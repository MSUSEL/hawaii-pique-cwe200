import json
import os
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from tqdm import tqdm
import hashlib
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# JSON parsing functions (unchanged)
def read_data_flow_file(file):
    with open(file, "r") as f:
        data_flows = json.load(f)
    return data_flows

def camel_case_split(str_input):
    words = [[str_input[0]]]
    for c in str_input[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append([c])
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def text_preprocess(feature_text):
    words = camel_case_split(feature_text)
    preprocessed_text = ' '.join(words).lower()
    return preprocessed_text

def process_data_flows(labeled_flows_dir):
    processed_data_flows = []
    seen_flow_hashes = set()
    
    total_flows = 0
    duplicate_flows = 0
    kept_flows = 0
    
    for file_name in os.listdir(labeled_flows_dir):
        data_flows = read_data_flow_file(os.path.join(labeled_flows_dir, file_name))
        for cwe in data_flows.keys():
            for result in data_flows[cwe]:
                result_index = result['resultIndex']
                flow_file_name = result['fileName']
                
                for flow in result['flows']:
                    total_flows += 1
                    
                    if not flow['flow'] or 'label' not in flow:
                        continue
                    
                    label = 1 if flow['label'] == 'Yes' else 0 if flow['label'] == 'No' else None
                    if label is None:
                        continue
                    
                    data_flow_string = f"Filename = {flow_file_name} Flows = "
                    for step in flow['flow']:
                        data_flow_string += str(step)
                    processed_text = text_preprocess(data_flow_string)
                    
                    flow_hash = hashlib.sha256(processed_text.encode('utf-8')).hexdigest()
                    if flow_hash in seen_flow_hashes:
                        duplicate_flows += 1
                        continue
                    
                    seen_flow_hashes.add(flow_hash)
                    kept_flows += 1
                    
                    instruction = f"Given the following data flow, determine if it exposes sensitive information (Yes) or not (No):\n{processed_text}"
                    response = "Yes" if label == 1 else "No"
                    processed_data_flows.append({"instruction": instruction, "response": response})
    
    print(f"Total flows processed: {total_flows}")
    print(f"Duplicate flows excluded: {duplicate_flows}")
    print(f"Flows kept for training: {kept_flows}")
    
    with open('processed_data_flows_for_llama.json', 'w', encoding='utf-8') as json_file:
        json.dump(processed_data_flows, json_file, indent=4)
    
    return processed_data_flows

# Prepare dataset
def prepare_dataset(data):
    dataset_dict = {
        "text": [f"<|startoftext|>### Instruction: {item['instruction']}\n### Response: {item['response']}<|endoftext|>" 
                 for item in data],
        "instruction": [item["instruction"] for item in data],  # For evaluation
        "response": [item["response"] for item in data]         # For evaluation
    }
    return Dataset.from_dict(dataset_dict)

# Custom evaluation function
def evaluate_model(model, tokenizer, eval_dataset, max_seq_length):
    model.eval()
    FastLanguageModel.for_inference(model)
    predictions = []
    true_labels = []

    yes_token_id = tokenizer(" Yes", add_special_tokens=False).input_ids[0]
    no_token_id = tokenizer(" No", add_special_tokens=False).input_ids[0]

    print("Evaluating model on evaluation dataset...")

    for example in tqdm(eval_dataset, desc="Evaluating"):
        torch.cuda.empty_cache()  # Clear before each forward pass

        instruction = example["instruction"]
        true_response = example["response"]

        prompt = f"<|startoftext|>### Instruction: {instruction}\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        next_token_logits = logits[0, -1]  # Grab logits for next token prediction
        pred_label = 1 if next_token_logits[yes_token_id] > next_token_logits[no_token_id] else 0
        true_label = 1 if true_response.lower() == "yes" else 0

        predictions.append(pred_label)
        true_labels.append(true_label)

        torch.cuda.empty_cache()  # Clear after each pass for safety

    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    return accuracy, f1, recall, precision



if __name__ == "__main__":
    # Configuration
    labeled_flows_dir = os.path.join('testing', 'Labeling', 'FlowData')
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "./llama_finetuned_model"
    max_seq_length = 10000
    
    # Step 1: Process data flows
    print("Processing data flows...")
    processed_data = process_data_flows(labeled_flows_dir)
    
    # Step 2: Create dataset
    print("Preparing dataset for fine-tuning...")
    dataset = prepare_dataset(processed_data)
    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
    
    # Step 3: Load model with 4-bit quantization and LoRA
    print("Loading Llama-3.1-8B-Instruct with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    
    # Step 4: Set up training arguments with epochs
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=1,  # Increased from implicit 1 epoch (via max_steps) to 3 epochs
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        # save_steps=50,
        optim="adamw_8bit",
        report_to="none",
        # load_best_model_at_end=True,
        save_total_limit=1,                    # Keep only the best checkpoint
        
        evaluation_strategy="no",
        # eval_steps=10,
        # metric_for_best_model="eval_loss",     
        # greater_is_better=False,
        # prediction_loss_only=True
    )
    
    # Step 5: Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    
    # Step 6: Train the model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Step 7: Evaluate the model
    evaluate_model(model, tokenizer, eval_dataset, max_seq_length)
    
    # Step 8: Save the fine-tuned model
    print("Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning and evaluation complete!")