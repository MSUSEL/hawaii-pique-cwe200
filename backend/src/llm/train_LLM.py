import json
import os
import numpy as np
import random
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from tqdm import tqdm
import hashlib
import re
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def read_data_flow_file(file):
    with open(file, "r") as f:
        return json.load(f)

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
    return ' '.join(words).lower()

def process_data_flows(labeled_flows_dir):
    processed_data_flows = []
    seen_flow_hashes = set()

    total_flows = 0
    duplicate_flows = 0
    kept_flows = 0

    for file_name in os.listdir(labeled_flows_dir):
        data_flows = read_data_flow_file(os.path.join(labeled_flows_dir, file_name))
        for cwe in data_flows:
            for result in data_flows[cwe]:
                flow_file_name = result['fileName']

                for flow in result['flows']:
                    total_flows += 1
                    if not flow['flow'] or 'label' not in flow:
                        continue

                    label = 1 if flow['label'] == 'Yes' else 0 if flow['label'] == 'No' else None
                    if label is None:
                        continue

                    # Add preprocessing here
                    preprocessed_steps = [text_preprocess(str(step)) for step in flow['flow']]
                    data_flow_string = f"Filename = {flow_file_name}, CWE = {cwe}, Flows = " + ' -> '.join(preprocessed_steps)

                    flow_hash = hashlib.sha256(data_flow_string.encode('utf-8')).hexdigest()
                    if flow_hash in seen_flow_hashes:
                        duplicate_flows += 1
                        continue

                    seen_flow_hashes.add(flow_hash)
                    kept_flows += 1

                    instruction = f"Does this data flow expose sensitive information?\n{data_flow_string}"
                    response = "Yes" if label == 1 else "No"
                    processed_data_flows.append({"instruction": instruction, "response": response})

    print(f"Total flows processed: {total_flows}")
    print(f"Duplicate flows excluded: {duplicate_flows}")
    print(f"Flows kept for training: {kept_flows}")

    with open('processed_data_flows_for_llama.json', 'w', encoding='utf-8') as json_file:
        json.dump(processed_data_flows, json_file, indent=4)

    return processed_data_flows

def balance_dataset(data):
    yes_data = [item for item in data if item['response'] == "Yes"]
    no_data = [item for item in data if item['response'] == "No"]

    if len(yes_data) == 0:
        return no_data

    oversampled_yes = random.choices(yes_data, k=len(no_data))
    balanced_data = no_data + oversampled_yes
    random.shuffle(balanced_data)
    print(f"Balanced dataset: {len(yes_data)} Yes -> {len(oversampled_yes)} oversampled | Total: {len(balanced_data)}")
    return balanced_data

def prepare_dataset(data):
    dataset_dict = {
        "text": [f"<|startoftext|>### Instruction: {item['instruction']}\n### Response: {item['response']}<|endoftext|>" for item in data],
        "instruction": [item["instruction"] for item in data],
        "response": [item["response"] for item in data]
    }
    return Dataset.from_dict(dataset_dict)

def evaluate_model(model, tokenizer, eval_dataset, max_seq_length):
    model.eval()
    FastLanguageModel.for_inference(model)
    predictions = []
    true_labels = []
    prediction_logs = []

    print("Evaluating model on evaluation dataset...")

    for example in tqdm(eval_dataset, desc="Evaluating"):
        torch.cuda.empty_cache()

        instruction = example["instruction"]
        true_response = example["response"]

        prompt = f"<|startoftext|>### Instruction: {instruction}\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # pred_label = 1 if re.search(r"\\byes\\b", output.lower()) else 0
        pred_label = 1 if re.search(r"\byes\b", output.lower().strip(". \n")) else 0

        true_label = 1 if true_response.lower() == "yes" else 0

        predictions.append(pred_label)
        true_labels.append(true_label)

        prediction_logs.append({
            "instruction": instruction,
            "true_response": true_response,
            "predicted_output": output.strip(),
            "predicted_label": "Yes" if pred_label == 1 else "No"
        })

    with open("eval_predictions.json", "w") as f:
        json.dump(prediction_logs, f, indent=2)

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
    labeled_flows_dir = os.path.join('testing', 'Labeling', 'FlowData')
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "./llama_finetuned_model"
    max_seq_length = 15000

    print("Processing data flows...")
    processed_data = process_data_flows(labeled_flows_dir)

    print("Preparing dataset...")
    full_dataset = prepare_dataset(processed_data)

    print("Splitting dataset...")
    train_dataset, eval_dataset = full_dataset.train_test_split(test_size=0.1, seed=42).values()

    print("Balancing training dataset...")
    train_data = train_dataset.to_dict()
    balanced_train_data = balance_dataset(
        [{"instruction": i, "response": r} for i, r in zip(train_data["instruction"], train_data["response"])]
    )
    train_dataset = prepare_dataset(balanced_train_data)

    print("Loading model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=10,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        report_to="none",
        evaluation_strategy="no",
        save_total_limit=1
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating model...")
    evaluate_model(model, tokenizer, eval_dataset, max_seq_length)

    print("Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Fine-tuning and evaluation complete!")
