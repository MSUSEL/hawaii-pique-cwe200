import json
import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import hashlib
import torch
from transformers import AutoTokenizer, AutoModel
from classifier_model import ClassifierModel


# Suppress TensorFlow logs if any (not used here, but left for compatibility)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

##########################################
# Data Processing Functions
##########################################
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

def read_data_flow_file(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data_flows = json.load(f)
    return data_flows

def process_data_flows_for_inference(data_flows):
    processed_data_flows = []
    flow_references = []
    for cwe in data_flows.keys():
        for result in data_flows[cwe]:
            result_index = result['resultIndex']
            flow_file_name = result['fileName']
            for flow in result['flows']:
                code_flow_index = flow['codeFlowIndex']
                # Construct a string from filename and flow steps.
                data_flow_string = f"Filename = {flow_file_name} Flows = "
                for step in flow['flow']:
                    data_flow_string += str(step)
                processed_data_flows.append(text_preprocess(data_flow_string))
                flow_references.append((cwe, result_index, code_flow_index))
    return processed_data_flows, flow_references

##########################################
# Embedding Functions with GraphCodeBERT + LSTM
##########################################
# LSTM aggregator class (same as in training)
import torch.nn as nn
class RNNAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(RNNAggregator, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        output, (h_n, c_n) = self.lstm(x)
        return h_n  # (num_layers, batch, hidden_size)

def embed_sentence(sentence, tokenizer, model, aggregator, device, max_length=512):
    """
    If the tokenized sentence exceeds max_length, split into segments, encode each segment,
    then aggregate via LSTM.
    """
    encoding = tokenizer(sentence, return_tensors="pt", truncation=False)
    input_ids = encoding["input_ids"][0]
    
    if input_ids.size(0) <= max_length:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the first token ([CLS]) embedding
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    else:
        # Split the tokens into segments
        tokens = input_ids.tolist()
        segment_embeddings = []
        for i in range(0, len(tokens), max_length):
            segment_tokens = tokens[i:i+max_length]
            segment_tensor = torch.tensor(segment_tokens).unsqueeze(0).to(device)
            attention_mask = torch.ones(segment_tensor.shape, dtype=torch.long).to(device)
            with torch.no_grad():
                outputs = model(input_ids=segment_tensor, attention_mask=attention_mask)
            seg_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            segment_embeddings.append(seg_emb)
        # Stack segments into (1, num_segments, embedding_dim)
        seg_tensor = torch.tensor(np.stack(segment_embeddings), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h_n = aggregator(seg_tensor)  # shape: (num_layers, batch, embedding_dim)
        aggregated_embedding = h_n[-1].squeeze(0).cpu().numpy()
        return aggregated_embedding

def calculate_graphcodebert_vectors(sentences, model_name='microsoft/graphcodebert-base', max_length=512, device='cuda'):
    print(f"Using device {device} for encoding")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    embedding_dim = model.config.hidden_size
    aggregator = RNNAggregator(embedding_dim)
    aggregator.to(device)
    aggregator.eval()
    
    embeddings = []
    for sentence in tqdm(sentences, desc="Embedding with GraphCodeBERT"):
        emb = embed_sentence(sentence, tokenizer, model, aggregator, device, max_length=max_length)
        embeddings.append(emb)
    return np.vstack(embeddings)

##########################################
# Model Loading and Inference Functions
##########################################
def load_skorch_model(model_path):
    print(f"Loading PyTorch model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_labels(model, embeddings):
    print("Running inference...")
    predicted_probs = model.predict_proba(embeddings.astype(np.float32))[:, 1]
    predicted_classes = (predicted_probs > 0.5).astype(int)
    predicted_labels = ["Yes" if pred == 1 else "No" for pred in predicted_classes]
    return predicted_labels, predicted_probs

def update_json_with_predictions(data_flows, flow_references, predicted_labels, predicted_probs):
    print("Updating JSON with predictions...")
    if len(predicted_probs.shape) > 1:
        predicted_probs = predicted_probs.flatten()
    # Zip references with predictions
    for (cwe, result_index, code_flow_index), label, prob in zip(flow_references, predicted_labels, predicted_probs):
        for result in data_flows[cwe]:
            if result['resultIndex'] == result_index:
                for flow in result['flows']:
                    if flow['codeFlowIndex'] == code_flow_index:
                        flow['label'] = label
                        flow['probability'] = float(prob)
                        break
    return data_flows

def save_updated_json(data_flows, input_file_path):
    output_file_path = os.path.splitext(input_file_path)[0] + ".json"
    print(f"Saving updated JSON to {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_flows, f, indent=4)
    return output_file_path

##########################################
# Main Inference Function
##########################################
def run(project_path, model_path):
    input_json_path = os.path.join(project_path, 'flowMapsByCWE.json')
    # Load the pickled Skorch model
    model = load_skorch_model(model_path)
    
    # Read and process input JSON
    print(f"Reading data flows from {input_json_path}...")
    data_flows = read_data_flow_file(input_json_path)
    processed_flows, flow_references = process_data_flows_for_inference(data_flows)
    
    # Calculate embeddings using GraphCodeBERT with LSTM aggregator
    embeddings = calculate_graphcodebert_vectors(processed_flows, max_length=512, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Predict labels using the loaded model
    predicted_labels, predicted_probs = predict_labels(model, embeddings)
    
    # Update the JSON with predictions
    updated_data_flows = update_json_with_predictions(data_flows, flow_references, predicted_labels, predicted_probs)
    
    # Save updated JSON
    output_path = save_updated_json(updated_data_flows, input_json_path)
    print(f"Inference complete! Updated JSON saved to {output_path}")

if __name__ == "__main__":
    print(f"Arguments: {sys.argv}")
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
        project_path = os.path.join(os.getcwd(), "Files", project_name)
        input_json_path = os.path.join(project_path, 'flowMapsByCWE.json')
        model_path = os.path.join(os.getcwd(), 'src', 'bert', 'models', 'verify_flows_skorch.pkl')
    else:
        project_name = "snowflake-jdbc-3.23.2"  # Change default project name as needed
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
        input_json_path = os.path.join(project_path, 'flowMapsByCWE.json')
        model_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models", "verify_flows_skorch.pkl")
    
    print(f"Project name: {project_name}")
    run(project_path, model_path)
