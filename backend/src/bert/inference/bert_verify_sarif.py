import json
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

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
                data_flow_string = f"CWE = {cwe}, Flows = "
                for step in flow['flow']:
                    data_flow_string += str(step)
                processed_data_flows.append(data_flow_string)
                flow_references.append((cwe, result_index, code_flow_index))
    return processed_data_flows, flow_references

def format_data_flows_for_graphcodebert(processed_flows):
    return processed_flows  # Already a list of strings

##########################################
# Embedding Functions with GraphCodeBERT + LSTM
##########################################
import torch.nn as nn
class RNNAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(RNNAggregator, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return h_n
    
class TransformerAggregator(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, num_layers=1):
        super(TransformerAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: shape (batch_size, sequence_length, embedding_dim)
        returns: shape (batch_size, embedding_dim)
        """
        encoded = self.transformer_encoder(x)
        # Aggregate by mean pooling over the sequence
        return torch.mean(encoded, dim=1)

def embed_sentence(sentence, tokenizer, model, aggregator, device, max_length=512):
    encoding = tokenizer(sentence, return_tensors="pt", truncation=False)
    input_ids = encoding["input_ids"][0]
    
    if input_ids.size(0) <= max_length:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    else:
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
        seg_tensor = torch.tensor(np.stack(segment_embeddings), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h_n = aggregator(seg_tensor)
        aggregated_embedding = h_n[-1].squeeze(0).cpu().numpy()
        return aggregated_embedding

def calculate_graphcodebert_vectors(sentences, model_path, model_name='microsoft/graphcodebert-base', max_length=512, device='cuda', aggregator_cls=RNNAggregator):
    print(f"Using device {device} for encoding")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    embedding_dim = model.config.hidden_size
    aggregator = aggregator_cls(embedding_dim)
    aggregator.to(device)
    aggregator_path = os.path.join(os.path.dirname(model_path), 'aggregator.pt')
    if not os.path.exists(aggregator_path):
        raise FileNotFoundError(f"Aggregator state dict not found at {aggregator_path}")
    aggregator.load_state_dict(torch.load(aggregator_path, map_location=device))
    aggregator.eval()
    
    embeddings = []
    for sentence in tqdm(sentences, desc="Embedding with GraphCodeBERT"):
        emb = embed_sentence(sentence, tokenizer, model, aggregator, device, max_length=max_length)
        embeddings.append(emb)
    return np.vstack(embeddings)

##########################################
# Model Loading and Inference Functions
##########################################
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def predict_labels(model, embeddings):
    print("Running inference...")
    device = next(model.parameters()).device
    batch_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
    print("Sample outputs:", outputs[:5].cpu().numpy())
    predicted_probs = outputs.cpu().numpy().flatten()
    predicted_classes = (predicted_probs > 0.5).astype(int)
    predicted_labels = ["Yes" if pred == 1 else "No" for pred in predicted_classes]
    return predicted_labels, predicted_probs

def update_json_with_predictions(data_flows, flow_references, predicted_labels, predicted_probs):
    print("Updating JSON with predictions...")
    if len(predicted_probs.shape) > 1:
        predicted_probs = predicted_probs.flatten()
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
    model = load_model(model_path)
    
    print(f"Reading data flows from {input_json_path}...")
    data_flows = read_data_flow_file(input_json_path)
    processed_flows, flow_references = process_data_flows_for_inference(data_flows)
    processed_flows = format_data_flows_for_graphcodebert(processed_flows)
    
    embeddings = calculate_graphcodebert_vectors(processed_flows, model_path, max_length=512, device='cuda' if torch.cuda.is_available() else 'cpu', aggregator_cls=TransformerAggregator)
    
    predicted_labels, predicted_probs = predict_labels(model, embeddings)
    
    updated_data_flows = update_json_with_predictions(data_flows, flow_references, predicted_labels, predicted_probs)
    
    output_path = save_updated_json(updated_data_flows, input_json_path)
    print(f"Inference complete! Updated JSON saved to {output_path}")

if __name__ == "__main__":
    print(f"Arguments: {sys.argv}")
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
        project_path = os.path.join(os.getcwd(), "Files", project_name)
        model_path = os.path.join(os.getcwd(), 'src', 'bert', 'models', 'verify_flows_final_model.pt')
    else:
        project_name = "CWEToyDataset"
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
        model_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models", "verify_flows_final_model.pt")
    
    print(f"Project name: {project_name}")
    run(project_path, model_path)