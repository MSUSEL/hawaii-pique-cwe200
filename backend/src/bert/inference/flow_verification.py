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
def read_data_flow_file(file_path):
    """Read a JSON file containing labeled data flows.
    Loads raw labeled flow data for later processing.

    :param file: path to JSON file
    :returns: loaded JSON object
    """
    with open(file_path, "r", encoding='utf-8') as f:
        data_flows = json.load(f)
    return data_flows

def process_data_flows_for_inference(data_flows):
    """Process the loaded data flows for inference.
    Extracts relevant information and formats it for embedding.
    :param data_flows: loaded JSON object
    :returns: processed data flows and their references
    """
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
    """Format the processed data flows for GraphCodeBERT embedding.
    In the training script, the processed flow had other information, such as the file name and the result index.
    Since this is inference, the processed flow is just a string already. I kept this function for consistency.
    :param processed_flows: list of processed data flows
    :returns: formatted data flows
    """
    return processed_flows  # Already a list of strings

##########################################
# Embedding Functions with GraphCodeBERT + LSTM
##########################################
import torch.nn as nn    
class TransformerAggregator(nn.Module):
    """Transformer-based aggregator for embeddings.
    This module uses a Transformer encoder to process the embeddings and aggregate them.
    This is needed because GraphCodeBERT only has a context window of 512 tokens.
    The transformer encoder will process the embeddings and aggregate them into a single vector 
    so that extra data don't get trunicated.
    """
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
    """Embed a single sentence using GraphCodeBERT and the aggregator.
    :param sentence: input sentence to embed
    :param tokenizer: tokenizer for GraphCodeBERT
    :param model: GraphCodeBERT model
    :param aggregator: aggregator for embeddings
    :param device: device to run the model on (CPU or GPU)
    :param max_length: maximum length of the input sequence
    :returns: aggregated embedding for the sentence
    """
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

def calculate_graphcodebert_vectors(sentences, model_path, model_name='microsoft/graphcodebert-base', max_length=512, device='cuda', aggregator_cls=TransformerAggregator):
    """Calculate GraphCodeBERT embeddings for a list of sentences.
    :param sentences: list of sentences to embed
    :param model_path: path to the model file
    :param model_name: name of the GraphCodeBERT model
    :param max_length: maximum length of the input sequence
    :param device: device to run the model on (CPU or GPU)
    :param aggregator_cls: class of the aggregator to use (default is RNNAggregator)
    :returns: numpy array of embeddings for the sentences
    """
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
    """Load the trained model from the specified path.
    :param model_path: path to the model file
    :returns: loaded model
    """
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def predict_labels(model, embeddings):
    """Run inference on the embeddings using the loaded model.
    :param model: loaded model
    :param embeddings: numpy array of embeddings
    :returns: predicted labels and probabilities
    """
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
    """Update the original JSON with the predicted labels and probabilities.
    :param data_flows: original JSON object
    :param flow_references: list of references to the flows in the original JSON
    :param predicted_labels: list of predicted labels
    :param predicted_probs: list of predicted probabilities
    :returns: updated JSON object
    """
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
    """Save the updated JSON object to a new file.
    :param data_flows: updated JSON object
    :param input_file_path: path to the original JSON file
    :returns: path to the saved JSON file
    """
    output_file_path = os.path.splitext(input_file_path)[0] + ".json"
    print(f"Saving updated JSON to {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_flows, f, indent=4)
    return output_file_path

##########################################
# Main Inference Function
##########################################
def run(project_path, model_path):
    """Main function to run the inference pipeline.
    :param project_path: path to the project directory
    :param model_path: path to the trained model file
    """
    print(f"Running inference for project: {project_path}")
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