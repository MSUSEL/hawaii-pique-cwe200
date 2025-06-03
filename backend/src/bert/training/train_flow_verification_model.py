import os
import json
import sys
import numpy as np
from tqdm import tqdm
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", FutureWarning)

from skorch import NeuralNetClassifier

##########################################
# Data Processing Functions
##########################################
def read_data_flow_file(file):
    """Read a JSON file containing labeled data flows.
    Loads raw labeled flow data for later processing.

    :param file: path to JSON file
    :returns: loaded JSON object
    """
    with open(file, "r") as f:
        data_flows = json.load(f)
    return data_flows

def process_data_flows(labeled_flows_dir):
    """Parse, deduplicate, and label all data flows from directory.
    Prepares clean and unique flows for training and evaluation.

    :param labeled_flows_dir: directory containing labeled flow files
    :returns: NumPy array of processed flows
    """
    processed_data_flows = []
    seen_flow_hashes = set()
    total_flows = 0
    duplicate_flows = 0
    kept_flows = 0
    
    for file_name in os.listdir(labeled_flows_dir):
        if not file_name.endswith('.json'):
            continue
        data_flows = read_data_flow_file(os.path.join(labeled_flows_dir, file_name))
        for cwe in data_flows.keys():
            for result in data_flows[cwe]:
                result_index = result['resultIndex']
                file_name = result['fileName']
                for flow in result['flows']:
                    total_flows += 1
                    
                    if not flow['flow'] or 'label' not in flow:
                        continue
                    label = 1 if flow['label'] == 'Yes' else 0 if flow['label'] == 'No' else None
                    if label is None:
                        continue
                    
                    # Build the original “data_flow_string”  
                    original_data_flow_string = f"CWE = {cwe}, Flows = "
                    for step in flow["flow"]:
                        step_string = f"step={step.get('step', '')}, "
                        step_string += f"variableName={step.get('variableName', '')}, "
                        step_string += f"type={step.get('type', '')}, "
                        step_string += f"code={step.get('code', '')}, "
                        step_string = {step_string}
                        original_data_flow_string += str(step_string)

                    # ─── Build a normalized “signature” for dedup’ing: only (variableName, type) ───
                    step_signature = []
                    for step in flow["flow"]:
                        varname = step.get("variableName", "").strip()
                        vartype = step.get("type", "").strip()
                        step_signature.append(f"{varname}::{vartype}")

                    signature_str = f"CWE={cwe}|" + "→".join(step_signature)
                    flow_hash = hashlib.sha256(signature_str.encode("utf-8")).hexdigest()

                    if flow_hash in seen_flow_hashes:
                        duplicate_flows += 1
                        continue

                    seen_flow_hashes.add(flow_hash)
                    kept_flows += 1

                    # ─── Store the original_data_flow_string (not signature_str) ───
                    processed_data_flows.append([
                        file_name,
                        result_index,
                        flow["codeFlowIndex"],
                        original_data_flow_string,
                        label
                    ])
    
    print(f"Total flows processed: {total_flows}")
    print(f"Duplicate flows excluded: {duplicate_flows}")
    print(f"Flows kept for training: {kept_flows}")
    # with open('processed_data_flows.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(processed_data_flows, json_file, indent=4)
    return np.array(processed_data_flows)

def format_data_flows_for_graphcodebert(processed_flows):
    """Extract flow text strings from preprocessed flows.
    Needed to prepare text for GraphCodeBERT input.

    :param processed_flows: list or array of processed data flow rows
    :returns: list of flow text strings
    """
    formatted_flows = []
    for row in processed_flows:
        flow_text = row[3]
        formatted_flows.append(flow_text)
    return formatted_flows

##########################################
# Embedding Functions with LSTM Aggregation
##########################################
class RNNAggregator(nn.Module):
    """ [Unused] LSTM-based aggregator for encoding segmented input.
    Aggregates embeddings across token segments using LSTM.

    :param embedding_dim: dimension of each token embedding
    """
    def __init__(self, embedding_dim):
        super(RNNAggregator, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return h_n  # Use the final hidden state

class TransformerAggregator(nn.Module):
    """Transformer encoder for aggregating token sequences.
    Captures contextual relationships using self-attention.

    :param embedding_dim: input/output embedding size
    :param num_heads: number of attention heads
    :param num_layers: number of encoder layers
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
    """Embed sentence using model and aggregator with optional segmentation.
    Handles long sequences by segmenting and recombining with RNN/Transformer.

    :param sentence: input string
    :param tokenizer: tokenizer instance from Hugging Face
    :param model: transformer model instance
    :param aggregator: module to aggregate segments (RNN or Transformer)
    :param device: target device (e.g., 'cuda')
    :param max_length: token cutoff length per segment
    :returns: sentence embedding as a 1D NumPy array
    """
    encoding = tokenizer(sentence, return_tensors="pt", truncation=False)
    input_ids = encoding["input_ids"][0]
    
    if input_ids.size(0) <= max_length:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=max_length)
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
            segment_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            segment_embeddings.append(segment_embedding)
        seg_tensor = torch.tensor(np.stack(segment_embeddings), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h_n = aggregator(seg_tensor)
        aggregated_embedding = h_n[-1].squeeze(0).cpu().numpy()
        return aggregated_embedding

def calculate_graphcodebert_vectors(sentences, model_name='microsoft/graphcodebert-base', max_length=512, device='cuda', aggregator_cls=RNNAggregator):
    """Compute embeddings using GraphCodeBERT + sequence aggregator.
    Generates contextual vector embeddings for input flows.

    :param sentences: list of flow text strings
    :param model_name: Hugging Face model name
    :param max_length: max tokens per segment
    :param device: target computation device
    :param aggregator_cls: class for aggregation (e.g., RNN, Transformer)
    :returns: stacked NumPy array of embeddings
    """
    print(f"Using device {device} for encoding")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    embedding_dim = model.config.hidden_size
    aggregator = aggregator_cls(embedding_dim)
    aggregator.to(device)
    aggregator.eval()
    
    embeddings = []
    for sentence in tqdm(sentences, desc="Embedding with GraphCodeBERT"):
        embedding = embed_sentence(sentence, tokenizer, model, aggregator, device, max_length=max_length)
        embeddings.append(embedding)
    torch.save(aggregator.state_dict(), os.path.join(model_dir, 'aggregator.pt'))
    print(f"Aggregator state dict saved at {os.path.join(model_dir, 'aggregator.pt')}")
    return np.vstack(embeddings)

##########################################
# PyTorch Classifier Model
##########################################
def get_activation(act_name):
    """Return PyTorch activation function by name.
    Enables dynamic activation configuration.

    :param act_name: string name of activation function
    :returns: PyTorch activation function instance
    """
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_name == 'elu':
        return nn.ELU()
    elif act_name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {act_name}")

class ClassifierModel(nn.Module):
    """Feed-forward neural network for binary classification.
    Predicts sensitive vs. non-sensitive flows using dense layers.

    :param embedding_dim: input feature size per sample
    :param dropout_rate: dropout rate for regularization
    :param activation: string name for activation function
    """
    def __init__(self, embedding_dim, dropout_rate=0.2, activation='elu'):
        super(ClassifierModel, self).__init__()
        units1 = embedding_dim
        units2 = embedding_dim * 3 // 4
        units3 = embedding_dim // 2
        units4 = embedding_dim // 4
        
        self.act = get_activation(activation)
        
        self.fc1 = nn.Linear(embedding_dim, units1)
        self.bn1 = nn.BatchNorm1d(units1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_res1 = nn.Linear(units1, units2)
        self.bn_res1 = nn.BatchNorm1d(units2)
        self.dropout_res1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(units1, units2)
        self.fc3 = nn.Linear(units2, units2)
        
        self.fc_res2 = nn.Linear(units2, units3)
        self.bn_res2 = nn.BatchNorm1d(units3)
        self.dropout_res2 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(units2, units3)
        self.fc5 = nn.Linear(units3, units3)
        
        self.fc_final = nn.Linear(units3, units4)
        self.bn_final = nn.BatchNorm1d(units4)
        self.dropout_final = nn.Dropout(dropout_rate)
        self.out = nn.Linear(units4, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        res1 = self.fc_res1(x)
        res1 = self.bn_res1(res1)
        res1 = self.act(res1)
        res1 = self.dropout_res1(res1)
        
        x2 = self.fc2(x)
        x2 = x2 + res1
        x3 = self.fc3(x2)
        x3 = x3 + res1
        
        res2 = self.fc_res2(x3)
        res2 = self.bn_res2(res2)
        res2 = self.act(res2)
        res2 = self.dropout_res2(res2)
        
        x4 = self.fc4(x3)
        x4 = x4 + res2
        x5 = self.fc5(x4)
        x5 = x5 + res2
        
        x6 = self.fc_final(x5)
        x6 = self.bn_final(x6)
        x6 = self.act(x6)
        x6 = self.dropout_final(x6)
        
        out = self.out(x6)
        return torch.sigmoid(out).squeeze(1)

##########################################
# Customized RandomizedSearchCV with TQDM
##########################################
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler

class TqdmRandomizedSearchCV(RandomizedSearchCV):
    """RandomizedSearchCV with progress tracking via tqdm.
    Enhances visibility into hyperparameter tuning progress.

    :param RandomizedSearchCV: base class from sklearn
    """
    def _run_search(self, evaluate_candidates):
        param_list = list(ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state))
        for params in tqdm(param_list, desc="Total fits"):
            evaluate_candidates([params])

##########################################
# Training, Hyperparameter Tuning, and Evaluation
##########################################
if __name__ == "__main__":
    """Run end-to-end training and evaluation pipeline.
    Coordinates preprocessing, embedding, model tuning, and saving.

    :returns: None
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    labeled_flows_dir = os.path.join('testing', 'Labeling', 'Flow Verification', 'FlowData')
    model_dir = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Processing data flows...")
    processed_data_flows = process_data_flows(labeled_flows_dir)
   
    print("Formatting data flows for GraphCodeBERT...")
    formatted_flows = format_data_flows_for_graphcodebert(processed_data_flows)
    
    print("Calculating GraphCodeBERT embeddings...")
    embeddings = calculate_graphcodebert_vectors(formatted_flows, max_length=512, device=device, aggregator_cls=TransformerAggregator)
    embedding_dim = embeddings.shape[1]
    labels = processed_data_flows[:, 4].astype(np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    net = NeuralNetClassifier(
        module=ClassifierModel,
        module__embedding_dim=embedding_dim,
        module__dropout_rate=0.2,
        module__activation='elu',
        max_epochs=50,
        lr=0.0001,
        batch_size=32,
        optimizer=optim.Adam,
        optimizer__weight_decay=0.0001,
        criterion=nn.BCELoss,
        device=device,
        iterator_train__shuffle=True,
        verbose=0,
    )
    
    param_grid = {
        'lr': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'module__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'module__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'optimizer__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'max_epochs': [60, 80, 100],
    }
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if device == 'cuda':
        n_jobs = 3
    else:
        n_jobs = -1

    random_search = TqdmRandomizedSearchCV(
        estimator=net,
        param_distributions=param_grid,
        n_iter=10,
        cv=kfold,
        scoring='f1',
        n_jobs=n_jobs,
        error_score='raise',
        random_state=42,
        verbose=0,
    )
    
    print("Starting hyperparameter tuning...")
    random_search_result = random_search.fit(X_train.astype(np.float32), y_train.astype(np.float32))
    
    print(f"Best CV F1 Score: {random_search_result.best_score_:.4f} using {random_search_result.best_params_}")
    best_params = random_search_result.best_params_
    
    final_net = NeuralNetClassifier(
        module=ClassifierModel,
        module__embedding_dim=embedding_dim,
        module__dropout_rate=best_params['module__dropout_rate'],
        module__activation=best_params['module__activation'],
        max_epochs=best_params['max_epochs'],
        lr=best_params['lr'],
        batch_size=best_params['batch_size'],
        optimizer=optim.Adam,
        optimizer__weight_decay=best_params['optimizer__weight_decay'],
        criterion=nn.BCELoss,
        device=device,
        iterator_train__shuffle=True,
        verbose=1,
    )
    
    print("Training final model...")
    final_net.fit(X_train.astype(np.float32), y_train.astype(np.float32))
    
    print("Evaluating final model on test set...")
    y_pred = final_net.predict(X_test.astype(np.float32))
    predicted_probs = final_net.predict_proba(X_test.astype(np.float32))[:, 1]
    
    print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {metrics.f1_score(y_test, y_pred):.4f}")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, predicted_probs):.4f}")
    print(metrics.classification_report(y_test, y_pred, target_names=["Non-sensitive", "Sensitive"]))
    print(metrics.confusion_matrix(y_test, y_pred))
    
    scripted_model = torch.jit.script(final_net.module_)
    model_path = os.path.join(model_dir, 'verify_flows_final_model.pt')
    scripted_model.save(model_path)
    print(f"Final model saved at {model_path}")
    
    print("Training complete!")