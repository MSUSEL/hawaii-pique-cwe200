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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", FutureWarning)



# Import skorch for wrapping the PyTorch model as a scikit-learn estimator.
from skorch import NeuralNetClassifier

##########################################
# Data Processing Functions
##########################################
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
                    
                    data_flow_string = f"Filename = {flow_file_name}, CWE = {cwe}, Flows = "
                    for step in flow['flow']:
                        data_flow_string += str(step)
                    
                    flow_hash = hashlib.sha256(data_flow_string.encode('utf-8')).hexdigest()
                    if flow_hash in seen_flow_hashes:
                        duplicate_flows += 1
                        continue
                    seen_flow_hashes.add(flow_hash)
                    kept_flows += 1
                    
                    processed_data_flows.append([
                        file_name,
                        result_index,
                        flow['codeFlowIndex'],
                        data_flow_string,
                        label
                    ])
    
    print(f"Total flows processed: {total_flows}")
    print(f"Duplicate flows excluded: {duplicate_flows}")
    print(f"Flows kept for training: {kept_flows}")
    with open('processed_data_flows.json', 'w', encoding='utf-8') as json_file:
        json.dump(processed_data_flows, json_file, indent=4)
    return np.array(processed_data_flows)

def format_data_flows_for_graphcodebert(processed_flows):
    formatted_flows = []
    for row in processed_flows:
        flow_text = row[3]
        formatted_flows.append(flow_text)
    return formatted_flows

##########################################
# Embedding Functions with LSTM Aggregation
##########################################
# LSTM aggregator to combine segment embeddings.
class RNNAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(RNNAggregator, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        output, (h_n, c_n) = self.lstm(x)
        return h_n  # Use the final hidden state

def embed_sentence(sentence, tokenizer, model, aggregator, device, max_length=512):
    """
    Embeds a sentence using GraphCodeBERT. If the tokenized sentence exceeds max_length,
    it is split into segments and each segment is processed. Their embeddings are then
    aggregated using an LSTM.
    """
    # Tokenize without truncation to determine full token count.
    encoding = tokenizer(sentence, return_tensors="pt", truncation=False)
    input_ids = encoding["input_ids"][0]
    
    if input_ids.size(0) <= max_length:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Return the [CLS] token embedding.
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    else:
        # Long sequence: split into segments.
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
        # Stack segment embeddings and convert to tensor: shape (1, num_segments, embedding_dim)
        seg_tensor = torch.tensor(np.stack(segment_embeddings), dtype=torch.float32).unsqueeze(0).to(device)
        # Use the LSTM aggregator to combine segment embeddings.
        with torch.no_grad():
            h_n = aggregator(seg_tensor)  # h_n: (num_layers, batch, hidden_size)
        aggregated_embedding = h_n[-1].squeeze(0).cpu().numpy()  # Use last layer's hidden state.
        return aggregated_embedding

def calculate_graphcodebert_vectors(sentences, model_name='microsoft/graphcodebert-base', max_length=512, device='cuda'):
    """
    Calculates embeddings for a list of sentences using GraphCodeBERT.
    If a sentence exceeds max_length tokens, it is segmented and aggregated using an LSTM.
    """
    print(f"Using device {device} for encoding")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    embedding_dim = model.config.hidden_size
    # Initialize the LSTM aggregator.
    aggregator = RNNAggregator(embedding_dim)
    aggregator.to(device)
    aggregator.eval()
    
    embeddings = []
    for sentence in tqdm(sentences, desc="Embedding with GraphCodeBERT"):
        embedding = embed_sentence(sentence, tokenizer, model, aggregator, device, max_length=max_length)
        embeddings.append(embedding)
    return np.vstack(embeddings)

##########################################
# PyTorch Classifier Model
##########################################
def get_activation(act_name):
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
    def __init__(self, embedding_dim, dropout_rate=0.2, activation='elu'):
        super(ClassifierModel, self).__init__()
        # Define units similar to the Keras model.
        units1 = embedding_dim
        units2 = embedding_dim * 3 // 4
        units3 = embedding_dim // 2
        units4 = embedding_dim // 4
        
        self.act = get_activation(activation)
        
        # First block
        self.fc1 = nn.Linear(embedding_dim, units1)
        self.bn1 = nn.BatchNorm1d(units1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Residual block 1
        self.fc_res1 = nn.Linear(units1, units2)
        self.bn_res1 = nn.BatchNorm1d(units2)
        self.dropout_res1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(units1, units2)
        self.fc3 = nn.Linear(units2, units2)
        
        # Residual block 2
        self.fc_res2 = nn.Linear(units2, units3)
        self.bn_res2 = nn.BatchNorm1d(units3)
        self.dropout_res2 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(units2, units3)
        self.fc5 = nn.Linear(units3, units3)
        
        # Final layers
        self.fc_final = nn.Linear(units3, units4)
        self.bn_final = nn.BatchNorm1d(units4)
        self.dropout_final = nn.Dropout(dropout_rate)
        self.out = nn.Linear(units4, 1)
        
    def forward(self, x):
        # x shape: (batch, embedding_dim)
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
        return torch.sigmoid(out).squeeze(1)  # Squeeze to get shape (batch,)

##########################################
# Customized RandomizedSearchCV with TQDM
##########################################
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler

class TqdmRandomizedSearchCV(RandomizedSearchCV):
    def _run_search(self, evaluate_candidates):
        # Generate candidate parameter settings using ParameterSampler.
        param_list = list(ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state))
        # Evaluate each candidate individually with a tqdm progress bar.
        for params in tqdm(param_list, desc="Total fits"):
            evaluate_candidates([params])

##########################################
# Training, Hyperparameter Tuning, and Evaluation
##########################################
if __name__ == "__main__":
    # Use GPU if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # Paths
    labeled_flows_dir = os.path.join('testing', 'Labeling', 'FlowData')
    model_dir = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Data processing and encoding.
    print("Processing data flows...")
    processed_data_flows = process_data_flows(labeled_flows_dir)
   
    print("Formatting data flows for GraphCodeBERT...")
    formatted_flows = format_data_flows_for_graphcodebert(processed_data_flows)
    
    print("Calculating GraphCodeBERT embeddings...")
    embeddings = calculate_graphcodebert_vectors(formatted_flows, max_length=512, device=device)
    embedding_dim = embeddings.shape[1]
    labels = processed_data_flows[:, 4].astype(np.int32)
    
    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Set up the PyTorch model wrapped in a skorch NeuralNetClassifier.
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
        # Set the criterion to binary cross-entropy loss.
        criterion=nn.BCELoss,
        device=device,  # Use GPU
        iterator_train__shuffle=True,
        verbose=0,
    )
    
    # Hyperparameter grid for RandomizedSearchCV.
    param_grid = {
        'lr': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'module__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'module__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'optimizer__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'max_epochs': [60, 80, 100],
    }
    
    # Use stratified K-Fold cross-validation.
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if device == 'cuda':
        n_jobs = 4
    else:
        n_jobs = -1

    # Use the customized RandomizedSearchCV with tqdm.
    random_search = TqdmRandomizedSearchCV(
        estimator=net,
        param_distributions=param_grid,
        n_iter=1000,  # Adjust iterations as needed.
        cv=kfold,
        scoring='f1',
        n_jobs=n_jobs,  # Use a single job to avoid GPU conflicts
        error_score='raise',
        random_state=42,
        verbose=0,  # disable internal verbosity
    )
    
    print("Starting hyperparameter tuning...")
    random_search_result = random_search.fit(X_train.astype(np.float32), y_train.astype(np.float32))
    
    print(f"Best CV F1 Score: {random_search_result.best_score_:.4f} using {random_search_result.best_params_}")
    best_params = random_search_result.best_params_
    
    # Create a final model with the best parameters.
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
    
    # Save the final model (skorch models can be pickled).
    final_model_path = os.path.join(model_dir, 'verify_flows_skorch_new.pkl')
    with open(final_model_path, 'wb') as f:
        import pickle
        pickle.dump(final_net, f)
    
    print("Training complete!")
