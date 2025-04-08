import os
import json
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer

# ------------------------------ Utilities ------------------------------------

sink_type_mapping = {
    0: "N/A",
    1: "I/O Sink",
    2: "Print Sink",
    3: "Network Sink",
    4: "Log Sink",
    5: "Database Sink",
    6: "Email Sink",
    7: "IPC Sink"
}

sink_type_mapping_rev = {
    "N/A": 0,
    "I/O Sink": 1,
    "Print Sink": 2,
    "Network Sink": 3,
    "Log Sink": 4,
    "Database Sink": 5,
    "Email Sink": 6,
    "IPC Sink": 7
}

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

def read_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        return json.load(file)

def calculate_sentbert_vectors(sentences, batch_size=64):
    from sentence_transformers import SentenceTransformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    print("Encoding sentences with SentenceTransformer...")
    embeddings = model_transformer.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def calculate_t5_vectors(sentences, model_name='t5-small', batch_size=32):
    from transformers import T5Tokenizer, T5EncoderModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif not isinstance(batch, list):
                batch = [str(batch)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.detach().cpu().numpy())
    return np.vstack(embeddings)

def calculate_roberta_vectors(sentences, model_name='roberta-base', batch_size=32):
    from transformers import RobertaTokenizer, RobertaModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif not isinstance(batch, list):
                batch = [str(batch)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.detach().cpu().numpy())
    return np.vstack(embeddings)

def calculate_codebert_vectors(sentences, model_name='microsoft/codebert-base', batch_size=32):
    from transformers import AutoTokenizer, AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from tqdm import tqdm  # For progress tracking
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Calculating CodeBERT Vectors"):
            batch = sentences[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif not isinstance(batch, list):
                batch = [str(batch)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.detach().cpu().numpy())
    return np.vstack(embeddings)


def calculate_codellama_vectors(sentences, model_name='codellama/CodeLlama-7b', batch_size=32):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif not isinstance(batch, list):
                batch = [str(batch)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)

def calculate_distilbert_vectors(sentences, model_name='distilbert-base-uncased', batch_size=32):
    from transformers import DistilBertTokenizer, DistilBertModel
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif not isinstance(batch, list):
                batch = [str(batch)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)

def calculate_albert_vectors(sentences, model_name='albert-base-v2', batch_size=32):
    from transformers import AlbertTokenizer, AlbertModel
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif not isinstance(batch, list):
                batch = [str(batch)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def calculate_longformer_vectors(sentences, model_name='allenai/longformer-base-4096', batch_size=32):
    from transformers import LongformerTokenizer, LongformerModel
    from tqdm import tqdm
    import torch

    # Determine device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)
    model.to(device)  # Move model to GPU
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding with Longformer"):
            batch = sentences[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif not isinstance(batch, list):
                batch = [str(batch)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            # Move inputs to GPU
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.detach().cpu().numpy())
    return np.vstack(embeddings)



def concat_name_and_context(name_vecs, context_vecs):
    total_vecs = []
    for idx in range(len(name_vecs)):
        total_vecs.append(np.concatenate((name_vecs[idx], context_vecs[idx]), axis=None))
    return total_vecs

# ------------------------------ Model Definitions ------------------------------------

def get_activation(act_name):
    act_name = act_name.lower()
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'elu':
        return nn.ELU()
    elif act_name == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {act_name}")

class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, weight_decay, activation):
        super().__init__()
        self.units1 = embedding_dim
        self.units2 = embedding_dim * 3 // 4
        self.units3 = embedding_dim // 2
        self.units4 = embedding_dim // 4
        act = get_activation(activation)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, self.units1)
        self.bn1 = nn.BatchNorm1d(self.units1)
        self.res1_fc = nn.Linear(self.units1, self.units2)
        self.res1_bn = nn.BatchNorm1d(self.units2)
        self.skip1 = nn.Linear(self.units1, self.units2)
        self.res2_fc = nn.Linear(self.units2, self.units3)
        self.res2_bn = nn.BatchNorm1d(self.units3)
        self.skip2 = nn.Linear(self.units2, self.units3)
        self.fc2 = nn.Linear(self.units3, self.units4)
        self.bn2 = nn.BatchNorm1d(self.units4)
        self.out = nn.Linear(self.units4, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        res1 = self.res1_fc(x)
        res1 = self.res1_bn(res1)
        res1 = self.act(res1)
        res1 = self.dropout(res1)
        skip1 = self.skip1(x)
        x = skip1 + res1
        res2 = self.res2_fc(x)
        res2 = self.res2_bn(res2)
        res2 = self.act(res2)
        res2 = self.dropout(res2)
        skip2 = self.skip2(x)
        x = skip2 + res2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.out(x)
        out = torch.sigmoid(out)
        return out

class MultiClassClassifier(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, weight_decay, activation):
        super().__init__()
        units = embedding_dim // 3
        act = get_activation(activation)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.fc2 = nn.Linear(units, units // 2)
        self.bn2 = nn.BatchNorm1d(units // 2)
        self.fc3 = nn.Linear(units // 2, units // 4)
        self.bn3 = nn.BatchNorm1d(units // 4)
        self.out = nn.Linear(units // 4, 8)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        out = self.out(x)
        return out

# ------------------------------ Training Utilities ------------------------------------

def evaluate_model(model, dataloader, device, category, print_report=False):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if category == 'sinks':
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = (outputs > 0.5).int().cpu().numpy().flatten()
            preds.extend(predictions)
            trues.extend(labels.numpy())
    
    trues = np.array(trues)
    preds = np.array(preds)
    
    precision = metrics.precision_score(trues, preds, average='weighted', zero_division=0)
    recall = metrics.recall_score(trues, preds, average='weighted', zero_division=0)
    f1_score = metrics.f1_score(trues, preds, average='weighted', zero_division=0)
    accuracy = metrics.accuracy_score(trues, preds)
    
    if print_report:
        print("\nFinal Evaluation Results:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")
        print(f"Accuracy: {accuracy}")
        print('------------------------------------------------')
        if category == 'sinks':
            target_names = [sink_type_mapping[i] for i in range(8)]
            print("Classification Report (Multi-class):")
            print(metrics.classification_report(trues, preds, target_names=target_names, zero_division=0))
        else:
            print("Classification Report (Binary):")
            print(metrics.classification_report(trues, preds, target_names=["Non-sensitive", "Sensitive"], zero_division=0))
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(trues, preds))
    
    return f1_score


def train_model(model, optimizer, criterion, train_loader, val_loader, device, epochs, early_stop_patience=10, category='binary'):
    best_f1 = -1
    best_state = None
    patience = 0
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape[1] == 1:
                loss = criterion(outputs.squeeze(), labels.float())
            else:
                loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        val_f1 = evaluate_model(model, val_loader, device, category, print_report=False)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1, model

# ------------------------------ Data Processing ------------------------------------

def get_context(labels, context, category):
    data = []
    for label_entry in labels:
        file_name = label_entry['fileName']
        if file_name not in context:
            print(f"Warning: {file_name} not found in context data.")
            continue
        file_context = context[file_name]
        method_code_map = file_context.get('methodCodeMap', {})
        if category not in label_entry:
            continue
        for label_item in label_entry[category]:
            matched_context_item = next(
                (context_item for context_item in file_context.get(category, [])
                 if label_item['name'].strip() == str(context_item['name']).strip()), None)
            if matched_context_item:
                binary_label = 1 if label_item['IsSensitive'] == 'Yes' else 0
                aggregated_context = ''
                if category == 'variables':
                    methods = matched_context_item.get('methods', [])
                    aggregated_context = f"Type: {matched_context_item['type']}, Context: "
                    for method in methods:
                        if method != 'global' and method in method_code_map:
                            aggregated_context += method_code_map[method]
                elif category == 'strings':
                    methods = matched_context_item.get('methods', [])
                    aggregated_context = "Context: "
                    for method in methods:
                        if method != 'global' and method in method_code_map:
                            aggregated_context += method_code_map[method]
                elif category == 'sinks':
                    methods = matched_context_item.get('methods', [])
                    for method in methods:
                        if method != 'global' and method in method_code_map:
                            aggregated_context += method_code_map[method]
                if category in ['variables', 'strings']:
                    data.append([
                        text_preprocess(label_item['name']),
                        text_preprocess(aggregated_context),
                        binary_label
                    ])
                elif category == 'comments':
                    data.append([
                        text_preprocess(label_item['name']),
                        None,
                        binary_label
                    ])

                elif category == 'sinks':
                    sink_type = label_item['type']
                    if sink_type not in sink_type_mapping_rev:
                        print(f"Missing sink type: {sink_type} for label: {label_item}")
                        continue
                    sink_value = sink_type_mapping_rev.get(sink_type, 0)  # Default to 0 if missing
                    data.append([
                        text_preprocess(label_item['name']),
                        text_preprocess(aggregated_context),
                        sink_value
                    ])

    return data

# ------------------------------ Training Function ------------------------------------

def train(category, data, param_grid, create_model_fn, embedding_model='sentbert', embedding_dim=384*2):
    variable_array = np.array(data, dtype=object)
    if embedding_model == 'sentbert':
        get_embeddings = calculate_sentbert_vectors
    elif embedding_model == 't5':
        get_embeddings = calculate_t5_vectors
    elif embedding_model == 'roberta':
        get_embeddings = calculate_roberta_vectors
    elif embedding_model == 'codebert':
        get_embeddings = calculate_codebert_vectors
    elif embedding_model == 'codellama':
        get_embeddings = calculate_codellama_vectors
    elif embedding_model == 'distilbert':
        get_embeddings = calculate_distilbert_vectors
    elif embedding_model == 'albert':
        get_embeddings = calculate_albert_vectors
    elif embedding_model == 'longformer':  # Added support for Longformer
        get_embeddings = calculate_longformer_vectors
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

    print("Encoding values")
    names = variable_array[:, 0]
    name_vectors = get_embeddings(names)
    
    if category != 'comments':
        print("Encoding context")
        contexts = variable_array[:, 1]
        context_vectors = get_embeddings(contexts)
        concatenated_vectors = concat_name_and_context(name_vectors, context_vectors)
        print(f"Name vector width: {name_vectors.shape[1]}")
        print(f"Context vector width: {context_vectors.shape[1]}")
        print(f"Concatenated vector width: {np.array(concatenated_vectors)[0].size}")
    else:
        concatenated_vectors = name_vectors
        embedding_dim //= 2

    X = np.array(concatenated_vectors, dtype=np.float32)
    y = variable_array[:, 2].astype(np.int32)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42)
    
    if category != 'sinks':
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        X_train_val, y_train_val = SMOTE(random_state=42).fit_resample(X_train_val, y_train_val)
    
    def create_loader(X, y, batch_size, shuffle=True):
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Retrieve number of candidate models to try and convert to int if needed.
    n_iter = param_grid.get('n_iter', 5)
    if isinstance(n_iter, list):
        n_iter = n_iter[0]
        
    best_val_f1 = -1
    best_params = None
    best_model_state = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Candidate model progress bar
    for i in tqdm(range(n_iter), desc="Training Candidate Models"):
        params = {k.split('__')[1]: random.choice(v) for k, v in param_grid.items() if k.startswith('model__')}
        batch_size = random.choice(param_grid.get('batch_size', [32]))
        epochs = random.choice(param_grid.get('epochs', [60]))
        
        if category == 'sinks':
            model = create_model_fn(embedding_dim=embedding_dim, **params)
            criterion = nn.CrossEntropyLoss()
            cat_type = 'sinks'
        else:
            model = create_model_fn(embedding_dim=embedding_dim, **params)
            criterion = nn.BCEWithLogitsLoss()
            cat_type = 'binary'
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate'), weight_decay=params.get('weight_decay'))
        train_loader = create_loader(X_train, y_train, batch_size)
        val_loader = create_loader(X_val, y_val, batch_size, shuffle=False)
        candidate_f1, _ = train_model(model, optimizer, criterion, train_loader, val_loader, device, epochs, category=cat_type)
        if candidate_f1 > best_val_f1:
            best_val_f1 = candidate_f1
            best_params = params.copy()
            best_params.update({'batch_size': batch_size, 'epochs': epochs})
            best_model_state = model.state_dict()
    print(f"\nBest Validation F1: {best_val_f1} with Params: {best_params}")
    
    if category == 'sinks':
        final_model = create_model_fn(embedding_dim=embedding_dim, **{k: best_params[k] for k in best_params if k not in ['batch_size','epochs']})
        criterion = nn.CrossEntropyLoss()
        cat_type = 'sinks'
    else:
        final_model = create_model_fn(embedding_dim=embedding_dim, **{k: best_params[k] for k in best_params if k not in ['batch_size','epochs']})
        criterion = nn.BCEWithLogitsLoss()
        cat_type = 'binary'
    final_model.to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params.get('learning_rate'), weight_decay=best_params.get('weight_decay'))
    
    train_val_loader = create_loader(X_train_val, y_train_val, best_params.get('batch_size'))
    test_loader = create_loader(X_test, y_test, best_params.get('batch_size'), shuffle=False)
    
    print("Training final model on train+val data...")
    train_model(final_model, optimizer, criterion, train_val_loader, val_loader, device, best_params.get('epochs'), category=cat_type)
    
    print("Final evaluation on test data:")
    evaluate_model(final_model, test_loader, device, category if category == 'sinks' else 'binary', print_report=True)
    
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{embedding_model}_final_model_{category}.pt')
    scripted_model = torch.jit.script(final_model)
    scripted_model.save(model_path)  
    print(f"{embedding_model} model saved at {model_path}")  

# ------------------------------ Model Creation Functions ------------------------------------

def create_model(learning_rate=0.0001, dropout_rate=0.3, weight_decay=0.0001, activation='relu', embedding_dim=None):
    if embedding_dim is None:
        raise ValueError("Embedding dimension not found")
    return BinaryClassifier(embedding_dim=embedding_dim, dropout_rate=dropout_rate, weight_decay=weight_decay, activation=activation)

def create_model_sinks(learning_rate=0.0001, dropout_rate=0.2, weight_decay=0.0001, activation='elu', embedding_dim=None):
    if embedding_dim is None:
        raise ValueError("Embedding dimension not found")
    # print("Creating multi-class model for sinks")
    return MultiClassClassifier(embedding_dim=embedding_dim, dropout_rate=dropout_rate, weight_decay=weight_decay, activation=activation)

# ------------------------------ Main Script ------------------------------------

if __name__ == '__main__':
    base_path = os.path.join(os.getcwd(), "backend", "src", "bert", "training", "data")
    context = read_json(os.path.join(base_path, 'parsedResults.json'))
    labels = read_json(os.path.join(base_path, 'labels.json'))
    
    variables_param_grid = {
        'model__learning_rate': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'model__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'epochs': [60, 80, 100],
        'n_iter': [4000]
    }
    
    strings_param_grid = {
        'model__learning_rate': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'model__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'epochs': [60, 80, 100],
        'n_iter': [4000]
    }
    
    sinks_param_grid = {
        'model__learning_rate': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'model__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'epochs': [60, 80, 100],
        'n_iter': [1000]
    }
    
    comments_param_grid = {
        'model__learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'model__dropout_rate': [0.2, 0.3],
        'model__activation': ['elu', 'relu'],
        'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64],
        'epochs': [50, 60],
        'n_iter': [5]
    }
    
    params_map = {
        "variables": variables_param_grid,
        "strings": strings_param_grid,
        "comments": comments_param_grid,
        "sinks": sinks_param_grid
    }
    
    categories = [
        # "variables",
        # "strings",
        # "comments",
        "sinks"
    ]
    
    embedding_models = {
        'sentbert': 384 * 2,
        # 't5': 512 * 2,
        # 'roberta': 768 * 2,
        # 'codebert': 768 * 2,
        # 'codellama': 4096 * 2,
        # 'distilbert': 768 * 2,
        # 'albert': 768 * 2,
        # 'longformer': 768 * 2,
    }
    
    for embedding_model, embedding_dim in embedding_models.items():
        for category in categories:
            data = get_context(labels, context, category)
            print(f"{len(data)} {category} entries found using embedding model {embedding_model}.")
            if category == 'sinks':
                model_fn = create_model_sinks
            else:
                model_fn = create_model
            train(category, data, params_map.get(category), model_fn, embedding_model=embedding_model, embedding_dim=embedding_dim)
