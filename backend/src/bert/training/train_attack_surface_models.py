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


def read_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        return json.load(file)

def calculate_sentbert_vectors(sentences, batch_size=64):
    """Calculate Sentence-BERT embeddings in parallel using threads.
    Speeds up vector generation for large datasets.

    :param data_list: list of items containing preprocessed text
    :param data_type: type of data (e.g., variables, strings)
    :param item_type: text field to encode ('name' or 'context')
    :param batch_size: number of items per batch
    :returns: list of embeddings aligned with data_list order
    """
    from sentence_transformers import SentenceTransformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    print("Encoding sentences with SentenceTransformer...")
    embeddings = model_transformer.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def calculate_codet5_vectors(sentences,
                             model_name='Salesforce/codet5-base',
                             batch_size=32):
    """Calculate CodeT5 embeddings in parallel using threads.
    Speeds up vector generation for large datasets.

    :param data_list: list of items containing preprocessed text
    :param data_type: type of data (e.g., variables, strings)
    :param item_type: text field to encode ('name' or 'context')
    :param batch_size: number of items per batch
    :returns: list of embeddings aligned with data_list order
    """
    from transformers import AutoTokenizer, T5EncoderModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer & encoder?only model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model     = T5EncoderModel.from_pretrained(model_name)
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

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)                            # encoder output only
            last_hidden = outputs.last_hidden_state              # [batch, seq_len, dim]
            pooled      = last_hidden.mean(dim=1)                # mean over seq_len
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def calculate_codebert_vectors(sentences, model_name='microsoft/codebert-base', batch_size=32):
    """Calculate Code-BERT embeddings in parallel using threads.
    Speeds up vector generation for large datasets.

    :param data_list: list of items containing preprocessed text
    :param data_type: type of data (e.g., variables, strings)
    :param item_type: text field to encode ('name' or 'context')
    :param batch_size: number of items per batch
    :returns: list of embeddings aligned with data_list order
    """
    
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


def calculate_longformer_vectors(sentences, model_name='allenai/longformer-base-4096', batch_size=32):
    """Calculate longformer embeddings in parallel using threads.
    Speeds up vector generation for large datasets.

    :param data_list: list of items containing preprocessed text
    :param data_type: type of data (e.g., variables, strings)
    :param item_type: text field to encode ('name' or 'context')
    :param batch_size: number of items per batch
    :returns: list of embeddings aligned with data_list order
    """
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
    """Concatenate name and context vectors for each item.
    Combines semantic info from both the item and its context.
    
    :param name_vec: list of name embeddings
    :param context_vec: list of context embeddings
    :returns: list of concatenated embeddings
    """
    total_vecs = []
    for idx in range(len(name_vecs)):
        total_vecs.append(np.concatenate((name_vecs[idx], context_vecs[idx]), axis=None))
    return total_vecs

# ------------------------------ Model Definitions ------------------------------------

def get_activation(act_name):
    """Get the activation function based on the provided name.
    param act_name: name of the activation function
    :returns: activation function object
    """
    if act_name is None:
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
    """Binary classifier model with residual connections and dropout.
    This model is designed to process embeddings and predict binary labels.
    """
    def __init__(self, embedding_dim, dropout_rate, weight_decay, activation):
        """Initialize the model with the given parameters.
        :param embedding_dim: dimension of the input embeddings
        :param dropout_rate: dropout rate for regularization
        :param weight_decay: weight decay for optimizer
        :param activation: activation function to use in the model
        """
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
        """Forward pass through the model.
        :param x: input tensor of shape (batch_size, embedding_dim)
        :returns: output tensor of shape (batch_size, 1) after sigmoid activation
        """
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
    """Multi-class classifier model with dropout.
    This model is designed to process embeddings and predict multiple classes. Used for sink types.
    """
    def __init__(self, embedding_dim, dropout_rate, weight_decay, activation):
        """Initialize the model with the given parameters.
        :param embedding_dim: dimension of the input embeddings
        :param dropout_rate: dropout rate for regularization
        :param weight_decay: weight decay for optimizer
        :param activation: activation function to use in the model
        """
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
        """Forward pass through the model.
        :param x: input tensor of shape (batch_size, embedding_dim)
        :returns: output tensor of shape (batch_size, 8) after softmax activation
        """
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

def evaluate_model(model, loader, device, category):
    """Evaluate the model on the given data loader.
    :param model: trained model to evaluate
    :param loader: data loader for evaluation
    :param device: device to run the model on (CPU or GPU)
    :param category: type of data (variables, sinks, etc.)
    :returns: dictionary of evaluation metrics
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            out = model(Xb)
            if category == 'sinks':
                p = torch.argmax(out, dim=1).cpu().numpy()
            else:
                p = (out > 0.5).int().cpu().numpy().flatten()
            preds.extend(p)
            trues.extend(yb.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    # overall metrics
    prec = metrics.precision_score(trues, preds, average='weighted', zero_division=0)
    rec  = metrics.recall_score(   trues, preds, average='weighted', zero_division=0)
    f1   = metrics.f1_score(      trues, preds, average='weighted', zero_division=0)
    acc  = metrics.accuracy_score(trues, preds)

    # classification report string + confusion matrix
    names      = list(sink_type_mapping.values()) if category=='sinks' else ["Non-sensitive","Sensitive"]
    report_str = metrics.classification_report(trues, preds, target_names=names, zero_division=0)
    conf_mat   = metrics.confusion_matrix(trues, preds)

    return {
        'precision':       prec,
        'recall':          rec,
        'f1':              f1,
        'accuracy':        acc,
        'report_str':      report_str,
        'confusion_matrix': conf_mat
    }



def train_model(model, optimizer, criterion, train_loader, val_loader, device, epochs, early_stop_patience=10, category='binary'):
    """Train the model for a specified number of epochs.
    :param model: model to train
    :param optimizer: optimizer for training
    :param criterion: loss function
    :param train_loader: data loader for training
    :param val_loader: data loader for validation
    :param device: device to run the model on (CPU or GPU)
    :param epochs: number of epochs to train
    :param early_stop_patience: number of epochs to wait for improvement before stopping
    :param category: type of data (variables, sinks, etc.)
    :returns: best validation F1 score and the trained model
    """
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
        val_metrics = evaluate_model(model, val_loader, device, category)
        val_f1 = val_metrics['f1']

        
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
    """Extract context data for a specific category from the labels and context.
    context comes from the parsedResults.json file.
    :param labels: list of label entries
    :param context: dictionary of context data
    :param category: category to extract (variables, strings, sinks, comments)
    :returns: list of data entries for the specified category
    """
    data = []
    for label_entry in labels:
        file_name = label_entry.get('fileName')
        if file_name not in context:
            print(f"Warning: {file_name} not found in context data.")
            continue
        file_context = context[file_name]
        method_code_map = file_context.get('methodCodeMap', {})

        # Skip if this category isn't in the labels
        if category not in label_entry:
            continue

        for label_item in label_entry[category]:
            # Guard against missing or non?string names
            name = label_item.get('name')
            if not isinstance(name, str) or not name.strip():
                print(
                    f"??  Skipping invalid name ({name!r}) "
                    f"in category '{category}' for file '{file_name}': {label_item}"
                )
                continue
            name = name.strip()

            # Find the matching context item by name
            matched_context_item = next(
                (
                    context_item
                    for context_item in file_context.get(category, [])
                    if name == str(context_item.get('name', '')).strip()
                ),
                None
            )
            if not matched_context_item:
                continue

            # Binary/sink label
            if category == 'sinks':
                binary_label = sink_type_mapping_rev.get(label_item.get('type'), 0)
            else:
                binary_label = 1 if label_item.get('IsSensitive') == 'Yes' else 0

            # Build aggregated context
            aggregated_context = ''
            methods = matched_context_item.get('methods', [])
            if category == 'variables':
                aggregated_context = f"Type: {matched_context_item.get('type')}, Context: "
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]
            elif category == 'strings':
                aggregated_context = "Context: "
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]
            elif category == 'sinks':
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]

            # Append to data in the correct format
            if category in ['variables', 'strings']:
                data.append([
                    name,
                    aggregated_context,
                    binary_label
                ])
            elif category == 'comments':
                data.append([
                    name,
                    None,
                    binary_label
                ])
            elif category == 'sinks':
                data.append([
                    name,
                    aggregated_context,
                    binary_label
                ])
    return data


# ------------------------------ Training Function ------------------------------------

def train(category, data, param_grid, create_model_fn, embedding_model='sentbert', embedding_dim=384*2):
    """Train a model on the provided data using the specified parameters.
    :param category: category of data (variables, strings, sinks, comments)
    :param data: list of data entries for the specified category
    :param param_grid: dictionary of hyperparameters for the model
    :param create_model_fn: function to create the model
    :param embedding_model: name of the embedding model to use
    :param embedding_dim: dimension of the embeddings
    :returns: dictionary of evaluation metrics
    """
    variable_array = np.array(data, dtype=object)
    if embedding_model == 'sentbert':
        get_embeddings = calculate_sentbert_vectors
    elif embedding_model == 'codet5':
        get_embeddings = calculate_codet5_vectors
    elif embedding_model == 'codebert':
        get_embeddings = calculate_codebert_vectors
    elif embedding_model == 'longformer': 
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

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{embedding_model}_final_model_{category}.pt')
    scripted_model = torch.jit.script(final_model)
    scripted_model.save(model_path)  
    print(f"{embedding_model} model saved at {model_path}")  


    final_metrics = evaluate_model(
        final_model,
        test_loader,
        device,
        category if category == 'sinks' else 'binary'
    )
    return final_metrics

# ------------------------------ Model Creation Functions ------------------------------------

def create_model(learning_rate=0.0001, dropout_rate=0.3, weight_decay=0.0001, activation='relu', embedding_dim=None):
    """Create a binary classifier model with the specified parameters.
    :param learning_rate: learning rate for the optimizer
    :param dropout_rate: dropout rate for regularization
    :param weight_decay: weight decay for optimizer
    :param activation: activation function to use in the model
    :param embedding_dim: dimension of the input embeddings
    :returns: BinaryClassifier instance
    """
    if embedding_dim is None:
        raise ValueError("Embedding dimension not found")
    return BinaryClassifier(embedding_dim=embedding_dim, dropout_rate=dropout_rate, weight_decay=weight_decay, activation=activation)

def create_model_sinks(learning_rate=0.0001, dropout_rate=0.2, weight_decay=0.0001, activation='elu', embedding_dim=None):
    """Create a multi-class classifier model for sink types with the specified parameters.
    :param learning_rate: learning rate for the optimizer
    :param dropout_rate: dropout rate for regularization
    :param weight_decay: weight decay for optimizer
    :param activation: activation function to use in the model
    :param embedding_dim: dimension of the input embeddings
    :returns: MultiClassClassifier instance
    """
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
        'n_iter': [250]
    }
    
    strings_param_grid = {
        'model__learning_rate': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'model__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'epochs': [60, 80, 100],
        'n_iter': [250]
    }
    
    sinks_param_grid = {
        'model__learning_rate': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'model__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'epochs': [60, 80, 100],
        'n_iter': [250]
    }
    
    comments_param_grid = {
        'model__learning_rate': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'model__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
        'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
        'batch_size': [32, 64, 96],
        'epochs': [60, 80, 100],
        'n_iter': [250]
    }
    
    params_map = {
        "variables": variables_param_grid,
        "strings": strings_param_grid,
        "comments": comments_param_grid,
        "sinks": sinks_param_grid
    }
    
    categories = [
        "variables",
        "strings",
        "comments",
        "sinks"
    ]
    
    embedding_models = {
        'sentbert': 384 * 2,
        't5': 512 * 2,
        'codet5': 768 * 2,
        'codebert': 768 * 2,
        # 'longformer': 768 * 2,
    }

    summary = {}
    for emb_name, emb_dim in embedding_models.items():
        summary[emb_name] = {}
        for cat in categories:
            data = get_context(labels, context, cat)
            print(f"\n--- {emb_name.upper()} / {cat} ({len(data)} samples) ---")
            factory = create_model_sinks if cat == 'sinks' else create_model
            metrics_dict = train(cat, data, params_map[cat], factory, embedding_model=emb_name, embedding_dim=emb_dim)
            summary[emb_name][cat] = metrics_dict

    output_path = os.path.join(os.getcwd(), "backend", "src", "bert", "training", "results.txt")

    with open(output_path, "w") as out:
        out.write("="*60 + "\n")
        out.write("      FINAL SUMMARY     \n")
        out.write("="*60 + "\n\n")

        for emb, cat_map in summary.items():
            for cat, m in cat_map.items():
                out.write("\n" + "="*50 + "\n")
                out.write(f"Model: {emb}  -  Category: {cat}\n")
                out.write("="*50 + "\n")
                out.write("Final Evaluation Results:\n")
                out.write(f"Precision: {m['precision']}\n")
                out.write(f"Recall:    {m['recall']}\n")
                out.write(f"F1 Score:  {m['f1']}\n")
                out.write(f"Accuracy:  {m['accuracy']}\n")
                out.write("-"*48 + "\n")
                out.write(m['report_str'] + "\n")
                out.write("Confusion Matrix:\n")
                out.write(np.array2string(m['confusion_matrix']) + "\n\n")

    print(f"Summary written to {output_path}")