import os
import sys
import json
import time
import asyncio
import concurrent.futures
import numpy as np
import nltk
from nltk.data import find
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
# from keras.models import load_model
# import tensorflow as tf
import torch
import aiofiles
from collections import deque
from progress_tracker import ProgressTracker

# Reconfigure stdout and stderr to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
DIM = 768

# -----------------------------------------------------------------------------
# PyTorch Model Definitions (from training)
# -----------------------------------------------------------------------------
import torch.nn as nn

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
        super(BinaryClassifier, self).__init__()
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
        super(MultiClassClassifier, self).__init__()
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

# -----------------------------------------------------------------------------
# Other Helper Functions
# -----------------------------------------------------------------------------

lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
def check_and_download_nltk_data(package):
    try:
        find(f'{package}')
    except LookupError:
        nltk.download(package)
for package in packages:
    check_and_download_nltk_data(package)

projectAllVariables = {
    'variables': [],
    'strings': [],
    'comments': [],
    'sinks': [] 
}

thresholds = {
    'variables': 0.75,
    'strings': 0.95,
    'comments': 0.75,
    'sinks': 0.5, 
}

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

def load_stop_words():
    stop_words = stopwords.words('english')
    try:
        with open(os.path.join(os.getcwd(), "src", "bert", "java_keywords.txt"), "r") as javaKeywordsFile:
            keywords = javaKeywordsFile.readlines()
            for keyword in keywords:
                stop_words.append(keyword.strip())
        return stop_words
    except FileNotFoundError:
        return stop_words

def camel_case_split(s):
    words = [[s[0]]]
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def text_preprocess(feature_text):
    words = camel_case_split(feature_text)
    preprocessed_text = ' '.join(words).lower()
    return preprocessed_text

def list_to_string(lst):
    return ' '.join(lst)

def concat_name_and_context(name_vec, context_vec):
    return [np.concatenate((name_vec[idx], context_vec[idx]), axis=None) for idx in range(len(name_vec))]

def encode_batch(batch_texts, identifier):
    batch_embeddings = model.encode(batch_texts)
    return identifier, batch_embeddings

def calculate_sentbert_vectors_concurrent(data_list, data_type, item_type, batch_size=64):
    embeddings = {}
    total_batches = len(data_list) // batch_size + int(len(data_list) % batch_size != 0)
    progress_tracker = ProgressTracker(total_batches, f"{data_type}-{item_type}-encoding")
    index = "preprocessed_item" if item_type == "name" else "context"
    data_batches = [data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]
    batch_texts_list = [[item[index] for item in batch] for batch in data_batches]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(encode_batch, batch_texts, identifier): identifier
            for identifier, batch_texts in enumerate(batch_texts_list)
        }
        for future in concurrent.futures.as_completed(futures):
            identifier, batch_embeddings = future.result()
            for j, embedding in enumerate(batch_embeddings):
                idx = identifier * batch_size + j
                embeddings[data_list[idx]["identifier"]] = embedding
            progress_tracker.update_progress(1)
    return [embeddings[item["identifier"]] for item in data_list]

# -----------------------------------------------------------------------------
# Prediction Function Using PyTorch
# -----------------------------------------------------------------------------

def get_predictions(model_pt, test_x, data_type, batch_size=64):
    total_batches = test_x.shape[0] // batch_size + int(test_x.shape[0] % batch_size != 0)
    prediction_tracker = ProgressTracker(total_batches, f"{data_type}-prediction")
    predictions = []
    model_pt.to(device)
    model_pt.eval()
    with torch.no_grad():
        for i in range(0, test_x.shape[0], batch_size):
            batch_data = test_x[i:i+batch_size]
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)
            batch_output = model_pt(batch_tensor)
            if not isinstance(batch_output, torch.Tensor):
                batch_output = torch.tensor(batch_output)
            predictions.extend(batch_output.cpu().numpy())
            prediction_tracker.update_progress(1)
    return np.array(predictions)

# -----------------------------------------------------------------------------
# Data Processing Functions
# -----------------------------------------------------------------------------

def process_files(data, data_type):
    total_progress = sum(len(data[file_name][data_type]) for file_name in data)
    progress_tracker = ProgressTracker(total_progress, f"{data_type}-processing")
    output = []
    identifier = 0
    for file_name in data:
        items = data[file_name][data_type]
        for item in items:
            item_name = item['name']
            item_methods = item['methods']
            context = None
            try:
                preprocessed_item = text_preprocess(item_name)
                if data_type == 'variables':
                    item_type = item['type']
                    context = f"Type: {item_type}, Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]
                    context = text_preprocess(context)
                elif data_type == 'strings':
                    context = "Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]
                    context = text_preprocess(context)
                # elif data_type == 'comments':
                #     output.append((file_name, preprocessed_item))
                elif data_type == 'sinks': 
                    context = "Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]
                    context = text_preprocess(context)
                    # output.append((file_name, preprocessed_item, context))
                output.append({
                    "identifier": identifier,
                    "file_name": file_name,
                    "preprocessed_item": preprocessed_item,
                    "context": context,
                    "original_name": item_name
                })
                identifier += 1
            except Exception as e:
                print(f"Error processing file {file_name} and {data_type[:-1]} {item}: {e}")
            progress_tracker.update_progress(1)
    return output

def process_data_type(data_type, data_list, final_results, model_path):
    if data_list:
        print(f"Encoding {data_type} name data")
        start = time.time()
        name_vectors = calculate_sentbert_vectors_concurrent(data_list, data_type, "name")
        print(f"Encoding {data_type} name data took {time.time() - start} seconds")
        if data_type != 'comments':
            context_vectors = calculate_sentbert_vectors_concurrent(data_list, data_type, "context")
            print(f"Encoding {data_type} context data")
            concatenated_vectors = concat_name_and_context(name_vectors, context_vectors)
        else:
            concatenated_vectors = np.asarray(name_vectors)

        # Instantiate appropriate model and load its state dictionary
        if data_type != 'sinks':
            model_pt = BinaryClassifier(embedding_dim=DIM, dropout_rate=0.3, weight_decay=0.0001, activation='relu')
        else:
            model_pt = MultiClassClassifier(embedding_dim=DIM, dropout_rate=0.2, weight_decay=0.0001, activation='elu')
        state_dict = torch.load(os.path.join(model_path, f"{data_type}.pt"), map_location=device)
        model_pt.load_state_dict(state_dict)

        test_x = np.reshape(concatenated_vectors, (-1, DIM)) if data_type != 'comments' else concatenated_vectors
        y_predict = get_predictions(model_pt, test_x, data_type)

        for idx, prediction in enumerate(y_predict):
            item = data_list[idx]
            file_name, original_name = item["file_name"], item["original_name"]
            if data_type != "sinks":
                confidence = str(prediction[0])
            else:
                class_idx = np.argmax(prediction)
                confidence = str(prediction[class_idx])
            if data_type == "sinks":
                if np.argmax(prediction) != 0:  # Ignore "non-sink" class
                    sink_type = sink_type_mapping[int(np.argmax(prediction))]
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name]["sinks"].append({"name": original_name, "type": sink_type, "confidence": confidence})
            else: 
                if prediction >= thresholds.get(data_type):
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name][data_type].append({"name": original_name, "confidence": confidence})

# -----------------------------------------------------------------------------
# Async File I/O and Main Execution
# -----------------------------------------------------------------------------

async def read_parsed_data(file_path):
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as json_vars:
            return json.loads(await json_vars.read())
    except Exception as e:
        print(f"Failed to read parsed data from {file_path}: {e}")
        return None

stop_words = load_stop_words()

async def main():
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        project_path = os.path.abspath(os.path.join(os.getcwd(), project_path))
        model_path = os.path.join(os.getcwd(), "src", "bert", "models")
    else:
        project_name = "PowerJob-5.1.1"
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
        model_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    parsed_data_file_path = os.path.join(project_path, 'parsedResults.json')
    parsed_data = await read_parsed_data(parsed_data_file_path)
    if not parsed_data:
        print("No parsed data found. Exiting.")
        return
    final_results = {}
    print("Processing files")
    
    variables = process_files(parsed_data, 'variables')
    process_data_type('variables', variables, final_results, model_path)
    
    # strings = process_files(parsed_data, 'strings')
    # process_data_type('strings', strings, final_results, model_path)
    
    # comments = process_files(parsed_data, 'comments')
    # process_data_type('comments', comments, final_results, model_path)
    
    sinks = process_files(parsed_data, 'sinks')
    process_data_type('sinks', sinks, final_results, model_path)
    
    print("Predicting data done")
    formatted_results = [{"fileName": file_name, **details} for file_name, details in final_results.items()]
    async with aiofiles.open(os.path.join(project_path, 'data.json'), 'w') as f:
        await f.write(json.dumps(formatted_results, indent=4))

if __name__ == '__main__':
    asyncio.run(main())
