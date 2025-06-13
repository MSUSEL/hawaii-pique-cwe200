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
import torch
import aiofiles
from collections import deque
from progress_tracker import ProgressTracker

# Reconfigure stdout and stderr to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Constants
DIM = 768

# -----------------------------------------------------------------------------
# Other Helper Functions
# -----------------------------------------------------------------------------

# lemmatizer = WordNetLemmatizer()

# Load SentBert model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

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

def concat_name_and_context(name_vec, context_vec):
    """Concatenate name and context vectors for each item.
    Combines semantic info from both the item and its context.
    
    :param name_vec: list of name embeddings
    :param context_vec: list of context embeddings
    :returns: list of concatenated embeddings
    """
    return [np.concatenate((name_vec[idx], context_vec[idx]), axis=None) for idx in range(len(name_vec))]

def encode_batch(batch_texts, identifier):
    """Encode a batch of text using Sentence-BERT model.
    Encodes inputs into vector form for downstream ML usage with our model.
    
    :param batch_texts: list of strings to encode
    :param identifier: unique batch identifier
    :returns: tuple of (identifier, list of embeddings)
    """
    batch_embeddings = model.encode(batch_texts)
    return identifier, batch_embeddings

def calculate_sentbert_vectors_concurrent(data_list, data_type, item_type, batch_size=64):
    """Calculate Sentence-BERT embeddings in parallel using threads.
    Speeds up vector generation for large datasets.

    :param data_list: list of items containing preprocessed text
    :param data_type: type of data (e.g., variables, strings)
    :param item_type: text field to encode ('name' or 'context')
    :param batch_size: number of items per batch
    :returns: list of embeddings aligned with data_list order
    """
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
    """Run predictions using a PyTorch model on input data.
    Generates classification confidence scores for each input.
    
    :param model_pt: PyTorch model
    :param test_x: input data as NumPy array
    :param data_type: data type (e.g., variables, sinks)
    :param batch_size: number of samples per inference batch
    :returns: array of predictions
    """
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
            # Apply appropriate activation to get confidence scores between 0 and 1
            if data_type == "sinks":
                # For multi-class classification, use softmax along the class dimension (assumed dim=1)
                batch_output = torch.softmax(batch_output, dim=1)
            predictions.extend(batch_output.cpu().numpy())
            prediction_tracker.update_progress(1)
    return np.array(predictions)


# -----------------------------------------------------------------------------
# Data Processing Functions
# -----------------------------------------------------------------------------

def process_files(data, data_type):
    """Extract and preprocess data items from the parsed dataset.
    Converts raw analysis data into model-compatible format.
    
    :param data: full parsed JSON data (From ParseJava)
    :param data_type: type of items to extract (e.g., variables, sinks)
    :returns: list of processed items with metadata and context
    """
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
                preprocessed_item = item_name
                if data_type == 'variables':
                    item_type = item['type']
                    context = f"Type: {item_type}, Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]
                    context = context
                elif data_type == 'strings':
                    context = "Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]
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
    """Encode data items, run predictions, and store confident results.
    Full pipeline step for encoding, inference, and result filtering.
    
    :param data_type: type of item (variables, strings, sinks, etc.)
    :param data_list: preprocessed data list
    :param final_results: output dictionary to populate with predictions
    :param model_path: path to PyTorch model directory
    :returns: None
    """
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

        model_pt = torch.jit.load(os.path.join(model_path, f"{data_type}.pt"), map_location=device)

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
                if float(prediction[0]) >= thresholds.get(data_type):
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name][data_type].append({"name": original_name, "confidence": confidence})

# -----------------------------------------------------------------------------
# Async File I/O and Main Execution
# -----------------------------------------------------------------------------

async def read_parsed_data(file_path):
    """Read parsed JSON data asynchronously.
    Loads data from disk efficiently without blocking.
    
    :param file_path: path to the parsed JSON file
    :returns: dictionary loaded from JSON file, or None on failure
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as json_vars:
            return json.loads(await json_vars.read())
    except Exception as e:
        print(f"Failed to read parsed data from {file_path}: {e}")
        return None


async def main() -> None:
    """Main function to run encoding and prediction pipeline asynchronously.
    Coordinates all processing steps for a given project’s data.
    
    :returns: None
    """
    if len(sys.argv) > 1:
        print(f"Args = {sys.argv}")
        project_path = sys.argv[1]
        project_path = os.path.abspath(os.path.join(os.getcwd(), project_path))
        model_path = os.path.join(os.getcwd(), "src", "bert", "models")
    else:
        project_name = "CWEToyDataset"
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
        model_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    parsed_data_file_path = os.path.join(project_path, 'parsedResults.json')
    parsed_data = await read_parsed_data(parsed_data_file_path)
    if not parsed_data:
        print("No parsed data found. Exiting.")
        return
    final_results = {}
    print("Processing files")
    
    # Uncomment the types you want to process
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
