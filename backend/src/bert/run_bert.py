import os
import sys
import json
import numpy as np
import nltk
from nltk.data import find
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
from keras.models import load_model
import tensorflow as tf
import asyncio
import aiofiles
from collections import deque
from progress_tracker import ProgressTracker
import concurrent.futures
import time


# Reconfigure stdout and stderr to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Constants
DIM = 768

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Ensure the necessary NLTK data packages are downloaded
packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
def check_and_download_nltk_data(package):
    try:
        find(f'{package}')
    except LookupError:
        nltk.download(package)

for package in packages:
    check_and_download_nltk_data(package)

# Dictionary to hold all variables for different types
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

# Sink Type Mapping
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

# Load stop words and add Java keywords to stop words
def load_stop_words():
    stop_words = stopwords.words('english')
    try:
        with open(os.path.join(os.getcwd(), "src", "bert", "java_keywords.txt"), "r") as javaKeywordsFile:
            keywords = javaKeywordsFile.readlines()
            for keyword in keywords:
                new_keyword = keyword.strip()
                stop_words.append(new_keyword)
        return stop_words
    except FileNotFoundError as e:
        return stop_words

# Split camel case words
def camel_case_split(s):
    words = [[s[0]]]
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def text_preprocess(feature_text):
    # Split camel case
    words = camel_case_split(feature_text)
    # Join words back into a string and convert to lowercase
    preprocessed_text = ' '.join(words).lower()
    return preprocessed_text

# Define the function to encode a single batch
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
        # Submit encoding tasks for each precomputed batch
        futures = {
            executor.submit(encode_batch, batch_texts, identifier): identifier
            for identifier, batch_texts in enumerate(batch_texts_list)
        }

        for future in concurrent.futures.as_completed(futures):
            identifier, batch_embeddings = future.result()
            # Store embeddings by their identifier for ordered retrieval later
            for j, embedding in enumerate(batch_embeddings):
                idx = identifier * batch_size + j
                embeddings[data_list[idx]["identifier"]] = embedding

            # Optionally offload progress tracking to another thread
            progress_tracker.update_progress(1)

    # Convert embeddings dictionary to a list, maintaining the original order
    return [embeddings[item["identifier"]] for item in data_list]


# Convert list to string
def list_to_string(lst):
    return ' '.join(lst)

# Concatenate name vectors and context vectors
def concat_name_and_context(name_vec, context_vec):
    return [np.concatenate((name_vec[idx], context_vec[idx]), axis=None) for idx in range(len(name_vec))]

def get_predictions(model, test_x, data_type, batch_size=64):
    # Reshape input data if needed
    total_batches = test_x.shape[0] // batch_size + int(test_x.shape[0] % batch_size != 0)
    prediction_tracker = ProgressTracker(total_batches, f"{data_type}-prediction")

    # Generate predictions in batches
    predictions = []
    for batch_num, i in enumerate(range(0, test_x.shape[0], batch_size), start=1):
        batch_data = test_x[i:i+batch_size]
        batch_predictions = model.predict(batch_data)
        predictions.extend(batch_predictions)
        prediction_tracker.update_progress(1)
    
    return np.array(predictions)

# Process files to extract relevant information and context
def process_files(data, data_type):
    # Set up the progress tracker
    total_progress = sum(len(data[file_name][data_type]) for file_name in data)
    progress_tracker = ProgressTracker((total_progress), f"{data_type}-processing")

    # Preallocate the list with the required size
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
                    context = f"Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]

                    context = text_preprocess(context)

                # elif data_type == 'comments':
                #     output.append((file_name, preprocessed_item))

                elif data_type == 'sinks': 
                    context = f"Context: "
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

def get_context_str(file, var_name):
    context = " "
    for sent in file:
        if len(sent.strip()) <= 2 or sent.strip()[0] == '*' or (sent.strip()[0] == '\\' and sent.strip()[1] == '\\') or (sent.strip()[0] == '\\' and sent.strip()[1] == '*'):
            continue
        if '\''+var_name+'\'' in sent or '"'+var_name+'"' in sent:
            sent = sent.replace(var_name, ' ')
            context = context + sent + " "
    return text_preprocess(context)

# Read parsed data from a JSON file asynchronously
async def read_parsed_data(file_path):
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as json_vars:
            return json.loads(await json_vars.read())
    except Exception as e:
        print(f"Failed to read parsed data from {file_path}: {e}")
        pass

# Load stop words
stop_words = load_stop_words()

# Process each type of data
def process_data_type(data_type, data_list, final_results, model_path):
    if data_list:
        data_array = np.squeeze(np.array(data_list, dtype=object))

        # file_info = data_array[:, 0]
        print(f"Encoding {data_type} name data")
        start = time.time()
        name_vectors = calculate_sentbert_vectors_concurrent(data_list, data_type, "name")
        # name_vectors = calculate_sentbert_vectors(data_array[:, 1], data_type, "name")
        print(f"Encoding {data_type} name data took {time.time() - start} seconds")
        
        if data_type != 'comments':
           context_vectors = calculate_sentbert_vectors_concurrent(data_list, data_type, "context")
           
           print(f"Encoding {data_type} context data")
           concatenated_vectors = concat_name_and_context(name_vectors, context_vectors)
        else:
            concatenated_vectors = np.asarray(name_vectors)

        # Load the model
        if data_type != 'comments':
            model = load_model(os.path.join(model_path, f"{data_type}.keras"))
        else:
            model = load_model(os.path.join(model_path, f"{data_type}.h5"))


        # Run the model to get predictions
        test_x = np.reshape(concatenated_vectors, (-1, DIM)) if data_type != 'comments' else concatenated_vectors
        y_predict = get_predictions(model, test_x, data_type)

        # Collect predictions and update saving progress
        # Collect predictions and save with correct links
        for idx, prediction in enumerate(y_predict):
            item = data_list[idx]
            file_name, original_name = item["file_name"], item["original_name"]
            confidence = str(prediction[0]) if data_type != "sinks" else str(prediction[np.argmax(prediction)])
            
            if data_type == "sinks":
                prediction = np.argmax(prediction)
                if prediction != 0:  # Ignore "non-sink" class
                    sink_type = sink_type_mapping[int(prediction)]
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name]["sinks"].append({"name": original_name, "type": sink_type, "confidence": confidence})
            else: 
                if prediction >= thresholds.get(data_type):
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name][data_type].append({"name": original_name, "confidence": confidence})

# Main function
async def main():
    # Set the project path and parsed data file path
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        project_path = os.path.abspath(os.path.join(os.getcwd(), project_path))

        model_path = os.path.join(os.getcwd(), "src", "bert", "models")
    else:
        project_name = "comments"
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
        model_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    parsed_data_file_path = os.path.join(project_path, 'parsedResults.json')

    parsed_data = await read_parsed_data(parsed_data_file_path)
    final_results = {}

    print("processing files")
    
    # Process files to extract variables, strings, comments, and sinks concurrently
    variables = process_files(parsed_data, 'variables')
    process_data_type('variables', variables, final_results, model_path)

    # strings = process_files(parsed_data, 'strings')
    # process_data_type('strings', strings, final_results, model_path)

    # comments = process_files(parsed_data, 'comments')
    # process_data_type('comments', comments, final_results, model_path)

    sinks = process_files(parsed_data, 'sinks')
    process_data_type('sinks', sinks, final_results, model_path)

    print("Predicting data done")

    # Format results as JSON
    formatted_results = [{"fileName": file_name, **details} for file_name, details in final_results.items()]

    # Write results to a JSON file asynchronously
    async with aiofiles.open(os.path.join(project_path, 'data.json'), 'w') as f:
        await f.write(json.dumps(formatted_results, indent=4))

if __name__ == '__main__':
    asyncio.run(main())