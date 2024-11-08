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


# Reconfigure stdout and stderr to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Constants
NUM_CLASSES = 14  # Including non-sink
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
    'sinks': []  # Added for sinks
}

thresholds = {
    'variables': 0.5,
    'strings': 0.95,
    'comments': 0.95,
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

# Calculate sentence embeddings using SentenceTransformer
def calculate_sentbert_vectors(sentences, data_type, item_type, batch_size=64):
    embeddings = []
    total_batches = len(sentences) // batch_size + int(len(sentences) % batch_size != 0)
    progress_tracker = ProgressTracker(total_batches, f"{data_type}-{item_type}-encoding")

    for batch_num, i in enumerate(range(0, len(sentences), batch_size), start=1):
        batch_texts = sentences[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
        progress_tracker.update_progress(1)
    
    return embeddings

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

    for file_name in data:
        items = data[file_name][data_type]
        for item in items:
            item_name = item['name']
            item_methods = item['methods']
            try:
                preprocessed_item = text_preprocess(item_name)

                if data_type == 'variables':
                    item_type = item['type']
                    context = f"Type: {item_type}, Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]
                    
                    context = text_preprocess(context)
                    output.append((file_name, preprocessed_item, context))


                elif data_type == 'strings':
                    # if len(preprocessed_item) < 1:
                    #     continue
                    # else:
                    context = f"Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]

                    context = text_preprocess(context)
                    output.append((file_name, preprocessed_item, context))

                elif data_type == 'comments':
                    output.append((file_name, preprocessed_item))


                elif data_type == 'sinks': 
                    context = f"Context: "
                    for method in item_methods:
                        if method in data[file_name]['methodCodeMap']:
                            context += data[file_name]['methodCodeMap'][method]

                    context = text_preprocess(context)
                    output.append((file_name, preprocessed_item, context))

                projectAllVariables[data_type].append([file_name, item['name']])

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
async def process_data_type(data_type, data_list, final_results, model_path):
    if data_list:
        data_array = np.squeeze(np.array(data_list))

        file_info = data_array[:, 0]
        print(f"Encoding {data_type} name data")
        name_vectors = calculate_sentbert_vectors(data_array[:, 1], data_type, "name")

        if data_type != 'comments':
           context_vectors = calculate_sentbert_vectors(data_array[:, 2], data_type, "context")
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
        for idx, prediction in enumerate(y_predict):
            if data_type == "sinks":  # Special handling for sinks (categorical)
                predicted_category = np.argmax(prediction)  # Get the predicted class
                # print(predicted_category)
                if predicted_category != 0:  # Ignore "non-sink" class (0)
                    sink_type = sink_type_mapping[predicted_category]  # Convert the index to a sink category
                    file_name, sink_name = projectAllVariables[data_type][idx]
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name]["sinks"].append({"name": sink_name, "type": sink_type})
            else:  # Handle non-sink types (strings, variables, comments)
                if prediction >= thresholds.get(data_type):
                    file_name, data = projectAllVariables[data_type][idx]
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name][data_type].append({"name": data})


# Main function
async def main():
    # Set the project path and parsed data file path
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        project_path = os.path.abspath(os.path.join(os.getcwd(), project_path))

        model_path = os.path.join(os.getcwd(), "src", "bert", "models")
    else:
        project_name = "CWEToyDataset"
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
        model_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    parsed_data_file_path = os.path.join(project_path, 'parsedResults.json')

    parsed_data = await read_parsed_data(parsed_data_file_path)

    print("processing files")
    # Process files to extract variables, strings, comments, and sinks concurrently
    variables = process_files(parsed_data, 'variables')
    # strings = process_files(parsed_data, 'strings')
    # comments = process_files(parsed_data, 'comments')
    sinks = process_files(parsed_data, 'sinks')

    final_results = {}

    await process_data_type('variables', variables, final_results, model_path)
    # await process_data_type('strings', strings, final_results, model_path)
    # await process_data_type('comments', comments, final_results, model_path)
    await process_data_type('sinks', sinks, final_results, model_path)

    print("Predicting data done")

    # Format results as JSON
    formatted_results = [{"fileName": file_name, **details} for file_name, details in final_results.items()]

    # Write results to a JSON file asynchronously
    async with aiofiles.open(os.path.join(project_path, 'data.json'), 'w') as f:
        await f.write(json.dumps(formatted_results, indent=4))

if __name__ == '__main__':
    asyncio.run(main())