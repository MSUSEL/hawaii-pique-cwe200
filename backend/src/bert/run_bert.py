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


# Reconfigure stdout and stderr to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Constants
BATCH_SIZE = 32
EPOCHS = 500
DIM = 768
NUM_CLASSES = 14  # Including non-sink

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Ensure the necessary NLTK data packages are downloaded
packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
def check_and_download_nltk_data(package):
    try:
        find(f'tokenizers/{package}')
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
    'variables': 0.7,
    'strings': 0.95,
    'comments': 0.95,
    'sinks': 0.99 
}

# Sink Type Mapping
sink_type_mapping = {
    0: "non-sink",  # Non-sink class
    1: "I/O Sink",
    2: "Print Sink",
    3: "Network Sink",
    4: "Log Sink",
    5: "Database Sink",
    6: "Email Sink",
    7: "IPC Sink",
    8: "Clipboard Sink",
    9: "GUI Display Sink",
    10: "RPC Sink",
    11: "Environment Variable Sink",
    12: "Command Execution Sink",
    13: "Configuration Sink"
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

# Preprocess text by tokenizing, removing stop words, and lemmatizing
def text_preprocess(feature_text):
    word_tokens = word_tokenize(feature_text)
    if '\n' in word_tokens:
        word_tokens.remove('\n')
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    for i in range(len(filtered_sentence)):
        filtered_sentence[i:i + 1] = camel_case_split(filtered_sentence[i])
    tagged_sentence = pos_tag(filtered_sentence)
    lemmatized_sentence = []
    for word, tag in tagged_sentence:
        ntag = tag[0].lower()
        if ntag in ['a', 'r', 'n', 'v']:
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), ntag))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower()))
    return list_to_string(lemmatized_sentence)

# Calculate sentence embeddings using SentenceTransformer
async def calculate_sentbert_vectors(api_lines):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = []
    for api in api_lines:
        preprocessed_tokens = text_preprocess(api)
        api = ' '.join(preprocessed_tokens)
        sentences.append(api)
    embeddings = await asyncio.to_thread(model.encode, sentences)
    return embeddings

# Convert list to string
def list_to_string(lst):
    return ' '.join(lst)

# Concatenate name vectors and context vectors
def concat_name_and_context(name_vec, context_vec):
    return [np.concatenate((name_vec[idx], context_vec[idx]), axis=None) for idx in range(len(name_vec))]


# Process files to extract relevant information and context
def process_files(data, files_dict, data_type):
    # Set up the progress tracker
    total_progress = sum(len(data[file_name][data_type]) for file_name in data)
    progress_tracker = ProgressTracker((total_progress), f"{data_type}-processing")

    # Preallocate the list with the required size
    output = [None] * total_progress
    index = 0  # To keep track of the current index in the preallocated list

    for file_name in data:
        items = data[file_name][data_type]
        for item in items:
            try:
                preprocessed_item = text_preprocess(item)

                if data_type == 'variables':
                    context = get_context(files_dict[file_name], item)
                    output[index] = (file_name, preprocessed_item, context)

                elif data_type == 'strings':
                    if len(preprocessed_item) == 0:
                        context = ""
                    else:
                        context = get_context_str(files_dict[file_name], item)
                    output[index] = (file_name, preprocessed_item, context)

                elif data_type == 'comments':
                    context = get_context(files_dict[file_name], item)
                    output[index] = (file_name, preprocessed_item)

                elif data_type == 'sinks': 
                    context = get_context(files_dict[file_name], item)
                    output[index] = (file_name, preprocessed_item, context)

                projectAllVariables[data_type].append([file_name, item])
                index += 1  # Move to the next position in the preallocated list

            except Exception as e:
                print(f"Error processing file {file_name} and {data_type[:-1]} {item}: {e}")
            progress_tracker.update_progress(1)

    return output



# Get the context of a variable within a file
def get_context(file, var_name):
    context = ""
    for sentence in file.split('\n'):
        sentence_tokens = word_tokenize(sentence)
        if var_name in sentence_tokens:
            sentence = sentence.replace(var_name, '')  # Remove the variable name from the context
            context = context + sentence + " "
    return text_preprocess(context)

async def get_context_str(file, var_name):
    context = " "
    for sent in file:
        if len(sent.strip()) <= 2 or sent.strip()[0] == '*' or (sent.strip()[0] == '\\' and sent.strip()[1] == '\\') or (sent.strip()[0] == '\\' and sent.strip()[1] == '*'):
            continue
        if '\''+var_name+'\'' in sent or '"'+var_name+'"' in sent:
            sent = sent.replace(var_name, ' ')
            context = context + sent + " "
    return await text_preprocess(context)

# Read Java files from a directory asynchronously
async def read_java_files(directory):
    print(directory)
    java_files_with_content = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        file_content = await f.read()
                    java_files_with_content[os.path.basename(file_path)] = file_content
                except Exception as e:
                    print(e)
    return java_files_with_content

# Read parsed data from a JSON file asynchronously
async def read_parsed_data(file_path):
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as json_vars:
            return json.loads(await json_vars.read())
    except Exception as e:
        print(f"Failed to read parsed data from {file_path}: {e}")
        pass

def filter_parsed_data(parsed_data, files_dict):
    # Create a set of keys from files_dict for fast lookups
    files_dict_keys = set(files_dict.keys())

    # Loop through parsed_data and remove any items not present in files_dict
    keys_to_remove = [key for key in parsed_data if key not in files_dict_keys]

    # Remove the keys from parsed_data
    for key in keys_to_remove:
        del parsed_data[key]

    return parsed_data

# Load stop words
stop_words = load_stop_words()

# Process each type of data
async def process_data_type(data_type, data_list, final_results, model_path):
    if data_list:
        data_array = np.squeeze(np.array(data_list))
        prediction_tracker = ProgressTracker(len(data_array), f'{data_type}-prediction')
        saving_tracker = ProgressTracker(len(data_array), f'{data_type}-saving')

        file_info = data_array[:, 0]
        name_vectors = await calculate_sentbert_vectors(data_array[:, 1])

        if data_type != 'comments':
            context_vectors = await calculate_sentbert_vectors(data_array[:, 2])
            concatenated_vectors = concat_name_and_context(name_vectors, context_vectors)
        else:
            concatenated_vectors = name_vectors

        # Load the model
        if data_type == 'sinks':
            model = load_model(os.path.join(model_path, f"{data_type}.keras"))
        else:
            model = load_model(os.path.join(model_path, f"{data_type}.h5"))


        # Run the model to get predictions
        test_x = np.reshape(concatenated_vectors, (-1, DIM)) if data_type != 'comments' else concatenated_vectors
        prediction_tracker.total_steps =  test_x.shape[0]
        y_predict = await asyncio.to_thread(model.predict, test_x)

        prediction_tracker.update_progress(len(data_array))
        prediction_tracker.complete()

        # Collect predictions and update saving progress
        for idx, prediction in enumerate(y_predict):
            if data_type == "sinks":  # Special handling for sinks (categorical)
                predicted_category = np.argmax(prediction)  # Get the predicted class
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

        saving_tracker.complete()

# Main function
async def main():
    # Set the project path and parsed data file path
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        project_path = os.path.abspath(os.path.join(os.getcwd(), project_path))

        model_path = os.path.join(os.getcwd(), "src", "bert", "models")
    else:
        project_name = "SmallTest"
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
        model_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    parsed_data_file_path = os.path.join(project_path, 'parsedResults.json')

    # Read Java files and parsed data asynchronously
    print(project_path)
    files_dict = await read_java_files(project_path)
    parsed_data = await read_parsed_data(parsed_data_file_path)

    # In case a file can't be found in the directory, remove it from the parsed data
    parsed_data = filter_parsed_data(parsed_data, files_dict)

    print(len(parsed_data))
    print(len(files_dict))

    print("processing files")
    # Process files to extract variables, strings, comments, and sinks concurrently
    variables = process_files(parsed_data, files_dict, 'variables'),


    final_results = {}

    print("Predicting data")
    await process_data_type('variables', variables, final_results, model_path)
    print("Predicting data done")


    # Format results as JSON
    formatted_results = [{"fileName": file_name, **details} for file_name, details in final_results.items()]

    # Write results to a JSON file asynchronously
    async with aiofiles.open(os.path.join(project_path, 'data.json'), 'w') as f:
        await f.write(json.dumps(formatted_results, indent=4))

class ProgressTracker:
    def __init__(self, total_steps, progress_type):
        self.total_steps = total_steps
        self.current_progress = 0
        self.progress_type = progress_type
        self.last_progress_percentage = -1  # Keep track of the last printed progress percentage

    def update_progress(self, step_increment):
        if self.total_steps > 0:
            self.current_progress += step_increment
            progress_percentage = min(round((self.current_progress / self.total_steps) * 100), 100)
            if progress_percentage != self.last_progress_percentage:  # Only print if the percentage has changed
                print(json.dumps({'type': self.progress_type, 'progress': progress_percentage}), flush=True)
                self.last_progress_percentage = progress_percentage  # Update the last progress percentage
        else:
            print(json.dumps({'type': self.progress_type, 'progress': 100}))

    def complete(self):
        self.current_progress = self.total_steps
        print(json.dumps({'type': self.progress_type, 'progress': 100}))


if __name__ == '__main__':
    asyncio.run(main())
