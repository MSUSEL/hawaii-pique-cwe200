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

# Reconfigure stdout and stderr to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure the necessary NLTK data packages are downloaded
packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
def check_and_download_nltk_data(package):
    try:
        find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

for package in packages:
    check_and_download_nltk_data(package)

# Initialize lists to hold different types of data
variables = []
strings = []
comments = []
sinks = []

# Dictionary to hold all variables for different types
projectAllVariables = {
    'variables': [],
    'strings': [],
    'comments': [],
    'sinks': []
}

# Constants
BATCH_SIZE = 32
EPOCHS = 500
DIM = 768

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

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

class ProgressTracker:
    def __init__(self, total_steps, progress_type):
        self.total_steps = total_steps
        self.current_progress = 0
        self.progress_type = progress_type

    def update_progress(self, step_increment):
        self.current_progress += step_increment
        progress_percentage = min(round((self.current_progress / self.total_steps) * 100), 100)
        print(json.dumps({'type': self.progress_type, 'progress': progress_percentage}))

    def complete(self):
        self.current_progress = self.total_steps
        print(json.dumps({'type': self.progress_type, 'progress': 100}))

# Process files to extract relevant information and context
async def process_files(var_data, files_dict, data_type, output_list, project_all_vars, progress_tracker):
    total_progress = len(var_data) * len(var_data[list(var_data.keys())[0]][data_type])
    for file_name in var_data:
        all_variables = var_data[file_name][data_type]
        for v in all_variables:
            try:
                context = get_context(files_dict[file_name], v)
                var = text_preprocess(v)

                if data_type == 'variables':
                    output_list.append((file_name, var, context))

                elif data_type == 'strings':
                    if len(var) == 0:
                        context = ""
                    output_list.append((file_name, var, context))

                elif data_type == 'comments':
                    output_list.append((file_name, var))

                project_all_vars.append([file_name, v])

            except Exception as e:
                print(f"Error processing file {file_name} and {data_type[:-1]} {v}: {e}")
            progress_tracker.update_progress(1)
    # Ensure progress reaches 100%
    progress_tracker.complete()

# Get the context of a variable within a file
def get_context(file, var_name):
    context = ""
    for sent in file.split('\n'):
        sent_tokens = word_tokenize(sent)
        if var_name in sent_tokens:
            sent = sent.replace(var_name, '')  # 2, Updated to remove the variable name from the context
            context = context + sent + " "
    return text_preprocess(context)

# Read Java files from a directory asynchronously
async def read_java_files(directory):
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
                    print(f"Failed to read file {file_path}: {e}")
    return java_files_with_content

# Read parsed data from a JSON file asynchronously
async def read_parsed_data(file_path):
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as json_vars:
            return json.loads(await json_vars.read())
    except Exception as e:
        print(f"Failed to read parsed data from {file_path}: {e}")
        return {}

# Load stop words
stop_words = load_stop_words()

# Process each type of data
async def process_data_type(data_type, data_list, project_all_vars, final_results):
    if data_list:
        data_array = np.array(data_list)
        processing_tracker = ProgressTracker(len(data_array), f'{data_type}-processing')
        prediction_tracker = ProgressTracker(len(data_array), f'{data_type}-prediction')
        saving_tracker = ProgressTracker(len(data_array), f'{data_type}-saving')

        file_info = data_array[:, 0]
        name_vectors = await calculate_sentbert_vectors(data_array[:, 1])
        processing_tracker.update_progress(len(data_array))

        if data_type != 'comments':
            context_vectors = await calculate_sentbert_vectors(data_array[:, 2])
            concatenated_vectors = concat_name_and_context(name_vectors, context_vectors)
        else:
            concatenated_vectors = name_vectors

        processing_tracker.complete()

        # Load the model
        model = load_model(os.path.join(os.getcwd(), "src", "bert", "models", f"{data_type}.h5"))
        # For testing
        # model = load_model(os.path.join(os.getcwd(), "backend", "src", "bert", "models", f"{data_type}.h5"))

        # Run the model to get predictions
        test_x = np.reshape(concatenated_vectors, (-1, DIM)) if data_type != 'comments' else concatenated_vectors
        y_predict = await asyncio.to_thread(model.predict, test_x)

        prediction_tracker.update_progress(len(data_array))
        prediction_tracker.complete()

        # Collect predictions and update saving progress
        for idx, prediction in enumerate(y_predict):
            if prediction > .8:
                file_name, data = project_all_vars[data_type][idx]
                if file_name not in final_results:
                    final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                final_results[file_name][data_type].append({"name": data})
            saving_tracker.update_progress(1)

        # Ensure progress reaches 100%
        saving_tracker.complete()

# Main function
async def main():
    # Set the project path and parsed data file path
    project_path = sys.argv[1]
    # For testing 
    # project_path = os.path.join(os.getcwd(), "backend", "src", "bert", "testdata")
    parsed_data_file_path = os.path.join(project_path, 'parsedResults.json')

    # Read Java files and parsed data asynchronously
    files_dict = await read_java_files(project_path)
    parsed_data = await read_parsed_data(parsed_data_file_path)

    # Process files to extract variables, strings, and comments concurrently
    await asyncio.gather(
        process_files(parsed_data, files_dict, 'variables', variables, projectAllVariables['variables'], ProgressTracker(len(parsed_data) * len(parsed_data[list(parsed_data.keys())[0]]['variables']), 'variables-processing')),
        process_files(parsed_data, files_dict, 'strings', strings, projectAllVariables['strings'], ProgressTracker(len(parsed_data) * len(parsed_data[list(parsed_data.keys())[0]]['strings']), 'strings-processing')),
        # process_files(parsed_data, files_dict, 'comments', comments, projectAllVariables['comments'], ProgressTracker(len(parsed_data) * len(parsed_data[list(parsed_data.keys())[0]]['comments']), 'comments-processing')),
        # process_files(parsed_data, files_dict, 'sinks', sinks, projectAllVariables['sinks'], ProgressTracker(len(parsed_data) * len(parsed_data[list(parsed_data.keys())[0]]['sinks']), 'sinks-processing'))
    )

    # Combine all data into a single dictionary
    all_data = {
        'variables': variables,
        'strings': strings,
        # 'comments': comments,
        # 'sinks': sinks
    }

    final_results = {}

    # Process each type of data concurrently
    await asyncio.gather(
        *[process_data_type(data_type, data_list, projectAllVariables, final_results) for data_type, data_list in all_data.items()]
    )

    # Format results as JSON
    formatted_results = [{"fileName": file_name, **details} for file_name, details in final_results.items()]

    # Write results to a JSON file asynchronously
    async with aiofiles.open(os.path.join(project_path, 'data.json'), 'w') as f:
        await f.write(json.dumps(formatted_results, indent=4))

if __name__ == '__main__':
    asyncio.run(main())
