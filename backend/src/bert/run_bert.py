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
        with open(os.path.join(os.getcwd(),"src", "bert","java_keywords.txt"), "r") as javaKeywordsFile:
            keywords = javaKeywordsFile.readlines()
            for keyword in keywords:
                new_keyword = keyword.strip()
                stop_words.append(new_keyword)
        return stop_words
    except FileNotFoundError as e:
        # print(f"Error: {e}")
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
def Text_Preprocess(feature_text):
    word_tokens = word_tokenize(feature_text)
    if '\n' in word_tokens:
        word_tokens.remove('\n')
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words] # 1, Updated to keep numbers, and Char Lines
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
    return listToString(lemmatized_sentence)

# Calculate sentence embeddings using SentenceTransformer
def calculate_SentBert_Vectors(API_Lines):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    Sentences = []
    for api in API_Lines:
        preprocessedTokens = Text_Preprocess(api)
        api = ' '.join(preprocessedTokens)
        Sentences.append(api)
    embeddings = model.encode(Sentences)
    return embeddings

# Convert list to string
def listToString(lst):
    return ' '.join(lst)

# Concatenate name vectors and context vectors
def concatNameandContext(nameVec, contextVec):
    return [np.concatenate((nameVec[idx], contextVec[idx]), axis=None) for idx in range(len(nameVec))]

# Process files to extract relevant information and context
def process_files(varData, files_dict, type, output_list, project_all_vars, progress_type):
    total_progress = len(varData) * len(varData[list(varData.keys())[0]][type])
    progress = 0
    for f in varData:
        fileName = f
        allVariables = varData[f][type]
        for v in allVariables:
            try:
                context = get_context(files_dict[fileName], v)
                var = Text_Preprocess(v)

                if type == 'variables':
                    output_list.append((fileName, var, context))

                elif type == 'strings':
                    if len(var) == 0:
                        context = ""
                    output_list.append((fileName, var, context))

                elif type == 'comments':
                    output_list.append((fileName, var))

                project_all_vars.append([fileName, v])

            except Exception as e:
                print(f"Error processing file {fileName} and {type[:-1]} {v}: {e}")
            progress += 1
            # Emit progress after processing each variable
            print(json.dumps({'type': progress_type, 'progress': ((progress / total_progress)/2) * 100}))
    

# Get the context of a variable within a file
def get_context(file, varName):
    context = ""
    for sent in file.split('\n'):
        sent_tokens = word_tokenize(sent)
        if varName in sent_tokens:
            sent = sent.replace(varName,'') # 2, Updated to remove the variable name from the context
            context = context + sent + " "
    return Text_Preprocess(context)

# Read Java files from a directory
def read_java_files(directory):
    java_files_with_content = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                java_files_with_content[os.path.basename(file_path)] = file_content
    return java_files_with_content

# Read parsed data from a JSON file
def read_parsed_data(file_path):
    with open(file_path, "r", encoding="utf-8") as jsonVars:
        return json.load(jsonVars)

# Load stop words
stop_words = load_stop_words()

# Main function
def main():
    # Set the project path and parsed data file path
    project_path = sys.argv[1]
    # For testing 
    # project_path = os.path.join(os.getcwd(),"backend" ,"src", "bert", "testdata")
    parsed_data_file_path = os.path.join(project_path, 'parsedResults.json')

    # Read Java files and parsed data
    files_dict = read_java_files(project_path)
    parsed_data = read_parsed_data(parsed_data_file_path)

    # Process files to extract variables, strings, and comments
    process_files(parsed_data, files_dict, 'variables', variables, projectAllVariables['variables'], 'GPTProgress-variables')
    process_files(parsed_data, files_dict, 'strings', strings, projectAllVariables['strings'], 'GPTProgress-strings')
    process_files(parsed_data, files_dict, 'comments', comments, projectAllVariables['comments'], 'GPTProgress-comments')
    # process_files(parsed_data, files_dict, 'sinks', sinks, projectAllVariables['sinks'], 'GPTProgress-sinks')

    # Combine all data into a single dictionary
    all_data = {
        'variables': variables,
        'strings': strings,
        'comments': comments,
        # 'sinks': sinks
    }

    final_results = {}

    # Process each type of data
    for data_type, data_list in all_data.items():
        if data_list:
            data_array = np.array(data_list)
            file_info = data_array[:, 0]
            name_vectors = calculate_SentBert_Vectors(data_array[:, 1])

            if data_type != 'comments':
                context_vectors = calculate_SentBert_Vectors(data_array[:, 2])
                concatenated_vectors = concatNameandContext(name_vectors, context_vectors)
            else:
                concatenated_vectors = name_vectors

            # Load the model
            model = load_model(os.path.join(os.getcwd(),"src", "bert", "models", f"{data_type}.h5"))

            # For testing
            # model = load_model(os.path.join(os.getcwd(),"backend" ,"src", "bert", "models", f"{data_type}.h5"))

            # Run the model to get predictions
            if data_type == 'comments':
                test_x = concatenated_vectors
            else:
                test_x = np.reshape(concatenated_vectors, (-1, DIM))
            yPredict = model.predict(test_x)

            # Collect predictions
            for idx, prediction in enumerate(yPredict):
                if prediction.round() > 0:
                    file_name, data = projectAllVariables[data_type][idx]
                    if file_name not in final_results:
                        final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                    final_results[file_name][data_type].append({"name": data})

            # Emit progress after model predictions
            print(json.dumps({'type': f'GPTProgress-{data_type}', 'progress': 100}))  # 100% after model predictions

    # Format results as JSON
    formatted_results = [{"fileName": file_name, **details} for file_name, details in final_results.items()]
    
    # Write results to a JSON file
    with open(os.path.join(project_path, 'data.json'), 'w') as f:
        json.dump(formatted_results, f, indent=4)
    
if __name__ == '__main__':
    main()
