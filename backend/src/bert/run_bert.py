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

variables = []
projectAllVariables = []

BATCH_SIZE = 32
EPOCHS = 500
DIM = 768

lemmatizer = WordNetLemmatizer()

def load_stop_words():
    stop_words = stopwords.words('english')
    try:
        with open("java_keywords.txt", "r") as javaKeywordsFile:
            keywords = javaKeywordsFile.readlines()
            for keyword in keywords:
                new_keyword = keyword.strip()
                stop_words.append(new_keyword)
        return stop_words
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return stop_words

def camel_case_split(s):
    words = [[s[0]]]
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def Text_Preprocess(feature_text):
    word_tokens = word_tokenize(feature_text)
    if '\n' in word_tokens:
        word_tokens.remove('\n')
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and w.isalnum()]
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

def calculate_SentBert_Vectors(API_Lines):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    Sentences = []
    for api in API_Lines:
        preprocessedTokens = Text_Preprocess(api)
        api = ' '.join(preprocessedTokens)
        Sentences.append(api)
    embeddings = model.encode(Sentences)
    return embeddings

def listToString(lst):
    return ' '.join(lst)

def concatNameandContext(nameVec, contextVec):
    return [np.concatenate((nameVec[idx], contextVec[idx]), axis=None) for idx in range(len(nameVec))]

def process_files(varData, files_dict):
    for f in varData:
        fileName = f
        allVariables = varData[f]['variables']
        for v in allVariables:
            try:
                context = get_context(files_dict[fileName], v)
                variables.append((fileName, Text_Preprocess(v), context))
                projectAllVariables.append([fileName, v])
            except Exception as e:
                print(f"Error processing file {fileName} and variable {v}: {e}")

def get_context(file, varName):
    context = ""
    for sent in file.split('\n'):
        sent_tokens = word_tokenize(sent)
        if varName in sent_tokens:
            context += " " + sent
    return Text_Preprocess(context)

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

def read_parsed_data(file_path):
    with open(file_path, "r") as jsonVars:
        return json.load(jsonVars)

stop_words = load_stop_words()

def main():
    # project_path = 'testdata'
    project_path = sys.argv[1]
    parsed_data_file_path = os.path.join(project_path, 'data.json')

    files_dict = read_java_files(project_path)
    parsed_data = read_parsed_data(parsed_data_file_path)
    process_files(parsed_data, files_dict)

    variable_array = np.array(variables)
    file_info = variable_array[:, 0]
    variable_vectors = calculate_SentBert_Vectors(variable_array[:, 1])
    context_vectors = calculate_SentBert_Vectors(variable_array[:, 2])
    concatenated_variable_vectors = concatNameandContext(variable_vectors, context_vectors)

    model = load_model('src/bert/sensInfo_variables_01_0.605.h5')

    test_x = np.reshape(concatenated_variable_vectors, (-1, DIM))
    yPredict = model.predict(test_x)
    
    results = {}
    for idx, prediction in enumerate(yPredict):
        if prediction.round() > 0:
            file_name, variable = projectAllVariables[idx]
            if file_name not in results:
                results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
            results[file_name]["variables"].append({"name": variable})
    
    final_results = [{"fileName": file_name, **details} for file_name, details in results.items()]
    
    with open(os.path.join(project_path,'results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    print("finished")

if __name__ == '__main__':
    main()
