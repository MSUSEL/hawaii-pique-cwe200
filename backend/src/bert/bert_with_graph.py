# This script runs BERT with backward slice graphs to predict if a variable is sensitive or not. 
# The hope is that with this extra context, the model will be able to make more accurate predictions.
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

# Dictionary to hold all variables for different types
projectAllVariables = {
    'variables': []
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


def camel_case_split(str):
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]


def Text_Preprocess(feature_text):
    word_tokens = word_tokenize(feature_text)
    if '\n' in word_tokens:
        word_tokens.remove('\n')
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    for i in range(0, len(filtered_sentence)):
        filtered_sentence[i:i + 1] = camel_case_split(filtered_sentence[i])
    tagged_sentence = pos_tag(filtered_sentence)
    lemmatized_sentence = []
    for word, tag in tagged_sentence:
        ntag = tag[0].lower()
        if (ntag in ['a', 'r', 'n', 'v']):
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), ntag))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower()))
    return listToString(lemmatized_sentence)


def listToString(lst):
    str=""
    for item in lst:
        str=str+item+" "
    return str


def readToyData(jsonFile):
    with open(jsonFile, "r", encoding='UTF-8') as jsonVars:
        varData = json.load(jsonVars)
        for f in varData:
            fileName = f['fileName']
            allVariables = f['variables']

            for v in allVariables:
                VarName = v['name']
                projectAllVariables['variables'].append([fileName, VarName])
                graph = v['graph']
                context = ""
                for node in graph:
                    if str(node['name']).strip() != str(VarName).strip():
                        context = context + node['type'] + ' '
                        # ------split parameters------
                        dnodeName = str(node['name'])
                        if '(' in dnodeName:
                            dnodeName = dnodeName.split('(')[1].split(')')[0]
                        context = context + dnodeName + ' '
                    else:  # add variable type to the context
                        context = context + node['type'] + ' '
                # print("context: ", context)
                variables.append([Text_Preprocess(VarName), Text_Preprocess(context)])

def calculate_SentBert_Vectors(API_Lines):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    Sentences=[]
    for api in API_Lines:
        preprocessedTokens=Text_Preprocess(api)
        api=''
        for token in preprocessedTokens:
            api=api+token+' '
        Sentences.append(api)
    embeddings = model.encode(Sentences)
    return embeddings

def concatNameandContext(nameVec,contextVex):
    totalVec=[]
    for idx, vec in enumerate(nameVec):
        totalVec.append(np.concatenate((nameVec[idx], contextVex[idx]), axis=None))
    return totalVec

# Load stop words
stop_words = load_stop_words()

def main():
    # For normal use
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        json_input_path = os.path.join(project_path, "variables_graph.json")
        model = load_model(os.path.join(os.getcwd(), "src", "bert", "models", "variables_dfg.h5"))


    # For testing purposes
    else:
        project_path = os.getcwd()
        json_input_path = os.path.join(os.getcwd(), "backend", "Files","SmallTest", "variables_graph.json")
        model = load_model(os.path.join(os.getcwd(), "backend", "src", "bert", "models", "variables_dfg.h5"))


    readToyData(json_input_path)
    variableArray=np.array(variables)
    variable_vectors=calculate_SentBert_Vectors(variableArray[:,0])
    variable_context_vectors=calculate_SentBert_Vectors(variableArray[:,1])
    concatenated_variable_vectors=concatNameandContext(variable_vectors,variable_context_vectors)

    x_test = np.reshape(concatenated_variable_vectors, (-1, DIM))
    y_predict = model.predict(x_test)

    final_results = {}

    # Collect predictions and update saving progress

    for idx, prediction in enumerate(y_predict):
                    if prediction >= .7:
                        file_name, data = projectAllVariables['variables'][idx]
                        if file_name not in final_results:
                            final_results[file_name] = {"variables": [], "strings": [], "comments": [], "sinks": []}
                        final_results[file_name]['variables'].append({"name": data})
                    else: 
                        file_name, data = projectAllVariables['variables'][idx]
                        # print(f"{data} removed")
    print("done")

    # Format results as JSON
    formatted_results = [{"fileName": file_name, **details} for file_name, details in final_results.items()]

    # Write results to a JSON file
    with open(os.path.join(project_path, 'sensitiveVariables.json'), 'w') as f:
        json.dump(formatted_results, f, indent=4)

if __name__ == "__main__":
    main()
