#read toydataset too and it to data

# Importing necessary libraries
import pandas as pd
import nltk
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
import numpy
from sklearn.model_selection import train_test_split
import torch.nn as nn
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Bidirectional
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import ReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard, CSVLogger
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json
import sys
import os

os.chdir(os.path.join(os.getcwd(), "backend", "src", "bert"))

BATCH_SIZE = 32
EPOCHS = 500
DIM=768
# DIM=384

lemmatizer = WordNetLemmatizer()
# w2v_model = Word2Vec.load("fastText_Models/Word2Vec_StandardJavaAPIs")

file_name = ""
variables=[]
strings=[]
comments=[]


def parse_values(fileParsedResults):
    try:
        parsed_results = json.loads(fileParsedResults)

        # Extract the values and assign them to the respective lists
        global variables, strings, comments, file_name
        variables = parsed_results.get("variables", [])
        strings = parsed_results.get("strings", [])
        comments = parsed_results.get("comments", [])
        file_name = parsed_results.get("filename", "")
    
    except:
        pass


def load_stop_words():
    #NLP PReprocessing-------------------------------------
    nltk.download('stopwords')
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


def camel_case_split(str):
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]


def Text_Preprocess(feature_text):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    word_tokens = word_tokenize(feature_text)
    if '\n' in word_tokens:
        word_tokens.remove('\n')
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and w.isalnum()]
    for i in range(0, len(filtered_sentence)):
        filtered_sentence[i:i + 1] = camel_case_split(filtered_sentence[i])
    # print(filtered_sentence)
    tagged_sentence = pos_tag(filtered_sentence)
    # print(tagged_sentence)
    # filtered_API_Desc_tokens = [w for w in wo if not w[0].lower() in stop_words and w[0].isalnum()]
    # ------------------------ add lemitization -------------------------------
    lemmatized_sentence = []
    for word, tag in tagged_sentence:
        ntag = tag[0].lower()
        if (ntag in ['a', 'r', 'n', 'v']):
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), ntag))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower()))
    return listToString(lemmatized_sentence)


def calculate_SentBert_Vectors(API_Lines):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    Sentences=[]
    for api in API_Lines:
        preprocessedTokens=Text_Preprocess(api)
        api=''
        for token in preprocessedTokens:
            api=api+token+' '
        Sentences.append(api)
    print(Sentences)
    embeddings = model.encode(Sentences)
    return embeddings


def readContext(fileName,varName):
    # with open("data/ReviewSensFiles/"+fileName,"r") as f:
    with open(fileName, "r") as f:
        context=""
        fileSentences=f.readlines()
        for sent in fileSentences:
            sent_tokens = word_tokenize(sent)
            if varName in sent_tokens:
                context=context+" "+sent
                continue

    return  Text_Preprocess(context)
def listToString(lst):
    # print(lst)
    str=""
    for item in lst:
        str=str+item+" "
    return str



def calculate_SentBert_Vectors(API_Lines):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 
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
        totalVec.append(numpy.concatenate((nameVec[idx], contextVex[idx]), axis=None))
        # print(totalVec)
        # print(len(totalVec[0]))
    return totalVec

stop_words = load_stop_words()


def main():
    # filecontents = args[1]
    # fileParsedResults = args[2]

    file_contents = 'test'

    fileParsedResults = r"""
{
    "variables": [
        "dataKey",
        "cipher",
        "args",
        "APIKey",
        "data",
        "e",
        "encryptedData",
        "KEY"
    ],
    "filename": "C:\\Users\\kyler\\OneDrive\\Documents\\Work\\cwe200\\backend\\Files\\CWEToyDataset\\CWEToyDataset\\src\\main\\java\\com\\mycompany\\app\\CWE-201\\GOOD\\GOOD_EncryptDataBeforeTransmission.java",
    "comments": [],
    "strings": [
        "The API token is 123",
        "An error has occurred.",
        "Encrypted Data:",
        "Bar12345Bar12345",
        "AES",
        "AES/ECB/PKCS5Padding"
    ]
}
"""


    file_contents = """
    import javax.crypto.Cipher;
    import javax.crypto.spec.SecretKeySpec;
    import java.util.Base64;

    public class GOOD_EncryptDataBeforeTransmission {
        private static final String KEY = "Bar12345Bar12345";
        
        public static String encryptData(String data) throws Exception {
            SecretKeySpec dataKey = new SecretKeySpec(KEY.getBytes(), "AES");
            Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, dataKey);
            
            byte[] encryptedData = cipher.doFinal(data.getBytes());
            return Base64.getEncoder().encodeToString(encryptedData);
        }

        public static void main(String[] args) {
            try {
                String APIKey = "The API token is 123";
                String encryptedData = encryptData(APIKey);
                System.out.println("Encrypted Data: " + encryptedData);
            } catch (Exception e) {
                System.err.println("An error has occurred.");
            }
        }
    }

    """

    # Parse the JSON
    parse_values(fileParsedResults)
    variable_array=numpy.array(variables)
    contents_array = numpy.array(file_contents.split())

    variable_vectors=calculate_SentBert_Vectors(variable_array)
    context_vectors=calculate_SentBert_Vectors(contents_array)
    concatenated_variable_vectors=concatNameandContext(variable_vectors, context_vectors)
    print(1)
    # Load model
    model = load_model('sensInfo_variables_01_0.605.h5')

    # Run model
    test_x= numpy.reshape(concatenated_variable_vectors,(-1,DIM))
    # print("test shape:", numpy.array(test_x).shape)
    # test_y= numpy.reshape(test_set_y_id,(-1,1)).astype(numpy.float32)
    yPredict = model.predict(test_x)
    for idx, val in enumerate(yPredict):
        print(projectAllVariables[idx]," ", yPredict[idx].round())

    # Save Sensitive Info

    # Turn into JSON



    # Text_Preprocess(stop_words, file_contents)




if __name__ == '__main__':
    # main(sys.argv)
    print("------------------------------------ " + os.getcwd())
    # main()
