#read toydataset too and it to data

# Importing necessary libraries
import pandas as pd
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
import numpy
from sklearn.model_selection import train_test_split
import torch.nn as nn
from keras.models import Sequential
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
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.metrics
import os
from progress_tracker import ProgressTracker

#######------------------------------parameters------------------------------------
peer_reviewed_file_name="data/Peer Review Data Files-3.xlsx"
BATCH_SIZE = 32
EPOCHS = 500
DIM=768
model_dir="model/"
modelname="sensInfo_variables_dfg"
#######----------------------------------------------------------------------------

variables=[]
strings=[]
comments=[]
category=1

#NLP PReprocessing-------------------------------------
# stop_words = set(stopwords.words('english'))
stop_words = stopwords.words('english')
# with open("data/java_keywords.txt","r") as javaKeywordsFile:
#     keywords=javaKeywordsFile.readlines()
#     for keyword in keywords:
#         newkeyword=keyword.strip().replace('\n','')
#         # print(keyword,newkeyword)
#         stop_words.append(newkeyword)


lemmatizer = WordNetLemmatizer()
# w2v_model = Word2Vec.load("fastText_Models/Word2Vec_StandardJavaAPIs")


def camel_case_split(str):
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]


def text_preprocess(feature_text):
    word_tokens = word_tokenize(feature_text)
    if '\n' in word_tokens:
        word_tokens.remove('\n')
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    for i in range(0, len(filtered_sentence)):
        filtered_sentence[i:i + 1] = camel_case_split(filtered_sentence[i])
    tagged_sentence = pos_tag(filtered_sentence)

    # ------------------------ add lemitization -------------------------------
    lemmatized_sentence = []
    for word, tag in tagged_sentence:
        ntag = tag[0].lower()
        if (ntag in ['a', 'r', 'n', 'v']):
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), ntag))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower()))
    return listToString(lemmatized_sentence)

def listToString(lst):
    # print(lst)
    str=""
    for item in lst:
        str=str+item+" "
    return str

def readInputData(file):
    with open(file, "r", encoding='UTF-8') as jsonVars:
        positives = 0
        negatives = 0
        varData = json.load(jsonVars)
        for f in varData:
            fileName = f['fileName']
            allVariables = f['variables']

            for v in allVariables:
                isSensitive = v["isSensitive"]
                VarName = v['name']
                graph = v['graph']
                context = ""
                for i, node in enumerate(graph):
                    # ------split parameters------
                    dnodeName = str(node['name'])
                    type = str(node['type'])
                    next_node = node['nextNode']
                    node_context = node['context']
                    if '(' in dnodeName:
                        dnodeName = dnodeName.split('(')[1].split(')')[0]
                    context += f"(Node Name = {dnodeName}, Type = {type}, Context = {node_context} Next Node = {next_node})"
                    if i != len(graph) - 1:
                        context += '->'
                print("context: ", context)
                if str(isSensitive).strip() == 'yes':
                    variables.append([text_preprocess(VarName), text_preprocess(context), 1])
                    positives = positives + 1
                else:
                    variables.append([text_preprocess(VarName), text_preprocess(context), 0])
                    negatives = negatives + 1
        print("cve data variables -->", "  positives-samples:", positives, " negativesamples: ", negatives)




def calculate_SentBert_Vectors(API_Lines):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    Sentences=[]
    for api in API_Lines:
        preprocessedTokens=text_preprocess(api)
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

    return totalVec

base_path = os.path.join(os.getcwd(), "backend", "src", "bert")
# readInputData(os.path.join(base_path, 'Combined_JSON_Toy.json'))
# readInputData(os.path.join(base_path, 'CVE.json'))


def read_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        return json.load(file)

def get_context(labels, context, category):

    for label_entry in labels:
        file_name = label_entry['fileName']

        # Check if file_name exists in context data
        if file_name not in context:
            print(f"Warning: {file_name} not found in context data.")
            continue

        file_context = context[file_name]
        method_code_map = file_context.get('methodCodeMap', {})

        if category not in label_entry:
            continue

        for label_item in label_entry[category]:
            # Find matching context item for current category
            matched_context_item = next((context_item for context_item in file_context.get(category, []) if label_item['name'] == context_item['name']), None)
            
            if matched_context_item:
                methods = matched_context_item.get('methods', [])
                aggregated_context = f"Type: {matched_context_item['type']}, Context: "

                # Retrieve method code if not global
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]
                
                binary_label = 1 if label_item['IsSensitive'] == 'Yes' else 0
                
                variables.append([text_preprocess(label_item['name']), text_preprocess(aggregated_context), binary_label])

                


           

context = read_json(os.path.join(base_path, 'parsedResults.json'))
labels = read_json(os.path.join(base_path, 'labels.json'))
labels_to_context_mapping = get_context(labels, context, 'variables')


variableArray=numpy.array(variables)
variable_vectors=calculate_SentBert_Vectors(variableArray[:,0])
variable_context_vectors=calculate_SentBert_Vectors(variableArray[:,1])

concatenated_variable_vectors=concatNameandContext(variable_vectors,variable_context_vectors)

first_set_x, test_set_x, first_set_y_id, test_set_y_id = train_test_split(concatenated_variable_vectors, numpy.array(variableArray[:,2]), test_size=0.2, random_state=42, shuffle=True)

train_set_x, valid_set_x, train_set_y_id, valid_set_y_id = train_test_split(first_set_x, first_set_y_id, test_size=0.1, random_state=42, shuffle=True)

train_y = numpy.reshape(train_set_y_id,(-1,1)).astype(numpy.float32)
num_zeros = numpy.count_nonzero(train_y == 0)
num_ones = numpy.count_nonzero(train_y == 1)
print("train samples----> positive=",num_ones," negatives=",num_zeros)
train_x = numpy.reshape(train_set_x,(-1,DIM))

valid_x = numpy.reshape(valid_set_x,(-1,DIM))
valid_y = numpy.reshape(valid_set_y_id,(-1,1)).astype(numpy.float32)
num_zeros = numpy.count_nonzero(valid_y == 0)
num_ones = numpy.count_nonzero(valid_y == 1)
print("validation samples----> positive=",num_ones," negatives=",num_zeros)
test_x= numpy.reshape(test_set_x,(-1,DIM))
test_y= numpy.reshape(test_set_y_id,(-1,1)).astype(numpy.float32)
num_zeros = numpy.count_nonzero(test_y == 0)
num_ones = numpy.count_nonzero(test_y == 1)
print("test samples----> positive=",num_ones," negatives=",num_zeros)
print( "valid_set_x", " ", numpy.array(valid_set_x).shape," ",numpy.array(valid_set_y_id).shape)
#----------------------------------------- Create Deep Learning Structure and Add Layers ---------------------------------------------------
print('------------------------------------------------')
print('Create Deep Learning Structure and Add Layers ...')
model = Sequential()
# model.add(Embedding(len(word_index) + 1,
#                     EMBEDDING_DIM,
#                     weights=[embedding_matrix],
#                     input_length=MAX_LEN,
#                     trainable=False))
# model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
# model.add(Bidirectional(LSTM(512, activation='tanh', return_sequences=True)))
# model.add((Conv1D(512, 3, activation='tanh')))
# model.add(LSTM(512, activation='tanh', return_sequences=True, input_shape=(768,1)))
# Reduce the spatial size of the representation
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#----------------------------------------- Configuring Deep Learning Model -----------------------------------------------
print('------------------------------------------------')
print('Configure Deep Learning Model ...')
#config the model with losses and metrics with model.compile()
f1_score=keras.metrics.F1Score(name='f1_score')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              #sara:optimizer='adam',
              metrics=[f1_score])
              # metrics=['accuracy'])

# Save result of best training epoch: monitor val_loss
checkpoint=ModelCheckpoint(filepath=model_dir + modelname + '_{epoch:02d}_{val_loss:.3f}.keras', monitor='val_loss',
                    verbose=2, save_best_only=True),
# checkpoint=ModelCheckpoint(filepath=model_dir + modelname + '_{epoch:02d}_{f1_score:.3f}.h5', monitor=f1_score,
#                     verbose=2, save_best_only=True, period=1),
callbacks = [checkpoint]
#     EarlyStopping(monitor='val_loss', patience=60, verbose=2, mode="min"),
#     TensorBoard(log_dir=log_path, batch_size=BATCH_SIZE, write_graph=True, write_grads=True, write_images=True,
#                 embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
#     CSVLogger(log_path + saved_model_name + '.log')]

print("start training the model...")

#----------------------------------------- Training Deep Learning Model -----------------------------------------------
print('------------------------------------------------')
print('Start Training Deep Learning Model ...')
#We call fit(), which will train the model by slicing the dacallbacks=callbacksta into "batches" of size "batch_size", and repeatedly iterating over the entire dataset for a given number of "epochs".

history = model.fit(train_x, train_y,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          shuffle=False,
          validation_data=(valid_x, valid_y),
          verbose=2,
          callbacks=callbacks)

print("history",history.history.keys())
#----------------------------------------- Prediction based on Deep Learning Model --------------------------------------
print('------------------------------------------------')
print("Start predicting....")

predicted_classes = model.predict(test_x, batch_size=BATCH_SIZE, verbose=2)
# test_accuracy = numpy.mean(numpy.equal(test_y, predicted_classes))
# print("test_y",test_y)
# print("test_accuracy" +  str(test_accuracy))
F1_score = metrics.f1_score(test_y, predicted_classes.round())
precision=metrics.precision_score(test_y, predicted_classes.round())
recall=metrics.recall_score(test_y, predicted_classes.round())
print("Precision: ",precision)
print("recal: ",recall)
print("F1_score" +  str(F1_score))
accuracy = metrics.accuracy_score(test_y, predicted_classes.round())
print("accuracy" +  str(accuracy))
# target_names = ["Non-vulnerable", "Vulnerable"]  # non-vulnerable->0, vulnerable->1
# print("test_y",test_y)
# print("predicted_classes",predicted_classes)
# print(confusion_matrix(test_y, predicted_classes, labels=[0, 1]))
# print("\r\n")
# print("\r\n")
# print(classification_report(test_y, predicted_classes, target_names=target_names))

# print(deep_learning_model + " prediction completed.")

# K.clear_session()

