import os
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras import regularizers
import keras.metrics


# Parameters
BATCH_SIZE = 32
EPOCHS = 500
DIM = 768 
NUM_CLASSES = 14  # 13 sink types + 1 for non-sink
MODEL_DIR = "model/"
MODEL_NAME = "sink_detection_model"

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load stop words
stop_words = stopwords.words('english')

# Sink Type Mapping
sink_type_mapping = {
    "non-sink": 0,  # Non-sink class
    "I/O Sink": 1,
    "Print Sink": 2,
    "Network Sink": 3,
    "Log Sink": 4,
    "Database Sink": 5,
    "Email Sink": 6,
    "IPC Sink": 7,
    "Clipboard Sink": 8,
    "GUI Display Sink": 9,
    "RPC Sink": 10,
    "Environment Variable Sink": 11,
    "Command Execution Sink": 12,
    "Configuration Sink": 13
}

def camel_case_split(s):
    words = [[s[0]]]
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return None  # Default None for no matching tag

def Text_Preprocess(feature_text):
    word_tokens = word_tokenize(feature_text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    for i in range(len(filtered_sentence)):
        filtered_sentence[i:i + 1] = camel_case_split(filtered_sentence[i])
    tagged_sentence = pos_tag(filtered_sentence)
    
    lemmatized_sentence = []
    for word, tag in tagged_sentence:
        wordnet_pos = get_wordnet_pos(tag) or 'n'  # Default to noun if no POS match
        lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), wordnet_pos))
    
    return ' '.join(lemmatized_sentence)

def readContext(fileName, methodName, java_files):
    context = ""
    if fileName in java_files:
        fileSentences = java_files[fileName].splitlines()
        for sent in fileSentences:
            sent_tokens = word_tokenize(sent)
            if methodName in sent_tokens:
                sent = sent.replace(methodName, ' ')
                context += " " + sent
    return Text_Preprocess(context)

def readInputData(file, java_files):
    method_calls = []
    with open(file, "r", encoding='UTF-8') as jsonVars:
        data = json.load(jsonVars)
        for f in data:
            fileName = f['fileName']
            allMethodCalls = f['sinks']

            for mc in allMethodCalls:
                methodName = mc['name']
                isSink = mc['isSink']
                if isSink == "yes":
                    sink_type = mc['type'].strip()
                else:
                    sink_type = "non-sink"
                
                # Generate context from the surrounding code
                context = readContext(fileName, methodName, java_files)

                preprocessed_name = Text_Preprocess(methodName)
                preprocessed_context = Text_Preprocess(context)

                # Use sink_type_mapping to map the sink type to an integer
                try:
                    method_calls.append([preprocessed_name, preprocessed_context, sink_type_mapping[sink_type]])
                except KeyError:
                    print(f"Unknown sink type: {sink_type}")
                    continue

    return method_calls

def calculate_SentBert_Vectors(API_Lines):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(API_Lines)
    return embeddings

# Keep the concatNameandContext function as is
def concatNameandContext(nameVec, contextVex):
    totalVec = []
    for idx, _ in enumerate(nameVec):
        totalVec.append(np.concatenate((nameVec[idx], contextVex[idx]), axis=None))
    totalVec = np.array(totalVec)
    return totalVec

# Read Java files from a directory synchronously
def read_java_files(directory):
    java_files_with_content = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    java_files_with_content[os.path.basename(file_path)] = file_content
                except Exception as e:
                    print(f"Failed to read file {file_path}: {e}")
    return java_files_with_content

# Read data
base_path = os.path.join(os.getcwd(), "backend", "src", "bert") 
project_path = os.path.join(os.getcwd(), "backend", "Files", "TrainingDataset")  # This holds both CVE and Toy dataset files
java_files = read_java_files(project_path)
method_calls = readInputData(os.path.join(base_path, 'combined_sinks.json'), java_files) # This should be the path to the combined_sinks.json file

# Prepare data
method_call_array = np.array(method_calls)
method_vectors = calculate_SentBert_Vectors(method_call_array[:, 0])  # Preprocessed method names
method_context_vectors = calculate_SentBert_Vectors(method_call_array[:, 1])  # Preprocessed method contexts
concatenated_method_vectors = concatNameandContext(method_vectors, method_context_vectors)

# Class weights to handle class imbalance
class_weights = {i: 1.0 for i in range(NUM_CLASSES)}

# Split data into train-validation-test sets (80-10-10)
first_set_x, test_set_x, first_set_y_id, test_set_y_id = train_test_split(concatenated_method_vectors, method_call_array[:, 2], test_size=0.1, random_state=42, shuffle=True)
train_set_x, valid_set_x, train_set_y_id, valid_set_y_id = train_test_split(first_set_x, first_set_y_id, test_size=0.1111, random_state=42, shuffle=True)

# One-hot encode the sink type labels
train_y = np.eye(NUM_CLASSES)[train_set_y_id.astype(int)]  # One-hot encode the labels
valid_y = np.eye(NUM_CLASSES)[valid_set_y_id.astype(int)]
test_y = np.eye(NUM_CLASSES)[test_set_y_id.astype(int)]

# Create Deep Learning Model with reduced dropout and L2 regularization (Solution 2)
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(DIM,), kernel_regularizer=regularizers.l2(0.001)))  # Reduced L2 penalty
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Reduced dropout rate
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Reduced L2 penalty
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Reduced dropout rate
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Reduced L2 penalty
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Reduced dropout rate
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
f1_score = keras.metrics.F1Score(name='f1_score')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_score])

# Callbacks (Solution 1)
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, MODEL_NAME + '_best.keras'), monitor='val_loss', verbose=2, save_best_only=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# Train the model with the new callback
history = model.fit(np.array(train_set_x), train_y,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(np.array(valid_set_x), valid_y),
                    callbacks=[checkpoint, reduce_lr_on_plateau],  # Dynamic LR scheduler
                    class_weight=class_weights)

# Evaluate on the test set
print("Evaluating on the test set ...")
predicted_classes = model.predict(np.array(test_set_x), batch_size=BATCH_SIZE, verbose=2)

# Convert predictions to labels
predicted_labels = np.argmax(predicted_classes, axis=1)
true_labels = np.argmax(test_y, axis=1)

# Metrics
accuracy = metrics.accuracy_score(true_labels, predicted_labels)
f1_score = metrics.f1_score(true_labels, predicted_labels, average='weighted')
precision = metrics.precision_score(true_labels, predicted_labels, average='weighted')
recall = metrics.recall_score(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
