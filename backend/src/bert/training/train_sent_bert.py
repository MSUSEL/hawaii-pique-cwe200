# Import necessary libraries
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier  # Changed import
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.utils import class_weight
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))



# ------------------------------ Parameters ------------------------------------
BATCH_SIZE = 32
EPOCHS = 50  # Reduced due to cross-validation
DIM = 768  # Keep as is due to concatenation
model_dir = "model/"
modelname = "sensInfo_variables_dfg"

# -----------------------------------------------------------------------------

variables = []
category = 'variables'

# NLP Preprocessing
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def camel_case_split(str_input):
    words = [[str_input[0]]]

    for c in str_input[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]

# def text_preprocess(feature_text):
#     word_tokens = word_tokenize(feature_text)
#     if '\n' in word_tokens:
#         word_tokens.remove('\n')
#     filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#     for i in range(0, len(filtered_sentence)):
#         filtered_sentence[i:i + 1] = camel_case_split(filtered_sentence[i])
#     tagged_sentence = pos_tag(filtered_sentence)

#     # Add lemmatization
#     lemmatized_sentence = []
#     for word, tag in tagged_sentence:
#         ntag = tag[0].lower()
#         if ntag in ['a', 'r', 'n', 'v']:
#             lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), ntag))
#         else:
#             lemmatized_sentence.append(lemmatizer.lemmatize(word.lower()))
#     return list_to_string(lemmatized_sentence)

def text_preprocess(feature_text):
    # Split camel case
    words = camel_case_split(feature_text)
    # Join words back into a string and convert to lowercase
    preprocessed_text = ' '.join(words).lower()
    return preprocessed_text

def list_to_string(lst):
    return ' '.join(lst)

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
            matched_context_item = next(
                (context_item for context_item in file_context.get(category, []) if label_item['name'] == context_item['name']), None)

            if matched_context_item:
                methods = matched_context_item.get('methods', [])
                aggregated_context = f"Type: {matched_context_item['type']}, Context: "

                # Retrieve method code if not global
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]

                binary_label = 1 if label_item['IsSensitive'] == 'Yes' else 0

                variables.append([text_preprocess(label_item['name']), text_preprocess(aggregated_context), binary_label])

# Read input data
base_path = os.path.join(os.getcwd(), "backend", "src", "bert")
context = read_json(os.path.join(base_path, 'parsedResults.json'))
labels = read_json(os.path.join(base_path, 'labels.json'))
get_context(labels, context, category)

# Convert variables to NumPy array
variable_array = np.array(variables)

# Initialize the SentenceTransformer model
model_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Outputs 384-dimensional embeddings

def calculate_sentbert_vectors(sentences):
    embeddings = model_transformer.encode(sentences)
    return embeddings

def concat_name_and_context(name_vecs, context_vecs):
    total_vecs = []
    for idx in range(len(name_vecs)):
        total_vecs.append(np.concatenate((name_vecs[idx], context_vecs[idx]), axis=None))
    return total_vecs

# Calculate embeddings
variable_vectors = calculate_sentbert_vectors(variable_array[:, 0])
variable_context_vectors = calculate_sentbert_vectors(variable_array[:, 1])

# Concatenate name and context embeddings
concatenated_variable_vectors = concat_name_and_context(variable_vectors, variable_context_vectors)

# Prepare data for training
X = np.array(concatenated_variable_vectors)
y = variable_array[:, 2].astype(np.float32)

# Define a function to create the model (used for KerasClassifier)
def create_model(optimizer='adam', learning_rate=0.001, dropout_rate=0.5, weight_decay=0.001, units=64):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=DIM, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 2, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 4, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(1, activation='sigmoid'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = optimizer

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])
    return model

# Wrap the model using the KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)  # Updated for scikeras

# Define the grid search parameters
param_grid = {
    'model__optimizer': ['adam'],
    'model__learning_rate': [5e-5, 1e-4, 2e-4],  # Around 0.0001
    'model__dropout_rate': [0.15, 0.2, 0.25],    # Around 0.2
    'model__weight_decay': [5e-5, 1e-4, 2e-4],   # Around 0.0001
    'model__units': [192, 256, 320],             # Around 256
    'batch_size': [32, 64, 128],                 # Around 64
    'epochs': [40, 50, 60]                       # Around 50
}

# Stratified k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='f1', n_jobs=-1)
grid_result = grid.fit(X, y, class_weight=class_weight_dict)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Build the final model with the best parameters
best_params = grid_result.best_params_

final_model = create_model(optimizer=best_params['model__optimizer'],
                           learning_rate=best_params['model__learning_rate'],
                           dropout_rate=best_params['model__dropout_rate'],
                           weight_decay=best_params['model__weight_decay'],
                           units=best_params['model__units'])

# Split data into train and test using the same k-fold for consistency
train_index, test_index = next(kfold.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Create a directory to save the model if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

checkpoint_filepath = os.path.join(model_dir, 'best_model.keras')
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                                   verbose=1, save_best_only=True)

# Fit the final model
history = final_model.fit(X_train, y_train,
                          validation_split=0.1,
                          epochs=best_params['epochs'],
                          batch_size=best_params['batch_size'],
                          class_weight=class_weight_dict,
                          callbacks=[early_stopping, model_checkpoint],
                          verbose=2)

# Evaluate the model
print('------------------------------------------------')
print('Evaluating the Model...')
# Load the best model
final_model.load_weights(checkpoint_filepath)

predicted_probs = final_model.predict(X_test)
predicted_classes = (predicted_probs > 0.5).astype(int)

# Calculate metrics
precision = metrics.precision_score(y_test, predicted_classes)
recall = metrics.recall_score(y_test, predicted_classes)
f1_score = metrics.f1_score(y_test, predicted_classes)
accuracy = metrics.accuracy_score(y_test, predicted_classes)
auc = metrics.roc_auc_score(y_test, predicted_probs)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy}")
print(f"AUC: {auc}")

# Print classification report
print('------------------------------------------------')
print('Classification Report:')
print(metrics.classification_report(y_test, predicted_classes, target_names=["Non-sensitive", "Sensitive"]))

# Confusion Matrix
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, predicted_classes))

# Clear session to free memory
tf.keras.backend.clear_session()