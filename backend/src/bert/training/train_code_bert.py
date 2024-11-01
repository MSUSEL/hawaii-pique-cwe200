# Import necessary libraries
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     Activation)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV)
from sklearn import metrics
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import sys

# ------------------------------ Parameters ------------------------------------
DIM = 768  # Embedding dimension (CodeBERT outputs 768-dim embeddings)
model_dir = "model/"
modelname = "sensInfo_variables_dfg"

# -----------------------------------------------------------------------------

variables = []
category = 'variables'

def camel_case_split(str_input):
    words = [[str_input[0]]]
    for c in str_input[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append([c])
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def text_preprocess(feature_text):
    # Split camel case and convert to lowercase
    words = camel_case_split(feature_text)
    preprocessed_text = ' '.join(words).lower()
    return preprocessed_text

def read_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        return json.load(file)

def get_context(labels, context, category):
    for label_entry in labels:
        file_name = label_entry['fileName']
        if file_name not in context:
            print(f"Warning: {file_name} not found in context data.")
            continue
        file_context = context[file_name]
        method_code_map = file_context.get('methodCodeMap', {})
        if category not in label_entry:
            continue
        if category == 'variables':
            for label_item in label_entry[category]:
                # Check if the variable is present in the context JSON
                matched_context_item = next(
                    (context_item for context_item in file_context.get(
                        category, []) if label_item['name'] == context_item['name']
                    ), None)
                # If the variable is found in the context JSON
                if matched_context_item:
                    methods = matched_context_item.get('methods', [])
                    aggregated_context = (
                        f"Type: {matched_context_item['type']}, Context: "
                    )
                    # Aggregate the context of all methods
                    for method in methods:
                        if method != 'global' and method in method_code_map:
                            aggregated_context += method_code_map[method]
                    # Append the variable name, aggregated context, and label
                    binary_label = 1 if label_item['IsSensitive'] == 'Yes' else 0
                    variables.append([
                        text_preprocess(label_item['name']),
                        text_preprocess(aggregated_context),
                        binary_label
                    ])

# Read input data
base_path = os.path.join(os.getcwd(), "backend", "src", "bert")
context = read_json(os.path.join(base_path, 'parsedResults.json'))
labels = read_json(os.path.join(base_path, 'labels.json'))
get_context(labels, context, category)

# Convert variables to NumPy array
variable_array = np.array(variables)

# Initialize the SentenceTransformer model (CodeBERT)
model_save_path = os.path.join(os.getcwd(), "backend", "src", "bert", "models", "codebert-base-mean-pooling")

if not os.path.exists(model_save_path):
    # print("CodeBERT model not found. Downloading and saving the model...")
    model_transformer = SentenceTransformer('microsoft/codebert-base')
    model_transformer.save(model_save_path)
    print("Model loaded successfully.")

else:
    print("Loading CodeBERT model...")
    model_transformer = SentenceTransformer(model_save_path)
    print("Model loaded successfully.")



# model_transformer = SentenceTransformer('microsoft/codebert-base')

def calculate_sentbert_vectors(sentences, batch_size=64):
    embeddings = model_transformer.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def concat_name_and_context(name_vecs, context_vecs):
    total_vecs = []
    for idx in range(len(name_vecs)):
        total_vecs.append(
            np.concatenate((name_vecs[idx], context_vecs[idx]), axis=None)
        )
    return total_vecs

# Calculate embeddings
variable_vectors = calculate_sentbert_vectors(variable_array[:, 0])
variable_context_vectors = calculate_sentbert_vectors(variable_array[:, 1])

# Concatenate name and context embeddings
concatenated_variable_vectors = concat_name_and_context(
    variable_vectors, variable_context_vectors
)

# Prepare data for training
X = np.array(concatenated_variable_vectors)
y = variable_array[:, 2].astype(np.float32)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Define a function to create the model
def create_model(optimizer='adam', learning_rate=0.0001, dropout_rate=0.2,
                 weight_decay=0.0001, units=256, activation='elu'):
    model = Sequential()
    model.add(Dense(units, kernel_regularizer=regularizers.l2(weight_decay),
                    input_dim=DIM))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units // 2,
                    kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units // 4,
                    kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Updated param_grid for fine-tuning
param_grid = {
    'model__learning_rate': [5e-5, 1e-4, 2e-4],
    'model__dropout_rate': [0.15, 0.2, 0.25],
    'model__weight_decay': [5e-5, 1e-4, 2e-4],
    'model__units': [192, 256, 320],
    'model__activation': ['elu', 'relu'],
    'batch_size': [32, 64, 128],
    'epochs': [40, 50, 60]
}

# Stratified k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

n_iter_search = 50  # Adjust based on computational resources
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=n_iter_search,
    cv=kfold,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search_result = random_search.fit(X_resampled, y_resampled)

# Summarize results
print("Best: %f using %s" % (random_search_result.best_score_,
                             random_search_result.best_params_))

# Build the final model with the best parameters
best_params = random_search_result.best_params_

final_model = create_model(
    learning_rate=best_params['model__learning_rate'],
    dropout_rate=best_params['model__dropout_rate'],
    weight_decay=best_params['model__weight_decay'],
    units=best_params['model__units'],
    activation=best_params['model__activation']
)

# Split data into train and test
train_index, test_index = next(kfold.split(X_resampled, y_resampled))
X_train, X_test = X_resampled[train_index], X_resampled[test_index]
y_train, y_test = y_resampled[train_index], y_resampled[test_index]

# Early stopping, learning rate reduction, and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=5, min_lr=1e-6, verbose=1)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

checkpoint_filepath = os.path.join(model_dir, 'best_model.keras')
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   monitor='val_loss',
                                   verbose=1, save_best_only=True)

callbacks = [early_stopping, lr_reduction, model_checkpoint]

# Fit the final model
history = final_model.fit(X_train, y_train,
                          validation_split=0.1,
                          epochs=best_params['epochs'],
                          batch_size=best_params['batch_size'],
                          callbacks=callbacks,
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
print(metrics.classification_report(y_test, predicted_classes,
                                    target_names=["Non-sensitive",
                                                  "Sensitive"]))

# Confusion Matrix
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, predicted_classes))

# Clear session to free memory
tf.keras.backend.clear_session()
