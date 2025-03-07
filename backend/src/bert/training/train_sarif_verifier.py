import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress info and warnings
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Activation, Input)
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV, train_test_split)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Parameter grid
param_grid = {
    'model__learning_rate': [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4],
    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3],
    'model__activation': ['leaky_relu', 'relu', 'elu', 'gelu'],
    'model__weight_decay': [1e-5, 3e-5, 5e-5, 1e-4],
    'model__batch_size': [32, 64, 96],
    'model__epochs': [60, 80, 100]
}

def read_data_flow_file(file):
    with open(file, "r") as f:
        data_flows = json.load(f)
    return data_flows

def camel_case_split(str_input):
    words = [[str_input[0]]]
    for c in str_input[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append([c])
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def text_preprocess(feature_text):
    words = camel_case_split(feature_text)
    preprocessed_text = ' '.join(words).lower()
    return preprocessed_text

import os
import numpy as np

def process_data_flows(labeled_flows_dir):
    processed_data_flows = []
    # Dictionary to track seen variableNames per resultIndex
    seen_variable_names_by_result = {}
    
    # Counters for statistics
    total_flows = 0
    duplicate_flows = 0
    kept_flows = 0
    
    for file_name in os.listdir(labeled_flows_dir):
        data_flows = read_data_flow_file(os.path.join(labeled_flows_dir, file_name))
        for cwe in data_flows.keys():
            for result in data_flows[cwe]:
                result_index = result['resultIndex']
                flow_file_name = result['fileName']
                
                # Initialize set for this resultIndex if not already present
                if result_index not in seen_variable_names_by_result:
                    seen_variable_names_by_result[result_index] = set()
                
                for flow in result['flows']:
                    total_flows += 1  # Increment total flows
                    
                    # Get the variableName of the first step in the flow
                    if not flow['flow']:  # Handle empty flow case
                        continue
                    first_step_variable_name = flow['flow'][0]['variableName']
                    
                    # Check for duplicate
                    if first_step_variable_name in seen_variable_names_by_result[result_index]:
                        duplicate_flows += 1  # Increment duplicate counter
                        continue
                    
                    # Add the variableName to the seen set
                    seen_variable_names_by_result[result_index].add(first_step_variable_name)
                    kept_flows += 1  # Increment kept flows
                    
                    # Process the flow as before
                    data_flow_string = f"Filename = {flow_file_name} Flows = "
                    codeFlowIndex = flow['codeFlowIndex']
                    label = 1 if flow['label'] == 'Yes' else 0
                    for step in flow['flow']:
                        data_flow_string += str(step)
                    processed_data_flows.append([
                        file_name,
                        result_index,
                        codeFlowIndex,
                        text_preprocess(data_flow_string),
                        label
                    ])
    
    # Print statistics
    print(f"Total flows processed: {total_flows}")
    print(f"Duplicate flows excluded: {duplicate_flows}")
    print(f"Flows kept for training: {kept_flows}")

    with open('processed_data_flows.json', 'w') as json_file:
        json.dump(processed_data_flows, json_file, indent=4)
    
    return np.array(processed_data_flows)

def calculate_sentbert_vectors(sentences, batch_size=64):
    model_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model_transformer.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def create_model(learning_rate=0.0001, dropout_rate=0.2, weight_decay=0.0001, units=256, activation='elu', embedding_dim=None):
    if embedding_dim is None:
        raise ValueError("Embedding dimension not found")
    
    units = embedding_dim // 3
    model = Sequential()
    model.add(Input(shape=(embedding_dim,)))
    model.add(Dense(units, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 2, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 4, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return model

def evaluate_model(final_model, X_test, y_test, category):
    print("Evaluating model on test set...")
    predicted_probs = final_model.predict(X_test, verbose=0)
    predicted_classes = (predicted_probs > 0.5).astype(int)
    precision = metrics.precision_score(y_test, predicted_classes)
    recall = metrics.recall_score(y_test, predicted_classes)
    f1_score = metrics.f1_score(y_test, predicted_classes)
    accuracy = metrics.accuracy_score(y_test, predicted_classes)
    auc = metrics.roc_auc_score(y_test, predicted_probs)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print('------------------------------------------------')
    print('Classification Report (Binary):')
    print(metrics.classification_report(y_test, predicted_classes, target_names=["Non-sensitive", "Sensitive"]))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, predicted_classes))

if __name__ == "__main__":
    # Configuration
    embedding_dim = 384
    embedding_model = 'paraphrase-MiniLM-L6-v2'
    labeled_flows_dir = os.path.join('Testing', 'Labeling', 'FlowData')
    model_dir = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    os.makedirs(model_dir, exist_ok=True)
    scoring = 'f1'
    category = 'flows'

    # Step 1: Process all data flows
    print("Processing data flows...")
    processed_data_flows = process_data_flows(labeled_flows_dir)
    print("Calculating Sentence-BERT embeddings...")
    embeddings = calculate_sentbert_vectors(processed_data_flows[:, 3])
    labels = processed_data_flows[:, 4]

    # Step 2: Prepare X and y
    X = np.array(embeddings)
    y = labels.astype(np.int32)

    # Step 3: Train/test split (before SMOTE)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Step 4: Hyperparameter tuning with pipeline
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    model = KerasClassifier(
        model=create_model,
        epochs=50,
        batch_size=32,
        verbose=0,
        embedding_dim=embedding_dim,
        learning_rate=0.0001,
        dropout_rate=0.2,
        activation='elu',
        weight_decay=0.0001
    )
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('model', model)])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=50,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=2
    )
    random_search_result = random_search.fit(X_train, y_train)
    sys.stdout.flush()
    print(f"Best CV F1 Score: {random_search_result.best_score_:.4f} using {random_search_result.best_params_}")

    # Step 5: Train final model with best parameters
    best_params = random_search_result.best_params_
    final_model = create_model(
        learning_rate=best_params['model__learning_rate'],
        dropout_rate=best_params['model__dropout_rate'],
        activation=best_params['model__activation'],
        weight_decay=best_params['model__weight_decay'],
        embedding_dim=embedding_dim
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    print("Applying SMOTE to training data for final fit...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Training final model with {best_params['model__epochs']} epochs and batch size {best_params['model__batch_size']}...")
    history = final_model.fit(
        X_train_smote, y_train_smote,
        validation_split=0.1,
        epochs=best_params['model__epochs'],
        batch_size=best_params['model__batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # Step 6: Evaluate
    print('Evaluating the Model...')
    evaluate_model(final_model, X_test, y_test, category)

    # Save model for inference
    print("Saving final model...")
    final_model.save(os.path.join(model_dir, 'verify_flows.keras'))
    tf.keras.backend.clear_session()
    print("Training complete!")