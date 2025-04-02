import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Activation, Input, Add)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV, train_test_split)
from imblearn.pipeline import Pipeline
from tqdm import tqdm
import hashlib
from transformers import AutoTokenizer, AutoModel
import torch

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

def process_data_flows(labeled_flows_dir):
    processed_data_flows = []
    seen_flow_hashes = set()
    total_flows = 0
    duplicate_flows = 0
    kept_flows = 0
    for file_name in os.listdir(labeled_flows_dir):
        data_flows = read_data_flow_file(os.path.join(labeled_flows_dir, file_name))
        for cwe in data_flows.keys():
            for result in data_flows[cwe]:
                result_index = result['resultIndex']
                flow_file_name = result['fileName']
                for flow in result['flows']:
                    total_flows += 1
                    if not flow['flow'] or 'label' not in flow:
                        continue
                    label = 1 if flow['label'] == 'Yes' else 0 if flow['label'] == 'No' else None
                    if label is None:
                        continue
                    data_flow_string = f"Filename = {flow_file_name}, CWE = {cwe}, Flows = "
                    for step in flow['flow']:
                        data_flow_string += str(step)
                    processed_text = text_preprocess(data_flow_string)
                    flow_hash = hashlib.sha256(processed_text.encode('utf-8')).hexdigest()
                    if flow_hash in seen_flow_hashes:
                        duplicate_flows += 1
                        continue
                    seen_flow_hashes.add(flow_hash)
                    kept_flows += 1
                    processed_data_flows.append([
                        file_name,
                        result_index,
                        flow['codeFlowIndex'],
                        processed_text,
                        label
                    ])
    print(f"Total flows processed: {total_flows}")
    print(f"Duplicate flows excluded: {duplicate_flows}")
    print(f"Flows kept for training: {kept_flows}")
    with open('processed_data_flows.json', 'w', encoding='utf-8') as json_file:
        json.dump(processed_data_flows, json_file, indent=4)
    return np.array(processed_data_flows)

def format_data_flows_for_graphcodebert(processed_flows):
    formatted_flows = []
    for row in processed_flows:
        flow_text = row[3]
        formatted_flows.append(flow_text)
    return formatted_flows

def calculate_graphcodebert_vectors(sentences, model_name='microsoft/graphcodebert-base', batch_size=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding with GraphCodeBERT"):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def create_model(learning_rate=0.0001, dropout_rate=0.2, weight_decay=0.0001, units=256, activation='elu', embedding_dim=None):
    if embedding_dim is None:
        raise ValueError("Embedding dimension not found")
    units1 = embedding_dim
    units2 = embedding_dim * 3 // 4
    units3 = embedding_dim // 2
    units4 = embedding_dim // 4
    inputs = Input(shape=(embedding_dim,))
    x = Dense(units1, kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    res1 = Dense(units2, kernel_regularizer=regularizers.l2(weight_decay))(x)
    res1 = BatchNormalization()(res1)
    res1 = Activation(activation)(res1)
    res1 = Dropout(dropout_rate)(res1)
    x = Dense(units2)(x)
    x = Add()([x, res1])
    res2 = Dense(units3, kernel_regularizer=regularizers.l2(weight_decay))(x)
    res2 = BatchNormalization()(res2)
    res2 = Activation(activation)(res2)
    res2 = Dropout(dropout_rate)(res2)
    x = Dense(units3)(x)
    x = Add()([x, res2])
    x = Dense(units4, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    opt = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return model

def evaluate_model(final_model, X_test, y_test, category):
    print("Evaluating model on test set...")
    predicted_probs = final_model.predict(X_test, verbose=0)
    predicted_classes = (predicted_probs > 0.5).astype(int)
    print(f"Precision: {metrics.precision_score(y_test, predicted_classes):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, predicted_classes):.4f}")
    print(f"F1 Score: {metrics.f1_score(y_test, predicted_classes):.4f}")
    print(f"Accuracy: {metrics.accuracy_score(y_test, predicted_classes):.4f}")
    print(f"AUC: {metrics.roc_auc_score(y_test, predicted_probs):.4f}")
    print(metrics.classification_report(y_test, predicted_classes, target_names=["Non-sensitive", "Sensitive"]))
    print(metrics.confusion_matrix(y_test, predicted_classes))

if __name__ == "__main__":
    labeled_flows_dir = os.path.join('testing', 'Labeling', 'FlowData')
    model_dir = os.path.join(os.getcwd(), "backend", "src", "bert", "models")
    os.makedirs(model_dir, exist_ok=True)
    scoring = 'f1'
    category = 'flows'
    print("Processing data flows...")
    processed_data_flows = process_data_flows(labeled_flows_dir)
    print("Formatting data flows for GraphCodeBERT...")
    formatted_flows = format_data_flows_for_graphcodebert(processed_data_flows)
    print("Calculating GraphCodeBERT embeddings...")
    embeddings = calculate_graphcodebert_vectors(formatted_flows)
    embedding_dim = embeddings.shape[1]
    labels = processed_data_flows[:, 4].astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, stratify=labels, random_state=42)
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
    pipeline = Pipeline([('model', model)])
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
    print(f"Best CV F1 Score: {random_search_result.best_score_:.4f} using {random_search_result.best_params_}")
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
    print(f"Training final model with {best_params['model__epochs']} epochs and batch size {best_params['model__batch_size']}...")
    history = final_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=best_params['model__epochs'],
        batch_size=best_params['model__batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    print('Evaluating the Model...')
    evaluate_model(final_model, X_test, y_test, category)
    print("Saving final model...")
    final_model.save(os.path.join(model_dir, 'verify_flows.keras'))
    tf.keras.backend.clear_session()
    print("Training complete!")
