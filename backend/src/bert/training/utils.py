import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                       ReduceLROnPlateau)

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from transformers import T5Tokenizer, TFT5Model
from transformers import RobertaTokenizer, TFRobertaModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ------------------------------ Parameters ------------------------------------
model_dir = "model/"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# -----------------------------------------------------------------------------

sink_type_mapping = {
    0: "N/A",
    1: "I/O Sink",
    2: "Print Sink",
    3: "Network Sink",
    4: "Log Sink",
    5: "Database Sink",
    6: "Email Sink",
    7: "IPC Sink"
}

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
    
def calculate_sentbert_vectors(sentences, batch_size=64):
    model_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Outputs 384-dimensional embeddings
    embeddings = model_transformer.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def concat_name_and_context(name_vecs, context_vecs):
    total_vecs = []
    for idx in range(len(name_vecs)):
        total_vecs.append(
            np.concatenate((name_vecs[idx], context_vecs[idx]), axis=None)
        )
    return total_vecs

def evaluate_model(final_model, checkpoint_filepath, X_test, y_test, category):
    # Load the best weights
    final_model.load_weights(checkpoint_filepath)
    
    # Predict class probabilities
    predicted_probs = final_model.predict(X_test)
    
    if category == 'sinks':
        # Multi-class case with 8 classes
        predicted_classes = np.argmax(predicted_probs, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        # Calculate metrics for multi-class
        precision = metrics.precision_score(true_classes, predicted_classes, average='weighted')
        recall = metrics.recall_score(true_classes, predicted_classes, average='weighted')
        f1_score = metrics.f1_score(true_classes, predicted_classes, average='weighted')
        accuracy = metrics.accuracy_score(true_classes, predicted_classes)
        auc = metrics.roc_auc_score(y_test, predicted_probs, average='weighted', multi_class='ovr')

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")
        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")

        # Print classification report and confusion matrix for multi-class
        print('------------------------------------------------')
        print('Classification Report (Multi-class):')
        target_names = [sink_type_mapping[i] for i in range(8)]

        print(metrics.classification_report(true_classes, predicted_classes,
                                            target_names=target_names))  # Update based on class names if available

        print('Confusion Matrix:')
        print(metrics.confusion_matrix(true_classes, predicted_classes))

    else:
        # Binary case
        predicted_classes = (predicted_probs > 0.5).astype(int)
        
        # Calculate metrics for binary
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

        # Print classification report and confusion matrix for binary
        print('------------------------------------------------')
        print('Classification Report (Binary):')
        print(metrics.classification_report(y_test, predicted_classes,
                                            target_names=["Non-sensitive", "Sensitive"]))

        print('Confusion Matrix:')
        print(metrics.confusion_matrix(y_test, predicted_classes))

def train(category, data, param_grid, create_model, embedding_model='sentbert', embedding_dim=384):
    # Convert data to NumPy array
    variable_array = np.array(data)

    if embedding_model == 'sentbert':
        get_embeddings = calculate_sentbert_vectors
    elif embedding_model == 't5':
        get_embeddings = calculate_t5_vectors
    elif embedding_model == 'roberta':
        get_embeddings = calculate_roberta_vectors
    elif embedding_model == 'codebert':
        get_embeddings = calculate_codebert_vectors
    elif embedding_model == 'codellama':
        get_embeddings = calculate_codellama_vectors
    elif embedding_model == 'distilbert':
        get_embeddings = calculate_distilbert_vectors
    elif embedding_model == 'albert':
        get_embeddings = calculate_albert_vectors


    # Calculate embeddings
    print("Encoding values")
    variable_vectors = get_embeddings(variable_array[:, 0])
    print("Encoding context")
    variable_context_vectors = get_embeddings(variable_array[:, 1])  # Assuming shared embeddings

    # variable_context_vectors = variable_vectors

    # Concatenate name and context embeddings
    concatenated_variable_vectors = concat_name_and_context(
        variable_vectors, variable_context_vectors
    )

    print(f"Width of the name vector {variable_vectors.shape[1]}")
    print(f"Width of the context vector {variable_context_vectors.shape[1]}")
    print(f"Width of the concatenated vector {concatenated_variable_vectors[0].size}")

    # Prepare data for training
    X = np.array(concatenated_variable_vectors)
    y = variable_array[:, 2].astype(np.int32)

    # Handle class imbalance using SMOTE
    if category != 'sinks':
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    # Define scoring metric
    scoring = 'f1_weighted' if category == 'sinks' else 'f1'

    # Wrap model in KerasClassifier
    model = KerasClassifier(
        model=create_model,  # Pass the model factory
        verbose=0,
        embedding_dim=embedding_dim
    )

    # Stratified k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Randomized Search for Hyperparameter Tuning
    random_search = TqdmRandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1,
        random_state=42

    )

    # Perform the search
    random_search_result = random_search.fit(X, y)

    # Summarize the best parameters
    print(f"Best: {random_search_result.best_score_} using {random_search_result.best_params_}")

    # Build and train the final model using best parameters
    best_params = random_search_result.best_params_
    final_model = create_model(
        learning_rate=best_params['model__learning_rate'],
        dropout_rate=best_params['model__dropout_rate'],
        # weight_decay=best_params['model__weight_decay'],
        # units=best_params['model__units'],
        activation=best_params['model__activation'],
        embedding_dim=embedding_dim
    )

    # Train/test split for evaluation
    train_index, test_index = next(kfold.split(X, y))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # One-hot encode labels if necessary
    if category == 'sinks':
        y_train = to_categorical(y_train, num_classes=8)
        y_test = to_categorical(y_test, num_classes=8)

    # Define callbacks
    checkpoint_filepath = os.path.join(model_dir, f'{embedding_model}_best_model_{category}.keras')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True)
    ]

    # Train the final model
    history = final_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate the model
    print('Evaluating the Model...')
    evaluate_model(final_model, checkpoint_filepath, X_test, y_test, category)
    tf.keras.backend.clear_session()

def calculate_t5_vectors(sentences, model_name='t5-small', batch_size=32):
    """
    Calculate fixed-size embeddings using T5 as an encoder with TensorFlow.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = TFT5Model.from_pretrained(model_name)    
    print("T5 model loaded successfully.")

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch_sentences = sentences[i:i+batch_size]

        # Ensure input is a list of strings
        if isinstance(batch_sentences, np.ndarray):  # If it's a NumPy array, convert it
            batch_sentences = batch_sentences.tolist()
        elif not isinstance(batch_sentences, list):  # Ensure it's a list
            batch_sentences = [str(batch_sentences)]

        inputs = tokenizer(batch_sentences, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs["input_ids"], decoder_input_ids=inputs["input_ids"])
        hidden_states = outputs.last_hidden_state
        pooled_embeddings = tf.reduce_mean(hidden_states, axis=1)  # Mean pooling to get fixed-size embeddings
        embeddings.append(pooled_embeddings.numpy())

    return np.vstack(embeddings)

def calculate_roberta_vectors(sentences, model_name='roberta-base', batch_size=32):
    """
    Calculate fixed-size embeddings using RoBERTa as an encoder with TensorFlow.
    """
    from transformers import RobertaTokenizer, TFRobertaModel

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = TFRobertaModel.from_pretrained(model_name)
    print("RoBERTa model loaded successfully.")

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch_sentences = sentences[i:i+batch_size]

        # Ensure input is a list of strings
        if isinstance(batch_sentences, np.ndarray):  # If it's a NumPy array, convert it
            batch_sentences = batch_sentences.tolist()
        elif not isinstance(batch_sentences, list):  # Ensure it's a list
            batch_sentences = [str(batch_sentences)]

        inputs = tokenizer(batch_sentences, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = outputs.last_hidden_state
        pooled_embeddings = tf.reduce_mean(hidden_states, axis=1)  # Mean pooling to get fixed-size embeddings
        embeddings.append(pooled_embeddings.numpy())

    return np.vstack(embeddings)

def calculate_codebert_vectors(sentences, model_name='microsoft/codebert-base', batch_size=32):
    """
    Calculate fixed-size embeddings using CodeBERT as an encoder with TensorFlow.
    """
    from transformers import AutoTokenizer, TFAutoModel

    # Load CodeBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    print("CodeBERT model loaded successfully.")

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch_sentences = sentences[i:i+batch_size]

        # Ensure input is a list of strings
        if isinstance(batch_sentences, np.ndarray):  # If it's a NumPy array, convert it
            batch_sentences = batch_sentences.tolist()
        elif not isinstance(batch_sentences, list):  # Ensure it's a list
            batch_sentences = [str(batch_sentences)]

        # Tokenize the input batch
        inputs = tokenizer(batch_sentences, return_tensors="tf", padding=True, truncation=True, max_length=512)

        # Forward pass through CodeBERT
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = outputs.last_hidden_state

        # Apply mean pooling over token embeddings to create sentence embeddings
        pooled_embeddings = tf.reduce_mean(hidden_states, axis=1)
        embeddings.append(pooled_embeddings.numpy())

    # Combine all embeddings into a single NumPy array
    return np.vstack(embeddings)


def calculate_codellama_vectors(sentences, model_name='codellama/CodeLlama-7b', batch_size=32):
    """
    Calculate fixed-size embeddings using Code LLaMA as an encoder with TensorFlow.
    """
    from transformers import AutoTokenizer, TFAutoModel

    # Load Code LLaMA tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = TFAutoModel.from_pretrained(model_name)
    print("Code LLaMA model loaded successfully.")

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch_sentences = sentences[i:i+batch_size]

        # Ensure input is a list of strings
        if isinstance(batch_sentences, np.ndarray):  # If it's a NumPy array, convert it
            batch_sentences = batch_sentences.tolist()
        elif not isinstance(batch_sentences, list):  # Ensure it's a list
            batch_sentences = [str(batch_sentences)]

        # Tokenize the input batch
        inputs = tokenizer(batch_sentences, return_tensors="tf", padding=True, truncation=True, max_length=512)

        # Forward pass through Code LLaMA
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = outputs.last_hidden_state

        # Apply mean pooling over token embeddings to create sentence embeddings
        pooled_embeddings = tf.reduce_mean(hidden_states, axis=1)
        embeddings.append(pooled_embeddings.numpy())

    # Combine all embeddings into a single NumPy array
    return np.vstack(embeddings)

def calculate_distilbert_vectors(sentences, model_name='distilbert-base-uncased', batch_size=32):
    """
    Calculate fixed-size embeddings using DistilBERT as an encoder with TensorFlow.
    """
    from transformers import DistilBertTokenizer, TFDistilBertModel

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = TFDistilBertModel.from_pretrained(model_name)
    print("DistilBERT model loaded successfully.")

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch_sentences = sentences[i:i+batch_size]

        # Ensure input is a list of strings
        if isinstance(batch_sentences, np.ndarray):  # If it's a NumPy array, convert it
            batch_sentences = batch_sentences.tolist()
        elif not isinstance(batch_sentences, list):  # Ensure it's a list
            batch_sentences = [str(batch_sentences)]

        inputs = tokenizer(batch_sentences, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = outputs.last_hidden_state
        pooled_embeddings = tf.reduce_mean(hidden_states, axis=1)  # Mean pooling to get fixed-size embeddings
        embeddings.append(pooled_embeddings.numpy())

    return np.vstack(embeddings)

def calculate_albert_vectors(sentences, model_name='albert-base-v2', batch_size=32):
    """
    Calculate fixed-size embeddings using ALBERT as an encoder with TensorFlow.
    """
    from transformers import AlbertTokenizer, TFAlbertModel

    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = TFAlbertModel.from_pretrained(model_name)
    print("ALBERT model loaded successfully.")

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch_sentences = sentences[i:i+batch_size]

        # Ensure input is a list of strings
        if isinstance(batch_sentences, np.ndarray):  # If it's a NumPy array, convert it
            batch_sentences = batch_sentences.tolist()
        elif not isinstance(batch_sentences, list):  # Ensure it's a list
            batch_sentences = [str(batch_sentences)]

        inputs = tokenizer(batch_sentences, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = outputs.last_hidden_state
        pooled_embeddings = tf.reduce_mean(hidden_states, axis=1)  # Mean pooling to get fixed-size embeddings
        embeddings.append(pooled_embeddings.numpy())

    return np.vstack(embeddings)

# Wrap the RandomizedSearchCV with tqdm
class TqdmRandomizedSearchCV(RandomizedSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score=np.nan, return_train_score=False):
        super().__init__(estimator=estimator,
                         param_distributions=param_distributions,
                         n_iter=n_iter,
                         scoring=scoring,
                         n_jobs=n_jobs,
                         refit=refit,
                         cv=cv,
                         verbose=verbose,
                         pre_dispatch=pre_dispatch,
                         random_state=random_state,
                         error_score=error_score,
                         return_train_score=return_train_score)
        self._progress_bar = None

    def fit(self, X, y=None, **fit_params):
        self._progress_bar = tqdm(total=self.n_iter, desc="Random Search Progress")
        self._progress_counter = 0

        def on_fit_iter_start():
            self._progress_counter += 1
            self._progress_bar.update(1)
            self._progress_bar.set_description(f"Model {self._progress_counter}/{self.n_iter}")

        self._on_fit_iter_start = on_fit_iter_start
        result = super().fit(X, y, **fit_params)
        self._progress_bar.close()
        return result

    def _fit_and_score(self, *args, **kwargs):
        if hasattr(self, '_on_fit_iter_start'):
            self._on_fit_iter_start()
        return super()._fit_and_score(*args, **kwargs)