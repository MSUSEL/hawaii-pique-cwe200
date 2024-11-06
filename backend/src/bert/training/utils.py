import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                       ReduceLROnPlateau)

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical

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


def train(category, data, param_grid, create_model):
    # Convert data to NumPy array
    variable_array = np.array(data)

    # Calculate embeddings
    print("Encoding values")
    variable_vectors = calculate_sentbert_vectors(variable_array[:, 0])
    print("Encoding context")
    variable_context_vectors = calculate_sentbert_vectors(variable_array[:, 1])

    # Concatenate name and context embeddings
    concatenated_variable_vectors = concat_name_and_context(
        variable_vectors, variable_context_vectors
    )

    # Prepare data for training
    X = np.array(concatenated_variable_vectors)
    y = variable_array[:, 2].astype(np.int32)  # Use integer labels

    # Set model loss function and scoring based on category
    if category == 'sinks':
        loss = 'categorical_crossentropy' # Sinks used multi-class classification 
        scoring = 'f1_weighted'
    else:
        loss = 'binary_crossentropy'
        scoring = 'f1'
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    # Wrap the model using KerasClassifier with adjusted loss function
    model = KerasClassifier(model=create_model, verbose=0, loss=loss)

    # Stratified k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search with RandomizedSearchCV
    n_iter_search = 50  # Adjust based on computational resources
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )

    # Perform grid search with integer labels (not one-hot encoded)
    random_search_result = random_search.fit(X, y)

    # Summarize results
    print("Best: %f using %s" % (random_search_result.best_score_,
                                random_search_result.best_params_))

    # Build the final model with the best parameters
    best_params = random_search_result.best_params_

    # Create final model with the best parameters
    final_model = create_model(
        learning_rate=best_params['model__learning_rate'],
        dropout_rate=best_params['model__dropout_rate'],
        units=best_params['model__units'],
        activation=best_params['model__activation']
    )

    # Split data into train and test
    train_index, test_index = next(kfold.split(X, y))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # One-hot encode labels if the category is `sinks`
    if category == 'sinks':
        y_train = to_categorical(y_train, num_classes=8)
        y_test = to_categorical(y_test, num_classes=8)

    # Early stopping, learning rate reduction, and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_filepath = os.path.join(model_dir, f'best_model_sent_{category}.keras')
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
    evaluate_model(final_model, checkpoint_filepath, X_test, y_test, category)

    # Clear session to free memory
    tf.keras.backend.clear_session()


