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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ------------------------------ Parameters ------------------------------------
model_dir = "model/"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# -----------------------------------------------------------------------------

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


def evaluate_model(final_model, checkpoint_filepath, X_test, y_test):
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


def train(category, data, param_grid, create_model):
    # Convert variables to NumPy array
    variable_array = np.array(data)

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

    # Wrap the model using KerasClassifier
    model = KerasClassifier(model=create_model, verbose=0)

    # Stratified k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search with RandomizedSearchCV
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_filepath = os.path.join(model_dir, f'best_model_sent{category}.keras')
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

    evaluate_model(final_model, checkpoint_filepath, X_test, y_test)

    # Clear session to free memory
    tf.keras.backend.clear_session()
