import os
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     Activation, Input)
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from utils import train, read_json, text_preprocess

# ------------------------------ Parameters ------------------------------------
DIM = 768 # Sent BERT dims = 384 * 2 (Value + Context)
# -----------------------------------------------------------------------------
sink_type_mapping = {
    "N/A": 0,
    "I/O Sink": 1,
    "Print Sink": 2,
    "Network Sink": 3,
    "Log Sink": 4,
    "Database Sink": 5,
    "Email Sink": 6,
    "IPC Sink": 7
}

def get_context(labels, context, category):
    data = []
    for label_entry in labels:
        file_name = label_entry['fileName']
        if file_name not in context:
            print(f"Warning: {file_name} not found in context data.")
            continue
        file_context = context[file_name]
        method_code_map = file_context.get('methodCodeMap', {})
        if category not in label_entry:
            continue
        # if category == 'variables' or category == 'strings':
        for label_item in label_entry[category]:
            # Check if the variable is present in the context JSON
            matched_context_item = next(
                (context_item for context_item in file_context.get(
                    category, []) if label_item['name'] == context_item['name']
                ), None)
            
            if matched_context_item:
                # Append the variable name, aggregated context, and label
                binary_label = 1 if label_item['IsSensitive'] == 'Yes' else 0
                aggregated_context = ''
                
                if category == 'variables':
                # If the variable is found in the context JSON
                    methods = matched_context_item.get('methods', [])
                    aggregated_context = f"Type: {matched_context_item['type']}, Context: "
                    # Aggregate the context of all methods
                    for method in methods:
                        if method != 'global' and method in method_code_map:
                            aggregated_context += method_code_map[method]
                
                elif category == 'strings':
                    methods = matched_context_item.get('methods', [])
                    aggregated_context = f"Context: "
                    for method in methods:
                        if method != 'global' and method in method_code_map:
                            aggregated_context += method_code_map[method]
                
                elif category == 'sinks':
                    methods = matched_context_item.get('methods', [])
                    for method in methods:
                        if method != 'global' and method in method_code_map:
                            aggregated_context += method_code_map[method]

                # Save the processed item, with it's context, and label
                if category != 'sinks': 
                    data.append([
                        text_preprocess(label_item['name']),
                        text_preprocess(aggregated_context),
                        binary_label
                    ])
                else:
                    data.append([
                        text_preprocess(label_item['name']),
                        text_preprocess(aggregated_context),
                        sink_type_mapping.get(label_item['type'])
                    ]) 
    return data



def create_model(learning_rate=0.0001, dropout_rate=0.2, weight_decay=0.0001, units=256, activation='elu',  embedding_dim=None):
    if embedding_dim == None:
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
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'), 
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model


def create_model_sinks(learning_rate=0.0001, dropout_rate=0.2, weight_decay=0.0001, units=256, activation='elu'):
    model = Sequential()
    model.add(Input(shape=(DIM,)))
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

    # Update the output layer for 8 classes
    model.add(Dense(8, activation='softmax'))

    opt = Adam(learning_rate=learning_rate)

    model.compile(
        loss='categorical_crossentropy',  # Updated for multi-class classification
        optimizer=opt,
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'), 
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

# Read input files
base_path = os.path.join(os.getcwd(), "backend", "src", "bert")
context = read_json(os.path.join(base_path, 'parsedResults.json'))
labels = read_json(os.path.join(base_path, 'labels.json'))


# Updated param_grid for fine-tuning
variables_param_grid = {
    'model__learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
    'model__dropout_rate': [0.2, 0.3],
    # 'model__weight_decay': [5e-5, 1e-4, 2e-4],
    # 'model__units': [192, 256, 320],
    'model__activation': ['elu', 'relu'],
    'batch_size': [32, 64],
    'epochs': [50, 60]
}


strings_param_grid = {
    'model__learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
    'model__dropout_rate': [0.2, 0.3],
    # 'model__weight_decay': [5e-5, 1e-4, 2e-4],
    # 'model__units': [192, 256, 320],
    'model__activation': ['elu', 'relu'],
    'batch_size': [32, 64],
    'epochs': [50, 60]
}

sinks_param_grid = {
    'model__learning_rate': [0.1, 0.01, 0.001, 0.0001], 
    'model__dropout_rate': [0.2],
    'model__weight_decay': [5e-5, 1e-4, 2e-4],
    'model__units': [192, 256, 320],
    'model__activation': ['elu', 'relu'],
    'batch_size': [32, 64],
    'epochs': [50, 60]
}

params_map = {
    "variables": variables_param_grid,
    "strings": strings_param_grid,
    "comments": None,
    "sinks": sinks_param_grid
}

categories = [
    # "variables",
    "strings",
    # "comments",
    # "sinks"
]

# Embedding models, and their respective output dimensions
embedding_models = {
    't5': 512 * 2,           # T5 model produces 512-dimensional embeddings; concatenating value and context.
    # 'roberta': 768 * 2,       # RoBERTa model produces 768-dimensional embeddings; concatenating value and context.
    # 'sent_bert': 384 * 2,     # Sentence-BERT produces 384-dimensional embeddings; concatenating value and context.
    # 'codellama': 4096 * 2,    # Code LLaMA produces 4096-dimensional embeddings; concatenating value and context.
    # 'codebert': 768 * 2,      # CodeBERT produces 768-dimensional embeddings; concatenating value and context.
    # 'albert': 768 * 2,        # ALBERT produces 768-dimensional embeddings; concatenating value and context.
}


for embedding_model, embedding_dim in embedding_models.items():
    for category in categories:
        # Parse input data
        data = get_context(labels, context, category)
        
        if category == 'sinks':
            model = create_model_sinks
        else:
            if embedding_dim is None:
                raise ValueError(f"Embedding model {embedding_model} not found in embedding_models.")
            model = create_model
        # Train the model 
        train(category, 
              data, 
              params_map.get(category), 
              model, 
              embedding_model=embedding_model, 
              embedding_dim=embedding_dim)
