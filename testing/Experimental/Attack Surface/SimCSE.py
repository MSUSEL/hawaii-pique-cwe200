'''
The approach here is based on contrastive learning, a technique originally popularized in computer vision tasks 
such as OpenAI’s CLIP model (https://arxiv.org/abs/2103.00020).
The core idea is to train a model to pull together similar data points (e.g., sensitive code examples) 
and push apart dissimilar ones (e.g., sensitive vs. non-sensitive) in the embedding space.

In this case, we apply contrastive learning to text/code data using SimCSE (Simple Contrastive Sentence Embeddings), 
which builds on the same base architecture as BERT, but is trained with a contrastive loss instead of traditional 
masked language modeling. This produces embeddings that are more discriminative and better suited for 
downstream classification tasks such as sensitive data detection.
'''

import os
import sys
from .shared import get_context, read_json, text_preprocess, BinaryClassifier, MultiClassClassifier, \
    evaluate_model
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


def combine_name_and_context(input_data: list[list[str]]) -> list[str]:
    """
    Combines the variable name and its code context into a single input string.
    SimCSE is designed to work with full sentences or unified text inputs,
    so we concatenate the variable name and its context with a separator.
    """
    return [
        f"{text_preprocess(name)} [SEP] {text_preprocess(context)}"
        for name, context in zip(input_data[0], input_data[1])
    ]


def main():
    base_path = os.path.join("backend", "src", "bert", "training", "data")

    # Read in the context 
    context = read_json(os.path.join(base_path, 'parsedResults.json'))
    # Read in the labels
    labels = read_json(os.path.join(base_path, 'labels.json'))

    # Extract relevant context for each category
    variable_context = get_context(labels, context, 'variables')
    string_context = get_context(labels, context, 'strings')
    sink_context = get_context(labels, context, 'sinks')
    comment_context = get_context(labels, context, 'comments')

    # In the original code, we embedded the variable name and code context separately. 
    # However, SimCSE is designed to operate on full sentences or unified text inputs.
    # So, for this implementation, we concatenate the name and context into a single string 
    # before embedding, allowing SimCSE to capture their combined semantics.

    data = combine_name_and_context(variable_context)

    # 1. load the SimCSE model
    model_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
    simcse = SentenceTransformer(model_name)


    # 2. Encode the data
    print("Encoding data with SimCSE...")
    X = simcse.encode(data, batch_size=32, show_progress_bar=True)
    y = np.array([item[2] for item in variable_context], dtype=np.int32)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Create DataLoaders
    def get_loader(X, y, batch_size=32, shuffle=True):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = get_loader(X_train, y_train)
    test_loader = get_loader(X_test, y_test, shuffle=False)

    print("SimCSE main function placeholder. Implement the training logic here.")

    # 5. Define classifier
    binary_classifier = BinaryClassifier(
        embedding_dim=X.shape[1],
        dropout_rate=0.5,
        weight_decay=0.01,
        activation='relu'
    )

    # 6. Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 7. Evaluate model

