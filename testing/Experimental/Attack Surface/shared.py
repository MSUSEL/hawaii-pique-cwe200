import json
import torch
import torch.nn as nn

def get_context(labels, context, category):
    """Extract context data for a specific category from the labels and context.
    context comes from the parsedResults.json file.
    :param labels: list of label entries
    :param context: dictionary of context data
    :param category: category to extract (variables, strings, sinks, comments)
    :returns: list of data entries for the specified category
    """
    data = []
    for label_entry in labels:
        file_name = label_entry.get('fileName')
        if file_name not in context:
            print(f"Warning: {file_name} not found in context data.")
            continue
        file_context = context[file_name]
        method_code_map = file_context.get('methodCodeMap', {})

        # Skip if this category isn't in the labels
        if category not in label_entry:
            continue

        for label_item in label_entry[category]:
            # Guard against missing or non?string names
            name = label_item.get('name')
            if not isinstance(name, str) or not name.strip():
                print(
                    f"??  Skipping invalid name ({name!r}) "
                    f"in category '{category}' for file '{file_name}': {label_item}"
                )
                continue
            name = name.strip()

            # Find the matching context item by name
            matched_context_item = next(
                (
                    context_item
                    for context_item in file_context.get(category, [])
                    if name == str(context_item.get('name', '')).strip()
                ),
                None
            )
            if not matched_context_item:
                continue

            # Binary/sink label
            if category == 'sinks':
                binary_label = sink_type_mapping_rev.get(label_item.get('type'), 0)
            else:
                binary_label = 1 if label_item.get('IsSensitive') == 'Yes' else 0

            # Build aggregated context
            aggregated_context = ''
            methods = matched_context_item.get('methods', [])
            if category == 'variables':
                aggregated_context = f"Type: {matched_context_item.get('type')}, Context: "
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]
            elif category == 'strings':
                aggregated_context = "Context: "
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]
            elif category == 'sinks':
                for method in methods:
                    if method != 'global' and method in method_code_map:
                        aggregated_context += method_code_map[method]

            # Append to data in the correct format
            if category in ['variables', 'strings']:
                data.append([
                    text_preprocess(name),
                    text_preprocess(aggregated_context),
                    binary_label
                ])
            elif category == 'comments':
                data.append([
                    text_preprocess(name),
                    None,
                    binary_label
                ])
            elif category == 'sinks':
                data.append([
                    text_preprocess(name),
                    text_preprocess(aggregated_context),
                    binary_label
                ])
    return data

def read_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        return json.load(file)

def camel_case_split(str_input):
    """Split a camelCase or PascalCase string into separate words.
    Helps normalize identifier names for better text processing.
    
    :param s: camelCase or PascalCase string
    :returns: list of split words
    """
    words = [[str_input[0]]]
    for c in str_input[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append([c])
        else:
            words[-1].append(c)
    return [''.join(word) for word in words]

def text_preprocess(feature_text) -> str:
    """Preprocess a string by splitting camel case and converting to lowercase.
    Prepares code tokens for consistent embedding and comparison.
    
    :param feature_text: raw text to preprocess
    :returns: preprocessed string
    """
    words = camel_case_split(feature_text)
    return ' '.join(words).lower()


class BinaryClassifier(nn.Module):
    """Binary classifier model with residual connections and dropout.
    This model is designed to process embeddings and predict binary labels.
    """
    def __init__(self, embedding_dim, dropout_rate, weight_decay, activation):
        """Initialize the model with the given parameters.
        :param embedding_dim: dimension of the input embeddings
        :param dropout_rate: dropout rate for regularization
        :param weight_decay: weight decay for optimizer
        :param activation: activation function to use in the model
        """
        super().__init__()
        self.units1 = embedding_dim
        self.units2 = embedding_dim * 3 // 4
        self.units3 = embedding_dim // 2
        self.units4 = embedding_dim // 4
        act = get_activation(activation)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, self.units1)
        self.bn1 = nn.BatchNorm1d(self.units1)
        self.res1_fc = nn.Linear(self.units1, self.units2)
        self.res1_bn = nn.BatchNorm1d(self.units2)
        self.skip1 = nn.Linear(self.units1, self.units2)
        self.res2_fc = nn.Linear(self.units2, self.units3)
        self.res2_bn = nn.BatchNorm1d(self.units3)
        self.skip2 = nn.Linear(self.units2, self.units3)
        self.fc2 = nn.Linear(self.units3, self.units4)
        self.bn2 = nn.BatchNorm1d(self.units4)
        self.out = nn.Linear(self.units4, 1)
        
    def forward(self, x):
        """Forward pass through the model.
        :param x: input tensor of shape (batch_size, embedding_dim)
        :returns: output tensor of shape (batch_size, 1) after sigmoid activation
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        res1 = self.res1_fc(x)
        res1 = self.res1_bn(res1)
        res1 = self.act(res1)
        res1 = self.dropout(res1)
        skip1 = self.skip1(x)
        x = skip1 + res1
        res2 = self.res2_fc(x)
        res2 = self.res2_bn(res2)
        res2 = self.act(res2)
        res2 = self.dropout(res2)
        skip2 = self.skip2(x)
        x = skip2 + res2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.out(x)
        out = torch.sigmoid(out)
        return out

def evaluate_model(model, loader, device, category):
    """Evaluate the model on the given data loader.
    :param model: trained model to evaluate
    :param loader: data loader for evaluation
    :param device: device to run the model on (CPU or GPU)
    :param category: type of data (variables, sinks, etc.)
    :returns: dictionary of evaluation metrics
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            out = model(Xb)
            if category == 'sinks':
                p = torch.argmax(out, dim=1).cpu().numpy()
            else:
                p = (out > 0.5).int().cpu().numpy().flatten()
            preds.extend(p)
            trues.extend(yb.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    # overall metrics
    prec = metrics.precision_score(trues, preds, average='weighted', zero_division=0)
    rec  = metrics.recall_score(   trues, preds, average='weighted', zero_division=0)
    f1   = metrics.f1_score(      trues, preds, average='weighted', zero_division=0)
    acc  = metrics.accuracy_score(trues, preds)

    # classification report string + confusion matrix
    names      = list(sink_type_mapping.values()) if category=='sinks' else ["Non-sensitive","Sensitive"]
    report_str = metrics.classification_report(trues, preds, target_names=names, zero_division=0)
    conf_mat   = metrics.confusion_matrix(trues, preds)

    return {
        'precision':       prec,
        'recall':          rec,
        'f1':              f1,
        'accuracy':        acc,
        'report_str':      report_str,
        'confusion_matrix': conf_mat
    }

class MultiClassClassifier(nn.Module):
    """Multi-class classifier model with dropout.
    This model is designed to process embeddings and predict multiple classes. Used for sink types.
    """
    def __init__(self, embedding_dim, dropout_rate, weight_decay, activation):
        """Initialize the model with the given parameters.
        :param embedding_dim: dimension of the input embeddings
        :param dropout_rate: dropout rate for regularization
        :param weight_decay: weight decay for optimizer
        :param activation: activation function to use in the model
        """
        super().__init__()
        units = embedding_dim // 3
        act = get_activation(activation)
        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.fc2 = nn.Linear(units, units // 2)
        self.bn2 = nn.BatchNorm1d(units // 2)
        self.fc3 = nn.Linear(units // 2, units // 4)
        self.bn3 = nn.BatchNorm1d(units // 4)
        self.out = nn.Linear(units // 4, 8)
        
    def forward(self, x):
        """Forward pass through the model.
        :param x: input tensor of shape (batch_size, embedding_dim)
        :returns: output tensor of shape (batch_size, 8) after softmax activation
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        out = self.out(x)
        return out

def get_activation(act_name):
    """Get the activation function based on the provided name.
    param act_name: name of the activation function
    :returns: activation function object
    """
    if act_name is None:
    act_name = act_name.lower()
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'elu':
        return nn.ELU()
    elif act_name == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {act_name}")