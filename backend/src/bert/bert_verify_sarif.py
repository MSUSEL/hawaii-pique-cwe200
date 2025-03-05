import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from keras.models import load_model
import tensorflow as tf

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configuration
embedding_model_name = 'paraphrase-MiniLM-L6-v2'
project_name = "CWEToyDataset"
project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
input_json_path = os.path.join(project_path, 'flowMapsByCWE.json')
model_path = os.path.join(os.getcwd(), "backend", 'src', 'bert', 'models', 'verify_flows.keras')

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

def read_data_flow_file(file_path):
    with open(file_path, "r") as f:
        data_flows = json.load(f)
    return data_flows

def process_data_flows_for_inference(data_flows):
    processed_data_flows = []
    flow_references = []
    for cwe in data_flows.keys():
        for result in data_flows[cwe]:
            result_index = result['resultIndex']
            for flow in result['flows']:
                data_flow_string = ""
                codeFlowIndex = flow['codeFlowIndex']
                for step in flow['flow']:
                    data_flow_string += str(step)
                processed_data_flows.append(text_preprocess(data_flow_string))
                flow_references.append((cwe, result_index, codeFlowIndex))
    return processed_data_flows, flow_references

def calculate_sentbert_vectors(sentences, batch_size=64):
    print("Calculating Sentence-BERT embeddings for inference...")
    model_transformer = SentenceTransformer(embedding_model_name)
    embeddings = model_transformer.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def load_keras_model(model_path):
    print(f"Loading model from {model_path}...")
    # model = tf.keras.models.load_model(model_path)
    model = load_model(model_path)

    return model

def predict_labels(model, embeddings):
    print("Running inference...")
    predicted_probs = model.predict(embeddings, verbose=1)
    predicted_classes = (predicted_probs > 0.5).astype(int)  # 0 = "No", 1 = "Yes"
    predicted_labels = ["Yes" if pred == 1 else "No" for pred in predicted_classes]
    return predicted_labels

def update_json_with_predictions(data_flows, flow_references, predicted_labels):
    print("Updating JSON with predictions...")
    for (cwe, result_index, code_flow_index), label in zip(flow_references, predicted_labels):
        for result in data_flows[cwe]:
            if result['resultIndex'] == result_index:
                for flow in result['flows']:
                    if flow['codeFlowIndex'] == code_flow_index:
                        flow['label'] = label
                        break
    return data_flows

def save_updated_json(data_flows, input_file_path):
    output_file_path = os.path.splitext(input_file_path)[0] + "_test.json"
    print(f"Saving updated JSON to {output_file_path}...")
    with open(output_file_path, 'w') as f:
        json.dump(data_flows, f, indent=4)
    return output_file_path

if __name__ == "__main__":
    # Step 1: Load the trained model
    model = load_keras_model(model_path)

    # Step 2: Read and process the input JSON
    print(f"Reading data flows from {input_json_path}...")
    data_flows = read_data_flow_file(input_json_path)
    processed_texts, flow_references = process_data_flows_for_inference(data_flows)

    # Step 3: Calculate embeddings
    embeddings = calculate_sentbert_vectors(processed_texts)

    # Step 4: Predict labels
    predicted_labels = predict_labels(model, embeddings)

    # Step 5: Update JSON with predictions
    updated_data_flows = update_json_with_predictions(data_flows, flow_references, predicted_labels)

    # Step 6: Save the updated JSON
    output_path = save_updated_json(updated_data_flows, input_json_path)
    print(f"Inference complete! Updated JSON saved to {output_path}")