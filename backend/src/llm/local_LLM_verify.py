import json
import os
import sys
import torch  # Import PyTorch explicitly
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress unnecessary TF logs (though Gemma uses PyTorch, not TensorFlow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load Gemma 3 4B model and processor
model_name = "google/gemma-3-4b-it"
logger.info(f"Loading processor for {model_name}...")
try:
    processor = AutoProcessor.from_pretrained(model_name)
    logger.info("Processor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load processor: {e}")
    raise

logger.info(f"Loading model for {model_name}...")
try:
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    model.to(device)  # Move model to GPU if available
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

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
    with open(file_path, "r", encoding='utf-8') as f:
        data_flows = json.load(f)
    return data_flows

def process_data_flows_for_inference(data_flows):
    processed_data_flows = []
    flow_references = []
    filenames = []
    for cwe in data_flows.keys():
        for result in data_flows[cwe]:
            result_index = result['resultIndex']
            flow_file_name = result['fileName']
            for flow in result['flows']:
                data_flow_string = f"Filename = {flow_file_name} Flows = "
                codeFlowIndex = flow['codeFlowIndex']
                for step in flow['flow']:
                    data_flow_string += str(step)
                processed_data_flows.append(text_preprocess(data_flow_string))
                flow_references.append((cwe, result_index, codeFlowIndex))
                filenames.append(flow_file_name)
    return processed_data_flows, flow_references, filenames

def query_gemma(data_flow):
    prompt = f"""Does this data flow expose sensitive information in the last step? 
    I don't consider usernames to be sensitive. 
    Answer only with 'Yes' or 'No'. Data flow: {data_flow}"""
    # Process text input
    inputs = processor(text=prompt, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate text output
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
    )
    # Decode the output (no need to move outputs to CPU explicitly for decoding)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    response = response.lower()
    if "yes" in response:
        return "Yes"
    elif "no" in response:
        return "No"
    else:
        return "No"  # Fallback

def predict_labels_with_gemma(processed_flows, flow_references, filenames):
    logger.info("Running inference with Gemma 3 4B...")
    predicted_labels = []
    false_positive = 0
    
    for i, (flow, (cwe, result_index, code_flow_index), filename) in enumerate(zip(processed_flows, flow_references, filenames)):
        label = query_gemma(flow)
        logger.info(f"Flow {i + 1}/{len(processed_flows)}: Filename = {filename}, ResultIndex = {result_index}, CodeFlowIndex = {code_flow_index}, Label = {label}")
        predicted_labels.append(label)
        if label == "No":
            false_positive += 1
    
    sys.stdout.write(f"Removed {false_positive} flows out of {len(predicted_labels)}\n")
    return predicted_labels

def update_json_with_predictions(data_flows, flow_references, predicted_labels):
    logger.info("Updating JSON with predictions...")
    for (cwe, result_index, code_flow_index), label in zip(flow_references, predicted_labels):
        for result in data_flows[cwe]:
            if result['resultIndex'] == result_index:
                for flow in result['flows']:
                    if flow['codeFlowIndex'] == code_flow_index:
                        flow['label'] = label
                        break
    return data_flows

def save_updated_json(data_flows, input_file_path):
    output_file_path = os.path.splitext(input_file_path)[0] + ".json"
    logger.info(f"Saving updated JSON to {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_flows, f, indent=4)
    return output_file_path

def run(project_path):
    input_json_path = os.path.join(project_path, 'flowMapsByCWE.json')
    logger.info(f"Reading data flows from {input_json_path}...")
    data_flows = read_data_flow_file(input_json_path)
    processed_flows, flow_references, filenames = process_data_flows_for_inference(data_flows)
    predicted_labels = predict_labels_with_gemma(processed_flows, flow_references, filenames)
    updated_data_flows = update_json_with_predictions(data_flows, flow_references, predicted_labels)
    output_path = save_updated_json(updated_data_flows, input_json_path)
    logger.info(f"Inference complete! Updated JSON saved to {output_path}")

if __name__ == "__main__":
    logger.info(f"Arguments: {sys.argv}")
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
        project_path = os.path.join(os.getcwd(), "Files", project_name)
    else:
        project_name = "PowerJob-5.1.1"
        project_path = os.path.join(os.getcwd(), "backend", "Files", project_name)
    
    logger.info(f"Project name: {project_name}")
    run(project_path)