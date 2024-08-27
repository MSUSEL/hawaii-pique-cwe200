import json
import os

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_output(labeled_data):
    extracted_data = []
    for entry in labeled_data:
        output = json.loads(entry['output'])
        file_name = output['files'][0]['fileName']
        for attribute_type in ['variables', 'strings', 'comments', 'sinks']:
            items = output['files'][0].get(attribute_type, [])
            if items:
                for item in items:
                    name = item['name']
                    is_sensitive = item['isSensitive'] if attribute_type != 'sinks' else item['isSink']
                    extracted_data.append({
                        'fileName': file_name,
                        'type': attribute_type,
                        'name': name,
                        'isSensitive': is_sensitive
                    })
    return extracted_data

def compare_classifications(labeled_data, tool_data):
    tp = tn = fp = fn = 0

    for labeled_item in labeled_data:
        file_name = labeled_item['fileName']
        attribute_type = labeled_item['type']
        name = labeled_item['name']
        labeled_value = labeled_item['isSensitive']

        tool_file = next((f for f in tool_data if f['fileName'] == file_name), None)
        tool_items = tool_file[attribute_type] if tool_file else []

        tool_item = next((item for item in tool_items if item['name'] == name), None)
        tool_value = tool_item['isSensitive'] if tool_item else 'no'

        if labeled_value == 'yes' and tool_value == 'yes':
            tp += 1
        elif labeled_value == 'yes' and tool_value == 'no':
            fn += 1
        elif labeled_value == 'no' and tool_value == 'yes':
            fp += 1
        elif labeled_value == 'no' and tool_value == 'no':
            tn += 1

    return tp, tn, fp, fn

def calculate_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def main():
    labeled_data = load_jsonl(os.path.join('backend', 'Files', 'TrainingDataset', 'testing_data.jsonl'))
    tool_data = load_json(os.path.join('backend', 'Files', 'TrainingDataset', 'data.json'))

    extracted_labeled_data = extract_output(labeled_data)
    
    results = {}

    for attribute_type in ['variables', 'strings', 'comments', 'sinks']:
        filtered_labeled_data = [item for item in extracted_labeled_data if item['type'] == attribute_type]
        tp, tn, fp, fn = compare_classifications(filtered_labeled_data, tool_data)
        precision, recall, f1 = calculate_metrics(tp, tn, fp, fn)

        results[attribute_type] = {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    for attribute_type, metrics in results.items():
        print(f"Results for {attribute_type}:")
        print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        print(f"  Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1 Score: {metrics['f1']:.2f}\n")

if __name__ == '__main__':
    main()
