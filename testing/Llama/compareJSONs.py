import json
from collections import defaultdict
import os

def load_json_file(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        return json.load(file)

def is_in_json_file(file_data, category, name):
    for entry in file_data.get(category, []):
        if entry.get('name') == name:
            return entry.get('isSensitive', 'no') if category != 'sinks' else entry.get('isSink', 'no')
    return 'no'

def compare_files(labeled_file, json_file):
    # Counters for the comparison results
    results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    processed_entries = set()  # To keep track of processed entries

    # First pass: use labeled_file as the driver
    for labeled_file_data in labeled_file:
        file_name = labeled_file_data['fileName']
        json_file_data = next((item for item in json_file if item['fileName'] == file_name), None)

        for category in ['variables', 'strings', 'comments', 'sinks']:
            for entry in labeled_file_data[category]:
                name = entry.get('name')
                if not name:
                    continue

                labeled_label = 'yes'
                json_label = is_in_json_file(json_file_data, category, name) if json_file_data else 'no'

                if json_label == 'yes' and labeled_label == 'yes':
                    results[category]['tp'] += 1
                elif json_label == 'no' and labeled_label == 'yes':
                    results[category]['fn'] += 1

                processed_entries.add((file_name, category, name))

    # Second pass: check for entries in json_file that aren't in labeled_file
    for json_file_data in json_file:
        file_name = json_file_data['fileName']
        labeled_file_data = next((item for item in labeled_file if item['fileName'] == file_name), None)

        for category in ['variables', 'strings', 'comments', 'sinks']:
            for entry in json_file_data[category]:
                name = entry.get('name')
                if not name:
                    continue

                if (file_name, category, name) in processed_entries:
                    continue  # Skip already processed entries

                json_label = entry.get('isSensitive', 'no') if category != 'sinks' else entry.get('isSink', 'no')
                labeled_label = 'yes' if labeled_file_data and is_in_json_file(labeled_file_data, category, name) == 'yes' else 'no'

                if json_label == 'yes' and labeled_label == 'no':
                    results[category]['fp'] += 1
                elif json_label == 'no' and labeled_label == 'no':
                    results[category]['tn'] += 1

    return results

def compute_metrics(results):
    metrics = {}
    for category, counts in results.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        tn = counts['tn']
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        metrics[category] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    return metrics

# File paths
# json_file_path = os.path.join('backend', 'Files', 'TrainingDataset', 'dataBase.json')
# json_file_path = os.path.join('backend', 'Files', 'TrainingDataset', 'dataFineTuned.json')
# json_file_path = os.path.join('backend', 'Files', 'TrainingDataset', 'data.json')
labeled_file_path = os.path.join('backend', 'Files', 'TrainingDataset', 'labeledData.json')

# Load the files
json_file = load_json_file(json_file_path)
labeled_file = load_json_file(labeled_file_path)

# Compare files and compute results
comparison_results = compare_files(labeled_file, json_file)

# Compute precision, recall, and F1 score
metrics = compute_metrics(comparison_results)

# Output the results
for category, metric in metrics.items():
    print(f"{category.capitalize()} - Precision: {metric['precision']:.2f}, Recall: {metric['recall']:.2f}, F1 Score: {metric['f1']:.2f}")
    print(f"TP: {metric['tp']}, FP: {metric['fp']}, FN: {metric['fn']}, TN: {metric['tn']}")

