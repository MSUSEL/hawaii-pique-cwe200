import pandas as pd
from collections import defaultdict
import json

def parse_review(file_path, sheet_names, reviewers, classes):
    """
    Create a nested dictionary structure for all sheets.

    :param file_path: Path to the Excel file.
    :param sheet_names: List of sheet names to process.
    :param reviewers: List of reviewers.
    :param classes: List of classification sections.
    :return: A nested dictionary with all the data.
    """
    # Initialize the final structure
    all_sheets_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for sheet_name in sheet_names:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Extract the file name
        file_name = df.iloc[0, 0]
        
        # Find the locations of the labels
        locations = df.where(df == "Label").stack().index.tolist()
        
        # Populate the structure with classifications
        for i in range(len(locations)):
            start_row = locations[i][0] + 1
            end_row = locations[i+1][0] if i+1 < len(locations) else len(df)
            
            for j, reviewer in enumerate(reviewers, start=2):
                labels = df.iloc[start_row:end_row, 0]
                keys = df.iloc[start_row:end_row, 1]
                classifications = clean_classifications(df.iloc[start_row:end_row, j])
                all_sheets_structure[sheet_name][file_name][classes[i]][reviewer] = classifications
                all_sheets_structure[sheet_name][file_name][classes[i]]['labels'] = labels.tolist()
                all_sheets_structure[sheet_name][file_name][classes[i]]['keys'] = clean_keys(keys.tolist())
    
    return all_sheets_structure

def clean_classifications(classifications):
    """
    Clean classifications by splitting and handling empty values.

    :param classifications: Series of classifications to clean.
    :return: Cleaned list of classifications.
    """
    classifications = classifications.tolist()
    classifications = [str(item) for item in classifications]

    for i in range(len(classifications)):
        if not classifications[i]:
            classifications[i] = "None"
        else:
            classifications[i] = classifications[i].split(" ")[0].upper()
    return classifications

def clean_keys(keys):
    """
    Clean keys by removing empty values, extra "s, and converting to strings.

    :param keys: List of keys to clean.
    :return: Cleaned list of keys.
    """
    keys = [str(key).replace('"', '') for key in keys if str(key) != 'nan']
    return keys

def find_agreed_classifications(nested_dict, reviewers, classes):
    """
    Find agreed-upon classifications across reviewers.

    :param nested_dict: The nested dictionary structure.
    :param reviewers: List of reviewers.
    :param classes: List of classification sections.
    :return: A new nested dictionary with agreed-upon classifications.
    """
    agreed_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for sheet_name, sheet_data in nested_dict.items():
        for file_name, file_data in sheet_data.items():
            for class_name in classes:
                labels = file_data[class_name]['labels']
                keys = file_data[class_name]['keys']
                for idx, label in enumerate(labels):
                        # Now instead of looking for 2 yeses, we just check to see if a label exists.
                        if idx < len(keys):
                            if str(label) != 'nan':
                                classification = 'Y'
                            else:
                                classification = 'N'
        
                            # print(f"Agreed: {sheet_name}, {file_name}, {class_name}, {label}, {keys[idx]} {idx} {classification}")

                            agreed_dict[sheet_name][file_name][class_name].append({
                                'label': label,
                                'key': keys[idx],
                                'classification': classification
                            })
                # If no agreement, add all classifications
                for key in keys:
                    if key not in [entry['key'] for entry in agreed_dict[sheet_name][file_name][class_name]]:
                        agreed_dict[sheet_name][file_name][class_name].append({
                            'label': label,
                            'key': key,
                            'classification': 'None'
                        })

                        
    
    return agreed_dict

def compare_classifications(agreed_dict, chatGPT_output, classes):
    """
    Compare the agreed classifications with the chatGPT's output for each file and class.
    
    :param agreed_dict: The dictionary with agreed classifications.
    :param chatGPT_output: The list with chatGPT output from JSON.
    :param classes: List of classification sections.
    :return: Metrics for each file, class, and totals.
    """
    file_metrics = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 
                                                            'tp_keys': [], 'fp_keys': [], 'fn_keys': []}))
    total_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    label_metrics = {}
    label_scores = {}

    
    for sheet_name, sheet_data in agreed_dict.items():
        for file_name, file_data in sheet_data.items():
            for class_name in classes:
                agreed_set = set((entry['key'], entry['classification'], entry['label']) for entry in file_data[class_name])
                
                chatGPT_set = set()
                for file in chatGPT_output:
                    if 'fileName' in file and file['fileName'] == file_name:
                        chatGPT_set = set((item['name']) for item in file[class_name])
                        break

                for key, classification, label in agreed_set:
                    # if class_name == 'comments' and classification == 'Y':
                    #     print(f"Agreed: {sheet_name}, {file_name}, {class_name}, {key}, {classification}")

                    if label not in label_metrics:
                        label_metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0}

                    if classification == 'Y':
                        if key in chatGPT_set:

                            file_metrics[file_name][class_name]['tp'] += 1
                            file_metrics[file_name][class_name]['tp_keys'].append(key)
                            total_metrics[class_name]['tp'] += 1
                            label_metrics[label]['tp'] += 1


                        else:
                            file_metrics[file_name][class_name]['fn'] += 1
                            file_metrics[file_name][class_name]['fn_keys'].append(key)
                            total_metrics[class_name]['fn'] += 1
                            label_metrics[label]['fn'] += 1

                    else:
                        if key in chatGPT_set:
                            file_metrics[file_name][class_name]['fp'] += 1
                            file_metrics[file_name][class_name]['fp_keys'].append(key)
                            total_metrics[class_name]['fp'] += 1
                            label_metrics[label]['fp'] += 1
                    label_metrics[label]['total'] += 1

                
                file_metrics[file_name][class_name]['accuracy'] = file_metrics[file_name][class_name]['tp'] / (file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp'] + file_metrics[file_name][class_name]['fn']) if file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp'] + file_metrics[file_name][class_name]['fn'] > 0 else 0
                file_metrics[file_name][class_name]['precision'] = file_metrics[file_name][class_name]['tp'] / (file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp']) if file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp'] > 0 else 0
                file_metrics[file_name][class_name]['recall'] = file_metrics[file_name][class_name]['tp'] / (file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fn']) if file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fn'] > 0 else 0
                file_metrics[file_name][class_name]['fscore'] = 2 * file_metrics[file_name][class_name]['precision'] * file_metrics[file_name][class_name]['recall'] / (file_metrics[file_name][class_name]['precision'] + file_metrics[file_name][class_name]['recall']) if file_metrics[file_name][class_name]['precision'] + file_metrics[file_name][class_name]['recall'] > 0 else 0


    # Calculate total metrics for each class
    for class_name in classes:
        tp = total_metrics[class_name]['tp']
        fp = total_metrics[class_name]['fp']
        fn = total_metrics[class_name]['fn']
        
        # Calculate total metrics for each class
        total_metrics[class_name]['accuracy'] = (tp) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        total_metrics[class_name]['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        total_metrics[class_name]['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        total_metrics[class_name]['fscore'] = 2 * total_metrics[class_name]['precision'] * total_metrics[class_name]['recall'] / (total_metrics[class_name]['precision'] + total_metrics[class_name]['recall']) if total_metrics[class_name]['precision'] + total_metrics[class_name]['recall'] > 0 else 0

    # Calculate total metrics for all classes combined
    total_tp = sum(total_metrics[class_name]['tp'] for class_name in classes)
    total_fp = sum(total_metrics[class_name]['fp'] for class_name in classes)
    total_fn = sum(total_metrics[class_name]['fn'] for class_name in classes)
    total_accuracy = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn > 0 else 0
    total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0

    # Calculate label metrics
    for label in label_metrics:
        tp = label_metrics[label]['tp']
        fp = label_metrics[label]['fp']
        fn = label_metrics[label]['fn']
        total = label_metrics[label]['total']

        accuracy = tp / total if total > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        label_scores[label] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }



    return file_metrics, total_metrics, total_accuracy, total_precision, total_recall, label_scores

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_label_statistics(nested_dict):
    """
    Get statistics on the labels for each class.

    :param nested_dict: The dictionary with all the data.
    :return: A dictionary with the statistics.
    """
    label_stats = defaultdict(int)

    for sheet_name, sheet_data in nested_dict.items():
        for file_name, file_data in sheet_data.items():
            for class_name, class_data in file_data.items():
                labels = class_data['labels']
                for label in labels:
                    label = str(label)
                    if label != 'nan':
                        label_stats[label] += 1
    
    return label_stats


def calculate_label_metrics(label_metrics):
    label_scores = {}
    for label, metrics in label_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        total = metrics['total']

        accuracy = tp / total if total > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        label_scores[label] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    return label_scores

# Usage
file_path = 'testing/ChatGPT/Peer Review Data Files.xlsx'
reviewers = ['david', 'sara', 'samantha']
classes = ['variables', 'strings', 'comments']

# Read the Excel file to get sheet names
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
sheet_names = sheet_names[2:]
# sheet_names = ['Plugin.java']
print(sheet_names)

# Create structure for all sheets
nested_dict = parse_review(file_path, sheet_names, reviewers, classes)

# Find agreed-upon classifications
agreed_dict = find_agreed_classifications(nested_dict, reviewers, classes)

# Load the JSON data
chatGPT_output = load_json('backend/Files/ReviewSensFiles/data.json')

# Compare classifications
file_metrics, total_metrics, total_accuracy, total_precision, total_recall, label_scores = compare_classifications(agreed_dict, chatGPT_output, classes)

# Get label statistics
label_stats = sorted(get_label_statistics(nested_dict).items())

with open('testing/ChatGPT/gpt_results.txt', 'w') as f:
    
    # Print the metrics for each file
    for file, metrics_by_class in file_metrics.items():
        f.write("====================================================\n")  
        f.write(f"Metrics for {file}:\n")
        for class_name, metrics in metrics_by_class.items():
            f.write(f"  {class_name.capitalize()} Metrics:\n")
            f.write(f"    Accuracy: {metrics['accuracy']:.2f}\n")
            f.write(f"    Precision: {metrics['precision']:.2f}\n")
            f.write(f"    Recall: {metrics['recall']:.2f}\n")
            f.write(f"    F1 Score: {metrics['fscore']:.2f}\n")
            f.write(f"    TP Keys: {metrics['tp_keys']}\n")
            f.write(f"    FP Keys: {metrics['fp_keys']}\n")
            f.write(f"    FN Keys: {metrics['fn_keys']}\n")
            f.write("\n") 
        f.write("\n")
    

    # Print the total metrics
    f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++\n") 
    f.write("Total Metrics:\n")
    for class_name, metrics in total_metrics.items():
        f.write(f"  {class_name.capitalize()} Metrics:\n")
        f.write(f"    Accuracy: {metrics['accuracy']:.2f}\n")
        f.write(f"    Precision: {metrics['precision']:.2f}\n")
        f.write(f"    Recall: {metrics['recall']:.2f}\n")
        f.write(f"    F1 Score: {metrics['fscore']:.2f}\n")
        f.write("\n")

    # Print the label statistics
    f.write("----------------------------------------------------\n")
    f.write("Label Statistics:\n")
    
    for label, count in label_stats:
        f.write(f" {count}: {label}\n")

    f.write("----------------------------------------------------\n")
    f.write("Label Metrics:\n")
    for label, scores in label_scores.items():
        if label != 'nan':
            f.write(f"  Label: {label}\n")
            f.write(f"    Accuracy: {scores['accuracy']:.2f}\n")
            f.write(f"    Precision: {scores['precision']:.2f}\n")
            f.write(f"    Recall: {scores['recall']:.2f}\n")
            f.write(f"    F1 Score: {scores['f1_score']:.2f}\n")
            f.write("\n")