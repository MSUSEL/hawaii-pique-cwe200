import pandas as pd
from collections import defaultdict
import json
from sklearn.metrics import cohen_kappa_score
import itertools

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
        try:
            # Read the Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Extract the file name
            file_name = df.iloc[0, 0]
            
            # Find the locations of the labels
            locations = df.where(df == "Label").stack().index.tolist()
            
            # Check if the number of sections matches the number of classes
            if len(locations) > len(classes):
                print(f"Warning: Sheet '{sheet_name}' has {len(locations)} 'Label' sections, but only {len(classes)} classes ({classes}) are defined. Skipping extra sections.")
            
            # Populate the structure with classifications
            for i in range(len(locations)):
                # Skip if index exceeds classes length
                if i >= len(classes):
                    print(f"Skipping section {i+1} in sheet '{sheet_name}' as it exceeds defined classes.")
                    continue
                
                start_row = locations[i][0] + 1
                end_row = locations[i+1][0] if i+1 < len(locations) else len(df)
                
                for j, reviewer in enumerate(reviewers, start=2):
                    labels = df.iloc[start_row:end_row, 0]
                    keys = df.iloc[start_row:end_row, 1]
                    classifications = clean_classifications(df.iloc[start_row:end_row, j])
                    all_sheets_structure[sheet_name][file_name][classes[i]][reviewer] = classifications
                    all_sheets_structure[sheet_name][file_name][classes[i]]['labels'] = labels.tolist()
                    all_sheets_structure[sheet_name][file_name][classes[i]]['keys'] = clean_keys(keys.tolist())
        
        except Exception as e:
            print(f"Error processing sheet '{sheet_name}': {str(e)}")
            continue
    
    return all_sheets_structure

def clean_classifications(classifications):
    """
    Clean classifications by splitting and handling empty values.
    """
    options = ['Y', 'N']    
    classifications = classifications.tolist()
    classifications = [str(item) for item in classifications]
    for i in range(len(classifications)):
        if not classifications[i] or classifications[i].lower() == 'nan':
            classifications[i] = "N"  # Treat empty or 'nan' as negative
        else:
            current = classifications[i][0].upper()
            if current in options:
                classifications[i] = current
            else:
                print(f"Warning: Unexpected classification '{current}' found. Defaulting to 'N'.")
                classifications[i] = "N"
    return classifications

def clean_keys(keys):
    """
    Clean keys by removing empty values, extra quotes, and converting to strings.
    """
    keys = [str(key).replace('"', '') for key in keys if str(key) != 'nan']
    return keys

def find_agreed_classifications(nested_dict, reviewers, classes):
    """
    Find agreed-upon classifications across reviewers.
    """
    agreed_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sheet_name, sheet_data in nested_dict.items():
        for file_name, file_data in sheet_data.items():
            for class_name in classes:
                labels = file_data[class_name]['labels']
                keys = file_data[class_name]['keys']
                for idx, label in enumerate(labels):
                    if idx < len(keys):
                        if str(label) != 'nan':
                            classification = 'Y'
                        else:
                            classification = 'N'
                        agreed_dict[sheet_name][file_name][class_name].append({
                            'label': label,
                            'key': keys[idx],
                            'classification': classification
                        })
                for key in keys:
                    if key not in [entry['key'] for entry in agreed_dict[sheet_name][file_name][class_name]]:
                        agreed_dict[sheet_name][file_name][class_name].append({
                            'label': label,
                            'key': key,
                            'classification': 'None'
                        })
    outputJSON(agreed_dict)
    return agreed_dict

def outputJSON(agreed_dict):
    json_output = []
    for sheet_name, sheet_data in agreed_dict.items():
        for file_name, file_data in sheet_data.items():
            file_entry = {"fileName": file_name, "variables": []}
            for class_name, entries in file_data.items():
                for entry in entries:
                    if entry['classification'] == 'Y' and class_name == 'variables':
                        file_entry["variables"].append({
                            "name": entry['key'],
                        })
            if file_entry["variables"]:
                json_output.append(file_entry)
    output_path = 'testing/ChatGPT/agreed_classifications.json'
    with open(output_path, 'w') as json_file:
        json.dump(json_output, json_file, indent=4)

from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import json
import itertools

from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import json
import itertools

def compute_cohen_kappa(nested_dict, reviewers, classes):
    """
    Compute Cohen's Kappa for each pair of reviewers per class and overall, with manual override for variables
    and weighted overall Kappa based on class-specific Kappas and item counts.
    """
    kappa_results = defaultdict(lambda: defaultdict(list))  # reviewer_pair -> class -> [(kappa, num_items)]
    all_classifications = defaultdict(list)  # reviewer_pair -> all classifications
    item_counts = defaultdict(lambda: defaultdict(int))  # reviewer_pair -> class -> total items
    reviewers = reviewers[:2]  # Only consider the first two reviewers for Kappa calculation
    
    for sheet_name, sheet_data in nested_dict.items():
        for file_name, file_data in sheet_data.items():
            for class_name in classes:
                keys = file_data[class_name]['keys']
                classifications_by_reviewer = {
                    reviewer: file_data[class_name][reviewer] for reviewer in reviewers
                }
                
                for rev1, rev2 in itertools.combinations(reviewers, 2):
                    pair = f"{rev1}_vs_{rev2}"
                    rev1_classifications = classifications_by_reviewer[rev1]
                    rev2_classifications = classifications_by_reviewer[rev2]
                    
                    if len(rev1_classifications) == len(rev2_classifications):
                        if not rev1_classifications:
                            print(f"Warning: Empty classifications in {sheet_name}/{file_name}/{class_name} for {pair}. Skipping.")
                            continue
                            
                        num_items = len(rev1_classifications)
                        item_counts[pair][class_name] += num_items
                        
                        if len(set(rev1_classifications)) == 1 and len(set(rev2_classifications)) == 1:
                            if rev1_classifications[0] == rev2_classifications[0]:
                                print(f"Perfect agreement in {sheet_name}/{file_name}/{class_name} for {pair} (all {rev1_classifications[0]}, {num_items} items). Assigning Kappa = 1.0.")
                                kappa_results[pair][class_name].append((1.0, num_items))
                                all_classifications[pair].extend(zip(rev1_classifications, rev2_classifications))
                            else:
                                print(f"Warning: Uniform but conflicting classifications in {sheet_name}/{file_name}/{class_name} for {pair}. Skipping Kappa.")
                                print(f"  {rev1}: {rev1_classifications[:10]}...")
                                print(f"  {rev2}: {rev2_classifications[:10]}...")
                            continue
                        
                        try:
                            kappa = cohen_kappa_score(rev1_classifications, rev2_classifications, labels=['Y', 'N'])
                            if not pd.isna(kappa):
                                kappa_results[pair][class_name].append((kappa, num_items))
                                all_classifications[pair].extend(zip(rev1_classifications, rev2_classifications))
                            else:
                                print(f"Warning: Undefined Kappa in {sheet_name}/{file_name}/{class_name} for {pair}. Skipping.")
                                print(f"  {rev1}: {rev1_classifications[:10]}...")
                                print(f"  {rev2}: {rev2_classifications[:10]}...")
                        except ValueError as e:
                            print(f"Error computing Kappa in {sheet_name}/{file_name}/{class_name} for {pair}: {str(e)}")
                            print(f"  {rev1}: {rev1_classifications[:10]}...")
                            print(f"  {rev2}: {rev2_classifications[:10]}...")
                            continue
    
    # Compute weighted average Kappa per class and overall per reviewer pair
    kappa_summary = {}
    for pair in kappa_results:
        kappa_summary[pair] = {}
        for class_name in classes:
            kappas = kappa_results[pair][class_name]
            if kappas:
                total_kappa = sum(k * n for k, n in kappas)
                total_count = sum(n for _, n in kappas)
                computed_kappa = total_kappa / total_count if total_count > 0 else None
                # Manually override variables Kappa for david_vs_sara
                # if pair == "david_vs_sara" and class_name == "variables":
                    # kappa_summary[pair][class_name] = 0.6826608810410321
                    # print(f"Manually setting {pair}/{class_name} Kappa to 0.6826608810410321 (computed: {computed_kappa})")
                # else:
                kappa_summary[pair][class_name] = computed_kappa
            else:
                kappa_summary[pair][class_name] = None
        # Compute overall Kappa as weighted average of class-specific Kappas
        try:
            total_kappa = 0
            total_items = 0
            for class_name in classes:
                class_kappa = kappa_summary[pair][class_name]
                class_items = item_counts[pair][class_name]
                if class_kappa is not None and class_items > 0:
                    total_kappa += class_kappa * class_items
                    total_items += class_items
            overall_kappa = total_kappa / total_items if total_items > 0 else None
            kappa_summary[pair]['overall'] = overall_kappa
            # Log overall Kappa calculation details
            print(f"\nOverall Kappa Calculation for {pair}:")
            print(f"  Variables: Kappa = {kappa_summary[pair]['variables']:.3f}, Items = {item_counts[pair]['variables']}")
            print(f"  Strings: Kappa = {kappa_summary[pair]['strings']:.3f}, Items = {item_counts[pair]['strings']}")
            print(f"  Comments: Kappa = {kappa_summary[pair]['comments']:.3f}, Items = {item_counts[pair]['comments']}")
            print(f"  Total items: {total_items}, Weighted Kappa: {overall_kappa:.3f}")
        except Exception as e:
            print(f"Error computing overall Kappa for {pair}: {str(e)}")
            kappa_summary[pair]['overall'] = None

    print("\nCohen's Kappa Results:")
    print(json.dumps(kappa_summary, indent=2))
    print("====================================================")
    
    return kappa_summary

def compare_classifications(agreed_dict, chatGPT_output, classes):
    """
    Compare the agreed classifications with the chatGPT's output for each file and class.
    """
    file_metrics = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                                                            'tp_keys': [], 'fp_keys': [], 'fn_keys': [], 'tn_keys': []}))
    total_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    label_metrics = {}
    
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
                    if label not in label_metrics:
                        label_metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total': 0, 
                                                'var_fn_set': set(), 'str_fn_set': set(), 'com_fn_set': set()}
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
                            if class_name == 'variables':
                                label_metrics[label]['var_fn_set'].add(key)
                            elif class_name == 'strings':
                                label_metrics[label]['str_fn_set'].add(key)
                            elif class_name == 'comments':
                                label_metrics[label]['com_fn_set'].add(key)
                    elif classification == 'N':
                        if key in chatGPT_set:
                            file_metrics[file_name][class_name]['fp'] += 1
                            file_metrics[file_name][class_name]['fp_keys'].append(key)
                            total_metrics[class_name]['fp'] += 1
                            label_metrics[label]['fp'] += 1
                        else:
                            file_metrics[file_name][class_name]['tn'] += 1
                            file_metrics[file_name][class_name]['tn_keys'].append(key)
                            total_metrics[class_name]['tn'] += 1
                            label_metrics[label]['tn'] += 1
                    label_metrics[label]['total'] += 1
                
                file_metrics[file_name][class_name]['accuracy'] = (file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['tn']) / (file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp'] + file_metrics[file_name][class_name]['fn'] + file_metrics[file_name][class_name]['tn']) if file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp'] + file_metrics[file_name][class_name]['fn'] + file_metrics[file_name][class_name]['tn'] > 0 else 0
                file_metrics[file_name][class_name]['precision'] = file_metrics[file_name][class_name]['tp'] / (file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp']) if file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fp'] > 0 else 0
                file_metrics[file_name][class_name]['recall'] = file_metrics[file_name][class_name]['tp'] / (file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fn']) if file_metrics[file_name][class_name]['tp'] + file_metrics[file_name][class_name]['fn'] > 0 else 0
                file_metrics[file_name][class_name]['fscore'] = 2 * file_metrics[file_name][class_name]['precision'] * file_metrics[file_name][class_name]['recall'] / (file_metrics[file_name][class_name]['precision'] + file_metrics[file_name][class_name]['recall']) if file_metrics[file_name][class_name]['precision'] + file_metrics[file_name][class_name]['recall'] > 0 else 0

    for class_name in classes:
        tp = total_metrics[class_name]['tp']
        fp = total_metrics[class_name]['fp']
        fn = total_metrics[class_name]['fn']
        tn = total_metrics[class_name]['tn']
        total_metrics[class_name]['accuracy'] = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        total_metrics[class_name]['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        total_metrics[class_name]['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        total_metrics[class_name]['fscore'] = 2 * total_metrics[class_name]['precision'] * total_metrics[class_name]['recall'] / (total_metrics[class_name]['precision'] + total_metrics[class_name]['recall']) if total_metrics[class_name]['precision'] + total_metrics[class_name]['recall'] > 0 else 0

    total_tp = sum(total_metrics[class_name]['tp'] for class_name in classes)
    total_fp = sum(total_metrics[class_name]['fp'] for class_name in classes)
    total_fn = sum(total_metrics[class_name]['fn'] for class_name in classes)
    total_tn = sum(total_metrics[class_name]['tn'] for class_name in classes)
    total_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0
    total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0

    for label in label_metrics:
        tp = label_metrics[label]['tp']
        fp = label_metrics[label]['fp']
        fn = label_metrics[label]['fn']
        tn = label_metrics[label]['tn']
        total = label_metrics[label]['total']
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        label_metrics[label]['accuracy'] = accuracy
        label_metrics[label]['precision'] = precision
        label_metrics[label]['recall'] = recall
        label_metrics[label]['f1_score'] = f1_score

    return file_metrics, total_metrics, total_accuracy, total_precision, total_recall, label_metrics

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_label_statistics(nested_dict):
    """
    Get statistics on the labels for each class.
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

# Usage
file_path = 'testing/ChatGPT/Peer Review Data Files.xlsx'
reviewers = ['david', 'sara', 'samantha']
classes = ['variables', 'strings', 'comments']

xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
sheet_names = sheet_names[6:]
print(sheet_names)

# Create structure for all sheets
nested_dict = parse_review(file_path, sheet_names, reviewers, classes)

# Compute Cohen's Kappa
kappa_summary = compute_cohen_kappa(nested_dict, reviewers, classes)

# Find agreed-upon classifications
agreed_dict = find_agreed_classifications(nested_dict, reviewers, classes)

# Load the JSON data
chatGPT_output = load_json('backend/Files/ReviewSensFiles/data.json')

# Compare classifications
file_metrics, total_metrics, total_accuracy, total_precision, total_recall, label_scores = compare_classifications(agreed_dict, chatGPT_output, classes)

# Get label statistics
label_stats = sorted(get_label_statistics(nested_dict).items())

with open('testing/ChatGPT/gpt_results.txt', 'w') as f:
    # Print Cohen's Kappa results
    f.write("====================================================\n")
    f.write("Cohen's Kappa Between Reviewers:\n")
    for pair, kappa_scores in kappa_summary.items():
        f.write(f"\n{pair.replace('_vs_', ' vs. ')}:\n")
        for class_name in classes:
            kappa = kappa_scores[class_name]
            f.write(f"  {class_name.capitalize()}: {kappa:.3f}\n" if kappa is not None else f"  {class_name.capitalize()}: N/A\n")
        overall_kappa = kappa_scores['overall']
        f.write(f"  Overall: {overall_kappa:.3f}\n" if overall_kappa is not None else f"  Overall: N/A\n")
    f.write("\n")

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
            f.write(f"    TP Keys: [{len(metrics['tp_keys'])}]{metrics['tp_keys']}\n")
            f.write(f"    FP Keys: [{len(metrics['fp_keys'])}]{metrics['fp_keys']}\n")
            f.write(f"    FN Keys: [{len(metrics['fn_keys'])}]{metrics['fn_keys']}\n")
            f.write("\n")
        f.write("\n")

    # Print the total metrics
    f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    f.write("Total Metrics:\n")
    for class_name, metrics in total_metrics.items():
        f.write(f"  {class_name.capitalize()} Metrics:\n")
        f.write(f"    TP {metrics['tp']} | FP {metrics['fp']} | FN {metrics['fn']}\n")
        f.write(f"    Accuracy: {metrics['accuracy']:.2f}\n")
        f.write(f"    Precision: {metrics['precision']:.2f}\n")
        f.write(f"    Recall: {metrics['recall']:.2f}\n")
        f.write(f"    F1 Score: {metrics['fscore']:.2f}\n")
        f.write("\n")

    f.write("----------------------------------------------------\n")
    f.write("Label Metrics:\n")
    f.write(f"*** Warning since these labels all exist FP will always be 0, so only recall can be calculated ***\n\n")

    # Sort the label_scores dictionary by the label name
    sorted_label_scores = dict(sorted(label_scores.items(), key=lambda item: str(item[0])))
    for label, scores in sorted_label_scores.items():
        if str(label) != 'nan':
            f.write(f"label {label}\n")
            f.write(f" Total {scores['total']} | TP {scores['tp']} | FN {scores['fn']} | Recall {scores['recall']:.2f}\n")
            f.write(f" FN Variables: {', '.join(scores['var_fn_set'])}\n")
            f.write(f" FN Strings: {', '.join(scores['str_fn_set'])}\n")
            f.write(f" FN Comments: {', '.join(scores['com_fn_set'])}\n")
            f.write("\n")