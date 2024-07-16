import json
import os
from collections import defaultdict

PROJECT = "CWEToyDataset"
CHATGPTPATH = f"Files\\{PROJECT}\\data.json"
CODEQLPATH = f"../out.sarif"
SRCCODEPATH = f"Files\\{PROJECT}"
CWESPATH = f"Files\\{PROJECT}\\src\\main\\java\\com\\mycompany\\app"

def main():
    os.chdir("backend/")
    java_files = get_java_files(SRCCODEPATH)
    codeql_data = read_data(CODEQLPATH)

    codeql_results = analyze_codeql_results(codeql_data, java_files)
    codeql_results = count_nulls(codeql_data, codeql_results["cwe_specific_results"])
    codeql_results = dict(sorted(codeql_results.items(), key=lambda item: int(item[0].split('-')[1])))
    complete_codeql_results = check_missed(codeql_results, java_files)
    calculate_f1_score(complete_codeql_results)
    save_results(complete_codeql_results, "../testing/codeql_results.json")

    print("Analysis complete, results saved to codeql_results.json")
        
def read_data(path):
    with open(path, "r") as file:
        return json.load(file)

def get_java_files(path):
    java_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def get_directories_in_dir(directory_path):
    return [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

def has_vulnerability(results):
    return True if len(results) > 0 else False

def dict_to_json(data):
    for outer_key, inner_dict in data.items():
        for inner_key, the_set in inner_dict.items():
            data[outer_key][inner_key] = list(the_set)
    return data

def save_results(results, path):
    data = dict_to_json(results)
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

def check_missed(codeql_results, java_files):
    for cwe in codeql_results:
        cwe_related_files = set()
        for java_file in java_files:
            file_name = java_file.split("/")[-1]
            if cwe in java_file:
                cwe_related_files.add(java_file)
        for file in cwe_related_files:
            file = file.split("\\")[-1]
            if (file not in codeql_results[cwe]["true_positives"] and 
                file not in codeql_results[cwe]["true_negatives"] and 
                file not in codeql_results[cwe]["false_positives"] and 
                file not in codeql_results[cwe]["false_negatives"]):
                    if file.startswith("BAD"):
                        codeql_results[cwe]["false_negatives"].add(file)
                    elif file.startswith("GOOD"):
                        codeql_results[cwe]["true_negatives"].add(file)
    return codeql_results        

def calculate_f1_score(data):
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0
    cwe_results_dic = {}

    for cwe in data:
        true_positives = len(data[cwe]["true_positives"])
        false_positives = len(data[cwe]["false_positives"])
        true_negatives = len(data[cwe]["true_negatives"])
        false_negatives = len(data[cwe]["false_negatives"])

        total_true_positives += true_positives
        total_true_negatives += true_negatives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        
        if true_positives + false_positives != 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        
        if true_positives + false_negatives != 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0

        if precision + recall != 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        cwe_results_dic[cwe] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score * 100
        }
        print(f"{cwe} query - F1 Score: {f1_score * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%")
        
    if total_true_positives + total_false_positives != 0:
        total_precision = total_true_positives / (total_true_positives + total_false_positives)
    else:
        total_precision = 0

    if total_true_positives + total_false_negatives != 0:
        total_recall = total_true_positives / (total_true_positives + total_false_negatives)
    else:
        total_recall = 0

    if total_precision + total_recall != 0:
        total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall)
    else:
        total_f1_score = 0

    print(f"Total F1 Score: {total_f1_score * 100:.2f}%")
    print(f"Total precision: {total_precision * 100:.2f}%")
    print(f"Total recall: {total_recall * 100:.2f}%")

def analyze_codeql_results(codeql_results, java_files):

    cwe_specific_results = defaultdict(lambda: defaultdict(set))
    cwe_extra_results = defaultdict(lambda: defaultdict(set))

    results = codeql_results['runs'][0]['results']
    num_results = len(results)
        
    for res in results:
        query_cwe = "CWE-" + res['ruleId'].split('/')[-1]
        directory = res['locations'][0]['physicalLocation']['artifactLocation']['uri']
        split = directory.split("/")
        cwe = get_cwe(split)
        file_name = split[-1]

        if query_cwe == cwe:
            message = res['message']
            if file_name.startswith("BAD") and has_vulnerability(message):
                cwe_specific_results[cwe]["true_positives"].add(file_name)
            elif file_name.startswith("GOOD") and has_vulnerability(message):
                cwe_specific_results[cwe]["false_positives"].add(file_name)
            elif file_name.startswith("BAD") and not has_vulnerability(message):
                cwe_specific_results[cwe]["false_negatives"].add(file_name)
            elif file_name.startswith("GOOD") and not has_vulnerability(message):
                cwe_specific_results[cwe]["true_negatives"].add(file_name)
        else:
            cwe_extra_results[query_cwe + " query results"][cwe].add(file_name)

    return {
    'cwe_specific_results': cwe_specific_results,
    'cwe_extra_results': cwe_extra_results 
}

def get_cwe(items):
    for item in items:
        if item.startswith("CWE-"):
            return item

def count_nulls(codeql_results, cwe_specific_results):
    cwes_queries = list(set([c['properties']['cwe'] for c in codeql_results['runs'][0]['tool']['driver']['rules']]))
    for cwe_query in cwes_queries:
        if cwe_query not in cwe_specific_results:
            cwe_specific_results[cwe_query] = {
                "true_positives": set(),
                "false_positives": set(),
                "true_negatives": set(),
                "false_negatives": set(),
            }
    return cwe_specific_results

if __name__ == "__main__":
    main()
