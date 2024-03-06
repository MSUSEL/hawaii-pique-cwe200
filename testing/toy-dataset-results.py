# The purpose of this file is to analyze the results of the codeql analysis on the CWE Toy Dataset
# The results are then analyzed to calculate the accuracy of the codeql analysis 
# Then the results are saved to a JSON file called codeql_results.json

import json
import os
from collections import defaultdict
# from matplotlib import pyplot as plt

PROJECT = "CWEToyDataset"
CHATGPTPATH = f"Files\\{PROJECT}\\data.json"
# CODEQLPATH = f"Files\\{PROJECT}\\results.sarif"
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
        calculate_accuracy(complete_codeql_results)
        save_results(complete_codeql_results, "../testing/codeql_results.json")

        print("Analysis complete, results saved to codeql_results.json")
        
def read_data(path):
    with open(path, "r") as file:
        return json.load(file)

# Get all the java files in the directory
def get_java_files(path):
    java_files = []
    for root, _, files in os.walk(path):  # Remove 'dirs' variable
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

# Get all the directories in a directory
def get_directories_in_dir(directory_path):
    return [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

# Check if a result has a vulnerability denoted by having a length greater than 0
def has_vulnerability(results):
    return True if len(results) > 0 else False

# Convert the set to a list and then convert the dictionary to a JSON
def dict_to_json(data):
    # Convert the set to a list
    for outer_key, inner_dict in data.items():
        for inner_key, the_set in inner_dict.items():
            # Convert the set to a list
            data[outer_key][inner_key] = list(the_set)
    return data

# Save the results as a JSON file
def save_results(results, path):
    data = dict_to_json(results)
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

# Check if there are any missed files that were not caught by the codeql analysis and add them to the results
def check_missed(codeql_results, java_files):
    for cwe in codeql_results:
        cwe_related_files = set()
        for java_file in java_files:
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
                    else:
                        codeql_results[cwe]["true_negatives"].add(file)
    return codeql_results        

# Calculate the accuracy of the codeql results
def calculate_accuracy(data):
    total_true_positives = 0
    total_true_negatives = 0
    total_samples = 0
    cwe_results_dic = {}

    for cwe in data:
        # Calculate the accuracy of each codeql query
        true_positives = len(data[cwe]["true_positives"])
        false_positives = len(data[cwe]["false_positives"])
        true_negatives = len(data[cwe]["true_negatives"])
        false_negatives = len(data[cwe]["false_negatives"])

        total = true_positives + false_positives + true_negatives + false_negatives
        
        total_samples += total
        total_true_positives += true_positives
        total_true_negatives += true_negatives
        
        if total != 0: accuracy = ((true_positives + true_negatives) / total) * 100
        else: accuracy = 0
        cwe_results_dic[cwe] = accuracy
        print(f"Accuracy of {cwe} query: {accuracy}%")
        
    print("\n")
    # Calculate the total accuracy of all codeql queries
    if total_samples != 0: print(f"Total accuracy: {((total_true_positives + total_true_negatives) / total_samples)*100}%")
    else: print("Total accuracy: 0%")

# Analyze the results of the codeql analysis
def analyze_codeql_results(codeql_results, java_files):

    cwe_specific_results = defaultdict(lambda: defaultdict(set))
    cwe_extra_results = defaultdict(lambda: defaultdict(set))

    results = codeql_results['runs'][0]['results']
    num_results = len(results)
        
    for res in results:
        query_cwe = res['ruleId']
        directory = res['locations'][0]['physicalLocation']['artifactLocation']['uri']
        split = directory.split("/")
        cwe = split[-2]
        file_name = split[-1]

        # We only care about the CWE query matching the result for testing purposes.
        if query_cwe == cwe:
            message = res['message']
            # Check that we get a true positive on an expected vulnerabal file
            if file_name.startswith("BAD") and has_vulnerability(message):
                cwe_specific_results[cwe]["true_positives"].add(file_name)
                # print(f"True positive on {file_name} with CWE {cwe}")
            
            # Check that we get a false positive on a non-vulnerable file
            elif file_name.startswith("GOOD") and has_vulnerability(message):
                cwe_specific_results[cwe]["false_positives"].add(file_name)
                # print(f"False positive on {file_name} with CWE {cwe}")
            
            # Check to see if we get a false negative on an expected vulnerabal file
            elif file_name.startswith("BAD") and not has_vulnerability(message):
                cwe_specific_results[cwe]["false_negatives"].add(file_name)
                # print(f"False negative on {file_name} with CWE {cwe}")

            # Check to see if we get a true negative on a non-vulnerabal file
            elif file_name.startswith("GOOD") and not has_vulnerability(message):
                cwe_specific_results[cwe]["true_negatives"].add(file_name)
                # print(f"True negative on {file_name} with CWE {cwe}")
        
        else:
            cwe_extra_results[query_cwe + " query results"][cwe].add(file_name)

    return {
    'cwe_specific_results': cwe_specific_results,
    'cwe_extra_results': cwe_extra_results 
}

# This is used to count CWEs that had no results 0% accuracy)
def count_nulls(codeql_results, cwe_specific_results):
        cwes_queries = list(set([c['properties']['cwe'] for c in codeql_results['runs'][0]['tool']['driver']['rules']]))
        for cwe_query in cwes_queries:
            # Initialize the dictionary for cwe_query if not already present
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