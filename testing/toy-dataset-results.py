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
    # Read in data from CHatGPTPath as a JSON
        os.chdir("backend/")
        java_files = get_java_files(SRCCODEPATH)
        codeql_data = read_data(CODEQLPATH)

        codeql_results = analyze_codeql_results(codeql_data, java_files)
        # cwes = get_directories_in_dir(CWESPATH)
        print(1)
        

def read_data(path):
    with open(path, "r") as file:
        return json.load(file)

def get_java_files(path):
    java_files = []
    for root, _, files in os.walk(path):  # Remove 'dirs' variable
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def get_directories_in_dir(directory_path):
    directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    return directories


def has_vulnerability(chatgpt_results):
    return True if len(chatgpt_results) > 0 else False

def analyze_codeql_results(codeql_results, java_files):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    cwe_specific_results = defaultdict(lambda: defaultdict(list))
    num_results = len(codeql_results['runs'][0]['results'])

    for res in codeql_results['runs'][0]['results']:
        directory = res['locations'][0]['physicalLocation']['artifactLocation']['uri']
        split = directory.split("/")
        cwe = split[-1] # TODO
        file_name = split[-1]
        message = res['message']

         # Check that we get a true positive on an expected vulnerabal file
        if file_name.startswith("BAD") and has_vulnerability(message):
            true_positives += 1
            cwe_specific_results[cwe]["true_positives"].append(file_name)
            print(f"True positive on {file_name} with CWE {cwe}")
        
        # Check that we get a false positive on a non-vulnerable file
        elif file_name.startswith("GOOD") and has_vulnerability(message):
            false_positives += 1
            cwe_specific_results[cwe]["false_positives"].append(file_name)
            print(f"False positive on {file_name} with CWE {cwe}")
        
        # Check to see if we get a false negative on an expected vulnerabal file
        elif file_name.startswith("BAD") and not has_vulnerability(message):
            false_negatives += 1
            cwe_specific_results[cwe]["false_negatives"].append(file_name)
            print(f"False negative on {file_name} with CWE {cwe}")

        # Check to see if we get a true negative on a non-vulnerabal file
        elif file_name.startswith("GOOD") and not has_vulnerability(message):
            true_negatives += 1
            cwe_specific_results[cwe]["true_negatives"].append(file_name)
            print(f"True negative on {file_name} with CWE {cwe}")

    print(true_positives, false_positives)

    return {
    'true_positives': true_positives,
    'false_positives': false_positives,
    'cwe_specific_results': cwe_specific_results
}

if __name__ == "__main__":
    main()