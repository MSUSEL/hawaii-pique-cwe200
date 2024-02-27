import json
import os
from collections import defaultdict
PROJECT = "CWEToyDataset"
CHATGPTPATH = f"Files\\{PROJECT}\\data.json"
CODEQLPATH = f"Files\\{PROJECT}\\results.sarif"
SRCCODEPATH = f"Files\\{PROJECT}"
CWESPATH = f"Files\\{PROJECT}\\src\\main\\java\\com\\mycompany\\app"


def main():
    # Read in data from CHatGPTPath as a JSON
        os.chdir("backend/")
        java_files = get_java_files(SRCCODEPATH)
        chatgpt_results = read_data(CHATGPTPATH)
        cwes = get_directories_in_dir(CWESPATH)

        analyze_chatgpt_results(chatgpt_results, java_files)

        # codeql_results = read_data(CODEQLPATH)
        
        
        print(1)
        

def read_data(path):
    with open(path, "r") as file:
        
        if path.endswith(".json"):
            return json.load(file)
        
        if path.endswith(".sarif"):
            return json.read()

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


def analyze_chatgpt_results(chatgpt_results, java_files):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    cwe_specific_results = defaultdict(lambda: defaultdict(list))
    
    for i, file in enumerate(java_files):
        split = file.split("\\")
        file_name = split[-1]
        cwe = split[-2]

        for res in chatgpt_results:
            if res['key'] == file_name:
                val = res['value']

                # Check that we get a true positive on an expected vulnerabal file
                if file_name.startswith("BAD") and has_vulnerability(val):
                    true_positives += 1
                    cwe_specific_results[cwe]["true_positives"].append(file_name)
                    print(f"True positive on {file_name} with CWE {cwe}")
                
                # Check that get a false positive on a non-vulnerable file
                elif file_name.startswith("GOOD") and has_vulnerability(val):
                    false_positives += 1
                    cwe_specific_results[cwe]["false_positives"].append(file_name)
                    print(f"False positive on {file_name} with CWE {cwe}")
                
                # Check that if we get a false negative on an expected vulnerabal file
                elif file_name.startswith("BAD") and not has_vulnerability(val):
                    false_negatives += 1
                    cwe_specific_results[cwe]["false_negatives"].append(file_name)
                    print(f"False negative on {file_name} with CWE {cwe}")

                # Check that if we get a true negative on an expected non-vulnerabal file
                elif file_name.startswith("GOOD") and not has_vulnerability(val):
                    true_negatives += 1
                    cwe_specific_results[cwe]["true_negatives"].append(file_name)
                    print(f"True negative on {file_name} with CWE {cwe}")


    print(true_positives, false_positives)

    return {
    'true_positives': true_positives,
    'false_positives': false_positives,
    'cwe_specific_results': cwe_specific_results
}


def has_vulnerability(chatgpt_results):
    return True if len(chatgpt_results) > 0 else False


if __name__ == "__main__":
    main()