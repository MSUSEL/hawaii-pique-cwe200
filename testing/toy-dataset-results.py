import json
import os
from collections import defaultdict

CHATGPTPATH = "Files\\Demo\\data.json"
CODEQLPATH = "Files\\Demo\\results.sarif"
SRCCODEPATH = "Files\\Demo"


def main():
    # Read in data from CHatGPTPath as a JSON
        os.chdir("backend/")
        java_files = get_java_files(SRCCODEPATH)
        chatgpt_results = read_data(CHATGPTPATH)
        cwes = get_directories_in_dir(SRCCODEPATH)

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
    cwe_specific_results = defaultdict(list)
    
    for file in java_files:
        file_name = file.split("\\")[-1]
        
        # Check that we get a true positive on an expected vulnerabal file
        if file_name.startswith("BAD") and has_vulnerability(chatgpt_results[file]):
            print(chatgpt_results[file])
            true_positives += 1
        
        # Check that get a false positive on a non-vulnerable file
        if file_name.startswith("GOOD") and has_vulnerability(chatgpt_results[file]):
            print(chatgpt_results[file])
            false_positives += 1

        print(file)
        if file in chatgpt_results:
            print(chatgpt_results[file])

        # print(chatgpt_results[file]

def has_vulnerability(file, chatgpt_results):
    return True if len(chatgpt_results) > 0 else False


if __name__ == "__main__":
    main()