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
        complete_codeql_results = check_missed(codeql_results['cwe_specific_results'], java_files)
        save_results(complete_codeql_results, "../testing/codeql_results.json")

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
    return [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]


def has_vulnerability(results):
    return True if len(results) > 0 else False

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

def analyze_codeql_results(codeql_results, java_files):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    cwe_specific_results = defaultdict(lambda: defaultdict(set))
    cwe_extra_results = defaultdict(lambda: defaultdict(set))

    results = codeql_results['runs'][0]['results']
    # This tells us all the CWEs that were queried in the codeql analysis
    cwes_queries = [c['properties']['cwe'] for c in codeql_results['runs'][0]['tool']['driver']['rules']]

    num_results = len(results)

    # Match a specific CWE query to a result so that we can track the results for each CWE
    for cwe_query in cwes_queries:
        for res in results:
            directory = res['locations'][0]['physicalLocation']['artifactLocation']['uri']
            split = directory.split("/")
            cwe = split[-2]
            file_name = split[-1]

            # Match the CWE query to the CWE of the result. 
            # It is possible that the result will be a different CWE since we have overlapping CWEs, 
            # however, for testing purposes, we only care about the CWE query matching the result 
            # Since that is how we could them.
            if cwe_query == cwe:
                message = res['message']
                # Check that we get a true positive on an expected vulnerabal file
                if file_name.startswith("BAD") and has_vulnerability(message):
                    true_positives += 1
                    cwe_specific_results[cwe]["true_positives"].add(file_name)
                    print(f"True positive on {file_name} with CWE {cwe}")
                
                # Check that we get a false positive on a non-vulnerable file
                elif file_name.startswith("GOOD") and has_vulnerability(message):
                    false_positives += 1
                    cwe_specific_results[cwe]["false_positives"].add(file_name)
                    print(f"False positive on {file_name} with CWE {cwe}")
                
                # Check to see if we get a false negative on an expected vulnerabal file
                elif file_name.startswith("BAD") and not has_vulnerability(message):
                    false_negatives += 1
                    cwe_specific_results[cwe]["false_negatives"].add(file_name)
                    print(f"False negative on {file_name} with CWE {cwe}")

                # Check to see if we get a true negative on a non-vulnerabal file
                elif file_name.startswith("GOOD") and not has_vulnerability(message):
                    true_negatives += 1
                    cwe_specific_results[cwe]["true_negatives"].add(file_name)
                    print(f"True negative on {file_name} with CWE {cwe}")
            
            else:
                cwe_extra_results[cwe_query + " query results"][cwe].add(file_name)

    print(true_positives, false_positives)

    return {
    'true_positives': true_positives,
    'false_positives': false_positives,
    'cwe_specific_results': cwe_specific_results,
    'cwe_extra_results': cwe_extra_results 
}

if __name__ == "__main__":
    main()