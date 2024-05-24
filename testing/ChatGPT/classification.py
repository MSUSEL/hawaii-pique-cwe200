"""
This script is used to compare the classifications made by ChatGPT to the actual CWEs of the files in the CWEToyDataset.
We moslty want to know if just using ChatGPT for classification is a viable option and if it can beat our current results.
"""
import json
import collections
import os

INPUT = 'backend/Files/CWEToyDataset/data.json'
BASE_DIR = 'backend/Files/CWEToyDataset/CWEToyDataset/src/main/java/com'
chat_gpt_classifications = collections.defaultdict(list)

# Load in and parse chatGPT's classifications
with open(INPUT, 'r') as f:
    data = json.load(f)

    for file in data:
        classifications = []
        for classification in file['classification']:
            classifications.append(classification['name'])
            if file['fileName'] in chat_gpt_classifications:
                chat_gpt_classifications[file['fileName']].append(classification['name'])
            else:
                chat_gpt_classifications[file['fileName']] = [classification['name']]

# Get the actual classifications specified by the directory each file belongs to

cwe_dict = {}

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".java"):
            # Split the path and get the CWE part
            path_parts = root.split(os.sep)
            if len(path_parts) > 1:
                    for item in path_parts:
                        if item.startswith("CWE-"):
                            cwe_name = item
                
                    # Add the file to the CWE key in the dictionary
                    if cwe_name not in cwe_dict:
                        cwe_dict[cwe_name] = []
                    cwe_dict[cwe_name].append(file)

# Compare the classifications
correct = 0
incorrect = 0

for cwe, files in cwe_dict.items():
    print(f"CWE-{cwe}")
    for file in files:
        if cwe in chat_gpt_classifications[file]:
            print(f"    {file}: Correct classification {chat_gpt_classifications[file]}")
            correct += 1
        elif "GOOD" in file and 'non-vulnerable' in chat_gpt_classifications[file] or ("GOOD" in file and not chat_gpt_classifications[file]):
            print(f"    {file}: Correct classification {chat_gpt_classifications[file]}")
            correct += 1
        else:
            print(f"    {file}: Incorrect classification as {chat_gpt_classifications[file]}" )
            incorrect += 1
    print()

print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")

                
        