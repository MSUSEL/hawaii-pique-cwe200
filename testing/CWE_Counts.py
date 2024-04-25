import os
from pathlib import Path

def count_files(directory):
    cwe_counts = {}
    for cwe in os.listdir(directory):
        cwe_counts[cwe] = {'GOOD': 0, 'BAD': 0, 'TOTAL': 0}
        sub_dir = os.path.join(directory, cwe)
        for root, dirs, files in os.walk(sub_dir):
            java_files = [f for f in files if f.endswith('.java')]
            for file in java_files:
                if file.startswith('GOOD_'):
                    cwe_counts[cwe]['GOOD'] += 1
                elif file.startswith('BAD_'):
                    cwe_counts[cwe]['BAD'] += 1
                cwe_counts[cwe]['TOTAL'] += 1

    return cwe_counts

def print_counts(cwe_counts):
    good_total, bad_total, total = 0, 0, 0
    for cwe, counts in cwe_counts.items():
        good_total += counts['GOOD']
        bad_total += counts['BAD']
        total += counts['TOTAL']
        print(f"{cwe}: GOOD={counts['GOOD']}, BAD={counts['BAD']}, TOTAL={counts['TOTAL']}")
    print(f"Total: GOOD={good_total}, BAD={bad_total}, TOTAL={total}")

# Set the directory path
directory_path = 'backend/Files/CWEToyDataset/src/main/java/com/mycompany/app/'

# Count the files
cwe_file_counts = count_files(directory_path)

# Print the results
print_counts(cwe_file_counts)
