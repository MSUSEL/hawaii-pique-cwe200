import json
from collections import defaultdict

# Load the SARIF file
with open('Backend/Files/jenkins-jenkins-1.582/Bert-result.sarif', 'r') as file:
    sarif_data = json.load(file)

# Initialize a dictionary to track paths leading to the same sink
sink_paths = defaultdict(list)

# Variable to track total paths
total_paths = 0

# Traverse the SARIF results to extract sinks and their associated paths
for run in sarif_data['runs']:
    for result in run['results']:
        rule_id = result['ruleId'].split('/')[-1]
        
        # Count every path in total_paths, regardless of ruleId
        total_paths += 1
        
        # Skip if ruleId is 531 or 540 when considering duplicates
        if rule_id != '531' and rule_id != '540':
            # Extract the sink location
            locations = result['locations']
            sink = (
                f"{locations[-1]['physicalLocation']['artifactLocation']['uri']} "
                f"{locations[-1]['physicalLocation']['region']['startLine']} "
                f"{locations[-1]['physicalLocation']['region']['startColumn']} "
                f"{locations[-1]['physicalLocation']['region']['endColumn']} "
            )

            # Store the path to the sink
            if result['ruleId'] in sink_paths[sink]:
                print(f"Duplicate detected at sink {sink} with {result['ruleId']}.")
            sink_paths[sink].append(result['ruleId'])

# Identify and count duplicates
duplicate_count = 0
for sink, paths in sink_paths.items():
    if len(paths) > 1:
        # print(f"Duplicate detected at sink {sink} with {len(paths)} paths., {paths}")
        duplicate_count += len(paths) - 1

# Calculate percentage of duplicates
if total_paths > 0:
    duplicate_percentage = (duplicate_count / total_paths) * 100
else:
    duplicate_percentage = 0

print(f"Total number of duplicates: {duplicate_count}")
print(f"Total number of paths: {total_paths}")
print(f"Percentage of paths that are duplicates: {duplicate_percentage:.2f}%")
