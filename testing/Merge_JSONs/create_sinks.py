import json
import os

def process_sinks(labeled_file, parsed_file):
    """
    Processes a single labeled and parsed JSON file and returns combined data.
    """
    # Load labeled sinks JSON file
    with open(labeled_file, 'r') as file:
        labeled_sinks = json.load(file)

    # Load parsed sinks JSON file
    with open(parsed_file, 'r') as file:
        parsed_sinks = json.load(file)

    # Convert labeled sinks list into a dictionary for easier lookup
    labeled_sinks_dict = {item['fileName']: item['sinks'] for item in labeled_sinks}

    # List to store the combined results for this dataset
    combined_data = []

    # Loop through the parsed sinks
    for file_name, parsed_data in parsed_sinks.items():
        combined_sinks = []
        
        # Check if the file is in labeled sinks
        if file_name in labeled_sinks_dict:
            labeled_sinks_for_file = labeled_sinks_dict[file_name]
            
            # Loop through the parsed sinks and match with labeled sinks
            for sink in parsed_data['sinks']:
                is_sink = "no"
                sink_type = "non-sink"
                for labeled_sink in labeled_sinks_for_file:
                    if sink == labeled_sink['name']:
                        is_sink = "yes"
                        sink_type = labeled_sink['type']
                        break
                combined_sinks.append({
                    "name": sink,
                    "isSink": is_sink,
                    "type": sink_type
                })
        else:
            # If file is not in labeled sinks, all parsed sinks are non-sinks
            for sink in parsed_data['sinks']:
                combined_sinks.append({
                    "name": sink,
                    "isSink": "no",
                    "type": "non-sink"
                })
        
        # Add the combined data for this file to the output list
        combined_data.append({
            "fileName": file_name,
            "sinks": combined_sinks
        })
    
    return combined_data

def combine_sinks(labeled_files, parsed_files, output_file):
    """
    Combines sinks from multiple labeled and parsed JSON datasets.
    """
    combined_data = []

    # Process each labeled and parsed file pair
    for labeled_file, parsed_file in zip(labeled_files, parsed_files):
        combined_data += process_sinks(labeled_file, parsed_file)

    # Write the combined data to an output JSON file
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)

# Usage example

# Paths to the Toy dataset
toy_labeled_file = os.path.join('testing', 'Merge_JSONs', 'Labeled_JSONs', 'Toy_dataset_labeled_sinks.json')
toy_parsed_file = os.path.join('backend', 'Files', 'CWEToyDataset', 'parsedResults.json')

# Paths to the CVE dataset
CVE_labeled_file = os.path.join('testing', 'Merge_JSONs', 'Labeled_JSONs', 'Labeled_dataset_sinks.json')
CVE_parsed_file = os.path.join('backend', 'Files', 'ReviewSensFiles', 'parsedResults.json')

# List of labeled and parsed file pairs
labeled_files = [toy_labeled_file, CVE_labeled_file]
parsed_files = [toy_parsed_file, CVE_parsed_file]

# Output file for combined results
output_file = os.path.join('testing', 'Merge_JSONs', 'combined_sinks.json')

# Combine and write the results
combine_sinks(labeled_files, parsed_files, output_file)
