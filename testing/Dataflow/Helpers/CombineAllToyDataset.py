import json
import os

# Load the JSON data
with open(os.path.join('testing', 'Merge_JSONs', 'Labeled_JSONs', 'Toy_dataset_labeled.json'), 'r') as labeled_json_file:
    labeled_json_data = json.load(labeled_json_file)

with open(os.path.join('backend', 'Files', 'CWEToyDataset', 'parsedResults.json'), 'r') as parsed_json_file:
    parsed_json_data = json.load(parsed_json_file)

with open(os.path.join('testing', 'Merge_JSONs', 'Labeled_JSONs', 'Toy_dataset_labeled_sinks.json'), 'r') as labeled_sinks_json_file:
    labeled_sinks_json_data = json.load(labeled_sinks_json_file)

with open(os.path.join('testing', 'Dataflow', 'graph.json'), 'r') as graph_json_file:
    graph_json_data = json.load(graph_json_file)

# Function to check if an item is labeled as sensitive
def is_labeled_sensitive(file_name, item_name, labeled_json, item_type):
    for entry in labeled_json:
        if entry["fileName"] == file_name:
            return any(item.get("name") == item_name for item in entry.get(item_type, []))
    return False

# Function to get the graph for a variable
def get_variable_graph(file_name, variable_name, graph_json):
    for entry in graph_json:
        if entry["fileName"] == file_name:
            for var in entry["variables"]:
                if var["name"] == variable_name:
                    return var.get("graph", [])
    return []

# Initialize the combined JSON structure
combined_json = []

# Iterate over the parsed JSON data
for file_name, file_data in parsed_json_data.items():
    combined_entry = {
        "fileName": file_name,
        "variables": [],
        "strings": [],
        "comments": [],
        "sinks": []
    }

    # Process variables
    for variable in file_data["variables"]:
        is_sensitive = "yes" if is_labeled_sensitive(file_name, variable, labeled_json_data, "variables") else "no"
        graph = get_variable_graph(file_name, variable, graph_json_data)
        combined_entry["variables"].append({
            "name": variable,
            "isSensitive": is_sensitive,
            "graph": graph
        })

    # Process strings
    for string in file_data["strings"]:
        is_sensitive = "yes" if is_labeled_sensitive(file_name, string, labeled_json_data, "strings") else "no"
        combined_entry["strings"].append({
            "name": string,
            "isSensitive": is_sensitive
        })

    # Process comments
    for comment in file_data["comments"]:
        is_sensitive = "yes" if is_labeled_sensitive(file_name, comment, labeled_json_data, "comments") else "no"
        combined_entry["comments"].append({
            "name": comment,
            "isSensitive": is_sensitive
        })

    # Process sinks
    for sink in file_data["sinks"]:
        is_sink = "yes" if is_labeled_sensitive(file_name, sink, labeled_sinks_json_data, "sinks") else "no"
        combined_entry["sinks"].append({
            "name": sink,
            "isSink": is_sink
        })

    combined_json.append(combined_entry)

# Save the combined JSON data to a file
output_path = os.path.join('testing', 'Dataflow', 'Helpers', 'Combined_JSON.json')
with open(output_path, 'w') as combined_json_file:
    json.dump(combined_json, combined_json_file, indent=4)

print(f"Combined JSON file has been generated successfully at {output_path}.")
