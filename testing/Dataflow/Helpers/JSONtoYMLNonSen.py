import json
import os

# Load the labeled JSON data (true positives) from a file
with open(os.path.join('testing', 'Merge_JSONs', 'Labeled_JSONs', 'Toy_dataset_labeled.json'), 'r') as labeled_json_file:
    labeled_json_data = json.load(labeled_json_file)

# Load the parsed JSON data (all variables) from a file
with open(os.path.join('backend', 'Files', 'CWEToyDataset', 'parsedResults.json'), 'r') as parsed_json_file:
    parsed_json_data = json.load(parsed_json_file)

# Create a set of (fileName, variableName) for all true positives
labeled_variables = set()
for entry in labeled_json_data:
    file_name = entry["fileName"]
    for variable in entry["variables"]:
        labeled_variables.add((file_name, variable["name"]))

# Initialize the YAML content as a string
yaml_content = """extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveVariables
    data:
"""

# Extract variables from parsed JSON and add them to the YAML content if not in labeled variables
for file_name, file_data in parsed_json_data.items():
    variables = file_data["variables"]
    
    for variable_name in variables:
        # Only include the variable if it's not in the labeled JSON
        if (file_name, variable_name) not in labeled_variables:
            yaml_content += f'    - ["{file_name}", "{variable_name}"]\n'

# Output the YAML data to a file
output_path = os.path.join('testing', 'Dataflow', 'Helpers', 'NonSensitiveVariables.yml')
with open(output_path, 'w') as yaml_file:
    yaml_file.write(yaml_content)

print(f"YAML file has been generated successfully at {output_path}.")
