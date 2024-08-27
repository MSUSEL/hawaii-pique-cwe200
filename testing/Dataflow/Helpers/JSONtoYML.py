import json
import os

# Load the JSON data from a file
with open(os.path.join('testing', 'Merge_JSONs', 'Labeled_JSONs', 'Toy_dataset_labeled.json'), 'r') as json_file:
    json_data = json.load(json_file)

# Initialize the YAML content as a string
yaml_content = """extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveVariables
    data:
"""

# Extract variables from JSON and add them to the YAML content
for entry in json_data:
    file_name = entry["fileName"]
    variables = entry["variables"]
    
    for variable in variables:
        variable_name = variable["name"]
        # Add each entry as a string in the desired format
        yaml_content += f'    - ["{file_name}", "{variable_name}"]\n'

# Output the YAML data to a file
output_path = os.path.join('testing', 'Dataflow', 'Helpers', 'SensitiveVariables.yml')
with open(output_path, 'w') as yaml_file:
    yaml_file.write(yaml_content)

print(f"YAML file has been generated successfully at {output_path}.")
