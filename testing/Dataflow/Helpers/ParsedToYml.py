import json
import os

# Load the parsed JSON data from a file
with open(os.path.join('backend', 'Files', 'CWEToyDataset', 'parsedResults.json'), 'r') as parsed_json_file:
    parsed_json_data = json.load(parsed_json_file)

# Initialize the YAML content as a string
yaml_content = """extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveVariables
    data:
"""

# Add all variables from the parsed JSON to the YAML content
for file_name, file_data in parsed_json_data.items():
    for variable_name in file_data["variables"]:
        yaml_content += f'    - ["{file_name}", "{variable_name}"]\n'

# Output the YAML data to a file
output_path = os.path.join('testing', 'Dataflow', 'Helpers', 'SensitiveVariables.yml')
with open(output_path, 'w') as yaml_file:
    yaml_file.write(yaml_content)

print(f"YAML file has been generated successfully at {output_path}.")
