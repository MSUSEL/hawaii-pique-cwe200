import csv
import json

# Function to read the CSV and process the data
def process_csv_to_json(csv_file, json_file):
    data = {"files": []}

    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_name = row['File Name']
            variables = [element.strip().strip('"').strip("'").replace('?', '') for element in row['Sensitive Variables'].split(',') if element.strip().strip('"').strip("'")]
            strings = [element.strip().strip('"').strip("'") for element in row['Sensitive Strings'].split(',') if element.strip().strip('"').strip("'")]
            comments = [element.strip().strip('"').strip("'") for element in row['Sensitive Comments'].split(',') if element.strip().strip('"').strip("'")]

            # Check if the file already exists in the data
            existing_file = next((item for item in data['files'] if item['fileName'] == file_name), None)
            if existing_file:
                existing_file['variables'].extend(variables)
                existing_file['strings'].extend(strings)
                existing_file['comments'].extend(comments)
            elif file_name != "":
                new_file_entry = {
                    "fileName": file_name,
                    "variables": variables,
                    "strings": strings,
                    "comments": comments
                }
                data['files'].append(new_file_entry)

    # Writing to json file
    with open(json_file, 'w') as json_out:
        json.dump(data, json_out, indent=4)

# Usage
csv_file = 'testing/ChatGPT/input.csv'
json_file = 'testing/ChatGPT/output.json'
process_csv_to_json(csv_file, json_file)
