import json

# Load the JSON files
with open('backend/Files/ReviewSensFiles/output_prompts.json', 'r', encoding='utf-8') as f1, open('testing/ChatGPT/agreed_classifications.json', 'r', encoding='utf-8') as f2:
    format1_data = json.load(f1)
    format2_data = json.load(f2)

# Convert format2 data to a dictionary for easier access
format2_dict = {file['fileName']: file for file in format2_data['files']}

# Combine the two formats
combined_data = {}

for file_name in format1_data:
    
    # Initialize the result structure for the current file
    file_result = {
        "variables": {"input": "", "output": ""},
        "strings": {"input": "", "output": ""},
        "comments": {"input": "", "output": ""}
    }

    # Check if the file exists in format2 data and update the results accordingly
    if file_name in format2_dict:
        sensitive_variables = format2_dict[file_name].get('sensitiveVariables', [])
        sensitive_strings = format2_dict[file_name].get('sensitiveStrings', [])
        sensitive_comments = format2_dict[file_name].get('sensitiveComments', [])
        
        file_result['variables']['output'] = sensitive_variables
        file_result['strings']['output'] = sensitive_strings
        file_result['comments']['output'] = sensitive_comments
        file_result['variables']['input'] = format1_data[file_name]['variables']['input']
        file_result['strings']['input'] = format1_data[file_name]['strings']['input']
        file_result['comments']['input'] = format1_data[file_name]['comments']['input']

    
    combined_data[file_name] = file_result

# Save the combined JSON to a new file
with open('testing/ChatGPT/training_data.json', 'w', encoding='utf-8') as combined_file:
    json.dump(combined_data, combined_file, indent=4)

print("JSON files combined successfully.")
