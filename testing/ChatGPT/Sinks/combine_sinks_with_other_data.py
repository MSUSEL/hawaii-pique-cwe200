import json

'''
Used to combine the labeled toy dataset, variables, strings, and comments, with the labeled sinks.
'''

def combine_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
        
        # Convert data1 and data2 to dictionaries for quick lookup
        data1_dict = {entry["fileName"]: entry for entry in data1}
        data2_dict = {entry["fileName"]: entry for entry in data2}
        
        # Iterate through data1 and replace sinks with those from data2
        for file_name, entry in data1_dict.items():
            if file_name in data2_dict:
                sinks = data2_dict[file_name]["sinks"]
                formatted_sinks = [{"name": sink["name"], "type": sink["type"]} for sink in sinks]
                entry["sinks"] = formatted_sinks
        
        # Convert the dictionary back to a list
        combined_data = list(data1_dict.values())
        
        # Save the combined data back to a new JSON file
        with open(output_path, 'w') as outfile:
            json.dump(combined_data, outfile, indent=4)

# Example usage
file1_path = 'path_to_file1.json'
file2_path = 'path_to_file2.json'
output_path = 'path_to_output_file.json'

combine_files(file1_path, file2_path, output_path)
print(f'Combined file saved to {output_path}')
