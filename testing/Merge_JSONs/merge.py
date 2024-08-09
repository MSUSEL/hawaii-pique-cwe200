import os
import json

def merge_json_files(directory):
    merged_data = {}

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'r', encoding='UTF-8') as file:
                data = json.load(file)
                
                # Merge each entry in the JSON file
                for entry in data:
                    file_name = entry['fileName']
                    
                    # If the file already exists in the merged_data, merge the contents
                    if file_name not in merged_data:
                        merged_data[file_name] = {
                            "variables": [],
                            "strings": [],
                            "comments": [],
                            "sinks": []
                        }
                    
                    # Merge the fields
                    merged_data[file_name]["variables"].extend(entry.get("variables", []))
                    merged_data[file_name]["strings"].extend(entry.get("strings", []))
                    merged_data[file_name]["comments"].extend(entry.get("comments", []))
                    merged_data[file_name]["sinks"].extend(entry.get("sinks", []))

    # Convert the merged_data dictionary back into a list
    merged_list = [{"fileName": file_name, **contents} for file_name, contents in merged_data.items()]

    # Save the merged data into a new JSON file
    output_filepath = os.path.join("testing", "Merge_JSONs", "all_labeled_data.json")
    with open(output_filepath, 'w', encoding='UTF-8') as output_file:
        json.dump(merged_list, output_file, indent=4)

    print(f"JSON files merged into {output_filepath}")

# Replace 'your_directory_path' with the path to the directory containing the JSON files
merge_json_files(os.path.join("testing", "Merge_JSONs", "Labeled_JSONs"))
