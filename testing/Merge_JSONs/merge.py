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
                    
                    # Function to add items and avoid duplicates
                    def add_unique_items(existing_list, new_items, key="name"):
                        seen = {item[key] for item in existing_list}
                        for item in new_items:
                            name = item.get(key)
                            if name and name not in seen:
                                existing_list.append(item)
                                seen.add(name)
                            elif name and name in seen:
                                # Merge additional attributes for duplicates
                                for existing_item in existing_list:
                                    if existing_item[key] == name:
                                        for attr in ["type", "reason", "description"]:
                                            if attr in item and attr not in existing_item:
                                                existing_item[attr] = item[attr]

                    # Merge the fields, ensuring no duplicates
                    add_unique_items(merged_data[file_name]["variables"], entry.get("variables", []))
                    add_unique_items(merged_data[file_name]["strings"], entry.get("strings", []))
                    add_unique_items(merged_data[file_name]["comments"], entry.get("comments", []))
                    add_unique_items(merged_data[file_name]["sinks"], entry.get("sinks", []), key="name")

    # Convert the merged_data dictionary back into a list
    merged_list = [{"fileName": file_name, **contents} for file_name, contents in merged_data.items()]

    # Save the merged data into a new JSON file
    output_filepath = os.path.join("testing", "Merge_JSONs", "all_labeled_data.json")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='UTF-8') as output_file:
        json.dump(merged_list, output_file, indent=4)

    print(f"JSON files merged into {output_filepath}")

# Replace 'your_directory_path' with the path to the directory containing the JSON files
merge_json_files(os.path.join("testing", "Merge_JSONs", "Labeled_JSONs"))
