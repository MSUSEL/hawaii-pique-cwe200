import pandas as pd
import json
import os
from collections import defaultdict

def parse_excel_to_json(excel_file, stats):
    # Load the Excel file
    df = pd.read_excel(excel_file, sheet_name=0, header=None)
    
    json_data = []
    current_file = None

    file_start = 0
    while file_start < len(df):
        # Check if the current row marks the start of a new file
        if pd.notna(df.iloc[file_start, 0]) and str(df.iloc[file_start, 0]).strip().endswith('.java'):
            # Extract the file name and path
            file_name = str(df.iloc[file_start, 0]).strip()
            file_path = str(df.iloc[file_start, 1]).strip()
        
            index = file_start + 2  # Move to the next row after the file name and path
        
            # Initialize starting and ending indices for categories
            variables_start, variables_end = None, None
            strings_start, strings_end = None, None
            comments_start, comments_end = None, None
            sinks_start, sinks_end = None, None

            # Iterate through rows until the next file or the end of the file
            while not (pd.notna(df.iloc[index, 0]) and str(df.iloc[index, 0]).strip().endswith('.java')) and index < len(df) - 1:
                current_value = str(df.iloc[index, 0]).strip()

                # Mark the start of each category
                if current_value == 'Variables':
                    variables_start = index + 1  # Move to the first data row after 'Variables'
                elif current_value == 'Strings':
                    strings_start = index + 1
                    variables_end = index - 1 if variables_start is not None else None
                elif current_value == 'Comments':
                    comments_start = index + 1
                    strings_end = index - 1 if strings_start is not None else None
                elif current_value == 'Sinks':
                    sinks_start = index + 1
                    comments_end = index - 1 if comments_start is not None else None
                
                index += 1

            # Determine the end bounds for the last category
            if variables_start is not None and variables_end is None:
                variables_end = index - 1
            if strings_start is not None and strings_end is None:
                strings_end = index - 1
            if comments_start is not None and comments_end is None:
                comments_end = index - 1
            if sinks_start is not None:
                sinks_end = index - 1
            
            file_start = index  # Move file_start to the beginning of the next file

            # Initialize the file info structure
            current_file = {
                "fileName": file_name,
                "filePath": file_path,
                "variables": [],
                "strings": [],
                "comments": [],
                "sinks": []
            }

            # Helper function to collect items within a category
            def collect_data(start, end, category):
                for i in range(start, end + 1):
                    if pd.isna(df.iloc[i, 1]):  # Stop if column 1 is empty
                        break
                    item = str(df.iloc[i, 1])
                    classification = str(df.iloc[i, 3]).strip().lower()

                    if pd.notna(item) and classification != 'nan':
                        is_sensitive = 'Yes' if classification == 'yes' else 'No'
                        current_file[category].append({"name": item.strip(), "IsSensitive": is_sensitive})

                        # Update stats
                        if is_sensitive == 'Yes':
                            stats[category]['Yes'] += 1
                        else:
                            stats[category]['No'] += 1

            # Collect data for each category
            if variables_start is not None:
                collect_data(variables_start, variables_end, 'variables')
            if strings_start is not None:
                collect_data(strings_start, strings_end, 'strings')
            if comments_start is not None:
                collect_data(comments_start, comments_end, 'comments')
            if sinks_start is not None:
                # Special handling for sinks as we track sink types too
                for i in range(sinks_start, sinks_end + 1):
                    if pd.isna(df.iloc[i, 1]):  # Stop if column 1 is empty
                        break
                    item = str(df.iloc[i, 1])
                    sink_type = str(df.iloc[i, 0]).strip()
                    classification = str(df.iloc[i, 3]).strip().lower() 

                    if pd.notna(item) and classification != 'nan':
                        is_sensitive = 'Yes' if classification == 'yes' else 'No'
                        sink_type = "N/A" if sink_type == 'nan' else sink_type
                        current_file['sinks'].append({"name": item.strip(), "type": sink_type, "IsSensitive": is_sensitive})
                        
                        if is_sensitive == 'No' and sink_type != 'N/A':
                            print(f"Sink '{item}' has a conflicting classification. {file_name}") 
                        elif is_sensitive == 'Yes' and sink_type == 'N/A':
                            print(f"Sink '{item}' is missing a sink type. {file_name}") 


                        # Update stats for sinks
                        if is_sensitive == 'Yes':
                            stats['sinks']['Yes'] += 1
                            # print(sink_type)
                        elif is_sensitive == 'No':
                            stats['sinks']['No'] += 1
                        else:
                            print(f"Invalid classification for sink: {classification}")

                        # Track the count of each sink type
                        stats['sink_types'][sink_type] += 1

            # Add the current file info to the json_data list
            if current_file['variables'] or current_file['strings'] or current_file['comments'] or current_file['sinks']:
                json_data.append(current_file)
        else:
            file_start += 1

    return json_data


def parse_all_excels_in_directory(directory_path):
    combined_json_data = []

    # Initialize statistics
    stats = {
        'variables': {'Yes': 0, 'No': 0},
        'strings': {'Yes': 0, 'No': 0},
        'comments': {'Yes': 0, 'No': 0},
        'sinks': {'Yes': 0, 'No': 0},
        'sink_types': defaultdict(int)  # Track each type of sink
    }

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")
            # Parse the Excel file and append the results
            json_data = parse_excel_to_json(file_path, stats)
            combined_json_data.extend(json_data)

    return combined_json_data, stats


# Example usage:
data_dir = os.path.join('testing', 'Labeling', 'Data')
combined_json_data, stats = parse_all_excels_in_directory(data_dir)

# Convert the combined data to a JSON string
json_output = json.dumps(combined_json_data, indent=4)

# Save the combined JSON data to a file
output_file_path = os.path.join('backend', 'src', 'bert', 'training', 'data', 'labels.json')
with open(output_file_path, 'w') as json_file:
    json_file.write(json_output)

print(f"Combined JSON file '{output_file_path}' created successfully.")

# Print the collected statistics
print("\nClassification Statistics:")
print(f"Variables - Sensitive: {stats['variables']['Yes']}, Non-Sensitive: {stats['variables']['No']}")
print(f"Strings - Sensitive: {stats['strings']['Yes']}, Non-Sensitive: {stats['strings']['No']}")
print(f"Comments - Sensitive: {stats['comments']['Yes']}, Non-Sensitive: {stats['comments']['No']}")
print(f"Sinks - Sensitive: {stats['sinks']['Yes']}, Non-Sensitive: {stats['sinks']['No']}")

print("\nSink Type Counts:")
for sink_type, count in stats['sink_types'].items():
    print(f"{sink_type}: {count}")
