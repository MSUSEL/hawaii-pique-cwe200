"""
This script reads in the Excel files containing the labeled data and converts it to a JSON file. 
The JSON file will be used to train the model for the next iteration.
"""

import pandas as pd
import json
import os

def parse_excel_to_json(excel_file):
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
        
            index = file_start + 2 # Move to the next row after the file name and path
        
            variables_start, variables_end = 0, 0
            strings_start, strings_end = 0, 0
            comments_start, comments_end = 0, 0
            sinks_start, sinks_end = 0, 0

            while not str(df.iloc[index, 0]).strip().endswith('.java') and index < len(df) - 1:
                
                if str(df.iloc[index, 0]).strip() == 'Variables':
                    variables_start = index
                elif str(df.iloc[index, 0]).strip() == 'Strings':
                    variables_end = index - 1
                    strings_start = index
                elif str(df.iloc[index, 0]).strip() == 'Comments':
                    strings_end = index - 1
                    comments_start = index
                elif str(df.iloc[index, 0]).strip() == 'Sinks':
                    comments_end = index - 1
                    sinks_start = index
                
                
                index += 1
                        
            sinks_end = index - 1
            file_start = index        


            # Initialize the file info structure
            current_file = {
                "fileName": file_name,
                "filePath": file_path,
                "variables": [],
                "strings": [],
                "comments": [],
                "sinks": []
            }

            # Collect variables
            for i in range(variables_start, variables_end + 1):
                if pd.isna(df.iloc[i, 1]):
                    continue
                item = str(df.iloc[i, 1])
                classification = str(df.iloc[i, 3]).strip().lower()

                if pd.notna(item) and classification != 'nan':
                    is_sensitive = 'Yes' if classification == 'yes' else 'No'
                    current_file['variables'].append({"name": item.strip(), "IsSensitive": is_sensitive})

            # Collect strings
            for i in range(strings_start, strings_end + 1):
                if pd.isna(df.iloc[i, 1]):
                    continue
                item = str(df.iloc[i, 1])
                classification = str(df.iloc[i, 3]).strip().lower()

                if pd.notna(item) and classification != 'nan':
                    is_sensitive = 'Yes' if classification == 'yes' else 'No'
                    current_file['strings'].append({"name": item.strip(), "IsSensitive": is_sensitive})
            
            # Collect comments
            for i in range(comments_start, comments_end + 1):
                if pd.isna(df.iloc[i, 1]):
                    continue
                item = str(df.iloc[i, 1])
                classification = str(df.iloc[i, 3]).strip().lower()

                if pd.notna(item) and classification != 'nan':
                    is_sensitive = 'Yes' if classification == 'yes' else 'No'
                    current_file['comments'].append({"name": item.strip(), "IsSensitive": is_sensitive})

            # Collect sinks
            for i in range(sinks_start, sinks_end + 1):
                if pd.isna(df.iloc[i, 1]):
                    continue
                item = str(df.iloc[i, 1])
                sink_type = str(df.iloc[i, 0]).strip()
                classification = str(df.iloc[i, 3]).strip().lower() 

                if pd.notna(item) and classification != 'nan' and classification != 'kyler':
                    is_sensitive = 'Yes' if classification == 'yes' else 'No'
                    current_file['sinks'].append({"name": item.strip(), "type": sink_type, "IsSensitive": is_sensitive})

        
            # Add the current file info to the json_data list
            if current_file['variables'] or current_file['strings'] or current_file['comments'] or current_file['sinks']:
                json_data.append(current_file)
        else:
            file_start += 1

    # Convert the list of dictionaries to a JSON string
    json_output = json.dumps(json_data, indent=4)
    return json_output
    


# Example usage:
data_dir = os.path.join('testing', 'Labeling', 'Data')
json_output = parse_excel_to_json(os.path.join(data_dir, 'CWEToyDataset.xlsx'))

# Save the JSON data to a file
with open(os.path.join('testing', 'Labeling', 'Data.json'), 'w') as json_file:
    json_file.write(json_output)

print("JSON file 'output.json' created successfully.")
