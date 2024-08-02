import pandas as pd
from collections import defaultdict
import json

'''
Used to parse the sinks sheet from the excel file
'''

def parse_review(file_path, sheet_names):
    # Initialize the final structure
    sinks = defaultdict(list)
    for sheet_name in sheet_names:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Find the locations of the labels
        locations = df.where(df == "Label").stack().index.tolist()

        # Populate the structure with classifications
        for i in range(len(locations)):
            file_name = (df.iloc[locations[i][0] - 1, 0])
            start_row = locations[i][0] + 1
            end_row = locations[i+1][0] if i+2 < len(locations) else len(df)
            
            labels = df.iloc[start_row:end_row, 0].tolist()
            names = df.iloc[start_row:end_row, 1].tolist()

            sink_list = []
            for label, name in zip(labels, names):
                if str(label) != 'nan' and str(name) != 'nan':
                    sink_list.append({"name": name, "type": label})
            
            sinks[file_name] = sink_list
    return sinks

def outputJSON(sinks_dict):
    json_output = []
    for file_name, sinks in sinks_dict.items():
        file_entry = {"fileName": file_name, "sinks": []}
        for sink in sinks:
            file_entry["sinks"].append({"name": sink["name"], "type": sink["type"]})
        json_output.append(file_entry)

    # Output the JSON to a file
    output_path = 'testing/ChatGPT/Sinks/labeled_sinks.json'
    with open(output_path, 'w') as json_file:
        json.dump(json_output, json_file, indent=4)

# Read the Excel file to get sheet names
file_path = 'testing/ChatGPT/Sinks/Sinks.xlsx'
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
sheet_names = sheet_names[3:]  # Adjust this if needed
sheet_names = ['Sinks']  # Use specific sheet names if needed

print(sheet_names)

# Create structure for all sheets
sinks = parse_review(file_path, sheet_names)
outputJSON(sinks)
