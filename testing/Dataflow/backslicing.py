import os
import json
import networkx as nx
import re
from collections import defaultdict

# Load the SARIF file
file_path = 'slice.sarif'
java_project_path = 'backend/Files/CWEToyDataset/CWEToyDataset'
# java_project_path = 'backend/Files/test/test'


with open(file_path, 'r', encoding='UTF-8') as file:
    sarif_data = json.load(file)

# Function to clean up value formatting
def clean_value(value):
    # Remove semicolons, closing parentheses, and commas from the end
    value = value.replace(';', '').strip().rstrip(',).')

    # If the value ends with a period, it likely indicates an incomplete call
    # Keep reading until a space or termination point is found
    if value.endswith('.'):
        value = re.match(r'.*\)', value) or value

    # Ensure the full call is captured by checking for matched parentheses
    if value.count('(') > value.count(')'):
        value += ')'

    return value

# Function to read a file and extract a snippet based on the region with context
def extract_snippet_with_context(file_path, start_line, start_column, end_line=None, end_column=None, context_radius=5):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
        # Extract the exact value within the region
        value = lines[start_line - 1][start_column-1:end_column] if start_column and end_column else lines[start_line - 1].strip()
        
        return clean_value(value.strip())

# Function to build the graph from the SARIF data
def build_flow_graph(results):
    graphs = defaultdict(list)

    for result in results:
        # Extract the flow path (if it exists)
        if 'codeFlows' in result:
            for thread_flow in result['codeFlows'][0]['threadFlows']:
                locations = thread_flow['locations']
                
                nodes = []

                # Process each location in order to build the graph
                for i in range(len(locations)):

                    # current_location = locations[i]['location']['physicalLocation']
                    # current_file = os.path.join(java_project_path, current_location['artifactLocation']['uri'])
                    # current_region = current_location['region']


                    # value = extract_snippet_with_context(
                    #     current_file,
                    #     current_region['startLine'],
                    #     current_region.get('startColumn', 0),
                    #     current_region.get('endLine'),
                    #     current_region.get('endColumn')
                    # )

                    value = locations[i]['location']['message']['text'].split(":")[0].strip()
                    if i == len(locations) - 1:
                        type = result['message']['text']
                    else:
                        type = locations[i]['location']['message']['text'].split(":")[-1].strip()

                    context = locations[i]['location']['physicalLocation']['contextRegion']['snippet']['text']

                    node = (value, context, type)
                    nodes.append(node)

                file_path = locations[-1]['location']['physicalLocation']['artifactLocation']['uri']   
                key = (locations[-1]['location']['message']['text'])
                graphs[key+" ---"+file_path].append(nodes)    
    return graphs



def build_json(graphs):
    data = defaultdict(list)  # Use defaultdict to group by file name

    for main_variable in graphs:
        # Extract variable name and file name
        variable_name = main_variable.split(" ---")[0]
        file_name = main_variable.split(" ---")[1].split("/")[-1]

        # Initialize a dictionary to keep track of combined flows
        combined_graph = {}
        
        # Iterate over all graphs for the current variable
        for variable_graphs in graphs[main_variable]:
            previous_node = None  # Keep track of the previous node in the sequence

            for node_index, node in enumerate(variable_graphs):
                node_name = node[0]
                node_context = node[1]
                type = node[2]
                
                # If the node already exists, append the context to the existing node
                if node_name in combined_graph:
                    combined_graph[node_name]["contexts"].append(node_context)
                else:
                    # Otherwise, add the node with its context
                    combined_graph[node_name] = {
                        "name": node_name,
                        "type": type,
                        "contexts": [node_context],
                        "nextNode": None  # Initialize nextNode as None
                    }
                
                # Set the nextNode for the previous node
                if previous_node:
                    combined_graph[previous_node]["nextNode"] = node_name

                # Update the previous node
                previous_node = node_name
            
            # Mark the last node as "end"
            if previous_node:
                combined_graph[previous_node]["nextNode"] = "end"
        
        # Convert the combined graph dictionary into a list of nodes
        combined_flow = []
        for node_name, node_data in combined_graph.items():
            # Combine the contexts into one, or keep them separate if needed
            combined_flow.append({
                "name": node_data["name"],
                "type": node_data["type"],
                "context": " | ".join(node_data["contexts"]),  # Combine contexts with a separator
                "nextNode": node_data["nextNode"]
            })

        # Final pass: ensure nodes are correctly ordered
        ordered_flow = []
        end_node_index = None

        # Find the index of the node marked with "end"
        for i, node in enumerate(combined_flow):
            if node["nextNode"] == "end":
                end_node_index = i
                break

        if end_node_index is not None:
            # Reorder nodes to group those with the same nextNode together before the end node
            for i, node in enumerate(combined_flow):
                if i > end_node_index and node["nextNode"] != "end":
                    # Find the correct position for this node based on its nextNode
                    for j in range(len(ordered_flow)):
                        if ordered_flow[j]["name"] == node["nextNode"]:
                            ordered_flow.insert(j, node)
                            break
                else:
                    ordered_flow.append(node)
        else:
            ordered_flow = combined_flow

        # Add the variable and its associated ordered graph to the file group
        data[file_name].append({
            "name": variable_name,
            "graph": ordered_flow  # Directly output the graph
        })

    # Prepare the final data structure
    final_data = []
    for file_name, variables in data.items():
        final_data.append({
            "fileName": file_name,
            "variables": variables
        })

    # Write the JSON data to a file
    with open(os.path.join("testing", "Dataflow", "graph.json"), "w") as file:
        json.dump(final_data, file, indent=4)

        


# Extract results from SARIF data
results_data = sarif_data['runs'][0]['results']

# Build the graph from the SARIF data
flow_graphs = build_flow_graph(results_data)

build_json(flow_graphs)


