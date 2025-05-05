import os
import json
import re
from collections import defaultdict
import sys

# Load the SARIF file
def load_sarif(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        sarif_data = json.load(file)
    return sarif_data

# Function to build the graph from the SARIF data
def build_flow_graph(results):
    graphs = defaultdict(list)
    variables = defaultdict(list)
    for result in results:
        if 'codeFlows' in result:
            for thread_flow in result['codeFlows'][0]['threadFlows']:
                locations = thread_flow['locations']
                nodes = []
                for i in range(len(locations)):
                    value = locations[i]['location']['message']['text'].split(":")[0].strip()
                    if i == len(locations) - 1:
                        type = result['message']['text']
                    else:
                        type = locations[i]['location']['message']['text'].split(":")[-1].strip()
                    context = locations[i]['location']['physicalLocation']['contextRegion']['snippet']['text'].strip()
                    node = (value, context, type)
                    nodes.append(node)
                file_path = locations[-1]['location']['physicalLocation']['artifactLocation']['uri']
                key = (locations[-1]['location']['message']['text'])
                graphs[key+" ---"+file_path].append(nodes) 
        elif '|' in result['message']['text']:
            # This is for the case where there is no code flow (e.g. an exception, parameter, etc.), but I still want to capture the context
            message = result['message']['text']
            key = message.split("|")[0].strip()
            if key.startswith('this.'):
                key = key.replace('this.', '')
            context = result['locations'][0]['physicalLocation']['contextRegion']['snippet']['text'].strip()
            file_path = result['locations'][0]['physicalLocation']['artifactLocation']['uri']
            type = message.split("|")[1].strip()
            nodes = [(key, context, type)]
            variables[key+" ---"+file_path].append(nodes)
    # Add only the variables that don't have data flow
    for variable in variables:
        if not variable in graphs:
            graphs[variable].extend(variables[variable])   
    return graphs

def build_json(graphs, output_path):
    data = defaultdict(list)
    for main_variable in graphs:
        variable_name = main_variable.split(" ---")[0]
        file_name = main_variable.split(" ---")[1].split("/")[-1]
        combined_graph = {}
        for variable_graphs in graphs[main_variable]:
            previous_node = None
            for node_index, node in enumerate(variable_graphs):
                node_name = node[0]
                node_context = node[1]
                type = node[2]
                if node_name in combined_graph:
                    if node_context not in combined_graph[node_name]["contexts"]:
                        combined_graph[node_name]["contexts"].append(node_context)
                else:
                    combined_graph[node_name] = {
                        "name": node_name,
                        "type": type,
                        "contexts": [node_context],
                        "nextNode": None
                    }
                if previous_node:
                    combined_graph[previous_node]["nextNode"] = node_name
                previous_node = node_name
            if previous_node:
                combined_graph[previous_node]["nextNode"] = "end"
        combined_flow = []
        for node_name, node_data in combined_graph.items():
            combined_flow.append({
                "name": node_data["name"],
                "type": node_data["type"],
                "context": " | ".join(node_data["contexts"]),
                "nextNode": node_data["nextNode"]
            })
        ordered_flow = []
        end_node_index = None
        for i, node in enumerate(combined_flow):
            if node["nextNode"] == "end":
                end_node_index = i
                break
        if end_node_index is not None:
            for i, node in enumerate(combined_flow):
                if i > end_node_index and node["nextNode"] != "end":
                    for j in range(len(ordered_flow)):
                        if ordered_flow[j]["name"] == node["nextNode"]:
                            ordered_flow.insert(j, node)
                            break
                else:
                    ordered_flow.append(node)
        else:
            ordered_flow = combined_flow
        data[file_name].append({
            "name": variable_name,
            "graph": ordered_flow
        })
    final_data = []
    for file_name, variables in data.items():
        final_data.append({
            "fileName": file_name,
            "variables": variables
        })
    with open(output_path, "w") as file:
        json.dump(final_data, file, indent=4)
    return final_data

def run(input_path='slice.sarif', output_path='output.json'):
    # Extract results from SARIF data
    sarif_data = load_sarif(input_path)
    results_data = sarif_data['runs'][0]['results']

    # Build the graph from the SARIF data
    flow_graphs = build_flow_graph(results_data)

    return build_json(flow_graphs, output_path)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        run(input_path, output_path)
    else:
        run()