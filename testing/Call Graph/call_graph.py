import os
import json
from collections import defaultdict
import re

# Load the SARIF file
def load_sarif(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        sarif_data = json.load(file)
    return sarif_data


# Function to build the graph from the SARIF data
def build_graph(results):
    graphs = defaultdict(list)
    variables = defaultdict(list)
    for result in results:
        if 'codeFlows' in result:
            sink = re.search(r'\[(.*?)\]', result['message']['text'].split("||")[-1].strip()).group(1) if re.search(r'\[(.*?)\]', result['message']['text'].split("||")[-1].strip()) else None
            print(sink)
            for thread_flow in result['codeFlows'][0]['threadFlows']:
                locations = thread_flow['locations']
                nodes = []
                last_java_file = None
                for i in range(len(locations)):
                    value = locations[i]['location']['message']['text'].split(":")[0].strip()
                    # print(value)
                    physicalLocation = locations[i]['location']['physicalLocation']
                    context = ""

                    # Handle missing contextRegion
                    if 'contextRegion' in physicalLocation and 'snippet' in physicalLocation['contextRegion']:
                        context = physicalLocation['contextRegion']['snippet']['text'].strip()
                    else:
                        context = "No context available"

                    # Ensure 'type' is defined
                    type = "MethodCall"  # or assign the appropriate type based on your context
                    if i == len(locations) - 1:
                        value = sink
                    node = (value, context, type)
                    nodes.append(node)
                    
                file_path = locations[-1]['location']['physicalLocation']['artifactLocation']['uri']
                key = sink
                graphs[key + " ---" + file_path].append(nodes)
        else:
            print("No codeFlows in result")
    for variable in variables:
        if variable not in graphs:
            graphs[variable].extend(variables[variable])
    return graphs


def build_json(graphs, main_file_name):
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
    with open(os.path.join("testing", "Call Graph", main_file_name + ".json"), "w") as file:
        json.dump(final_data, file, indent=4)
    return final_data

def run(file_path='callgraph.sarif'):
    file_name = os.path.basename(file_path).split(".")[0]
    # Extract results from SARIF data
    sarif_data = load_sarif(file_path)
    results_data = sarif_data['runs'][0]['results']

    # Build the graph from the SARIF data
    flow_graphs = build_graph(results_data)

    return build_json(flow_graphs, file_name)

if __name__ == '__main__':
    run()
