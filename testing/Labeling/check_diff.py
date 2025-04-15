import os
import json
import hashlib
from glob import glob

###############################################################################
# Helper functions for hashing flows
###############################################################################

def hash_single_step(step):
    """
    Returns a string that captures only these parts of a single 'step':
      - step (index)
      - variableName
      - uri
      - type
      - code
    Other fields are ignored.
    """
    step_num     = step.get('step', '')
    variableName = step.get('variableName', '')
    step_type    = step.get('type', '')
    code         = step.get('code', '')

    return f"{step_num}|{variableName}|{step_type}|{code}"

def hash_flow(flow_entry):
    """
    Compute a hash of the entire flow (list of steps) by concatenating
    the per-step strings and hashing them. This helps match flows
    between labeled and tool outputs without relying on index or order.
    """
    step_hashes = []
    for step in flow_entry.get("flow", []):
        step_hashes.append(hash_single_step(step))
    big_string = "~~".join(step_hashes)
    return hashlib.md5(big_string.encode('utf-8')).hexdigest()

###############################################################################
# Main script
###############################################################################

def main():
    labeled_dir = os.path.join("testing", "Labeling", "FlowData")
    backend_dir = os.path.join("backend", "Files")

    same_projects = []
    diff_projects = []

    # Collect all labeled JSON files
    labeled_json_files = glob(os.path.join(labeled_dir, "*.json"))

    for labeled_path in labeled_json_files:
        filename = os.path.basename(labeled_path)
        dataset_name = filename.replace(".json", "")  # e.g. "CWEToyDataset"

        # Corresponding tool output path
        tool_json_path = os.path.join(backend_dir, dataset_name, "flowMapsByCWE.json")
        if not os.path.isfile(tool_json_path):
            print(f"!! No tool output found for {dataset_name}, skipping...")
            continue

        # Load labeled JSON
        with open(labeled_path, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)

        # Load tool JSON
        with open(tool_json_path, "r", encoding="utf-8") as f:
            tool_data = json.load(f)

        # Collect hashed flows from labeled data
        labeled_flows_set = set()
        for cwe_id, file_list in labeled_data.items():
            for file_item in file_list:
                flows = file_item.get("flows", [])
                for flow_entry in flows:
                    labeled_flows_set.add(hash_flow(flow_entry))

        # Collect hashed flows from tool data
        tool_flows_set = set()
        for cwe_id, file_list in tool_data.items():
            for file_item in file_list:
                flows = file_item.get("flows", [])
                for flow_entry in flows:
                    tool_flows_set.add(hash_flow(flow_entry))

        # Compare the sets of hashed flows
        if labeled_flows_set == tool_flows_set:
            same_projects.append(dataset_name)
        else:
            diff_projects.append(dataset_name)

    # Print summary
    print("\n=== Comparison Results (Based on Flow Hashes) ===")
    if diff_projects:
        print("Projects that differ:")
        for proj in diff_projects:
            print(f"  - {proj}")
    else:
        print("No projects differ based on flow hashes.")

    if same_projects:
        print("\nProjects that are the same:")
        for proj in same_projects:
            print(f"  - {proj}")
    else:
        print("\nNo projects are the same.")

if __name__ == "__main__":
    main()
