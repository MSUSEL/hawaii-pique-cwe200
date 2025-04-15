import os
import json
from glob import glob

def main():
    labeled_dir = os.path.join("testing", "Labeling", "FlowDataOld")
    backend_dir = os.path.join("backend", "Files")

    same_projects = []
    diff_projects = []

    # Collect all labeled JSON files in labeled_dir
    labeled_json_files = glob(os.path.join(labeled_dir, "*.json"))

    for labeled_path in labeled_json_files:
        filename = os.path.basename(labeled_path)
        dataset_name = filename.replace(".json", "")  # e.g. "CWEToyDataset"

        # Corresponding tool output
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

        # Compare them directly as Python objects
        if labeled_data == tool_data:
            same_projects.append(dataset_name)
        else:
            diff_projects.append(dataset_name)

    # Print summary
    print("\n=== Comparison Results ===")
    if diff_projects:
        print("Projects that differ:")
        for proj in diff_projects:
            print(f"  - {proj}")
    else:
        print("No projects differ.")

    if same_projects:
        print("\nProjects that are the same:")
        for proj in same_projects:
            print(f"  - {proj}")
    else:
        print("\nNo projects are the same.")

if __name__ == "__main__":
    main()
