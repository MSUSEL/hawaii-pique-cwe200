import os
import json
import numpy as np
from tqdm import tqdm
import hashlib

def read_data_flow_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def process_data_flows(labeled_flows_file):
    processed_data_flows = []
    seen_flow_hashes = set()
    total_flows = 0
    duplicate_flows = 0
    kept_flows = 0

    data_flows = read_data_flow_file(labeled_flows_file)

    for cwe, results in data_flows.items():
        for result in results:
            result_index = result["resultIndex"]
            for flow in result["flows"]:
                total_flows += 1
                if not flow.get("flow") or "label" not in flow:
                    continue

                label = 1 if flow["label"] == "Yes" else 0 if flow["label"] == "No" else None
                if label is None:
                    continue

                # Build the original “data_flow_string”  
                original_data_flow_string = f"CWE = {cwe}, Flows = "
                for step in flow["flow"]:
                    step_string = f"step={step.get('step', '')}, "
                    step_string += f"variableName={step.get('variableName', '')}, "
                    step_string += f"type={step.get('type', '')}, "
                    step_string += f"code={step.get('code', '')}, "
                    step_string = {step_string}
                    # print(step_string)
                    original_data_flow_string += str(step_string)

                # ─── Build a normalized “signature” for dedup’ing: only (variableName, type) ───
                step_signature = []
                for step in flow["flow"]:
                    varname = step.get("variableName", "").strip()
                    vartype = step.get("type", "").strip()
                    step_signature.append(f"{varname}::{vartype}")

                signature_str = f"CWE={cwe}|" + "→".join(step_signature)
                flow_hash = hashlib.sha256(signature_str.encode("utf-8")).hexdigest()

                if flow_hash in seen_flow_hashes:
                    duplicate_flows += 1
                    continue

                seen_flow_hashes.add(flow_hash)
                kept_flows += 1

                # ─── Store the original_data_flow_string (not signature_str) ───
                processed_data_flows.append([
                    result_index,
                    flow["codeFlowIndex"],
                    original_data_flow_string,
                    label
                ])

    print(f"Total flows processed: {total_flows}")
    print(f"Duplicate flows excluded: {duplicate_flows}")
    print(f"Flows kept for training: {kept_flows}")

    with open("processed_data_flows.json", "w", encoding="utf-8") as json_file:
        json.dump(processed_data_flows, json_file, indent=4)

    return np.array(processed_data_flows)


if __name__ == "__main__":
    project_name = "infinispan-15.2.1.Final.json"
    labeled_flows_dir = os.path.join("testing", "Labeling", "Flow Verification", "FlowData")
    project = os.path.join(labeled_flows_dir, project_name)

    print("Processing data flows…")
    processed_data_flows = process_data_flows(project)
