import os
import json
import hashlib
from glob import glob

###############################################################################
# Helper functions
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
    uri          = step.get('uri', '')
    step_type    = step.get('type', '')
    code         = step.get('code', '')

    return f"{step_num}|{variableName}|{uri}|{step_type}|{code}"

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

def compute_metrics(tp, fp, fn, tn):
    """
    Returns (precision, recall, f1) for the 'Yes' (positive) class.

    Definitions:
      precision = TP / (TP + FP)  if TP+FP > 0 else 0.0
      recall    = TP / (TP + FN)  if TP+FN > 0 else 0.0
      f1        = 2 * precision * recall / (precision + recall),
                  or 0.0 if both precision and recall are 0.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def compute_accuracy(tp, fp, fn, tn):
    """
    Accuracy = (TP + TN) / (TP + FP + FN + TN).
    Rewards both correct positives (TP) and correct negatives (TN).
    """
    total = tp + fp + fn + tn
    if total == 0:
        return 0.0
    return (tp + tn) / total

def compute_specificity(tp, fp, fn, tn):
    """
    Specificity (True Negative Rate) = TN / (TN + FP).
    Measures how well the tool predicts 'No' flows correctly.
    """
    denom = tn + fp
    if denom == 0:
        return 0.0
    return tn / denom

def compute_balanced_accuracy(tp, fp, fn, tn):
    """
    Balanced Accuracy = 0.5 * (TPR + TNR),
    where TPR = TP / (TP + FN), TNR = TN / (TN + FP).

    Balanced accuracy equally weighs performance on positives
    and negatives, making it useful for imbalanced datasets.
    """
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # TPR
    spec = tn / (tn + fp) if (tn + fp) else 0.0    # TNR
    return 0.5 * (recall + spec)

###############################################################################
# Main script
###############################################################################

def main():
    labeled_dir = "testing/Labeling/FlowData"
    backend_dir = "backend/Files"

    # Global confusion-matrix totals
    g_wv_TP = g_wv_FP = g_wv_FN = g_wv_TN = 0
    g_nv_TP = g_nv_FP = g_nv_FN = g_nv_TN = 0

    # Tally how many projects improved vs. stayed the same vs. got worse (by F1)
    improved_count = 0
    same_count = 0
    worse_count = 0

    # Gather final listing of which projects we actually compared
    compared_projects = []

    # Iterate over labeled JSON files
    labeled_json_files = glob(os.path.join(labeled_dir, "*.json"))

    for labeled_path in labeled_json_files:
        filename = os.path.basename(labeled_path)
        dataset_name = filename.replace(".json", "")  # e.g. "CWEToyDataset"
        tool_json_path = os.path.join(backend_dir, dataset_name, "flowMapsByCWE.json")

        if not os.path.isfile(tool_json_path):
            print(f"!! No tool output found for {dataset_name} -> skipping comparison.")
            continue

        print(f"** Found tool output for {dataset_name} -> comparing... **")
        compared_projects.append(dataset_name)

        # 1) Load Labeled JSON (Ground Truth)
        with open(labeled_path, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)  # structure like { "201": [ ... ], "536": [ ... ] }

        # Build dict: flow_hash -> ground_truth_label
        labeled_flows = {}
        for cwe_id, file_list in labeled_data.items():
            for file_item in file_list:
                flows = file_item.get("flows", [])
                for flow_entry in flows:
                    flow_hash_val = hash_flow(flow_entry)
                    label = flow_entry.get("label", "No")  # default "No"
                    labeled_flows[flow_hash_val] = label

        # 2) Load Tool’s JSON (Predictions)
        with open(tool_json_path, "r", encoding="utf-8") as f:
            tool_data = json.load(f)

        # Dict: flow_hash -> tool_label (Yes/No)
        # or "Yes" by default if no explicit label.
        tool_flows = {}
        for cwe_id, file_list in tool_data.items():
            for file_item in file_list:
                flows = file_item.get("flows", [])
                for flow_entry in flows:
                    tool_label = flow_entry.get("label", "Yes")
                    flow_hash_val = hash_flow(flow_entry)
                    tool_flows[flow_hash_val] = tool_label

        # Local confusion matrix for this project
        wv_TP = wv_FP = wv_FN = wv_TN = 0
        nv_TP = nv_FP = nv_FN = nv_TN = 0

        # 3) Compare flows
        for flow_hash_val, true_label in labeled_flows.items():
            gt_yes = (true_label.lower() == "yes")

            # WITH VERIFICATION: use the tool's actual label if the flow is found
            predicted_label_wv = tool_flows[flow_hash_val] if flow_hash_val in tool_flows else "No"
            pred_yes_wv = (predicted_label_wv.lower() == "yes")

            if gt_yes and pred_yes_wv:
                wv_TP += 1
            elif (not gt_yes) and pred_yes_wv:
                wv_FP += 1
            elif gt_yes and (not pred_yes_wv):
                wv_FN += 1
            else:
                wv_TN += 1

            # WITHOUT VERIFICATION: presence in tool_flows => "Yes", otherwise "No"
            pred_yes_nv = (flow_hash_val in tool_flows)

            if gt_yes and pred_yes_nv:
                nv_TP += 1
            elif (not gt_yes) and pred_yes_nv:
                nv_FP += 1
            elif gt_yes and (not pred_yes_nv):
                nv_FN += 1
            else:
                nv_TN += 1

        # 4) Compute local (per-project) metrics
        loc_wv_precision, loc_wv_recall, loc_wv_f1 = compute_metrics(wv_TP, wv_FP, wv_FN, wv_TN)
        loc_nv_precision, loc_nv_recall, loc_nv_f1 = compute_metrics(nv_TP, nv_FP, nv_FN, nv_TN)

        loc_wv_accuracy = compute_accuracy(wv_TP, wv_FP, wv_FN, wv_TN)
        loc_nv_accuracy = compute_accuracy(nv_TP, nv_FP, nv_FN, nv_TN)

        loc_wv_specificity = compute_specificity(wv_TP, wv_FP, wv_FN, wv_TN)
        loc_nv_specificity = compute_specificity(nv_TP, nv_FP, nv_FN, nv_TN)

        loc_wv_bal_acc = compute_balanced_accuracy(wv_TP, wv_FP, wv_FN, wv_TN)
        loc_nv_bal_acc = compute_balanced_accuracy(nv_TP, nv_FP, nv_FN, nv_TN)

        # Delta metrics (With verification minus Without verification)
        delta_f1     = loc_wv_f1 - loc_nv_f1
        delta_acc    = loc_wv_accuracy - loc_nv_accuracy
        delta_spec   = loc_wv_specificity - loc_nv_specificity
        delta_balacc = loc_wv_bal_acc - loc_nv_bal_acc

        # Tally improvement vs. same vs. worse for F1
        if delta_f1 > 0:
            improved_count += 1
        elif delta_f1 < 0:
            worse_count += 1
        else:
            same_count += 1

        # 5) Print local project metrics
        print(f"\n--- Metrics for {dataset_name} ---")

        # -- With Verification
        print("With Verification (tool's actual label):")
        print(f"  TP={wv_TP}, FP={wv_FP}, FN={wv_FN}, TN={wv_TN}")
        print(f"  Precision={loc_wv_precision:.3f}, Recall={loc_wv_recall:.3f}, F1={loc_wv_f1:.3f}, "
              f"Accuracy={loc_wv_accuracy:.3f}, Specificity={loc_wv_specificity:.3f}, BalancedAcc={loc_wv_bal_acc:.3f}")

        # -- Without Verification
        print("Without Verification (presence => Yes):")
        print(f"  TP={nv_TP}, FP={nv_FP}, FN={nv_FN}, TN={nv_TN}")
        print(f"  Precision={loc_nv_precision:.3f}, Recall={loc_nv_recall:.3f}, F1={loc_nv_f1:.3f}, "
              f"Accuracy={loc_nv_accuracy:.3f}, Specificity={loc_nv_specificity:.3f}, BalancedAcc={loc_nv_bal_acc:.3f}")

        # -- Delta
        print(f"Δ F1={delta_f1:.3f}, Δ Acc={delta_acc:.3f}, Δ Spec={delta_spec:.3f}, Δ BalAcc={delta_balacc:.3f}\n")

        # 6) Update GLOBAL confusion matrix
        g_wv_TP += wv_TP
        g_wv_FP += wv_FP
        g_wv_FN += wv_FN
        g_wv_TN += wv_TN

        g_nv_TP += nv_TP
        g_nv_FP += nv_FP
        g_nv_FN += nv_FN
        g_nv_TN += nv_TN

    # -------------------------------------------------------------------------
    # Compute final (global) metrics across all projects
    # -------------------------------------------------------------------------
    g_wv_precision, g_wv_recall, g_wv_f1 = compute_metrics(g_wv_TP, g_wv_FP, g_wv_FN, g_wv_TN)
    g_nv_precision, g_nv_recall, g_nv_f1 = compute_metrics(g_nv_TP, g_nv_FP, g_nv_FN, g_nv_TN)

    g_wv_accuracy    = compute_accuracy(g_wv_TP, g_wv_FP, g_wv_FN, g_wv_TN)
    g_nv_accuracy    = compute_accuracy(g_nv_TP, g_nv_FP, g_nv_FN, g_nv_TN)
    g_wv_specificity = compute_specificity(g_wv_TP, g_wv_FP, g_wv_FN, g_wv_TN)
    g_nv_specificity = compute_specificity(g_nv_TP, g_nv_FP, g_nv_FN, g_nv_TN)
    g_wv_bal_acc     = compute_balanced_accuracy(g_wv_TP, g_wv_FP, g_wv_FN, g_wv_TN)
    g_nv_bal_acc     = compute_balanced_accuracy(g_nv_TP, g_nv_FP, g_nv_FN, g_nv_TN)

    # Global deltas
    g_delta_f1     = g_wv_f1 - g_nv_f1
    g_delta_acc    = g_wv_accuracy - g_nv_accuracy
    g_delta_spec   = g_wv_specificity - g_nv_specificity
    g_delta_balacc = g_wv_bal_acc - g_nv_bal_acc

    print("\n=== Final Metrics for All Compared Projects ===")
    if compared_projects:
        print(f"Compared Projects: {', '.join(compared_projects)}")
    else:
        print("No projects were compared because no tool outputs matched the labeled JSONs.")

    # Global metrics: With Verification
    print("\n--- Global With Verification (tool's actual label) ---")
    print(f"TP={g_wv_TP}, FP={g_wv_FP}, FN={g_wv_FN}, TN={g_wv_TN}")
    print(f"Precision={g_wv_precision:.3f}, Recall={g_wv_recall:.3f}, F1={g_wv_f1:.3f}, "
          f"Accuracy={g_wv_accuracy:.3f}, Specificity={g_wv_specificity:.3f}, BalancedAcc={g_wv_bal_acc:.3f}")

    # Global metrics: Without Verification
    print("\n--- Global Without Verification (presence => Yes) ---")
    print(f"TP={g_nv_TP}, FP={g_nv_FP}, FN={g_nv_FN}, TN={g_nv_TN}")
    print(f"Precision={g_nv_precision:.3f}, Recall={g_nv_recall:.3f}, F1={g_nv_f1:.3f}, "
          f"Accuracy={g_nv_accuracy:.3f}, Specificity={g_nv_specificity:.3f}, BalancedAcc={g_nv_bal_acc:.3f}")

    print(f"\nGlobal Δ F1={g_delta_f1:.3f}, Δ Acc={g_delta_acc:.3f}, "
          f"Δ Spec={g_delta_spec:.3f}, Δ BalAcc={g_delta_balacc:.3f}")

    # Print how many improved vs. same vs. worse across all projects (by F1)
    print("\n=== Verification Impact Across Projects (By F1) ===")
    print(f"  Improved : {improved_count}")
    print(f"  Same     : {same_count}")
    print(f"  Worse    : {worse_count}")

if __name__ == "__main__":
    main()
