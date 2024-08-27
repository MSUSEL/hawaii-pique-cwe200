import json

def calculate_metrics(predictions, labeled_data):
    variable_tp = 0
    variable_fp = 0
    variable_fn = 0

    string_tp = 0
    string_fp = 0
    string_fn = 0

    comment_tp = 0
    comment_fp = 0
    comment_fn = 0

    predictions_dict = {p["fileName"]: p for p in predictions}
    labeled_data_dict = {l["fileName"]: l for l in labeled_data}

    for filename, labeled in labeled_data_dict.items():
        prediction = predictions_dict.get(filename)
        if prediction:
            for variable in prediction["variables"]:
                if variable["name"] in [v["name"] for v in labeled["variables"]]:
                    variable_tp += 1
                else:
                    variable_fp += 1

            for string in prediction["strings"]:
                if string["name"] in [s["name"] for s in labeled["strings"]]:
                    string_tp += 1
                else:
                    string_fp += 1

            for comment in prediction.get("comments", []):
                if "name" in comment:
                    for labeled_comment in labeled.get("comments", []):
                        if "name" in labeled_comment and labeled_comment["name"].lower() in comment["name"].lower():
                            comment_tp += 1
                            break
                    else:
                        comment_fp += 1

            for variable in labeled["variables"]:
                if variable["name"] not in [v["name"] for v in prediction["variables"]]:
                    variable_fn += 1

            for string in labeled["strings"]:
                if string["name"] not in [s["name"] for s in prediction["strings"]]:
                    string_fn += 1

            for comment in labeled.get("comments", []):
                if "name" in comment:
                    for predicted_comment in prediction.get("comments", []):
                        if "name" in predicted_comment and comment["name"].lower() in predicted_comment["name"].lower():
                            break
                    else:
                        comment_fn += 1
        else:
            # If the filename is not in the predictions, count all variables, strings, and comments as false negatives
            variable_fn += len(labeled["variables"])
            string_fn += len(labeled["strings"])
            comment_fn += len(labeled.get("comments", []))

    variable_precision = variable_tp / (variable_tp + variable_fp) if (variable_tp + variable_fp) != 0 else 0
    variable_recall = variable_tp / (variable_tp + variable_fn) if (variable_tp + variable_fn) != 0 else 0
    variable_f1 = 2 * (variable_precision * variable_recall) / (variable_precision + variable_recall) if (variable_precision + variable_recall) != 0 else 0

    string_precision = string_tp / (string_tp + string_fp) if (string_tp + string_fp) != 0 else 0
    string_recall = string_tp / (string_tp + string_fn) if (string_tp + string_fn) != 0 else 0
    string_f1 = 2 * (string_precision * string_recall) / (string_precision + string_recall) if (string_precision + string_recall) != 0 else 0

    comment_precision = comment_tp / (comment_tp + comment_fp) if (comment_tp + comment_fp) != 0 else 0
    comment_recall = comment_tp / (comment_tp + comment_fn) if (comment_tp + comment_fn) != 0 else 0
    comment_f1 = 2 * (comment_precision * comment_recall) / (comment_precision + comment_recall) if (comment_precision + comment_recall) != 0 else 0

    return {
        "variables": {
            "tp": variable_tp,
            "fp": variable_fp,
            "fn": variable_fn,
            "precision": variable_precision,
            "recall": variable_recall,
            "f1": variable_f1
        },
        "strings": {
            "tp": string_tp,
            "fp": string_fp,
            "fn": string_fn,
            "precision": string_precision,
            "recall": string_recall,
            "f1": string_f1
        },
        "comments": {
            "tp": comment_tp,
            "fp": comment_fp,
            "fn": comment_fn,
            "precision": comment_precision,
            "recall": comment_recall,
            "f1": comment_f1
        }
    }

def main():
    with open("backend/Files/CWEToyDataset/ChatGPT-data-final.json", "r") as f:
        predictions = json.load(f)

    with open("backend/Files/CWEToyDataset/Toy_dataset_data-labeled.json", "r") as f:
        labeled_data = json.load(f)

    metrics = calculate_metrics(predictions, labeled_data)

    for type, type_metrics in metrics.items():
        print(f"Metrics for {type}:")
        print(f"True Positives: {type_metrics['tp']}")
        print(f"False Positives: {type_metrics['fp']}")
        print(f"False Negatives: {type_metrics['fn']}")
        print(f"Precision: {type_metrics['precision']}")
        print(f"Recall: {type_metrics['recall']}")
        print(f"F1: {type_metrics['f1']}")
        print()

if __name__ == "__main__":
    main()