import json
from collections import defaultdict

'''
Used to compare the predicted sinks (Either from ChatGPT or Bert) to the labeled sinks.
'''

def calculate_metrics(predictions, labeled_data):
    sink_tp = 0
    sink_fp = 0
    sink_fn = 0

    predictions_dict = {p["fileName"]: p for p in predictions}
    labeled_data_dict = {l["fileName"]: l for l in labeled_data}

    for filename, labeled in labeled_data_dict.items():
        prediction = predictions_dict.get(filename)
        if prediction:
            for sink in prediction["sinks"]:
                if sink["name"] in [s["name"] for s in labeled["sinks"]]:
                    sink_tp += 1
                else:
                    sink_fp += 1

            for sink in labeled["sinks"]:
                if sink["name"] not in [s["name"] for s in prediction["sinks"]]:
                    sink_fn += 1
        else:
            # If the filename is not in the predictions, count all sinks as false negatives
            sink_fn += len(labeled["sinks"])

    sink_precision = sink_tp / (sink_tp + sink_fp) if (sink_tp + sink_fp) != 0 else 0
    sink_recall = sink_tp / (sink_tp + sink_fn) if (sink_tp + sink_fn) != 0 else 0
    sink_f1 = 2 * (sink_precision * sink_recall) / (sink_precision + sink_recall) if (sink_precision + sink_recall) != 0 else 0

    return {
        "sinks": {
            "tp": sink_tp,
            "fp": sink_fp,
            "fn": sink_fn,
            "precision": sink_precision,
            "recall": sink_recall,
            "f1": sink_f1
        }
    }

def main():
    with open("backend/Files/CWEToyDataset/ChatGPT-data-final.json", "r", encoding='UTF-8') as f:
        predictions = json.load(f)

    with open("testing/ChatGPT/Sinks/ToySinkLabel.json", "r") as f:
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
