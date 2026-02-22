import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from config import RESULTS_DIR


def evaluate(model, texts, labels):
    """
    Runs prediction and returns accuracy and F1 for a single run.
    """
    predictions = model.predict(texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }


def aggregate_across_seeds(results_per_seed):
    """
    Takes a list of dicts like [{"accuracy": 0.91, "f1": 0.90}, ...]
    and returns mean and std for each metric.
    This is what goes into the paper-style result tables.
    """
    metrics = list(results_per_seed[0].keys())
    aggregated = {}

    for metric in metrics:
        values = [r[metric] for r in results_per_seed]
        aggregated[f"{metric}_mean"] = round(float(np.mean(values)), 4)
        aggregated[f"{metric}_std"] = round(float(np.std(values)), 4)

    return aggregated


def save_results(results, filename):
    """
    Dumps results as JSON into the results directory.
    filename should be something descriptive like 'noise_sweep_logreg'.
    """
    path = RESULTS_DIR / f"{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved → {path}")


def load_results(filename):
    path = RESULTS_DIR / f"{filename}.json"
    with open(path, "r") as f:
        return json.load(f)
