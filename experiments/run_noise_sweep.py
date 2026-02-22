import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from config import SEEDS, NOISE_LEVELS
from data.loader import load_sst2
from noise.injector import inject_label_noise
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds, save_results


def run_noise_sweep(model_name):
    print(f"\nrunning noise sweep — {model_name}")
    print(f"noise levels: {NOISE_LEVELS}")
    print(f"seeds: {SEEDS}\n")

    splits = load_sst2()
    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]
    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]

    results = {}

    for noise_level in tqdm(NOISE_LEVELS, desc="noise levels"):
        seed_results = []

        for seed in SEEDS:
            noisy_texts, noisy_labels = inject_label_noise(
                train_texts, train_labels, noise_level, seed=seed
            )

            model = train(model_name, noisy_texts, noisy_labels, seed=seed)
            metrics = evaluate(model, test_texts, test_labels)
            seed_results.append(metrics)

        aggregated = aggregate_across_seeds(seed_results)
        results[str(noise_level)] = aggregated

        print(
            f"  noise={noise_level:.0%} | "
            f"acc={aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f} | "
            f"f1={aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}"
        )

    save_results(results, f"noise_sweep_{model_name}")
    print(f"\ndone. results saved to results/noise_sweep_{model_name}.json")
    return results


if __name__ == "__main__":
    for model_name in ["logreg", "distilbert"]:
        run_noise_sweep(model_name)
