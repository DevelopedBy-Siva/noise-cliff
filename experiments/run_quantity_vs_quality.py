import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from config import SEEDS, QUANTITY_VS_QUALITY
from data.loader import load_sst2_subset, load_toxicchat_quality_split
from noise.injector import inject_label_noise
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds, save_results


def run_quantity_vs_quality_sst2(model_name):
    print(f"\nrunning quantity vs quality -- {model_name} on sst2")
    print(
        f"  scenario A: {QUANTITY_VS_QUALITY['noisy_size']} samples @ {QUANTITY_VS_QUALITY['noisy_noise_level']:.0%} label noise"
    )
    print(f"  scenario B: {QUANTITY_VS_QUALITY['clean_size']} samples @ 0% label noise")
    print(f"  seeds: {SEEDS}\n")

    noisy_results = []
    clean_results = []

    for seed in SEEDS:
        noisy_splits = load_sst2_subset(QUANTITY_VS_QUALITY["noisy_size"], seed=seed)
        test_texts = noisy_splits["test"]["texts"]
        test_labels = noisy_splits["test"]["labels"]

        noisy_texts, noisy_labels = inject_label_noise(
            noisy_splits["train"]["texts"],
            noisy_splits["train"]["labels"],
            QUANTITY_VS_QUALITY["noisy_noise_level"],
            seed=seed,
        )

        noisy_model = train(model_name, noisy_texts, noisy_labels, seed=seed)
        noisy_metrics = evaluate(noisy_model, test_texts, test_labels)
        noisy_results.append(noisy_metrics)

        clean_splits = load_sst2_subset(QUANTITY_VS_QUALITY["clean_size"], seed=seed)
        clean_model = train(
            model_name,
            clean_splits["train"]["texts"],
            clean_splits["train"]["labels"],
            seed=seed,
        )
        clean_metrics = evaluate(clean_model, test_texts, test_labels)
        clean_results.append(clean_metrics)

        print(
            f"  seed={seed} | "
            f"noisy(50k,30%) f1={noisy_metrics['f1_macro']:.4f} | "
            f"clean(20k) f1={clean_metrics['f1_macro']:.4f}"
        )

    noisy_agg = aggregate_across_seeds(noisy_results)
    clean_agg = aggregate_across_seeds(clean_results)

    delta_f1 = round(clean_agg["f1_macro_mean"] - noisy_agg["f1_macro_mean"], 4)
    delta_acc = round(clean_agg["accuracy_mean"] - noisy_agg["accuracy_mean"], 4)

    results = {
        "noisy_50k_30pct": noisy_agg,
        "clean_20k": clean_agg,
        "delta": {
            "f1_macro": delta_f1,
            "accuracy": delta_acc,
            "clean_wins": delta_f1 > 0,
        },
    }

    _print_summary_sst2(noisy_agg, clean_agg, delta_f1, delta_acc)
    save_results(results, f"quantity_vs_quality_{model_name}_sst2")
    print(
        f"\ndone. results saved to results/quantity_vs_quality_{model_name}_sst2.json"
    )
    return results


def run_quantity_vs_quality_toxicchat(model_name):
    print(f"\nrunning quantity vs quality -- {model_name} on toxicchat")
    print("  scenario A: all data including weak labels (human_annotation=False)")
    print("  scenario B: human_annotation=True only -- smaller but higher quality")
    print(f"  seeds: {SEEDS}\n")

    all_data_results = []
    human_only_results = []

    for seed in SEEDS:
        splits = load_toxicchat_quality_split(seed=seed)

        test_texts = splits["all_data"]["test"]["texts"]
        test_labels = splits["all_data"]["test"]["labels"]

        all_model = train(
            model_name,
            splits["all_data"]["train"]["texts"],
            splits["all_data"]["train"]["labels"],
            seed=seed,
        )
        all_metrics = evaluate(all_model, test_texts, test_labels)
        all_data_results.append(all_metrics)

        human_model = train(
            model_name,
            splits["human_only"]["train"]["texts"],
            splits["human_only"]["train"]["labels"],
            seed=seed,
        )
        human_metrics = evaluate(human_model, test_texts, test_labels)
        human_only_results.append(human_metrics)

        print(
            f"  seed={seed} | "
            f"all_data prauc={all_metrics['prauc']:.4f} | "
            f"human_only prauc={human_metrics['prauc']:.4f}"
        )

    all_agg = aggregate_across_seeds(all_data_results)
    human_agg = aggregate_across_seeds(human_only_results)

    delta_prauc = round(human_agg["prauc_mean"] - all_agg["prauc_mean"], 4)
    delta_f1 = round(human_agg["f1_macro_mean"] - all_agg["f1_macro_mean"], 4)

    results = {
        "all_data": all_agg,
        "human_only": human_agg,
        "delta": {
            "prauc": delta_prauc,
            "f1_macro": delta_f1,
            "human_only_wins": delta_prauc > 0,
        },
    }

    _print_summary_toxicchat(all_agg, human_agg, delta_prauc, delta_f1)
    save_results(results, f"quantity_vs_quality_{model_name}_toxicchat")
    print(
        f"\ndone. results saved to results/quantity_vs_quality_{model_name}_toxicchat.json"
    )
    return results


def _print_summary_sst2(noisy_agg, clean_agg, delta_f1, delta_acc):
    print(f"\n{'—' * 55}")
    print(
        f"  noisy 50k  | acc={noisy_agg['accuracy_mean']:.4f} ± {noisy_agg['accuracy_std']:.4f} | "
        f"f1={noisy_agg['f1_macro_mean']:.4f} ± {noisy_agg['f1_macro_std']:.4f}"
    )
    print(
        f"  clean 20k  | acc={clean_agg['accuracy_mean']:.4f} ± {clean_agg['accuracy_std']:.4f} | "
        f"f1={clean_agg['f1_macro_mean']:.4f} ± {clean_agg['f1_macro_std']:.4f}"
    )
    print(
        f"  delta      | acc={delta_acc:+.4f} | f1={delta_f1:+.4f} | clean wins: {delta_f1 > 0}"
    )
    print(f"{'—' * 55}")


def _print_summary_toxicchat(all_agg, human_agg, delta_prauc, delta_f1):
    print(f"\n{'—' * 55}")
    print(
        f"  all data   | prauc={all_agg['prauc_mean']:.4f} ± {all_agg['prauc_std']:.4f} | "
        f"f1={all_agg['f1_macro_mean']:.4f} ± {all_agg['f1_macro_std']:.4f}"
    )
    print(
        f"  human only | prauc={human_agg['prauc_mean']:.4f} ± {human_agg['prauc_std']:.4f} | "
        f"f1={human_agg['f1_macro_mean']:.4f} ± {human_agg['f1_macro_std']:.4f}"
    )
    print(
        f"  delta      | prauc={delta_prauc:+.4f} | f1={delta_f1:+.4f} | human only wins: {delta_prauc > 0}"
    )
    print(f"{'—' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["sst2", "toxicchat"],
        default="toxicchat",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "deberta"],
        default=["logreg", "deberta"],
    )
    args = parser.parse_args()

    for model_name in args.models:
        if args.dataset == "sst2":
            run_quantity_vs_quality_sst2(model_name)
        else:
            run_quantity_vs_quality_toxicchat(model_name)
