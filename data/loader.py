from datasets import load_dataset
from sklearn.model_selection import train_test_split
from config import DATA_DIR, SEEDS, TOXICCHAT_CONFIG


def load_sst2(seed=SEEDS[0]):
    dataset = load_dataset("glue", "sst2", cache_dir=str(DATA_DIR))

    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]

    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        random_state=seed,
        stratify=train_labels,
    )

    return {
        "train": {"texts": list(train_texts), "labels": list(train_labels)},
        "val": {"texts": list(val_texts), "labels": list(val_labels)},
        "test": {"texts": list(test_texts), "labels": list(test_labels)},
    }


def load_sst2_subset(n_samples, seed=SEEDS[0]):
    """Caps the training set at n_samples. Used in the quantity vs quality experiment."""
    splits = load_sst2(seed=seed)

    texts = splits["train"]["texts"]
    labels = splits["train"]["labels"]

    if n_samples >= len(texts):
        return splits

    subset_texts, _, subset_labels, _ = train_test_split(
        texts,
        labels,
        train_size=n_samples,
        random_state=seed,
        stratify=labels,
    )

    splits["train"]["texts"] = subset_texts
    splits["train"]["labels"] = subset_labels

    return splits


def load_toxicchat(seed=SEEDS[0]):
    """
    Loads ToxicChat using the existing train/test splits from HuggingFace.
    Carves a validation set out of train using TOXICCHAT_CONFIG val_size.
    Uses user_input as text and toxicity as label.
    model_output is intentionally excluded -- it's multilingual and not a valid feature.
    """
    cfg = TOXICCHAT_CONFIG
    dataset = load_dataset(
        cfg["dataset_name"],
        cfg["dataset_version"],
        cache_dir=str(DATA_DIR),
    )

    train_texts = dataset["train"][cfg["text_column"]]
    train_labels = dataset["train"][cfg["label_column"]]
    test_texts = dataset["test"][cfg["text_column"]]
    test_labels = dataset["test"][cfg["label_column"]]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=cfg["val_size"],
        random_state=seed,
        stratify=train_labels,
    )

    return {
        "train": {"texts": list(train_texts), "labels": list(train_labels)},
        "val": {"texts": list(val_texts), "labels": list(val_labels)},
        "test": {"texts": list(test_texts), "labels": list(test_labels)},
    }


def load_toxicchat_quality_split(seed=SEEDS[0]):
    """
    Returns two training sets for the quantity vs quality experiment.
    Scenario A -- all data including weak labels (human_annotation=False).
    Scenario B -- human_annotation=True only, smaller but higher quality.
    Test set is the same for both so the comparison is fair.
    Val set is carved from the full train split.
    """
    cfg = TOXICCHAT_CONFIG
    dataset = load_dataset(
        cfg["dataset_name"],
        cfg["dataset_version"],
        cache_dir=str(DATA_DIR),
    )

    train_data = dataset["train"]
    test_texts = list(dataset["test"][cfg["text_column"]])
    test_labels = list(dataset["test"][cfg["label_column"]])

    all_texts = train_data[cfg["text_column"]]
    all_labels = train_data[cfg["label_column"]]
    all_human = train_data[cfg["human_annotation_column"]]

    human_texts = [t for t, h in zip(all_texts, all_human) if h]
    human_labels = [l for l, h in zip(all_labels, all_human) if h]

    all_train_texts, all_val_texts, all_train_labels, all_val_labels = train_test_split(
        all_texts,
        all_labels,
        test_size=cfg["val_size"],
        random_state=seed,
        stratify=all_labels,
    )

    human_train_texts, human_val_texts, human_train_labels, human_val_labels = (
        train_test_split(
            human_texts,
            human_labels,
            test_size=cfg["val_size"],
            random_state=seed,
            stratify=human_labels,
        )
    )

    return {
        "all_data": {
            "train": {"texts": list(all_train_texts), "labels": list(all_train_labels)},
            "val": {"texts": list(all_val_texts), "labels": list(all_val_labels)},
            "test": {"texts": test_texts, "labels": test_labels},
        },
        "human_only": {
            "train": {
                "texts": list(human_train_texts),
                "labels": list(human_train_labels),
            },
            "val": {"texts": list(human_val_texts), "labels": list(human_val_labels)},
            "test": {"texts": test_texts, "labels": test_labels},
        },
    }
