# NoiseCliff

> Detect when noisy labels will break a safety classifier **before retraining on bad data.**

Machine learning systems often retrain classifiers as new labeled data arrives.
If those labels contain too much noise, the retrained model silently degrades.

**NoiseCliff** is a **pre-training quality gate** that checks a label batch and estimates whether the dataset is too noisy to safely retrain on.

Instead of discovering label problems **after training fails**, the gate detects them **before the training job starts**.

---

# What This Project Provides

NoiseCliff includes two things:

### 1️⃣ A practical tool

A CLI that checks annotation batches and blocks retraining if the data appears too noisy.

```bash
python gate/check.py --batch new_labels.csv --baseline toxicchat
```

Example output:

```
NoiseCliff Quality Gate

Batch: week_47_labels.csv
Samples: 2,847
Toxic rate: 9.4%

Estimated noise rate: 24.3%
Estimated PR-AUC if trained: 0.481 (baseline 0.628)

Collapse risk: DANGER
Safe to retrain: NO
```

The gate can run automatically in CI before any training job begins.

---

### 2️⃣ Analysis of when label noise becomes dangerous

Experiments show classifier performance degrades gradually until roughly **~20% label noise**, after which performance collapses rapidly.

Below the threshold:

```
~0.02 PR-AUC loss per 5% noise interval
```

Beyond it:

```
~0.09 PR-AUC loss per interval
```

This creates a **noise cliff** where model reliability drops sharply.

Standard metrics like **F1 macro can hide this failure**, especially on imbalanced datasets.

---

# Why This Matters

In production ML systems:

- annotation pipelines drift
- weak labels get mixed with human labels
- guidelines change over time
- reviewers disagree

All of these introduce **label noise**.

Without monitoring, models can be retrained on corrupted data and degrade silently.

NoiseCliff acts as a **data quality guardrail for ML pipelines**.

---

# Quickstart

Clone the repository:

```bash
git clone https://github.com/DevelopedBy-Siva/noise-cliff.git
cd noise-cliff
pip install -r requirements.txt
```

Initialize the baseline:

```bash
python gate/bootstrap_toxicchat_baseline.py
```

Check a label batch:

```bash
python gate/check.py --batch labels.csv
```

CSV format:

```
text,label
"This prompt is toxic",1
"This prompt is safe",0
```

---

# Quality Gate CLI

Basic usage:

```bash
python gate/check.py --batch labels.csv --baseline toxicchat
```

Useful options:

```
--save       Save report as JSON
--json       Output machine-readable report
--ci         Exit with error code if unsafe
--explain    Show suspicious examples
```

Example:

```bash
python gate/check.py \
  --batch week_47_labels.csv \
  --baseline toxicchat \
  --save
```

---

# CI Integration

NoiseCliff can block retraining jobs automatically.

Example GitHub Actions step:

```yaml
- name: NoiseCliff quality gate
  run: |
    python gate/check.py \
      --batch data/new_labels.csv \
      --baseline toxicchat \
      --ci
```

Exit codes:

| Code | Meaning                |
| ---- | ---------------------- |
| 0    | Safe batch             |
| 1    | Unsafe batch (CI fail) |
| 2    | Invalid input          |

---

# How Noise Estimation Works

The gate estimates noise **without external ground truth**.

Steps:

1. Train a lightweight **TF-IDF + Logistic Regression probe**
2. Run **3-fold cross-validation**
3. Measure uncertainty signals from predictions

Signals used:

| Signal                   | Meaning                                       |
| ------------------------ | --------------------------------------------- |
| Prediction entropy       | noisy labels produce uncertain predictions    |
| Margin                   | clean labels produce confident predictions    |
| Near-threshold mass      | noise concentrates predictions near boundary  |
| Label distribution drift | dataset shifts may indicate annotation issues |

These signals are compared against calibrated baseline curves to estimate the noise rate.

---

# Calibration

The gate ships with a baseline calibrated for **ToxicChat**.

If your dataset differs significantly, calibrate a new baseline.

```bash
python gate/calibrate.py --csv my_data.csv --name my_dataset
```

Then use it:

```bash
python gate/check.py --batch new_labels.csv --baseline my_dataset
```

Calibration:

1. Injects controlled noise levels
2. measures PR-AUC degradation
3. finds the dataset’s tipping point
4. stores calibration curves

---

# Gate Accuracy

Validated across **9 injected noise levels × 3 seeds** on ToxicChat.

| Injected | Estimated | Error  |
| -------- | --------- | ------ |
| 0%       | 0.000     | 0.000  |
| 5%       | 0.050     | ~0     |
| 10%      | 0.104     | +0.004 |
| 20%      | 0.195     | -0.005 |
| 30%      | 0.311     | +0.011 |
| 40%      | 0.357     | -0.043 |

Mean absolute error: **1.1 percentage points**

Operational zones:

| Zone     | Noise  |
| -------- | ------ |
| CLEAN    | <10%   |
| WATCH    | 10–18% |
| DANGER   | 18–30% |
| CRITICAL | >30%   |

---

# Experiments

The project includes experiments evaluating classifier robustness to label noise.

Two datasets were used:

### SST-2

Used to validate the experimental setup.

### ToxicChat

Real-world dataset of user prompts from the Vicuna LLM demo.

- 10,165 prompts
- 7.33% toxic
- 2.01% jailbreak prompts
- labeled by multiple annotators

---

# Noise Experiments (ToxicChat)

Primary metric: **PR-AUC** (appropriate for imbalanced classification).

### Degradation curves

| Noise | LogReg PR-AUC | DeBERTa PR-AUC |
| ----- | ------------- | -------------- |
| 0%    | 0.628         | 0.845          |
| 10%   | 0.605         | 0.804          |
| 20%   | 0.532         | 0.750          |
| 40%   | 0.185         | 0.243          |

Performance degrades slowly until **~20% noise**, then collapses rapidly.

---

# Metric Pitfall

On imbalanced datasets, **F1 macro can increase even while the classifier gets worse**.

Example from experiments:

```
F1_macro rises from 0.606 → 0.656
while PR-AUC drops from 0.627 → 0.382
```

Monitoring only F1 hides model failure.

PR-AUC reflects ranking quality across thresholds and is more reliable for toxicity classification.

---

# Cleaning Strategy Results

Cleaning approaches evaluated:

| Strategy             | Result                       |
| -------------------- | ---------------------------- |
| Loss filtering       | consistent recovery          |
| Heuristic filtering  | small improvements           |
| Confidence filtering | dangerous on imbalanced data |

At **40% noise**, no strategy fully recovers the classifier.

Preventing noisy retraining batches is safer than post-hoc fixes.

---

# Quantity vs Quality

Experiments show weak labels can sometimes help.

On ToxicChat:

```
All data (weak labels included) performs better
than smaller human-only datasets.
```

Reason: weak labels from moderation models still contain signal.

Noise characteristics matter as much as noise amount.

---

# Business Interpretation

Assume:

```
100k daily interactions
7.3% toxic prompts
```

That means roughly **7,300 toxic prompts daily**.

Training with **20% label noise** reduces PR-AUC:

```
0.628 → 0.532
```

At **40% noise**, ranking approaches random.

Both missed toxic prompts and false blocks increase.

The **~20% noise threshold** is where operational risk begins to accelerate.

---

# Project Structure

```
noise-cliff/

gate/              # quality gate CLI
models/            # classifiers
experiments/       # experiment scripts
noise/             # noise injection utilities
training/          # training pipeline
evaluation/        # evaluation metrics
cleaning/          # label cleaning strategies
results/           # experiment outputs
```

---

# Running Experiments

```
python experiments/find_tipping_point.py
python experiments/run_noise_sweep.py
python experiments/run_cleaning.py
python experiments/run_quantity_vs_quality.py
```

---

# Tech Stack

Python 3.10,
PyTorch,
HuggingFace Transformers,
scikit-learn,
pandas,
matplotlib
