# data-quality-bench

Does cleaner data beat more data? Benchmarking label noise, model degradation, and data cleaning strategies on NLP sentiment classification.

---

## What this is

Most ML projects obsess over model architecture. This one obsesses over the data.

The idea is simple: take a clean dataset, deliberately corrupt it in controlled ways, measure how fast models break, then try to fix it and see how much you can recover. Do that across two very different models and three seeds, and you start to get answers that actually hold up.

The headline experiment is blunt: 20,000 clean training samples vs 50,000 samples with 30% of the labels flipped. One question -- does cleaner data beat more data?

---

## Results

### Degradation -- how fast do models break?

| Model      | Noise 0%   | Noise 10%  | Noise 20%  | Noise 40%  |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| LogReg     | `F1 ± std` | `F1 ± std` | `F1 ± std` | `F1 ± std` |
| DistilBERT | `F1 ± std` | `F1 ± std` | `F1 ± std` | `F1 ± std` |

![degradation curves](results/plot_degradation_curves.png)

### Cleaning recovery -- how much can you get back?

| Strategy          | Noise 10%  | Noise 20%  | Noise 40%  |
| ----------------- | ---------- | ---------- | ---------- |
| Noisy baseline    | `F1 ± std` | `F1 ± std` | `F1 ± std` |
| Confidence filter | `F1 ± std` | `F1 ± std` | `F1 ± std` |
| Loss filter       | `F1 ± std` | `F1 ± std` | `F1 ± std` |
| Heuristic filter  | `F1 ± std` | `F1 ± std` | `F1 ± std` |

![cleaning recovery logreg](results/plot_cleaning_recovery_logreg.png)
![cleaning recovery distilbert](results/plot_cleaning_recovery_distilbert.png)

### The headline -- quantity vs quality

| Scenario               | F1 (mean ± std) |
| ---------------------- | --------------- |
| 50k samples, 30% noise | `F1 ± std`      |
| 20k samples, clean     | `F1 ± std`      |
| Delta                  | `+/- X.XXXX`    |

> Clean wins: `True / False`

![quantity vs quality](results/plot_quantity_vs_quality_distilbert.png)

---

## How it works

**Dataset:** SST-2 from HuggingFace. Binary sentiment classification, widely understood, fast to iterate on. The dataset choice is boring on purpose -- the experiment is the interesting part.

**Noise types:**

- Label noise: randomly flips labels at 10%, 20%, and 40% of training samples
- Text noise: word deletions, character swaps, token duplication
- Structural noise: minority class duplication and junk short samples

**Models:**

- Logistic Regression with TF-IDF (classical baseline)
- DistilBERT fine-tuned (transformer)

Both models expose the same interface. Everything else -- optimizer, epochs, batch size, eval split -- is held constant across every experiment.

**Cleaning strategies:**

- Confidence filter: drops samples the model is uncertain about
- Loss filter: drops high-loss samples, which correlate strongly with mislabeled examples
- Heuristic filter: removes duplicates and samples too short to carry signal

**Reproducibility:** all runs use fixed seeds `[42, 43, 44]`. Results are reported as mean ± std across the three seeds. Every result file is saved as JSON in `results/`.

---

## Project structure

```
data-quality-bench/
├── config.py                         -- all hyperparameters and paths live here
├── data/
│   └── loader.py                     -- loads SST-2, carves out a test split
├── noise/
│   └── injector.py                   -- label, text, and structural noise
├── models/
│   ├── logreg.py                     -- TF-IDF + logistic regression
│   └── distilbert.py                 -- DistilBERT fine-tuning wrapper
├── training/
│   └── trainer.py                    -- seed control, model init
├── evaluation/
│   └── evaluator.py                  -- metrics, seed aggregation, result saving
├── cleaning/
│   └── strategies.py                 -- confidence, loss, and heuristic filters
├── experiments/
│   ├── run_noise_sweep.py            -- experiment 1: degradation curves
│   ├── run_cleaning.py               -- experiment 2: recovery study
│   └── run_quantity_vs_quality.py    -- experiment 3: the headline
├── notebooks/
│   └── plots.ipynb                   -- all visualizations
└── results/                          -- JSON results and PNG plots land here
```

---

## Running it

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the experiments in order**

```bash
python experiments/run_noise_sweep.py
python experiments/run_cleaning.py
python experiments/run_quantity_vs_quality.py
```

Each experiment prints a live summary as it runs and saves results to `results/`. Then open `notebooks/plots.ipynb` to generate all the charts.

If you only have time for one, run `run_quantity_vs_quality.py` -- it is the most self-contained and produces the clearest finding.

---

## Key findings

`[fill in after running -- a few sentences describing what you actually found. was the degradation linear or did it cliff? did distilbert hold up better than logreg under noise? which cleaning strategy worked best? did clean beat noisy in the headline experiment? by how much?]`

---

## Things worth knowing

Running DistilBERT on CPU is slow. On a modern GPU the full noise sweep takes around X hours. On CPU expect Y hours. The LogReg experiments run in minutes either way.

All three seeds are required for the results to be meaningful. Do not report single-seed results.

The confidence and loss filters both require a trained model to score samples. That model is trained on noisy data by design -- that is not a bug. The idea is to use the model's own uncertainty to find samples that do not belong.

---

## Stack

Python 3.10, PyTorch, HuggingFace Transformers and Datasets, scikit-learn, pandas, matplotlib, seaborn
