"""
setfit_customer_reviews_multiseed.py
------------------------------------
Replication of the paper's few-shot setting on the Customer Reviews dataset
(SetFit/CR): 8 examples per class (16 in total), repeated over several random
splits. Each split trains a new model and is tested on the full test set.

Output (written to the folder you run the command from):
    customer_reviews_multiseed_runs.csv      one row per split (saved after every run)
    customer_reviews_multiseed_results.txt   mean / SD, compared with the paper

The script can be stopped and started again: splits already in the CSV are skipped.

Usage (inside the venv):
    python setfit_customer_reviews_multiseed.py
"""

import csv
import os
import platform
import time
from datetime import datetime

import numpy as np
import setfit
import torch
import transformers
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from transformers import set_seed

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATASET_NAME = "SetFit/CR"
SEEDS = list(range(10))          # 10 splits, as in the paper
N_PER_CLASS = 8                  # 8 per class = 16 training examples
BASE_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
BATCH_SIZE = 8
NUM_ITERATIONS = 10
NUM_EPOCHS = 1
PAPER_MEAN, PAPER_SD = 88.5, 1.9  # SetFit MPNet, Customer Reviews, N=8 (Table 2)

RUNS_FILE = "customer_reviews_multiseed_runs.csv"
RESULTS_FILE = "customer_reviews_multiseed_results.txt"
LINE = "=" * 72


def read_done():
    done = {}
    if os.path.exists(RUNS_FILE):
        with open(RUNS_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[int(row["seed"])] = float(row["accuracy"])
    return done


def append_run(seed, acc, n_pos, n_neg, minutes):
    new_file = not os.path.exists(RUNS_FILE)
    with open(RUNS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["seed", "accuracy", "train_positive", "train_negative", "minutes"])
        w.writerow([seed, f"{acc:.4f}", n_pos, n_neg, f"{minutes:.1f}"])


# ----------------------------------------------------------------------
# Run every split
# ----------------------------------------------------------------------
dataset = load_dataset(DATASET_NAME)
eval_dataset = dataset["test"]
done = read_done()
print(f"Splits already done: {sorted(done)}")

for seed in SEEDS:
    if seed in done:
        continue
    start = time.time()
    set_seed(seed)
    train_dataset = sample_dataset(dataset["train"], label_column="label",
                                   num_samples=N_PER_CLASS, seed=seed)
    labels = list(train_dataset["label"])

    model = SetFitModel.from_pretrained(BASE_MODEL)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_class=CosineSimilarityLoss,
        batch_size=BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        num_epochs=NUM_EPOCHS,
    )
    trainer.train()
    acc = trainer.evaluate()["accuracy"]
    minutes = (time.time() - start) / 60

    append_run(seed, acc, labels.count(1), labels.count(0), minutes)
    done[seed] = acc
    print(f"[seed {seed}] accuracy {acc:.4f}  ({minutes:.1f} min)  "
          f"{len(done)}/{len(SEEDS)} splits done")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
accs = np.array([done[s] for s in SEEDS]) * 100
mean, sd = accs.mean(), accs.std(ddof=1)

lines = [
    LINE,
    "SetFit on Customer Reviews: results over several random splits",
    LINE,
    f"Run time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Python        : {platform.python_version()}",
    f"setfit        : {setfit.__version__}",
    f"transformers  : {transformers.__version__}",
    f"torch         : {torch.__version__}",
    "",
    f"Dataset       : {DATASET_NAME} ({len(eval_dataset)} test reviews)",
    f"Training set  : {N_PER_CLASS} examples per class ({2 * N_PER_CLASS} in total), "
    f"new random sample for each split",
    f"Splits        : {len(SEEDS)} (seeds {SEEDS[0]} to {SEEDS[-1]})",
    f"Settings      : batch_size={BATCH_SIZE}, num_iterations={NUM_ITERATIONS}, "
    f"num_epochs={NUM_EPOCHS}",
    "",
    LINE,
    "ACCURACY PER SPLIT (%)",
    LINE,
]
for s, a in zip(SEEDS, accs):
    lines.append(f"  seed {s:<3}: {a:.1f}")
lines += [
    "",
    LINE,
    "SUMMARY",
    LINE,
    f"This replication : {mean:.1f} +/- {sd:.1f}   (mean +/- standard deviation, "
    f"min {accs.min():.1f}, max {accs.max():.1f})",
    f"Paper (Table 2)  : {PAPER_MEAN} +/- {PAPER_SD}",
    f"Difference       : {mean - PAPER_MEAN:+.1f} points",
    "",
]
text = "\n".join(lines)
print("\n" + text)
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print(f"Results saved to {RESULTS_FILE}")
