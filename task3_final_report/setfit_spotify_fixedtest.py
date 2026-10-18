"""
setfit_spotify_fixedtest.py
---------------------------
Few-shot SetFit on the new Spotify review dataset with a FIXED test set.
A class-balanced test set (45 positive + 45 negative reviews) is drawn once with a
fixed seed and never used for training. Each of the 10 splits then draws a new set of
16 training reviews (8 positive + 8 negative) from the remaining reviews. This follows
the protocol of the SetFit paper (fixed test set, 10 random training splits).

Output (written to the folder you run the command from):
    spotify_fixedtest_testset.csv       the fixed test set (review_id, label)
    spotify_fixedtest_runs.csv          one row per split (saved after every run)
    spotify_fixedtest_results.txt       mean / SD over all splits

The script can be stopped and started again: splits already in the CSV are skipped.
If you change the labels in the labelled CSV, delete spotify_fixedtest_runs.csv first.

Usage (inside the venv, from the folder that holds the labelled CSV):
    python setfit_spotify_fixedtest.py
"""

import csv
import os
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
LABELED_FILE = "spotify_reviews_labeled_VERIFIED.csv"
SEEDS = list(range(10))
N_PER_CLASS_TRAIN = 8
BASE_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
BATCH_SIZE = 8
NUM_ITERATIONS = 10
NUM_EPOCHS = 1
TEST_PER_CLASS = 45
TEST_SEED = 42

RUNS_FILE = "spotify_fixedtest_runs.csv"
RESULTS_FILE = "spotify_fixedtest_results.txt"
TESTSET_FILE = "spotify_fixedtest_testset.csv"
LINE = "=" * 72

# ----------------------------------------------------------------------
# Load the labelled data
# ----------------------------------------------------------------------
df = pd.read_csv(LABELED_FILE, encoding="utf-8-sig")
df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
n_all = len(df)
df = df[df["label"].isin(["positive", "negative"])].copy()
df["y"] = (df["label"] == "positive").astype(int)
n_pos_all, n_neg_all = int((df["y"] == 1).sum()), int((df["y"] == 0).sum())


# Fixed test set: drawn once, never used for training
test = pd.concat([
    df[df["y"] == 1].sample(TEST_PER_CLASS, random_state=TEST_SEED),
    df[df["y"] == 0].sample(TEST_PER_CLASS, random_state=TEST_SEED),
]).sample(frac=1, random_state=TEST_SEED)
test[["review_id", "label"]].to_csv(TESTSET_FILE, index=False)
pool = df.drop(test.index)
n_pos_pool, n_neg_pool = int((pool["y"] == 1).sum()), int((pool["y"] == 0).sum())


def make_split(seed):
    pos = pool[pool["y"] == 1].sample(frac=1, random_state=seed)
    neg = pool[pool["y"] == 0].sample(frac=1, random_state=seed)
    train = pd.concat([pos.iloc[:N_PER_CLASS_TRAIN], neg.iloc[:N_PER_CLASS_TRAIN]])
    train = train.sample(frac=1, random_state=seed)
    return train, test


def read_done():
    done = {}
    if os.path.exists(RUNS_FILE):
        with open(RUNS_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[int(row["seed"])] = row
    return done


def append_run(row):
    new_file = not os.path.exists(RUNS_FILE)
    with open(RUNS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["seed", "accuracy", "f1_macro", "test_size", "errors", "minutes"])
        w.writerow(row)


# ----------------------------------------------------------------------
# Run every split
# ----------------------------------------------------------------------
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score, f1_score
from transformers import set_seed
import setfit
import torch
import transformers

done = read_done()
print(f"Labelled reviews: {len(df)} ({n_pos_all} positive / {n_neg_all} negative)")
print(f"Fixed test set: {len(test)} reviews; training pool: {len(pool)} ({n_pos_pool} positive / {n_neg_pool} negative)")
print(f"Splits already done: {sorted(done)}")


def to_ds(frame):
    return Dataset.from_pandas(
        frame[["review_text", "y"]].rename(columns={"review_text": "text", "y": "label"}),
        preserve_index=False,
    )


for seed in SEEDS:
    if seed in done:
        continue
    start = time.time()
    set_seed(seed)
    train, test = make_split(seed)

    model = SetFitModel.from_pretrained(BASE_MODEL)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=to_ds(train),
        eval_dataset=to_ds(test),
        loss_class=CosineSimilarityLoss,
        batch_size=BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        num_epochs=NUM_EPOCHS,
    )
    trainer.train()

    y_true = test["y"].to_numpy()
    y_pred = np.asarray(model.predict(test["review_text"].tolist())).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    errors = int((y_true != y_pred).sum())
    minutes = (time.time() - start) / 60

    row = [seed, f"{acc:.4f}", f"{f1:.4f}", len(test), errors, f"{minutes:.1f}"]
    append_run(row)
    done[seed] = dict(zip(["seed", "accuracy", "f1_macro", "test_size", "errors", "minutes"], row))
    print(f"[seed {seed}] accuracy {acc:.4f}  F1 {f1:.4f}  errors {errors}/{len(test)}  "
          f"({minutes:.1f} min)  {len(done)}/{len(SEEDS)} splits done")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
accs = np.array([float(done[s]["accuracy"]) for s in SEEDS]) * 100
f1s = np.array([float(done[s]["f1_macro"]) for s in SEEDS]) * 100
test_size = int(done[SEEDS[0]]["test_size"])

lines = [
    LINE,
    "SetFit on the Spotify review dataset: results over several random splits",
    LINE,
    f"Run time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Python        : {platform.python_version()}",
    f"setfit        : {setfit.__version__}",
    f"transformers  : {transformers.__version__}",
    f"torch         : {torch.__version__}",
    "",
    f"Data file     : {LABELED_FILE}",
    f"Labelled      : {len(df)} reviews ({n_pos_all} positive / {n_neg_all} negative), "
    f"{n_all - len(df)} excluded",
    f"Training set  : {N_PER_CLASS_TRAIN} positive + {N_PER_CLASS_TRAIN} negative, "
    f"new random sample for each split, drawn from a pool of {len(pool)} reviews",
    f"Test set      : {test_size} reviews ({TEST_PER_CLASS} per class), fixed for all splits (seed {TEST_SEED})",
    f"Splits        : {len(SEEDS)} (seeds {SEEDS[0]} to {SEEDS[-1]})",
    f"Settings      : batch_size={BATCH_SIZE}, num_iterations={NUM_ITERATIONS}, "
    f"num_epochs={NUM_EPOCHS}",
    "",
    LINE,
    "RESULT PER SPLIT (%)",
    LINE,
]
for s, a, f in zip(SEEDS, accs, f1s):
    lines.append(f"  seed {s:<3}: accuracy {a:.1f}   macro-F1 {f:.1f}")
lines += [
    "",
    LINE,
    "SUMMARY",
    LINE,
    f"Accuracy : {accs.mean():.1f} +/- {accs.std(ddof=1):.1f}   "
    f"(mean +/- standard deviation, min {accs.min():.1f}, max {accs.max():.1f})",
    f"Macro-F1 : {f1s.mean():.1f} +/- {f1s.std(ddof=1):.1f}",
    "Baseline : 50.0 (always guessing one class on a balanced test set)",
    "",
]
text = "\n".join(lines)
print("\n" + text)
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print(f"Results saved to {RESULTS_FILE}")
