"""
setfit_spotify_experiment.py
----------------------------
Few-shot SetFit on the NEW constructed dataset (Spotify app-store reviews).

  - Train : 16 labelled reviews (8 positive + 8 negative), same few-shot size
            as the Customer Reviews feasibility run
  - Test  : all remaining labelled reviews, class-balanced (equal positive /
            negative) so that accuracy is easy to interpret
  - Output (written to the folder you run the command from):
        spotify_experiment_results.txt      full report (also printed on screen)
        spotify_experiment_errors.csv       test reviews the model got WRONG
        spotify_experiment_predictions.csv  every test review with true / predicted label

Usage (inside the venv, from the folder that holds the labelled CSV):
    python setfit_spotify_experiment.py
    python setfit_spotify_experiment.py --dry-run     # only shows the train/test split
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
LABELED_FILE = "spotify_reviews_labeled_DRAFT.csv"
N_PER_CLASS_TRAIN = 8
SEED = 42
BASE_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
BATCH_SIZE = 8
NUM_ITERATIONS = 10
NUM_EPOCHS = 1

RESULTS_FILE = "spotify_experiment_results.txt"
ERRORS_FILE = "spotify_experiment_errors.csv"
PREDS_FILE = "spotify_experiment_predictions.csv"

LINE = "=" * 72
THIN = "-" * 72

_lines = []


def log(message=""):
    print(message)
    _lines.append(str(message))


def section(title):
    log("")
    log(LINE)
    log(title)
    log(LINE)


def shorten(text, width=170):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


# ----------------------------------------------------------------------
# Step 1: load the labelled data
# ----------------------------------------------------------------------
df = pd.read_csv(LABELED_FILE, encoding="utf-8-sig")
df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
n_all = len(df)

df = df[df["label"].isin(["positive", "negative"])].copy()
df["y"] = (df["label"] == "positive").astype(int)   # 1 = positive, 0 = negative

# ----------------------------------------------------------------------
# Step 2: few-shot train set + class-balanced test set
# ----------------------------------------------------------------------
pos = df[df["y"] == 1].sample(frac=1, random_state=SEED)
neg = df[df["y"] == 0].sample(frac=1, random_state=SEED)

train = pd.concat([pos.iloc[:N_PER_CLASS_TRAIN], neg.iloc[:N_PER_CLASS_TRAIN]])
train = train.sample(frac=1, random_state=SEED)

pos_rest = pos.iloc[N_PER_CLASS_TRAIN:]
neg_rest = neg.iloc[N_PER_CLASS_TRAIN:]
n_test_per_class = min(len(pos_rest), len(neg_rest))
test = pd.concat([pos_rest.iloc[:n_test_per_class], neg_rest.iloc[:n_test_per_class]])
test = test.sample(frac=1, random_state=SEED)

log(LINE)
log("SetFit on the new constructed dataset (Spotify reviews)")
log(LINE)
log(f"Run time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Data file : {LABELED_FILE}")

section("1. DATASET")
log(f"Reviews in file            : {n_all}")
log(f"Usable labelled reviews    : {len(df)}  "
    f"({int((df['y'] == 1).sum())} positive / {int((df['y'] == 0).sum())} negative)")
log(f"Train set (few-shot)       : {len(train)}  "
    f"({int((train['y'] == 1).sum())} positive / {int((train['y'] == 0).sum())} negative)")
log(f"Test set (class-balanced)  : {len(test)}  "
    f"({int((test['y'] == 1).sum())} positive / {int((test['y'] == 0).sum())} negative)")
log(f"Unused labelled reviews    : {len(df) - len(train) - len(test)}")
log(f"Random seed                : {SEED}")

if "--dry-run" in sys.argv:
    log("\n--dry-run: stopping before training.")
    sys.exit(0)

# ----------------------------------------------------------------------
# Step 3: train SetFit (same settings as the Customer Reviews feasibility run)
# ----------------------------------------------------------------------
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from transformers import set_seed

set_seed(SEED)

train_ds = Dataset.from_pandas(
    train[["review_text", "y"]].rename(columns={"review_text": "text", "y": "label"}),
    preserve_index=False,
)
test_ds = Dataset.from_pandas(
    test[["review_text", "y"]].rename(columns={"review_text": "text", "y": "label"}),
    preserve_index=False,
)

section("2. MODEL")
log(f"Base model  : {BASE_MODEL}")
log(f"Settings    : batch_size={BATCH_SIZE}, num_iterations={NUM_ITERATIONS}, "
    f"num_epochs={NUM_EPOCHS}")

model = SetFitModel.from_pretrained(BASE_MODEL)
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=BATCH_SIZE,
    num_iterations=NUM_ITERATIONS,
    num_epochs=NUM_EPOCHS,
)
trainer.train()

# ----------------------------------------------------------------------
# Step 4: evaluate
# ----------------------------------------------------------------------
texts = test["review_text"].tolist()
y_true = test["y"].to_numpy()
y_pred = np.asarray(model.predict(texts)).astype(int)

acc = accuracy_score(y_true, y_pred)
n_correct = int((y_true == y_pred).sum())
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

section("3. RESULTS ON THE NEW DATASET")
log(f"Accuracy : {acc:.4f}  ({n_correct}/{len(y_true)} correct)")
log("Baseline: 0.5000  (always guessing one class on this balanced test set)")
log("")
log("Confusion matrix (rows = true label, columns = predicted label)")
log("")
log("                    pred negative   pred positive")
log(f"  true negative     {cm[0][0]:>13}   {cm[0][1]:>13}")
log(f"  true positive     {cm[1][0]:>13}   {cm[1][1]:>13}")
log("")
log("Per-class scores")
log("")
log(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=3))

# ----------------------------------------------------------------------
# Step 5: save predictions and the mistakes (for error analysis)
# ----------------------------------------------------------------------
out = test[["review_id", "review_text", "star_rating", "label"]].copy()
out = out.rename(columns={"label": "true_label"})
out["predicted_label"] = np.where(y_pred == 1, "positive", "negative")
out["correct"] = (y_true == y_pred)
out.to_csv(PREDS_FILE, index=False, encoding="utf-8-sig")

errors = out[~out["correct"]]
errors.to_csv(ERRORS_FILE, index=False, encoding="utf-8-sig")

pos_as_neg = errors[errors["true_label"] == "positive"]
neg_as_pos = errors[errors["true_label"] == "negative"]

section("4. ERROR SUMMARY")
log(f"Misclassified reviews : {len(errors)} of {len(out)}  "
    f"({len(errors) / len(out):.1%})")
log(f"  positive predicted as negative : {len(pos_as_neg)}")
log(f"  negative predicted as positive : {len(neg_as_pos)}")
log("")
log("Star rating of the mistakes (shows where stars and text disagree)")
for stars, n in errors["star_rating"].value_counts().sort_index().items():
    log(f"  {stars} star : {n}")

section("5. ALL MISCLASSIFIED REVIEWS")
for title, subset in [("TRUE POSITIVE, PREDICTED NEGATIVE", pos_as_neg),
                      ("TRUE NEGATIVE, PREDICTED POSITIVE", neg_as_pos)]:
    log("")
    log(f"{title}  ({len(subset)})")
    log(THIN)
    for i, (_, r) in enumerate(subset.iterrows(), start=1):
        log(f"{i:>2}. [{r['star_rating']} star] {shorten(r['review_text'])}")

section("6. OUTPUT FILES")
log(f"  {RESULTS_FILE:<36} this report")
log(f"  {ERRORS_FILE:<36} the {len(errors)} misclassified reviews")
log(f"  {PREDS_FILE:<36} prediction for every test review")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nResults saved to {RESULTS_FILE}")
