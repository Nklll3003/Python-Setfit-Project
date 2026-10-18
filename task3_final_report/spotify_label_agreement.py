"""
spotify_label_agreement.py
--------------------------
Agreement measures for the Spotify review labels.

Comparison 1 (always): first labelling pass vs second labelling pass, from
    spotify_reviews_labeled_VERIFIED.csv (columns draft_label, second_pass_label;
    an empty value means the review was excluded in that pass).
Comparison 2 (only if filled in): your own labels vs the final labels, from
    spotify_human_check_sample.csv (column my_label: positive, negative or exclude).

Output: spotify_label_agreement.txt (also printed on screen)

Usage (inside the venv, from the folder that holds the CSV files):
    python spotify_label_agreement.py
"""

import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

VERIFIED = "spotify_reviews_labeled_VERIFIED.csv"
SAMPLE = "spotify_human_check_sample.csv"
OUT = "spotify_label_agreement.txt"
LINE = "=" * 72
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def table(a, b, names):
    cm = confusion_matrix(a, b, labels=names)
    log(f"{'':>12}" + "".join(f"{('2nd ' + n):>14}" for n in names))
    for n, row in zip(names, cm):
        log(f"{('1st ' + n):>12}" + "".join(f"{v:>14}" for v in row))


df = pd.read_csv(VERIFIED, encoding="utf-8-sig").fillna("")
first = df["draft_label"].replace("", "exclude")
second = df["second_pass_label"].replace("", "exclude")
names = ["positive", "negative", "exclude"]

log(LINE)
log("1. FIRST PASS vs SECOND PASS (all reviews)")
log(LINE)
log(f"Reviews                : {len(df)}")
log(f"Raw agreement          : {(first == second).mean():.1%}  ({int((first == second).sum())}/{len(df)})")
log(f"Cohen's kappa (3 classes: positive / negative / exclude) : {cohen_kappa_score(first, second):.3f}")
log("")
table(first, second, names)
both = df[(df["draft_label"] != "") & (df["second_pass_label"] != "")]
log("")
log(f"Reviews labelled positive or negative in both passes: {len(both)}")
log(f"  agreement : {(both['draft_label'] == both['second_pass_label']).mean():.1%}")
log("")
log("Final labels: a label is kept only when both passes agree on positive or")
log("negative. Every other review is excluded as ambiguous.")
fin = df["label"].value_counts()
log(f"  positive {int(fin.get('positive', 0))} | negative {int(fin.get('negative', 0))} | "
    f"excluded {int((df['label'] == '').sum())}")

if os.path.exists(SAMPLE):
    s = pd.read_csv(SAMPLE, encoding="utf-8-sig").fillna("")
    s = s[s["my_label"].str.strip() != ""].copy()
    if len(s):
        s["my_label"] = s["my_label"].str.strip().str.lower()
        s = s.merge(df[["review_id", "label"]], on="review_id", how="left")
        s["label"] = s["label"].replace("", "exclude")
        log("")
        log(LINE)
        log("2. YOUR LABELS vs FINAL LABELS (sample)")
        log(LINE)
        log(f"Reviews labelled by you : {len(s)}")
        log(f"Raw agreement           : {(s['my_label'] == s['label']).mean():.1%}")
        log(f"Cohen's kappa           : {cohen_kappa_score(s['my_label'], s['label']):.3f}")
        log("")
        table(s["my_label"], s["label"], names)
        log("(rows = your label, columns = final label)")
        diff = s[s["my_label"] != s["label"]]
        if len(diff):
            log("")
            log("Reviews where you and the final label differ:")
            for _, r in diff.iterrows():
                log(f"  you={r['my_label']:<9} final={r['label']:<9} {' '.join(str(r['review_text']).split())[:110]}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved to {OUT}")
