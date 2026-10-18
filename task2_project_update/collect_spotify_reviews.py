"""
collect_spotify_reviews.py
-------------------------
Collects ~500 English app reviews from Google Play, sampled PER STAR RATING so
that negative reviews (1-2 stars) are well represented, then saves:
    - spotify_reviews_raw.csv       (data, ready for manual labelling)
    - spotify_collection_log.txt    (everything printed on screen + settings used)

Sampling plan (documented so it can be described in the report):
    1 star: 150 | 2 stars: 100 | 4 stars: 100 | 5 stars: 150
    3-star reviews are skipped on purpose (mostly neutral / hard to label).

Cleaning steps:
    - drop empty / very short reviews (< 15 characters)
    - drop mostly non-English reviews (ASCII-letter ratio < 0.9)
    - drop duplicates (case-insensitive)
    - shuffle with a fixed seed (reproducible)

Usage (run on your own computer, inside the venv):
    pip install google-play-scraper pandas
    python collect_spotify_reviews.py

Output files are written to the folder you run the command from.
Older files from earlier runs are not touched.
"""

import time
from datetime import datetime

import pandas as pd
from google_play_scraper import reviews, Sort

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
APP_ID = "com.spotify.music"      # id= part of the Google Play URL
PER_STAR = {1: 150, 2: 100, 4: 100, 5: 150}
OUTPUT_FILE = "spotify_reviews_raw.csv"
LOG_FILE = "spotify_collection_log.txt"
MIN_CHARS = 15
SEED = 42

# ----------------------------------------------------------------------
# Logging helper: print to screen AND remember the line for the .txt file
# ----------------------------------------------------------------------
_log_lines = []


def log(message=""):
    print(message)
    _log_lines.append(str(message))


def save_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")


def mostly_english(text: str) -> bool:
    """Rough filter: >= 90% of the letters are plain ASCII letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if c.isascii()]
    return len(ascii_letters) / len(letters) >= 0.9


# ----------------------------------------------------------------------
# Header: record the settings used for this run
# ----------------------------------------------------------------------
log("=== App review collection log ===")
log(f"Run time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"App ID          : {APP_ID}")
log(f"Language/Country: en / us")
log(f"Reviews per star: {PER_STAR}")
log(f"Min characters  : {MIN_CHARS}")
log(f"Random seed     : {SEED}")
log("")

# ----------------------------------------------------------------------
# Step 1: collect per star rating
# ----------------------------------------------------------------------
rows = []
for stars, n in PER_STAR.items():
    log(f"Collecting up to {n} reviews with {stars} star(s)...")
    result, _ = reviews(
        APP_ID,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=n,
        filter_score_with=stars,
    )
    log(f"  got {len(result)}")
    for r in result:
        rows.append({
            "review_id": r["reviewId"],
            "review_text": r["content"],
            "star_rating": r["score"],
            "date": r["at"],
        })
    time.sleep(2)  # be polite to the server

df = pd.DataFrame(rows)
log("")
log(f"Collected {len(df)} reviews in total (before cleaning)")

# ----------------------------------------------------------------------
# Step 2: cleaning (counts are logged so they can be quoted in the report)
# ----------------------------------------------------------------------
df = df.dropna(subset=["review_text"]).copy()
df["review_text"] = (
    df["review_text"].str.strip().str.replace(r"\s+", " ", regex=True)
)
n_start = len(df)

df = df[df["review_text"].str.len() >= MIN_CHARS]
n_len = len(df)

df = df[df["review_text"].apply(mostly_english)]
n_eng = len(df)

df = df[~df["review_text"].str.lower().duplicated()]
n_dedup = len(df)

log("Cleaning summary:")
log(f"  after removing empty                : {n_start}")
log(f"  after removing very short           : {n_len}")
log(f"  after removing non-English          : {n_eng}")
log(f"  after removing duplicates           : {n_dedup}")

# ----------------------------------------------------------------------
# Step 3: shuffle (fixed seed), add empty label column, save
# ----------------------------------------------------------------------
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df["label"] = ""          # fill in: positive / negative (leave blank if unclear)
df["note"] = ""           # optional: why excluded / ambiguous

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
log("")
log(f"Saved {len(df)} reviews to {OUTPUT_FILE}")
log("Reviews per star rating:")
log(df["star_rating"].value_counts().sort_index().to_string())

# ----------------------------------------------------------------------
# Step 4: write the .txt log
# ----------------------------------------------------------------------
save_log()
print(f"\nLog saved to {LOG_FILE}")
print("Next: open the CSV in Google Sheets and fill in the 'label' column.")
print("Tip: do NOT copy the star rating blindly - label by the TEXT.")
