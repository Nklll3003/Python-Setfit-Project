"""
collect_spotify_pilot.py
------------------------
PILOT (40 reviews, first test only; the real dataset uses collect_spotify_reviews.py).
Collects app reviews from the Google Play Store and saves them to a CSV
file, ready for you to open in Excel / Google Sheets and label yourself
(positive/negative).

No API key or account needed - uses the public google-play-scraper library.

Usage (run this on your own computer, not in this sandbox):
    1) Install the required libraries (once):
       pip install google-play-scraper pandas
    2) Run the script:
       python collect_spotify_pilot.py
    3) You'll get a file spotify_pilot_raw.csv in the same folder
"""

from google_play_scraper import reviews, Sort
import pandas as pd

# ----------------------------------------------------------------------
# Step 1: Settings - change these if you want to collect reviews for a
# different app
# ----------------------------------------------------------------------

# The app ID is the part after "id=" in the app's Google Play URL, e.g.
# https://play.google.com/store/apps/details?id=com.spotify.music
# -> APP_ID = "com.spotify.music"
APP_ID = "com.spotify.music"

# Number of reviews to collect (collect extra up front, since some
# reviews will get filtered out)
NUM_REVIEWS = 40

# Output file name
OUTPUT_FILE = "spotify_pilot_raw.csv"

# ----------------------------------------------------------------------
# Step 2: Fetch reviews from the Google Play Store
# ----------------------------------------------------------------------

print(f"Collecting up to {NUM_REVIEWS} reviews for app '{APP_ID}'...")

result, _ = reviews(
    APP_ID,
    lang="en",          # fetch English-language reviews
    country="us",
    sort=Sort.NEWEST,    # newest reviews first
    count=NUM_REVIEWS,
)

print(f"Collected {len(result)} reviews in total")

# ----------------------------------------------------------------------
# Step 3: Keep only the fields we need + do basic cleaning
# ----------------------------------------------------------------------

rows = []
for r in result:
    text = r["content"]
    if text is None:
        continue
    text = text.strip().replace("\n", " ")
    if len(text) < 5:          # skip reviews that are too short/empty
        continue
    rows.append({
        "review_text": text,
        "star_rating": r["score"],       # 1-5 stars, a rough hint for the label
        "date": r["at"],
        "label": ""                      # <-- fill this in yourself: "positive" or "negative"
    })

df = pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Step 4: Save to CSV so you can label it in Excel / Google Sheets
# ----------------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"Saved {len(df)} cleaned reviews to {OUTPUT_FILE}")
print("Open this file and fill in the 'label' column with 'positive' or 'negative'")
print("Tip: star_rating 4-5 is usually positive, 1-2 is usually negative "
      "- use this as a rough guide, but always read the actual review before labeling")