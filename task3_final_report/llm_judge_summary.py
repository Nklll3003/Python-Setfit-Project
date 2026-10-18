"""
llm_judge_summary.py
--------------------
Summarises the LLM-as-judge check of the SetFit errors on the Spotify dataset.

Input : spotify_llm_judge_errors.csv
        The 19 test reviews that SetFit got wrong in the first single run on the draft
        labels (spotify_experiment_errors.csv from task2_project_update). Columns:
        review_id, review_text, star_rating, draft_label, setfit_prediction,
        claude_label (positive / negative / mixed / unclear), claude_reason,
        chatgpt_label (positive / negative / ambiguous).
Output: llm_judge_summary.txt (also printed on screen)

Judge task (one review at a time): decide the overall opinion of the reviewer about
the app. positive, negative, or ambiguous (mixed opinions, or no clear opinion).
Claude's "mixed" and "unclear" answers are both counted as "ambiguous" here.
Note: ChatGPT was given the file of errors, which includes the draft label and the
SetFit prediction. It states that it did not use them, but this cannot be verified.

Usage (inside the venv, from the folder that holds the CSV):
    python llm_judge_summary.py
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv("spotify_llm_judge_errors.csv", encoding="utf-8-sig")
df["claude"] = df["claude_label"].replace({"mixed": "ambiguous", "unclear": "ambiguous"})
df["chatgpt"] = df["chatgpt_label"]
LINE = "=" * 72
out = [LINE, "LLM-AS-JUDGE CHECK OF SETFIT ERRORS (Spotify, first single run)", LINE,
       f"Misclassified reviews judged : {len(df)}", ""]


def group(r, col):
    if r[col] == "ambiguous":
        return "ambiguous"
    return "clear error (judge agrees with draft label)" if r[col] == r["draft_label"] \
        else "judge agrees with SetFit (draft label doubtful)"


for name in ("claude", "chatgpt"):
    out.append(f"{name}:")
    for g, n in df.apply(lambda r: group(r, name), axis=1).value_counts().items():
        out.append(f"  {g:<50}: {n:>2}  ({n / len(df):.0%})")
out.append("")
agree = int((df["claude"] == df["chatgpt"]).sum())
out.append(f"Claude vs ChatGPT agreement : {agree}/{len(df)} ({agree / len(df):.1%})")
out.append(f"Cohen's kappa (3 classes)   : {cohen_kappa_score(df['claude'], df['chatgpt']):.3f}")
both = df[(df["claude"] != "ambiguous") & (df["chatgpt"] != "ambiguous")]
out.append(f"Both picked a side          : {len(both)} reviews, agree on {int((both['claude'] == both['chatgpt']).sum())}")
out.append(f"Ambiguous by both judges    : {int(((df['claude'] == 'ambiguous') & (df['chatgpt'] == 'ambiguous')).sum())}")
out.append(f"Ambiguous by at least one   : {int(((df['claude'] == 'ambiguous') | (df['chatgpt'] == 'ambiguous')).sum())}")
out += ["", "Claude (rows) vs ChatGPT (columns):", pd.crosstab(df["claude"], df["chatgpt"]).to_string(), ""]
text = "\n".join(out)
print(text)
open("llm_judge_summary.txt", "w", encoding="utf-8").write(text + "\n")
