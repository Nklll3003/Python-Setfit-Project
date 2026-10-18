# SetFit replication and new dataset (COMP8240)

Project for COMP8240 Applications of Data Science (Macquarie University).

The project replicates **SetFit**, the few-shot text classification method from
Tunstall et al. (2022), *Efficient Few-Shot Learning Without Prompts*
(https://arxiv.org/abs/2209.11055), and tests it on a newly constructed dataset of
Spotify app reviews from Google Play.

- Original code: https://github.com/huggingface/setfit
- Original dataset used here: https://huggingface.co/datasets/SetFit/CR (Customer Reviews)

## Repository structure

| Folder | Contents |
|---|---|
| `task2_project_update/` | Work for the Project Update: feasibility check on Customer Reviews, collection and first experiment on the Spotify dataset |
| `task3_final_report/` | Work for the Final Report: repeated runs over 10 random splits on both datasets |

Each folder is self-contained. Run its scripts from inside that folder, because the
scripts read and write files in the current folder.

## Environment

- Python 3.10
- `setfit==1.0.3` and `transformers==4.36.2` (newer versions break the original code)
- CPU is enough. One training run takes about 2 to 7 minutes on a laptop CPU.

```
pip install setfit==1.0.3 transformers==4.36.2 pandas scikit-learn
pip install google-play-scraper     # only for collect_spotify_reviews.py
```

Base model: `sentence-transformers/paraphrase-mpnet-base-v2`.
Training settings used here: batch size 8, 10 iterations (R=10), 1 epoch, 16 training examples (8 per class).
The paper reports batch size 16 and R=20 (Tunstall et al., 2022), so these two settings differ from the original.

## task2_project_update

| File | Purpose |
|---|---|
| `setfit_customer_reviews_feasibility.py` | One run of SetFit on Customer Reviews (16 training examples, 376 test reviews). Writes `customer_reviews_feasibility_results.txt` |
| `collect_spotify_reviews.py` | Collects 500 Spotify reviews from Google Play, cleans them to 373. Writes `spotify_reviews_raw.csv` and `spotify_collection_log.txt`. Do not re-run once labelling has started, because it fetches new reviews |
| `collect_spotify_pilot.py` | Small pilot collection (40 reviews) |
| `setfit_spotify_experiment.py` | One run of SetFit on the Spotify dataset. Needs `spotify_reviews_labeled_DRAFT.csv`. Writes `spotify_experiment_results.txt`, `spotify_experiment_errors.csv`, `spotify_experiment_predictions.csv` |
| `1.run_commands.txt` | The commands in the order they were run |

## task3_final_report

| File | Purpose |
|---|---|
| `setfit_customer_reviews_multiseed.py` | SetFit on Customer Reviews over 10 random splits (8 examples per class). Writes `customer_reviews_multiseed_runs.csv` and `customer_reviews_multiseed_results.txt` |
| `setfit_spotify_fixedtest.py` | **Main Spotify result.** SetFit on the checked labels over 10 training splits with one fixed test set (45 reviews per class, seed 42). Writes `spotify_fixedtest_testset.csv`, `spotify_fixedtest_runs.csv`, `spotify_fixedtest_results.txt` |
| `setfit_financial_phrasebank_fixedtest.py` | **Main Financial PhraseBank result.** Same protocol: fixed test set of 150 sentences per class (seed 42), 10 training splits. Needs internet to download the dataset. Writes `financial_phrasebank_fixedtest_runs.csv` and `financial_phrasebank_fixedtest_results.txt` |
| `setfit_spotify_multiseed.py` | *Earlier protocol (test set drawn again for each split).* SetFit on the Spotify dataset over 10 splits, using the first draft labels (`spotify_reviews_labeled_DRAFT.csv`). Writes `spotify_multiseed_runs.csv` and `spotify_multiseed_results.txt` |
| `setfit_spotify_multiseed_verified.py` | *Earlier protocol.* Same experiment on the checked labels (`spotify_reviews_labeled_VERIFIED.csv`). Writes `spotify_multiseed_verified_runs.csv` and `spotify_multiseed_verified_results.txt` |
| `spotify_label_agreement.py` | Agreement between the two labelling passes, and between a human annotator and the final labels. Writes `spotify_label_agreement.txt` |
| `spotify_reviews_labeled_VERIFIED.csv` | The 373 reviews with first-pass label, second-pass label and final label |
| `spotify_human_check_sample.csv` | 40 random reviews from the final labels. `my_label` holds the human annotator's labels (18 rows). `claude_label` holds AI labels for the other 22 rows |
| `setfit_financial_phrasebank_multiseed.py` | *Earlier protocol.* SetFit on Financial PhraseBank (existing dataset, positive vs negative) over 10 random splits. Needs internet to download the dataset from Hugging Face. Writes `financial_phrasebank_multiseed_runs.csv`, `financial_phrasebank_multiseed_results.txt` and `financial_phrasebank_binary_sample.csv` |
| `spotify_llm_judge_errors.csv` | The 19 reviews SetFit got wrong in the first single run, with the LLM judge's label and reason |
| `llm_judge_summary.py` | Summarises the judge check. Writes `llm_judge_summary.txt` |
| `setfit_multiseed_results.xlsx` | Summary table of the earlier runs (earlier protocol) |

All SetFit scripts save after every split and skip splits already in the runs CSV, so they
can be stopped and started again. Delete the runs CSV if you change the labels.

```
python setfit_customer_reviews_multiseed.py
python spotify_label_agreement.py
python setfit_spotify_fixedtest.py
python setfit_financial_phrasebank_fixedtest.py
python llm_judge_summary.py

# earlier protocol (test set drawn again for each split), kept for comparison
python setfit_spotify_multiseed.py
python setfit_spotify_multiseed_verified.py
python setfit_financial_phrasebank_multiseed.py
```

## Results (10 random splits, accuracy %)

Protocol used in the report: 10 random training splits (seeds 0 to 9, 8 examples per class)
and **one fixed test set per dataset**, as in the paper. Mean ± sample standard deviation.

| Dataset | Mean ± SD | Min / Max | Test examples |
|---|---|---|---|
| Customer Reviews, paper (Table 2, N=8) | 88.5 ± 1.9 | | |
| Customer Reviews, this replication | 90.0 ± 1.0 | 88.8 / 91.5 | 376 (original test split) |
| Financial PhraseBank, positive vs negative | 95.2 ± 0.5 | 94.3 / 96.0 | 300 (150 per class) |
| Spotify, checked labels | 92.6 ± 1.7 | 88.9 / 94.4 | 90 (45 per class) |

Macro-F1: Financial PhraseBank 95.2 ± 0.5, Spotify 92.5 ± 1.7.

Earlier protocol (test set drawn again for each split, so test sets overlap between splits):

| Dataset | Mean ± SD | Min / Max | Test examples |
|---|---|---|---|
| Financial PhraseBank | 94.5 ± 1.8 | 91.2 / 96.8 | 590 |
| Spotify, checked labels | 91.4 ± 5.6 | 76.4 / 96.0 | 174 |
| Spotify, first draft labels | 88.0 ± 4.6 | 79.7 / 95.3 | 192 |

The Spotify test set has only 90 reviews, so one review is 1.1 points of accuracy. The checked
set also leaves out ambiguous reviews, so the Spotify and Customer Reviews numbers are not
directly comparable. The draft-label run used the earlier protocol.

## The new dataset (Spotify reviews)

- Source: Google Play reviews for `com.spotify.music`, sampled by star rating
  (150 one-star, 100 two-star, 100 four-star, 150 five-star). Three-star reviews are skipped.
- Cleaning: 500 reviews, 373 after removing very short reviews, non-English reviews and duplicates.
- Labelling, two passes by the same annotator (an AI model, Claude), from the review text:
  a label is kept only when both passes agree on positive or negative. Result: 95 positive,
  209 negative, 69 excluded as ambiguous.
- Agreement between the two passes: 93.6% raw agreement, Cohen's kappa 0.887 (three classes).
  This is the consistency of one annotator, not agreement between two annotators.
- Human check: the repository owner labelled 18 of 40 randomly drawn reviews without seeing the
  final labels. Agreement with the final labels is 100% (kappa 1.000). The sample is small,
  and it only contains reviews that have a final positive or negative label.
- The first draft pass appears to have used the star rating to settle some mixed reviews,
  although the intended rule was to label from the text.

## LLM-as-judge

Two LLM judges (Claude and ChatGPT) labelled the 19 reviews that SetFit misclassified in the
first single run as positive, negative or ambiguous. Claude judged 9 of 19 as clear SetFit
errors and 10 as ambiguous; ChatGPT judged 14 as clear errors and 5 as ambiguous. The two
judges agree on 12 of 19 reviews (63%, Cohen's kappa 0.46); where both picked a side they
agree 8 of 8 times. Four errors are ambiguous for both judges, eleven for at least one.
Limitations: ChatGPT was given the file of errors including the draft label and the SetFit
prediction, so its judgement is not blind. Claude is also the model that produced the labels.
The check covers 19 errors from one run, so it is an error analysis, not evidence of label quality.
