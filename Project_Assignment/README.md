# Comparative Study of Traditional ML and Transformer Models for Spam Classification

A comparison of a classical TF-IDF + Logistic Regression / Naive Bayes pipeline against a
fine-tuned **DistilBERT** transformer for binary spam/ham text classification, built with a
custom PyTorch training loop (no `Trainer` API).

## Overview

This project implements and evaluates two approaches to spam classification on the same
data splits:

- **Part A — Traditional ML**: text cleaning → TF-IDF vectorization → Logistic Regression
  and Multinomial Naive Bayes.
- **Part B — Transformer**: raw text → DistilBERT tokenizer → fine-tuned
  `distilbert-base-uncased` with a partially frozen encoder, trained via a hand-written
  PyTorch loop.

Both pipelines are scored with Accuracy, Precision, Recall, F1-score, and confusion
matrices on an identical held-out test set, and compared on training cost.

## Results Summary

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.9704 | 0.8440 | 0.9388 | 0.8889 |
| Naive Bayes | 0.9807 | 1.0000 | 0.8469 | 0.9171 |
| DistilBERT | 0.9910 | 0.9789 | 0.9490 | 0.9637 |

DistilBERT outperforms both classical models on every metric, at roughly 2–3 orders of
magnitude more training time (~115s for 3 epochs vs. sub-second for the classical models).
See `Spam_Classification_Technical_Report.docx` for full methodology, error analysis, and
discussion.

## Dataset

The notebook uses the **SMS Spam Collection** dataset (5,572 labeled messages, ham/spam),
loaded directly from a public GitHub mirror:

```
https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv
```

> **Note:** The original assignment specifies the Kaggle
> [Spam Email Classification](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification)
> dataset. This project uses the SMS Spam Collection dataset instead, since it has the same
> two-column (`label`, `text`) schema and similar class-imbalance characteristics. The
> pipeline is dataset-agnostic and will run unchanged on the originally specified Kaggle
> file — swap the loading cell for a `pd.read_csv()` of the downloaded Kaggle CSV if exact
> compliance with the assignment's dataset is required.

**Preprocessing**: duplicate messages are removed (403 found, 7.23%), labels are mapped to
`ham=0` / `spam=1`, and the cleaned data is split 70% / 15% / 15% into train / validation /
test with stratification on the label to preserve class balance across splits.

## Project Structure

```
.
├── Project_Assignment.ipynb                    # Main notebook: EDA, Part A, Part B, visualizations
├── Spam_Classification_Technical_Report.docx   # Full written report (methodology, results, discussion)
├── visualizations/
│   ├── metric_comparison.png                   # Accuracy/Precision/Recall/F1 bar chart
│   ├── confusion_matrices.png                  # Confusion matrices for all 3 models
│   ├── training_curves.png                     # DistilBERT loss/accuracy/F1 across epochs
│   └── training_time_comparison.png            # Training time bar chart
└── README.md
```

## Requirements

```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
torch
transformers
```

Install with:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn torch transformers
```

The notebook downloads required NLTK corpora (`stopwords`, `wordnet`, `omw-1.4`)
automatically on first run.

## Running the Notebook

1. Open `Project_Assignment.ipynb` in Jupyter or Google Colab.
2. Run all cells top to bottom. Sections are ordered:
   1. **Get Dataset** — download and load the CSV
   2. **EDA** — class distribution, deduplication, message-length analysis
   3. **Train/Val/Test split** — stratified 70/15/15
   4. **Part A** — text cleaning, TF-IDF, Logistic Regression, Naive Bayes
   5. **Part B** — DistilBERT tokenization, model setup, custom training loop
   6. **Visualizations** — metric comparison, confusion matrices, training curves, training-time comparison
3. A GPU is recommended for Part B (falls back to CPU automatically via
   `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, but CPU training will be
   noticeably slower than the ~115s reported for GPU/Colab).

## Part A — Traditional ML Configuration

- **Cleaning**: lowercasing, URL/email/number placeholder substitution, non-alphabetic
  character stripping (keeping `!` and `$` as spam signals), stopword removal,
  WordNet lemmatization.
- **TF-IDF**: `max_features=5000`, `ngram_range=(1,2)`, `min_df=2`.
- **Logistic Regression**: `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.
- **Naive Bayes**: `MultinomialNB` with default smoothing.

## Part B — DistilBERT Configuration

- **Base model**: `distilbert-base-uncased`, `max_length=128`.
- **Layer freezing**: embeddings + first 2 of 6 transformer blocks frozen
  (28.9M / 67.0M parameters trainable, 43.2%).
- **Optimizer**: AdamW, `lr=2e-5`, `weight_decay=0.01`.
- **Schedule**: linear decay with 10% warm-up (`get_linear_schedule_with_warmup`).
- **Batch size**: 16. **Epochs**: 3. **Gradient clipping**: `max_norm=1.0`.
- **Model selection**: checkpoint with best validation F1 saved to `best_distilbert.pt` and
  reloaded for final test evaluation.

## Limitations

- Dataset substitution (SMS vs. Kaggle email dataset) — see note above.
- Single train/val/test split and single training run per model; no cross-validation or
  multi-seed averaging, so metrics are point estimates.
- Test set is small for the minority class (98 spam examples), so individual
  misclassifications shift precision/recall noticeably.
- DistilBERT hyperparameters (LR, batch size, epochs, freezing depth) follow standard
  fine-tuning heuristics rather than a systematic search.
- No evaluation of robustness to adversarial, out-of-distribution, or non-English inputs.

Full discussion in `Spam_Classification_Technical_Report.docx`.
