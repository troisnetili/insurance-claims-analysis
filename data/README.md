# Data

This project uses the **Insurance Claims Fraud Data** dataset from
[Kaggle](https://www.kaggle.com/datasets/mastmustu/insurance-claims-fraud-data)
(10,000 rows, 38 columns).

**This folder does not include the raw CSV.** Kaggle dataset licenses vary
and aren't always clear about redistribution, so rather than guess, download
it directly:

1. Visit the link above (requires a free Kaggle account)
2. Download and extract the archive
3. Place `insurance_data.csv` in this folder (`data/insurance_data.csv`)
4. From the repo root, run `python scripts/build_notebook.py` to regenerate
   the notebook against the real data, or open
   `notebooks/claim_amount_analysis.ipynb` directly in Jupyter

All code in `src/` and `notebooks/` references the dataset's real column
names directly, so no changes are needed once the file is in place.

## A note on `CLAIM_STATUS`

Despite the dataset's "fraud data" name, its only categorical status column
(`CLAIM_STATUS`, values `A`/`D`) does not correlate with any other field —
see Section 6 of the notebook for the check (a model using every available
feature scores 0.50 ROC AUC trying to predict it, i.e. no better than
chance). It's treated here as a non-fraud administrative code rather than a
modeling target, and the project instead predicts `CLAIM_AMOUNT`, which the
data supports well.

## Other files in the original download

The Kaggle archive also includes `employee_data.csv` and `vendor_data.csv`
(agent and vendor contact lookup tables). They were checked and found to add
no claim-level predictive signal, so they aren't used in this analysis.
