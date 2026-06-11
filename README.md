# Predicting Bitcoin Price

Same-minute Bitcoin price regression with PySpark and Spark MLlib (LinearRegression).
![image](https://user-images.githubusercontent.com/111542025/234733071-3e1869ad-eb1d-416a-b276-3755500ee4f1.png)

## This is the 1st version

## What this project does

It takes minute-level Bitcoin market data (open price and trading volumes) and trains a
linear regression model to estimate the `Weighted_Price` for that same minute. It walks
through the full Spark ML workflow end to end: load, clean, explore, normalize, train,
tune with cross-validation, and score new rows.

## Business Problem

Bitcoin is the oldest and most well-known cryptocurrency, first released as open source in
2009 by the anonymous Satoshi Nakamoto. It serves as a decentralized medium of digital
exchange, with transactions verified and recorded on a distributed public ledger (the
blockchain) without a central record-keeping authority.

The goal here is to estimate a Bitcoin price point from contemporaneous market features
using a simple, interpretable model, and to practice the end-to-end PySpark ML pipeline on
a large (millions of rows) time series.

We use historical Bitcoin quote data from 2011 to 2021. The year 2022 was atypical for
Bitcoin, so it was left out. You can include later years and retrain if you wish.

## Data

The dataset is the public bitstampUSD 1-minute Bitcoin history (Timestamp, Open, High, Low,
Close, Volume_(BTC), Volume_(Currency), Weighted_Price), roughly 4.86 million rows from
2011-12-31 onward.

The original `bitcoincharts.com` source is no longer reachable. Use the equivalent Kaggle
dataset instead: "Bitcoin Historical Data" by mczielinski
(https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data). Download the
bitstampUSD 1-minute CSV and place it at `Data/dataset.csv` (this is the path the notebook
reads in the load cell). The `Data/` folder is gitignored, so it will not be committed.

## How to run

Requirements: Python 3.8 to 3.10 and a JDK (8, 11, or 17) on your PATH (Spark needs Java).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# place the dataset at Data/dataset.csv (see the Data section)
jupyter notebook "Predicting Bitcoin Price.ipynb"
```

Spark runs in local mode, so no cluster is needed. Run the cells top to bottom.

## Solution Strategy

The target is a continuous price, so the model is linear regression, built in two versions
through the full Machine Learning process end to end.
* Step 01: Preparing the Spark Environment
* Step 02: Loading the Data and Data Understanding
* Step 03: Data Wrangling with Spark SQL
* Step 04: Exploratory Analysis
* Step 05: Data Normalization
* Step 06: Machine Learning, Model Version 01, LinearRegression
* Step 07: Model Evaluation, Model Version 01
* Step 08: Machine Learning, Model Version 02, LinearRegression with hyperparameter tuning (CrossValidator)
* Step 09: Model Evaluation, Model Version 02
* Step 10: Scoring new data

## Results

Reported on the held-out 30% test split (`randomSplit([.7, .3])`):

| Model | Test RMSE | Notes |
|-------|-----------|-------|
| v2 (LinearRegression + CrossValidator) | 10.4263 | best model from hyperparameter tuning |

Training-fit reference (these are training-set metrics from the Spark `LinearRegression`
training summary, not test performance): v1 training MAE 5.0306, v1 training RMSE 11.339,
v2 training MAE 3.3923.

These numbers are the values stored in the notebook outputs. Note that `randomSplit` is not
seeded, so re-running the notebook will produce slightly different figures. Set a seed in
the split cell if you need reproducible numbers.

## Limitations

Read these before treating the model as a price predictor.

* Target leakage from `Open`. The model predicts the same-minute `Weighted_Price` from the
  same-minute `Open` price (plus volumes). `Open` and `Weighted_Price` for the same minute
  are nearly identical, so the prediction essentially mirrors `Open`. In the scoring step,
  Open 20546.29 maps to a predicted 20538.51, and Open 21620.85 maps to 21612.50. The low
  error reflects this leakage, not genuine forecasting skill.
* Not a forecast and not real time. There is no streaming component and no future horizon.
  The "Scoring new data" step scores a couple of hand-entered rows in batch. To make this a
  real predictor you would use lagged features (predict minute t from data up to t-1).
* Random split on a time series. The train/test split is random rather than chronological,
  which leaks information across time. A time-ordered split is more honest for a time series.

A natural next version is a lagged-feature model with a chronological split, which would
give a far more realistic (and higher) error but a genuinely useful target.

## Models

`models/model_v1/` and `models/model_v2/` contain the saved Spark MLlib LinearRegression
artifacts (saved with Spark 3.3.2). Loading them requires the same feature pipeline used in
the notebook; they are included as deliverables of the training run, not as a standalone
predictor.

## Next Steps

* Add a lagged-feature version with a chronological train/test split (genuine forecasting).
* Improve hyperparameters and feature engineering.

## Disclaimer

A good part of this project was done in the Data Science Academy "Big Data Real-Time
Analytics with Python and Spark" course (part of the Data Scientist training).
