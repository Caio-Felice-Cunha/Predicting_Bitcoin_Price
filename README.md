# Predicting Bitcoin Price
Predicting Cryptocurrency Quotes in Real Time with PySpark and Machine Learning<br>
![image](https://user-images.githubusercontent.com/111542025/234733071-3e1869ad-eb1d-416a-b276-3755500ee4f1.png)<br>

## This is the 1st version

## Business Problem
> Data source: the data was extracted from the [website](https://bitcoincharts.com/charts/bitstampUSD#rg60ztgSzm1g10zm2g25zv)

Bitcoin is the oldest and most well-known cryptocurrency, first released as open source in 2009 by the anonymous Satoshi Nakamoto.<br>

Bitcoin serves as a decentralized medium of digital exchange, with transactions verified and recorded on a distributed public ledger (the Blockchain) without the need for a record-keeping authority or central intermediary.<br>

Transaction blocks contain an SHA-256 cryptographic hash of previous transaction blocks and are therefore "chained" together, serving as an immutable record of all transactions that have ever taken place.<br>

We will use historical Bitcoin quote data from 2011 to 2021. As the year 2022 was quite atypical for Bitcoin, we chose not to use data from that year. You can include data from 2022 if you wish and retrain the machine learning model.<br>

## Solution Strategy
As our objective is to predict the value of a certain asset, we will use linear regression, creating different versions of the model with different algorithms and going through the entire Machine Learning process from end to end.
* Step 01: Preparing the Spark Environment;
* Step 02: Loading the Data and Data Understandment;
* Step 03: Data Wrangling with SparkSQL;
* Step 04: Exploratory Analysis;
* Step 05: Data Normalization;
* Step 06: Machine Learning - Model Version 01 - LinearRegression;
* Step 07: Model Evaluation - Model Version 01 - LinearRegression;
* Step 08: Machine Learning - Model Version 02 - LinearRegression with Hyperparameter Optimization;
* Step 09: Model Evaluation - Model Version 02 - LinearRegression with Hyperparameter Optimization;
* Step 10: Real Time Forecasts.


## Next Steps
* Improve hyperparameters and data classification.

## Disclaimer
This project was largely done in the Data Science Academy, Big Data Real-Time Analytics with Python and Spark course (part of the Data Scientist training)
