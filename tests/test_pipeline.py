"""Smoke tests for the Bitcoin price regression pipeline.

These tests do not need the full Kaggle dataset. They build a small synthetic
DataFrame with the same columns the notebook uses (Open, VolBTC, VolCurrency ->
Weighted_Price) and exercise the exact PySpark ML steps the notebook performs:
VectorAssembler, MinMaxScaler, LinearRegression, and the corrected metric
evaluation (train metrics from the model summary, test metrics from a
RegressionEvaluator on the held-out split).

Run with:  pytest -q
Requires:  pyspark and a JDK on the PATH.
"""
import random

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("bitcoin-price-tests")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def scaler_and_splits(spark):
    """Build synthetic data and return (scaler_model, train_scaled, test_scaled)."""
    random.seed(42)
    rows = []
    for _ in range(1500):
        open_price = random.uniform(1.0, 100.0)
        vol_btc = random.uniform(0.0, 50.0)
        vol_currency = vol_btc * open_price
        # Weighted_Price closely tracks Open (mirrors the same-minute leakage).
        weighted_price = open_price + random.gauss(0.0, 2.0)
        rows.append((open_price, vol_btc, vol_currency, weighted_price))

    df = spark.createDataFrame(
        rows, ["Open", "VolBTC", "VolCurrency", "Weighted_Price"]
    )
    assembled = VectorAssembler(
        inputCols=["Open", "VolBTC", "VolCurrency"], outputCol="features"
    ).transform(df)

    training_data, test_data = assembled.randomSplit([0.7, 0.3], seed=42)
    scaler_model = MinMaxScaler(
        inputCol="features", outputCol="scaled_features"
    ).fit(training_data)
    return (
        scaler_model,
        scaler_model.transform(training_data),
        scaler_model.transform(test_data),
    )


@pytest.fixture(scope="module")
def scaled_splits(scaler_and_splits):
    _, training_scaled, test_scaled = scaler_and_splits
    return training_scaled, test_scaled


@pytest.fixture(scope="module")
def model_v1(scaled_splits):
    training_scaled, _ = scaled_splits
    lr = LinearRegression(
        featuresCol="scaled_features",
        labelCol="Weighted_Price",
        predictionCol="Predicted_price",
        maxIter=100,
        regParam=0.3,
        elasticNetParam=0.8,
    )
    return lr.fit(training_scaled)


def test_split_is_non_empty(scaled_splits):
    training_scaled, test_scaled = scaled_splits
    assert training_scaled.count() > 0
    assert test_scaled.count() > 0


def test_predictions_have_expected_columns(model_v1, scaled_splits):
    _, test_scaled = scaled_splits
    preds = model_v1.transform(test_scaled)
    assert "Predicted_price" in preds.columns
    assert "Weighted_Price" in preds.columns


def test_train_metrics_come_from_summary(model_v1):
    """model.summary.* are TRAINING-set metrics (the bug the audit flagged was
    labeling these as 'test' metrics)."""
    assert model_v1.summary.meanAbsoluteError >= 0.0
    assert model_v1.summary.rootMeanSquaredError >= 0.0


def test_test_metrics_use_regression_evaluator(model_v1, scaled_splits):
    """The corrected code evaluates the TEST split with a RegressionEvaluator,
    not the training summary. Both MAE and RMSE must be computable on the
    held-out predictions."""
    _, test_scaled = scaled_splits
    test_preds = model_v1.transform(test_scaled)

    mae = RegressionEvaluator(
        labelCol="Weighted_Price", predictionCol="Predicted_price", metricName="mae"
    ).evaluate(test_preds)
    rmse = RegressionEvaluator(
        labelCol="Weighted_Price", predictionCol="Predicted_price", metricName="rmse"
    ).evaluate(test_preds)

    assert mae >= 0.0
    assert rmse >= 0.0
    # With near-linear synthetic data the fit should be reasonable.
    assert rmse < 50.0


def test_prediction_tracks_open(model_v1, scaler_and_splits, spark):
    """Because Weighted_Price is essentially Open, a prediction for a new row
    should land near its Open value. This is the leakage documented in the
    README Limitations section, asserted here so the behavior is explicit."""
    scaler_model, _, _ = scaler_and_splits
    row = spark.createDataFrame(
        [(50.0, 10.0, 500.0)], ["Open", "VolBTC", "VolCurrency"]
    )
    assembled = VectorAssembler(
        inputCols=["Open", "VolBTC", "VolCurrency"], outputCol="features"
    ).transform(row)
    scaled = scaler_model.transform(assembled)
    predicted = model_v1.transform(scaled).collect()[0]["Predicted_price"]
    # Prediction should be within 20% of the Open price (it essentially mirrors it).
    assert abs(predicted - 50.0) < 10.0
