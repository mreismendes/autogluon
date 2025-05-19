"""Example of training a DeepVAR model using AutoGluon contrib module."""
import pandas as pd
from autogluon.contrib.timeseries import DeepVARPredictor

# Example synthetic dataset with two target variables
index = pd.date_range("2020-01-01", periods=30, freq="D")
train_data = pd.DataFrame({
    "target1": range(30),
    "target2": range(30, 60)
}, index=index)

predictor = DeepVARPredictor(prediction_length=5, freq="D", epochs=1)
predictor.fit(train_data, target_columns=["target1", "target2"])

forecasts = predictor.predict(train_data, target_columns=["target1", "target2"])
for f in forecasts:
    print(f.mean)
