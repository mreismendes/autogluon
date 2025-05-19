# DeepVAR Quick Start

This tutorial demonstrates how to train a multivariate time series forecasting model with the DeepVAR estimator from GluonTS using AutoGluon's contrib interface.

```python
import pandas as pd
from autogluon.contrib.timeseries import DeepVARPredictor

# prepare toy dataset
index = pd.date_range("2020-01-01", periods=30, freq="D")
data = pd.DataFrame({
    "target1": range(30),
    "target2": range(30, 60)
}, index=index)

predictor = DeepVARPredictor(prediction_length=5, freq="D", epochs=1)
predictor.fit(data, target_columns=["target1", "target2"])

forecasts = predictor.predict(data, target_columns=["target1", "target2"])
for f in forecasts:
    print(f.mean)
```
