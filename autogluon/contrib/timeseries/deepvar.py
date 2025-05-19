"""Wrapper for training multivariate DeepVAR models using GluonTS."""
from typing import Optional

import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import MultivariateEvaluator


class DeepVARPredictor:
    """Simple interface for multivariate time series forecasting with DeepVAR."""

    def __init__(self, prediction_length: int, freq: str = "D", epochs: int = 5):
        self.prediction_length = prediction_length
        self.freq = freq
        self.epochs = epochs
        self.estimator: Optional[DeepVAREstimator] = None
        self.predictor = None

    def fit(self, data: pd.DataFrame, target_columns):
        """Train DeepVAR model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing a datetime index and target columns.
        target_columns : list
            Names of the columns to forecast.
        """
        dataset = PandasDataset(
            data,
            target=target_columns,
            freq=self.freq,
        )
        trainer = Trainer(epochs=self.epochs)
        self.estimator = DeepVAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            trainer=trainer,
        )
        self.predictor = self.estimator.train(dataset)
        return self

    def predict(self, data: pd.DataFrame, target_columns):
        """Generate forecasts for the given data."""
        if self.predictor is None:
            raise RuntimeError("The model must be fit before calling predict().")
        dataset = PandasDataset(data, target=target_columns, freq=self.freq)
        forecasts = list(self.predictor.predict(dataset))
        return forecasts

    def evaluate(self, data: pd.DataFrame, target_columns):
        """Evaluate the predictor on holdout data."""
        if self.predictor is None:
            raise RuntimeError("The model must be fit before calling evaluate().")
        dataset = PandasDataset(data, target=target_columns, freq=self.freq)
        evaluator = MultivariateEvaluator(target_agg_funcs={})
        agg_metrics, _ = evaluator(self.predictor.predict(dataset), dataset)
        return agg_metrics
