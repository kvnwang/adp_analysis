import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge

class ForecastModel(ABC):
    name: str

    @abstractmethod
    def predict(self, history: list[float]) -> float:
        raise NotImplementedError

class NaiveModel(ForecastModel):
    name = "naive"

    def predict(self, history):
        if not history:
            raise ValueError("History cannot be empty")
        return float(history[-1])

class MovingAverageModel(ForecastModel):
    name = "moving_average"

    def __init__(self, window=3):
        self.window = window

    def predict(self, history):
        if len(history) < self.window:
            raise ValueError(f"Need at least {self.window} observations")
        return float(np.mean(history[-self.window:]))

class SeasonalNaiveModel(ForecastModel):
    name = "seasonal_naive"

    def __init__(self, period=12):
        self.period = period

    def predict(self, history):
        if len(history) < self.period:
            raise ValueError(f"Need at least {self.period} observations")
        return float(history[-self.period])

class RidgeLagModel(ForecastModel):
    name = "ridge_lag"

    def __init__(self, lags=3, alpha=10.0):
        self.lags = lags
        self.alpha = alpha

    def _features(self, values):
        values = np.asarray(values, dtype=float)
        rows, targets = [], []
        for i in range(self.lags, len(values)):
            recent = values[i-self.lags:i]
            rows.append([
                *recent,
                np.mean(recent),
                recent[-1] - recent[0],
            ])
            targets.append(values[i])
        return np.asarray(rows), np.asarray(targets)

    def predict(self, history):
        if len(history) <= self.lags:
            raise ValueError(f"Need more than {self.lags} observations")
        X, y = self._features(history)
        model = Ridge(alpha=self.alpha)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            model.fit(X, y)

        recent = np.asarray(history[-self.lags:], dtype=float)
        X_next = np.asarray([[
            *recent,
            np.mean(recent),
            recent[-1] - recent[0],
        ]])
        return float(model.predict(X_next)[0])

def get_models(period=12):
    return [
        NaiveModel(),
        MovingAverageModel(window=3),
        SeasonalNaiveModel(period=period),
        RidgeLagModel(lags=3, alpha=10.0),
    ]
