import pytest
from adp_forecast.forecast import NaiveModel, MovingAverageModel, RidgeLagModel, SeasonalNaiveModel, get_models

def test_naive():
    assert NaiveModel().predict([100, 200, 300]) == 300

def test_naive_rejects_empty_history():
    with pytest.raises(ValueError):
        NaiveModel().predict([])

def test_moving_average():
    assert MovingAverageModel(3).predict([100, 200, 300]) == 200

def test_moving_average_rejects_insufficient_history():
    with pytest.raises(ValueError):
        MovingAverageModel(3).predict([100, 200])

def test_ridge_model():
    result = RidgeLagModel(3).predict([100, 120, 140, 160, 180, 200])
    assert isinstance(result, float)

def test_ridge_model_rejects_insufficient_history():
    with pytest.raises(ValueError):
        RidgeLagModel(lags=3).predict([100, 120, 140])

def test_seasonal_naive():
    history = list(range(100, 100 + 24))
    assert SeasonalNaiveModel().predict(history) == history[-12]

def test_seasonal_naive_requires_full_period():
    with pytest.raises(ValueError):
        SeasonalNaiveModel().predict([100, 200, 300])

def test_get_models_uses_given_seasonal_period():
    seasonal = next(m for m in get_models(period=52) if m.name == "seasonal_naive")
    assert seasonal.period == 52
