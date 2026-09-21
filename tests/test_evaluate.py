from adp_forecast.evaluate import walk_forward_evaluate, walk_forward_predict, evaluate_models
from adp_forecast.forecast import NaiveModel, SeasonalNaiveModel


def test_walk_forward_respects_min_train_size():
    values = list(range(1, 21))
    result = walk_forward_evaluate(values, NaiveModel(), min_train_size=15)
    assert result.predictions == 20 - 15


def test_evaluate_models_forwards_min_train_size_for_long_period_models():
    values = list(range(100, 100 + 60))
    results = evaluate_models(values, [SeasonalNaiveModel(period=52)], min_train_size=52)
    assert results[0].predictions == 60 - 52


def test_walk_forward_predict_returns_predictions_and_actuals():
    values = [100, 200, 300, 400, 500]
    predictions, actuals = walk_forward_predict(values, NaiveModel(), min_train_size=3)
    assert actuals == [400, 500]
    assert predictions == [300, 400]
