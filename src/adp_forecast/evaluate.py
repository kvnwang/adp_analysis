from dataclasses import dataclass
import numpy as np
from .forecast import ForecastModel

@dataclass
class EvaluationResult:
    model: str
    mae: float
    rmse: float
    predictions: int

def walk_forward_predict(values, model: ForecastModel, min_train_size=12):
    predictions, actuals = [], []

    for i in range(min_train_size, len(values)):
        history = values[:i]
        predictions.append(model.predict(history))
        actuals.append(values[i])

    return predictions, actuals

def walk_forward_evaluate(values, model: ForecastModel, min_train_size=12):
    predictions, actuals = walk_forward_predict(values, model, min_train_size=min_train_size)

    errors = np.asarray(predictions) - np.asarray(actuals)
    return EvaluationResult(
        model=model.name,
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        predictions=len(predictions),
    )

def evaluate_models(values, models, min_train_size=12):
    return sorted(
        [walk_forward_evaluate(values, m, min_train_size=min_train_size) for m in models],
        key=lambda r: r.mae,
    )
