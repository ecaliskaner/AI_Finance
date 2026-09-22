"""Turning a prediction into a position.

`docs/PLAN.md` Phase 3 insists on a split that is easy to skip and expensive to
skip: **the model predicts, a separate deterministic rule decides.**

The tempting alternative is one model that outputs a position directly. It is
also nearly impossible to debug. When an end-to-end model loses money you have a
weight vector and a shrug. When a predictor loses money you can ask two separate
questions with two separate answers — was the forecast wrong, or was the
forecast fine and the sizing rule bad? — and fix whichever it was.

The rule itself is where the cost arithmetic finally bites. A model can be right
about direction most of the time and still lose money, because being right about
a 5 bp move is worthless when the round trip costs 22. So :class:`PredictionPolicy`
refuses to trade unless the *predicted* move clears the cost by a margin.

There is no learning in this module. It is arithmetic, and it is meant to stay
that way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.strategy.base import MarketState, Signal
from ai_finance.strategy.baselines import _TargetTracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionPolicy:
    """Converts a predicted return into a target weight.

    Args:
        costs: the cost model the backtest is charging. The threshold is derived
            from it, so raising costs automatically makes the policy pickier
            rather than silently unprofitable.
        safety_factor: required edge as a multiple of the round-trip cost.
            1.0 means "trade whenever the prediction covers costs exactly",
            which is a coin flip after estimation error. The default demands
            half again as much.
        full_size_return: predicted return at which the position reaches
            ``max_weight``. Predictions between the threshold and this scale in
            linearly, so a marginal forecast gets a small position.
        max_weight: largest fraction of equity this policy will ever ask for.
            The risk engine may cut it further; it never raises it.
        allow_short: whether a negative prediction may open a short. Spot crypto
            cannot short, so this is off by default and a bearish forecast
            simply means "hold nothing".
    """

    costs: CostModel
    safety_factor: float = 1.5
    full_size_return: float = 0.02
    max_weight: float = 1.0
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.safety_factor <= 0:
            raise ValueError("safety_factor must be positive")
        if self.full_size_return <= 0:
            raise ValueError("full_size_return must be positive")
        if not 0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")

    @property
    def threshold(self) -> float:
        """Smallest predicted move worth acting on.

        With default costs (22 bp round trip) and the default safety factor,
        this is 33 bp — which at a 4-hour horizon is a move of roughly a third
        of one standard deviation. Most bars will not clear it, and that is the
        intended behaviour, not a bug.
        """
        return self.costs.round_trip_cost * self.safety_factor

    def weight(self, predicted_return: float) -> float:
        """Target weight for ``predicted_return``. Zero means stay flat."""
        if not np.isfinite(predicted_return):
            return 0.0
        if abs(predicted_return) < self.threshold:
            return 0.0
        if predicted_return < 0 and not self.allow_short:
            return 0.0

        size = min(self.max_weight, abs(predicted_return) / self.full_size_return)
        return size if predicted_return > 0 else -size

    def explain(self, predicted_return: float) -> str:
        """Why the policy did what it did, for the signal's ``reason`` field."""
        if not np.isfinite(predicted_return):
            return "no prediction available"
        if abs(predicted_return) < self.threshold:
            return (
                f"predicted {predicted_return:+.3%} < {self.threshold:.3%} needed "
                f"to clear costs; staying flat"
            )
        if predicted_return < 0 and not self.allow_short:
            return f"predicted {predicted_return:+.3%} but shorting is disabled"
        return f"predicted {predicted_return:+.3%} vs {self.threshold:.3%} threshold"


class ModelStrategy(_TargetTracker):
    """Trades a precomputed series of predictions through a policy.

    Predictions are supplied rather than computed here, keyed by the bar's
    ``close_time``. That is deliberate: a model fitted on a vectorised feature
    matrix and a model re-implemented bar by bar inside the event loop will
    eventually disagree, and the disagreement will be silent. Passing in a
    prediction series removes the possibility entirely. What guarantees the
    predictions are honest is that the walk-forward runner only ever fits on
    rows that closed before the window being predicted — and that the features
    themselves are proven point-in-time by
    :func:`ai_finance.features.pipeline.assert_point_in_time`.

    A bar with no prediction produces no signal, which the engine reads as "no
    change". The position simply persists, which is what a live system would do
    if a model run were skipped.
    """

    name = "ml"

    def __init__(
        self,
        predictions: pd.Series,
        policy: PredictionPolicy,
        *,
        rebalance_threshold: float = 0.05,
    ) -> None:
        super().__init__(threshold=rebalance_threshold)
        self.policy = policy
        self._predictions: dict[Any, float] = {
            key: float(value) for key, value in predictions.items()
        }

    def on_bar(self, state: MarketState) -> Signal | None:
        predicted = self._predictions.get(state.timestamp)
        if predicted is None:
            return None
        return self.emit(
            state.symbol,
            self.policy.weight(predicted),
            self.policy.explain(predicted),
        )


def make_model(kind: str, **params: Any):
    """Build an unfitted estimator.

    ``ridge`` first, always. A linear model on standardised features is
    interpretable — the coefficients say which features the edge came from —
    and on data this noisy it is frequently no worse than anything fancier.
    Reach for ``gbm`` only when ridge has established a baseline to beat.

    ``gbm`` is scikit-learn's ``HistGradientBoostingRegressor``, which is the
    same histogram-based algorithm LightGBM popularised. Using it avoids a
    dependency for no loss at this scale.

    Deliberately absent: anything with hidden layers. A few thousand rows of
    data with a signal-to-noise ratio this low is not where a neural network
    earns its keep; it is where one memorises the training set.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        options = {"alpha": 1.0, **params}
        return make_pipeline(StandardScaler(), Ridge(**options))
    if kind == "gbm":
        options = {
            "max_depth": 3,
            "max_iter": 200,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
            "random_state": 0,
            **params,
        }
        return HistGradientBoostingRegressor(**options)
    raise ValueError(f"unknown model {kind!r}; expected 'ridge' or 'gbm'")


MODEL_KINDS = ("ridge", "gbm")


def linear_coefficients(model, feature_names: list[str]) -> pd.Series | None:
    """Standardised coefficients of a linear model, largest magnitude first.

    Returns ``None`` for models that have no such thing. This is the cheapest
    route to the pre-live checklist's hardest item — being able to say in a
    paragraph *why* the strategy makes money.
    """
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is None:
        return None
    series = pd.Series(np.ravel(coefficients), index=feature_names)
    return series.reindex(series.abs().sort_values(ascending=False).index)
