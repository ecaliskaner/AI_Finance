"""Walk-forward evaluation of a supervised model.

Same discipline as :mod:`ai_finance.research.walkforward`, with the parameter
search replaced by a model fit — and with purging, because a supervised label
reaches forward in time and a plain split would let it reach into the test set.

The result reports three things that have to agree before any of them mean
anything:

1. **Directional accuracy**, with a z-score. 52% on 300 samples is noise; 52% on
   30,000 is a finding. The percentage alone cannot tell them apart.
2. **Information coefficient** — the correlation between predicted and realised
   returns, which unlike accuracy is sensitive to getting the *big* moves right.
3. **What the strategy actually earned**, after costs, against buy-and-hold.

A model can score well on the first two and still lose money, because accuracy
counts every bar equally while the cost model charges per trade. When that
happens it is not a contradiction, it is the whole thesis of this project
showing up in one table.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.backtest.ledger import Trade
from ai_finance.backtest.metrics import BacktestResult
from ai_finance.execution.base import Fill
from ai_finance.features.pipeline import (
    Dataset,
    assert_point_in_time,
    build_dataset,
    directional_accuracy,
    information_coefficient,
)
from ai_finance.research.purged_cv import PurgedSplit, purged_splits
from ai_finance.research.registry import Registry
from ai_finance.research.walkforward import _quiet_risk_engine
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.ml import (
    ModelStrategy,
    PredictionPolicy,
    linear_coefficients,
    make_model,
)

log = logging.getLogger(__name__)


def accuracy_z_score(accuracy: float, n_samples: int) -> float:
    """How many standard errors ``accuracy`` sits above a coin flip.

    The standard error of a proportion near 0.5 is ``sqrt(0.25 / n)``. Anything
    under about 2 is comfortably explained by chance, however pleasing the
    percentage looks.
    """
    if n_samples < 2 or not np.isfinite(accuracy):
        return float("nan")
    return (accuracy - 0.5) / math.sqrt(0.25 / n_samples)


@dataclass(frozen=True)
class FoldReport:
    """One purged train/test fold."""

    split: PurgedSplit
    accuracy: float
    ic: float
    confident_accuracy: float
    confident_fraction: float
    backtest: BacktestResult

    @property
    def window(self) -> str:
        index = self.backtest.equity_curve.index
        return f"{index[0]:%Y-%m-%d} .. {index[-1]:%Y-%m-%d}"


@dataclass
class MLResult:
    """Pooled out-of-sample performance of a model across every fold."""

    model_kind: str
    horizon: int
    symbol: str
    interval: str
    folds: list[FoldReport]
    oos: BacktestResult
    predictions: pd.Series
    actuals: pd.Series
    policy: PredictionPolicy
    coefficients: pd.Series | None = None
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def accuracy(self) -> float:
        """Directional accuracy pooled over every fold."""
        return directional_accuracy(self.predictions.to_numpy(), self.actuals.to_numpy())

    @property
    def accuracy_z(self) -> float:
        return accuracy_z_score(self.accuracy, len(self.predictions))

    @property
    def ic(self) -> float:
        return information_coefficient(self.predictions.to_numpy(), self.actuals.to_numpy())

    @property
    def confident_mask(self) -> np.ndarray:
        """Bars where the prediction was large enough for the policy to act."""
        return np.abs(self.predictions.to_numpy()) >= self.policy.threshold

    @property
    def confident_accuracy(self) -> float:
        """Accuracy on the bars that were actually traded.

        The number that matters. Overall accuracy counts the thousands of bars
        the policy sat out; this counts only the ones where it committed money.
        """
        mask = self.confident_mask
        if not mask.any():
            return float("nan")
        return directional_accuracy(
            self.predictions.to_numpy()[mask], self.actuals.to_numpy()[mask]
        )

    @property
    def confident_fraction(self) -> float:
        mask = self.confident_mask
        return float(mask.mean()) if len(mask) else 0.0

    @property
    def confident_z(self) -> float:
        return accuracy_z_score(self.confident_accuracy, int(self.confident_mask.sum()))

    def to_text(self) -> str:
        lines = [
            f"Model walk-forward: {self.model_kind} on {self.symbol} ({self.interval})",
            "=" * 62,
            f"folds         {self.n_folds}   horizon {self.horizon} bars   "
            f"{len(self.predictions):,} out-of-sample predictions",
            f"threshold     {self.policy.threshold:.3%} predicted move to act "
            f"({self.policy.safety_factor:.1f}x the "
            f"{self.oos.costs.round_trip_cost:.2%} round trip)",
            "",
            "SIGNAL QUALITY",
            "-" * 62,
            f"accuracy      {_pct(self.accuracy)}   z={_num(self.accuracy_z)}  "
            f"({'above' if self.accuracy_z > 2 else 'NOT above'} chance)",
            f"  when traded {_pct(self.confident_accuracy)}   z={_num(self.confident_z)}   "
            f"on {self.confident_fraction:.1%} of bars",
            f"info coef     {_num(self.ic)}   "
            f"({'plausible' if abs(self.ic) < 0.15 else 'SUSPICIOUSLY HIGH - check for a leak'})",
            "",
            "WHAT IT EARNED",
            "-" * 62,
        ]
        lines += self.oos.to_text().splitlines()[2:]

        if self.coefficients is not None and not self.coefficients.empty:
            lines += ["", "Largest standardised coefficients (last fold):"]
            lines += [
                f"  {name:<16} {value:+.5f}" for name, value in self.coefficients.head(6).items()
            ]
        return "\n".join(lines)


def run_ml_walk_forward(
    bars: pd.DataFrame,
    *,
    horizon: int,
    model_kind: str = "ridge",
    train_size: int,
    test_size: int,
    embargo: int = 0,
    anchored: bool = False,
    max_splits: int | None = None,
    symbol: str = "",
    interval: str = "",
    initial_equity: float = 10_000.0,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    policy: PredictionPolicy | None = None,
    registry: Registry | None = None,
    verify_point_in_time: bool = True,
    model_params: dict | None = None,
) -> MLResult:
    """Fit on purged history, predict the next window, roll forward.

    Args:
        horizon: label horizon in bars — the model predicts the return this far
            ahead.
        model_kind: ``"ridge"`` or ``"gbm"``.
        train_size: training rows per fold, before purging.
        test_size: test rows per fold, and the step between folds.
        embargo: extra rows dropped from the end of training, beyond the purge.
        verify_point_in_time: recompute features on truncated data and check
            nothing moved. On by default and worth the seconds it costs — it is
            the only thing standing between a vectorised feature and a leak.

    Raises:
        PointInTimeError: if a feature turns out to depend on the future.
        ValueError: on unknown models or data too short for a single fold.
    """
    costs = costs if costs is not None else CostModel()
    limits = limits if limits is not None else RiskLimits()
    policy = policy if policy is not None else PredictionPolicy(costs)
    periods_per_year = _periods_per_year(bars)

    if verify_point_in_time:
        assert_point_in_time(bars, periods_per_year)

    dataset = build_dataset(bars, horizon=horizon, periods_per_year=periods_per_year)
    if len(dataset) == 0:
        raw = build_dataset(
            bars, horizon=horizon, periods_per_year=periods_per_year, drop_incomplete=False
        )
        always_missing = [name for name in raw.features.columns if raw.features[name].isna().all()]
        detail = (
            f"; feature(s) {always_missing} are missing for every row"
            if always_missing
            else f"; {len(bars):,} bars may be fewer than the longest feature window"
        )
        raise ValueError(f"no usable rows after building features{detail}")

    splits = purged_splits(
        len(dataset),
        train_size=train_size,
        test_size=test_size,
        horizon=horizon,
        embargo=embargo,
        anchored=anchored,
        max_splits=max_splits,
    )
    if not splits:
        raise ValueError(
            f"{len(dataset):,} usable rows is too short for train={train_size} + "
            f"test={test_size} with horizon={horizon}"
        )

    folds: list[FoldReport] = []
    pooled_predictions: list[pd.Series] = []
    pooled_actuals: list[pd.Series] = []
    equity_parts: list[pd.Series] = []
    benchmark_parts: list[pd.Series] = []
    weight_parts: list[pd.Series] = []
    trades: list[Trade] = []
    fills: list[Fill] = []
    running_equity = initial_equity
    running_benchmark = initial_equity
    coefficients: pd.Series | None = None

    with _quiet_risk_engine():
        for split in splits:
            model = make_model(model_kind, **(model_params or {}))
            train_x, train_y = dataset.slice(split.train_positions)
            model.fit(train_x, train_y)
            coefficients = linear_coefficients(model, dataset.feature_names)

            test_x, test_y = dataset.slice(split.test_positions)
            raw = np.asarray(model.predict(test_x), dtype=float)
            times = dataset.index[split.test_positions]
            predictions = pd.Series(raw, index=times, name="prediction")
            actuals = pd.Series(test_y, index=times, name="actual")

            fold_bars = _bars_between(bars, times[0], times[-1])
            if len(fold_bars) < 2:
                log.warning("fold %d covers fewer than 2 bars; skipping", split.index)
                continue

            result = run_backtest(
                fold_bars,
                ModelStrategy(predictions, policy),
                symbol=symbol,
                initial_equity=running_equity,
                costs=costs,
                limits=limits,
            )
            if registry is not None:
                registry.record_result(
                    result,
                    params={
                        "model": model_kind,
                        "horizon": horizon,
                        "train_size": train_size,
                        **(model_params or {}),
                    },
                    interval=interval,
                    kind="test",
                    note=f"ml fold {split.index} out-of-sample",
                )

            confident = np.abs(raw) >= policy.threshold
            folds.append(
                FoldReport(
                    split=split,
                    accuracy=directional_accuracy(raw, test_y),
                    ic=information_coefficient(raw, test_y),
                    confident_accuracy=(
                        directional_accuracy(raw[confident], test_y[confident])
                        if confident.any()
                        else float("nan")
                    ),
                    confident_fraction=float(confident.mean()),
                    backtest=result,
                )
            )
            pooled_predictions.append(predictions)
            pooled_actuals.append(actuals)

            benchmark = result.benchmark_curve / result.initial_equity * running_benchmark
            equity_parts.append(result.equity_curve)
            benchmark_parts.append(benchmark)
            weight_parts.append(result.weight_curve)
            trades.extend(result.trades)
            fills.extend(result.fills)
            running_equity = result.final_equity
            running_benchmark = float(benchmark.iloc[-1])

    if not folds:
        raise ValueError("no fold produced a usable backtest window")

    return MLResult(
        model_kind=model_kind,
        horizon=horizon,
        symbol=symbol,
        interval=interval,
        folds=folds,
        oos=BacktestResult(
            symbol=symbol,
            strategy_name=f"{model_kind} (walk-forward)",
            costs=costs,
            initial_equity=initial_equity,
            equity_curve=pd.concat(equity_parts),
            weight_curve=pd.concat(weight_parts),
            benchmark_curve=pd.concat(benchmark_parts),
            trades=trades,
            fills=fills,
        ),
        predictions=pd.concat(pooled_predictions),
        actuals=pd.concat(pooled_actuals),
        policy=policy,
        coefficients=coefficients,
        feature_names=dataset.feature_names,
    )


def dataset_for(bars: pd.DataFrame, horizon: int) -> Dataset:
    """Convenience wrapper that infers the cadence from the bars themselves."""
    return build_dataset(bars, horizon=horizon, periods_per_year=_periods_per_year(bars))


def _periods_per_year(bars: pd.DataFrame) -> float:
    seconds = (bars["close_time"].iloc[0] - bars["open_time"].iloc[0]).total_seconds() + 0.001
    return 365.0 * 24.0 * 3600.0 / seconds


def _bars_between(bars: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    mask = (bars["close_time"] >= first) & (bars["close_time"] <= last)
    return bars.loc[mask].reset_index(drop=True)


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:.2f}%"


def _num(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2f}"
