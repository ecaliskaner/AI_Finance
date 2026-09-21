"""The experiment registry: a count of how many times you rolled the dice.

If you test 200 strategy variants and report the best one, you have not found a
strategy. You have found the highest of 200 draws from a distribution centred
near zero, and it will look excellent right up until it meets new data.

The defence is not cleverness, it is bookkeeping. Every backtest this project
runs is appended here — strategy, parameters, window, costs, results — and the
count of trials is used to compute what the *best* result would have looked like
if nothing worked at all. A Sharpe of 1.5 after 5 trials is interesting. The
same 1.5 after 500 trials is the null hypothesis behaving normally.

The log is append-only JSON Lines: one self-describing record per line, readable
by anything, and impossible to accidentally rewrite when a later run disagrees
with an earlier one.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd

from ai_finance.backtest.metrics import BacktestResult
from ai_finance.config import data_dir

#: Euler-Mascheroni constant, from the expected-maximum formula below.
EULER_MASCHERONI = 0.5772156649015329

_NORMAL = NormalDist()


@dataclass(frozen=True)
class Experiment:
    """One recorded backtest."""

    recorded_at: str
    strategy: str
    params: dict[str, Any]
    symbol: str
    interval: str
    start: str
    end: str
    n_bars: int
    kind: str
    costs: dict[str, float] = field(default_factory=dict)
    total_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    sharpe: float = float("nan")
    max_drawdown: float = 0.0
    n_trades: int = 0
    cost_drag: float = 0.0
    note: str = ""


class Registry:
    """Append-only log of every backtest run.

    Args:
        path: where the log lives. Defaults to ``<data_dir>/experiments.jsonl``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else data_dir() / "experiments.jsonl"

    def record(self, experiment: Experiment) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _sanitize(asdict(experiment))
        with self.path.open("a", encoding="utf-8") as handle:
            # allow_nan=False turns any non-finite value that slipped past
            # _sanitize into an exception instead of a bare NaN literal, which
            # Python writes happily but which is not valid JSON and which every
            # other parser rejects.
            handle.write(json.dumps(payload, allow_nan=False, default=_json_safe) + "\n")

    def record_result(
        self,
        result: BacktestResult,
        *,
        params: dict[str, Any],
        interval: str,
        kind: str = "adhoc",
        note: str = "",
    ) -> Experiment:
        """Record a :class:`BacktestResult`.

        Args:
            kind: ``"train"`` for a parameter-selection run, ``"test"`` for an
                out-of-sample evaluation, ``"adhoc"`` for anything else. Only
                ``train`` runs count as trials when deflating a Sharpe: the
                out-of-sample evaluations are the answer, not the search.
        """
        experiment = Experiment(
            recorded_at=pd.Timestamp.now("UTC").isoformat(),
            strategy=result.strategy_name,
            params=dict(params),
            symbol=result.symbol,
            interval=interval,
            start=result.equity_curve.index[0].isoformat(),
            end=result.equity_curve.index[-1].isoformat(),
            n_bars=len(result.equity_curve),
            kind=kind,
            costs={
                "fee_rate": result.costs.fee_rate,
                "half_spread_bps": result.costs.half_spread_bps,
                "slippage_bps": result.costs.slippage_bps,
            },
            total_return=result.total_return,
            benchmark_return=result.benchmark_return,
            excess_return=result.excess_return,
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            n_trades=result.n_trades,
            cost_drag=result.cost_drag,
            note=note,
        )
        self.record(experiment)
        return experiment

    def all(self) -> pd.DataFrame:
        """Every recorded experiment, oldest first."""
        if not self.path.exists():
            return pd.DataFrame(columns=[f.name for f in Experiment.__dataclass_fields__.values()])
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)

    def count(self, kind: str | None = None) -> int:
        frame = self.all()
        if frame.empty:
            return 0
        if kind is None:
            return len(frame)
        return int((frame["kind"] == kind).sum())

    def report(self, n_observations: int | None = None) -> str:
        """Human-readable summary, including the multiple-testing warning."""
        frame = self.all()
        if frame.empty:
            return "No experiments recorded yet."

        lines = [
            "Experiment registry",
            "=" * 46,
            f"file          {self.path}",
            f"experiments   {len(frame):,}",
        ]
        for kind, count in frame["kind"].value_counts().items():
            lines.append(f"  {kind:<11} {count:,}")

        by_strategy = frame.groupby("strategy")["excess_return"].agg(["count", "max"])
        lines += ["", "by strategy (count, best excess return):"]
        lines += [
            f"  {name:<22} {int(row['count']):>5,}  {row['max'] * 100:+.2f}%"
            for name, row in by_strategy.iterrows()
        ]

        trials = self.count("train") or len(frame)
        if n_observations:
            threshold = expected_max_sharpe(trials, n_observations)
            lines += [
                "",
                f"MULTIPLE TESTING: {trials:,} search runs recorded.",
                f"With that many tries on {n_observations:,} observations, the *best*",
                f"strategy would be expected to show a Sharpe of {threshold:.2f} even if",
                "none of them had any edge at all. Treat anything below that as noise.",
            ]
        return "\n".join(lines)


def expected_max_sharpe(
    n_trials: int, n_observations: int, periods_per_year: float = 365.0
) -> float:
    """Annualised Sharpe the best of ``n_trials`` worthless strategies would show.

    Uses the standard approximation for the expected maximum of ``N`` draws from
    a standard normal::

        E[max] ~= (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e))

    with ``g`` the Euler-Mascheroni constant, scaled by the standard error of a
    Sharpe estimate, ``sqrt(periods_per_year / n_observations)``.

    Concretely: 200 variants tested on 1,000 daily observations produce an
    expected best Sharpe of about 1.7 **from luck alone**. If your winner scores
    1.4, you have not found anything.

    Assumes independent trials and normally distributed returns. Neither holds
    exactly — grid-search variants are correlated, which makes the real
    threshold somewhat lower, and returns have fat tails, which pushes it back
    up. It is a sanity check, not a p-value.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_observations < 2:
        raise ValueError("n_observations must be at least 2")
    if n_trials == 1:
        return 0.0

    expected_max_z = (1.0 - EULER_MASCHERONI) * _NORMAL.inv_cdf(
        1.0 - 1.0 / n_trials
    ) + EULER_MASCHERONI * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    standard_error = math.sqrt(periods_per_year / n_observations)
    return expected_max_z * standard_error


def _sanitize(value: Any) -> Any:
    """Replace non-finite floats with ``None``, recursively.

    ``nan`` and ``inf`` are ordinary Python floats, so ``json.dumps`` serialises
    them as the bare literals ``NaN`` and ``Infinity`` rather than handing them
    to a ``default=`` hook. Those literals are not JSON. Python reads them back,
    almost nothing else does, and a Sharpe that could not be computed is
    genuinely absent rather than a number — so ``null`` is also the more honest
    encoding.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    """Last resort for types :func:`_sanitize` left alone."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
