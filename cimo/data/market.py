"""Market data and the portfolio instances used in the numerical study.

Decision of date t
    z = (z_1, ..., z_S, s) with S the number of stocks.
    z_i >= 0, sum_i z_i = 1, and s >= volatility_{t,i} * z_i.
    s is an epigraph variable: paying its cost minimizes an upper bound on
    volatility-weighted holdings.

Context of date t predicts the next date's return (and volume, if requested).
The feasible set of observation t uses the volatility recorded on date t.
The last calendar date has no successor, so it is dropped from the sample.
"""

from dataclasses import dataclass
import pickle
import random

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

from cimo.models import forward_problem
from cimo.paths import CACHE_DIR


# Liquid US names. A run draws ``n_stocks`` of them, under the given seed.
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "BRK-B", "JNJ", "V", "JPM", "WMT",
    "NVDA", "HD", "PG", "UNH", "MA", "DIS", "PYPL", "NFLX", "PEP", "KO",
]

# Market series that can enter the context. A run draws ``n_market`` of them.
MARKET_UNIVERSE = [
    "^DJI",   # Dow Jones
    "^VIX",   # implied volatility index
    "^GSPC",  # S&P 500
    "CL=F",   # crude oil future
    "GC=F",   # gold future
    "^TNX",   # 10-year Treasury yield
]


@dataclass
class PortfolioSample:
    """One simulated inverse-optimization dataset, already solved in the forward direction.

    Training and test dicts are keyed by ``0 .. n-1`` inside each split.
    Constraint matrices are split the same way, so index ``n`` in a split is
    the ``A`` that belongs to that split's context ``n``.
    """

    market_frame: pd.DataFrame
    stocks: list
    market_features: list
    criterion_names: list
    context_dimensions: np.ndarray
    n_decisions: int
    n_constraints: int
    preference_weights: np.ndarray
    theta: dict
    matrices_train: list
    matrices_test: list
    constraint_rhs: np.ndarray
    contexts_train: dict
    contexts_test: dict
    decisions_train: dict
    decisions_test: dict
    ideals_train: dict
    ideals_test: dict

    @property
    def n_criteria(self):
        """Number of linear criteria, including the risk epigraph."""
        return len(self.theta)

    @property
    def n_context_entries(self):
        """Total number of context coordinates across criteria."""
        return int(self.context_dimensions.sum())


def build_portfolio_instances(
    seed,
    n_stocks,
    n_market,
    n_train,
    n_test,
    gamma,
    start,
    end,
    what_in_context,
    volume_in_objective,
    sparse_model,
    preference_weights=None,
):
    """Download (or reload) prices, fit the surrogates, and solve every forward problem.

    Parameters
    ----------
    what_in_context :
        ``"market"`` uses the sampled market series.
        ``"past_t-1"`` uses the lagged value of the criterion being predicted.
        ``"all"`` uses both. Every design includes an intercept.
    volume_in_objective :
        When True, next-day volume is a criterion of its own, placed between
        return and risk. The default preference vector ``[0.3, 0.7]`` is then
        the wrong length and ``preference_weights`` must be passed.
    sparse_model :
        Fit each asset from the market series plus that asset's own lag, and
        leave the other assets' lags at zero. Ignored for ``what_in_context
        == "market"``, which has no per-asset lag.
    preference_weights :
        ``w`` of the ground-truth forward model. Defaults to ``[0.3, 0.7]``
        (return, risk), which sums to one.
    """
    frame, stocks, market_features = load_market_data(seed, n_stocks, n_market, start, end)
    theta, contexts, matrices, rhs, criterion_names = fit_linear_surrogates(
        frame, stocks, market_features, what_in_context, volume_in_objective, sparse_model
    )
    n_available = next(iter(contexts.values())).shape[0]
    if n_train + n_test > n_available:
        raise ValueError(
            f"Requested {n_train} + {n_test} observations but only {n_available} "
            "dates have both a context and a next-day target."
        )

    n_criteria = len(theta)
    if preference_weights is None:
        if n_criteria != 2:
            raise ValueError(
                "The default preference vector [0.3, 0.7] is for return and risk only. "
                f"This sample has criteria {criterion_names}; pass preference_weights."
            )
        preference_weights = np.array([0.3, 0.7])
    preference_weights = np.asarray(preference_weights, dtype=float)
    if preference_weights.shape != (n_criteria,):
        raise ValueError(
            f"preference_weights has shape {preference_weights.shape}, expected {(n_criteria,)}."
        )

    n_constraints, n_decisions = matrices[0].shape
    context_dimensions = np.array([contexts[k].shape[1] for k in range(n_criteria)])
    print("Solving the forward problem on the training and test dates.")
    decisions, ideals, split_contexts = _solve_forward_sample(
        contexts,
        matrices,
        rhs,
        preference_weights,
        theta,
        gamma,
        n_decisions,
        n_criteria,
        n_train,
        n_test,
    )

    return PortfolioSample(
        market_frame=frame,
        stocks=list(stocks),
        market_features=list(market_features),
        criterion_names=criterion_names,
        context_dimensions=context_dimensions,
        n_decisions=n_decisions,
        n_constraints=n_constraints,
        preference_weights=preference_weights,
        theta=theta,
        matrices_train=matrices[:n_train],
        matrices_test=matrices[n_train : n_train + n_test],
        constraint_rhs=rhs,
        contexts_train=split_contexts["train"],
        contexts_test=split_contexts["test"],
        decisions_train=decisions["train"],
        decisions_test=decisions["test"],
        ideals_train=ideals["train"],
        ideals_test=ideals["test"],
    )


def load_market_data(seed, n_stocks, n_market, start, end):
    """Return ``(frame, stocks, market_features)``, using ``cache/`` when possible.

    The cache key is the seed and the download window, not the fitted model.
    A corrupt cache file is deleted and the download is repeated.
    """
    if n_stocks > len(STOCK_UNIVERSE):
        raise ValueError(f"n_stocks={n_stocks} exceeds the universe of {len(STOCK_UNIVERSE)}.")
    if n_market > len(MARKET_UNIVERSE):
        raise ValueError(f"n_market={n_market} exceeds the universe of {len(MARKET_UNIVERSE)}.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"market_s{seed}_n{n_stocks}_m{n_market}_{start}_{end}.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                return pickle.load(handle)
        except (pickle.UnpicklingError, EOFError):
            cache_path.unlink()

    # stdlib Random, not NumPy: ticker draws must not consume the stream that
    # later builds the estimation prior.
    sampler = random.Random(seed)
    stocks = sampler.sample(STOCK_UNIVERSE, n_stocks)
    market_features = sampler.sample(MARKET_UNIVERSE, n_market)
    frame = _download_feature_frame(stocks, market_features, start, end)

    with cache_path.open("wb") as handle:
        pickle.dump((frame, stocks, market_features), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return frame, stocks, market_features


def fit_linear_surrogates(frame, stocks, market_features, what_in_context, volume_in_objective, sparse_model):
    """Fit one linear map per criterion and build the date-wise feasible sets.

    Predictive regressions are multiplied by ``-1 / ||.||_F``. The forward
    model minimizes linear cost, while the investor wants to maximize predicted
    return (and volume, when that criterion is on). The sign flip is that
    change of orientation. The risk criterion is built directly as a cost on
    the slack variable and is already unit norm, so it is not flipped again.

    The last column of every predictive matrix is the coefficient on the risk
    slack. It is left at zero, then preserved by the normalization.
    """
    if what_in_context not in ("market", "past_t-1", "all"):
        raise ValueError(f"Unknown context design {what_in_context!r}.")

    n_assets = len(stocks)
    n_decisions = n_assets + 1  # asset weights plus the risk slack
    n_market = len(market_features)
    targets = ["_Return", "_Volume"] if volume_in_objective else ["_Return"]
    names = ["return", "volume"] if volume_in_objective else ["return"]

    theta = {}
    contexts = {}
    for k, suffix in enumerate(targets):
        n_features = _n_raw_features(what_in_context, n_market, n_assets)
        # +1 row for the intercept. Columns: one per asset, plus the slack.
        coefficient = np.zeros((n_features + 1, n_decisions))
        for asset_index, stock in enumerate(stocks):
            full_columns, sparse_columns = _regressor_columns(
                what_in_context, market_features, stocks, stock, suffix
            )
            design = _design_matrix(frame, full_columns)
            target = frame.loc[:, f"{stock}{suffix}"].to_numpy()[1:]
            contexts[k] = design
            coefficient = _write_asset_coefficients(
                coefficient,
                design,
                _design_matrix(frame, sparse_columns),
                target,
                asset_index,
                what_in_context,
                n_market,
                sparse_model,
            )
        theta[k] = -coefficient / np.linalg.norm(coefficient, "fro")

    # Risk criterion: cost vector (0, ..., 0, 1) on (weights, slack), context = 1.
    risk = len(theta)
    theta[risk] = np.hstack((np.zeros(n_assets), np.ones(1))).reshape(1, n_decisions)
    n_rows = contexts[risk - 1].shape[0]
    contexts[risk] = np.ones((n_rows, 1))
    names.append("risk")

    matrices, rhs = build_constraint_system(frame, stocks)
    # The supervised design drops the last date (no next-day target). Drop the
    # matching constraint matrix so index t refers to the same date everywhere.
    matrices = matrices[:n_rows]
    return theta, contexts, matrices, rhs, names


def build_constraint_system(frame, stocks):
    """Date-wise inequalities ``A_t z <= b`` for the portfolio forward problem.

    Rows of ``A_t``, in order:

    * ``-z_i <= 0`` for each asset (long only). The slack is free here.
    * ``volatility_{t,i} z_i - s <= 0`` (epigraph of holding-level volatility).
    * ``sum_i z_i <= 1`` and ``-sum_i z_i <= -1`` (budget equality).

    ``b = (0_S, 0_S, 1, -1)`` does not depend on the date. ``A_t`` does, through
    the volatility diagonal.
    """
    n_assets = len(stocks)
    volatility_columns = [f"{stock}_Volatility" for stock in stocks]
    matrices = []
    for date in frame.index:
        volatility = frame.loc[date, volatility_columns].to_numpy(dtype=float)
        long_only = np.hstack((-np.eye(n_assets), np.zeros((n_assets, 1))))
        risk_epigraph = np.hstack((np.diag(volatility), -np.ones((n_assets, 1))))
        budget_upper = np.hstack((np.ones((1, n_assets)), np.zeros((1, 1))))
        budget_lower = np.hstack((-np.ones((1, n_assets)), np.zeros((1, 1))))
        matrices.append(np.vstack((long_only, risk_epigraph, budget_upper, budget_lower)))
    rhs = np.hstack((np.zeros(n_assets), np.zeros(n_assets), np.ones(1), -np.ones(1)))
    return matrices, rhs


def _download_feature_frame(stocks, market_features, start, end):
    """Assemble return, volatility, volume, and same-day market levels.

    Column layout follows current yfinance: a MultiIndex ``(field, ticker)``.
    Return is the percent change of the close. Volatility is the high-low
    range divided by the midpoint of that range. Both are aligned on the
    intersection of trading dates, and the first return row is dropped because
    ``pct_change`` has nothing to difference against.
    """
    prices = yf.download(stocks, start=start, end=end).reset_index()
    if prices.shape[0] == 0:
        raise RuntimeError(f"No stock prices returned for {stocks} between {start} and {end}.")

    prices["DATE"] = prices.loc[:, ("Date", "")].astype("datetime64[ns]")
    prices = prices.set_index("DATE")

    returns = prices.loc[:, ("Close", slice(None))].copy()
    returns.columns = [f"{ticker}_Return" for _, ticker in returns.columns]
    returns = returns.pct_change().iloc[1:] * 100.0
    returns = returns.dropna()

    high = prices.loc[:, ("High", slice(None))]
    low = prices.loc[:, ("Low", slice(None))]
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    volatility = pd.DataFrame(
        (high_values - low_values) / ((high_values + low_values) / 2.0),
        index=high.index,
        columns=[f"{ticker}_Volatility" for _, ticker in high.columns],
    )

    volume = prices.loc[:, ("Volume", slice(None))].copy()
    volume.columns = [f"{ticker}_Volume" for _, ticker in volume.columns]

    frame = volume.join(volatility, how="inner").join(returns, how="inner")

    market = yf.download(market_features, start=start, end=end).reset_index()
    if market.shape[0] == 0:
        raise RuntimeError(
            f"No market data returned for {market_features} between {start} and {end}."
        )
    market["DATE"] = market.loc[:, ("Date", "")].astype("datetime64[ns]")
    market = market.set_index("DATE")
    levels = market.loc[:, ("Open", slice(None))].copy()
    levels.columns = [ticker for _, ticker in levels.columns]
    levels = levels.dropna()

    frame = levels.join(frame, how="inner")
    if frame.empty:
        raise RuntimeError("Stock and market calendars have an empty intersection.")
    return frame


def _n_raw_features(what_in_context, n_market, n_assets):
    """Width of the design before the intercept column is appended."""
    if what_in_context == "all":
        return n_market + n_assets
    if what_in_context == "past_t-1":
        return n_assets
    return n_market


def _regressor_columns(what_in_context, market_features, stocks, stock, suffix):
    """Full design columns, and the columns used when the sparse fit is on.

    The sparse design keeps the market block (if any) and the lagged criterion
    of ``stock`` only. Other assets' lags are filled with structural zeros
    after the regression, rather than being estimated and thresholded.
    """
    own_lag = [f"{stock}{suffix}"]
    all_lags = [f"{name}{suffix}" for name in stocks]
    if what_in_context == "all":
        return market_features + all_lags, market_features + own_lag
    if what_in_context == "past_t-1":
        return all_lags, own_lag
    return list(market_features), list(market_features)


def _design_matrix(frame, columns):
    """Rows ``0 .. T-2`` of ``columns``, with a trailing intercept column.

    Row t is paired by the caller with the target at t+1.
    """
    features = frame.loc[:, columns].to_numpy(dtype=float)[:-1]
    intercept = np.ones((features.shape[0], 1))
    return np.hstack((features, intercept))


def _write_asset_coefficients(
    coefficient,
    design,
    sparse_design,
    target,
    asset_index,
    what_in_context,
    n_market,
    sparse_model,
):
    """Write the ordinary-least-squares row block of one asset into ``coefficient``.

    ``fit_intercept=False`` because the intercept is already a column of the
    design. In the sparse ``"all"`` layout, market coefficients occupy rows
    ``0 .. n_market-1``, the own lag occupies row ``n_market + asset_index``,
    and the intercept occupies the last row.
    """
    regression = LinearRegression(fit_intercept=False)
    use_sparse = sparse_model and what_in_context != "market"
    if not use_sparse:
        regression.fit(design, target)
        coefficient[:, asset_index] = regression.coef_
        return coefficient

    regression.fit(sparse_design, target)
    if what_in_context == "all":
        coefficient[:n_market, asset_index] = regression.coef_[:n_market]
        coefficient[n_market + asset_index, asset_index] = regression.coef_[n_market]
        coefficient[-1, asset_index] = regression.coef_[-1]
    else:
        # "past_t-1": the sparse design is (own lag, intercept).
        coefficient[asset_index, asset_index] = regression.coef_[0]
        coefficient[-1, asset_index] = regression.coef_[-1]
    return coefficient


def _solve_forward_sample(
    contexts,
    matrices,
    rhs,
    preference_weights,
    theta,
    gamma,
    n_decisions,
    n_criteria,
    n_train,
    n_test,
):
    """Solve the ground-truth forward problem and split the series in time order.

    The first ``n_train`` dates are training. The next ``n_test`` dates are test.
    There is no shuffle: both splits stay contiguous blocks of the price path.
    """
    decisions = {"train": {}, "test": {}}
    ideals = {"train": {}, "test": {}}
    split_contexts = {"train": {}, "test": {}}
    for global_index in range(n_train + n_test):
        context = [contexts[k][global_index, :] for k in range(n_criteria)]
        decision, ideal, _ = forward_problem(
            context,
            preference_weights,
            theta,
            matrices[global_index],
            rhs,
            gamma,
            n_decisions,
            n_criteria,
        )
        if global_index < n_train:
            split, local_index = "train", global_index
            print(f"train {local_index}: z* = {np.round(decision, 2) if decision is not None else None}")
        else:
            split, local_index = "test", global_index - n_train
        decisions[split][local_index] = decision
        ideals[split][local_index] = ideal
        split_contexts[split][local_index] = context
    return decisions, ideals, split_contexts
