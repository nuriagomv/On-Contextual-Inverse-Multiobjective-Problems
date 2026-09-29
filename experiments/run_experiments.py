"""Grid runner for the portfolio inverse-optimization study.

Run from anywhere:

    python experiments/run_experiments.py

Each completed design point is appended to ``results/results.pkl``. A new
process starts that file over; it does not resume a partial grid.
"""

import pickle
import sys
import time
from pathlib import Path

import gurobipy as gp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cimo.data.market import build_portfolio_instances
from cimo.evaluation import (
    assess_recovered_solutions,
    decompose_estimates,
    evaluate_estimates,
    fractional_gap,
)
from cimo.models import inverse_problem
from cimo.paths import RESULTS_DIR


def run_inverse_optimization(
    seed,
    n_train,
    n_test,
    n_stocks,
    n_market,
    objective,
    time_limit=1800,
    solve_to="optimality",
    start="2023-01-01",
    end="2024-12-31",
    enforce_portfolio_support=True,
    what_in_context="past_t-1",
    volume_in_objective=False,
    gamma="l_2",
    tolerance=0.0,
    cluster_contexts=False,
    sparse_model=True,
):
    """Build one sample, test the true parameters, and estimate ``theta_tilde``.

    The prior is the true normalized direction, plus uniform noise on
    ``[-0.25, 0.25]``, times an independent Bernoulli(1/2) mask. It is a
    synthetic informative prior for the simulation, not something estimated
    from decisions. Draws use ``numpy.random.RandomState(seed)`` so the prior
    matches the legacy global NumPy stream and does not depend on whether the
    price cache was hit.

    ``cluster_contexts`` replaces the training sample by the medoids of a
    5-cluster K-medoids partition of the concatenated contexts, and estimates
    from those medoids only. This is the path previously controlled by
    ``noise``. Perturbing the observed decisions themselves is not implemented.

    Returns
    -------
    result, summary
        ``result`` keeps the sample and the estimated matrices.
        ``summary`` is a flat dict of scalars for the spreadsheet.
    """
    if enforce_portfolio_support and volume_in_objective:
        raise ValueError(
            "Portfolio support constraints assume criterion 0 is return and "
            "criterion 1 is risk. They do not apply when volume sits between them."
        )

    sample = build_portfolio_instances(
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
    )
    if enforce_portfolio_support and sample.criterion_names != ["return", "risk"]:
        raise ValueError(
            "Portfolio support constraints assume criteria ['return', 'risk'], "
            f"got {sample.criterion_names}."
        )

    _print_ground_truth(sample)
    # Feasibility of the ground truth against the KKT system, without the
    # support equalities. Those equalities are side information for estimation,
    # not part of the claim that the true parameters explain the decisions.
    inverse_problem(
        n_train,
        sample.contexts_train,
        sample.decisions_train,
        sample.matrices_train,
        sample.constraint_rhs,
        sample.context_dimensions,
        sample.n_decisions,
        sample.n_constraints,
        sample.n_criteria,
        gamma,
        objective=None,
        time_limit=time_limit,
        solve_to="optimality",
        fixed_parameters=(sample.preference_weights, sample.theta),
        tolerance=tolerance,
        enforce_portfolio_support=False,
    )

    prior = _noisy_prior(sample.theta, sample.n_criteria, seed)
    if cluster_contexts:
        n_fit, contexts_fit, decisions_fit, matrices_fit = _medoid_training_set(
            sample, n_train, seed
        )
    else:
        n_fit = n_train
        contexts_fit = sample.contexts_train
        decisions_fit = sample.decisions_train
        matrices_fit = sample.matrices_train

    started = time.perf_counter()
    model, theta_tilde = inverse_problem(
        n_fit,
        contexts_fit,
        decisions_fit,
        matrices_fit,
        sample.constraint_rhs,
        sample.context_dimensions,
        sample.n_decisions,
        sample.n_constraints,
        sample.n_criteria,
        gamma,
        objective,
        time_limit,
        solve_to,
        prior=prior,
        tolerance=tolerance,
        enforce_portfolio_support=enforce_portfolio_support,
    )
    runtime_seconds = time.perf_counter() - started
    if model.SolCount == 0:
        raise RuntimeError(f"Inverse model returned no solution (status {model.status}).")

    weights_hat, theta_hat = decompose_estimates([variable.X for variable in theta_tilde.values()])
    emd, cosine = evaluate_estimates(
        weights_hat, theta_hat, sample.preference_weights, sample.theta, sample.contexts_test, n_test
    )
    consistent, gaps_in, gaps_out = assess_recovered_solutions(
        n_train,
        n_test,
        gamma,
        sample.matrices_train,
        sample.matrices_test,
        sample.constraint_rhs,
        weights_hat,
        theta_hat,
        sample.preference_weights,
        sample.theta,
        sample.contexts_train,
        sample.decisions_train,
        sample.ideals_train,
        sample.contexts_test,
        sample.decisions_test,
        sample.ideals_test,
    )
    summary = _summarize(
        seed,
        n_train,
        n_test,
        n_stocks,
        n_market,
        objective,
        gamma,
        tolerance,
        what_in_context,
        sparse_model,
        enforce_portfolio_support,
        runtime_seconds,
        _mip_gap(model),
        sample,
        emd,
        cosine,
        consistent,
        gaps_in,
        gaps_out,
        weights_hat,
        theta_hat,
    )
    result = {
        "summary": summary,
        "prior": prior,
        "weights_hat": weights_hat,
        "theta_hat": theta_hat,
        "gaps_in": gaps_in,
        "gaps_out": gaps_out,
        "sample": sample,
    }
    return result, summary


def _noisy_prior(theta, n_criteria, seed):
    """True direction, corrupted by uniform noise and a random 0/1 mask.

    ``RandomState`` (not ``Generator``) reproduces the legacy ``np.random``
    stream used when this prior was drawn from the global NumPy RNG.
    """
    rng = np.random.RandomState(seed)
    prior = {}
    for k in range(n_criteria):
        noise = rng.uniform(-0.25, 0.25, size=theta[k].shape)
        mask = rng.randint(0, 2, size=theta[k].shape)
        prior[k] = (theta[k] + noise) * mask
    return prior


def _medoid_training_set(sample, n_train, seed, n_clusters=5):
    """Keep one medoid per cluster, with constraint matrices aligned to them.

    Medoid indices are data-point indices, so each retained context is an
    actual training observation. The returned dicts are reindexed from zero.
    """
    from sklearn_extra.cluster import KMedoids

    rows = np.vstack(
        [np.concatenate(tuple(sample.contexts_train[n])) for n in range(n_train)]
    )
    clustering = KMedoids(n_clusters=min(n_clusters, n_train), random_state=seed).fit(rows)
    indices = sorted({int(index) for index in clustering.medoid_indices_})
    contexts, decisions, matrices = {}, {}, []
    for local, index in enumerate(indices):
        contexts[local] = sample.contexts_train[index]
        decisions[local] = sample.decisions_train[index]
        matrices.append(sample.matrices_train[index])
    print(f"Estimating from {len(indices)} context medoids.")
    return len(indices), contexts, decisions, matrices


def _print_ground_truth(sample):
    """Print the simulated preferences and the share of structural zeros."""
    print("True preference weights:", sample.preference_weights)
    for name, matrix in zip(sample.criterion_names, sample.theta.values()):
        print(f"True theta[{name}] =\n{matrix}")


def _summarize(
    seed,
    n_train,
    n_test,
    n_stocks,
    n_market,
    objective,
    gamma,
    tolerance,
    what_in_context,
    sparse_model,
    enforce_portfolio_support,
    runtime_seconds,
    mip_gap,
    sample,
    emd,
    cosine,
    consistent,
    gaps_in,
    gaps_out,
    weights_hat,
    theta_hat,
):
    """Scalar record of one design point. Heavy arrays stay in ``result``."""
    # Last column is the risk slack, which is structurally zero for the return
    # surrogate. The percentage ignores it.
    return_block = sample.theta[0][:, :-1]
    percentage_zeros_true = float(np.sum(return_block == 0) / return_block.size)
    gap_in = _gap_moments(gaps_in)
    gap_out = _gap_moments(gaps_out)
    return {
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_stocks": n_stocks,
        "n_market": n_market,
        "gamma": gamma,
        "tolerance": tolerance,
        "objective": str(objective),
        "context": what_in_context,
        "sparse_model": sparse_model,
        "portfolio_support": enforce_portfolio_support,
        "runtime_seconds": runtime_seconds,
        "mip_gap": mip_gap,
        "percentage_zeros_true": percentage_zeros_true,
        "n_unique_decisions": _n_unique(sample.decisions_train.values()),
        "n_unique_contexts": _n_unique(sample.contexts_train[n][0] for n in range(n_train)),
        "emd": emd,
        "cosine_similarity": cosine,
        "consistent": consistent,
        "gap_in_mean": gap_in[0],
        "gap_in_std": gap_in[1],
        "gap_in_median": gap_in[2],
        "gap_out_mean": gap_out[0],
        "gap_out_std": gap_out[1],
        "gap_out_median": gap_out[2],
        # Exact-zero counts of w_hat[0] * matrix, minus one per decision coordinate.
        # Kept so spreadsheets stay comparable with earlier dumps of this study.
        "zeros_true_scaled": int(np.sum(weights_hat[0] * sample.theta[0] == 0) - sample.n_decisions),
        "zeros_estimated": int(np.sum(weights_hat[0] * theta_hat[0] == 0) - sample.n_decisions),
        "stocks": ",".join(sample.stocks),
        "market_features": ",".join(sample.market_features),
    }


def _mip_gap(model):
    """Relative MIP gap, or 0 when a continuous model was solved to optimality.

    ``min_prior`` alone has no integer variables, and Gurobi only exposes
    ``MIPGap`` on mixed-integer models.
    """
    if model.IsMIP:
        return model.MIPGap
    if model.status == gp.GRB.OPTIMAL:
        return 0.0
    return float("nan")


def _gap_moments(pairs):
    """Mean, standard deviation (divisor n), and median of the fractional gaps."""
    if not pairs:
        return (np.nan, np.nan, np.nan)
    ratios = np.array([fractional_gap(observed, reoptimized) for observed, reoptimized in pairs], dtype=float)
    return (float(np.nanmean(ratios)), float(np.nanstd(ratios)), float(np.nanmedian(ratios)))


def _n_unique(rows):
    """Count distinct vectors under exact equality, including floating-point identity."""
    return len({tuple(np.asarray(row).tolist()) for row in rows})


def main():
    """Active grid. Widen the lists below to repeat the larger design."""
    np.set_printoptions(suppress=True, precision=3)

    # Previously explored and left out of the default run:
    # train sizes 25, 50, 100, 150, 200, and the objective {0.8: min_prior, 0.2: max_sparsity}.
    seeds = [123, 456, 789]
    train_sizes = [1]
    stock_counts = [5]
    objectives = [
        {1.0: "min_prior"},
        {0.9: "min_prior", 0.1: "max_sparsity"},
    ]
    n_test = 100
    n_market = 6
    gamma = "l_2"
    tolerance = 0.005
    what_in_context = "past_t-1"
    sparse_model = True
    enforce_portfolio_support = True

    results = []
    summaries = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "results.pkl"

    for seed in seeds:
        for n_train in train_sizes:
            for n_stocks in stock_counts:
                for objective in objectives:
                    print(
                        f"\nSetting seed={seed} n_train={n_train} n_test={n_test} "
                        f"n_stocks={n_stocks} objective={objective}"
                    )
                    result, summary = run_inverse_optimization(
                        seed,
                        n_train,
                        n_test,
                        n_stocks,
                        n_market,
                        objective,
                        gamma=gamma,
                        tolerance=tolerance,
                        what_in_context=what_in_context,
                        enforce_portfolio_support=enforce_portfolio_support,
                        sparse_model=sparse_model,
                    )
                    results.append(result)
                    summaries.append(summary)
                    with out_path.open("wb") as handle:
                        pickle.dump((results, summaries), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(
                        "Result:",
                        f"emd={summary['emd']}",
                        f"consistent={summary['consistent']}",
                        f"gap_out_median={summary['gap_out_median']}",
                    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
