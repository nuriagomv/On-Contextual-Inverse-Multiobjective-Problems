"""Turn an inverse solution into weights, directions, and out-of-sample gaps."""

import math

import numpy as np
import ot
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

from cimo.models import forward_problem, linear_criteria


def decompose_estimates(theta_estimates):
    """Split each estimated matrix into a weight and a unit-norm direction.

    ``w_k = ||theta_tilde_k||_F`` and ``theta_k = theta_tilde_k / w_k``.
    A zero matrix is returned unchanged and contributes weight 0.

    Parameters
    ----------
    theta_estimates :
        Iterable of matrices in criterion order, typically
        ``[theta.X for theta in theta_tilde.values()]``.

    Returns
    -------
    weights : ndarray of shape ``(K,)``
    theta : dict ``k -> ndarray``
    """
    theta = {}
    weights = []
    for k, estimate in enumerate(theta_estimates):
        weight = float(np.linalg.norm(estimate, "fro"))
        weights.append(weight)
        theta[k] = estimate / weight if weight != 0.0 else estimate
        print(f"theta_hat[{k}] =\n{theta[k]}")
    weights = np.array(weights)
    print(f"w_hat = {weights}")
    return weights, theta


def evaluate_estimates(weights_hat, theta_hat, weights_true, theta_true, contexts_test, n_test):
    """Compare estimated preferences and estimated criterion rows on the test set.

    Preference error is the 1-Wasserstein distance between ``weights_true`` and
    ``weights_hat``, viewed as masses on the criterion indices ``0 .. K-1``.
    With two criteria this is the closed-form distance on the real line. With
    more criteria the ground cost is ``1 - I`` (zero on the diagonal, one
    elsewhere), solved as a transportation problem.

    Criterion error is the cosine similarity of ``x @ theta_true`` and
    ``x @ theta_hat`` on each test context. The reported triple is
    ``(mean, standard deviation with divisor n, median)`` across test observations.
    Both weights should already sum to one; they are not renormalized here.

    Returns
    -------
    emd : float
    cosine_by_criterion : dict ``k -> (mean, std, median)``
    """
    if len(weights_true) == 2:
        bins = np.arange(len(weights_true))
        emd = wasserstein_distance(bins, bins, weights_true, weights_hat)
    else:
        ground_cost = 1.0 - np.eye(len(weights_true))
        emd = ot.emd2(weights_true, weights_hat, ground_cost)

    cosine_by_criterion = {}
    for k in range(len(theta_true)):
        scores = []
        for n in range(n_test):
            true_row = (contexts_test[n][k] @ theta_true[k]).reshape(1, -1)
            hat_row = (contexts_test[n][k] @ theta_hat[k]).reshape(1, -1)
            scores.append(cosine_similarity(true_row, hat_row)[0, 0])
        cosine_by_criterion[k] = (
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.median(scores)),
        )
    return emd, cosine_by_criterion


def forward_objective_value(context, weights, theta, gamma, decision, ideal_points):
    """Scalarized cost of ``decision``, as excess over the criteria-wise ideals.

    ``l_1`` sums the excesses, ``l_2`` takes their Euclidean norm, and
    ``l_inf`` takes the maximum. The forward *solver* only implements ``l_1``
    and ``l_2``; ``l_inf`` is available here so a stored decision can still be
    scored under that scalarization.
    """
    costs = linear_criteria(context, weights, theta)
    excesses = [costs[k] @ (decision - ideal_points[k]) for k in range(len(ideal_points))]
    if gamma == "l_1":
        return float(np.sum(excesses))
    if gamma == "l_inf":
        return float(np.max(excesses))
    if gamma == "l_2":
        return float(np.sqrt(np.sum(np.square(excesses))))
    raise ValueError(f"Unsupported scalarization {gamma!r}.")


def fractional_gap(reference, candidate):
    """``candidate / reference``, with the usual 0/0 and x/0 conventions.

    In this project ``reference`` is the cost of the observed decision and
    ``candidate`` is the cost of the decision reoptimized under the same
    parameters. For a minimization problem and positive costs, a ratio below
    one means the reoptimized decision improves on the observation.
    """
    if reference != 0:
        return candidate / reference
    if candidate == 0:
        return 1.0
    return np.inf


def assess_recovered_solutions(
    n_train,
    n_test,
    gamma,
    matrices_train,
    matrices_test,
    constraint_rhs,
    weights_hat,
    theta_hat,
    weights_true,
    theta_true,
    contexts_train,
    decisions_train,
    ideals_train,
    contexts_test,
    decisions_test,
    ideals_test,
):
    """Re-solve the forward problem at the estimate and score the new decisions.

    Two different questions are answered with the same pair of decisions
    (the observation, and the forward reoptimization under the estimate):

    Consistency
        Under the *estimated* parameters, do the two decisions have the same
        scalarized cost? They should, up to solver tolerance, when the
        observation is optimal for the estimate. Ideal points used here are
        the ones returned by the reoptimization.
    Suboptimality
        Under the *true* parameters, what is the fractional gap between the
        cost of the reoptimized decision and the cost of the observation?
        Ideal points used here are the true criteria-wise optima.

    The count of exactly equal decision vectors is printed and is not the
    consistency test: two optimal solutions can differ and still share a cost.

    Returns
    -------
    consistent : bool
        True when the mean in-sample cost difference is within ``1e-5`` of 0
        and every training forward problem was solved.
    gaps_in, gaps_out :
        Lists of ``(cost of observation, cost of reoptimized decision)``
        under the true parameters.
    """
    n_decisions = len(decisions_train[0])
    n_criteria = len(theta_hat)
    n_equal = 0
    consistencies = []
    gaps_in = []

    for n in range(n_train):
        reoptimized, ideals_hat, _ = forward_problem(
            contexts_train[n],
            weights_hat,
            theta_hat,
            matrices_train[n],
            constraint_rhs,
            gamma,
            n_decisions,
            n_criteria,
        )
        if reoptimized is None:
            print(f"Training observation {n}: forward problem with the estimate is infeasible.")
            continue
        n_equal += int(np.array_equal(reoptimized, decisions_train[n]))
        cost_observed, cost_reoptimized = _pair_costs(
            contexts_train[n], weights_hat, theta_hat, gamma, decisions_train[n], reoptimized, list(ideals_hat.values())
        )
        # Equal optimal costs under the estimate. Sign is observed minus reoptimized.
        consistencies.append(round(cost_observed - cost_reoptimized, 5))
        gaps_in.append(
            _pair_costs(
                contexts_train[n],
                weights_true,
                theta_true,
                gamma,
                decisions_train[n],
                reoptimized,
                list(ideals_train[n].values()),
            )
        )

    gaps_out = []
    for n in range(n_test):
        reoptimized, _, _ = forward_problem(
            contexts_test[n],
            weights_hat,
            theta_hat,
            matrices_test[n],
            constraint_rhs,
            gamma,
            n_decisions,
            n_criteria,
        )
        if reoptimized is None:
            print(f"Test observation {n}: forward problem with the estimate is infeasible.")
            continue
        gaps_out.append(
            _pair_costs(
                contexts_test[n],
                weights_true,
                theta_true,
                gamma,
                decisions_test[n],
                reoptimized,
                list(ideals_test[n].values()),
            )
        )

    print(f"Exact decision matches: {n_equal} of {n_train}.")
    consistent = len(consistencies) == n_train and math.isclose(
        float(np.mean(consistencies)), 0.0, abs_tol=1e-5
    )
    print(f"In-sample consistency of objective values: {consistent}.")
    print(f"Median fractional gap in sample: {_median_gap(gaps_in)}")
    print(f"Median fractional gap out of sample: {_median_gap(gaps_out)}")
    return consistent, gaps_in, gaps_out


def _pair_costs(context, weights, theta, gamma, decision_observed, decision_reoptimized, ideal_points):
    """Costs of the observation and of the reoptimized decision, in that order."""
    observed = forward_objective_value(context, weights, theta, gamma, decision_observed, ideal_points)
    reoptimized = forward_objective_value(context, weights, theta, gamma, decision_reoptimized, ideal_points)
    return observed, reoptimized


def _median_gap(pairs):
    """Median of ``fractional_gap`` over ``(observed cost, reoptimized cost)`` pairs."""
    if not pairs:
        return np.nan
    return float(np.nanmedian([fractional_gap(observed, reoptimized) for observed, reoptimized in pairs]))
