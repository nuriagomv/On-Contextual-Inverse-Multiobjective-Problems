"""Forward and inverse models for contextual multiobjective optimization.

Notation used throughout this package
-------------------------------------
Decision
    z in R^{d_z}.
Context of observation n, criterion k
    x_{n,k} in R^{d_{x,k}}.
Criterion parameters
    theta_k in R^{d_{x,k} x d_z}.
Preference weights
    w in R^K, nonnegative.
Linear criterion
    c_k(x) = w_k * x_k @ theta_k, a row in R^{d_z}.
Feasible set
    Z(A, b) = {z : A z <= b}.

The inverse model does not estimate w and theta separately. It estimates the
product theta_tilde_k = w_k * theta_k, then splits that product into a scale
and a direction. See ``cimo.evaluation.decompose_estimates``.

Scalarizations currently built by the solvers
----------------------------------------------
``l_1``
    Minimize the sum of the linear criteria.
``l_2``
    Minimize the Euclidean distance, in criterion space, to the ideal point
    (the vector of criteria-wise optima).

``l_inf`` is not implemented. An earlier formulation, and a binary formulation
that forced each context feature to load on at most one criterion, were
removed from this file; both were fully commented out and untested.
"""

import gurobipy as gp
import numpy as np


def linear_criteria(context, weights, theta):
    """Build the K criterion rows evaluated at one context.

    Parameters
    ----------
    context :
        Sequence of length K. Entry k has shape ``(d_{x,k},)``.
    weights :
        Sequence of K nonnegative scalars.
    theta :
        Dict ``k -> array of shape (d_{x,k}, d_z)`` with keys ``0 .. K-1``.

    Returns
    -------
    ndarray of shape ``(K, d_z)``
        Row k is ``weights[k] * context[k] @ theta[k]``.
    """
    rows = [weights[k] * context[k] @ theta[k] for k in range(len(theta))]
    return np.vstack(rows)


def forward_problem(
    context,
    weights,
    theta,
    constraint_matrix,
    constraint_rhs,
    gamma,
    n_decisions,
    n_criteria,
):
    """Solve the contextual forward (compromise) problem at one observation.

    Criteria-wise ideals ``v_k`` solve ``min c_k @ v`` over ``A v <= b``.
    They are always computed, because the evaluation metrics report excess
    cost over the ideal point.

    For ``gamma == "l_1"`` the program minimizes ``sum_k c_k @ z``. Subtracting
    the ideal values ``c_k @ v_k`` would not change the argmin, so those
    constants are omitted here and added back only when the cost is reported.

    For ``gamma == "l_2"`` the program minimizes ``t >= 0`` subject to
    ``sum_k (c_k @ (z - v_k))^2 <= t^2``.

    Returns
    -------
    decision, ideals, model
        ``decision`` is the optimal ``z``. ``ideals`` maps criterion index to
        ``v_k``. All three are ``None`` when a criteria-wise problem is
        infeasible or the compromise problem itself is not optimal.
    """
    if gamma not in ("l_1", "l_2"):
        raise ValueError(f"Unsupported scalarization {gamma!r}. Use 'l_1' or 'l_2'.")

    costs = linear_criteria(context, weights, theta)
    ideals = {}
    for k in range(n_criteria):
        ideal = _minimize_linear(costs[k], constraint_matrix, constraint_rhs, n_decisions)
        if ideal is None:
            print(f"Criteria-wise problem {k} is infeasible.")
            return None, None, None
        ideals[k] = ideal

    model = gp.Model()
    model.setParam(gp.GRB.Param.OutputFlag, 0)
    decision = model.addMVar(
        shape=n_decisions,
        lb=-float("inf"),
        ub=float("inf"),
        vtype=gp.GRB.CONTINUOUS,
        name="z",
    )
    model.addConstr(constraint_matrix @ decision <= constraint_rhs, name="z_in_Z")

    if gamma == "l_1":
        model.setObjective(
            gp.quicksum(costs[k] @ decision for k in range(n_criteria)),
            sense=gp.GRB.MINIMIZE,
        )
    else:
        # Epigraph of the Euclidean distance to the ideal point.
        epigraph = model.addMVar(shape=1, lb=0.0, ub=float("inf"), name="epigraph")
        excesses = [costs[k] @ (decision - ideals[k]) for k in range(n_criteria)]
        model.addConstr(
            gp.quicksum(excess * excess for excess in excesses) <= epigraph**2,
            name="l2_epigraph",
        )
        model.setObjective(epigraph, sense=gp.GRB.MINIMIZE)

    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        print(f"Forward problem not solved (status {model.status}).")
        return None, None, model
    return decision.X, ideals, model


def inverse_problem(
    n_observations,
    contexts,
    decisions,
    constraint_matrices,
    constraint_rhs,
    context_dimensions,
    n_decisions,
    n_constraints,
    n_criteria,
    gamma,
    objective,
    time_limit,
    solve_to,
    prior=None,
    fixed_parameters=None,
    tolerance=0.0,
    enforce_portfolio_support=False,
):
    """Recover ``theta_tilde_k ≈ w_k * theta_k`` from observed optimal decisions.

    Parameters
    ----------
    contexts, decisions :
        Dicts keyed by ``0 .. n_observations-1``. ``contexts[n]`` is a list of
        K context vectors. ``decisions[n]`` is the observed optimal ``z``.
    constraint_matrices :
        List of length at least ``n_observations``. Matrix ``n`` is the ``A``
        of observation ``n``. Every matrix is ``n_constraints x n_decisions``.
        The right-hand side ``b`` is shared.
    context_dimensions :
        ``d_{x,k}`` for each criterion.
    gamma :
        ``"l_1"`` or ``"l_2"``. Selects the stationarity system below.
    objective :
        Map from a positive multiplier to ``"min_prior"`` and/or
        ``"max_sparsity"``. ``None`` leaves a zero objective, which is the
        right setting for a pure feasibility check.
    solve_to :
        ``"optimality"`` or ``"feasibility"``. The latter stops at the first
        feasible point (Gurobi ``SolutionLimit=1``).
    prior :
        Dict of matrices on the scale of the *direction* ``theta_k``, not of
        ``w_k * theta_k``. The prior term compares ``theta_tilde_k`` with
        ``norms[k] * prior[k]``.
    fixed_parameters :
        Optional ``(weights, theta)``. Pins every entry of ``theta_tilde`` to
        ``weights[k] * theta[k]`` so the solve only tests whether that point
        satisfies the inverse constraints.
    tolerance :
        Absolute slack on complementary slackness and on the sum of the
        Frobenius norms. Squared-norm *definitions* stay equalities. For
        ``l_1``, stationarity stays an equality as well.
    enforce_portfolio_support :
        Encode the known zeros of the portfolio surrogates: the return
        criterion does not price the risk-slack coordinate, and the risk
        criterion prices only that coordinate. This assumes criterion 0 is
        return and criterion 1 is risk.

    Optimality system
    -----------------
    The observed decision is data, so complementary slackness is linear in
    the dual multiplier ``mu >= 0`` of ``A_n z <= b``:

        mu_i * (A_n,i @ z_n - b_i) = 0.

    Stationarity depends on the scalarization.

    ``l_1``. With ``c_n = sum_k x_{n,k} @ theta_tilde_k``,

        -c_n = A_n^T mu_n.

    ``l_2``. ``y_{n,k} <= 0`` is the dual multiplier of the criteria-wise
    problem, linked to the estimated criterion by dual feasibility
    ``A_n^T y_{n,k} = x_{n,k} @ theta_tilde_k``. The compromise stationarity
    condition then weights each criterion by its excess over the ideal value.

    The normalization ``sum_k ||theta_tilde_k||_F = 1`` removes the scale
    invariance of the linear criteria. Bounds ``[-1, 1]`` are implied by that
    norm and are imposed explicitly to help the solver.

    Returns
    -------
    model, theta_tilde
        ``theta_tilde`` maps each criterion to its Gurobi matrix variable.
        Read a solution from ``theta_tilde[k].X`` when ``model.SolCount > 0``.
    """
    if gamma not in ("l_1", "l_2"):
        raise ValueError(f"Unsupported scalarization {gamma!r}. Use 'l_1' or 'l_2'.")
    if solve_to not in ("optimality", "feasibility"):
        raise ValueError(f"solve_to must be 'optimality' or 'feasibility', got {solve_to!r}.")
    if constraint_matrices[0].shape != (n_constraints, n_decisions):
        raise ValueError(
            "constraint_matrices[0].shape is "
            f"{constraint_matrices[0].shape}, expected {(n_constraints, n_decisions)}."
        )

    model = _new_model(
        time_limit=time_limit,
        stop_at_first_feasible=(solve_to == "feasibility"),
        quiet=(fixed_parameters is not None),
    )
    theta_tilde = _add_parameter_variables(
        model, context_dimensions, n_decisions, n_criteria, fixed_parameters
    )
    if enforce_portfolio_support:
        _add_portfolio_support(model, theta_tilde, n_criteria)
    norms = _add_normalization(model, theta_tilde, context_dimensions, n_decisions, n_criteria, tolerance)

    # c_{n,k} = x_{n,k} @ theta_tilde_k. The weight is already inside theta_tilde.
    criterion_rows = np.array(
        [
            [contexts[n][k] @ theta_tilde[k] for k in range(n_criteria)]
            for n in range(n_observations)
        ]
    )
    _add_optimality_conditions(
        model,
        criterion_rows,
        theta_tilde,
        contexts,
        decisions,
        constraint_matrices,
        constraint_rhs,
        n_observations,
        n_decisions,
        n_constraints,
        n_criteria,
        gamma,
        tolerance,
    )
    _add_estimation_objective(
        model,
        theta_tilde,
        norms,
        objective,
        prior,
        context_dimensions,
        n_decisions,
        n_criteria,
    )

    model.optimize()
    if fixed_parameters is not None:
        _report_fixed_parameter_feasibility(model)
    return model, theta_tilde


def _minimize_linear(cost, constraint_matrix, constraint_rhs, n_decisions):
    """Solve ``min cost @ v`` subject to ``A v <= b``. Return ``None`` if not optimal."""
    model = gp.Model()
    model.setParam(gp.GRB.Param.OutputFlag, 0)
    variable = model.addMVar(
        shape=n_decisions,
        lb=-float("inf"),
        ub=float("inf"),
        vtype=gp.GRB.CONTINUOUS,
        name="v",
    )
    model.addConstr(constraint_matrix @ variable <= constraint_rhs)
    model.setObjective(cost @ variable, sense=gp.GRB.MINIMIZE)
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        return None
    return variable.X


def _new_model(time_limit, stop_at_first_feasible, quiet):
    """Empty Gurobi model with the nonconvex quadratic setting this formulation needs.

    ``NonConvex=2`` is required: the norm identities, the prior objective, and
    the ``l_2`` stationarity conditions are nonconvex quadratic.
    """
    model = gp.Model()
    model.setParam(gp.GRB.Param.NonConvex, 2)
    model.setParam(gp.GRB.Param.TimeLimit, time_limit)
    if quiet:
        model.setParam(gp.GRB.Param.OutputFlag, 0)
    if stop_at_first_feasible:
        # SolutionLimit=1 makes Gurobi emphasize feasible-point heuristics.
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
    return model


def _add_parameter_variables(model, context_dimensions, n_decisions, n_criteria, fixed_parameters):
    """Create ``theta_tilde[k]``, free in ``[-1, 1]`` or pinned to a candidate."""
    theta_tilde = {}
    for k in range(n_criteria):
        if fixed_parameters is None:
            lower, upper = -1.0, 1.0
        else:
            weights, theta = fixed_parameters
            # A fixed point is encoded as identical lower and upper bounds.
            lower = upper = weights[k] * theta[k]
        theta_tilde[k] = model.addMVar(
            shape=(int(context_dimensions[k]), n_decisions),
            lb=lower,
            ub=upper,
            vtype=gp.GRB.CONTINUOUS,
            name=f"theta_tilde_{k}",
        )
    return theta_tilde


def _add_portfolio_support(model, theta_tilde, n_criteria):
    """Known zeros of the two-criterion portfolio surrogate.

    Decision layout is ``(asset weights, risk slack)``. Return does not depend
    on the slack, and the risk criterion depends only on the slack.
    """
    if n_criteria < 2:
        raise ValueError("Portfolio support constraints need a return criterion and a risk criterion.")
    n_decisions = theta_tilde[0].shape[1]
    model.addConstr(
        theta_tilde[0][:, -1] == np.zeros(theta_tilde[0].shape[0]),
        name="return_ignores_slack",
    )
    model.addConstr(
        theta_tilde[1][:, :-1] == np.zeros(n_decisions - 1),
        name="risk_prices_slack_only",
    )


def _add_normalization(model, theta_tilde, context_dimensions, n_decisions, n_criteria, tolerance):
    """``sum_k ||theta_tilde_k||_F = 1``, with ``norms[k]`` equal to each norm."""
    norms = model.addMVar(shape=n_criteria, lb=0.0, ub=1.0, name="norms")
    total = gp.quicksum(norms[k] for k in range(n_criteria))
    if tolerance == 0.0:
        model.addConstr(total == 1.0, name="unit_SumNorm")
    else:
        model.addConstr(total <= 1.0 + tolerance, name="unit_SumNorm_U")
        model.addConstr(total >= 1.0 - tolerance, name="unit_SumNorm_L")
    model.addConstrs(
        (
            norms[k] ** 2
            == gp.quicksum(
                theta_tilde[k][j, q] ** 2
                for j in range(int(context_dimensions[k]))
                for q in range(n_decisions)
            )
            for k in range(n_criteria)
        ),
        name="define_norm**2",
    )
    return norms


def _add_optimality_conditions(
    model,
    criterion_rows,
    theta_tilde,
    contexts,
    decisions,
    constraint_matrices,
    constraint_rhs,
    n_observations,
    n_decisions,
    n_constraints,
    n_criteria,
    gamma,
    tolerance,
):
    """KKT system that forces each observed decision to be optimal for ``theta_tilde``."""
    dual = model.addMVar(
        shape=(n_constraints, n_observations),
        lb=0.0,
        ub=float("inf"),
        vtype=gp.GRB.CONTINUOUS,
        name="mu",
    )
    for n in range(n_observations):
        _add_complementary_slackness(
            model, dual, constraint_matrices[n], constraint_rhs, decisions[n], n, n_constraints, tolerance
        )

    if gamma == "l_1":
        for n in range(n_observations):
            aggregated = np.array(
                [
                    gp.quicksum(criterion_rows[n][k][q] for k in range(n_criteria))
                    for q in range(n_decisions)
                ]
            )
            # Stationarity of min (sum_k c_k) @ z over A z <= b, with mu >= 0:
            # -(sum_k c_k) = A^T mu. Imposed exactly, even when tolerance > 0.
            model.addConstrs(
                (
                    -aggregated[q] == (constraint_matrices[n].T @ dual[:, n])[q]
                    for q in range(n_decisions)
                ),
                name=f"consistency_stationarity_{n}",
            )
        return

    # Dual multipliers of the criteria-wise problems. Sign y <= 0 matches
    # dual feasibility of a <= constrained linear program.
    criteria_dual = model.addMVar(
        shape=(n_constraints, n_criteria, n_observations),
        lb=-float("inf"),
        ub=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="criteria_dual",
    )
    for n in range(n_observations):
        for k in range(n_criteria):
            _add_criteria_dual_feasibility(
                model,
                criteria_dual[:, k, n],
                constraint_matrices[n],
                contexts[n][k] @ theta_tilde[k],
                tolerance,
            )
        # Excess of the observed decision over the criteria-wise ideal value,
        # written with the dual objective y @ b instead of an explicit ideal point.
        weighted_cost = np.array(
            [
                gp.quicksum(
                    criterion_rows[n][k][q]
                    * (
                        contexts[n][k] @ theta_tilde[k] @ decisions[n]
                        - criteria_dual[:, k, n] @ constraint_rhs
                    )
                    for k in range(n_criteria)
                )
                for q in range(n_decisions)
            ]
        )
        stationarity = constraint_matrices[n].T @ dual[:, n]
        _add_stationarity(model, weighted_cost, stationarity, n, n_decisions, tolerance)


def _add_complementary_slackness(
    model, dual, constraint_matrix, constraint_rhs, decision, observation, n_constraints, tolerance
):
    """``mu_i * (A_i z - b_i) = 0``. The residual is numeric, so this is linear in ``mu``.

    A zero residual makes the product identically zero, so the constraint is
    vacuous and is omitted. Gurobi also drops coefficients below ``1e-13``.
    """
    for i in range(n_constraints):
        residual = float(constraint_matrix[i, :] @ decision - constraint_rhs[i])
        if abs(residual) < 1e-13:
            continue
        product = dual[i, observation] * residual
        name = f"consistency_complementarity_{observation}_{i}"
        if tolerance == 0.0:
            model.addConstr(product == 0.0, name=name)
        else:
            model.addConstr(product <= tolerance, name=f"U_{name}")
            model.addConstr(product >= -tolerance, name=f"L_{name}")


def _add_criteria_dual_feasibility(model, dual_column, constraint_matrix, criterion_row, tolerance):
    """Dual feasibility ``A^T y = c`` for one criteria-wise linear program."""
    gradient = constraint_matrix.T @ dual_column
    if tolerance == 0.0:
        model.addConstr(gradient == criterion_row)
    else:
        model.addConstr(gradient <= criterion_row + tolerance)
        model.addConstr(gradient >= criterion_row - tolerance)


def _add_stationarity(model, weighted_cost, stationarity, observation, n_decisions, tolerance):
    """``-weighted_cost = A^T mu``, optionally relaxed by ``tolerance``."""
    for q in range(n_decisions):
        name = f"consistency_stationarity_{observation}_{q}"
        if tolerance == 0.0:
            model.addConstr(-weighted_cost[q] == stationarity[q], name=name)
        else:
            model.addConstr(-weighted_cost[q] <= stationarity[q] + tolerance, name=f"U_{name}")
            model.addConstr(-weighted_cost[q] >= stationarity[q] - tolerance, name=f"L_{name}")


def _add_estimation_objective(
    model,
    theta_tilde,
    norms,
    objective,
    prior,
    context_dimensions,
    n_decisions,
    n_criteria,
):
    """Weighted sum of a prior penalty and a sparsity penalty.

    Sparsity is counted only on criterion 0 and only on the asset columns.
    The last column is the risk slack: its zeros are structural (see
    ``_add_portfolio_support``) and are not a pattern to be learned.
    Entries of that column are already bounded, so a binary indicator with
    big-M equal to 1 is a valid linearization of "this entry is nonzero".
    """
    if not objective:
        model.setObjective(0.0, sense=gp.GRB.MINIMIZE)
        return

    expression = 0.0
    # Asset coordinates only. The trailing risk-slack column is excluded.
    asset_columns = range(n_decisions - 1)
    for weight, name in objective.items():
        if weight <= 0.0:
            continue
        if name == "min_prior":
            if prior is None:
                raise ValueError("Objective 'min_prior' requires a prior.")
            expression += weight * gp.quicksum(
                (theta_tilde[k][j, q] - norms[k] * prior[k][j, q]) ** 2
                for k in range(n_criteria)
                for j in range(int(context_dimensions[k]))
                for q in range(n_decisions)
            )
        elif name == "max_sparsity":
            # Only the first criterion is sparsified. Do not assign to
            # n_criteria here: that variable sizes the rest of the model.
            criterion = 0
            indicator = model.addMVar(
                shape=(int(context_dimensions[criterion]), n_decisions),
                vtype=gp.GRB.BINARY,
                name="sparse_entries_0",
            )
            model.addConstrs(
                (
                    theta_tilde[criterion][j, q] <= indicator[j, q]
                    for j in range(int(context_dimensions[criterion]))
                    for q in asset_columns
                )
            )
            model.addConstrs(
                (
                    -theta_tilde[criterion][j, q] <= indicator[j, q]
                    for j in range(int(context_dimensions[criterion]))
                    for q in asset_columns
                )
            )
            expression += weight * gp.quicksum(
                indicator[j, q]
                for j in range(int(context_dimensions[criterion]))
                for q in asset_columns
            )
        else:
            raise ValueError(f"Unknown objective term {name!r}.")
    model.setObjective(expression, sense=gp.GRB.MINIMIZE)


def _report_fixed_parameter_feasibility(model):
    """Print whether the pinned ``(w, theta)`` satisfies the inverse system.

    On failure, relax complementarity, stationarity, and the exact norm-sum
    equality, then print the positive artificial slacks. Squared-norm
    definitions and support constraints are left untouched: they define the
    variables rather than the optimality claim being tested.
    """
    if model.status == gp.GRB.OPTIMAL:
        print("Fixed (w, theta) is feasible for the inverse constraints.")
        return

    print(f"Fixed (w, theta) is infeasible (status {model.status}). Computing a relaxation.")
    original_n_variables = model.NumVars
    relaxed = model.copy()
    relaxed.setParam(gp.GRB.Param.OutputFlag, 0)
    relaxed.feasRelax(
        relaxobjtype=1,
        minrelax=False,
        vars=None,
        lbpen=None,
        ubpen=None,
        constrs=[c for c in relaxed.getConstrs() if _is_diagnostic_constraint(c.ConstrName)],
        rhspen=None,
    )
    relaxed.optimize()
    if relaxed.status != gp.GRB.OPTIMAL:
        print("Fixed (w, theta) stays infeasible after relaxing those constraints.")
        return

    print("Positive slacks in the relaxation:")
    meanings = {
        "ArtU": "increase the upper bound of variable",
        "ArtL": "decrease the lower bound of variable",
        "ArtP": "decrease the right-hand side of constraint",
        "ArtN": "increase the right-hand side of constraint",
    }
    for slack in relaxed.getVars()[original_n_variables:]:
        if slack.X <= 1e-9:
            continue
        kind = slack.VarName[:4]
        target = slack.VarName[5:]
        action = meanings.get(kind, kind)
        print(f"  {action} {target} by {slack.X:.9f}")


def _is_diagnostic_constraint(name):
    """Constraints whose violation is informative when a candidate parameter is rejected.

    The exact name ``unit_SumNorm`` is required so that the tolerance form
    (``unit_SumNorm_U`` / ``unit_SumNorm_L``) is not relaxed by accident.
    """
    return ("consistency" in name) or ("dualfeas" in name) or (name == "unit_SumNorm")
