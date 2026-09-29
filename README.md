# Contextual inverse multiobjective optimization

Recover the hidden linear criteria of a contextual multiobjective problem from observed optimal decisions, then measure how well those estimates reproduce the decisions in and out of sample. The numerical study is a long-only portfolio: the forward model trades off predicted return against an epigraph of holding-level volatility.

## Model

An observation is a context `x`, a feasible set `Z(A, b) = {z : A z <= b}`, and an observed decision `z*`. Criterion `k` is the linear form `c_k(x) = w_k * x_k @ theta_k`.

The forward problem is a compromise over these forms.

- `l_1` minimizes the sum of the criteria.
- `l_2` minimizes Euclidean distance, in criterion space, to the ideal point (the vector of criteria-wise optima).

The inverse problem does not estimate `w_k` and `theta_k` separately. It estimates the product `theta_tilde_k ≈ w_k * theta_k`, normalized so that the Frobenius norms sum to one, subject to the KKT system that makes every observed decision optimal. Two optional estimation terms can be combined by positive weights:

- `min_prior` pulls `theta_tilde_k` toward `||theta_tilde_k||_F` times a given direction.
- `max_sparsity` minimizes the number of nonzeros in the asset columns of the first criterion. The risk-slack column is excluded; those zeros are structural.

In the portfolio study the prior is synthetic: the true direction, plus uniform noise, times a random 0/1 mask. That is side information for the simulation, not a quantity computed from the decisions.

## Portfolio instance

The decision is `z = (z_1, ..., z_S, s)`: long-only weights summing to one, and a slack `s` with `s >= vol_{t,i} * z_i`. Return (and volume, if requested) is a regression of the next day's value on the chosen context, sign-flipped because the forward model minimizes. Risk is the unit cost on `s`. With portfolio support constraints turned on, the estimate is forced to respect that pattern: return does not price `s`, and risk prices only `s`.

Context designs are `market`, `past_t-1`, and `all`. The active grid uses lagged criterion values, a sparse own-lag regression, scalarization `l_2`, and tolerance `0.005`.

## Layout

```
cimo/models.py            forward and inverse formulations
cimo/evaluation.py        weights, cosine similarity, consistency, gaps
cimo/data/market.py       prices, surrogates, feasible sets, forward solves
experiments/run_experiments.py
experiments/summarize_results.py
cache/                    downloaded market frames
results/                  pickled runs and the Excel table
```

## Setup

Gurobi needs a license beyond the `gurobipy` wheel. Then:

```bash
pip install -r requirements.txt
```

`pot` is imported as `ot`.

## Run

```bash
python experiments/run_experiments.py
python experiments/summarize_results.py
```

Paths are resolved from the repository root, so the working directory does not matter. The default grid is seeds `123, 456, 789`, one training date, 100 test dates, and five stocks. Widen the lists at the bottom of `experiments/run_experiments.py` for the larger design (training sizes 25 to 200). Each finished point is written to `results/results.pkl`; a new process overwrites that file.
