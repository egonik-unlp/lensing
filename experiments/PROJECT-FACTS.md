# PROJECT FACTS — Price Guesser Models

Living, agent-maintained roll-up of empirical knowledge: the leaderboard,
noise bands, dataset lineage, per-family field guide and hard-won pitfalls.
The campaign reports in experiments/*.md are PRIMARY; this file is their
index. When they disagree, the newest report wins — and this file is stale
and must be updated. Writers: the experiment-designer agent and the
model-definitions experiment workflow reconcile this file as part of every
campaign report.

Last updated: never — no facts yet; the first campaign seeds this.

## Best on record (leaderboard)

| model | dataset | MAE | medAPE | R² | source |
|---|---|---|---|
| _none yet_ | | | |

## Noise bands & significance thresholds

- Unknown — run a same-split repeat and a seed study early; until then,
  treat small margins as noise.

## Dataset lineage

| dataset (recipe) | feeds | notes |
|---|---|---|
| _none yet_ | | |

## Known pitfalls per predictor family

Generic priors (replace with measured facts as campaigns land):

- Bounded-by-construction families (trees: leaf averages; rbf kernels:
  revert-to-intercept) cannot blow up through a log-target inverse;
  unbounded ones (linear models, MLPs) can — ship MLPs with
  `clamp_output: true` until proven safe.
- Features usually beat architecture: scan dataset-level axes before deep
  hyperparameter scans.
