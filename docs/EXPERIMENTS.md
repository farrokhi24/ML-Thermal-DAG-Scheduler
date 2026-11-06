# Reproducible combined-system experiments

Run from the repository root, after installing `requirements.txt`:

```sh
MPLBACKEND=Agg python3 run_thermal_experiments.py \
  --count 30 --cores 2 4 --seconds-per-unit .05 \
  --out outputs/thermal_validation --export-traces
```

If a sandbox cannot write font caches, set `MPLCONFIGDIR` to a writable local
folder as well. The plotter uses the noninteractive Agg backend.

This runs 30 independent test DAGs on two platforms with four control policies
(240 policy runs), plus frozen/online predictor comparisons on identical
uncontrolled trajectories. The three calibration DAG seeds 9000..9002 are
strictly disjoint from test seeds 2026..2055. A model is fitted once per platform
and cloned for every independent evaluation run; test updates never leak into
the next graph or another policy.

For each (DAG, core count), all policies share the input hash, task-power seed,
RC parameters, initial temperature, priorities and relative deadline. Deadline
is an explicit project assumption: `D = T = deadline_factor * certified
uncontrolled bound`, default factor 1.25, fixed before evaluating any controller.
This benchmark simulates ONE release per DAG, not multiple overlapping jobs.
It compares observed completion times to D; it does not certify thermal timing.

Outputs:

- `results.csv`: every requested policy run, including errors and messages.
- `summary.json`: requested/completed/failed counts; per-policy mean/min/max;
  paired deltas against `none`, missing-pair counts, prediction scores,
  full configuration, calibration seeds, git base, package versions and source hashes.
- `inputs/`: exact test DAG JSON, including generator settings and seed.
- `model_cores_*.json`: the independent offline models before test updates.
- `comparison.png`: descriptive means for time, peak temperature, energy and
  sampled thermal exposure. It is not a confidence/significance plot.
- With `--export-traces`: schedules, action reasons, temperatures, causal learning
  pairs, predictions and task powers for every case/policy.

Successful-case means never hide failed runs: failure counts and paired coverage
are shown next to them. The CLI exits nonzero if ANY policy run failed. Fatal
calibration/configuration errors abort the experiment instead of manufacturing
scores. As in the DAG paper, generated topology depth defaults to 5..8 and layer
width to 2..p; these are not packet-traffic experiments from the thermal paper.

Sensitivity examples (keep the test/training seeds fixed):

```sh
# Shorter sensing interval: more samples and a different one-step prediction horizon.
MPLBACKEND=Agg python3 run_thermal_experiments.py --count 30 --cores 2 4 \
  --seconds-per-unit .05 --dt .25 --out outputs/thermal_fast_sensing
# Hot start, retaining an explicit project limit and RC environment.
MPLBACKEND=Agg python3 run_thermal_experiments.py --count 30 --cores 2 4 \
  --seconds-per-unit .05 --initial-temperature 80 --out outputs/thermal_hot_start
```

`--limit`, `--margin`, `--hysteresis`, `--levels`, `--resistance`, `--capacitance`,
`--coupling`, `--ambient`, `--initial-temperature`, `--power-min`, `--power-max`,
`--idle-power`, `--dt` and `--mu` are exposed and saved. Tune on separate validation
seeds, not on the held-out test results. No claim of universal predictive-policy
superiority or reproduction of the paper's percentage improvements is made.

## Large DAG timing analysis

`analyze_timing` no longer enumerates the full local path list. Its dynamic
program retains per-ending-node states indexed by unions of higher/lower
interference sets. States with equal signatures retain the greatest residual
path workload; their future scoring is identical, so this merge is exact for
the EXISTING experimental recurrence. A representative path is kept for audits.

The exported `path_statistics` contains the number of represented complete paths,
retained states and terminal signatures. Detailed `paths` contains representative
states, not necessarily all paths. `local_paths` is null in the compact result.

This removes the old 10,000-COMPLETE-PATH limit, not the intrinsic combinatorics of
all interference signatures. `max_states` defaults to 200,000 per block. Exceeding
it reports `not_evaluated_state_budget`, with `bound: null`, while retaining the
independent certified unthrottled bound. No truncated set is called exact. The
legacy `local_paths` helper still supports capped explicit enumeration for small
reference/oracle tests; the production analysis does not call it.

Tests compare compact evaluation to independent exhaustive path lists on small
seeded DAGs and exercise the >10,000-path example. Algorithm 3's original safety
status is preserved: improving scalability does not prove its recurrence sound.
