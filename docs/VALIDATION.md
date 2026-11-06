# Validation record (2026-11-06)

Base pulled from origin/main: `beddf26`. This record covers the local thermal
policy, compact-analysis and paired-benchmark changes. Raw runs are deliberately
ignored by git; rerun the commands below to regenerate them. Source hashes and
environment versions are preserved in each experiment's `summary.json`.

## Automated checks

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dag-mpl python3 -m unittest discover -s tests -q
```

68 tests passed. New coverage includes compact-vs-exhaustive analysis on 30
seeded graphs and three core counts, the old >10,000-path fixture, explicit
state-budget fallback, budget-policy action decisions, partial-interval thermal
exposure, shared inputs and deadlines, disjoint training seeds, and preservation
of failed benchmark runs. No test turns the experimental candidate into a proof.

## Executed experiments

| Scenario | Test DAGs | Core counts | Policy runs | Failures |
| --- | ---: | --- | ---: | ---: |
| Default sensing, initial 45 C | 30 | 2, 4 | 240 | 0 |
| dt=.25, initial 45 C | 10 | 2, 4 | 80 | 0 |
| Initial 80 C | 10 | 2, 4 | 80 | 0 |

All use workload 300, test seeds starting at 2026, training seeds 9000..9002,
seconds_per_unit=.05 and temperature limit 55 C. Each policy sees the same DAG
and task powers for each seed. These are controlled synthetic experiments, not
hardware measurements or the original paper's traffic benchmarks.

Commands:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dag-mpl python3 run_thermal_experiments.py --count 30 --cores 2 4 --seconds-per-unit .05 --out outputs/thermal_validation --export-traces
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dag-mpl python3 run_thermal_experiments.py --count 10 --cores 2 4 --seconds-per-unit .05 --dt .25 --out outputs/thermal_fast_sensing
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dag-mpl python3 run_thermal_experiments.py --count 10 --cores 2 4 --seconds-per-unit .05 --initial-temperature 80 --out outputs/thermal_hot_start
```

## Default-scenario descriptive means (30 DAGs per row)

| Cores | Policy | Makespan (units) | Event peak (C) | Energy (J) | Observed deadline met |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | budget | 201.55 | 54.55 | 237.56 | 100.0% |
| 2 | none | 169.21 | 57.33 | 231.09 | 100.0% |
| 2 | predictive | 201.31 | 54.55 | 237.51 | 100.0% |
| 2 | reactive | 217.28 | 55.07 | 240.71 | 96.7% |
| 4 | budget | 141.80 | 54.43 | 253.97 | 100.0% |
| 4 | none | 132.56 | 55.37 | 250.28 | 100.0% |
| 4 | predictive | 141.60 | 54.43 | 253.89 | 100.0% |
| 4 | reactive | 146.23 | 54.81 | 255.74 | 100.0% |

The predictive controller trades longer execution and more idle energy relative
to no control for lower temperatures. It finishes sooner than the reactive
controller in these default cases. The linear duty/power model does NOT show
energy savings relative to no control, and none are claimed.

For the 60 independent uncontrolled predictor validations (30 DAGs x 2 platforms),
the mean of per-case prediction MAEs is:

- Frozen ANN: 0.08931 C.
- ANN+LMS: 0.05669 C.
- Persistence: 0.08881 C.

This is a macro-average of per-case MAEs, not a pooled sample MAE. Frozen and
online models were scored on the SAME uncontrolled thermal trajectories.
Training/validation here is synthetic and does not establish real-sensor accuracy.

Hot-start peak temperature is 80 C for every policy because the initial state
is already hot. Exposure and completion, rather than a lower initial peak, are
the appropriate comparisons there. Observed deadline rates depend on the
explicit project deadline factor (1.25) and are not formal guarantees.

## Scaling smoke checks

The old adversarial local-path test represents 16,385 complete paths and now
returns an experimental candidate rather than `not_evaluated_path_limit`.
Compact-state results match exhaustive path results on the small oracle suite.
Six additional DAGs with 45..151 nodes (depths 10, 20, 30; seeds 340, 341; p=8;
m=4) completed the compact analysis; the local smoke-run times were about
0.001..0.014 seconds. This is not a cross-machine performance benchmark.
The maximum interference-state budget remains explicit; combinatorial state
growth has not been mathematically eliminated.

## Remaining scientific limitations

- Algorithm 3 remains experimental/unverified, even after the colleague's fix
  changes the supplied counterexample from 17 to 23.5 (observed time 18).
- Existing He2019/Zhao2020 code compares priority heuristics' observed schedules;
  it does not reproduce their complete timing analyses or reserved-core policy.
- `budget` adapts the one-step power-budget mechanism, retaining the project's
  global guard; it is not exact distributed NoC/DFS reproduction.
- Thermal parameters, core mapping and deadlines are project assumptions.
- Continuous thermal safety and controlled worst-case timing are not proven by
  these observed benchmarks.
