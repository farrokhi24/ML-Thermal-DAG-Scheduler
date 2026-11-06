# Thermal policy: source audit and project adaptation

## What is reproduced, and what is not

Primary source: Chen & Liao, ISCAS 2020, *Adaptive Machine Learning-based
Temperature Prediction Scheme for Thermal-aware NoC System*,
DOI [10.1109/ISCAS45731.2020.9180475](https://doi.org/10.1109/ISCAS45731.2020.9180475).

Section III describes a single-neuron temperature predictor with nine local
and neighboring temperatures and nine activity-throttling inputs, ReLU, and LMS
weight adaptation. Section IV-A evaluates it using distributed PDTM "similar
to [5]". It does not give a complete controller pseudocode, cooldown duration,
threshold-selection rule or offline ANN optimizer. We do not fill these gaps
and describe them as paper parameters.

Reference [5]: Chen et al., TPDS 2015, *RC-Based Temperature Prediction Scheme
for Proactive Dynamic Thermal Management in Throttle-Based 3D NoCs*,
DOI [10.1109/TPDS.2014.2308206](https://doi.org/10.1109/TPDS.2014.2308206).
[Author-hosted full text](https://access.ee.ntu.edu.tw/files/journals/J63_2015.pdf),
Section 4.2, Figure 10 (journal page 214).
This reference distinguishes measured thermal emergencies from forecast ones
and distributes a power budget for partial throttling. For the one-step
prediction example, partial activity is 1/2. Full throttling handles a measured
emergency. We implement that one-step mechanism as an explicitly ADAPTED
comparison, not the complete 3D NoC/DFS experiment.

The later 2023 TVLSI paper with a similar title adds reinforcement learning;
it is not the four-page 2020 source and its RL controller is not implemented.

## Available policies

| Name | Decision | Attribution |
| --- | --- | --- |
| `none` | Full activity, no thermal guard | Project baseline |
| `reactive` | Full activity until measured emergency; global full cooling | Project reactive baseline |
| `predictive` | Search descending duty levels with ANN; choose first acceptable forecast | Project policy, not the 2020 algorithm |
| `budget` | Predict at full activity; forecast emergency selects half duty for k=1 | Adapted one-step [5] power-budget mechanism |

All controlled modes share a measured-temperature GLOBAL cooldown guard. It
pauses occupied cores at `max(T) >= limit`, resumes once `max(T) <= limit -
hysteresis`, and is a project choice for this coupled core-grid plant. The
reference discusses local/distributed NoC control. Thus `budget` is not called
an exact paper reproduction. NoC traffic throttling is abstracted as duty-
averaged CPU progress here; voltage, packets, buffers and routing are absent.

`margin`, `hysteresis`, duty levels, grid geometry, ambient/initial temperature,
R, C, coupling, power distribution and sensing interval are project parameters.
With default levels {1, .75, .5, .25, 0}, `budget` requests .5. For custom levels
it rounds down to the largest positive level <= .5; a configuration without
one is rejected. Forecasting uses a one-sensing-step horizon only.

`predictive` retains the minimum-progress fallback outside measured cooldown;
this may intentionally override a pessimistic forecast. Its activation count is
reported. Global cooling can overshoot the limit between sensor samples; no
continuous-time hard thermal guarantee is made.

## Predictor adaptations

- Each core has its own 18 weights, with missing grid neighbors zero-masked.
- Temperatures/targets are scaled by 100. `mu = dt * seconds_per_unit` is the
  default choice on the SCALED feature space; this is not numerical equivalence
  to an unscaled analog LMS implementation.
- Offline initialization is ridge least squares toward persistence. The 2020
  paper does not specify this training solver. Training graphs are independent
  from evaluation graphs.
- Online LMS receives stored causal features only when the full-horizon target
  arrives; partial terminal intervals cannot train a full-step prediction.
- The RC plant is an uncalibrated synthetic data source, not the paper's 8x8 NoC
  traffic/thermal co-simulator. The default 45 C initial temperature is a project
  setting; the paper uses 80 C. The CLI exposes it for sensitivity checks.

## Fair interpretation

Duty scales dynamic power AND execution progress by the same ratio. For a fixed
set of tasks, dynamic execution energy is consequently almost invariant in this
linear model. Slower completion adds idle energy; temperature reduction does not
imply energy reduction. Report both, including unfavorable results.

Time above limit and integrated excess temperature use left sensor samples,
in seconds and core-seconds. `peak_event_temperature_celsius` also samples task
completion events, but is still not a continuous-time maximum.

The DAG paper's unthrottled Graham bound does not apply unchanged to throttled
execution. Deadline flags in the benchmark are OBSERVED outcomes under explicit
project deadlines. Algorithm 3 remains experimental even if a regression is
fixed or a candidate lies above the observed schedule.
