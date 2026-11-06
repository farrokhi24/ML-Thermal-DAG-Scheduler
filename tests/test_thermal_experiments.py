import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ann_lms_predictor import ANNLMSPredictor
from dag_baseline import DAG
from run_thermal_experiments import run_thermal_experiments
from thermal_scheduler import ThrottleController, simulate_thermal_dag


class ThermalBenchmarkTests(unittest.TestCase):
    def test_reference_budget_half_duty_and_measured_guard(self):
        model = ANNLMSPredictor(2)
        model.weights[:, 0] = 2
        controller = ThrottleController(2, margin=0)
        np.testing.assert_equal(controller.decide([45, 45], [True, False], 'budget', model), [.5, 1])
        self.assertEqual(controller.last_reason, 'one_step_power_budget')
        np.testing.assert_equal(controller.decide([55, 45], [True, True], 'budget', model), [0, 0])
        self.assertEqual(controller.last_reason, 'measured_global_cooldown')
        np.testing.assert_equal(controller.decide([54, 45], [True, True], 'budget', model), [0, 0])
        np.testing.assert_equal(controller.decide([52, 45], [True, True], 'budget', model), [.5, .5])

    def test_sampled_violation_duration_accounts_for_partial_terminal_interval(self):
        dag = DAG({1: 2.5}, [])
        result = simulate_thermal_dag(dag, {1: 1}, 1, policy='none', seconds_per_unit=.1,
                                      dt=1, initial_temperature=60, limit=55)
        self.assertAlmostEqual(result['metrics']['sampled_violation_core_seconds'], .25)
        self.assertGreater(result['metrics']['sampled_excess_degree_core_seconds'], 0)

    def test_benchmark_pairs_policies_and_preserves_all_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_thermal_experiments(directory, count=2, cores=(1, 2),
                                             workload=35, parallelism=3,
                                             seconds_per_unit=.05, training_graphs=1,
                                             export_traces=True)
            self.assertEqual(report['requested_runs'], 16)
            self.assertEqual(report['failed_runs'], 0)
            self.assertEqual(len(report['prediction_validation']), 4)
            frame = pd.read_csv(Path(directory) / 'results.csv')
            self.assertEqual(len(frame), 16)
            for _, group in frame.groupby(['seed', 'cores']):
                self.assertEqual(group.input_sha256.nunique(), 1)
                self.assertEqual(group.deadline.nunique(), 1)
                self.assertEqual(group.power_seed.nunique(), 1)
                self.assertEqual(set(group.policy), {'none', 'reactive', 'predictive', 'budget'})
            self.assertTrue((Path(directory) / 'seed_2026/cores_2/budget/control.csv').is_file())
            self.assertTrue(all(g['completed'] == 2 for g in report['by_policy']))
            models = [json.loads((Path(directory) / f'model_cores_{c}.json').read_text()) for c in [1, 2]]
            self.assertTrue(all(m['online_updates'] == 0 for m in models))

    def test_failed_runs_remain_visible(self):
        import run_thermal_experiments as module
        original = module.simulate_thermal_dag
        def failing(*args, **kwargs):
            if kwargs.get('policy') == 'budget':
                raise ValueError('forced benchmark failure')
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'simulate_thermal_dag', side_effect=failing):
            result = run_thermal_experiments(directory, count=1, cores=(1,), workload=20,
                                              parallelism=2, seconds_per_unit=.05, training_graphs=1)
            self.assertEqual(result['failed_runs'], 1)
            group = next(g for g in result['by_policy'] if g['policy'] == 'budget')
            self.assertEqual(group['completed'], 0)
            pairs = next(g for g in result['paired_deltas_vs_none'] if g['policy'] == 'budget')
            self.assertEqual(pairs['unpaired_samples'], 1)

    def test_calibration_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                run_thermal_experiments(directory, seed=9000, count=1, cores=(2,))
