import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_experiments import run_experiments

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_demo_exports_candidate_status_safe_bound_and_thermal_data(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            run = subprocess.run([sys.executable, str(ROOT / 'run_demo.py'), '--out', str(out),
                                  '--thermal', '--seconds-per-unit', '.01', '--thermal-dt', '.25'],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['analytical_algorithm_3_bound'], 16.5)
            self.assertEqual(report['simulated_makespan'], 15)
            self.assertEqual(report['model_safe_bound'], 20.5)
            self.assertEqual(report['algorithm_3_status'], 'experimental_unverified')
            for name in ['schedule.csv', 'thermal_trace.csv', 'thermal_trace_learning.csv',
                         'thermal_trace_node_power.csv', 'thermal_trace.metadata.json', 'thermal_trace.png']:
                self.assertTrue((out / name).is_file(), name)

    def test_demo_counterexample_is_exported_with_failed_sanity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([sys.executable, str(ROOT / 'run_demo.py'), '--out', directory,
                                  '--input', str(ROOT / 'examples/algorithm_3_counterexample.json')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((Path(directory) / 'report.json').read_text())
            self.assertEqual(report['algorithm_3_status'], 'experimental_unverified')
            self.assertFalse(report['algorithm_3_details']['sanity_issues'])

    def test_thermal_cli_requires_an_explicit_time_conversion(self):
        run = subprocess.run([sys.executable, str(ROOT / 'run_demo.py'), '--thermal'],
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('--seconds-per-unit', run.stderr)

    def test_small_experiment_uses_same_inputs_and_preserves_all_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiments(directory, count=3, cores=(2, 4), thermal_count=1,
                                     seconds_per_unit=.001, thermal_dt=10)
            root = Path(directory)
            self.assertEqual(result['requested_cases'], 6)
            self.assertEqual(sum(result['statuses'].values()), 6)
            self.assertNotIn('error', result['statuses'])
            self.assertEqual(len(list((root / 'inputs').glob('*.json'))), 3)
            self.assertTrue((root / 'sample_0000/cores_2/thermal_trace_learning.csv').is_file())
            self.assertTrue((root / 'sample_0000/cores_4/thermal_trace_learning.csv').is_file())

    def test_experiment_retains_a_path_limit_error(self):
        # Exercise explicit accounting of failed cases without relying on a
        # random seed to exceed the path cap.
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            with patch('run_experiments.assign_priorities', side_effect=ValueError('Local path limit exceeded')):
                result = run_experiments(directory, count=2, cores=(2, 4))
            self.assertEqual(result['statuses'], {'error': 4})
            self.assertEqual(len((Path(directory) / 'results.csv').read_text().splitlines()), 5)
