import csv
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from thermal_model import generate_thermal_trace


class ThermalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_trace(self, rows, **kwargs):
        schedule = self.root / 'schedule.csv'
        with schedule.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['node', 'core', 'start', 'finish'])
            writer.writeheader()
            writer.writerows(dict(zip(writer.fieldnames, row)) for row in rows)
        return generate_thermal_trace(schedule, self.root / 'trace.csv', **kwargs)

    def test_constant_power_matches_closed_form_at_fractional_samples(self):
        frame = self.run_trace([(1, 0, 0, 2.5)], T_amb=20, R=2, C=3, dt=.7,
                               node_powers={1: 10}, seconds_per_unit=1)
        self.assertEqual(frame.Time.iloc[-1], 2.5)
        for _, row in frame.iterrows():
            self.assertAlmostEqual(row.Temp_Core_0, 20 + 20 * (1 - math.exp(-row.Time / 6)), places=12)
        self.assertEqual(frame.Power_Core_0.iloc[0], 10)
        self.assertEqual(frame.Power_Core_0.iloc[-1], 2)

    def test_switches_inside_sample_and_fractional_final_interval(self):
        frame = self.run_trace([(1, 0, 0, .25), (2, 0, .25, .75)],
                               T_amb=20, R=2, C=3, dt=2, P_idle=0,
                               seconds_per_unit=2, cooldown=.5, node_powers={1: 10, 2: 20})
        # Independent piecewise scalar solution over .5s, 1s, 1s.
        state = 20 * (1 - math.exp(-.5 / 6))
        state = 40 + (state - 40) * math.exp(-1 / 6)
        state *= math.exp(-1 / 6)
        self.assertEqual(list(frame.Time), [0, 1.25])
        self.assertAlmostEqual(frame.Temp_Core_0.iloc[-1], 20 + state, places=12)

    def test_sampling_step_does_not_change_underlying_final_temperature(self):
        rows = [(1, 0, .1, .4), (2, 0, .4, 1.1), (3, 1, 0, .8)]
        kwargs = dict(node_powers={1: 10, 2: 20, 3: 7}, cores=3,
                      coupling=.2, cooldown=.3, seconds_per_unit=.5)
        coarse = self.run_trace(rows, dt=2, **kwargs)
        fine = self.run_trace(rows, dt=.05, **kwargs)
        for c in range(3):
            self.assertAlmostEqual(coarse[f'Temp_Core_{c}'].iloc[-1],
                                   fine[f'Temp_Core_{c}'].iloc[-1], places=11)

    def test_temperature_is_not_reset_at_a_task_boundary(self):
        frame = self.run_trace([(1, 0, 0, .5), (2, 0, .5, 1)], dt=.5,
                               node_powers={1: 10, 2: 10})
        self.assertGreater(frame.Temp_Core_0.iloc[1], frame.Temp_Core_0.iloc[0])
        self.assertGreater(frame.Temp_Core_0.iloc[2], frame.Temp_Core_0.iloc[1])
        self.assertEqual(frame.Active_Core_0.iloc[1], 1)

    def test_idle_cores_are_exported_and_initial_temperature_cools(self):
        frame = self.run_trace([(1, 0, 0, 1)], cores=3, T_amb=20, initial_temperature=30,
                               P_idle=0, node_powers={1: 0}, R=2, C=3)
        for c in range(3):
            self.assertAlmostEqual(frame[f'Temp_Core_{c}'].iloc[-1], 20 + 10 * math.exp(-1/6))
        self.assertTrue((frame.Active_Core_2 == 0).all())

    def test_coupling_matches_two_core_closed_form(self):
        frame = self.run_trace([(1, 0, 0, 2)], cores=2, T_amb=20,
                               R=2, C=3, coupling=.5, P_idle=0, node_powers={1: 10}, dt=2)
        mean = 10 * (1 - math.exp(-2/6))
        difference = (5 / 1.5) * (1 - math.exp(-1.5 * 2 / 3))
        self.assertAlmostEqual(frame.Temp_Core_0.iloc[-1], 20 + mean + difference, places=12)
        self.assertAlmostEqual(frame.Temp_Core_1.iloc[-1], 20 + mean - difference, places=12)
        self.assertGreater(frame.Temp_Core_1.iloc[-1], 20)

    def test_learning_inputs_and_targets_have_correct_times_and_masks(self):
        frame = self.run_trace([(1, 0, 0, 1.25)], cores=2, dt=.5,
                               seconds_per_unit=.01, node_powers={1: 10})
        learning = pd.read_csv(self.root / 'trace_learning.csv')
        first = learning.iloc[0]
        self.assertEqual(first['time'], 0)
        self.assertEqual(first['target_time'], .5)
        self.assertAlmostEqual(first['horizon_seconds'], .005)
        self.assertEqual(first['slot_0_temperature'], frame.Temp_Core_0.iloc[0])
        self.assertEqual(first['target_temperature'], frame.Temp_Core_0.iloc[1])
        self.assertEqual(first['slot_1_mask'], 0)  # north boundary
        self.assertEqual(first['slot_3_mask'], 1)  # east: core 1
        self.assertEqual(len(learning), (len(frame) - 1) * 2)
        self.assertAlmostEqual(learning.horizon_seconds.iloc[-1], .0025)
        metadata = json.loads((self.root / 'trace.metadata.json').read_text())
        self.assertEqual(metadata['parameters']['seconds_per_schedule_unit'], .01)
        self.assertEqual(metadata['topology']['neighbors'], {'0': [1], '1': [0]})

    def test_future_changes_do_not_leak_into_current_features(self):
        rows = [(1, 0, 0, 1), (2, 0, 1, 2)]
        self.run_trace(rows, dt=.5, node_powers={1: 10, 2: 15})
        first = pd.read_csv(self.root / 'trace_learning.csv').iloc[0]
        self.run_trace(rows, dt=.5, node_powers={1: 10, 2: 100})
        changed = pd.read_csv(self.root / 'trace_learning.csv').iloc[0]
        pd.testing.assert_series_equal(first, changed)

    def test_seed_noise_and_global_rng_isolation(self):
        random.seed(2026)
        before = random.getstate()
        np_before = np.random.get_state()
        first = self.run_trace([(1, 0, 0, 1)], dt=.25, sensor_noise_std=.1, seed=12)
        second = self.run_trace([(1, 0, 0, 1)], dt=.25, sensor_noise_std=.1, seed=12)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(random.getstate(), before)
        np_after = np.random.get_state()
        np.testing.assert_equal(np_before[1], np_after[1])
        self.assertEqual(np_before[2:], np_after[2:])
        self.assertFalse((first.Temp_Core_0 == first.Latent_Temp_Core_0).all())

    def test_invalid_schedules_and_physical_parameters(self):
        for rows in [[], [(1, 0, 0, 2), (2, 0, 1, 3)], [(1, -1, 0, 1)],
                     [(1, 0, 1, 1)], [(1, 0, 0, float('nan'))],
                     [(1, 0, 0, 1), (1, 1, 0, 1)]]:
            with self.assertRaises(ValueError):
                self.run_trace(rows)
        for kwargs in [dict(dt=0), dict(R=-1), dict(C=0), dict(seconds_per_unit=0),
                       dict(sensor_noise_std=-1), dict(coupling=-1), dict(cores=0),
                       dict(power_min=20, power_max=10), dict(cores=True),
                       dict(node_powers={2: 10}), dict(node_powers={1: -1})]:
            with self.assertRaises(ValueError):
                self.run_trace([(1, 0, 0, 1)], **kwargs)
