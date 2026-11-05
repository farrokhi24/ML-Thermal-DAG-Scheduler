from pathlib import Path
import json
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ann_lms_predictor import ANNLMSPredictor
from dag_baseline import DAG, assign_priorities, divide_subgraphs, simulate
from dag_scheduler import DAGGenerator
from run_system import run_system
from thermal_model import generate_thermal_trace
from thermal_scheduler import ThrottleController, simulate_thermal_dag

ROOT = Path(__file__).resolve().parents[1]


def priorities(dag):
    _, blocks, tail = divide_subgraphs(dag)
    return assign_priorities(dag, blocks, tail)


class ClosedLoopTests(unittest.TestCase):
    def test_no_control_matches_event_scheduler_for_generated_graphs(self):
        for seed in range(12):
            wcet, edges = DAGGenerator(seed=seed, total_workload=50).generate_dag()
            dag = DAG(wcet, edges)
            p = priorities(dag)
            for cores in [1, 2, 4]:
                standard = simulate(dag, p, cores)
                run = simulate_thermal_dag(dag, p, cores, policy='none', seconds_per_unit=.01, dt=1)
                self.assertAlmostEqual(run['metrics']['makespan'], max(e['finish'] for e in standard), places=8)
                by_node = {e['node']: e for e in standard}
                for event in run['schedule'].to_dict('records'):
                    self.assertEqual(event['core'], by_node[event['node']]['core'])
                    self.assertAlmostEqual(event['start'], by_node[event['node']]['start'], places=8)
                    self.assertAlmostEqual(event['finish'], by_node[event['node']]['finish'], places=8)

    def test_no_control_plant_matches_postprocessed_thermal_trace(self):
        dag = DAG({1: 1.25, 2: 2.5}, [(1, 2)])
        run = simulate_thermal_dag(dag, priorities(dag), 2, policy='none', seconds_per_unit=.2,
                                   node_powers={1: 10, 2: 20}, dt=.7)
        with tempfile.TemporaryDirectory() as directory:
            schedule = Path(directory) / 'schedule.csv'
            run['schedule'].to_csv(schedule, index=False)
            frame = generate_thermal_trace(schedule, Path(directory) / 'trace.csv', cores=2,
                                           seconds_per_unit=.2, dt=.7, coupling=.15,
                                           node_powers={1: 10, 2: 20})
            for core in range(2):
                np.testing.assert_allclose(frame[f'Temp_Core_{core}'], run['trace'][f'Temp_Core_{core}'], atol=1e-10)

    def test_duty_changes_execution_duration_power_and_energy(self):
        dag = DAG({1: 4}, [])
        with patch.object(ThrottleController, 'decide', return_value=np.array([.5])):
            run = simulate_thermal_dag(dag, priorities(dag), 1, policy='reactive',
                                       seconds_per_unit=1, node_powers={1: 10}, coupling=0)
        self.assertEqual(run['metrics']['makespan'], 8)
        # P=Pidle + duty*(Pactive-Pidle)=2+.5*8=6W for 8 seconds.
        self.assertAlmostEqual(run['metrics']['energy_joules'], 48)
        self.assertTrue((run['trace'].Power_Core_0.iloc[:-1] == 6).all())
        self.assertAlmostEqual(run['trace'].Temp_Core_0.iloc[-1], 45 + 4.8 * (1 - math.exp(-8/2.4)))

    def test_global_cooldown_hysteresis(self):
        controller = ThrottleController(2, limit=55, hysteresis=2)
        np.testing.assert_equal(controller.decide([55, 45], [1, 1], 'reactive'), [0, 0])
        np.testing.assert_equal(controller.decide([54, 45], [1, 1], 'reactive'), [0, 0])
        np.testing.assert_equal(controller.decide([52.9, 45], [1, 1], 'reactive'), [1, 1])

    def test_bad_forecast_cannot_stall_a_cool_platform(self):
        model = ANNLMSPredictor(1)
        model.weights[:] = 100
        dag = DAG({1: 2.5}, [])
        run = simulate_thermal_dag(dag, priorities(dag), 1, policy='predictive', predictor=model,
                                   adapt=False, seconds_per_unit=.01, node_powers={1: 10})
        self.assertEqual(run['metrics']['makespan'], 10)
        self.assertGreater(run['metrics']['progress_fallbacks'], 0)
        self.assertGreaterEqual(run['metrics']['conservative_controlled_response_bound'], 10)

    def test_lms_waits_for_full_horizon_and_does_not_mutate_offline_model(self):
        model = ANNLMSPredictor(1)
        before = model.to_dict()
        dag = DAG({1: 2.5}, [])
        run = simulate_thermal_dag(dag, priorities(dag), 1, policy='none', predictor=model,
                                   seconds_per_unit=.1, dt=1)
        self.assertEqual(model.to_dict(), before)
        self.assertEqual(run['model'].updates, 2)
        self.assertEqual(list(run['predictions'].time), [1, 2])

    def test_reactive_control_cools_hot_workload_and_completes_dependencies(self):
        dag = DAG({1: 30, 2: 30, 3: 40}, [(1, 2), (2, 3)])
        settings = dict(seconds_per_unit=.1, dt=.5, node_powers={1: 30, 2: 30, 3: 30}, coupling=0)
        baseline = simulate_thermal_dag(dag, priorities(dag), 1, policy='none', **settings)
        controlled = simulate_thermal_dag(dag, priorities(dag), 1, policy='reactive', **settings)
        self.assertLess(controlled['metrics']['peak_event_temperature_celsius'],
                        baseline['metrics']['peak_event_temperature_celsius'])
        self.assertGreater(controlled['metrics']['makespan'], baseline['metrics']['makespan'])
        self.assertGreater(controlled['metrics']['throttled_core_seconds'], 0)
        self.assertLessEqual(controlled['metrics']['makespan'],
                             controlled['metrics']['conservative_controlled_response_bound'])
        events = controlled['schedule'].set_index('node')
        self.assertGreaterEqual(events.loc[2, 'start'], events.loc[1, 'finish'])
        self.assertGreaterEqual(events.loc[3, 'start'], events.loc[2, 'finish'])
        self.assertEqual(baseline['node_powers'], controlled['node_powers'])

    def test_complete_pipeline_uses_disjoint_calibration_and_exports_real_model(self):
        data = json.loads((ROOT / 'examples/paper_12_nodes.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            scores = run_system(data, directory, cores=2, seconds_per_unit=.05,
                                 training_graphs=1, training_workload=40)
            self.assertEqual(scores['baseline']['makespan'], 15)
            self.assertEqual(scores['timing']['algorithm_3_candidate'], 16.5)
            self.assertGreater(scores['controlled']['lms_updates'], 0)
            self.assertEqual(scores['training']['dag_seeds'], [9000])
            self.assertEqual(scores['prediction_validation']['validation_seed'], 9001)
            self.assertEqual(set(scores['prediction_validation']),
                             {'validation_seed', 'frozen_ann', 'ann_lms', 'persistence'})
            model = ANNLMSPredictor.load(Path(directory) / 'model.json')
            self.assertEqual(model.weights.shape, (2, 18))
            self.assertEqual(model.updates, 0)
            for file in ['controlled/control.csv', 'controlled/thermal_trace_learning.csv',
                         'baseline/schedule.csv', 'validation_frozen/predictions.csv',
                         'validation_adaptive/predictions.csv', 'model_after_online_updates.json']:
                self.assertTrue((Path(directory) / file).is_file())

    def test_invalid_configs_terminate(self):
        dag = DAG({1: 1}, [])
        for settings in [dict(dt=0), dict(seconds_per_unit=0), dict(policy='unknown'),
                         dict(policy='predictive'), dict(policy='reactive', limit=47),
                         dict(policy='none', max_steps=1, dt=.01), dict(policy='none', max_time=.1),
                         dict(node_powers={1: 0}), dict(seed=True)]:
            with self.assertRaises(ValueError):
                simulate_thermal_dag(dag, priorities(dag), 1,
                                     **({'seconds_per_unit': .01, 'policy': 'none'} | settings))
        with self.assertRaisesRegex(ValueError, 'horizon'):
            simulate_thermal_dag(dag, priorities(dag), 1, seconds_per_unit=.01,
                                 policy='none', predictor=ANNLMSPredictor(1, horizon_seconds=.1))
        data = DAGGenerator(seed=9000).to_dict()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'seed'):
                run_system(data, directory, seconds_per_unit=.01)
