"""Focused checks for the environment and Q-learning update."""

import json
import unittest
from pathlib import Path

from environment import Environment
from q_learning import TrainingSettings, q_learning_update, train


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/baseline.json"


class QLearningTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_terminal_update_does_not_bootstrap(self):
        updated = q_learning_update(0, 100, 999, 0.2, 0.9, terminated=True)
        self.assertEqual(updated, 20)
        ordinary = q_learning_update(0, -1, 100, 0.2, 0.9, terminated=False)
        self.assertAlmostEqual(ordinary, 17.8)

    def test_custom_start_and_target_change_terminal_action_set(self):
        self.config["start_state"] = "A"
        self.config["target_state"] = "C"
        environment = Environment.from_config(self.config)
        self.assertEqual(environment.shortest_path(), ["A", "E", "D", "C"])
        self.assertEqual(environment.available_actions("C"), ())
        self.assertEqual(environment.available_actions("F"), ("B", "E"))
        self.assertEqual(environment.step("D", "C"), ("C", 100, True))
        with self.assertRaises(ValueError):
            environment.step("A", "C")

    def test_training_is_reproducible_and_records_full_table(self):
        self.config["training"]["episodes"] = 100
        environment = Environment.from_config(self.config)
        settings = TrainingSettings.from_config(self.config)
        first = train(environment, settings)
        second = train(environment, settings)
        self.assertEqual(first, second)
        self.assertEqual(len(first.records), 100)
        self.assertEqual(first.records[-1].q_values, first.final_q)
        self.assertNotIn(("F", "B"), first.final_q)
        self.assertTrue(all(value == 0 for value in first.initial_q.values()))


if __name__ == "__main__":
    unittest.main()
