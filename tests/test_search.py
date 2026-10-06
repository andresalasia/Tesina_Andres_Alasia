"""Regression tests for leakage, independent candidates and durable resumption."""
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from tesina import search


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "run": {"protocol": "grouped", "seed": 42},
            "basis": {"resolution": 24, "elements": ["Ti", "Ba", "O"], "degree": 2},
            "fit": {"ridge_1b": 1e-6, "ridge_2b": 0, "curvature_2b": 1e-8, "weight": 0.2},
            "search": {"resolutions": [24, 25], "ridge_1b": [1e-6],
                       "curvature_2b": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6], "weights": [0.2, 0.3]},
        }
        self.frame = pd.DataFrame({"energy": [1., 2., 3., 4., 999., 999.]}, index=[f"dft_{i}" for i in range(6)])
        self.splits = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
        self.fits = []
        self.features_seen = []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        mocks = {
            "make_basis": lambda config: copy.deepcopy(config["basis"]),
            "featurize": self.fake_features,
            "tuples": lambda features, keys, n_elements: tuple(keys),
            "fit_model": self.fake_fit,
            "evaluation": self.fake_evaluation,
            "save_model": self.fake_save,
        }
        for name, replacement in mocks.items():
            patcher = patch.object(search, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_features(self, frame, basis, workers):
        self.features_seen.append(frame.index.tolist())
        self.assertNotIn("dft_4", frame.index)
        self.assertNotIn("dft_5", frame.index)
        return frame

    def fake_fit(self, basis, train_tuples, fit_config, repulsion_bounds=None):
        self.assertEqual(train_tuples, ("dft_0", "dft_1"))
        model = SimpleNamespace(basis=basis, parameters=copy.deepcopy(fit_config), train=train_tuples)
        self.fits.append(model)
        return model

    @staticmethod
    def fake_evaluation(features, keys, model):
        score = model.basis["resolution"] + model.parameters["curvature_2b"] * 1e8
        return {search.ENERGY: score + model.parameters["weight"],
                search.FORCES: score - model.parameters["weight"],
                "structures": len(keys), "force_components": 6}, {}

    @staticmethod
    def fake_save(model, path):
        Path(path).write_text(json.dumps(model.parameters, sort_keys=True))

    def run_grid(self, name="run", **kwargs):
        return search.run_search(self.config, self.frame, self.splits, self.output / name,
                                 progress=None, **kwargs)

    def test_all_twenty_candidates_use_training_and_cache_each_basis(self):
        original_config = copy.deepcopy(self.config)
        original_splits = copy.deepcopy(self.splits)
        results = self.run_grid()
        self.assertEqual(len(results), 20)
        self.assertTrue(all(row["status"] == "complete" for row in results))
        self.assertEqual(len(self.fits), 20)
        self.assertEqual(len({id(model) for model in self.fits}), 20)
        self.assertEqual(len(self.features_seen), 2)
        self.assertEqual(self.config, original_config)
        self.assertEqual(self.splits, original_splits)
        self.assertEqual(len(list((self.output / "run" / "candidates").glob("*/model.json"))), 20)
        ranking = pd.read_csv(self.output / "run" / "ranking_energy.csv")
        self.assertTrue(ranking[search.ENERGY].is_monotonic_increasing)
        self.assertGreater(len(pd.read_csv(self.output / "run" / "pareto.csv")), 0)

    def test_reversing_grid_order_preserves_metrics_and_identity(self):
        forward = {row["candidate_id"]: row["metrics"] for row in self.run_grid("forward")}
        for key, values in self.config["search"].items():
            self.config["search"][key] = list(reversed(values))
        backward = {row["candidate_id"]: row["metrics"] for row in self.run_grid("backward")}
        self.assertEqual(forward, backward)
        count = len(self.fits)
        self.run_grid("forward", resume=True)
        self.assertEqual(len(self.fits), count)

    def test_resume_skips_completed_and_checks_data_config_and_split(self):
        self.run_grid(max_candidates=3)
        self.assertEqual(len(self.fits), 3)
        self.run_grid(resume=True, max_candidates=5)
        self.assertEqual(len(self.fits), 5)
        changes = (
            (self.config["fit"], "weight", 0.6),
            (self.config["run"], "seed", 43),
            (self.splits, "train", [1, 0]),
        )
        for mapping, key, value in changes:
            old = mapping[key]
            mapping[key] = value
            with self.assertRaisesRegex(ValueError, "cambiaron"):
                self.run_grid(resume=True)
            mapping[key] = old
        self.frame.loc["dft_0", "energy"] = 100.
        with self.assertRaisesRegex(ValueError, "cambiaron"):
            self.run_grid(resume=True)

    def test_interrupt_keeps_completed_candidate_and_resume_continues(self):
        calls = 0
        def interrupting_fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return self.fake_fit(*args, **kwargs)
        with patch.object(search, "fit_model", interrupting_fit):
            with self.assertRaises(KeyboardInterrupt):
                self.run_grid()
        checkpoint = json.loads((self.output / "run" / "search_checkpoint.json").read_text())
        self.assertEqual(len(checkpoint["candidates"]), 1)
        self.assertEqual(len(self.run_grid(resume=True)), 20)
        self.assertEqual(len(self.fits), 20)

    def test_failures_include_traceback_and_are_retried(self):
        calls = 0
        def failing_fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise np.linalg.LinAlgError("singular candidate")
            return self.fake_fit(*args, **kwargs)
        with patch.object(search, "fit_model", failing_fit):
            results = self.run_grid()
        failed = [row for row in results if row["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("LinAlgError", failed[0]["error"]["traceback"])
        self.assertEqual(failed[0]["error"]["message"], "singular candidate")
        retried = self.run_grid(resume=True)
        self.assertTrue(all(row["status"] == "complete" for row in retried))
        self.assertEqual(len(self.fits), 20)

    def test_resume_refuses_missing_or_modified_model(self):
        result = self.run_grid(max_candidates=1)[0]
        (self.output / "run" / result["model"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "falta o cambió"):
            self.run_grid(resume=True)

    def test_refuses_test_as_validation_and_overlapping_partitions(self):
        self.splits = {"train": [0, 1], "test": [2, 3]}
        with self.assertRaisesRegex(ValueError, "nunca utiliza test"):
            self.run_grid()
        self.splits = {"train": [0, 1], "holdout": [1, 2]}
        with self.assertRaisesRegex(ValueError, "comparten"):
            self.run_grid()

    def test_historical_holdout_is_supported(self):
        self.splits["holdout"] = self.splits.pop("validation")
        result = self.run_grid(max_candidates=1)
        self.assertEqual(result[0]["status"], "complete")
        checkpoint = json.loads((self.output / "run" / "search_checkpoint.json").read_text())
        self.assertEqual(checkpoint["evaluation_split"], "holdout")


class LegacyCompatibilityTests(unittest.TestCase):
    def test_reloading_utils_does_not_create_directories(self):
        from src import utils
        with patch("pathlib.Path.mkdir") as mkdir:
            importlib.reload(utils)
        mkdir.assert_not_called()

    def test_notebook_wrapper_uses_shared_fit_without_overwriting_training(self):
        from src import utils
        from tesina import science
        data = pd.DataFrame({"marker": [1, 1, 2]}, index=["dft_0", "dft_1", "dft_2"])
        fitted = []
        basis = SimpleNamespace()
        representation = SimpleNamespace(evaluate=lambda *args, **kwargs: data)
        def data_tuples(frame, **kwargs):
            a = frame.marker.to_numpy()
            return a, a, a, a
        def fit(basis, train, settings, repulsion_bounds=None):
            self.assertEqual(train[0].tolist(), [1, 1])
            fitted.append(settings.copy())
            return SimpleNamespace(predict=lambda x: np.asarray(x, dtype=float), coefficients=np.zeros(1))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(utils, "DATA_OUT_PATH", Path(directory)), \
             patch.object(utils, "create_res_map", return_value={}), \
             patch.object(utils.bspline, "BSplineBasis", return_value=basis), \
             patch.object(utils.process, "BasisFeaturizer", return_value=representation), \
             patch.object(utils.least_squares, "dataframe_to_tuples", side_effect=data_tuples), \
             patch.object(utils.least_squares, "arrange_coefficients", return_value={}), \
             patch.object(science, "fit_model", side_effect=fit), \
             patch.object(science, "save_model", side_effect=lambda model, path: path.write_text("{}")), \
             patch.object(utils, "plot_err") as plot_err, \
             patch.object(utils, "plot_pot") as plot_pot:
            result = utils.run_selec_hyper(SimpleNamespace(element_list=["Ti", "Ba", "O"]),
                [24, 25], [1e-6], [1e-10, 1e-9, 1e-8, 1e-7, 1e-6], [0.2, 0.3], {}, {2: {}}, {},
                data, None, "energy", 1, ["dft_0", "dft_1"], ["dft_2"], None, plot=True, zoom=True)
            self.assertEqual(len(fitted), 20)
            self.assertEqual(len(result["RMSE_E (meV/atom)"]), 20)
            self.assertTrue((utils.DATA_SELEC_HYP / "resultados.csv").is_file())
            self.assertEqual(plot_err.call_count, 20)
            self.assertTrue(all(call.kwargs["n_vecinos"] == 2 and call.kwargs["zoom"] is True for call in plot_pot.call_args_list))


if __name__ == "__main__":
    unittest.main()
