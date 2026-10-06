"""Small deterministic scientific checks; no full training set fit is run here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tesina import science

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/out/model_2.json"


class ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from uf3.data.io import DataCoordinator
        from uf3.forcefield.calculator import UFCalculator

        cls.model = science.load_model(ARCHIVE)
        cls.atoms = []
        energies, forces = [], []
        for i in range(4):
            atoms = science.cubic_structure(3.96 + i * 0.025)
            atoms.positions[1] += np.array([0.013 + i * 0.003, -0.007, 0.004])
            atoms.calc = UFCalculator(cls.model)
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
            cls.atoms.append(atoms)
        coordinator = DataCoordinator()
        coordinator.dataframe_from_lists(cls.atoms, energies=energies, forces=forces, prefix="sample")
        cls.frame = coordinator.consolidate()
        # Non-sorted input order exercises input-order preservation explicitly.
        cls.frame = cls.frame.iloc[[2, 0, 3, 1]]
        cls.features = science.featurize(cls.frame, cls.model.bspline_config)

    def test_archived_model_predictions_match_ase(self):
        from uf3.forcefield.calculator import UFCalculator

        metrics, predictions = science.evaluation(self.features, self.frame.index, self.model)
        self.assertLess(metrics["rmse_energy_meV_atom"], 1e-7)
        self.assertLess(metrics["rmse_force_meV_A"], 1e-7)
        expected_energy, expected_force = [], []
        for atoms in self.frame.geometry:
            candidate = atoms.copy()
            candidate.calc = UFCalculator(self.model)
            expected_energy.append(candidate.get_potential_energy() / len(candidate))
            expected_force.extend(candidate.get_forces().T.ravel())
        np.testing.assert_allclose(predictions["energy_pred"], expected_energy, atol=1e-10)
        np.testing.assert_allclose(predictions["force_pred"], expected_force, atol=1e-10)

    def test_roundtrip_preserves_knots_predictions_and_original_file(self):
        before = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = science.save_model(self.model, Path(tmp) / "model.json")
            payload = json.loads(path.read_text())
            self.assertIn("knots_map", payload)
            self.assertNotIn("knots", payload)
            loaded = science.load_model(path)
            for pair in self.model.bspline_config.interactions_map[2]:
                np.testing.assert_array_equal(
                    self.model.bspline_config.knots_map[pair], loaded.bspline_config.knots_map[pair]
                )
            np.testing.assert_array_equal(self.model.coefficients, loaded.coefficients)
            _, a = science.evaluation(self.features, self.frame.index, self.model)
            _, b = science.evaluation(self.features, self.frame.index, loaded)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(before, hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())

    def test_loader_handles_legacy_only_and_rejects_conflicting_pairs(self):
        payload = json.loads(ARCHIVE.read_text())
        payload.pop("knots_map")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(payload))
            loaded = science.load_model(path)
            np.testing.assert_array_equal(loaded.coefficients, self.model.coefficients)
            key = next(k for k in payload["coefficients"] if "-" in k and len(set(k.split("-"))) == 2)
            reverse = "-".join(reversed(key.split("-")))
            payload["coefficients"][reverse] = [v + 0.1 for v in payload["coefficients"][key]]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Conflicting canonical"):
                science.load_model(path)

    def test_loader_rejects_disagreeing_knot_schemas(self):
        payload = json.loads(ARCHIVE.read_text())
        pair = next(iter(payload["knots_map"]))
        payload["knots_map"][pair][4] += 0.01
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ambiguous.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "disagree"):
                science.load_model(path)

    def test_force_is_negative_energy_gradient(self):
        from uf3.forcefield.calculator import UFCalculator

        atoms = self.atoms[1].copy()
        atoms.calc = UFCalculator(self.model)
        reference = atoms.get_forces()
        step = 1e-5
        for atom_index, component in [(0, 0), (1, 0), (1, 1), (3, 2)]:
            center = atoms.positions[atom_index, component]
            atoms.positions[atom_index, component] = center + step
            plus = atoms.get_potential_energy()
            atoms.positions[atom_index, component] = center - step
            minus = atoms.get_potential_energy()
            atoms.positions[atom_index, component] = center
            self.assertAlmostEqual(reference[atom_index, component], -(plus - minus) / (2 * step), delta=2e-6)

    def test_spawn_features_match_serial_and_preserve_order(self):
        parallel = science.featurize(self.frame, self.model.bspline_config, workers=2)
        self.assertEqual(list(parallel.index), list(self.features.index))
        np.testing.assert_allclose(parallel.to_numpy(), self.features.to_numpy(), rtol=0, atol=1e-12)

    def test_fit_is_fresh_and_does_not_mutate_training_arrays(self):
        training = science.tuples(self.features, self.frame.index)
        snapshot = [array.copy() for array in training]
        settings = {"ridge_1b": 1e-7, "ridge_2b": 0.0, "curvature_2b": 1e-5, "weight": 0.1}
        first = science.fit_model(self.model.bspline_config, training, settings)
        second = science.fit_model(self.model.bspline_config, training, settings)
        self.assertIsNot(first, second)
        np.testing.assert_allclose(first.coefficients, second.coefficients, rtol=0, atol=1e-12)
        for previous, after in zip(snapshot, training):
            np.testing.assert_array_equal(previous, after)

    def test_table_uses_correct_double_counting_convention(self):
        from scipy.interpolate import BSpline
        from uf3.regression.least_squares import arrange_coefficients

        with tempfile.TemporaryDirectory() as tmp:
            path = science.export_table(self.model, Path(tmp) / "BTO.table", grid=7)
            sections = path.read_text().strip().split("\n\n\n")
        basis = self.model.bspline_config
        coefficients = arrange_coefficients(self.model.coefficients, basis)
        self.assertEqual(len(sections), 6)
        for pair, section in zip(basis.interactions_map[2], sections):
            rows = [line.split() for line in section.splitlines() if line and line[0].isdigit()]
            self.assertEqual(len(rows), 7)
            self.assertIn("UF_" + "-".join(pair), section)
            spline = BSpline(basis.knots_map[pair], coefficients[pair], 3)
            for row in rows:
                radius, energy, force = map(float, row[1:])
                self.assertAlmostEqual(energy, 2 * spline(radius), delta=2e-8)
                self.assertAlmostEqual(force, -2 * spline(radius, nu=1), delta=2e-7)

    def test_cubic_equation_of_state_matches_archived_reference(self):
        result = science.equation_of_state(self.model)
        self.assertAlmostEqual(result["a0_A"], 3.981419, delta=0.0001)
        self.assertAlmostEqual(result["bulk_GPa"], 137.0383, delta=0.05)


if __name__ == "__main__":
    unittest.main()
