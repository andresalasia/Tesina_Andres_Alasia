import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from tesina.config import load_config
from tesina.data import audit_data, geometry_hash, grouped_split, make_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def sample(i, energy=None):
    atoms = Atoms('BaTiO3', cell=[4 + i / 1000] * 3, scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=float(i if energy is None else energy), forces=np.zeros((5, 3)))
    return atoms


class DataTests(unittest.TestCase):
    def test_grouping_keeps_discrepant_labels_and_stratifies(self):
        structures = [sample(i) for i in range(120)] + [sample(0, 9)]
        labels = [f'source_{i // 20}' for i in range(120)] + ['source_0']
        hashes = [geometry_hash(a) for a in structures]
        splits = grouped_split(hashes, labels)
        self.assertEqual(splits, grouped_split(hashes, labels))
        validate_splits(splits, hashes)
        self.assertEqual(sum(map(len, splits.values())), 121)
        self.assertTrue(any(0 in v and 120 in v for v in splits.values()))
        for indices in splits.values():
            self.assertEqual(len({labels[i] for i in indices}), 6)
        audit = audit_data(structures, labels, splits)
        self.assertEqual(audit['redundant_records'], 1)
        self.assertEqual(audit['label_discrepancy_groups'][0]['indices'], [0, 120])
        self.assertEqual(audit['label_discrepancy_groups'][0]['max_energy_difference_eV'], 9)
        self.assertTrue(all(not x for x in audit['shared_geometry_indices'].values()))

    def test_smoke_samples_after_full_partition(self):
        config = load_config(ROOT / 'configs/m2.toml')
        structures = [sample(i) for i in range(120)]
        labels = [f'source_{i // 20}' for i in range(120)]
        active, full = make_splits(structures, labels, config, smoke=True)
        normal, _ = make_splits(structures, labels, config)
        self.assertEqual(full, normal)
        for key, values in active.items():
            self.assertEqual(len(values), 12)
            self.assertLessEqual(set(values), set(full[key]))

    def test_historical_preserves_train_order_and_lexical_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'indices.txt'
            path.write_text('4\n0\n8\n')
            config = load_config(ROOT / 'configs/m2.toml', 'historical')
            config['data']['training_indices'] = str(path)
            structures = [sample(i) for i in range(12)]
            splits, _ = make_splits(structures, ['a'] * 12, config)
            self.assertEqual(splits['train'], [4, 0, 8])
            self.assertEqual(splits['holdout'], [1, 10, 11, 2, 3, 5, 6, 7, 9])

    def test_rejects_geometry_leakage_and_unknown_cutoff(self):
        with self.assertRaisesRegex(ValueError, 'Geometrías'):
            validate_splits({'train': [0], 'test': [1]}, ['same', 'same'])
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'config.toml'
            p.write_text((ROOT / 'configs/m2.toml').read_text().replace('"O-O" = 4.42', '"O-O" = -1'))
            with self.assertRaisesRegex(ValueError, 'cutoff'):
                load_config(p)

    def test_paths_are_relative_to_configuration(self):
        config = load_config(ROOT / 'configs/m2.toml')
        self.assertEqual(Path(config['data']['trajectory']), ROOT / 'data/in/Training_set.xyz')
        self.assertEqual(Path(config['run']['output_dir']), ROOT / 'data/out/runs')


if __name__ == '__main__':
    unittest.main()
