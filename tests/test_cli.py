import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from html.parser import HTMLParser

import numpy as np

from tesina.cli import main, parser, save_predictions
from tesina.reporting import write_report

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in {'src', 'href'})


class CliTests(unittest.TestCase):
    def test_subcommands_and_explicit_model(self):
        for name in ['analyze', 'train', 'search']:
            args = parser().parse_args([name, '--config', 'm2.toml', '--smoke'])
            self.assertEqual(args.command, name)
        for name in ['evaluate', 'phonons', 'export']:
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args([name, '--config', 'm2.toml'])

    def test_import_creates_no_results(self):
        with tempfile.TemporaryDirectory() as temp:
            env = {**os.environ, 'PYTHONPATH': str(ROOT), 'MPLCONFIGDIR': temp + '/mpl'}
            result = subprocess.run([sys.executable, '-B', '-c', 'import tesina.cli; import src.utils'], cwd=temp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(temp) / 'data').exists())
            self.assertFalse((Path(temp) / 'out').exists())

    def test_archived_evaluation_smoke_and_offline_report(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            status = main(['evaluate', '--config', str(ROOT / 'configs/m2.toml'), '--model', str(ROOT / 'data/out/model_2.json'), '--protocol', 'historical', '--smoke', '--output-dir', temp])
            self.assertEqual(status, 0)
            output = next(Path(temp).iterdir())
            manifest = json.loads((output / 'manifest.json').read_text())
            splits = json.loads((output / 'splits.json').read_text())
            metrics = json.loads((output / 'metrics.json').read_text())
            self.assertTrue(manifest['smoke'])
            self.assertEqual(len(splits['full']['holdout']), 438)
            self.assertEqual(metrics['holdout']['structures'], 12)
            self.assertTrue((output / 'metrics.csv').exists())
            links = Links()
            html = (output / 'index.html').read_text()
            links.feed(html)
            self.assertIn('PRUEBA REDUCIDA', html)
            for link in links.links:
                self.assertNotIn('://', link)
                self.assertTrue((output / link).is_file(), link)

    def test_resume_mismatch_preserves_existing_run(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            path = Path(temp) / 'manifest.json'
            original = '{"signature": "incompatible", "status": "complete"}'
            path.write_text(original)
            result = main(['search', '--config', str(ROOT / 'configs/search_demo.toml'), '--smoke', '--resume', temp])
            self.assertEqual(result, 1)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_complex_eigenvectors_are_stored_without_pickle(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            bands = {'eigenvectors': [np.ones((2, 3, 3), dtype=complex) * (1 + 2j)], 'frequencies': [np.zeros((2, 3))]}
            with patch('tesina.science.phonons', return_value=(np.zeros((1, 1, 3, 3)), {}, bands)), patch('tesina.reporting.plot_phonons'):
                result = main(['phonons', '--config', str(ROOT / 'configs/m2.toml'), '--model', str(ROOT / 'data/out/model_2.json'), '--output-dir', temp])
            self.assertEqual(result, 0)
            output = next(Path(temp).iterdir())
            np.testing.assert_array_equal(np.load(output / 'phonon_eigenvectors.npz', allow_pickle=False)['segment_0'], bands['eigenvectors'][0])
            self.assertNotIn('eigenvectors', json.loads((output / 'phonon_bands_THz.json').read_text()))

    def test_force_storage_matches_ase_for_different_structure_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            canonical = [np.arange(6).reshape(2, 3), np.arange(9).reshape(3, 3) + 20]
            native = np.concatenate([a.T.ravel() for a in canonical])
            predictions = {'energy_ref': np.array([1., 2.]), 'energy_pred': np.array([1., 2.]),
                           'force_ref': native, 'force_pred': native + .5}
            save_predictions(Path(temp), 'holdout', predictions, [0, 1], ['a', 'b'], [2, 3], component_major=True)
            saved = np.load(Path(temp) / 'holdout_predictions.npz', allow_pickle=False)
            expected = np.concatenate([a.ravel() for a in canonical])
            np.testing.assert_array_equal(saved['force_ref'], expected)
            np.testing.assert_array_equal(saved['force_pred'], expected + .5)
            np.testing.assert_array_equal(predictions['force_ref'], native)
            self.assertEqual(str(saved['force_order']), 'structure_atom_xyz')

    def test_report_escapes_external_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            output = write_report(temp, '<script>label</script>', {'input': '<b>text</b>'}, warnings=['<img src=x>'])
            html = output.read_text()
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertIn('&lt;img', html)


if __name__ == '__main__':
    unittest.main()
