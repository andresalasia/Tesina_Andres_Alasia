"""Verifica artefactos de evaluate/train; no inicia otro ajuste completo."""
import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED = {'rmse_energy_meV_atom': 3.8711425, 'rmse_force_meV_A': 134.4983452}


def validate(evaluation_run, training_run=None):
    evaluation_run = Path(evaluation_run)
    manifest = json.loads((evaluation_run / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['smoke'] or manifest['protocol'] != 'historical':
        raise ValueError('Se requiere evaluate histórico completo, sin --smoke')
    hashes = manifest['hashes']
    if hashes['trajectory_sha256'] != '17bac1203b60d2bb19451c877daacaf36e993d8c89953920e0835f4ce6a9a5b3':
        raise ValueError('Dataset distinto del histórico')
    if hashes['input_model_sha256'] != '75fd4cae5a96c54e9850dde8611701854b3f277240d50e84ef74014d6cac9f52':
        raise ValueError('El modelo evaluado no es el M2 archivado')
    archived = json.loads((evaluation_run / 'metrics.json').read_text())['holdout']
    if archived['structures'] != 438:
        raise ValueError('El holdout histórico debe tener 438 estructuras')
    checks = {key: {'actual': archived[key], 'expected': value, 'tolerance': 1e-4,
                    'passed': abs(archived[key] - value) <= 1e-4} for key, value in EXPECTED.items()}
    eos = json.loads((evaluation_run / 'eos.json').read_text())
    for key, expected, tolerance in [('a0_A', 3.981419, 1e-4), ('bulk_GPa', 137.0383, .05)]:
        checks[key] = {'actual': eos[key], 'expected': expected, 'tolerance': tolerance, 'passed': abs(eos[key] - expected) <= tolerance}
    result = {'archived_M2': checks, 'versions': manifest['versions']}
    if training_run:
        training_run = Path(training_run)
        training_manifest = json.loads((training_run / 'manifest.json').read_text())
        if training_manifest['status'] != 'complete' or training_manifest['smoke'] or training_manifest['protocol'] != 'historical':
            raise ValueError('Se requiere train histórico completo, sin --smoke')
        for key in ['trajectory_sha256', 'training_indices_sha256']:
            if hashes[key] != training_manifest['hashes'][key]:
                raise ValueError(f'Diferencia entre ejecuciones: {key}')
        fitted = json.loads((training_run / 'metrics.json').read_text())['holdout']
        relative = {key: abs(fitted[key] - archived[key]) / archived[key] for key in EXPECTED}
        old = np.load(evaluation_run / 'holdout_predictions.npz', allow_pickle=False)
        new = np.load(training_run / 'holdout_predictions.npz', allow_pickle=False)
        np.testing.assert_array_equal(old['indices'], new['indices'])
        np.testing.assert_allclose(old['energy_ref'], new['energy_ref'], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(old['force_ref'], new['force_ref'])
        result['refit_M2'] = {'metrics': fitted, 'relative_RMSE_difference': relative,
                            'within_one_percent': all(x <= .01 for x in relative.values()),
                            'max_energy_prediction_difference_eV_atom': float(np.max(np.abs(old['energy_pred'] - new['energy_pred']))),
                            'max_force_prediction_difference_eV_A': float(np.max(np.abs(old['force_pred'] - new['force_pred'])))}
    passed = all(check['passed'] for check in checks.values()) and result.get('refit_M2', {}).get('within_one_percent', True)
    return result, passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-run', required=True, type=Path)
    p.add_argument('--training-run', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result, passed = validate(args.evaluation_run, args.training_run)
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    print(text)
    if args.output:
        args.output.write_text(text)
    raise SystemExit(0 if passed else 1)
