"""Lectura, procedencia y particiones que conservan los índices originales."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


DATASET_SHA256 = "17bac1203b60d2bb19451c877daacaf36e993d8c89953920e0835f4ce6a9a5b3"
SOURCES = [(0, 100, "NVT"), (100, 618, "initial"), (618, 828, "cubic_NPT"),
           (828, 1038, "tetragonal_NPT"), (1038, 1248, "orthorhombic_NPT"),
           (1248, 1458, "rhombohedral_NPT")]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_hash(atoms):
    digest = hashlib.sha256()
    for array in [atoms.numbers, atoms.positions, np.asarray(atoms.cell), atoms.pbc]:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_data(config):
    from ase.io import read
    from uf3.data.io import prepare_dataframe_from_lists

    path = Path(config["data"]["trajectory"])
    structures = read(path, index=":")
    if not structures:
        raise ValueError("El archivo de estructuras está vacío")
    energies, forces = [], []
    elements = set(config["basis"]["elements"])
    for i, atoms in enumerate(structures):
        if not len(atoms) or not set(atoms.get_chemical_symbols()) <= elements:
            raise ValueError(f"Composición no admitida en estructura {i}")
        energy, force = atoms.get_potential_energy(), atoms.get_forces()
        if not np.isfinite(energy) or not np.isfinite(force).all():
            raise ValueError(f"Etiquetas no finitas en estructura {i}")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
            raise ValueError(f"Geometría no finita en estructura {i}")
        energies.append(energy)
        forces.append(force)
    frame = prepare_dataframe_from_lists(structures, prefix="dft", energies=energies, forces=forces)
    checksum = file_hash(path)
    strata_path = config["data"].get("strata")
    if strata_path:
        labels = json.loads(Path(strata_path).read_text())
        if not isinstance(labels, list) or len(labels) != len(structures) or not all(isinstance(x, str) for x in labels):
            raise ValueError("data.strata debe apuntar a una lista JSON con una etiqueta por estructura")
    elif checksum == DATASET_SHA256 and len(structures) == 1458:
        labels = [label for lo, hi, label in SOURCES for _ in range(lo, hi)]
    elif config["run"]["protocol"] == "grouped":
        raise ValueError("Dataset distinto del original: proveer data.strata para no inventar su procedencia")
    else:
        labels = ["unknown"] * len(structures)
    return structures, frame, labels, checksum


def grouped_split(fingerprints, labels, seed=42):
    groups = defaultdict(list)
    for i, fingerprint in enumerate(fingerprints):
        groups[fingerprint].append(i)
    strata = defaultdict(list)
    for members in groups.values():
        # Keep cross-source duplicate groups together, in their own composite stratum.
        label = tuple(sorted({labels[i] for i in members}))
        strata[label].append(members)
    result = {"train": [], "validation": [], "test": []}
    rng = np.random.default_rng(seed)
    for label in sorted(strata):
        members = strata[label]
        order = rng.permutation(len(members))
        n = len(order)
        if n < 3:
            raise ValueError(f"Estrato {label} con menos de tres geometrías distintas; no permite tres particiones")
        n_train = min(max(1, int(round(n * 0.7))), n - 2)
        n_validation = min(max(1, int(round(n * 0.15))), n - n_train - 1)
        for name, selection in [("train", order[:n_train]),
                                ("validation", order[n_train:n_train + n_validation]),
                                ("test", order[n_train + n_validation:])]:
            for group_index in selection:
                result[name].extend(members[group_index])
    return {name: sorted(indices) for name, indices in result.items()}


def make_splits(structures, labels, config, smoke=False):
    fingerprints = [geometry_hash(atoms) for atoms in structures]
    if config["run"]["protocol"] == "historical":
        train = np.atleast_1d(np.loadtxt(config["data"]["training_indices"], dtype=int)).tolist()
        if len(set(train)) != len(train) or any(i < 0 or i >= len(structures) for i in train):
            raise ValueError("Índices históricos repetidos o fuera de rango")
        # Match notebook DataFrame.index.difference: lexicographic dft_* order.
        training = set(train)
        holdout = sorted((i for i in range(len(structures)) if i not in training), key=lambda i: f"dft_{i}")
        result = {"train": train, "holdout": holdout}
    else:
        result = grouped_split(fingerprints, labels, config["run"]["seed"])
    validate_splits(result, fingerprints, grouped=config["run"]["protocol"] == "grouped")
    full = {name: list(indices) for name, indices in result.items()}
    if smoke:
        result = {}
        for name, indices in full.items():
            counts = defaultdict(int)
            chosen = []
            for i in indices:
                if counts[labels[i]] < 2:
                    chosen.append(i)
                    counts[labels[i]] += 1
            result[name] = chosen
    return result, full


def validate_splits(splits, fingerprints, grouped=True):
    assigned = set()
    geometry_sets = []
    for name, indices in splits.items():
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"Partición {name} vacía o con índices repetidos")
        if assigned.intersection(indices) or any(i < 0 or i >= len(fingerprints) for i in indices):
            raise ValueError("Particiones con índices compartidos o fuera de rango")
        assigned.update(indices)
        current = {fingerprints[i] for i in indices}
        if grouped and any(current & other for other in geometry_sets):
            raise ValueError("Geometrías compartidas entre particiones agrupadas")
        geometry_sets.append(current)


def audit_data(structures, labels, splits):
    groups = defaultdict(list)
    for i, atoms in enumerate(structures):
        groups[geometry_hash(atoms)].append(i)
    duplicates = [indices for indices in groups.values() if len(indices) > 1]
    energies = [a.get_potential_energy() for a in structures]
    forces = [a.get_forces() for a in structures]
    overlapping = {}
    for left, left_indices in splits.items():
        for right, right_indices in splits.items():
            if left >= right:
                continue
            left_hash = {geometry_hash(structures[i]) for i in left_indices}
            overlap = [i for i in right_indices if geometry_hash(structures[i]) in left_hash]
            overlapping[f"{left}/{right}"] = overlap
    return {
        "structures": len(structures), "unique_exact_geometries": len(groups),
        "duplicate_groups": duplicates,
        "label_discrepancy_groups": [
            {"indices": g,
             "max_energy_difference_eV": float(np.ptp([energies[i] for i in g])),
             "max_force_difference_eV_A": float(max(np.max(np.abs(forces[g[0]] - forces[i])) for i in g))}
            for g in duplicates
            if np.ptp([energies[i] for i in g]) > 0 or any(np.any(forces[g[0]] != forces[i]) for i in g)
        ],
        "partition_provenance_counts": {name: {label: sum(labels[i] == label for i in indices)
                                                for label in sorted(set(labels))}
                                        for name, indices in splits.items()},
        "redundant_records": sum(len(group) - 1 for group in duplicates),
        "duplicate_max_energy_difference_eV": max((float(np.ptp([energies[i] for i in g])) for g in duplicates), default=0.0),
        "duplicate_max_force_difference_eV_A": max((float(np.max(np.abs(forces[g[0]] - forces[i]))) for g in duplicates for i in g), default=0.0),
        "shared_geometry_indices": overlapping,
        "partition_counts": {name: len(indices) for name, indices in splits.items()},
        "provenance_counts": {label: labels.count(label) for label in sorted(set(labels))},
        "limitations": "Agrupación exacta en el orden de átomos almacenado; no prueba independencia de trayectorias ni equivalencia bajo simetrías.",
    }
