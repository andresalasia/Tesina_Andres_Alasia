"""Reproducible grid search: immutable partitions and per-candidate checkpoints."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .science import evaluation, featurize, fit_model, make_basis, save_model, tuples

ENERGY = "rmse_energy_meV_atom"
FORCES = "rmse_force_meV_A"


def _jsonable(value):
    if hasattr(value, "get_atomic_numbers"):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(value):
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path, payload):
    """Replace a checkpoint only after its complete contents reach the filesystem."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path, rows, columns=None):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def candidates(config):
    """Semantic identifiers remain stable when a grid's input order changes."""
    grid = config.get("search", {})
    fit = config.get("fit", {})
    fields = (
        grid.get("resolutions", [config["basis"]["resolution"]]),
        grid.get("ridge_1b", [fit.get("ridge_1b", 1e-6)]),
        grid.get("curvature_2b", [fit.get("curvature_2b", 1e-8)]),
        grid.get("weights", [fit.get("weight", 0.2)]),
    )
    if any(not values for values in fields):
        raise ValueError("La grilla no puede contener listas vacías.")
    found = set()
    for resolution, ridge, curvature, weight in itertools.product(*fields):
        if int(resolution) != resolution or int(resolution) < 1:
            raise ValueError("Cada resolución debe ser un entero positivo.")
        if not all(math.isfinite(float(value)) for value in (ridge, curvature, weight)):
            raise ValueError("Los hiperparámetros deben ser finitos.")
        if float(ridge) < 0 or float(curvature) < 0 or not 0 <= float(weight) <= 1:
            raise ValueError("Regularizaciones no negativas y peso entre 0 y 1 requeridos.")
        parameters = {"resolution": int(resolution), "ridge_1b": float(ridge),
                      "ridge_2b": float(fit.get("ridge_2b", 0.0)),
                      "curvature_2b": float(curvature), "weight": float(weight)}
        identity = _digest(parameters)[:16]
        if identity not in found:
            found.add(identity)
            yield identity, parameters


def _partitions(frame, splits):
    if "validation" in splits:
        evaluation_split = "validation"
    elif "holdout" in splits:
        evaluation_split = "holdout"
    else:
        raise ValueError("search requiere validation o holdout; nunca utiliza test.")
    training = list(splits.get("train", []))
    validation = list(splits[evaluation_split])
    for name, indices in (("train", training), (evaluation_split, validation)):
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"La partición {name} debe contener índices únicos y no estar vacía.")
        if any(not isinstance(index, (int, np.integer)) or index < 0 or index >= len(frame) for index in indices):
            raise ValueError(f"Índices de {name} fuera del dataset.")
    if set(training) & set(validation):
        raise ValueError("Entrenamiento y evaluación comparten índices.")
    if not frame.index.is_unique:
        raise ValueError("Las etiquetas de las estructuras deben ser únicas.")
    return training, validation, evaluation_split


def _signature(config, frame, training, validation, evaluation_split, grid, repulsion_bounds):
    # Inspect only the permitted training/evaluation rows; test labels are never read.
    indices = training + validation
    rows = [{"key": str(frame.index[index]), "values": frame.iloc[index].to_dict()} for index in indices]
    scientific_config = {key: config[key] for key in ("basis", "fit", "data", "protocol", "repulsion") if key in config}
    scientific_config["run"] = {key: config.get("run", {}).get(key) for key in ("protocol", "seed")}
    return _digest({"format_version": 1, "config": scientific_config, "evaluation_split": evaluation_split,
                    "candidates": sorted(grid, key=lambda item: item[0]),
                    "train": training, "evaluation": validation,
                    "rows": rows, "repulsion_bounds": repulsion_bounds})


def _write_tables(output_dir, records):
    rows = []
    for record in records:
        row = {"candidate_id": record["candidate_id"], "status": record["status"], **record["parameters"]}
        row.update(record.get("metrics", {}))
        row["error"] = record.get("error", {}).get("message", "")
        rows.append(row)
    columns = ["candidate_id", "status", "resolution", "ridge_1b", "ridge_2b", "curvature_2b", "weight",
               ENERGY, FORCES, "structures", "force_components", "error"]
    _atomic_csv(output_dir / "results.csv", rows, columns)
    complete = [row for row in rows if row["status"] == "complete"]
    _atomic_csv(output_dir / "ranking_energy.csv", sorted(complete, key=lambda row: (row[ENERGY], row["candidate_id"])), columns)
    _atomic_csv(output_dir / "ranking_forces.csv", sorted(complete, key=lambda row: (row[FORCES], row["candidate_id"])), columns)
    pareto = [row for row in complete if not any(
        other[ENERGY] <= row[ENERGY] and other[FORCES] <= row[FORCES]
        and (other[ENERGY] < row[ENERGY] or other[FORCES] < row[FORCES]) for other in complete)]
    _atomic_csv(output_dir / "pareto.csv", pareto, columns)


def run_search(config, frame, splits, output_dir, repulsion_bounds=None, workers=1,
               resume=False, progress=print, max_candidates=None):
    """Fit a fresh model per candidate, checkpointing success and ordinary failures.

    ``splits`` holds positional integer indices into ``frame``. ``max_candidates``
    restricts the grid to its first N entries; it does not change checkpoint identity.
    Resume skips completed candidates and retries failures. KeyboardInterrupt propagates.
    """
    training, validation, evaluation_split = _partitions(frame, splits)
    grid = list(candidates(config))
    signature = _signature(config, frame, training, validation, evaluation_split, grid, repulsion_bounds)
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates debe ser positivo.")
    output_dir = Path(output_dir)
    checkpoint = output_dir / "search_checkpoint.json"
    records = {}
    if checkpoint.exists():
        if not resume:
            raise ValueError("Ya existe una búsqueda aquí; usá --resume o un directorio nuevo.")
        previous = json.loads(checkpoint.read_text())
        if previous.get("signature") != signature:
            raise ValueError("No se puede reanudar: cambiaron configuración, datos, particiones o corrección repulsiva.")
        records = previous["candidates"]
    elif resume:
        raise ValueError("No existe search_checkpoint.json para reanudar.")
    output_dir.mkdir(parents=True, exist_ok=True)

    def persist():
        ordered = [records[identity] for identity, _ in grid if identity in records]
        _atomic_json(checkpoint, {"version": 1, "signature": signature,
                                 "evaluation_split": evaluation_split, "candidates": records})
        _write_tables(output_dir, ordered)

    persist()
    active_basis = None
    basis = features = train_tuples = None
    training_keys = frame.index[training].tolist()
    validation_keys = frame.index[validation].tolist()
    subset = frame.iloc[training + validation]
    for identity, parameters in grid[:max_candidates]:
        previous = records.get(identity)
        if previous and previous["status"] == "complete":
            model_path = output_dir / previous["model"]
            if model_path.is_file() and hashlib.sha256(model_path.read_bytes()).hexdigest() == previous.get("model_sha256"):
                continue
            raise ValueError(f"El modelo del candidato {identity} falta o cambió; no se puede reanudar.")
        if progress is not None:
            progress(f"Candidato {identity}: {parameters}")
        row = {"candidate_id": identity, "parameters": parameters, "status": "running"}
        try:
            basis_config = copy.deepcopy(config)
            basis_config["basis"]["resolution"] = parameters["resolution"]
            basis_signature = _digest(basis_config["basis"])
            if basis_signature != active_basis:
                basis = make_basis(basis_config)
                features = featurize(subset, basis, workers=workers)
                train_tuples = tuples(features, training_keys, n_elements=len(config["basis"].get("elements", ["Ti", "Ba", "O"])))
                active_basis = basis_signature
            fit_config = {**config.get("fit", {}), **{key: value for key, value in parameters.items() if key != "resolution"}}
            model = fit_model(basis, train_tuples, fit_config, repulsion_bounds=repulsion_bounds)
            metrics, _ = evaluation(features, validation_keys, model)
            training_metrics, _ = evaluation(features, training_keys, model)
            if not all(math.isfinite(float(metrics[key])) for key in (ENERGY, FORCES)):
                raise ValueError("El candidato produjo métricas no finitas.")
            candidate_dir = output_dir / "candidates" / identity
            candidate_dir.mkdir(parents=True, exist_ok=True)
            model_path = candidate_dir / "model.json"
            save_model(model, model_path)
            row.update(status="complete", metrics=metrics, training_metrics=training_metrics,
                       model=str(model_path.relative_to(output_dir)),
                       model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
        except Exception as error:
            row.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                                "traceback": traceback.format_exc()})
            if progress is not None:
                progress(f"Falló {identity}: {type(error).__name__}: {error}")
        records[identity] = row
        persist()
    return [records[identity] for identity, _ in grid if identity in records]
