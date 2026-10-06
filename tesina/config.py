"""Configuraciones TOML explícitas, independientes del directorio de trabajo."""

from copy import deepcopy
import math
from pathlib import Path
import tomllib


DEFAULTS = {
    "run": {"protocol": "grouped", "seed": 42, "output_dir": "../data/out/runs"},
    "basis": {"elements": ["Ti", "Ba", "O"], "degree": 2, "resolution": 24,
              "r_min": 0.001, "leading_trim": 0, "trailing_trim": 3},
    "fit": {"ridge_1b": 1e-7, "ridge_2b": 0.0, "curvature_2b": 1e-5,
            "weight": 0.1, "repulsion": True},
    "phonons": {"lattice_constant_ang": 4.0, "displacement_ang": 0.05, "supercell": 3},
}


def load_config(filename, protocol=None):
    path = Path(filename).expanduser().resolve()
    with path.open("rb") as stream:
        supplied = tomllib.load(stream)
    config = deepcopy(DEFAULTS)
    for section, values in supplied.items():
        if not isinstance(values, dict):
            raise ValueError(f"La sección {section} debe ser una tabla TOML")
        config.setdefault(section, {}).update(values)
    if protocol:
        config["run"]["protocol"] = protocol
    config["config_path"] = str(path)
    for section, key in [("data", "trajectory"), ("data", "training_indices"),
                         ("data", "strata"), ("run", "output_dir")]:
        value = config.get(section, {}).get(key)
        if value:
            config[section][key] = str((path.parent / value).resolve())
    validate_config(config)
    return config


def _number(value, name, minimum=0, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    if value < minimum or (exclusive and value == minimum):
        raise ValueError(f"{name} fuera de rango")


def validate_config(config):
    if not config.get("data", {}).get("trajectory"):
        raise ValueError("Falta data.trajectory en la configuración")
    if config["run"]["protocol"] not in {"historical", "grouped"}:
        raise ValueError("run.protocol debe ser historical o grouped")
    if config["run"]["protocol"] == "historical" and not config["data"].get("training_indices"):
        raise ValueError("El protocolo historical requiere data.training_indices")
    seed = config["run"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("run.seed debe ser un entero no negativo")
    basis = config["basis"]
    if basis["degree"] != 2:
        raise ValueError("Esta versión admite únicamente potenciales de pares (degree=2)")
    if not isinstance(basis["resolution"], int) or basis["resolution"] < 4:
        raise ValueError("basis.resolution debe ser un entero >= 4")
    _number(basis["r_min"], "basis.r_min", exclusive=True)
    if not basis.get("cutoffs"):
        raise ValueError("Falta basis.cutoffs; los radios deben ser explícitos")
    elements = basis["elements"]
    if not elements or len(set(elements)) != len(elements):
        raise ValueError("basis.elements debe contener especies distintas")
    pairs = set()
    for pair, cutoff in basis["cutoffs"].items():
        symbols = pair.split("-")
        if len(symbols) != 2 or not set(symbols) <= set(elements):
            raise ValueError(f"Par desconocido en basis.cutoffs: {pair}")
        key = tuple(sorted(symbols))
        if key in pairs:
            raise ValueError(f"Par duplicado en basis.cutoffs: {pair}")
        pairs.add(key)
        _number(cutoff, f"cutoff {pair}", basis["r_min"], exclusive=True)
    expected = {tuple(sorted((a, b))) for a in elements for b in elements}
    if pairs != expected:
        raise ValueError("basis.cutoffs debe especificar todos los pares de elementos")
    for key in ["leading_trim", "trailing_trim"]:
        if not isinstance(basis[key], int) or not 0 <= basis[key] <= 3:
            raise ValueError(f"basis.{key} debe estar entre 0 y 3")
    for key in ["ridge_1b", "ridge_2b", "curvature_2b"]:
        _number(config["fit"][key], f"fit.{key}")
    _number(config["fit"]["weight"], "fit.weight", exclusive=True)
    if config["fit"]["weight"] >= 1:
        raise ValueError("fit.weight debe estar estrictamente entre 0 y 1")
    if not isinstance(config["fit"]["repulsion"], bool):
        raise ValueError("fit.repulsion debe ser booleano")
    for key in ["lattice_constant_ang", "displacement_ang"]:
        _number(config["phonons"][key], f"phonons.{key}", exclusive=True)
    if not isinstance(config["phonons"]["supercell"], int) or config["phonons"]["supercell"] < 1:
        raise ValueError("phonons.supercell debe ser un entero positivo")
    if "search" in config:
        for key in ["resolutions", "ridge_1b", "curvature_2b", "weights"]:
            values = config["search"].get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"search.{key} debe ser una lista no vacía")
            for value in values:
                _number(value, f"search.{key}")
                if key == "resolutions" and (not isinstance(value, int) or value < 4):
                    raise ValueError("search.resolutions requiere enteros >= 4")
                if key == "weights" and not 0 < value < 1:
                    raise ValueError("search.weights requiere valores entre 0 y 1")
