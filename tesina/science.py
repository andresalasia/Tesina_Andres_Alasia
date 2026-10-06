"""Scientific building blocks shared by the CLI and historical notebooks.

Energy feature rows are normalized by atom count; force rows are in eV/Å.
Importing this module performs no file writes, calculations or plotting.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np


def _canonical_map(mapping, *, name="interaction map"):
    """Use UF3's electronegativity ordering without changing caller data."""
    from uf3.data.composition import sort_interaction_symbols

    result = {}
    for key, value in mapping.items():
        if isinstance(key, str) and "-" in key:
            key = tuple(key.split("-"))
        if isinstance(key, tuple):
            key = sort_interaction_symbols(key)
        if key in result and not np.array_equal(result[key], value):
            raise ValueError(f"Conflicting canonical entries in {name}: {key}")
        result[key] = value
    return result


def make_basis(config: dict):
    """Construct the explicit two-body basis specified in a TOML config."""
    from uf3.data.composition import ChemicalSystem
    from uf3.representation.bspline import BSplineBasis

    settings = config["basis"]
    if settings.get("degree", 2) != 2:
        raise ValueError("This CLI currently supports pair potentials (degree=2).")
    system = ChemicalSystem(settings.get("elements", ["Ti", "Ba", "O"]), degree=2)
    pairs = system.interactions_map[2]
    cutoffs = _canonical_map(settings["cutoffs"], name="cutoffs")
    if set(cutoffs) != set(pairs):
        raise ValueError(f"cutoffs must cover exactly {pairs}; got {list(cutoffs)}")
    r_min = float(settings.get("r_min", 0.001))
    resolution = settings.get("resolution", 24)
    if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 1:
        raise ValueError("basis.resolution must be a positive integer.")
    if not np.isfinite(r_min) or r_min <= 0:
        raise ValueError("basis.r_min must be positive and finite.")
    if any(not np.isfinite(value) or value <= r_min for value in cutoffs.values()):
        raise ValueError("Every cutoff must be finite and greater than basis.r_min.")
    return BSplineBasis(
        system,
        r_min_map={pair: r_min for pair in pairs},
        r_max_map=cutoffs,
        resolution_map={pair: int(resolution) for pair in pairs},
        leading_trim=settings.get("leading_trim", 0),
        trailing_trim=settings.get("trailing_trim", 3),
    )


def featurize(frame, basis, workers=1):
    """Generate features in input order, closing all spawned workers on exit."""
    from uf3.representation.process import BasisFeaturizer

    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer.")
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Features require a non-empty frame with unique structure keys.")
    representation = BasisFeaturizer(basis)
    if workers == 1 or len(frame) == 1:
        features = representation.evaluate(frame, energy_key="energy", progress=False)
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers), mp_context=multiprocessing.get_context("spawn")
        ) as client:
            features = representation.evaluate_parallel(
                frame, client, energy_key="energy", n_jobs=min(len(frame), int(workers) * 4),
                shuffle=False, progress=False,
            )
    return features.loc[list(frame.index), :]


def tuples(features, keys, n_elements=3):
    """Return (X_energy, y_energy, X_force, y_force), normalizing energy/atom."""
    from uf3.regression.least_squares import dataframe_to_tuples

    keys = list(keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Evaluation/training keys must be non-empty and unique.")
    return dataframe_to_tuples(
        features.loc[keys, :], n_elements=n_elements, energy_key="energy"
    )


def apply_repulsion(model, bounds):
    """Apply the notebook's short-range correction in place and return model."""
    bounds = _canonical_map(bounds, name="repulsion lower bounds")
    for pair in model.bspline_config.interactions_map[2]:
        if pair not in bounds or not np.isfinite(bounds[pair]):
            raise ValueError(f"Missing finite repulsion lower bound for {pair}.")
        model.fix_repulsion_2b(pair, r_target=float(bounds[pair]), min_curvature=0.0)
    return model


def fit_model(basis, train_tuples, fit_config, repulsion_bounds=None):
    """Fit a fresh independent model; optionally apply the historical correction."""
    from uf3.regression.least_squares import WeightedLinearModel

    weight = float(fit_config.get("weight", 0.1))
    if not 0 <= weight <= 1:
        raise ValueError("fit.weight must be between 0 and 1 (higher favors energy).")
    if any(not np.all(np.isfinite(array)) for array in train_tuples):
        raise ValueError("Training arrays contain non-finite values.")
    if len(train_tuples[1]) < 2 or len(train_tuples[3]) < 2:
        raise ValueError("Fitting requires at least two energies and two force components.")
    if np.std(train_tuples[1]) == 0 or np.std(train_tuples[3]) == 0:
        raise ValueError("UF3 fitting requires nonzero energy and force variance.")
    regularization = {
        name: float(fit_config.get(name, default))
        for name, default in [("ridge_1b", 1e-7), ("ridge_2b", 0.0), ("curvature_2b", 1e-5)]
    }
    if any(not np.isfinite(value) or value < 0 for value in regularization.values()):
        raise ValueError("Regularization strengths must be finite and nonnegative.")
    model = WeightedLinearModel(
        basis, regularizer=basis.get_regularization_matrix(**regularization)
    )
    model.fit(*train_tuples, weight=weight)
    if not np.all(np.isfinite(model.coefficients)):
        raise ValueError("UF3 returned non-finite fitted coefficients.")
    if repulsion_bounds is not None:
        apply_repulsion(model, repulsion_bounds)
    return model


def evaluation(features, keys, model):
    """Return metrics in meV and reference/prediction vectors in eV units."""
    x_e, y_e, x_f, y_f = tuples(
        features, keys, n_elements=len(model.bspline_config.element_list)
    )
    p_e, p_f = model.predict(x_e), model.predict(x_f)
    if not len(y_e) or not len(y_f):
        raise ValueError("Evaluation requires both energy and force targets.")
    if any(not np.all(np.isfinite(array)) for array in (y_e, y_f, p_e, p_f)):
        raise ValueError("Evaluation targets or predictions contain non-finite values.")
    metrics = {
        "structures": len(y_e),
        "force_components": len(y_f),
        "rmse_energy_meV_atom": float(np.sqrt(np.mean((y_e - p_e) ** 2)) * 1000),
        "rmse_force_meV_A": float(np.sqrt(np.mean((y_f - p_f) ** 2)) * 1000),
    }
    return metrics, {"energy_ref": y_e, "energy_pred": p_e, "force_ref": y_f, "force_pred": p_f}


# A concise spelling is useful for callers using the CLI's command name.
evaluate = evaluation


def analyze_pairs(frame, basis):
    """Notebook histogram analysis; callers choose training-only vs historical."""
    from uf3.data.analyze import DataAnalyzer

    if frame.empty:
        raise ValueError("Pair analysis requires at least one structure.")
    analyzer = DataAnalyzer(basis.chemical_system, r_cut=10, bins=0.01, progress=False)
    analyzer.load_entries(frame["geometry"])
    return analyzer.analyze()


def load_model(path):
    """Load legacy ``knots`` or modern ``knots_map``, retaining every knot.

    UF3 0.4 expects canonical pair ordering and may otherwise modify a dictionary
    while iterating it. Canonicalize a fresh mapping and reject conflicting
    duplicates before asking UF3 to load coefficients. Never edit the input JSON.
    """
    from uf3.data.composition import ChemicalSystem
    from uf3.regression.least_squares import WeightedLinearModel
    from uf3.representation.bspline import BSplineBasis

    with Path(path).open(encoding="utf-8") as handle:
        dump = json.load(handle)
    if dump.get("degree", 2) != 2:
        raise ValueError("Only degree=2 model files are supported in this CLI.")
    knot_maps = [_canonical_map(dump[key], name=key) for key in ("knots", "knots_map") if key in dump]
    if not knot_maps:
        raise ValueError("Model has neither knots nor knots_map; explicit knots are required.")
    knots = knot_maps[0]
    if len(knot_maps) == 2 and (
        set(knots) != set(knot_maps[1])
        or any(not np.array_equal(value, knot_maps[1][key]) for key, value in knots.items())
    ):
        raise ValueError("Model knots and knots_map disagree.")
    system = ChemicalSystem(dump["element_list"], degree=2)
    if set(knots) != set(system.interactions_map[2]):
        raise ValueError("Model knots must define every expected pair, with no extra pairs.")
    for key, sequence in knots.items():
        values = np.asarray(sequence, dtype=float)
        if values.ndim != 1 or len(values) < 8 or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0):
            raise ValueError(f"Invalid knot sequence for {key}.")
    basis = BSplineBasis(
        system, knots_map={key: np.asarray(value, dtype=float) for key, value in knots.items()},
        leading_trim=dump.get("leading_trim", 0), trailing_trim=dump.get("trailing_trim", 3),
        knot_strategy=dump.get("knot_strategy", "linear"), offset_1b=dump.get("offset_1b", True),
    )
    coefficients = _canonical_map(dump["coefficients"], name="coefficients")
    expected = set(system.element_list) | set(system.interactions_map[2])
    if set(coefficients) != expected:
        raise ValueError("Model coefficients must define every expected element and pair.")
    if any(not np.all(np.isfinite(value)) for value in coefficients.values()):
        raise ValueError("Model contains non-finite coefficients.")
    model = WeightedLinearModel(basis)
    model.load(solution=coefficients)
    coverage = dump.get("data_coverage")
    if coverage is not None:
        coverage = np.asarray(coverage, dtype=bool)
        if coverage.shape != model.coefficients.shape:
            raise ValueError("Model data_coverage shape differs from coefficient shape.")
        model.data_coverage = coverage
    return model


def save_model(model, path):
    """Write UF3's modern knots_map schema, avoiding redundant legacy fields."""
    from uf3.util import json_io

    dump = model.as_dict()
    dump.pop("knots", None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json_io.dump_interaction_map(dump, write=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def cubic_structure(lattice_constant_ang=4.0):
    """Five-atom cubic BaTiO3 cell with the notebook's fractional coordinates."""
    from ase import Atoms

    a = float(lattice_constant_ang)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("Lattice constant must be finite and positive (Å).")
    return Atoms(
        "BaTiO3", scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]],
        cell=[a, a, a], pbc=True,
    )


def equation_of_state(model):
    """Cubic constrained minimum and finite-difference bulk, in Å and GPa.

    This is the audit's fixed-shape cubic calculation, not a relaxed phase EOS.
    B = E''(a)/(9a); 1 eV/Å³ = 160.21766208 GPa.
    """
    from scipy.optimize import minimize_scalar
    from uf3.forcefield.calculator import UFCalculator

    calc = UFCalculator(model)

    def energy(a):
        atoms = cubic_structure(a)
        atoms.calc = calc
        return float(atoms.get_potential_energy())

    minimum = minimize_scalar(energy, bounds=(3.9, 4.1), method="bounded", options={"xatol": 1e-10})
    if not minimum.success or not np.isfinite(minimum.fun):
        raise RuntimeError("Cubic energy minimization failed.")
    if min(minimum.x - 3.9, 4.1 - minimum.x) < 1e-4:
        raise RuntimeError("Cubic minimum lies at the [3.9, 4.1] Å search boundary.")
    step = 0.0005
    bulk = (energy(minimum.x + step) - 2 * minimum.fun + energy(minimum.x - step)) / step**2
    bulk *= 160.21766208 / (9 * minimum.x)
    a_values = np.linspace(3.9, 4.1, 41)
    return {
        "a0_A": float(minimum.x), "bulk_GPa": float(bulk), "energy_min_eV": float(minimum.fun),
        "difference_step_A": step, "lattice_constants_A": a_values.tolist(),
        "volumes_A3": (a_values ** 3).tolist(), "energies_eV": [energy(a) for a in a_values],
        "method": "cubic fixed-shape minimum; B=E''(a)/(9a)",
    }


def phonons(model, lattice_constant_ang=4.0, displacement_ang=0.05, supercell=3):
    """Return (force_constants, seekpath_data, bands) with frequencies in THz."""
    from uf3.forcefield.calculator import UFCalculator

    if not np.isfinite(displacement_ang) or displacement_ang <= 0:
        raise ValueError("Phonon displacement must be finite and positive (Å).")
    if isinstance(supercell, bool) or int(supercell) != supercell or supercell < 1:
        raise ValueError("Phonon supercell must be a positive integer.")
    result = UFCalculator(model).get_phonon_data(
        atoms=cubic_structure(lattice_constant_ang), n_super=int(supercell), disp=float(displacement_ang)
    )
    if result[0] is None or not result[2]:
        raise RuntimeError("Phonopy did not produce force constants and phonon bands.")
    return result


def export_table(model, path, grid=1000):
    """Export LAMMPS metal tables, retaining UF3's factor-two pair convention.

    Pair tables exclude the one-body energy constants; forces are unchanged,
    while comparing absolute energies requires adding sum(n_element * c_1b).
    """
    from uf3.forcefield.lammps import export_tabulated_potential
    from uf3.regression.least_squares import arrange_coefficients

    if isinstance(grid, bool) or int(grid) != grid or grid < 2:
        raise ValueError("LAMMPS table grid must be an integer >= 2.")
    basis = model.bspline_config
    if basis.degree != 2:
        raise ValueError("LAMMPS table export supports pair potentials only.")
    coefficients = arrange_coefficients(model.coefficients, basis)
    tables = [
        export_tabulated_potential(basis.knots_map[pair], coefficients[pair], pair, grid=int(grid), rounding=12)
        for pair in basis.interactions_map[2]
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n\n".join(tables) + "\n", encoding="utf-8")
    return path
