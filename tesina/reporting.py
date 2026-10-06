"""Informes locales y gráficos; ningún cálculo científico depende del HTML."""

from copy import deepcopy
import html
import json
from pathlib import Path

import numpy as np

from .artifacts import json_default


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    plt = _pyplot()
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
    finally:
        plt.close(fig)
    return Path(path)


def plot_predictions(predictions, output_dir, prefix=""):
    plt = _pyplot()
    paths = []
    for kind, label in [("energy", "Energía (eV/átomo)"), ("force", "Componente de fuerza (eV/Å)")]:
        actual = np.asarray(predictions[f"{kind}_ref"]).ravel()
        predicted = np.asarray(predictions[f"{kind}_pred"]).ravel()
        if not actual.size:
            continue
        # Subsampling affects visualization only; metrics use every prediction.
        indices = np.linspace(0, len(actual) - 1, min(10000, len(actual)), dtype=int)
        x, y = actual[indices], predicted[indices]
        fig, ax = plt.subplots(figsize=(6, 5))
        artist = ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="orangered", linewidth=1)
        ax.set(xlabel=f"Referencia · {label}", ylabel=f"Predicción · {label}", title=f"{prefix or 'Modelo'} · {len(actual):,} valores")
        fig.colorbar(artist, ax=ax, label="Puntos visualizados")
        paths.append(_save(fig, Path(output_dir) / f"{prefix}{kind}_predictions.png"))
    return paths


def plot_potentials(model, output_dir):
    from scipy.interpolate import BSpline
    from uf3.regression.least_squares import arrange_coefficients

    plt = _pyplot()
    basis = model.bspline_config
    solutions = arrange_coefficients(model.coefficients, basis)
    pairs = list(basis.interactions_map[2])
    rows = (len(pairs) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.8 * rows), squeeze=False)
    for ax, pair in zip(axes.flat, pairs):
        knots = basis.knots_map[pair]
        spline = BSpline(knots, solutions[pair], 3, extrapolate=False)
        radii = np.linspace(knots[3], knots[-4], 600)
        ax.plot(radii, spline(radii))
        ax.set(title="–".join(pair), xlabel="Distancia (Å)", ylabel="Potencial de par UF3 (eV)")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    return [_save(fig, Path(output_dir) / "pair_potentials.png")]


def plot_dataset(structures, output_dir):
    plt = _pyplot()
    energy = np.array([a.get_potential_energy() / len(a) for a in structures])
    lengths = np.array([a.cell.lengths() for a in structures])
    angles = np.array([a.cell.angles() for a in structures])
    volume = np.array([a.get_volume() for a in structures])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(energy - energy.min(), linewidth=.7)
    axes[0, 0].set(ylabel="E − E mínima (eV/átomo)")
    axes[0, 1].plot(volume, linewidth=.7)
    axes[0, 1].set(ylabel="Volumen de la estructura (Å³)")
    for i, name in enumerate(["a", "b", "c"]):
        axes[1, 0].plot(lengths[:, i], label=name, linewidth=.7)
    for i, name in enumerate(["α", "β", "γ"]):
        axes[1, 1].plot(angles[:, i], label=name, linewidth=.7)
    axes[1, 0].set(ylabel="Longitud de celda (Å)")
    axes[1, 1].set(ylabel="Ángulos (grados)")
    for ax in axes.flat:
        ax.set(xlabel="Índice de la muestra visualizada")
        ax.grid(alpha=.2)
    axes[1, 0].legend()
    axes[1, 1].legend()
    return [_save(fig, Path(output_dir) / "dataset.png")]


def plot_pairs(analysis, output_dir):
    _pyplot()
    from uf3.util.plotting import plot_pair_distributions
    figures = plot_pair_distributions(analysis, show_cutoffs=True)
    paths = []
    try:
        for i, (fig, _) in enumerate(figures):
            paths.append(_save(fig, Path(output_dir) / f"pair_distribution_{i + 1}.png"))
    finally:
        for fig, _ in figures:
            _pyplot().close(fig)
    return paths


def plot_phonons(path_data, bands_dict, output_dir):
    plt = _pyplot()
    from scipy.constants import c
    from uf3.forcefield.properties.phonon import plot_phonon_spectrum
    converted = deepcopy(bands_dict)
    converted["frequencies"] = [np.asarray(values) * 1e10 / c for values in bands_dict["frequencies"]]
    result = plot_phonon_spectrum(path_data=path_data, bands_dict=converted, lw=.7)
    fig, ax = result
    ax.set(ylabel="Frecuencia (cm⁻¹)", xlabel="Camino en la zona de Brillouin")
    ax.axhline(0, color="grey", linewidth=.5)
    return [_save(fig, Path(output_dir) / "phonons.png")]


def plot_eos(eos, output_dir):
    plt = _pyplot()
    energies = np.asarray(eos["energies_eV"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eos["lattice_constants_A"], (energies - energies.min()) * 1000, "o-")
    ax.set(xlabel="Parámetro cúbico a (Å)", ylabel="E − E mínima (meV/celda de 5 átomos)",
           title=f"a₀ = {eos['a0_A']:.6f} Å · B = {eos['bulk_GPa']:.3f} GPa")
    return [_save(fig, Path(output_dir) / "equation_of_state.png")]


def write_report(output_dir, title, summary, *, warnings=None):
    output_dir = Path(output_dir)
    escaped = html.escape
    warning_html = "".join(f"<li>{escaped(str(w))}</li>" for w in (warnings or []))
    blocks = []
    for key, value in summary.items():
        if isinstance(value, dict) and all(not isinstance(v, (dict, list)) for v in value.values()):
            rows = "".join(f"<tr><th>{escaped(str(k))}</th><td>{escaped(str(v))}</td></tr>" for k, v in value.items())
            body = f"<table>{rows}</table>"
        else:
            body = f"<pre>{escaped(json.dumps(value, ensure_ascii=False, indent=2, default=json_default))}</pre>"
        blocks.append(f"<section><h2>{escaped(key.replace('_', ' '))}</h2>{body}</section>")
    files = [p for p in sorted(output_dir.rglob("*")) if p.is_file() and p.name != "index.html" and p.suffix != ".tmp"]
    links = "".join(f'<li><a href="{escaped(p.relative_to(output_dir).as_posix(), quote=True)}">{escaped(p.relative_to(output_dir).as_posix())}</a></li>' for p in files)
    images = "".join(f'<figure><img loading="lazy" src="{escaped(p.relative_to(output_dir).as_posix(), quote=True)}" alt="{escaped(p.stem, quote=True)}"><figcaption>{escaped(p.stem.replace("_", " "))}</figcaption></figure>' for p in files if p.suffix.lower() == ".png")
    document = f'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped(title)}</title><style>
body{{margin:0;background:#f4f6f8;color:#172838;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}
header{{border-top:5px solid #147d80;padding:16px 0 24px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:20px;margin-top:0}}
section,figure,aside{{background:white;border:1px solid #dce3e9;border-radius:10px;padding:22px;margin:18px 0}}aside{{border-left:5px solid #c17a18}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #e5e9ec}}
th{{font-weight:500}}a{{color:#066c83}}img{{max-width:100%;height:auto}}figcaption{{color:#546676}}.eyebrow{{color:#147d80;letter-spacing:.12em;font-size:12px}}
</style></head><body><main><header><div class="eyebrow">TESINA · RESULTADOS REPRODUCIBLES</div><h1>{escaped(title)}</h1><p>{escaped(output_dir.name)}</p></header>
{f'<aside><strong>Alcance y limitaciones</strong><ul>{warning_html}</ul></aside>' if warning_html else ''}
{''.join(blocks)}{images}<section><h2>Archivos de esta ejecución</h2><ul>{links}</ul></section></main></body></html>'''
    path = output_dir / "index.html"
    path.write_text(document)
    return path
