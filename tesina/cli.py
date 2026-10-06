"""Comandos reproducibles: importar este módulo no calcula ni escribe archivos."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from .artifacts import create_run, environment, write_json
from .config import load_config


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("debe ser un entero positivo")
    return result


def parser():
    root = argparse.ArgumentParser(prog="tesina", description="Potenciales de pares de BaTiO3 · UF3")
    root.add_argument("--version", action="version", version="tesina 0.2.0")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in [("analyze", "Auditar datos y particiones"), ("train", "Ajustar y evaluar sin tocar test"),
                            ("evaluate", "Evaluar un modelo explícito"), ("search", "Buscar candidatos con validación"),
                            ("phonons", "Calcular fonones de un modelo explícito"), ("export", "Exportar tabla LAMMPS")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output-dir", type=Path, help="Directorio raíz; se crea una ejecución única")
        command.add_argument("--protocol", choices=["historical", "grouped"])
        if name in {"analyze", "train", "evaluate", "search"}:
            command.add_argument("--smoke", action="store_true", help="Muestreo reducido después de partir los datos")
        if name in {"train", "search"}:
            command.add_argument("--workers", type=positive_int, default=1)
        if name in {"evaluate", "phonons", "export"}:
            command.add_argument("--model", required=True, type=Path, help="Modelo JSON; ruta relativa al directorio actual")
        if name == "evaluate":
            command.add_argument("--split", choices=["train", "validation", "test", "holdout"], help="Por defecto: validation o holdout histórico")
            command.add_argument("--eos", action="store_true", help="Comprobar red cúbica y bulk")
        if name == "search":
            command.add_argument("--resume", type=Path, help="Directorio exacto de una búsqueda anterior")
            command.add_argument("--max-candidates", type=positive_int, help="Limitar la grilla; puede ampliarse al reanudar")
        if name == "export":
            command.add_argument("--grid", type=positive_int, default=1000)
    return root


def direct_predictions(structures, indices, model):
    import numpy as np
    from uf3.forcefield.calculator import UFCalculator

    energy_ref, energy_pred, force_ref, force_pred = [], [], [], []
    calculator = UFCalculator(model)
    for i in indices:
        original = structures[i]
        atoms = original.copy()
        atoms.calc = calculator
        energy_ref.append(original.get_potential_energy() / len(original))
        energy_pred.append(atoms.get_potential_energy() / len(atoms))
        force_ref.extend(original.get_forces().ravel())
        force_pred.extend(atoms.get_forces().ravel())
    predictions = {key: np.asarray(value) for key, value in locals().items()
                   if key in {"energy_ref", "energy_pred", "force_ref", "force_pred"}}
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Predicciones no finitas")
    metrics = {"structures": len(indices), "force_components": len(force_ref),
               "rmse_energy_meV_atom": float(np.sqrt(np.mean((predictions["energy_ref"] - predictions["energy_pred"]) ** 2)) * 1000),
               "rmse_force_meV_A": float(np.sqrt(np.mean((predictions["force_ref"] - predictions["force_pred"]) ** 2)) * 1000)}
    return metrics, predictions


def save_predictions(output, name, predictions, indices, labels, atom_counts, component_major=False):
    import numpy as np
    predictions = dict(predictions)
    if component_major:
        for key in ["force_ref", "force_pred"]:
            blocks, offset = [], 0
            for count in atom_counts:
                blocks.append(predictions[key][offset:offset + 3 * count].reshape(3, count).T.ravel())
                offset += 3 * count
            if offset != len(predictions[key]):
                raise ValueError("Cantidad de fuerzas incompatible con las estructuras")
            predictions[key] = np.concatenate(blocks)
    np.savez_compressed(output / f"{name}_predictions.npz", indices=indices, atom_counts=atom_counts,
                        force_order="structure_atom_xyz", **predictions)
    with (output / f"{name}_energies.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["structure_index", "source", "reference_eV_atom", "prediction_eV_atom"])
        writer.writerows((i, labels[i], ref, pred) for i, ref, pred in zip(indices, predictions["energy_ref"], predictions["energy_pred"]))


def save_metrics(output, metrics):
    write_json(output / "metrics.json", metrics)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "structures", "force_components", "rmse_energy_meV_atom", "rmse_force_meV_A"])
        writer.writerows([name, values["structures"], values["force_components"], values["rmse_energy_meV_atom"], values["rmse_force_meV_A"]] for name, values in metrics.items())


def _identity(config, args, hashes):
    effective = {k: v for k, v in config.items() if k != "config_path"}
    effective["run"] = {k: v for k, v in config["run"].items() if k != "output_dir"}
    value = {"config": effective, "hashes": hashes, "smoke": getattr(args, "smoke", False), "command": args.command}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main(argv=None):
    args = parser().parse_args(argv)
    output = None
    started = time.monotonic()
    manifest = {}
    warnings = []
    summary = {}
    try:
        from . import science, reporting
        from .data import load_data, make_splits, audit_data, file_hash
        config = load_config(args.config, args.protocol)
        hashes = {"config_sha256": file_hash(args.config)}
        if hasattr(args, "model"):
            hashes["input_model_sha256"] = file_hash(args.model)
        if args.command in {"analyze", "train", "evaluate", "search"}:
            print("Leyendo datos y construyendo particiones…", flush=True)
            structures, frame, labels, checksum = load_data(config)
            splits, full_splits = make_splits(structures, labels, config, args.smoke)
            hashes["trajectory_sha256"] = checksum
            for key in ["training_indices", "strata"]:
                if config["data"].get(key):
                    hashes[f"{key}_sha256"] = file_hash(config["data"][key])
            if args.command == "evaluate":
                split = args.split or ("holdout" if config["run"]["protocol"] == "historical" else "validation")
                if split not in splits:
                    raise ValueError(f"La partición {split} no existe en el protocolo {config['run']['protocol']}")
            warnings.append("Las geometrías se agrupan exactamente con el orden de átomos almacenado. Falta establecer independencia entre trayectorias.")
        if getattr(args, "smoke", False):
            warnings.append("PRUEBA REDUCIDA (--smoke): no representa los resultados del conjunto completo ni una reproducción histórica.")
        if config["run"]["protocol"] == "historical":
            warnings.append("Protocolo histórico: hay geometrías duplicadas entre train y holdout. La corrección repulsiva usa estadísticas del conjunto completo, como el notebook.")
        signature = _identity(config, args, hashes)
        # Validate before touching an earlier run, including its manifest and partitions.
        resume = getattr(args, "resume", None)
        candidate_output = create_run(config, args.command, args.output_dir, resume)
        if resume:
            previous = json.loads((candidate_output / "manifest.json").read_text())
            if previous.get("signature") != signature:
                raise ValueError("No se puede reanudar: cambiaron configuración, hashes o modo smoke")
        output = candidate_output
        print(f"Resultados: {output}", flush=True)
        manifest = {"command": args.command, "status": "running", "signature": signature,
                    "smoke": getattr(args, "smoke", False), "protocol": config["run"]["protocol"],
                    "arguments": vars(args), "hashes": hashes, **environment(Path(config["config_path"]).parent)}
        write_json(output / "manifest.json", manifest)
        write_json(output / "effective_config.json", config)
        summary["ejecución"] = {"comando": args.command, "protocolo": config["run"]["protocol"], "prueba_reducida": getattr(args, "smoke", False)}
        if args.command in {"analyze", "train", "evaluate", "search"}:
            write_json(output / "splits.json", {"full": full_splits, "active": splits})
            audit = audit_data(structures, labels, full_splits)
            write_json(output / "data_audit.json", audit)
            summary["particiones_completas"] = audit["partition_counts"]
            summary["particiones_activas"] = {k: len(v) for k, v in splits.items()}
            evaluation_split = "holdout" if "holdout" in splits else "validation"
        if args.command in {"analyze", "train", "search"}:
            basis = science.make_basis(config)
            statistics_indices = list(range(len(frame))) if config["run"]["protocol"] == "historical" else splits["train"]
            if args.smoke and config["run"]["protocol"] == "historical":
                statistics_indices = sorted(set(sum(splits.values(), [])))
            print(f"Analizando distancias de {len(statistics_indices)} estructuras…", flush=True)
            pair_analysis = science.analyze_pairs(frame.iloc[statistics_indices], basis)
            bounds = pair_analysis["lower_bounds"] if config["fit"]["repulsion"] else None
            write_json(output / "pair_statistics.json", {"indices": statistics_indices,
                       "lower_bounds_A": {"-".join(k): v for k, v in pair_analysis["lower_bounds"].items()}})
        if args.command == "analyze":
            reporting.plot_dataset([structures[i] for i in statistics_indices], output)
            reporting.plot_pairs(pair_analysis, output)
            summary["auditoría"] = audit
            summary["estadísticas"] = {"estructuras": len(statistics_indices), "origen": "todo el conjunto histórico" if config["run"]["protocol"] == "historical" else "solo entrenamiento"}
        elif args.command == "train":
            used = splits["train"] + splits[evaluation_split]
            print(f"Calculando descriptores de {len(used)} estructuras con {args.workers} proceso(s)…", flush=True)
            features = science.featurize(frame.iloc[used], basis, args.workers)
            training = science.tuples(features, frame.index[splits["train"]], len(config["basis"]["elements"]))
            print("Ajustando modelo…", flush=True)
            model = science.fit_model(basis, training, config["fit"])
            before = {name: science.evaluation(features, frame.index[splits[name]], model)[0] for name in ["train", evaluation_split]}
            if bounds is not None:
                science.apply_repulsion(model, bounds)
            science.save_model(model, output / "model.json")
            metrics = {}
            for name in ["train", evaluation_split]:
                metrics[name], predictions = science.evaluation(features, frame.index[splits[name]], model)
                save_predictions(output, name, predictions, splits[name], labels, [len(structures[i]) for i in splits[name]], component_major=True)
                reporting.plot_predictions(predictions, output, prefix=f"{name}_")
            write_json(output / "metrics_before_repulsion.json", before)
            save_metrics(output, metrics)
            summary["métricas"] = metrics
            reporting.plot_potentials(model, output)
        elif args.command == "evaluate":
            model = science.load_model(args.model)
            print(f"Evaluando {len(splits[split])} estructuras de {split}…", flush=True)
            metrics, predictions = direct_predictions(structures, splits[split], model)
            save_predictions(output, split, predictions, splits[split], labels, [len(structures[i]) for i in splits[split]])
            save_metrics(output, {split: metrics})
            summary["métricas"] = {split: metrics}
            reporting.plot_predictions(predictions, output, prefix=f"{split}_")
            reporting.plot_potentials(model, output)
            warnings.append("La partición de evaluación no garantiza que el modelo externo nunca haya visto esas estructuras; revisar su procedencia.")
            if args.eos:
                eos = science.equation_of_state(model)
                write_json(output / "eos.json", eos)
                summary["red_y_bulk"] = {k: eos[k] for k in ["a0_A", "bulk_GPa", "energy_min_eV"]}
                reporting.plot_eos(eos, output)
        elif args.command == "search":
            from .search import run_search
            rows = run_search(config, frame, splits, output, repulsion_bounds=bounds, workers=args.workers,
                              resume=bool(resume), max_candidates=args.max_candidates,
                              progress=lambda message: print(message, flush=True))
            summary["candidatos"] = rows
            complete = [row for row in rows if row["status"] == "complete"]
            if complete:
                plt = reporting._pyplot()
                fig, ax = plt.subplots(figsize=(7, 5))
                x = [row["metrics"]["rmse_energy_meV_atom"] for row in complete]
                y = [row["metrics"]["rmse_force_meV_A"] for row in complete]
                ax.scatter(x, y)
                for row, a, b in zip(complete, x, y):
                    ax.annotate(row["candidate_id"][:6], (a, b), fontsize=8)
                ax.set(xlabel="RMSE energía (meV/átomo)", ylabel="RMSE fuerzas (meV/Å)", title=f"Evaluación en {evaluation_split}")
                reporting._save(fig, output / "search_scores.png")
            if any(row["status"] != "complete" for row in rows):
                raise RuntimeError("Fallaron candidatos; ver search_checkpoint.json y reanudar tras corregir la causa")
        elif args.command == "phonons":
            import numpy as np
            model = science.load_model(args.model)
            print("Calculando constantes de fuerza y bandas de fonones…", flush=True)
            force_constants, path_data, bands_dict = science.phonons(model, **config["phonons"])
            np.save(output / "force_constants.npy", force_constants)
            write_json(output / "phonon_path.json", path_data)
            eigenvectors = bands_dict.get("eigenvectors")
            if eigenvectors is not None:
                np.savez_compressed(output / "phonon_eigenvectors.npz", **{f"segment_{i}": v for i, v in enumerate(eigenvectors)})
            write_json(output / "phonon_bands_THz.json", {k: v for k, v in bands_dict.items() if k != "eigenvectors"})
            summary["fonones"] = config["phonons"]
            warnings.extend(["Frecuencias negativas representan modos imaginarios; la fase cúbica puede ser inestable.",
                             "Perfil del notebook: a=4 Å, desplazamiento=0,05 Å, supercelda 3×3×3. El PDF describe a=4,012 Å y 0,05 % de a; son desplazamientos distintos."])
            reporting.plot_phonons(path_data, bands_dict, output)
        elif args.command == "export":
            model = science.load_model(args.model)
            science.export_table(model, output / "potential.table", grid=args.grid)
            summary["tabla"] = {"puntos_por_par": args.grid, "energía": "eV", "distancia": "Å", "fuerza": "eV/Å"}
            warnings.append("Formato tabulado histórico de LAMMPS con factor 2 de UF3. No incluye las constantes de un cuerpo: cambia el cero de energía absoluta.")
            reporting.plot_potentials(model, output)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started)
        if (output / "model.json").is_file():
            manifest["hashes"]["output_model_sha256"] = file_hash(output / "model.json")
        for artifact in ["splits.json", "effective_config.json"]:
            if (output / artifact).is_file():
                manifest["hashes"][artifact] = file_hash(output / artifact)
        write_json(output / "manifest.json", manifest)
        reporting.write_report(output, f"Tesina · {args.command}", summary, warnings=warnings)
        print(f"Completado. Informe: {output / 'index.html'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if output is not None:
            from .reporting import write_report
            manifest.update(status="interrupted" if interrupted else "failed", elapsed_seconds=time.monotonic() - started,
                            error={"type": type(error).__name__, "message": str(error)})
            write_json(output / "manifest.json", manifest)
            (output / "error.txt").write_text(traceback.format_exc())
            write_report(output, f"Tesina · {manifest['status']}", {**summary, "error": manifest["error"]}, warnings=warnings)
        print(f"{'Interrumpido' if interrupted else 'Error'}: {error}", file=sys.stderr, flush=True)
        return 130 if interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
