"""Archivos de ejecución y escritura atómica de manifiestos."""

from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import uuid


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Tipo no serializable: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def create_run(config, command, output_dir=None, resume=None):
    if resume:
        path = Path(resume).expanduser().resolve()
        if not path.is_dir() or not (path / "manifest.json").is_file():
            raise ValueError("--resume requiere un directorio de ejecución existente")
        return path
    root = Path(output_dir).expanduser().resolve() if output_dir else Path(config["run"]["output_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{command}-{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment(repo_path=None):
    versions = {"python": platform.python_version()}
    for name in ["uf3", "numpy", "scipy", "pandas", "ase", "phonopy", "matplotlib", "numba"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    try:
        repo = Path(repo_path).resolve() if repo_path else Path.cwd()
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    return {"versions": versions, "git_commit": commit, "tracked_changes": dirty,
            "package_source_sha256": source_hashes}
