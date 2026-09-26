"""Locate or download model files (Piper voice, RVC weights, RVC assets).

Nothing here is loaded into memory; this module only resolves paths.
Downloads are cached under ``/models`` and reused on later starts.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import Settings

_LOGGER = logging.getLogger(__name__)

PIPER_VOICE_PATTERN = re.compile(
    r"^(?P<family>[a-z]{2,3})_(?P<region>[A-Za-z0-9]+)-(?P<name>[^-]+)-(?P<quality>x_low|low|medium|high)$"
)
CONTENTVEC_FILES = ("Resources/embedders/contentvec/config.json", "Resources/embedders/contentvec/pytorch_model.bin")
RMVPE_FILE = "Resources/predictors/rmvpe.pt"
_EXTRACT_MARKER = ".extracted.json"
_MAX_MEMBER_BYTES = 4 << 30  # an RVC .pth/.index is well under 1 GB; guards against zip bombs


class ModelResolutionError(RuntimeError):
    """A required model file is missing, ambiguous, or could not be downloaded."""


@dataclass(frozen=True)
class PiperFiles:
    model: Path
    config: Path
    voice: str


@dataclass(frozen=True)
class RvcFiles:
    model: Path
    index: Path | None
    source: str


@dataclass(frozen=True)
class RvcAssets:
    contentvec_dir: Path
    rmvpe: Path


# --------------------------------------------------------------------------- Piper


def resolve_piper(settings: Settings) -> PiperFiles:
    """Explicit PIPER_MODEL wins; otherwise PIPER_VOICE is looked up or downloaded."""
    if settings.piper_model is not None:
        model = settings.piper_model
        config = settings.piper_config or Path(f"{model}.json")
        _require_file(model, "PIPER_MODEL")
        _require_file(config, "PIPER_CONFIG")
        return PiperFiles(model=model, config=config, voice=model.name.removesuffix(".onnx"))

    voice = settings.piper_voice
    match = PIPER_VOICE_PATTERN.match(voice)
    if not match:
        raise ModelResolutionError(
            f"PIPER_VOICE {voice!r} does not look like '<lang>_<REGION>-<name>-<quality>', e.g. es_AR-daniela-high"
        )
    lang_code = f"{match['family']}_{match['region']}"
    repo_dir = f"{match['family']}/{lang_code}/{match['name']}/{match['quality']}"
    data_dir = settings.piper_data_dir

    for base in (data_dir, data_dir / repo_dir):
        model, config = base / f"{voice}.onnx", base / f"{voice}.onnx.json"
        if _nonempty(model) and _nonempty(config):
            _LOGGER.info("Piper voice found in cache: %s", model)
            return PiperFiles(model=model, config=config, voice=voice)

    _LOGGER.info("Downloading Piper voice %s from %s", voice, settings.piper_voices_repo_id)
    from huggingface_hub import hf_hub_download

    paths = []
    for suffix in (".onnx", ".onnx.json"):
        try:
            paths.append(
                Path(
                    hf_hub_download(
                        repo_id=settings.piper_voices_repo_id,
                        filename=f"{repo_dir}/{voice}{suffix}",
                        local_dir=data_dir,
                    )
                )
            )
        except Exception as err:  # huggingface_hub raises several unrelated types
            raise ModelResolutionError(f"Could not download Piper voice {voice}{suffix}: {err}") from err
    return PiperFiles(model=paths[0], config=paths[1], voice=voice)


# --------------------------------------------------------------------------- RVC model


def resolve_rvc(settings: Settings) -> RvcFiles:
    """Resolve the RVC ``.pth`` and optional ``.index`` from local paths or a HF repo."""
    model_hint, index_hint = settings.rvc_model_file, settings.rvc_index_file

    if model_hint and Path(model_hint).is_absolute():
        model = Path(model_hint)
        _require_file(model, "RVC_MODEL_FILE")
        index = None
        if index_hint:
            index = Path(index_hint)
            if not index.is_absolute():
                raise ModelResolutionError("RVC_INDEX_FILE must be absolute when RVC_MODEL_FILE is absolute")
            _require_file(index, "RVC_INDEX_FILE")
        return RvcFiles(model=model, index=index, source=str(model.parent))

    snapshot = _snapshot(settings.rvc_repo_id, settings.rvc_revision)
    roots = [snapshot] + _extract_archives(snapshot, settings.rvc_data_dir / settings.rvc_repo_id.replace("/", "__"))
    pth = _collect(roots, ".pth")
    indexes = _collect(roots, ".index")
    _LOGGER.info(
        "RVC candidates in %s: models=%s indexes=%s",
        settings.rvc_repo_id,
        [str(p) for p in pth],
        [str(p) for p in indexes],
    )

    model = _pick(pth, model_hint, "RVC_MODEL_FILE", ".pth", required=True)
    assert model is not None
    index = _pick(indexes, index_hint, "RVC_INDEX_FILE", ".index", required=False)
    return RvcFiles(model=model.path, index=index.path if index else None, source=settings.rvc_repo_id)


@dataclass(frozen=True)
class _Candidate:
    path: Path
    rel: str

    def __str__(self) -> str:
        return self.rel


def _collect(roots: list[Path], suffix: str) -> list[_Candidate]:
    found: dict[str, _Candidate] = {}
    for root in roots:
        for path in sorted(root.rglob(f"*{suffix}")):
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
                rel = path.relative_to(root).as_posix()
                if root is not roots[0]:
                    rel = f"{root.name}.zip/{rel}"
                found[rel] = _Candidate(path=path, rel=rel)
    return list(found.values())


def _pick(
    candidates: list[_Candidate], hint: str | None, env_name: str, suffix: str, required: bool
) -> _Candidate | None:
    names = ", ".join(c.rel for c in candidates) or "none"
    if hint:
        wanted = PurePosixPath(hint.replace("\\", "/"))
        matches = [c for c in candidates if c.rel == wanted.as_posix() or PurePosixPath(c.rel).name == wanted.name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ModelResolutionError(f"{env_name}={hint!r} not found. Candidates: {names}")
        raise ModelResolutionError(f"{env_name}={hint!r} is ambiguous. Matches: {', '.join(m.rel for m in matches)}")

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        if required:
            raise ModelResolutionError(f"No {suffix} file found in the RVC repository")
        _LOGGER.warning("No %s file found; RVC will run without feature retrieval", suffix)
        return None
    raise ModelResolutionError(f"Multiple {suffix} files found; set {env_name} to one of: {names}")


def _extract_archives(snapshot: Path, dest_root: Path) -> list[Path]:
    """Extract ``.pth``/``.index`` members of every zip in the snapshot (once)."""
    extracted: list[Path] = []
    for archive in sorted(snapshot.rglob("*.zip")):
        dest = dest_root / archive.stem
        marker = dest / _EXTRACT_MARKER
        stamp = {"archive": archive.relative_to(snapshot).as_posix(), "size": archive.stat().st_size}
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == stamp:
            extracted.append(dest)
            continue

        _LOGGER.info("Extracting %s -> %s", archive.name, dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                seen: set[str] = set()
                for member in zf.infolist():
                    name = PurePosixPath(member.filename.replace("\\", "/")).name
                    if member.is_dir() or not name.lower().endswith((".pth", ".index")):
                        continue
                    if name in seen:
                        raise ModelResolutionError(f"{archive.name} has several files named {name!r}")
                    if member.file_size > _MAX_MEMBER_BYTES:
                        raise ModelResolutionError(f"{archive.name}: {name} is implausibly large")
                    seen.add(name)
                    # Flatten to the basename: prevents path traversal ("zip slip").
                    with zf.open(member) as src, open(dest / name, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        except zipfile.BadZipFile as err:
            raise ModelResolutionError(f"Corrupt archive {archive}: {err}") from err
        marker.write_text(json.dumps(stamp), encoding="utf-8")
        extracted.append(dest)
    return extracted


# --------------------------------------------------------------------------- RVC assets


def resolve_rvc_assets(settings: Settings) -> RvcAssets:
    """ContentVec embedder and RMVPE pitch model (from the Applio HF repo by default)."""
    repo, rev = settings.rvc_assets_repo_id, settings.rvc_assets_revision
    contentvec = [_hf_file(repo, name, rev) for name in CONTENTVEC_FILES]
    rmvpe = _hf_file(repo, RMVPE_FILE, rev)
    return RvcAssets(contentvec_dir=contentvec[0].parent, rmvpe=rmvpe)


# --------------------------------------------------------------------------- HF helpers


def _snapshot(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    patterns = ["*.pth", "*.index", "*.zip"]
    try:
        path = snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=patterns, local_files_only=True)
        _LOGGER.info("RVC repo %s found in cache: %s", repo_id, path)
        return Path(path)
    except Exception:
        pass
    _LOGGER.info("Downloading RVC repo %s (revision=%s)", repo_id, revision or "main")
    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=patterns))
    except Exception as err:
        raise ModelResolutionError(f"Could not download RVC repo {repo_id}: {err}") from err


def _hf_file(repo_id: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_files_only=True))
    except Exception:
        pass
    _LOGGER.info("Downloading %s from %s", filename, repo_id)
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    except Exception as err:
        raise ModelResolutionError(f"Could not download {filename} from {repo_id}: {err}") from err


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _require_file(path: Path, what: str) -> None:
    if not _nonempty(path):
        raise ModelResolutionError(f"{what} not found or empty: {path}")
