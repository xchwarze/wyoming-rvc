import zipfile
from pathlib import Path

import pytest

from wyoming_rvc import model_loader
from wyoming_rvc.config import Settings
from wyoming_rvc.model_loader import ModelResolutionError, resolve_piper, resolve_rvc


def settings(tmp_path: Path, **env: str) -> Settings:
    return Settings.from_env({"MODELS_DIR": str(tmp_path / "models"), **env})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    monkeypatch.setattr(model_loader, "_snapshot", lambda repo_id, revision: snap)
    return snap


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_single_model_and_index(tmp_path, repo):
    touch(repo / "sub" / "voice.pth")
    touch(repo / "voice.index")
    files = resolve_rvc(settings(tmp_path))
    assert files.model.name == "voice.pth" and files.index.name == "voice.index"


def test_index_optional(tmp_path, repo):
    touch(repo / "voice.pth")
    assert resolve_rvc(settings(tmp_path)).index is None


def test_multiple_models_fail_with_candidates(tmp_path, repo):
    touch(repo / "a.pth")
    touch(repo / "b.pth")
    with pytest.raises(ModelResolutionError, match="a.pth, b.pth"):
        resolve_rvc(settings(tmp_path))


def test_multiple_models_selected_by_name(tmp_path, repo):
    touch(repo / "a.pth")
    touch(repo / "dir" / "b.pth")
    assert resolve_rvc(settings(tmp_path, RVC_MODEL_FILE="b.pth")).model.name == "b.pth"
    assert resolve_rvc(settings(tmp_path, RVC_MODEL_FILE="dir/b.pth")).model.name == "b.pth"


def test_multiple_indexes_require_choice(tmp_path, repo):
    touch(repo / "a.pth")
    touch(repo / "added_x.index")
    touch(repo / "trained_x.index")
    with pytest.raises(ModelResolutionError, match="RVC_INDEX_FILE"):
        resolve_rvc(settings(tmp_path))
    files = resolve_rvc(settings(tmp_path, RVC_INDEX_FILE="added_x.index"))
    assert files.index.name == "added_x.index"


def test_missing_named_file(tmp_path, repo):
    touch(repo / "a.pth")
    with pytest.raises(ModelResolutionError, match="not found"):
        resolve_rvc(settings(tmp_path, RVC_MODEL_FILE="zzz.pth"))


def test_no_model(tmp_path, repo):
    with pytest.raises(ModelResolutionError, match="No .pth"):
        resolve_rvc(settings(tmp_path))


def test_zip_archive_is_extracted_once_and_flattened(tmp_path, repo):
    archive = repo / "Teto.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/Teto_240e.pth", b"weights")
        zf.writestr("../../evil.index", b"index")
        zf.writestr("readme.txt", b"ignored")
    s = settings(tmp_path)
    files = resolve_rvc(s)
    assert files.model.read_bytes() == b"weights"
    assert files.index is not None and files.index.parent == files.model.parent
    assert not (tmp_path / "evil.index").exists()
    assert not list(files.model.parent.glob("*.txt"))
    # second call reuses the extraction
    mtime = files.model.stat().st_mtime_ns
    assert resolve_rvc(s).model.stat().st_mtime_ns == mtime


def test_absolute_model_path_skips_download(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "_snapshot", lambda *a: pytest.fail("should not download"))
    model = touch(tmp_path / "m.pth")
    index = touch(tmp_path / "m.index")
    files = resolve_rvc(settings(tmp_path, RVC_MODEL_FILE=str(model), RVC_INDEX_FILE=str(index)))
    assert files.model == model and files.index == index


def test_piper_explicit_model(tmp_path):
    model = touch(tmp_path / "v.onnx")
    touch(tmp_path / "v.onnx.json", b"{}")
    files = resolve_piper(settings(tmp_path, PIPER_MODEL=str(model)))
    assert files.config == tmp_path / "v.onnx.json"


def test_piper_explicit_model_missing_config(tmp_path):
    model = touch(tmp_path / "v.onnx")
    with pytest.raises(ModelResolutionError, match="PIPER_CONFIG"):
        resolve_piper(settings(tmp_path, PIPER_MODEL=str(model)))


def test_piper_voice_from_cache(tmp_path):
    base = tmp_path / "models" / "piper"
    touch(base / "es_AR-daniela-high.onnx")
    touch(base / "es_AR-daniela-high.onnx.json", b"{}")
    assert resolve_piper(settings(tmp_path)).model == base / "es_AR-daniela-high.onnx"


def test_piper_bad_voice_name(tmp_path):
    with pytest.raises(ModelResolutionError, match="does not look like"):
        resolve_piper(settings(tmp_path, PIPER_VOICE="daniela"))
