import json
from pathlib import Path

import pytest
import yaml

from wyoming_rvc import addon
from wyoming_rvc.config import Settings

ROOT = Path(__file__).resolve().parents[1]
PROFILES = [
    ("addon", "cuda", "wyoming_rvc", "auto"),
    ("addon-cpu", "cpu", "wyoming_rvc_cpu", "cpu"),
    ("addon-rocm", "rocm", "wyoming_rvc_rocm", "cuda"),
]


def _config(folder):
    return yaml.safe_load((ROOT / folder / "config.yaml").read_text(encoding="utf-8"))


def test_apply_options_maps_to_env_and_keeps_explicit_env(tmp_path):
    path = tmp_path / "options.json"
    path.write_text(
        json.dumps({"device": "auto", "rvc_pitch": 3, "wyoming_streaming": True, "rvc_model_file": "", "x": None})
    )
    env = {"RVC_PITCH": "5", "MODELS_DIR": "/models"}  # the image sets MODELS_DIR
    assert addon.apply_options(env, path) is True
    assert env == {"RVC_PITCH": "5", "DEVICE": "auto", "WYOMING_STREAMING": "true", "MODELS_DIR": "/data"}


def test_apply_options_noop_outside_addon(tmp_path):
    env = {}
    assert addon.apply_options(env, tmp_path / "missing.json") is False
    assert env == {}


@pytest.mark.parametrize("folder", ["addon", "addon-cpu", "addon-rocm"])
def test_probe_text_option_reaches_runtime_settings(tmp_path, folder):
    config = _config(folder)
    assert "probe_text" in config["options"] and "probe_text" in config["schema"]
    options = {**config["options"], "probe_text": "Comprobación."}
    path = tmp_path / "options.json"
    path.write_text(json.dumps(options), encoding="utf-8")
    env = {}
    assert addon.apply_options(env, path) is True
    assert Settings.from_env(env).probe_text == "Comprobación."


@pytest.mark.parametrize(("folder", "gpu", "slug", "device"), PROFILES)
def test_profiles_use_the_matching_published_image(folder, gpu, slug, device):
    config = _config(folder)
    workflow = yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["image"]["strategy"]["matrix"]["include"]
    suffixes = {entry["gpu"]: entry["suffix"] for entry in matrix}
    assert config["image"] == f"ghcr.io/xchwarze/wyoming-rvc{suffixes[gpu]}"
    assert config["slug"] == slug and config["arch"] == ["amd64"]
    assert config["options"]["device"] == device
    assert config["ports"]["10200/tcp"] is None
    assert "8080/tcp" not in config["ports"]


@pytest.mark.parametrize("folder", ["addon-cpu", "addon-rocm"])
def test_profiles_share_voice_options_and_schema(folder):
    original, variant = _config("addon"), _config(folder)
    for section in ("options", "schema"):
        assert {k: v for k, v in variant[section].items() if k != "device"} == {
            k: v for k, v in original[section].items() if k != "device"
        }
    assert variant["map"] == original["map"]
    assert variant["backup_exclude"] == original["backup_exclude"]


def test_cpu_profile_requires_no_gpu_access():
    config = _config("addon-cpu")
    assert not config.get("devices") and not config.get("video", False)
    choices = config["schema"]["device"].removeprefix("list(").removesuffix(")").split("|")
    assert set(choices) == {"auto", "cpu"}


def test_rocm_profile_requests_compute_and_render_devices():
    config = _config("addon-rocm")
    assert config["devices"] == ["/dev/kfd"] and config["video"] is True
    assert config["stage"] == "experimental"


@pytest.mark.parametrize("folder", ["addon", "addon-cpu", "addon-rocm"])
def test_profiles_do_not_request_broad_host_privileges(folder):
    config = _config(folder)
    assert not config.get("full_access") and not config.get("docker_api")
    assert not config.get("privileged") and config.get("apparmor", True) is True


def test_ci_lints_every_addon_profile():
    workflow = yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text(encoding="utf-8"))
    linted = {
        step["with"]["path"].removeprefix("./")
        for step in workflow["jobs"]["test"]["steps"]
        if step.get("uses", "").startswith("frenck/action-app-linter@")
    }
    assert linted == {folder for folder, *_ in PROFILES}


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_announce_wyoming_posts_discovery(monkeypatch):
    monkeypatch.setattr(addon.socket, "gethostname", lambda: "addon-host")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.headers["Authorization"]
        seen["body"] = json.loads(request.data)
        return _Response()

    assert addon.announce_wyoming(10200, {"SUPERVISOR_TOKEN": "t0k"}, fake_urlopen) is True
    assert seen == {
        "url": "http://supervisor/discovery",
        "auth": "Bearer t0k",
        "body": {"service": "wyoming", "config": {"uri": "tcp://addon-host:10200"}},
    }


def test_announce_wyoming_skipped_without_token_and_tolerates_errors():
    assert addon.announce_wyoming(10200, {}) is False

    def boom(*_args, **_kwargs):
        raise OSError("no supervisor")

    assert addon.announce_wyoming(10200, {"SUPERVISOR_TOKEN": "t"}, boom) is False
