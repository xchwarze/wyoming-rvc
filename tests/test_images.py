"""Image dependencies and compose examples must agree with the published profiles."""

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

from wyoming_rvc.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def _locked_dependencies(filename):
    text = (ROOT / filename).read_text(encoding="utf-8")
    return dict(re.findall(r"^([a-z0-9-]+)==([^\s\\]+)", text, re.MULTILINE))


def _compose_service(filename):
    return yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))["services"]["wyoming-rvc"]


@pytest.mark.parametrize("filename", ["requirements.lock", "requirements-cpu.lock", "requirements-rocm.lock"])
def test_image_locks_satisfy_project_dependencies(filename):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = _locked_dependencies(filename)
    for dependency in project["dependencies"]:
        requirement = Requirement(dependency)
        assert requirement.name in locked, f"{filename}: missing {requirement.name}"
        assert locked[requirement.name] in requirement.specifier, f"{filename}: incompatible {dependency}"


def test_cpu_lock_excludes_gpu_runtimes():
    locked = _locked_dependencies("requirements-cpu.lock")
    gpu_packages = [name for name in locked if name.startswith(("cuda-", "nvidia-", "rocm", "triton"))]
    assert gpu_packages == []
    assert locked["torch"].endswith("+cpu")


@pytest.mark.parametrize(
    ("filename", "gpu"),
    [("docker-compose.yml", "cuda"), ("docker-compose.cpu.yml", "cpu"), ("docker-compose.rocm.yml", "rocm")],
)
def test_compose_uses_the_published_image_profile(filename, gpu):
    workflow = yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["image"]["strategy"]["matrix"]["include"]
    profiles = {entry["gpu"]: entry["suffix"] for entry in matrix}
    service = _compose_service(filename)
    assert service["image"] == f"ghcr.io/xchwarze/wyoming-rvc{profiles[gpu]}:latest"
    build = service["build"]
    assert (build["args"]["GPU"] if isinstance(build, dict) else "cuda") == gpu


def test_cpu_compose_needs_no_gpu_even_with_the_example_env():
    service = _compose_service("docker-compose.cpu.yml")
    env = {"DEVICE": "cuda"}  # .env.example selects CUDA for the default image.
    env.update(item.split("=", 1) for item in service["environment"])
    settings = Settings.from_env(env)
    assert settings.device == "cpu" and settings.piper_use_cuda is False
    assert "devices" not in service and "deploy" not in service
    assert "127.0.0.1:8080:8080" in service["ports"]
