"""Package metadata, add-on manifests and mirrored release notes stay in sync."""

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from wyoming_rvc import __version__

ROOT = Path(__file__).resolve().parents[1]
ADDON_DIRS = ("addon", "addon-cpu", "addon-rocm")


def test_version_is_the_same_everywhere():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    for folder in ADDON_DIRS:
        # Each add-on version is the image tag Home Assistant pulls.
        addon = yaml.safe_load((ROOT / folder / "config.yaml").read_text(encoding="utf-8"))["version"]
        changelog = (ROOT / folder / "CHANGELOG.md").read_text(encoding="utf-8")
        latest = re.findall(r"^## (\d+\.\d+\.\d+)\s*$", changelog, re.MULTILINE)[0]
        assert __version__ == pyproject == addon == latest, folder


def test_profile_changelogs_match_the_project_release_notes():
    canonical = (ROOT / "addon/CHANGELOG.md").read_text(encoding="utf-8")
    for folder in ADDON_DIRS[1:]:
        assert (ROOT / folder / "CHANGELOG.md").read_text(encoding="utf-8") == canonical, folder


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def release_repo(tmp_path):
    script = tmp_path / "scripts/release.py"
    script.parent.mkdir()
    shutil.copy2(ROOT / "scripts/release.py", script)
    files = {
        "pyproject.toml": '[project]\nversion = "0.6.1"\n',
        "src/wyoming_rvc/__init__.py": '__version__ = "0.6.1"\n',
    }
    notes = "# Changelog\n\n## Unreleased\n\n- Shared release notes.\n\n## 0.6.1\n\n- Previous release.\n"
    for folder in ADDON_DIRS:
        files[f"{folder}/config.yaml"] = 'version: "0.6.1"\n'
        files[f"{folder}/CHANGELOG.md"] = notes
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(tmp_path, "init", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "Release test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "config", "tag.gpgsign", "false")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "Initial fixture")
    return tmp_path


def test_release_updates_and_tags_every_profile(release_repo):
    subprocess.run(
        [sys.executable, "scripts/release.py", "0.6.2", "Test release"],
        cwd=release_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    project = tomllib.loads((release_repo / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == "0.6.2"
    init = (release_repo / "src/wyoming_rvc/__init__.py").read_text(encoding="utf-8")
    assert re.findall(r'^__version__ = "([^"]+)"$', init, re.MULTILINE) == ["0.6.2"]
    for folder in ADDON_DIRS:
        config = yaml.safe_load((release_repo / folder / "config.yaml").read_text(encoding="utf-8"))
        assert config["version"] == "0.6.2", folder
        changelog = (release_repo / folder / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## 0.6.2\n" in changelog and "## Unreleased" not in changelog
        assert "- Shared release notes." in changelog
    assert _git(release_repo, "rev-parse", "v0.6.2^{}") == _git(release_repo, "rev-parse", "HEAD")
    assert _git(release_repo, "status", "--porcelain") == ""


def test_release_refuses_diverged_changelog_without_changing_files(release_repo):
    path = release_repo / "addon-rocm/CHANGELOG.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nProfile-specific note.\n", encoding="utf-8")
    _git(release_repo, "add", "addon-rocm/CHANGELOG.md")
    _git(release_repo, "commit", "-m", "Diverged release notes")
    before = _git(release_repo, "rev-parse", "HEAD")
    result = subprocess.run(
        [sys.executable, "scripts/release.py", "0.6.2", "Test release"],
        cwd=release_repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "changelog" in result.stderr.lower()
    assert _git(release_repo, "rev-parse", "HEAD") == before
    assert _git(release_repo, "status", "--porcelain") == ""
    assert _git(release_repo, "tag", "--list", "v0.6.2") == ""
