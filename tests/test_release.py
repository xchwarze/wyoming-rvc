"""The version lives in four places; a release (scripts/release.py) changes them together."""

import re
import tomllib
from pathlib import Path

import yaml

from wyoming_rvc import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_the_same_everywhere():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    # The add-on version is the image tag Home Assistant pulls: it must be the released one.
    addon = yaml.safe_load((ROOT / "addon" / "config.yaml").read_text(encoding="utf-8"))["version"]
    changelog = (ROOT / "addon" / "CHANGELOG.md").read_text(encoding="utf-8")
    latest = re.findall(r"^## (\d+\.\d+\.\d+)\s*$", changelog, re.MULTILINE)[0]
    assert __version__ == pyproject == addon == latest
