#!/usr/bin/env python3
"""Cut a release: set the version everywhere, turn "## Unreleased" into it, commit and tag.

  python scripts/release.py 0.5.1 "short summary for the tag"

Nothing is pushed. Push the commit and the tag together, so the add-on's new version
(Home Assistant reads it from main) and the image built from the tag appear at once:

  git push --atomic origin main v0.5.1

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# (file, pattern of the line holding the version); each must match exactly once.
VERSION_LINES = [
    ("pyproject.toml", r'^version = "[^"]+"$', 'version = "{v}"'),
    ("src/wyoming_rvc/__init__.py", r'^__version__ = "[^"]+"$', '__version__ = "{v}"'),
    ("addon/config.yaml", r'^version: "[^"]+"$', 'version: "{v}"'),
]
CHANGELOG = "addon/CHANGELOG.md"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def replaced_once(path: Path, pattern: str, replacement: str) -> str:
    text = path.read_text(encoding="utf-8")
    if len(re.findall(pattern, text, flags=re.MULTILINE)) != 1:
        sys.exit(f"{path.relative_to(ROOT)}: expected exactly one line matching {pattern!r}")
    return re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="new version, e.g. 0.5.1")
    parser.add_argument("summary", help="one line for the tag message")
    args = parser.parse_args()
    version = args.version.removeprefix("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"version must look like 1.2.3, got {args.version!r}")
    if git("status", "--porcelain"):
        sys.exit("the working tree has uncommitted changes")
    if git("tag", "--list", f"v{version}"):
        sys.exit(f"tag v{version} already exists")
    changelog = ROOT / CHANGELOG
    unreleased = re.search(r"^## Unreleased\s*\n(.*?)(?=^## |\Z)", changelog.read_text(encoding="utf-8"), re.S | re.M)
    if not unreleased or not unreleased.group(1).strip():
        sys.exit(f'{CHANGELOG} has no "## Unreleased" entries to release')

    # Check every file before writing any, so a mismatch leaves nothing half-edited.
    edits = {
        ROOT / name: replaced_once(ROOT / name, pattern, line.format(v=version))
        for name, pattern, line in VERSION_LINES
    }
    edits[changelog] = replaced_once(changelog, r"^## Unreleased[ \t]*$", f"## {version}")
    for path, text in edits.items():
        path.write_text(text, encoding="utf-8", newline="")  # keep LF line endings on Windows too

    git("commit", "--quiet", "-am", f"release: {version}")
    git("tag", "-a", f"v{version}", "-m", f"v{version}: {args.summary}")
    print(f"Committed and tagged v{version}. Publish with:\n  git push --atomic origin main v{version}")


if __name__ == "__main__":
    main()
