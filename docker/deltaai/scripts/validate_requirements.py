#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as im
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that exact pinned requirements are present in the active environment."
    )
    parser.add_argument("requirements", type=Path, help="Path to a requirements.txt file containing exact == pins.")
    args = parser.parse_args()

    failures: list[str] = []
    for raw in args.requirements.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        name = canonicalize_name(req.name)
        if not req.specifier:
            failures.append(f"{line}: missing exact version pin")
            continue

        exact_pins = [spec.version for spec in req.specifier if spec.operator == "=="]
        if len(exact_pins) != 1:
            failures.append(f"{line}: expected exactly one == pin")
            continue

        expected = exact_pins[0]
        try:
            actual = im.version(name)
        except im.PackageNotFoundError:
            failures.append(f"{line}: package not installed")
            continue

        # Treat local-version suffixes as compatible if the public version matches.
        actual_base = actual.split("+", 1)[0]
        expected_base = expected.split("+", 1)[0]
        if actual_base != expected_base:
            failures.append(f"{line}: found {actual}")

    if failures:
        raise SystemExit("Requirement validation failed:\n- " + "\n- ".join(failures))

    print(f"Requirement validation passed for {args.requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
