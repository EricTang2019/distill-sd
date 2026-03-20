#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path

import yaml


DIST_NAME_OVERRIDES = {
    "flash_attn": "flash-attn",
    "flashinfer": "flashinfer-python",
    "sgl_kernel": "sgl-kernel",
}


def _import_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is None:
        raise RuntimeError(f"Module {module_name!r} has no __version__ attribute.")
    return str(version)


def _installed_version(module_name: str) -> str:
    dist_name = DIST_NAME_OVERRIDES.get(module_name, module_name)
    try:
        return str(importlib.metadata.version(dist_name))
    except importlib.metadata.PackageNotFoundError:
        return _import_version(module_name)


def _major_minor_python() -> str:
    return ".".join(platform.python_version().split(".")[:2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the active Python environment against a saved version snapshot.")
    parser.add_argument("spec", type=Path, help="Path to env.*.versions.yaml")
    parser.add_argument("--ignore-python-patch", action="store_true", help="Only compare major.minor for Python.")
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    expected_python = str(spec["python"])
    actual_python = platform.python_version()
    if args.ignore_python_patch:
        if _major_minor_python() != ".".join(expected_python.split(".")[:2]):
            raise SystemExit(
                f"Python mismatch: expected major.minor {'.'.join(expected_python.split('.')[:2])}, got {actual_python}"
            )
    elif actual_python != expected_python:
        raise SystemExit(f"Python mismatch: expected {expected_python}, got {actual_python}")

    failures: list[str] = []
    for module_name, expected in spec.get("packages", {}).items():
        try:
            actual = _installed_version(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: import failed: {exc!r}")
            continue
        if actual != str(expected):
            failures.append(f"{module_name}: expected {expected}, got {actual}")

    for module_name in spec.get("absent_packages", []):
        if importlib.util.find_spec(module_name) is None:
            continue
        failures.append(f"{module_name}: expected to be absent, but import succeeded")

    if failures:
        raise SystemExit("Environment validation failed:\n- " + "\n- ".join(failures))

    print(f"Environment validation passed for {args.spec}")


if __name__ == "__main__":
    main()
