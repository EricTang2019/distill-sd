#!/usr/bin/env python3

import argparse
import importlib
import importlib.metadata
import importlib.util
import platform
from pathlib import Path
from typing import List

import yaml

DIST_NAME_OVERRIDES = {
    "flash_attn": "flash-attn",
    "flashinfer": "flashinfer-python",
    "liger_kernel": "liger-kernel",
    "sgl_kernel": "sgl-kernel",
}


def _normalize_machine(machine: str) -> str:
    machine = machine.lower()
    if machine == "arm64":
        return "aarch64"
    return machine


def _installed_version(module_name: str) -> str:
    dist_name = DIST_NAME_OVERRIDES.get(module_name, module_name)
    try:
        return str(importlib.metadata.version(dist_name))
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is None:
            raise RuntimeError("Module {!r} has no __version__ attribute.".format(module_name))
        return str(version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the active Python environment against an exact saved snapshot.")
    parser.add_argument("spec", type=Path, help="Path to envs/*.versions.yaml")
    parser.add_argument("--ignore-python-patch", action="store_true", help="Only compare Python major.minor.")
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    expected_platform = spec.get("platform")
    if expected_platform is not None:
        actual_platform = "{}-{}".format(platform.system().lower(), _normalize_machine(platform.machine()))
        if actual_platform != str(expected_platform):
            raise SystemExit("Platform mismatch: expected {}, got {}".format(expected_platform, actual_platform))

    expected_python = str(spec["python"])
    actual_python = platform.python_version()
    if args.ignore_python_patch:
        expected_mm = ".".join(expected_python.split(".")[:2])
        actual_mm = ".".join(actual_python.split(".")[:2])
        if actual_mm != expected_mm:
            raise SystemExit("Python mismatch: expected major.minor {}, got {}".format(expected_mm, actual_python))
    elif actual_python != expected_python:
        raise SystemExit("Python mismatch: expected {}, got {}".format(expected_python, actual_python))

    failures = []  # type: List[str]
    for module_name, expected in spec.get("packages", {}).items():
        try:
            actual = _installed_version(module_name)
        except Exception as exc:
            failures.append("{}: import/version lookup failed: {!r}".format(module_name, exc))
            continue
        if actual != str(expected):
            failures.append("{}: expected {}, got {}".format(module_name, expected, actual))

    for module_name in spec.get("absent_packages", []):
        if importlib.util.find_spec(module_name) is None:
            continue
        failures.append("{}: expected to be absent, but import succeeded".format(module_name))

    if failures:
        raise SystemExit("Environment validation failed:\n- " + "\n- ".join(failures))

    print("Environment validation passed for {}".format(args.spec))


if __name__ == "__main__":
    main()
