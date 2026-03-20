#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as im
import re
import sys
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def main() -> int:
    env = default_environment()
    installed = {}
    for dist in im.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        installed[canonicalize_name(name)] = dist

    missing = []
    conflicts = []
    invalid_requires = []

    for dist in installed.values():
        dist_name = dist.metadata["Name"]
        for raw_req in dist.requires or ():
            try:
                req = Requirement(raw_req)
            except Exception as exc:  # pragma: no cover - defensive
                invalid_requires.append((dist_name, raw_req, str(exc)))
                continue

            if req.marker and not req.marker.evaluate(env):
                continue

            req_name = canonicalize_name(req.name)
            dep = installed.get(req_name)
            if dep is None:
                missing.append((dist_name, raw_req))
                continue

            if not req.specifier:
                continue

            dep_version = dep.version
            try:
                parsed_version = Version(dep_version)
            except InvalidVersion:
                # Keep behavior fail-safe but pragmatic for local/path installs.
                if not re.search(r"(git\+|file:|editable|local)", dep_version):
                    conflicts.append((dist_name, raw_req, dep.metadata["Name"], dep_version))
                continue

            if parsed_version not in req.specifier:
                conflicts.append((dist_name, raw_req, dep.metadata["Name"], dep_version))

    if invalid_requires or missing or conflicts:
        if invalid_requires:
            print("Invalid requirement strings:", file=sys.stderr)
            for dist_name, raw_req, exc in invalid_requires:
                print(f"  {dist_name}: {raw_req!r} ({exc})", file=sys.stderr)
        if missing:
            print("Missing dependencies:", file=sys.stderr)
            for dist_name, raw_req in missing:
                print(f"  {dist_name}: missing {raw_req}", file=sys.stderr)
        if conflicts:
            print("Version conflicts:", file=sys.stderr)
            for dist_name, raw_req, dep_name, dep_version in conflicts:
                print(
                    f"  {dist_name}: requires {raw_req}, found {dep_name}=={dep_version}",
                    file=sys.stderr,
                )
        return 1

    print("Dependency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
