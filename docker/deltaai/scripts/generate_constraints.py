#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


FLASH_ATTN_WHL_RE = re.compile(r"flash_attn-(?P<version>[0-9][^+]+)\+")

DIRECT_REF_OVERRIDES = {
    "megatron-core": "megatron-core==0.13.1",
    "transformer_engine": "transformer-engine==2.6.0+c90a720",
}

SKIP_PREFIXES = (
    "-e ",
)

SKIP_DIRECT_REF_PREFIXES = (
    "packaging @ file://",
    "pip @ file://",
)


def sanitize_line(raw: str) -> str | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith(SKIP_PREFIXES):
        return None
    if line.startswith(SKIP_DIRECT_REF_PREFIXES):
        return None
    if line.startswith("flash_attn @ "):
        match = FLASH_ATTN_WHL_RE.search(line)
        if not match:
            raise ValueError(f"Could not parse flash-attn version from line: {line}")
        return f"flash-attn=={match.group('version')}"
    if " @ " in line:
        package_name = line.split(" @ ", 1)[0]
        override = DIRECT_REF_OVERRIDES.get(package_name)
        if override is not None:
            return override
        return None
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sanitized pip constraints file from pip freeze output.")
    parser.add_argument("src", type=Path, help="Path to a raw pip-freeze file.")
    parser.add_argument("dst", type=Path, help="Path to the generated constraints file.")
    args = parser.parse_args()

    seen: set[str] = set()
    lines: list[str] = [
        "# Auto-generated from local pip freeze.",
        "# This file is intentionally sanitized for arm64/GH200 builds:",
        "# - editable installs are removed",
        "# - local file:// references are removed",
        "# - known x86_64 direct URLs are normalized to version pins",
        "",
    ]
    for raw in args.src.read_text().splitlines():
        normalized = sanitize_line(raw)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        lines.append(normalized)

    args.dst.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
