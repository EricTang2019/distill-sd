#!/usr/bin/env python3
"""Patch sglang StandaloneWorker to sync only target weights.

This patch is needed when speculative decoding uses a smaller draft model
than target model. In that setup, trainer-side weight sync tensors match
the target model shape and must not be applied to the draft worker.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


MARKER = "keep the draft model fixed"


def _ensure_import(text: str, anchor: str, line_to_add: str) -> str:
    if line_to_add in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Cannot find import anchor: {anchor}")
    return text.replace(anchor, f"{anchor}\n{line_to_add}", 1)


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False

    text = _ensure_import(
        text,
        "from sglang.srt.managers.tp_worker import TpModelWorker",
        "from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput",
    )
    text = _ensure_import(
        text,
        "from sglang.srt.utils import MultiprocessingSerializer, empty_context, get_bool_env_var, is_cuda",
        "from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions",
    )

    anchor = "        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)\n"
    if anchor not in text:
        raise RuntimeError("Cannot find StandaloneWorker __init__ tail anchor.")

    method = """
    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        # In standalone speculative decoding, keep the draft model fixed
        # and only synchronize target weights from the trainer.
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        return self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
"""

    text = text.replace(anchor, anchor + method, 1)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("[ERROR] Cannot locate installed sglang package.", file=sys.stderr)
        return 2

    target_file = Path(spec.origin).resolve().parent / "srt/speculative/standalone_worker.py"
    if not target_file.exists():
        print(f"[ERROR] Cannot find file: {target_file}", file=sys.stderr)
        return 2

    changed = patch_file(target_file)
    if changed:
        print(f"[OK] Patched: {target_file}")
    else:
        print(f"[OK] Already patched: {target_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
