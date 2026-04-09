from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot before/after remaining-budget weight distributions from compare_rembudget_offline JSON."
    )
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-png", type=str, default="")
    parser.add_argument("--output-svg", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args()


def _quantile_keys(quantiles: dict[str, float]) -> list[str]:
    keys = [k for k in quantiles.keys() if k.startswith("p")]
    return sorted(keys, key=lambda k: int(k[1:]))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    reserved = {"data_file", "gamma", "temperature", "num_valid_tokens", "delta"}
    labels = [key for key in payload.keys() if key not in reserved]
    if len(labels) != 2:
        raise ValueError(f"Expected exactly two model labels in compare JSON, got {labels!r}")
    before_label, after_label = labels

    before = payload[before_label]
    after = payload[after_label]
    delta = payload["delta"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    ax = axes[0]
    before_hist = before["histogram"]
    after_hist = after["histogram"]
    ax.plot(before_hist["bin_centers"], before_hist["density"], label=before_label, linewidth=2)
    ax.plot(after_hist["bin_centers"], after_hist["density"], label=after_label, linewidth=2)
    ax.set_title("Raw RemBudget Weight Density")
    ax.set_xlabel("remaining_budget_weight")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    delta_hist = delta["histogram_after_minus_before"]
    ax.plot(delta_hist["bin_centers"], delta_hist["density"], color="tab:purple", linewidth=2)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Delta Density")
    ax.set_xlabel(f"{after_label} - {before_label}")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)

    ax = axes[2]
    before_q = before["quantiles"]
    after_q = after["quantiles"]
    qkeys = _quantile_keys(before_q)
    xs = [int(k[1:]) for k in qkeys]
    ax.plot(xs, [before_q[k] for k in qkeys], marker="o", label=before_label, linewidth=2)
    ax.plot(xs, [after_q[k] for k in qkeys], marker="o", label=after_label, linewidth=2)
    ax.set_title("Quantile Shift")
    ax.set_xlabel("percentile")
    ax.set_ylabel("remaining_budget_weight")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    summary = (
        f"tokens={payload['num_valid_tokens']:,}\n"
        f"{before_label} mean={before['mean']:.4f}\n"
        f"{after_label} mean={after['mean']:.4f}\n"
        f"delta mean={delta['mean_after_minus_before']:.4f}\n"
        f"after>before={delta['fraction_after_gt_before']:.3f}\n"
        f"after<before={delta['fraction_after_lt_before']:.3f}"
    )
    fig.suptitle(args.title or "Remaining-Budget Weight Comparison", fontsize=14)
    fig.text(0.995, 0.02, summary, ha="right", va="bottom", fontsize=9, family="monospace")

    if args.output_png:
        out_png = Path(args.output_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
    if args.output_svg:
        out_svg = Path(args.output_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, bbox_inches="tight")
    if not args.output_png and not args.output_svg:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
