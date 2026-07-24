"""
Chart the effect of MoE layer CPU-offloading (-ncmoe) on:
  1. max usable context length
  2. prompt vs. generation throughput (tokens/s)

Designed for repeated sweep runs over the same ncmoe grid (e.g. several
CSVs from re-running the same sweep), plotting a min-max band plus an
average line per metric, in the xkcd/all-caps house style.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import pandas as pd


def _set_xkcd_style():
    plt.xkcd()
    # plt.xkcd() sets two independent things: path.sketch (the hand-drawn
    # wiggle) and path.effects (a withStroke that draws a fat white outline
    # behind every line and marker). The white outline erases narrow
    # fill_between bands that hug a line, so drop it and keep the wiggle.
    plt.rcParams["path.effects"] = []
    font_family = None
    for f in font_manager.fontManager.ttflist:
        if "xkcd" in f.name.lower():
            font_family = f.name
            break
    if font_family is not None:
        plt.rcParams["font.family"] = font_family


def load_sweep_runs(csv_paths):
    """Load a list of sweep-result CSVs (same ncmoe grid, repeated runs)
    and return (ncmoe_index, {metric_name: DataFrame[run_idx -> values]})."""
    runs = [pd.read_csv(p).set_index("ncmoe") for p in csv_paths]

    ncmoe = runs[0].index
    for df in runs[1:]:
        assert (df.index == ncmoe).all(), "sweep runs have mismatched ncmoe grids"

    combined = {}
    for col in ("n_ctx", "prompt_tps", "gen_tps", "vram_mib", "rss_mib"):
        combined[col] = pd.concat(
            [df[col].rename(i) for i, df in enumerate(runs)], axis=1
        )
    return ncmoe, combined


def _band_and_avg(ax, x, frame, color, label):
    """Shaded min-max band plus an average line across repeated runs.

    The band is narrow relative to the y-range, so it needs a reasonably
    strong alpha, small markers on the average line, and thin edge lines
    marking min/max so the extent stays readable where the band pinches.
    (This only works with the xkcd white-stroke path effect disabled --
    see _set_xkcd_style.)
    """
    lo = frame.min(axis=1)
    hi = frame.max(axis=1)
    avg = frame.mean(axis=1)

    ax.fill_between(x, lo, hi, color=color, alpha=0.35, linewidth=0, zorder=1)
    ax.plot(x, lo, color=color, linewidth=0.8, alpha=0.7, zorder=2)
    ax.plot(
        x, hi, color=color, linewidth=0.8, alpha=0.7, zorder=2,
        label=f"MIN\u2013MAX {label}",
    )
    ax.plot(
        x, avg, color=color, marker="o", markersize=4, linestyle="-",
        linewidth=2, zorder=3, label=f"AVG {label}",
    )


def generate_offload_charts(csv_paths, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_xkcd_style()

    ncmoe, metrics = load_sweep_runs(csv_paths)

    # --- Chart 1: max context length vs. MoE layers offloaded ---

    fig, ax_ctx = plt.subplots(figsize=(8, 6), dpi=100)

    _band_and_avg(ax_ctx, ncmoe, metrics["n_ctx"], "teal", "CONTEXT")

    ax_ctx.set_title("MAX CONTEXT LENGTH VS MOE LAYERS OFFLOADED")
    ax_ctx.set_xlabel("MOE LAYERS OFFLOADED TO CPU (-NCMOE)")
    ax_ctx.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_ctx.set_ylabel("MAX CONTEXT LENGTH (TOKENS)")
    # The curve rises steeply then plateaus at the top right, so "upper
    # right" sits right on top of the data. Lower right is empty (ncmoe
    # is high there, but context length only reads high, not low).
    ax_ctx.legend(loc="lower right", handlelength=2.0, handletextpad=0.6)

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_dir / "context-length-chart.png", bbox_inches="tight")
    plt.close(fig)

    # --- Chart 2: throughput vs. MoE layers offloaded (prompt + gen, separate axes) ---

    fig_tps, ax_prompt = plt.subplots(figsize=(8, 6), dpi=100)
    ax_gen = ax_prompt.twinx()

    _band_and_avg(ax_prompt, ncmoe, metrics["prompt_tps"], "blue", "PROMPT TPS")
    _band_and_avg(ax_gen, ncmoe, metrics["gen_tps"], "orange", "GEN TPS")

    ax_prompt.set_title("THROUGHPUT VS MOE LAYERS OFFLOADED")
    ax_prompt.set_xlabel("MOE LAYERS OFFLOADED TO CPU (-NCMOE)")
    ax_prompt.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_prompt.set_ylabel("PROMPT THROUGHPUT (TOKENS/S)", color="blue")
    ax_gen.set_ylabel("GENERATION THROUGHPUT (TOKENS/S)", color="orange")

    ax_prompt.tick_params(axis="y", labelcolor="blue")
    ax_gen.tick_params(axis="y", labelcolor="orange")

    # Merge legends from both axes into one box.
    lines_1, labels_1 = ax_prompt.get_legend_handles_labels()
    lines_2, labels_2 = ax_gen.get_legend_handles_labels()
    ax_prompt.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.6,
    )

    fig_tps.tight_layout(rect=(0, 0.12, 1, 1))
    fig_tps.savefig(output_dir / "throughput-chart.png", bbox_inches="tight")
    plt.close(fig_tps)


    # --- Chart 3: memory usage vs. MoE layers offloaded (VRAM + RAM, shared axis) ---

    fig_mem, ax_mem = plt.subplots(figsize=(8, 6), dpi=100)

    _band_and_avg(ax_mem, ncmoe, metrics["vram_mib"], "green", "VRAM")
    _band_and_avg(ax_mem, ncmoe, metrics["rss_mib"], "crimson", "RAM")

    ax_mem.set_title("MEMORY USAGE VS MOE LAYERS OFFLOADED")
    ax_mem.set_xlabel("MOE LAYERS OFFLOADED TO CPU (-NCMOE)")
    ax_mem.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_mem.set_ylabel("MEMORY (MIB)")
    # VRAM starts top-left and falls; RAM starts bottom-left and rises;
    # they cross on the right-hand side, leaving the middle-left empty.
    ax_mem.legend(loc="center left", handlelength=2.0, handletextpad=0.6)

    fig_mem.tight_layout(rect=(0, 0.12, 1, 1))
    fig_mem.savefig(output_dir / "memory-chart.png", bbox_inches="tight")
    plt.close(fig_mem)


if __name__ == "__main__":
    csv_paths = sys.argv[1:]
    print(f"Charting {csv_paths}")
    generate_offload_charts(csv_paths, "charts")
