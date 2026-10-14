#!/usr/bin/env python3
r"""Decompose the per-window CATCH-minus-DAMPA-consensus difference by pi bin.

For every species/window, the plotted endpoint is::

    CATCH - DAMPA consensus

Its two additive components are::

    CATCH - DAMPA representative
    DAMPA representative - DAMPA consensus

Each component is divided by the matched DAMPA-consensus probe count, and the
two resulting contributions are shown as boxplots within each pi bin.

Input contract
--------------
Python 3.10+, numpy, pandas, matplotlib and seaborn. No local helper modules
are required. --input-data takes one headered CSV per --species-labels value.
These are the per-window exports from figure4e_pt1_pi_vs_unique_prove_scatter.py,
computed for the three probe sets on the SAME alignment and retained windows.
Required columns are:

    window                     matching window identifier across probe sets
    probe_set                  label identifying one of the three designs
    mean_unique_probe_count    mean distinct-probe depth per retained position
    distinct_probe_count       number of distinct probe IDs overlapping window
    mean_pi                    per-site nucleotide diversity (fraction)

Use median_pi instead of mean_pi with --pi-summary median. Probe-set values
must exactly match --catch-label, --dampa-representative-label and
--dampa-consensus-label (all configurable). Labels must be distinct. There
must be at most one row per window/probe_set; pi must agree across probe sets
within a window. Extra columns/probe sets are ignored. Species labels are
supplied explicitly rather than inferred from filenames.

Contributions plotted as boxplots/strip points use differences in
mean_unique_probe_count divided by positive matched DAMPA-consensus depth.
The absolute-count tables instead use differences in distinct_probe_count;
these represent distinct probes per window, not mean per-position depths.
Incomplete windows and undefined non-positive-denominator ratios are removed.
Negative component differences are retained. Decomposition residuals are
checked against zero. Windows are pooled across species within each pi bin;
this is a window-level distribution, not an equal-species-weighted average.
--pi-bin-width, --zero-pi-bin and --maximum-pi control binning.

Outputs next to --output-file: *_data.csv, *_binned_summary.csv,
*_absolute_probe_counts_by_window.csv and
*_absolute_probe_counts_binned_summary.csv. Distinct probes can overlap
multiple windows: binned sums are sums of window counts, not a deduplicated
count for the entire probe library. Existing output files are overwritten.
All output defaults are relative to the working directory.

Example:
    python figure_scripts/supp_fig5d_plot_catch_dampa_difference_decomposition.py \
        --input-data data/species_a_data.csv data/species_b_data.csv \
        --species-labels SpeciesA SpeciesB \
        --output-file output/supp_fig5d.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


RAW_VALUE = "mean_unique_probe_count"
DISTINCT_PROBE_COUNT = "distinct_probe_count"
PI_VALUE = "pi_value"
TOTAL_DIFFERENCE = "catch_minus_dampa_consensus"
CATCH_COMPONENT = "catch_minus_dampa_representative"
DAMPA_COMPONENT = "dampa_representative_minus_dampa_consensus"
PANGENOME_IMPACT_COUNT = "pangenome_impact_probe_count"
CONSENSUS_IMPACT_COUNT = "consensus_impact_probe_count"
TOTAL_IMPACT_COUNT = "total_impact_probe_count"
TOTAL_CONTRIBUTION = "catch_minus_dampa_consensus_relative_to_dampa_consensus"
CATCH_CONTRIBUTION = (
    "catch_minus_dampa_representative_relative_to_dampa_consensus"
)
DAMPA_CONTRIBUTION = (
    "dampa_representative_minus_dampa_consensus_relative_to_dampa_consensus"
)
DEFAULT_WIDTH = 8.27 * 0.62
DEFAULT_HEIGHT = DEFAULT_WIDTH * 0.70
COMPONENT_COLORS = {
    CATCH_CONTRIBUTION: "#398D3E",
    DAMPA_CONTRIBUTION: "#9467BD",
}
CONTRIBUTION_LABELS = {
    CATCH_CONTRIBUTION: "Pangenome impact",
    DAMPA_CONTRIBUTION: "Consensus impact",
}


mpl.rcParams["pdf.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-data",
        nargs="+",
        required=True,
        help="Per-species pi/probe CSV files containing all three probe sets.",
    )
    parser.add_argument(
        "--species-labels",
        nargs="+",
        required=True,
        help="Species labels corresponding to --input-data, in the same order.",
    )
    parser.add_argument(
        "--output-file",
        "--output_file",
        dest="output_file",
        default="output/supp_fig5d/catch_dampa_difference_decomposition.pdf",
    )
    parser.add_argument("--catch-label", default="catch")
    parser.add_argument(
        "--dampa-representative-label", default="dampa_representative"
    )
    parser.add_argument("--dampa-consensus-label", default="dampa_consensus")
    parser.add_argument(
        "--pi-summary", choices=("mean", "median"), default="mean"
    )
    parser.add_argument("--pi-bin-width", type=float, default=0.05)
    parser.add_argument(
        "--maximum-pi",
        "--max-pi",
        dest="maximum_pi",
        type=float,
        help="Merge pi values above this threshold into a terminal >X bin.",
    )
    parser.add_argument(
        "--zero-pi-bin",
        "--separate-zero-pi-bin",
        dest="zero_pi_bin",
        action="store_true",
        help="Place exact-zero pi observations in a separate bin.",
    )
    parser.add_argument(
        "--point-size", type=float, default=10.0, help="Strip-point area (default: 10)."
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.7,
        help="Strip-point opacity (default: 0.7).",
    )
    parser.add_argument(
        "--box-alpha",
        type=float,
        default=0.7,
        help="Boxplot fill opacity (default: 0.7).",
    )
    parser.add_argument("--plot-width", type=float, default=DEFAULT_WIDTH)
    parser.add_argument("--plot-height", type=float, default=DEFAULT_HEIGHT)
    parser.add_argument("--font-size", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if len(args.input_data) != len(args.species_labels):
        parser.error("--input-data and --species-labels must have the same length")
    if len(set(args.species_labels)) != len(args.species_labels):
        parser.error("--species-labels values must be unique")
    if not np.isfinite([args.pi_bin_width, args.point_size, args.plot_width,
                       args.plot_height, args.font_size]).all():
        parser.error("Bin width, point size and plot dimensions must be finite")
    if len({args.catch_label, args.dampa_representative_label, args.dampa_consensus_label}) != 3:
        parser.error("The three probe-set labels must be distinct")
    if args.pi_bin_width <= 0:
        parser.error("--pi-bin-width must be greater than zero")
    if args.maximum_pi is not None and (not np.isfinite(args.maximum_pi) or args.maximum_pi <= 0):
        parser.error("--maximum-pi must be greater than zero")
    if args.point_size <= 0 or not 0 < args.point_alpha <= 1:
        parser.error("point size must be positive and point alpha must be in (0, 1]")
    if not 0 < args.box_alpha <= 1:
        parser.error("--box-alpha must be greater than 0 and at most 1")
    if args.plot_width <= 0 or args.plot_height <= 0 or args.font_size <= 0:
        parser.error("plot dimensions and font size must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be greater than zero")
    return args


def load_decomposition_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load, pair, and decompose the three probe-set values per window."""
    pi_column = f"{args.pi_summary}_pi"
    labels = {
        "catch": args.catch_label,
        "dampa_representative": args.dampa_representative_label,
        "dampa_consensus": args.dampa_consensus_label,
    }
    frames: list[pd.DataFrame] = []

    for input_file, species in zip(args.input_data, args.species_labels):
        data = pd.read_csv(input_file)
        required = {
            "window",
            "probe_set",
            RAW_VALUE,
            DISTINCT_PROBE_COUNT,
            pi_column,
        }
        missing_columns = required - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"{input_file} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        selected = data.loc[data["probe_set"].isin(labels.values())].copy()
        available = set(selected["probe_set"].dropna())
        missing_labels = [label for label in labels.values() if label not in available]
        if missing_labels:
            raise ValueError(
                f"{input_file} is missing probe set(s): {', '.join(missing_labels)}"
            )
        duplicates = selected.duplicated(["window", "probe_set"], keep=False)
        if duplicates.any():
            duplicate_pairs = selected.loc[duplicates, ["window", "probe_set"]]
            raise ValueError(
                f"{input_file} has duplicate window/probe_set rows: "
                f"{duplicate_pairs.head().to_dict(orient='records')}"
            )

        selected[RAW_VALUE] = pd.to_numeric(selected[RAW_VALUE], errors="coerce")
        selected[DISTINCT_PROBE_COUNT] = pd.to_numeric(
            selected[DISTINCT_PROBE_COUNT], errors="coerce"
        )
        selected[PI_VALUE] = pd.to_numeric(selected[pi_column], errors="coerce")
        pi_counts = selected.groupby("window")[PI_VALUE].nunique(dropna=True)
        inconsistent = pi_counts[pi_counts > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"{input_file} has inconsistent {pi_column} values among probe "
                f"sets for window(s): {', '.join(map(str, inconsistent.index[:5]))}"
            )

        wide = selected.pivot(index="window", columns="probe_set", values=RAW_VALUE)
        wide = wide.rename(
            columns={
                labels["catch"]: "catch_probe_count",
                labels["dampa_representative"]: "dampa_representative_probe_count",
                labels["dampa_consensus"]: "dampa_consensus_probe_count",
            }
        ).reset_index()
        distinct_wide = selected.pivot(
            index="window", columns="probe_set", values=DISTINCT_PROBE_COUNT
        )
        distinct_wide = distinct_wide.rename(
            columns={
                labels["catch"]: "catch_distinct_probe_count",
                labels["dampa_representative"]: (
                    "dampa_representative_distinct_probe_count"
                ),
                labels["dampa_consensus"]: "dampa_consensus_distinct_probe_count",
            }
        ).reset_index()
        wide = wide.merge(distinct_wide, on="window", validate="one_to_one")
        pi_by_window = selected.groupby("window", sort=False)[PI_VALUE].first()
        wide[PI_VALUE] = wide["window"].map(pi_by_window)
        wide["species"] = species
        wide["source_file"] = str(Path(input_file).expanduser())
        wide[CATCH_COMPONENT] = (
            wide["catch_probe_count"] - wide["dampa_representative_probe_count"]
        )
        wide[DAMPA_COMPONENT] = (
            wide["dampa_representative_probe_count"]
            - wide["dampa_consensus_probe_count"]
        )
        wide[TOTAL_DIFFERENCE] = (
            wide["catch_probe_count"] - wide["dampa_consensus_probe_count"]
        )
        wide["decomposition_residual"] = (
            wide[TOTAL_DIFFERENCE]
            - wide[CATCH_COMPONENT]
            - wide[DAMPA_COMPONENT]
        )
        denominator = wide["dampa_consensus_probe_count"]
        valid_denominator = np.isfinite(denominator) & denominator.gt(0)
        wide[CATCH_CONTRIBUTION] = (
            wide[CATCH_COMPONENT] / denominator.where(valid_denominator)
        )
        wide[DAMPA_CONTRIBUTION] = (
            wide[DAMPA_COMPONENT] / denominator.where(valid_denominator)
        )
        wide[TOTAL_CONTRIBUTION] = (
            wide[TOTAL_DIFFERENCE] / denominator.where(valid_denominator)
        )
        wide["contribution_residual"] = (
            wide[TOTAL_CONTRIBUTION]
            - wide[CATCH_CONTRIBUTION]
            - wide[DAMPA_CONTRIBUTION]
        )
        wide[PANGENOME_IMPACT_COUNT] = (
            wide["catch_distinct_probe_count"]
            - wide["dampa_representative_distinct_probe_count"]
        )
        wide[CONSENSUS_IMPACT_COUNT] = (
            wide["dampa_representative_distinct_probe_count"]
            - wide["dampa_consensus_distinct_probe_count"]
        )
        wide[TOTAL_IMPACT_COUNT] = (
            wide["catch_distinct_probe_count"]
            - wide["dampa_consensus_distinct_probe_count"]
        )
        wide["absolute_count_decomposition_residual"] = (
            wide[TOTAL_IMPACT_COUNT]
            - wide[PANGENOME_IMPACT_COUNT]
            - wide[CONSENSUS_IMPACT_COUNT]
        )
        frames.append(wide)

    combined = pd.concat(frames, ignore_index=True)
    value_columns = [
        PI_VALUE,
        "catch_probe_count",
        "dampa_representative_probe_count",
        "dampa_consensus_probe_count",
        "catch_distinct_probe_count",
        "dampa_representative_distinct_probe_count",
        "dampa_consensus_distinct_probe_count",
        CATCH_CONTRIBUTION,
        DAMPA_CONTRIBUTION,
    ]
    before = len(combined)
    combined = combined.dropna(subset=value_columns).reset_index(drop=True)
    if len(combined) < before:
        print(
            f"Warning: removed {before - len(combined):,} incomplete window(s)",
            file=sys.stderr,
        )
    if combined.empty:
        raise ValueError("No complete window observations remained")
    if not np.allclose(combined["decomposition_residual"], 0.0, atol=1e-10):
        raise ValueError("The difference decomposition identity failed")
    if not np.allclose(combined["contribution_residual"], 0.0, atol=1e-10):
        raise ValueError("The relative-contribution identity failed")
    if not np.allclose(
        combined["absolute_count_decomposition_residual"], 0.0, atol=1e-10
    ):
        raise ValueError("The absolute-count decomposition identity failed")
    if (combined[PI_VALUE] < 0).any():
        raise ValueError("Pi values must be non-negative")
    return combined


def add_pi_bins(
    data: pd.DataFrame,
    bin_width: float,
    maximum_pi: float | None,
    separate_zero: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Assign ordered pi bins, including optional zero and overflow bins."""
    data = data.copy()
    values = data[PI_VALUE].to_numpy(dtype=float)
    decimals = max(2, int(np.ceil(-np.log10(bin_width))) + 1) if bin_width < 1 else 2
    overflow = values > maximum_pi if maximum_pi is not None else np.zeros(len(data), bool)
    regular_values = values.copy()
    if maximum_pi is not None:
        regular_values = np.minimum(regular_values, np.nextafter(maximum_pi, -np.inf))
    indices = np.floor(regular_values / bin_width).astype(int)
    zero = (values == 0) & separate_zero
    regular_indices = sorted(np.unique(indices[~overflow & ~zero]))
    labels_by_index = {
        index: (
            f"{index * bin_width:.{decimals}f}–"
            f"{min((index + 1) * bin_width, maximum_pi) if maximum_pi is not None else (index + 1) * bin_width:.{decimals}f}"
        )
        for index in regular_indices
    }
    row_labels = [labels_by_index.get(index, "") for index in indices]
    order: list[str] = []
    if zero.any():
        for row_index in np.flatnonzero(zero):
            row_labels[row_index] = "0"
        order.append("0")
    order.extend(labels_by_index[index] for index in regular_indices)
    if maximum_pi is not None and overflow.any():
        overflow_label = f">{maximum_pi:.2f}"
        for row_index in np.flatnonzero(overflow):
            row_labels[row_index] = overflow_label
        order.append(overflow_label)
    data["pi_bin"] = pd.Categorical(row_labels, categories=order, ordered=True)
    data["pi_bin_overflow"] = overflow
    return data, order


def draw_decomposition_plot(
    data: pd.DataFrame,
    bin_order: list[str],
    args: argparse.Namespace,
) -> plt.Figure:
    """Draw the two DAMPA-consensus-relative contributions as boxplots."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(args.plot_width, args.plot_height))
    contribution_order = [CATCH_CONTRIBUTION, DAMPA_CONTRIBUTION]
    long = data.melt(
        id_vars=["species", "window", "pi_bin"],
        value_vars=contribution_order,
        var_name="contribution",
        value_name="relative_contribution",
    )
    long["Contribution"] = long["contribution"].map(CONTRIBUTION_LABELS)
    hue_order = [CONTRIBUTION_LABELS[column] for column in contribution_order]
    palette = {
        CONTRIBUTION_LABELS[column]: COMPONENT_COLORS[column]
        for column in contribution_order
    }
    sns.boxplot(
        data=long,
        x="pi_bin",
        y="relative_contribution",
        hue="Contribution",
        order=bin_order,
        hue_order=hue_order,
        palette=palette,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    for box in ax.patches:
        facecolor = box.get_facecolor()
        box.set_facecolor((*facecolor[:3], args.box_alpha))
    sns.stripplot(
        data=long,
        x="pi_bin",
        y="relative_contribution",
        hue="Contribution",
        order=bin_order,
        hue_order=hue_order,
        palette=palette,
        dodge=True,
        jitter=0.18,
        size=max(2.0, np.sqrt(args.point_size)),
        alpha=args.point_alpha,
        edgecolor="white",
        linewidth=0,
        marker=".",
        ax=ax,
    )

    ax.axhline(0, color="#404040", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel(
        f"{args.pi_summary.capitalize()} nucleotide diversity (π) bin",
        fontsize=args.font_size,
    )
    ax.set_ylabel(
        "Contribution relative to DAMPA-consensus probe count",
        fontsize=args.font_size,
    )
    ax.set_title(
        "Contributions to CATCH − DAMPA consensus",
        fontsize=args.font_size,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.tick_params(axis="both", labelsize=args.font_size)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        [unique[label] for label in hue_order],
        hue_order,
        frameon=False,
        fontsize=args.font_size,
        title=None,
        loc="best",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def summarize_bins(data: pd.DataFrame) -> pd.DataFrame:
    """Return distribution summaries for the total and both components."""
    long = data.melt(
        id_vars=["species", "window", "pi_bin"],
        value_vars=[TOTAL_CONTRIBUTION, CATCH_CONTRIBUTION, DAMPA_CONTRIBUTION],
        var_name="difference",
        value_name="value",
    )
    return (
        long.groupby(["pi_bin", "difference"], observed=True, sort=True)["value"]
        .agg(
            observations="size",
            mean="mean",
            q1=lambda values: values.quantile(0.25),
            median="median",
            q3=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )


def absolute_probe_counts_by_window(data: pd.DataFrame) -> pd.DataFrame:
    """Return tool counts and the two absolute count impacts for each window."""
    columns = [
        "species",
        "window",
        PI_VALUE,
        "pi_bin",
        "catch_distinct_probe_count",
        "dampa_representative_distinct_probe_count",
        "dampa_consensus_distinct_probe_count",
        PANGENOME_IMPACT_COUNT,
        CONSENSUS_IMPACT_COUNT,
        TOTAL_IMPACT_COUNT,
    ]
    return data.loc[:, columns].copy()


def summarize_absolute_probe_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Sum window-level distinct counts and impacts within each pi bin."""
    summary = (
        data.groupby("pi_bin", observed=True, sort=True)
        .agg(
            windows=("window", "size"),
            catch_summed_window_probe_count=(
                "catch_distinct_probe_count", "sum"
            ),
            dampa_representative_summed_window_probe_count=(
                "dampa_representative_distinct_probe_count", "sum"
            ),
            dampa_consensus_summed_window_probe_count=(
                "dampa_consensus_distinct_probe_count", "sum"
            ),
            pangenome_impact_probe_count_sum=(PANGENOME_IMPACT_COUNT, "sum"),
            consensus_impact_probe_count_sum=(CONSENSUS_IMPACT_COUNT, "sum"),
            total_impact_probe_count_sum=(TOTAL_IMPACT_COUNT, "sum"),
        )
        .reset_index()
    )
    summary["decomposition_residual"] = (
        summary["total_impact_probe_count_sum"]
        - summary["pangenome_impact_probe_count_sum"]
        - summary["consensus_impact_probe_count_sum"]
    )
    return summary


def main() -> None:
    args = parse_args()
    data = load_decomposition_data(args)
    data, bin_order = add_pi_bins(
        data, args.pi_bin_width, args.maximum_pi, args.zero_pi_bin
    )
    output_path = Path(args.output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_name(f"{output_path.stem}_data.csv")
    summary_path = output_path.with_name(f"{output_path.stem}_binned_summary.csv")
    absolute_window_path = output_path.with_name(
        f"{output_path.stem}_absolute_probe_counts_by_window.csv"
    )
    absolute_bin_path = output_path.with_name(
        f"{output_path.stem}_absolute_probe_counts_binned_summary.csv"
    )

    figure = draw_decomposition_plot(data, bin_order, args)
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)
    data.to_csv(data_path, index=False)
    summarize_bins(data).to_csv(summary_path, index=False)
    absolute_probe_counts_by_window(data).to_csv(absolute_window_path, index=False)
    summarize_absolute_probe_counts(data).to_csv(absolute_bin_path, index=False)
    print(f"Saved decomposition plot: {output_path}")
    print(f"Saved paired-window data: {data_path}")
    print(f"Saved binned summary: {summary_path}")
    print(f"Saved absolute counts by window: {absolute_window_path}")
    print(f"Saved binned absolute-count summary: {absolute_bin_path}")


if __name__ == "__main__":
    main()
