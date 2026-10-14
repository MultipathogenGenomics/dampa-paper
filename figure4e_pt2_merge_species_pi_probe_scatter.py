#!/usr/bin/env python3
r"""Merge per-species pi/probe CSV outputs into one scatterplot.

Input files are the ``*_data.csv`` files written by
``figure4e_pt1_pi_vs_unique_prove_scatter.py``. Probe sets are distinguished by color and
species by marker shape.

Input contract
--------------
Python 3.10+, numpy, pandas, matplotlib, seaborn and scipy (regression via
seaborn); keep plot_program_colors.py alongside this script. --input-data
accepts headered CSVs with one row per probe_set/window and required columns:
window, probe_set, mean_unique_probe_count, and mean_pi (or median_pi when
--pi-summary median is requested). Pi is a per-site fraction, not percent.
Window identifiers must refer to the same retained alignment block across
probe sets within each species. --species-labels supplies one display label
per input file. Other exported columns are retained in the combined data.

Relative/difference modes pair each window with the first DAMPA-labelled probe
set for that species. Undefined zero-denominator ratios are omitted. Binned
comparison to DAMPA medians is optional and controlled by explicit flags.
--tools optionally selects and orders probe sets; default is all input tools.

Outputs: --output-file, *_data.csv and *_regression_summary.csv; --bin-pi also
writes *_binned_summary.csv. Existing outputs are overwritten.

Example:
    python figure_scripts/figure4e_pt2_merge_species_pi_probe_scatter.py \
        --input-data data/species1_data.csv data/species2_data.csv \
        --species-labels Species1 Species2 --output-file output/figure4e.pdf
"""

from __future__ import annotations

import argparse
import inspect
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from plot_program_colors import distinct_program_colors, program_color


mpl.rcParams["pdf.fonttype"] = 42
RAW_VALUE = "mean_unique_probe_count"
RELATIVE_VALUE = "mean_unique_probe_count_relative_to_dampa"
DIFFERENCE_VALUE = "mean_unique_probe_count_difference_from_dampa"
REQUIRED_COLUMNS = {"window", "probe_set", RAW_VALUE}
PI_VALUE = "pi_value"
A4_WIDTH_IN = 8.27
COMPACT_FIGURE_WIDTH = A4_WIDTH_IN * 0.5
COMPACT_FIGURE_HEIGHT = COMPACT_FIGURE_WIDTH * 0.75
MIN_TEXT_SIZE = 6
TITLE_WRAP_WIDTH = 38
SPECIES_MARKERS = ("o", "x", "^", "s", "D", "v", "P", "X", "<", ">", "*")


def finish_compact_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    font_size: float,
) -> None:
    """Lay out a narrow figure without allowing auto-layout to squash its axes."""
    ax.set_title(textwrap.fill(title, width=TITLE_WRAP_WIDTH), fontsize=font_size)
    ax.xaxis.label.set_fontsize(font_size)
    ax.yaxis.label.set_fontsize(font_size)
    # On a 40%-of-A4 figure, tight_layout tries to accommodate long titles and
    # legends by collapsing the data axes. Fixed margins keep the usable plot
    # width stable and preserve the requested physical output width.
    fig.tight_layout()


def program_rank(label: str) -> int:
    """Return the requested plotting order for probe-design programs."""
    normalized = str(label).lower().replace("-", "_").replace(" ", "_")
    if "dampa" in normalized:
        return 0
    if "catch" in normalized:
        return 1
    if "probetools" in normalized:
        return 2
    return 3


def ordered_program_labels(labels) -> list[str]:
    """Order labels DAMPA, CATCH, Probetools, then retain unknown labels."""
    unique = list(dict.fromkeys(str(label) for label in labels))
    original_position = {label: index for index, label in enumerate(unique)}
    return sorted(unique, key=lambda label: (program_rank(label), original_position[label]))


def resolve_program_order(labels, tools: list[str] | None) -> list[str]:
    """Use the requested ``--tools`` order, or the default program order."""
    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    if not tools:
        return ordered_program_labels(unique_labels)

    labels_by_key: dict[str, list[str]] = {}
    for label in unique_labels:
        labels_by_key.setdefault(label.casefold(), []).append(label)

    order: list[str] = []
    requested_keys = list(dict.fromkeys(str(tool).casefold() for tool in tools))
    for key in requested_keys:
        order.extend(labels_by_key.get(key, []))
    return order


def comparison_value_column(relative_to_dampa: bool, difference_from_dampa: bool) -> str:
    """Return the response column selected by the DAMPA comparison mode."""
    if relative_to_dampa:
        return RELATIVE_VALUE
    if difference_from_dampa:
        return DIFFERENCE_VALUE
    return RAW_VALUE


def add_dampa_reference(
    ax: plt.Axes, swap_axes: bool, difference_from_dampa: bool
) -> None:
    """Draw the no-change reference for a DAMPA comparison."""
    color = program_color("DAMPA")
    line_options = {
        "color": color,
        "linestyle": "--",
        "linewidth": 1.2,
        "alpha": 0.9,
        "label": "_nolegend_",
    }
    reference = 0.0 if difference_from_dampa else 1.0
    if swap_axes:
        ax.axvline(reference, **line_options)
    else:
        ax.axhline(reference, **line_options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-species pi-versus-probe data into one scatterplot."
    )
    parser.add_argument(
        "--input-data",
        nargs="+",
        required=True,
        help="Data CSV files produced by figure4e_pt1_pi_vs_unique_prove_scatter.py.",
    )
    parser.add_argument(
        "--species-labels",
        nargs="+",
        required=True,
        help="Species labels corresponding to --input-data, in the same order.",
    )
    parser.add_argument(
        "--tools",
        "--probe-sets",
        dest="tools",
        nargs="+",
        help=(
            "Plot only these tools from the input probe_set column "
            "(case-insensitive exact matches), in the supplied plotting and legend "
            "order. By default, all tools are plotted in the standard order."
        ),
    )
    parser.add_argument(
        "--output-file",
        "--output_file",
        dest="output_file",
        default="merged_species_pi_vs_unique_probe_count.pdf",
        help="Output plot (.pdf, .png, .svg, etc.).",
    )
    parser.add_argument(
        "--relative-to-dampa",
        action="store_true",
        help=(
            "Plot each probe set relative to the first DAMPA probe set within "
            "the same species and window. Existing relative values are reused "
            "when unambiguous."
        ),
    )
    parser.add_argument(
        "--difference-from-dampa",
        action="store_true",
        help=(
            "Plot the absolute difference in probe count from the first DAMPA "
            "probe set within the same species and window (tool minus DAMPA)."
        ),
    )
    parser.add_argument(
        "--normalise-to-median",
        "--normalize-to-median",
        dest="normalise_to_median",
        action="store_true",
        help=(
            "With a DAMPA comparison mode and --bin-pi, compare each observation "
            "with the first DAMPA probe set's median for its species and pi bin "
            "instead of its paired window."
        ),
    )
    parser.add_argument(
        "--ignore-species",
        action="store_true",
        help=(
            "Pool species when plotting and fitting regressions. Points are "
            "distinguished only by probe set."
        ),
    )
    parser.add_argument(
        "--bin-pi",
        action="store_true",
        help="Group points into pi bins and draw stripplots over boxplots.",
    )
    parser.add_argument(
        "--pi-summary",
        choices=("mean", "median"),
        default="mean",
        help=(
            "Window-level pi statistic to read from mean_pi or median_pi "
            "(default: mean)."
        ),
    )
    parser.add_argument(
        "--pi-bin-width",
        type=float,
        default=0.05,
        help="Width of pi bins used with --bin-pi (default: 0.05).",
    )
    parser.add_argument(
        "--zero-pi-bin",
        "--separate-zero-pi-bin",
        dest="zero_pi_bin",
        action="store_true",
        help=(
            "With --bin-pi, place observations whose selected pi statistic is "
            "exactly zero in a separate zero bin."
        ),
    )
    parser.add_argument(
        "--maximum-pi",
        "--max-pi",
        dest="maximum_pi",
        type=float,
        help=(
            "With --bin-pi, retain pi values up to this threshold in regular "
            "bins and merge all larger values into a terminal >X bin."
        ),
    )
    parser.add_argument(
        "--remove-strip",
        action="store_true",
        help="With --bin-pi, omit strip points.",
    )
    parser.add_argument(
        "--boxtype",
        choices=("boxplot", "violinplot"),
        default="boxplot",
        help=(
            "With --bin-pi, choose the underlying distribution plot type "
            "(default: boxplot)."
        ),
    )
    parser.add_argument(
        "--swap-axes",
        action="store_true",
        help="Swap the x and y axes in the output plot.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Set the displayed y-axis limits.",
    )
    parser.add_argument(
        "--log2-y",
        action="store_true",
        help="Display the numeric y axis on a base-2 logarithmic scale.",
    )
    parser.add_argument(
        "--plot-width",
        "--figure-width",
        dest="plot_width",
        type=float,
        default=COMPACT_FIGURE_WIDTH,
        help=f"Plot width in inches (default: {COMPACT_FIGURE_WIDTH:g}).",
    )
    parser.add_argument(
        "--plot-height",
        "--figure-height",
        dest="plot_height",
        type=float,
        default=COMPACT_FIGURE_HEIGHT,
        help=f"Plot height in inches (default: {COMPACT_FIGURE_HEIGHT:g}).",
    )
    parser.add_argument(
        "--font-size",
        "--text-size",
        dest="font_size",
        type=float,
        default=MIN_TEXT_SIZE,
        help=f"Font size for titles, labels, ticks, and legends (default: {MIN_TEXT_SIZE:g}).",
    )
    parser.add_argument("--point-size", type=float, default=45.0,
                        help="Scatter point area (default: 45).")
    parser.add_argument("--point-alpha", type=float, default=0.75,
                        help="Scatter point opacity (default: 0.75).")
    args = parser.parse_args()
    if len(args.input_data) != len(args.species_labels):
        parser.error("--input-data and --species-labels must have the same number of values")
    if len(set(args.species_labels)) != len(args.species_labels):
        parser.error("--species-labels values must be unique")
    if args.point_size <= 0:
        parser.error("--point-size must be greater than zero")
    if args.plot_width <= 0 or args.plot_height <= 0:
        parser.error("--plot-width and --plot-height must be greater than zero")
    if args.font_size <= 0:
        parser.error("--font-size must be greater than zero")
    if not 0 < args.point_alpha <= 1:
        parser.error("--point-alpha must be greater than 0 and at most 1")
    if args.pi_bin_width <= 0:
        parser.error("--pi-bin-width must be greater than zero")
    if args.ylim is not None and (
        not all(np.isfinite(limit) for limit in args.ylim)
        or args.ylim[0] >= args.ylim[1]
    ):
        parser.error("--ylim requires two finite values with MIN less than MAX")
    if args.log2_y and args.ylim is not None and args.ylim[0] <= 0:
        parser.error("--log2-y requires a strictly positive --ylim minimum")
    if args.log2_y and args.bin_pi and args.swap_axes:
        parser.error("--log2-y cannot be used on the categorical y axis created by --bin-pi --swap-axes")
    if args.maximum_pi is not None and (
        not np.isfinite(args.maximum_pi) or args.maximum_pi <= 0
    ):
        parser.error("--maximum-pi must be a finite value greater than zero")
    if args.maximum_pi is not None and not args.bin_pi:
        parser.error("--maximum-pi requires --bin-pi")
    if args.zero_pi_bin and not args.bin_pi:
        parser.error("--zero-pi-bin requires --bin-pi")
    if args.remove_strip and not args.bin_pi:
        parser.error("--remove-strip requires --bin-pi")
    if args.relative_to_dampa and args.difference_from_dampa:
        parser.error(
            "--relative-to-dampa and --difference-from-dampa are mutually exclusive"
        )
    if args.normalise_to_median and not (
        args.relative_to_dampa or args.difference_from_dampa
    ):
        parser.error(
            "--normalise-to-median requires --relative-to-dampa or "
            "--difference-from-dampa"
        )
    if args.normalise_to_median and not args.bin_pi:
        parser.error("--normalise-to-median requires --bin-pi")
    return args


def calculate_dampa_comparison_values(
    data: pd.DataFrame,
    species: str,
    difference_from_dampa: bool,
) -> pd.DataFrame:
    """Return non-DAMPA rows compared with the first DAMPA set per window."""
    value_column = DIFFERENCE_VALUE if difference_from_dampa else RELATIVE_VALUE
    dampa_labels = [
        label for label in data["probe_set"].dropna().unique()
        if "dampa" in str(label).lower()
    ]
    if (
        value_column in data.columns
        and data[value_column].notna().any()
        and len(dampa_labels) <= 1
    ):
        dampa_mask = data["probe_set"].str.contains("dampa", case=False, na=False)
        return data.loc[~dampa_mask].copy()

    if not dampa_labels:
        raise ValueError(
            f"Species {species!r} requires at least one probe_set containing 'DAMPA'"
        )
    dampa_label = dampa_labels[0]
    if len(dampa_labels) > 1:
        print(
            f"Species {species!r} has multiple DAMPA probe sets; using the first "
            f"as the relative reference: {dampa_label}",
            file=sys.stderr,
        )
    denominator = data.loc[
        data["probe_set"] == dampa_label, ["window", RAW_VALUE]
    ].rename(columns={RAW_VALUE: "dampa_mean_unique_probe_count"})
    comparison = data.loc[data["probe_set"] != dampa_label].merge(
        denominator, on="window", how="left", validate="many_to_one"
    )
    denominator_values = comparison["dampa_mean_unique_probe_count"]
    if difference_from_dampa:
        valid = np.isfinite(denominator_values)
        comparison[DIFFERENCE_VALUE] = (
            comparison[RAW_VALUE] - denominator_values.where(valid)
        )
    else:
        valid = np.isfinite(denominator_values) & denominator_values.gt(0)
        comparison[RELATIVE_VALUE] = (
            comparison[RAW_VALUE] / denominator_values.where(valid)
        )
    return comparison


def calculate_relative_values(data: pd.DataFrame, species: str) -> pd.DataFrame:
    """Return rows relative to the first DAMPA probe set for each window."""
    return calculate_dampa_comparison_values(data, species, False)


def load_species_data(
    input_files: list[str],
    species_labels: list[str],
    relative_to_dampa: bool,
    difference_from_dampa: bool,
    pi_summary: str,
) -> pd.DataFrame:
    frames = []
    pi_column = f"{pi_summary}_pi"
    for input_file, species in zip(input_files, species_labels):
        data = pd.read_csv(input_file)
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(
                f"{input_file} is missing required columns: {', '.join(sorted(missing))}"
            )
        if pi_column not in data.columns:
            raise ValueError(
                f"{input_file} is missing {pi_column!r}; regenerate it with "
                f"figure4e_pt1_pi_vs_unique_prove_scatter.py --pi-summary {pi_summary}"
            )
        data = data.copy()
        data[PI_VALUE] = pd.to_numeric(data[pi_column], errors="coerce")
        data["pi_summary"] = pi_summary
        data["species"] = species
        data["source_file"] = str(Path(input_file).expanduser())
        if relative_to_dampa or difference_from_dampa:
            data = calculate_dampa_comparison_values(
                data, species, difference_from_dampa
            )
        frames.append(data)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    value_column = comparison_value_column(
        relative_to_dampa, difference_from_dampa
    )
    return combined.dropna(subset=[PI_VALUE, value_column]).reset_index(drop=True)


def select_tools(data: pd.DataFrame, tools: list[str] | None) -> pd.DataFrame:
    """Return only requested probe sets, matching labels case-insensitively."""
    if not tools:
        return data

    requested = list(dict.fromkeys(str(tool).casefold() for tool in tools))
    labels = data["probe_set"].dropna().astype(str)
    available_by_key: dict[str, list[str]] = {}
    for label in dict.fromkeys(labels):
        available_by_key.setdefault(label.casefold(), []).append(label)

    missing = [tool for tool in tools if str(tool).casefold() not in available_by_key]
    if missing:
        available = ", ".join(ordered_program_labels(labels)) or "none"
        raise ValueError(
            "Requested tool(s) not found in the input data: "
            f"{', '.join(map(str, missing))}. Available tools: {available}"
        )

    selected = data["probe_set"].astype(str).str.casefold().isin(requested)
    return data.loc[selected].copy().reset_index(drop=True)


def regression_summary(
    data: pd.DataFrame, value_column: str, ignore_species: bool
) -> pd.DataFrame:
    rows = []
    group_columns = ["probe_set"] if ignore_species else ["species", "probe_set"]
    for group_key, subset in data.groupby(group_columns, sort=False):
        if ignore_species:
            species = "all species"
            probe_set = group_key[0] if isinstance(group_key, tuple) else group_key
        else:
            species, probe_set = group_key
        x = subset[PI_VALUE].to_numpy(dtype=float)
        y = subset[value_column].to_numpy(dtype=float)
        if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            pearson_r = float(np.corrcoef(x, y)[0, 1])
            r_squared = pearson_r ** 2
        else:
            slope = intercept = pearson_r = r_squared = np.nan
        rows.append(
            {
                "species": species,
                "probe_set": probe_set,
                "pi_summary": str(subset["pi_summary"].iloc[0]),
                "response_variable": value_column,
                "windows": len(subset),
                "slope": slope,
                "intercept": intercept,
                "pearson_r": pearson_r,
                "r_squared": r_squared,
            }
        )
    return pd.DataFrame(rows)


def window_description(data: pd.DataFrame) -> str:
    if "positions_in_window" not in data.columns:
        return "non-overlapping windows"
    sizes = sorted(data["positions_in_window"].dropna().astype(int).unique())
    if len(sizes) == 1:
        return f"non-overlapping {sizes[0]}-position blocks"
    return "non-overlapping windows"


def add_pi_bins(
    data: pd.DataFrame,
    bin_width: float,
    maximum_pi: float | None = None,
    separate_zero: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Assign regular pi bins plus optional exact-zero and terminal bins."""
    data = data.copy()
    pi_values = data[PI_VALUE].to_numpy(dtype=float)
    decimals = max(2, int(np.ceil(-np.log10(bin_width))) + 1) if bin_width < 1 else 2

    overflow = np.zeros(len(data), dtype=bool)
    values_for_regular_bins = pi_values.copy()
    if maximum_pi is not None:
        overflow = pi_values > maximum_pi
        # Include values exactly equal to the threshold in the final regular
        # interval, even when the threshold is an exact bin-width multiple.
        values_for_regular_bins = np.minimum(
            values_for_regular_bins,
            np.nextafter(maximum_pi, -np.inf),
        )
    bin_index = np.floor(values_for_regular_bins / bin_width).astype(int)
    bin_starts = np.round(bin_index * bin_width, decimals)
    bin_ends = (bin_index + 1) * bin_width
    if maximum_pi is not None:
        bin_ends = np.minimum(bin_ends, maximum_pi)
    bin_ends = np.round(bin_ends, decimals)
    data["pi_bin_start"] = bin_starts
    data["pi_bin_end"] = bin_ends

    zero = (pi_values == 0.0) & separate_zero
    regular_indices = sorted(np.unique(bin_index[~overflow & ~zero]))
    labels_by_index = {
        index: (
            f"{index * bin_width:.{decimals}f}–"
            f"{min((index + 1) * bin_width, maximum_pi) if maximum_pi is not None else (index + 1) * bin_width:.{decimals}f}"
        )
        for index in regular_indices
    }
    labels = [labels_by_index.get(index, "") for index in bin_index]
    order = []
    if zero.any():
        zero_label = "0"
        for row_index in np.flatnonzero(zero):
            labels[row_index] = zero_label
        data.loc[zero, "pi_bin_start"] = 0.0
        data.loc[zero, "pi_bin_end"] = 0.0
        order.append(zero_label)
    order.extend(labels_by_index[index] for index in regular_indices)
    if maximum_pi is not None and overflow.any():
        overflow_label = f">{maximum_pi:.2f}"
        for row_index in np.flatnonzero(overflow):
            labels[row_index] = overflow_label
        data.loc[overflow, "pi_bin_start"] = np.round(maximum_pi, decimals)
        data.loc[overflow, "pi_bin_end"] = np.inf
        order.append(overflow_label)
    data["pi_bin"] = labels
    data["pi_bin_overflow"] = overflow
    data["pi_bin"] = pd.Categorical(data["pi_bin"], categories=order, ordered=True)
    return data, order


def calculate_relative_to_dampa_bin_median(
    data: pd.DataFrame,
    difference_from_dampa: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Compare with the first DAMPA set's median per species and pi bin."""
    reference_masks = []
    for species, group in data.groupby("species", sort=False):
        dampa_labels = [
            label for label in group["probe_set"].dropna().unique()
            if "dampa" in str(label).lower()
        ]
        if not dampa_labels:
            raise ValueError(
                "--normalise-to-median requires raw data containing a DAMPA "
                f"probe set for species {species!r}"
            )
        dampa_label = dampa_labels[0]
        if len(dampa_labels) > 1:
            print(
                f"Species {species!r} has multiple DAMPA probe sets; using the first "
                f"for median normalization: {dampa_label}",
                file=sys.stderr,
            )
        reference_masks.append(
            data["species"].eq(species) & data["probe_set"].eq(dampa_label)
        )
    if not reference_masks:
        raise ValueError(
            "--normalise-to-median requires raw per-species data containing DAMPA rows; "
            "a previously DAMPA-normalized CSV does not contain the required rows"
        )
    dampa_mask = np.logical_or.reduce(reference_masks)
    group_columns = ["species", "pi_bin_start", "pi_bin_end"]
    dampa_medians = (
        data.loc[dampa_mask]
        .groupby(group_columns, observed=True, sort=False)[RAW_VALUE]
        .median()
        .rename("dampa_median_unique_probe_count")
        .reset_index()
    )
    normalized = data.merge(
        dampa_medians, on=group_columns, how="left", validate="many_to_one"
    )
    denominator = normalized["dampa_median_unique_probe_count"]
    if difference_from_dampa:
        valid = np.isfinite(denominator)
        normalized[DIFFERENCE_VALUE] = normalized[RAW_VALUE] - denominator.where(valid)
    else:
        valid = np.isfinite(denominator) & denominator.gt(0)
        normalized[RELATIVE_VALUE] = normalized[RAW_VALUE] / denominator.where(valid)
    return normalized, int((~valid).sum())


def binned_summary(
    data: pd.DataFrame, value_column: str, ignore_species: bool
) -> pd.DataFrame:
    group_columns = ["pi_bin_start", "pi_bin_end", "pi_bin", "probe_set"]
    if not ignore_species:
        group_columns.insert(0, "species")
    summary = (
        data.groupby(group_columns, observed=True, sort=False)[value_column]
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
    if ignore_species:
        summary.insert(0, "species", "all species")
    summary.insert(
        summary.columns.get_loc("observations"), "response_variable", value_column
    )
    return summary


def draw_bin_normalized_violins(
    ax: plt.Axes,
    plot_data: pd.DataFrame,
    value_column: str,
    pi_bin_order: list[str],
    hue_order: list[str],
    palette: dict[str, str],
    swap_axes: bool,
) -> None:
    """Draw violins whose count-based widths are normalized within each pi bin."""
    violin_parameters = inspect.signature(sns.violinplot).parameters
    for bin_label in pi_bin_order:
        bin_data = plot_data.loc[plot_data["pi_bin"] == bin_label]
        if bin_data.empty:
            continue
        grouped_values = bin_data.groupby("Plot group", observed=True)[value_column]
        has_density = (
            grouped_values.transform("count").ge(2)
            & grouped_values.transform("nunique").ge(2)
        )
        density_data = bin_data.loc[has_density]
        if density_data.empty:
            continue
        options = {
            "data": density_data,
            "x": value_column if swap_axes else "pi_bin",
            "y": "pi_bin" if swap_axes else value_column,
            "hue": "Plot group",
            "order": pi_bin_order,
            "hue_order": hue_order,
            "palette": palette,
            "dodge": True,
            "cut": 0,
            "inner": None,
            "linewidth": 1.0,
            "ax": ax,
        }
        if "density_norm" in violin_parameters:
            options.update(density_norm="count", common_norm=True)
        else:
            # Seaborn <0.13 uses these names. scale_hue=True makes the
            # count-based normalization local to the current x-axis bin.
            options.update(scale="count", scale_hue=True)
        sns.violinplot(**options)

    # Draw these ourselves so the summaries remain visible even for groups
    # with too few observations for a kernel-density violin.
    total_width = 0.8
    hue_width = total_width / len(hue_order)
    hue_offsets = (
        np.arange(len(hue_order), dtype=float) * hue_width
        + hue_width / 2
        - total_width / 2
    )
    for bin_index, bin_label in enumerate(pi_bin_order):
        bin_data = plot_data.loc[plot_data["pi_bin"] == bin_label]
        for hue_index, group in enumerate(hue_order):
            values = bin_data.loc[
                bin_data["Plot group"] == group, value_column
            ].dropna()
            if values.empty:
                continue
            q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
            category_position = bin_index + hue_offsets[hue_index]
            if swap_axes:
                ax.hlines(
                    category_position,
                    q1,
                    q3,
                    color="#202020",
                    linewidth=2.3,
                    zorder=5,
                )
                point_x, point_y = median, category_position
            else:
                ax.vlines(
                    category_position,
                    q1,
                    q3,
                    color="#202020",
                    linewidth=2.3,
                    zorder=5,
                )
                point_x, point_y = category_position, median
            ax.scatter(
                [point_x],
                [point_y],
                s=28,
                marker="o",
                facecolor="white",
                edgecolor="#202020",
                linewidth=1.0,
                zorder=6,
            )


def remove_group_outliers(
    plot_data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Remove Tukey outliers independently within each pi-bin/program group."""
    groups = plot_data.groupby(["pi_bin", "Plot group"], observed=True)[value_column]
    q1 = groups.transform(lambda values: values.quantile(0.25))
    q3 = groups.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    keep = plot_data[value_column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return plot_data.loc[keep].copy()


def apply_log2_y_axis(ax: plt.Axes, values: pd.Series, enabled: bool) -> None:
    """Apply a base-2 y scale after verifying that all plotted values are positive."""
    if not enabled:
        return
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        raise ValueError("--log2-y requires at least one finite y value")
    nonpositive = int((finite <= 0).sum())
    if nonpositive:
        raise ValueError(
            f"--log2-y cannot display {nonpositive:,} y value(s) that are zero or negative"
        )
    ax.set_yscale("log", base=2)
    ax.set_ylabel(f"{ax.get_ylabel()} (log₂ scale)")


def plain_number_tick(value: float, _position: float) -> str:
    """Format a tick as an ordinary number without scientific notation."""
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1:
        decimals = 0 if np.isclose(value, round(value)) else 2
    else:
        decimals = min(12, max(1, int(np.ceil(-np.log10(magnitude))) + 2))
    formatted = f"{value:.{decimals}f}"
    return formatted.rstrip("0").rstrip(".") if decimals else formatted


def use_plain_y_tick_labels(ax: plt.Axes) -> None:
    """Prevent scientific or power notation on a numeric y axis."""
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number_tick))


def draw_species_styled_points(
    ax: plt.Axes,
    plot_data: pd.DataFrame,
    value_column: str,
    pi_bin_order: list[str],
    program_order: list[str],
    program_palette: dict[str, str],
    point_size: float,
    point_alpha: float,
    swap_axes: bool,
) -> tuple[list[Line2D], list[str]]:
    """Draw points in program colours with species-specific marker types."""
    species_order = list(dict.fromkeys(plot_data["species"].astype(str)))
    species_styles = {
        species: SPECIES_MARKERS[index % len(SPECIES_MARKERS)]
        for index, species in enumerate(species_order)
    }
    total_width = 0.8
    program_width = total_width / len(program_order)
    program_offsets = (
        np.arange(len(program_order), dtype=float) * program_width
        + program_width / 2
        - total_width / 2
    )
    rng = np.random.default_rng(4)
    for bin_index, bin_label in enumerate(pi_bin_order):
        for program_index, program in enumerate(program_order):
            group = plot_data.loc[
                (plot_data["pi_bin"] == bin_label)
                & (plot_data["probe_set"] == program)
            ]
            centre = bin_index + program_offsets[program_index]
            for species in species_order:
                values = group.loc[
                    group["species"].astype(str) == species, value_column
                ].dropna()
                if values.empty:
                    continue
                jitter = rng.uniform(
                    -0.28 * program_width,
                    0.28 * program_width,
                    size=len(values),
                )
                positions = centre + jitter
                marker = species_styles[species]
                point_options = {
                    "s": point_size,
                    "marker": marker,
                    "color": program_palette[program],
                    "alpha": point_alpha,
                    "zorder": 4,
                }
                if marker == "x":
                    point_options["linewidths"] = 0.8
                else:
                    point_options.update(edgecolors="white", linewidths=0)
                ax.scatter(
                    values if swap_axes else positions,
                    positions if swap_axes else values,
                    **point_options,
                )
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=species_styles[species],
            markerfacecolor="#606060" if species_styles[species] == "o" else "none",
            markeredgecolor="#606060",
            markeredgewidth=0.8,
            markersize=max(3.0, np.sqrt(point_size)),
            label=species,
        )
        for species in species_order
    ]
    return handles, species_order


def save_binned_plot(
    data: pd.DataFrame,
    pi_bin_order: list[str],
    program_order: list[str],
    output_path: Path,
    relative_to_dampa: bool,
    difference_from_dampa: bool,
    point_size: float,
    point_alpha: float,
    ignore_species: bool,
    remove_strip: bool,
    normalise_to_median: bool,
    boxtype: str,
    swap_axes: bool,
    plot_width: float,
    plot_height: float,
    font_size: float,
    ylim: list[float] | None,
    log2_y: bool,
) -> None:
    value_column = comparison_value_column(
        relative_to_dampa, difference_from_dampa
    )
    pi_summary = str(data["pi_summary"].iloc[0]).capitalize()
    plot_data = data.copy()
    # Distributions are pooled by program. When species are retained they are
    # encoded only in the overlaid point shade and marker, not as extra boxes.
    plot_data["Plot group"] = plot_data["probe_set"]
    legend_title = "Probe set" if ignore_species else "Probe set / Species"
    hue_order = program_order
    palette = dict(zip(hue_order, distinct_program_colors(hue_order)))
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    categorical_axis = {
        "x": value_column if swap_axes else "pi_bin",
        "y": "pi_bin" if swap_axes else value_column,
    }
    displayed_data = plot_data
    if boxtype == "boxplot":
        boxplot_palette = {group: "black" for group in hue_order}
        sns.boxplot(
            data=plot_data,
            **categorical_axis,
            hue="Plot group",
            order=pi_bin_order,
            fill=False,
            linewidth=0.5,
            hue_order=hue_order,
            palette=boxplot_palette,
            # Keep boxplot outlines slightly transparent to reduce overdraw.
            boxprops={"alpha": 0.6},
            whiskerprops={"alpha": 0.6},
            capprops={"alpha": 0.6},
            medianprops={"alpha": 0.6},
            # Outliers are supplied by the optional strip layer, avoiding
            # duplicate markers; --remove-strip therefore hides them fully.
            showfliers=False,
            ax=ax,
        )
    elif boxtype == "violinplot":
        displayed_data = remove_group_outliers(plot_data, value_column)
        draw_bin_normalized_violins(
            ax,
            displayed_data,
            value_column,
            pi_bin_order,
            hue_order,
            palette,
            swap_axes,
        )
    else:
        raise ValueError(f"Unsupported boxtype: {boxtype}")
    species_handles: list[Line2D] = []
    species_labels: list[str] = []
    if not remove_strip and ignore_species:
        sns.stripplot(
            # Keep all observations, including Tukey outliers, in the strip.
            # With --remove-strip they are absent from the plot entirely.
            data=plot_data,
            **categorical_axis,
            hue="Plot group",
            order=pi_bin_order,
            hue_order=hue_order,
            palette=palette,
            dodge=True,
            jitter=0.18,
            size=max(2.0, np.sqrt(point_size)),
            alpha=point_alpha,
            edgecolor="white",
            linewidth=0,
            marker=".",
            ax=ax,
        )
    elif not remove_strip:
        species_handles, species_labels = draw_species_styled_points(
            ax,
            plot_data,
            value_column,
            pi_bin_order,
            hue_order,
            palette,
            point_size,
            point_alpha,
            swap_axes,
        )
    if difference_from_dampa:
        add_dampa_reference(ax, swap_axes, True)
        value_axis_label = "Probe-count difference from DAMPA"
        metric_name = "DAMPA-difference probe count"
    elif relative_to_dampa:
        add_dampa_reference(ax, swap_axes, False)
        value_axis_label = "Relative composition bias"
        metric_name = "DAMPA-relative probe count"
    else:
        value_axis_label = "Unique probe count"
        metric_name = "Unique-probe count"
    program_handles = [
        Patch(facecolor="none", edgecolor=palette[label], linewidth=1.2, label=label)
        for label in hue_order
    ]
    unique_handles = program_handles + species_handles
    unique_labels = hue_order + species_labels
    ax.legend(
        unique_handles,
        unique_labels,
        title=legend_title,
        frameon=False,
        loc="upper right",
        title_fontsize=font_size,
        prop={"size": font_size},
    )
    pooling = (
        "pooled across species"
        if ignore_species
        else "species shown by point style"
    )
    bin_labels = data["pi_bin"].astype(str)
    bin_endpoints = [
        float(data.loc[bin_labels == bin_label, "pi_bin_end"].iloc[0])
        for bin_label in pi_bin_order
    ]
    bin_ticks = np.arange(len(pi_bin_order), dtype=float) + 0.5
    bin_tick_labels = [
        str(bin_label)
        if str(bin_label) == "0" or not np.isfinite(endpoint)
        else f"{endpoint:.2f}"
        for bin_label, endpoint in zip(pi_bin_order, bin_endpoints)
    ]
    if swap_axes:
        ax.set_yticks(bin_ticks)
        ax.set_yticklabels(bin_tick_labels)
        ax.set_ylim(-0.5, len(pi_bin_order) - 0.2)
        ax.set_xlabel(value_axis_label)
        ax.set_ylabel(f"{pi_summary} nucleotide diversity (π)")
    else:
        ax.set_xticks(bin_ticks)
        ax.set_xticklabels(bin_tick_labels)
        ax.set_xlim(-0.5, len(pi_bin_order) - 0.2)
        ax.set_xlabel(f"{pi_summary} nucleotide diversity (π)")
        ax.set_ylabel(value_axis_label)
    apply_log2_y_axis(ax, plot_data[value_column], log2_y)
    if ylim is not None:
        ax.set_ylim(ylim)
    if not swap_axes:
        use_plain_y_tick_labels(ax)
    ax.tick_params(axis="both", labelsize=font_size, labelrotation=0)
    sns.despine(ax=ax)
    finish_compact_figure(
        fig,
        ax,
        f"{metric_name} by {pi_summary.lower()} nucleotide-diversity bin, {pooling}",
        font_size,
    )
    fig.savefig(output_path)
    plt.close(fig)


def save_plot(
    data: pd.DataFrame,
    program_order: list[str],
    output_file: str,
    relative_to_dampa: bool,
    difference_from_dampa: bool,
    point_size: float,
    point_alpha: float,
    ignore_species: bool,
    swap_axes: bool,
    plot_width: float,
    plot_height: float,
    font_size: float,
    ylim: list[float] | None,
    log2_y: bool,
) -> Path:
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    value_column = comparison_value_column(
        relative_to_dampa, difference_from_dampa
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    plot_data = data.rename(columns={"probe_set": "Probe set", "species": "Species"})
    pi_summary = str(data["pi_summary"].iloc[0]).capitalize()
    probe_set_order = program_order
    probe_set_palette = dict(
        zip(probe_set_order, distinct_program_colors(probe_set_order))
    )
    scatter_options = {
        "data": plot_data,
        "x": value_column if swap_axes else PI_VALUE,
        "y": PI_VALUE if swap_axes else value_column,
        "hue": "Probe set",
        "hue_order": probe_set_order,
        "palette": probe_set_palette,
        "s": point_size,
        "alpha": point_alpha,
        "edgecolor": "white",
        "linewidth": 0.35,
        "ax": ax,
    }
    if not ignore_species:
        scatter_options["style"] = "Species"
    sns.scatterplot(**scatter_options)
    diversity_label = (
        f"{pi_summary} nucleotide diversity (π) per {window_description(data)}"
    )
    if difference_from_dampa:
        add_dampa_reference(ax, swap_axes, True)
        value_axis_label = "Probe-count difference from DAMPA"
        title = "Nucleotide diversity versus probe-count difference from DAMPA"
    elif relative_to_dampa:
        add_dampa_reference(ax, swap_axes, False)
        value_axis_label = "Relative composition bias"
        title = "Nucleotide diversity versus DAMPA-relative probe count"
    else:
        value_axis_label = "Unique probe count"
        title = "Nucleotide diversity versus unique-probe count"
    if swap_axes:
        ax.set_xlabel(value_axis_label)
        ax.set_ylabel(diversity_label)
    else:
        ax.set_xlabel(diversity_label)
        ax.set_ylabel(value_axis_label)
    y_column = PI_VALUE if swap_axes else value_column
    apply_log2_y_axis(ax, plot_data[y_column], log2_y)
    if ylim is not None:
        ax.set_ylim(ylim)
    use_plain_y_tick_labels(ax)
    title_suffix = " pooled across species" if ignore_species else " across species"
    handles, labels = ax.get_legend_handles_labels()
    if "DAMPA" in labels:
        dampa_index = labels.index("DAMPA")
        dampa_item = (handles.pop(dampa_index), labels.pop(dampa_index))
        insert_at = labels.index("Probe set") + 1 if "Probe set" in labels else 0
        handles.insert(insert_at, dampa_item[0])
        labels.insert(insert_at, dampa_item[1])
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper right",
        prop={"size": font_size},
        title_fontsize=font_size,
    )
    ax.tick_params(axis="both", labelsize=font_size, labelrotation=0)
    sns.despine(ax=ax)
    finish_compact_figure(fig, ax, title + title_suffix, font_size)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    compare_to_dampa = args.relative_to_dampa or args.difference_from_dampa
    compare_per_window = compare_to_dampa and not args.normalise_to_median
    data = load_species_data(
        args.input_data,
        args.species_labels,
        args.relative_to_dampa and compare_per_window,
        args.difference_from_dampa and compare_per_window,
        args.pi_summary,
    )
    # Median normalization still needs the DAMPA observations as references,
    # even when DAMPA itself is not part of the requested plotted subset.
    if not args.normalise_to_median:
        data = select_tools(data, args.tools)
    if data.empty:
        raise ValueError("No valid points remained after merging the input files")
    program_order = resolve_program_order(data["probe_set"], args.tools)
    rank_by_program = {label: rank for rank, label in enumerate(program_order)}
    data = (
        data.assign(_program_order=data["probe_set"].astype(str).map(rank_by_program))
        .sort_values("_program_order", kind="stable")
        .drop(columns="_program_order")
        .reset_index(drop=True)
    )
    value_column = comparison_value_column(
        args.relative_to_dampa, args.difference_from_dampa
    )
    binned_summary_path = None
    if args.bin_pi:
        data, pi_bin_order = add_pi_bins(
            data,
            args.pi_bin_width,
            args.maximum_pi,
            args.zero_pi_bin,
        )
        if args.normalise_to_median:
            data, undefined = calculate_relative_to_dampa_bin_median(
                data, args.difference_from_dampa
            )
            if undefined:
                requirement = "finite" if args.difference_from_dampa else "positive"
                print(
                    f"Warning: {undefined:,} observations lacked a {requirement} DAMPA "
                    "median in their pi bin",
                    file=sys.stderr,
                )
            data = select_tools(data, args.tools)
            if data.empty:
                raise ValueError("No valid points remained after selecting tools")
            data["pi_bin"] = data["pi_bin"].cat.remove_unused_categories()
            pi_bin_order = [str(label) for label in data["pi_bin"].cat.categories]
        output_path = Path(args.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_binned_plot(
            data,
            pi_bin_order,
            program_order,
            output_path,
            args.relative_to_dampa,
            args.difference_from_dampa,
            args.point_size,
            args.point_alpha,
            args.ignore_species,
            args.remove_strip,
            args.normalise_to_median,
            args.boxtype,
            args.swap_axes,
            args.plot_width,
            args.plot_height,
            args.font_size,
            args.ylim,
            args.log2_y,
        )
        binned_summary_path = output_path.with_name(
            f"{output_path.stem}_binned_summary.csv"
        )
        binned_summary(data, value_column, args.ignore_species).to_csv(
            binned_summary_path, index=False
        )
    else:
        output_path = save_plot(
            data,
            program_order,
            args.output_file,
            args.relative_to_dampa,
            args.difference_from_dampa,
            args.point_size,
            args.point_alpha,
            args.ignore_species,
            args.swap_axes,
            args.plot_width,
            args.plot_height,
            args.font_size,
            args.ylim,
            args.log2_y,
        )
    data_path = output_path.with_name(f"{output_path.stem}_data.csv")
    summary_path = output_path.with_name(f"{output_path.stem}_regression_summary.csv")
    data.to_csv(data_path, index=False)
    regression_summary(data, value_column, args.ignore_species).to_csv(summary_path, index=False)
    print(f"Saved merged plot: {output_path}")
    print(f"Saved merged data: {data_path}")
    if binned_summary_path is not None:
        print(f"Saved binned summary: {binned_summary_path}")
    print(f"Saved regression summary: {summary_path}")


if __name__ == "__main__":
    main()
