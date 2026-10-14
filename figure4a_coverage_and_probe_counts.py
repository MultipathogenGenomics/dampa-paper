#!/usr/bin/env python3
r"""Create Figure 4i: (missing coverage and) probe-set size by species.
Note only probe set size is shown in figure 4
The input directory is expected to have the following layout::

    <species>/<program>/evaluation/
        <species>_<program>_pergenome_stats_of_perc_coverage.csv
        <species>_<program>_summary_stats_of_perc_coverage.csv

Panel i uses the ``0`` column of each per-genome CSV (percentage of each
genome with zero probe coverage). Panel ii uses the ``total_probes`` row and
``mean`` column of the corresponding summary CSV. Missing species/program
combinations are left blank rather than plotted as zero.

Inputs and dependencies
-----------------------
Python 3.10+, numpy, pandas and matplotlib. --input-dir is required. CSVs
have a header and an index in the first column: per-genome rows are genome
identifiers, while summary rows are metric names. Coverage values are percent
(0-100), not fractions; total_probes must be a non-negative integer.
SPECIES and PROGRAMS define the publication's directory names, display labels
and ordering; edit those tables for a different comparison. The script never
runs probe design or evaluation. Its inputs are evaluation exports.

Outputs are a two-panel PDF and PNG under --output (a path stem), plus
*_missing_coverage_data.csv, *_probe_count_data.csv and a printed summary. Existing outputs are overwritten. Point jitter is
reproducible via --seed. The default output is relative to the working directory.

Example:
    python figure_scripts/figure4a_coverage_and_probe_counts.py \
        --input-dir data/evaluations --output output/figure4a
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


DEFAULT_OUTPUT = Path("output/figure4a/coverage_and_probe_counts")
DEFAULT_RASTER_DPI = 1200
MIN_RASTER_DPI = 600


@dataclass(frozen=True)
class Species:
    directory: str
    label: str


@dataclass(frozen=True)
class Program:
    directory: str
    label: str
    colour: str


# Ordering, names and colours match Figure 4 in the manuscript.
SPECIES = (
    Species("jev", "JEV"),
    Species("hiv_groupM", "HIV Group M"),
    Species("eva", "Enterovirus A"),
    Species("hhv4", "HHV4"),
    Species("dengue", "Dengue"),
    Species("pertussis_18", r"$\it{B.\ pertussis}$"),
)
PROGRAMS = (
    Program("dampa", "DAMPA", "#3279a8"),
    Program("catch", "CATCH", "#398d3e"),
    Program("catch_large", "CATCH_large", "#c43d3d"),
    Program("probetools", "Probetools", "#e58025"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root of the species/program evaluation results tree.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output path stem, or a path ending in .pdf/.png. Both PDF and PNG "
            f"are written (default stem: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed upper limit for panel i; by default it is data-driven.",
    )
    parser.add_argument(
        "--violin",
        action="store_true",
        help="Overlay outline-only violin plots on the strip and box plots.",
    )
    point_mode = parser.add_mutually_exclusive_group()
    point_mode.add_argument(
        "--rasterize-points",
        dest="rasterize_points",
        action="store_true",
        help="Rasterize dense scatter points in the PDF (default).",
    )
    point_mode.add_argument(
        "--vector-points",
        dest="rasterize_points",
        action="store_false",
        help="Keep scatter points as vectors for Illustrator compatibility.",
    )
    parser.set_defaults(rasterize_points=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="Random seed for reproducible point jitter (default: 4).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_RASTER_DPI,
        help=(
            "PNG resolution and embedded PDF raster-point resolution "
            f"(minimum: {MIN_RASTER_DPI}; default: {DEFAULT_RASTER_DPI})."
        ),
    )
    args = parser.parse_args()
    if args.dpi < MIN_RASTER_DPI:
        parser.error(f"--dpi must be at least {MIN_RASTER_DPI}")
    return args


def input_paths(root: Path, species: Species, program: Program) -> tuple[Path, Path]:
    evaluation = root / species.directory / program.directory / "evaluation"
    prefix = f"{species.directory}_{program.directory}"
    return (
        evaluation / f"{prefix}_pergenome_stats_of_perc_coverage.csv",
        evaluation / f"{prefix}_summary_stats_of_perc_coverage.csv",
    )


def read_missing_coverage(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    if "0" not in frame.columns:
        raise ValueError(f"Missing required '0' column in {path}")
    values = pd.to_numeric(frame["0"], errors="raise").dropna()
    if ((values < 0) | (values > 100)).any():
        raise ValueError(f"Missing-coverage values outside 0-100 in {path}")
    return values


def read_probe_count(path: Path) -> int:
    frame = pd.read_csv(path, index_col=0)
    if "total_probes" not in frame.index or "mean" not in frame.columns:
        raise ValueError(
            f"Expected a 'total_probes' row and 'mean' column in {path}"
        )
    value = float(frame.loc["total_probes", "mean"])
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError(f"Invalid total_probes value {value!r} in {path}")
    return int(value)


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_parts: list[pd.DataFrame] = []
    count_rows: list[dict[str, object]] = []

    for species in SPECIES:
        for program in PROGRAMS:
            per_genome, summary = input_paths(root, species, program)
            if not per_genome.exists() and not summary.exists():
                warnings.warn(
                    f"No evaluation files for {species.directory}/{program.directory}; "
                    "leaving this combination blank.",
                    stacklevel=2,
                )
                continue
            if not per_genome.exists() or not summary.exists():
                missing = per_genome if not per_genome.exists() else summary
                raise FileNotFoundError(f"Incomplete evaluation pair; missing {missing}")

            values = read_missing_coverage(per_genome)
            coverage_parts.append(
                pd.DataFrame(
                    {
                        "genome": values.index.astype(str),
                        "species": species.directory,
                        "program": program.directory,
                        "missing_coverage_percent": values.to_numpy(),
                    }
                )
            )
            count_rows.append(
                {
                    "species": species.directory,
                    "program": program.directory,
                    "probe_count": read_probe_count(summary),
                    "genome_count": len(values),
                }
            )

    if not coverage_parts:
        raise FileNotFoundError(f"No evaluation CSV pairs found below {root}")
    return pd.concat(coverage_parts, ignore_index=True), pd.DataFrame(count_rows)


def print_coverage_summary(coverage: pd.DataFrame) -> None:
    """Print missing-coverage statistics for each organism/program pair."""
    rows: list[dict[str, object]] = []
    for species in SPECIES:
        for program in PROGRAMS:
            values = coverage.loc[
                (coverage["species"] == species.directory)
                & (coverage["program"] == program.directory),
                "missing_coverage_percent",
            ]
            if values.empty:
                continue
            rows.append(
                {
                    "organism": species.directory,
                    "program": program.label,
                    "n": len(values),
                    "mean": values.mean(),
                    "Q1": values.quantile(0.25),
                    "median": values.median(),
                    "Q3": values.quantile(0.75),
                }
            )

    summary = pd.DataFrame(rows)
    print("\nMissing coverage (%) by organism and program:")
    print(
        summary.to_string(
            index=False,
            formatters={
                statistic: lambda value: f"{value:.4f}"
                for statistic in ("mean", "Q1", "median", "Q3")
            },
        )
    )


def nice_upper_limit(maximum: float) -> float:
    """Return a readable upper limit with a little headroom."""
    if maximum <= 5:
        step = 1.0
    elif maximum <= 20:
        step = 5.0
    else:
        step = 10.0
    return max(step, math.ceil(maximum * 1.06 / step) * step)


def draw_figure(
    coverage: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    y_max: float | None,
    seed: int,
    violin: bool,
    rasterize_points: bool,
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cm = 1 / 2.54
    fig, (ax_cov, ax_count) = plt.subplots(
        2,
        1,
        figsize=(19 * cm, 10.5 * cm),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.03},
    )
    species_x = np.arange(len(SPECIES), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(PROGRAMS))
    rng = np.random.default_rng(seed)

    for species_index, species in enumerate(SPECIES):
        for program_index, program in enumerate(PROGRAMS):
            selected = coverage.loc[
                (coverage["species"] == species.directory)
                & (coverage["program"] == program.directory),
                "missing_coverage_percent",
            ].to_numpy()
            if not len(selected):
                continue

            centre = species_x[species_index] + offsets[program_index]
            jitter = rng.uniform(-0.055, 0.055, size=len(selected))
            ax_cov.scatter(
                centre + jitter,
                selected,
                s=7,
                color=program.colour,
                edgecolors="none",
                alpha=0.24,
                rasterized=rasterize_points,
                zorder=1,
            )
            if violin:
                violin_width = 0.18
                if np.ptp(selected) == 0:
                    # A kernel-density violin is undefined for constant data.
                    ax_cov.plot(
                        [centre - violin_width / 2, centre + violin_width / 2],
                        [selected[0], selected[0]],
                        color=program.colour,
                        linewidth=0.9,
                        zorder=2,
                    )
                else:
                    violin_parts = ax_cov.violinplot(
                        selected,
                        positions=[centre],
                        widths=violin_width,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                        bw_method="scott",
                    )
                    for body in violin_parts["bodies"]:
                        body.set_facecolor("none")
                        body.set_edgecolor(program.colour)
                        body.set_linewidth(0.9)
                        body.set_alpha(1.0)
                        body.set_zorder(2)
            ax_cov.boxplot(
                selected,
                positions=[centre],
                widths=0.14,
                showfliers=False,
                patch_artist=False,
                manage_ticks=False,
                boxprops={"color": "black", "linewidth": 0.8, "alpha": 0.55},
                whiskerprops={"color": "black", "linewidth": 0.8, "alpha": 0.55},
                capprops={"color": "black", "linewidth": 0.8, "alpha": 0.55},
                medianprops={"color": "black", "linewidth": 0.9, "alpha": 0.7},
                zorder=3,
            )

    legend_handles = [Patch(facecolor=p.colour, label=p.label) for p in PROGRAMS]
    ax_cov.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.86),
        frameon=True,
        borderpad=0.4,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    ax_cov.set_ylabel("% of genome missing coverage")
    upper = y_max if y_max is not None else nice_upper_limit(
        float(coverage["missing_coverage_percent"].max())
    )
    if upper <= 0:
        raise ValueError("--y-max must be greater than zero")
    ax_cov.set_ylim(-0.015 * upper, upper)
    ax_cov.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_cov.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_cov.spines[["top", "right"]].set_visible(False)
    ax_cov.text(-0.05, 0.99, "i", transform=ax_cov.transAxes, fontsize=14, va="top")

    bar_width = 0.18
    for species_index, species in enumerate(SPECIES):
        species_counts = counts.loc[counts["species"] == species.directory]
        if species_counts.empty:
            continue
        normalizer = float(species_counts["probe_count"].max())
        for program_index, program in enumerate(PROGRAMS):
            selected = species_counts.loc[
                species_counts["program"] == program.directory, "probe_count"
            ]
            if selected.empty:
                continue
            original = int(selected.iloc[0])
            height = 100 * original / normalizer if normalizer else 0
            centre = species_x[species_index] + offsets[program_index]
            ax_count.bar(
                centre,
                height,
                width=bar_width,
                color=program.colour,
                edgecolor="none",
                zorder=2,
            )
            inside = height >= 22
            ax_count.text(
                centre,
                height / 2 if inside else height + 2,
                f"{original}",
                ha="center",
                va="center" if inside else "bottom",
                rotation=90,
                fontsize=5.6,
                color="white" if inside else "black",
                clip_on=False,
            )

    genome_counts = coverage.groupby("species")["genome"].nunique().to_dict()
    tick_labels = [
        f"{species.label}\n({genome_counts.get(species.directory, 0)})"
        for species in SPECIES
    ]
    ax_count.set_xticks(species_x, tick_labels)
    ax_count.set_xlim(-0.5, len(SPECIES) - 0.5)
    ax_count.set_ylim(0, 110)
    ax_count.set_yticks([])
    ax_count.set_ylabel("Probe number")
    ax_count.tick_params(axis="x", length=3, pad=3)
    ax_count.spines[["top", "right", "left"]].set_visible(False)
    ax_count.text(-0.09, 0.98, "ii", transform=ax_count.transAxes, fontsize=14, va="top")

    fig.align_ylabels((ax_cov, ax_count))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.98, bottom=0.16)
    return fig


def output_stem(path: Path) -> Path:
    return path.with_suffix("") if path.suffix.lower() in {".pdf", ".png"} else path


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")

    coverage, counts = load_data(args.input_dir)
    print_coverage_summary(coverage)
    stem = output_stem(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(stem.with_name(f"{stem.name}_missing_coverage_data.csv"), index=False)
    counts.to_csv(stem.with_name(f"{stem.name}_probe_count_data.csv"), index=False)

    figure = draw_figure(
        coverage,
        counts,
        y_max=args.y_max,
        seed=args.seed,
        violin=args.violin,
        rasterize_points=args.rasterize_points,
    )
    figure.savefig(stem.with_suffix(".pdf"), dpi=args.dpi)
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    plt.close(figure)

    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
