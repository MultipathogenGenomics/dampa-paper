#!/usr/bin/env python3
r"""Scatterplot nucleotide diversity against unique-probe count in 120mers.

Alignment columns are first filtered to retain only positions present (non-gap)
in strictly more than 90% of strains. The retained positions are then divided,
in alignment order, into non-overlapping blocks of 120 positions. Each plotted
point contains the block's selected per-site nucleotide-diversity summary (pi)
on the x axis and mean distinct-probe count per position on the y axis. The
exported data also contain the absolute number of distinct probe IDs that
overlap each window.

Inputs and outputs
------------------
Python 3.10+, numpy, pandas, matplotlib, seaborn, scipy and biopython.
Keep probe_count_normalized_by_pi.py, probe_coverage_boxplot_by_gene.py and
plot_program_colors.py alongside this script (and their local imports).

BLAST inputs are headerless nine-column TSVs in this exact order:
qseqid sseqid pident qstart qend sstart send evalue bitscore (tab-separated),
corresponding to BLAST -outfmt "6 qseqid sseqid pident qstart qend sstart send
evalue bitscore". qseqid identifies a probe; sseqid identifies an ungapped
subject genome whose ID matches a FASTA alignment record. Coordinates are
1-based inclusive and may be reversed. Do not use the standard twelve-column
outfmt 6 layout. Each file represents one probe set, paired with one label.
The alignment is a nucleotide FASTA MSA with unique IDs and equal row lengths;
BLAST subjects must use the ungapped sequences in that alignment.

Gap filtering, identity/hit-length cutoffs and window size are command-line
options. GFF annotations are not needed for this scatterplot. Output consists
of --output_file, a sibling *_data.csv (one row per probe set/window), and
*_regression_summary.csv. Existing files are overwritten. These data CSVs
are the input to figure4e_pt2_merge_species_pi_probe_scatter.py.

Example:
    python figure_scripts/figure4e_pt1_pi_vs_unique_prove_scatter.py \
        --alignment_file data/genomes.fasta --blast_file data/probes.tsv \
        --blast-labels DAMPA --output_file output/pi_probe.pdf
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
from Bio import SeqIO

from probe_count_normalized_by_pi import calculate_site_pi
from probe_coverage_boxplot_by_gene import (
    load_alignment,
    unique_probe_counts_with_positions,
)
from plot_program_colors import distinct_program_colors, program_color


mpl.rcParams["pdf.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scatterplot mean nucleotide diversity versus mean distinct-probe "
            "count in non-overlapping alignment windows."
        )
    )
    parser.add_argument("--blast_file", nargs="+", required=True,
                        help="One or more tab-separated BLAST result files.")
    parser.add_argument("--blast-labels", nargs="+", required=True,
                        help="Labels for --blast_file, in the same order.")
    parser.add_argument("--alignment_file", required=True,
                        help="Aligned genome FASTA used as the BLAST database source.")
    parser.add_argument("--output_file", default="pi_vs_unique_probe_count_120mers.pdf",
                        help="Output scatterplot (.pdf, .png, .svg, etc.).")
    parser.add_argument("--window-size", "--window_size", dest="window_size", type=int,
                        default=120, help="Non-overlapping window size (default: 120).")
    parser.add_argument(
        "--pi-summary",
        choices=("mean", "median"),
        default="mean",
        help="Statistic used to summarize per-site pi within each window (default: mean).",
    )
    parser.add_argument("--min-non-gap-proportion", type=float, default=0.9,
                        help=(
                            "Retain positions present in strictly more than this "
                            "proportion of strains (default: 0.9)."
                        ))
    parser.add_argument("--keep-partial-window", action="store_true",
                        help="Keep the final block if it contains fewer than --window-size positions.")
    parser.add_argument("--min-pident", type=float, default=80.0,
                        help="Minimum BLAST percent identity (default: 80).")
    parser.add_argument("--min-hit-length", type=int, default=81,
                        help="Minimum inclusive subject hit length (default: 81).")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="BLAST rows read per chunk (default: 100000).")
    parser.add_argument("--point-size", type=float, default=38.0,
                        help="Scatter point area (default: 38).")
    parser.add_argument("--point-alpha", type=float, default=0.75,
                        help="Scatter point opacity (default: 0.75).")
    parser.add_argument(
        "--relative-to-dampa",
        action="store_true",
        help=(
            "Plot each non-DAMPA window's mean probe count divided by the "
            "DAMPA mean for that window. Exactly one label must contain 'DAMPA'."
        ),
    )

    # Accepted so commands for the other alignment plotters can be reused.
    parser.add_argument("--gff_file", help=argparse.SUPPRESS)
    parser.add_argument("--strain_to_copy_annots", help=argparse.SUPPRESS)
    parser.add_argument("--overlay-blasts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bottom-panels", nargs="+", help=argparse.SUPPRESS)

    args = parser.parse_args()
    if len(args.blast_file) != len(args.blast_labels):
        parser.error("--blast_file and --blast-labels must have the same number of values")
    if len(set(args.blast_labels)) != len(args.blast_labels):
        parser.error("--blast-labels values must be unique")
    if args.relative_to_dampa:
        dampa_labels = [label for label in args.blast_labels if "dampa" in label.lower()]
        if len(dampa_labels) != 1:
            parser.error(
                "--relative-to-dampa requires exactly one --blast-labels value "
                "containing 'DAMPA'"
            )
        if len(args.blast_labels) < 2:
            parser.error("--relative-to-dampa requires at least one non-DAMPA probe set")
    if args.window_size < 1:
        parser.error("--window-size must be at least 1")
    if not 0 <= args.min_non_gap_proportion < 1:
        parser.error("--min-non-gap-proportion must be at least 0 and less than 1")
    if args.min_hit_length < 1:
        parser.error("--min-hit-length must be at least 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.point_size <= 0:
        parser.error("--point-size must be greater than zero")
    if not 0 < args.point_alpha <= 1:
        parser.error("--point-alpha must be greater than 0 and at most 1")
    return args


def first_alignment_id(alignment_file: str) -> str:
    record = next(SeqIO.parse(alignment_file, "fasta"), None)
    if record is None:
        raise ValueError("The alignment contains no FASTA records")
    return record.id


def add_enterovirus_a_blast_aliases(
    position_maps: dict[str, np.ndarray],
) -> int:
    """Alias legacy alignment IDs to the spelling used in BLAST hits.

    For example, the alignment ID::

        enterovirus-A~coxsackievirus_ON730871.1

    is also made available as::

        enterovirus~A~coxsackievirus_ON730871.1
    """
    alignment_prefix = "enterovirus-A~"
    blast_prefix = "enterovirus~A~"
    aliases = {}
    for alignment_id, position_map in position_maps.items():
        if not alignment_id.startswith(alignment_prefix):
            continue
        alias = blast_prefix + alignment_id[len(alignment_prefix):]
        if alias in position_maps:
            raise ValueError(
                "Cannot add Enterovirus A BLAST ID alias because both forms "
                f"are present in the alignment: {alignment_id!r} and {alias!r}"
            )
        aliases[alias] = position_map
    position_maps.update(aliases)
    return len(aliases)


def make_windows(
    non_gap_proportion: np.ndarray,
    threshold: float,
    window_size: int,
    keep_partial: bool,
) -> list[np.ndarray]:
    # This is deliberately strict: a column equal to 90% is excluded when the
    # threshold is 0.9, matching the requested >90% rule.
    retained_positions = np.flatnonzero(non_gap_proportion > threshold)
    if not len(retained_positions):
        raise ValueError(
            f"No alignment positions are present in more than {threshold:.1%} of strains"
        )
    windows = []
    for offset in range(0, len(retained_positions), window_size):
        positions = retained_positions[offset:offset + window_size]
        if len(positions) < window_size and not keep_partial:
            break
        windows.append(positions)
    if not windows:
        raise ValueError(
            f"Only {len(retained_positions)} positions passed the presence filter, "
            f"which is fewer than one {window_size}-position window"
        )
    return windows


def summarize_windows(
    windows: list[np.ndarray],
    site_pi: np.ndarray,
    counts_by_label: list[tuple[str, np.ndarray, np.ndarray]],
    pi_column: str,
) -> pd.DataFrame:
    rows = []
    retained_offset = 0
    for window_index, positions in enumerate(windows, start=1):
        mean_pi = float(np.nanmean(site_pi[positions]))
        median_pi = float(np.nanmedian(site_pi[positions]))
        for label, counts, distinct_counts in counts_by_label:
            rows.append(
                {
                    "window": window_index,
                    "probe_set": label,
                    "retained_position_start": retained_offset + 1,
                    "retained_position_end": retained_offset + len(positions),
                    "alignment_start": int(positions[0]) + 1,
                    "alignment_end": int(positions[-1]) + 1,
                    "positions_in_window": len(positions),
                    "mean_pi": mean_pi,
                    "median_pi": median_pi,
                    "mean_unique_probe_count": float(np.mean(counts[positions])),
                    "median_unique_probe_count": float(np.median(counts[positions])),
                    "distinct_probe_count": int(distinct_counts[window_index - 1]),
                }
            )
        retained_offset += len(positions)
    data = pd.DataFrame(rows)
    return data.dropna(subset=[pi_column, "mean_unique_probe_count"]).reset_index(drop=True)


def count_distinct_probes_by_window(
    windows: list[np.ndarray],
    probe_positions: dict[str, set[int]],
    alignment_length: int,
) -> np.ndarray:
    """Count each probe ID once in every retained-position window it overlaps."""
    position_to_window = np.full(alignment_length, -1, dtype=np.int32)
    for window_index, positions in enumerate(windows):
        position_to_window[positions] = window_index

    window_counts = np.zeros(len(windows), dtype=np.int32)
    for positions in probe_positions.values():
        if not positions:
            continue
        alignment_positions = np.fromiter(positions, dtype=np.int32)
        overlapping_windows = np.unique(position_to_window[alignment_positions])
        overlapping_windows = overlapping_windows[overlapping_windows >= 0]
        window_counts[overlapping_windows] += 1
    return window_counts


def make_relative_to_dampa(
    data: pd.DataFrame, labels: list[str]
) -> tuple[pd.DataFrame, list[str], int]:
    dampa_label = next(label for label in labels if "dampa" in label.lower())
    comparison_labels = [label for label in labels if label != dampa_label]
    denominator = data.loc[
        data["probe_set"] == dampa_label,
        ["window", "mean_unique_probe_count"],
    ].rename(columns={"mean_unique_probe_count": "dampa_mean_unique_probe_count"})
    relative = data.loc[data["probe_set"].isin(comparison_labels)].merge(
        denominator, on="window", how="left", validate="many_to_one"
    )
    invalid = (
        ~np.isfinite(relative["dampa_mean_unique_probe_count"])
        | relative["dampa_mean_unique_probe_count"].le(0)
    )
    relative["mean_unique_probe_count_relative_to_dampa"] = (
        relative["mean_unique_probe_count"]
        / relative["dampa_mean_unique_probe_count"].mask(invalid)
    )
    return relative, comparison_labels, int(invalid.sum())


def regression_summary(
    data: pd.DataFrame, value_column: str, pi_column: str
) -> pd.DataFrame:
    rows = []
    for label, subset in data.groupby("probe_set", sort=False):
        subset = subset.dropna(subset=[pi_column, value_column])
        x = subset[pi_column].to_numpy(dtype=float)
        y = subset[value_column].to_numpy(dtype=float)
        if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            correlation = float(np.corrcoef(x, y)[0, 1])
            r_squared = correlation ** 2
        else:
            slope = intercept = correlation = r_squared = np.nan
        rows.append(
            {
                "probe_set": label,
                "pi_summary": pi_column.removesuffix("_pi"),
                "response_variable": value_column,
                "windows": len(subset),
                "slope": slope,
                "intercept": intercept,
                "pearson_r": correlation,
                "r_squared": r_squared,
            }
        )
    return pd.DataFrame(rows)


def save_plot(
    data: pd.DataFrame,
    labels: list[str],
    output_file: str,
    window_size: int,
    presence_threshold: float,
    point_size: float,
    point_alpha: float,
    relative_to_dampa: bool,
    pi_summary: str,
) -> Path:
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = distinct_program_colors(labels)
    value_column = (
        "mean_unique_probe_count_relative_to_dampa"
        if relative_to_dampa
        else "mean_unique_probe_count"
    )
    pi_column = f"{pi_summary}_pi"
    for label, color in zip(labels, colors):
        subset = data[data["probe_set"] == label]
        ax.scatter(
            subset[pi_column],
            subset[value_column],
            label=label,
            color=color,
            s=point_size,
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_xlabel(
        f"{pi_summary.capitalize()} nucleotide diversity (π) per "
        f"{window_size}-position block"
    )
    if relative_to_dampa:
        ax.axhline(
            1.0,
            color=program_color("DAMPA"),
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="DAMPA",
        )
        ax.set_ylabel("Relative composition bias")
    else:
        ax.set_ylabel("Unique probe count")
    ax.set_title(
        f"Nucleotide diversity versus unique-probe count\n"
        f"positions present in >{presence_threshold:.0%} of strains"
    )
    ax.legend(title="Probe set", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    reference_id = first_alignment_id(args.alignment_file)
    position_maps, non_gap_proportion, alignment_length = load_alignment(
        args.alignment_file, reference_id
    )
    alias_count = add_enterovirus_a_blast_aliases(position_maps)
    if alias_count:
        print(
            f"Added {alias_count:,} Enterovirus A BLAST-ID aliases "
            "(enterovirus-A~... -> enterovirus~A~...)",
            file=sys.stderr,
        )
    site_pi = calculate_site_pi(args.alignment_file)
    if len(site_pi) != alignment_length:
        raise ValueError("Internal error: pi and alignment lengths differ")
    windows = make_windows(
        non_gap_proportion,
        args.min_non_gap_proportion,
        args.window_size,
        args.keep_partial_window,
    )
    retained_count = sum(len(window) for window in windows)
    print(
        f"Created {len(windows):,} non-overlapping windows from "
        f"{retained_count:,} retained alignment positions",
        file=sys.stderr,
    )

    counts_by_label = []
    for blast_file, label in zip(args.blast_file, args.blast_labels):
        print(f"Processing {label}: {blast_file}", file=sys.stderr)
        counts, probe_positions, stats = unique_probe_counts_with_positions(
            blast_file,
            position_maps,
            alignment_length,
            args.min_pident,
            args.min_hit_length,
            args.chunk_size,
        )
        distinct_counts = count_distinct_probes_by_window(
            windows, probe_positions, alignment_length
        )
        counts_by_label.append((label, counts, distinct_counts))
        print(
            f"  {stats['passing']:,}/{stats['rows']:,} hits passed; "
            f"{stats['unique_probes']:,} distinct probes; "
            f"{stats['unmatched_genome']:,} hits had no aligned genome",
            file=sys.stderr,
        )

    pi_column = f"{args.pi_summary}_pi"
    data = summarize_windows(windows, site_pi, counts_by_label, pi_column)
    if data.empty:
        raise ValueError("No windows had calculable nucleotide diversity and probe counts")
    plot_labels = args.blast_labels
    value_column = "mean_unique_probe_count"
    if args.relative_to_dampa:
        data, plot_labels, omitted = make_relative_to_dampa(data, args.blast_labels)
        value_column = "mean_unique_probe_count_relative_to_dampa"
        if omitted:
            print(
                f"{omitted:,} comparison windows are undefined because the "
                "DAMPA mean probe count was zero or unavailable",
                file=sys.stderr,
            )
    output_path = save_plot(
        data,
        plot_labels,
        args.output_file,
        args.window_size,
        args.min_non_gap_proportion,
        args.point_size,
        args.point_alpha,
        args.relative_to_dampa,
        args.pi_summary,
    )
    data_path = output_path.with_name(f"{output_path.stem}_data.csv")
    summary_path = output_path.with_name(f"{output_path.stem}_regression_summary.csv")
    data.to_csv(data_path, index=False)
    regression_summary(data, value_column, pi_column).to_csv(summary_path, index=False)
    print(f"Saved scatterplot: {output_path}")
    print(f"Saved plotted data: {data_path}")
    print(f"Saved regression summary: {summary_path}")


if __name__ == "__main__":
    main()
