#!/usr/bin/env python3
r"""Generate Figure 5e: enterovirus depth profiles and per-locus boxplots.

Dependencies: Python 3.9+, numpy, pandas, matplotlib, seaborn >= 0.13.

Inputs
------
* --panda-dir and --twist-dir: directories containing CASTANET sample outputs.
  Each must contain exactly one primary sample directory per selected strain:
  <strain>_S*/Depth_output/<target>-<strain>*_depth_by_pos.csv
  For example:
  POS-EV-002_S337_L001/Depth_output/
      enterovirus_a-POS-EV-002_S337_L001_depth_by_pos.csv
  The CSV has NO header or position column. Row 1 contains all-read depths;
  row 2 contains deduplicated-read depths. Columns are consecutive reference
  positions starting at position 1, including zero-depth positions. Both rows
  must contain finite non-negative numbers. Only all-read depths are plotted.
  Files shorter than the reference are rejected; trailing columns beyond the
  reference are discarded. Replicates named <strain>-01 etc. are not selected.
* --reference-fasta: single-record FASTA of the reference used for depth mapping
  (OQ091683.1 for the publication). Depths must use its coordinate system.
* --reference-gff3: annotations of the same reference, with 1-based inclusive
  coordinates. A ##sequence-region declaration is recommended; otherwise the
  largest feature end is used as the sequence length. Selected feature types
  are mature_protein_region_of_CDS, five_prime_UTR and three_prime_UTR.
  Protein labels are read from product, then Name, then ID. Labels must match
  LOCUS_PALETTE (VP4, VP2, VP3, VP1, 2A-2C and 3A-3D). Selected loci must
  be unique, non-overlapping and on the forward strand. The 3-prime UTR is
  excluded, and the plotted genome ends at the last included locus.

Calculations
------------
For each method/sample, locus means use unsmoothed all-read depths divided by
that sample's maximum across the entire reference. Profiles instead use a
centred rolling mean (120 nt by default), divided by its own maximum across
the reference. Incomplete windows at either end remain missing. Bold lines
show the position-wise median across samples. Boxplots show the interquartile
range and full-range whiskers, with mean lines and individual sample points;
the boxplots and profiles therefore use different normalisation denominators.
Twist is labelled CVRP in figures and tables.

Outputs
-------
--output-dir receives data/ (per-sample locus means, per-method summaries and
length-weighted means combining 2A-2C and 3A-3D) and pdf/ (two locus boxplot
figures and normalized_reads_all_sm_<window>window_median.pdf, Figure 5e).
Existing files with these names are overwritten.

Example (paths are relative to the working directory)::

    python figure_scripts/figure5e_normalised_depth_plots.py \
        --panda-dir data/panda --twist-dir data/twist \
        --reference-fasta data/reference.fasta \
        --reference-gff3 data/reference.gff3 --output-dir output/figure5e
"""

import argparse
from glob import glob
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_TARGET = "enterovirus_a"
DEFAULT_STRAINS = ("POS-EV-002", "POS-EV-003", "POS-EV-004", "POS-EV-005", "POS-EV-006")
SMOOTHING_WINDOW = 120
EXCLUDED_FEATURES = {"3′ UTR"}

# Exact RGB fills used in the Figure 5 enterovirus genome structure schematic.
LOCUS_PALETTE = {
    "5′ UTR": "#3974BB",
    "VP4": "#B3C8EB",
    "VP2": "#F46E16",
    "VP3": "#F9B162",
    "VP1": "#37A53E",
    "2A": "#97D776",
    "2B": "#D82327",
    "2C": "#F68191",
    "3A": "#8F56B1",
    "3B": "#C8A8DA",
    "3C": "#874A44",
    "3D": "#C1938E",
}


def fasta_length(fasta_path):
    """Return the number of bases in a single-record FASTA file."""
    with open(fasta_path) as fasta:
        lines = [line.strip() for line in fasta if line.strip()]
    if not lines or not lines[0].startswith(">") or sum(
        line.startswith(">") for line in lines
    ) != 1:
        raise ValueError(f"Expected a single-record FASTA in {fasta_path}")
    length = sum(len(line) for line in lines if not line.startswith(">"))
    if length == 0:
        raise ValueError(f"Empty reference sequence in {fasta_path}")
    return length


def gff3_sequence_length(gff3_path):
    """Return the declared GFF3 sequence-region length and validate feature bounds."""
    sequence_length = None
    maximum_feature_end = 0
    with open(gff3_path) as gff3:
        for line in gff3:
            if line.startswith("##sequence-region"):
                _, _, start, end = line.split()
                sequence_length = int(end) - int(start) + 1
            elif not line.startswith("#"):
                fields = line.rstrip().split("\t")
                if len(fields) == 9:
                    maximum_feature_end = max(maximum_feature_end, int(fields[4]))
    if sequence_length is None:
        sequence_length = maximum_feature_end
    if maximum_feature_end > sequence_length:
        raise ValueError(f"A feature in {gff3_path} extends beyond its sequence-region")
    return sequence_length


def parse_gff3_genes_and_utrs(gff3_path):
    """Read included non-overlapping proteins and UTRs in coordinate order."""
    feature_names = {
        "five_prime_UTR": "5′ UTR",
        "three_prime_UTR": "3′ UTR",
    }
    rows = []
    with open(gff3_path) as gff3:
        for line in gff3:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] not in {
                "mature_protein_region_of_CDS",
                *feature_names,
            }:
                continue
            if fields[6] != "+":
                raise ValueError("Figure 5e requires forward-strand reference annotations")
            if int(fields[3]) < 1 or int(fields[4]) < int(fields[3]):
                raise ValueError(f"Invalid feature coordinates in {gff3_path}")
            attributes = dict(
                attribute.split("=", 1)
                for attribute in fields[8].split(";")
                if "=" in attribute
            )
            rows.append(
                {
                    "gene": feature_names.get(fields[2])
                    or attributes.get("product")
                    or attributes.get("Name")
                    or attributes.get("ID", "unnamed_feature"),
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                }
            )
    if not rows:
        raise ValueError(f"No mature-protein or UTR annotations found in {gff3_path}")
    features = pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)
    features = features.loc[
        ~features["gene"].isin(EXCLUDED_FEATURES)
    ].reset_index(drop=True)
    if features.empty:
        raise ValueError(f"No mature-protein or UTR annotations found in {gff3_path}")
    if features["gene"].duplicated().any():
        raise ValueError("Selected annotation labels must be unique")
    missing_colours = set(features["gene"]) - set(LOCUS_PALETTE)
    if missing_colours:
        raise ValueError(f"Unsupported locus labels: {sorted(missing_colours)}")
    if (
        features["start"].iloc[1:].to_numpy()
        <= features["end"].iloc[:-1].to_numpy()
    ).any():
        raise ValueError("Selected annotations overlap; feature means would double-count positions")
    return features


def find_depth_file(method_dir, strain, target):
    """Find the primary sample for a method/strain, excluding -01 etc. replicates."""
    pattern = (
        method_dir
        / f"{strain}_S*"
        / "Depth_output"
        / f"{target}-{strain}*_depth_by_pos.csv"
    )
    matches = sorted(glob(str(pattern)))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one depth file in {method_dir} for {strain}; found {matches}")
    return Path(matches[0])


def load_normalized_depth(depth_path, reference_length):
    """Read two-row CASTANET CSV, trim to reference, and normalise all-read depth.

    CSV columns map to reference positions 1..N; the returned DataFrame has
    an explicit 1-based position column. Normalisation precedes locus slicing.
    """
    depth = pd.read_csv(depth_path, header=None).T
    if depth.shape[1] != 2:
        raise ValueError(f"Expected reads_all and reads_dedup rows in {depth_path}")
    if len(depth) < reference_length:
        raise ValueError(
            f"{depth_path} has {len(depth):,} positions, shorter than the "
            f"{reference_length:,}-nt reference"
        )
    try:
        depth = depth.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric depth in {depth_path}") from error
    if not np.isfinite(depth.to_numpy()).all() or (depth.to_numpy() < 0).any():
        raise ValueError(f"Depths must be finite and non-negative in {depth_path}")
    depth.columns = ["reads_all", "reads_dedup"]
    depth = depth.iloc[:reference_length].copy()
    depth["position"] = np.arange(1, reference_length + 1)
    maximum = depth["reads_all"].max()
    if maximum <= 0:
        raise ValueError(f"Cannot normalize {depth_path}: maximum reads_all is {maximum}")
    depth["normalized_reads_all"] = depth["reads_all"] / maximum
    return depth


def summarize_normalized_depth_by_gene(
    depth_dirs, reference_fasta, reference_gff3, strains, target
):
    """Calculate mean normalized_reads_all for each included annotated locus."""
    reference_length = fasta_length(reference_fasta)
    annotation_length = gff3_sequence_length(reference_gff3)
    if annotation_length != reference_length:
        raise ValueError(
            f"FASTA length ({reference_length:,}) and GFF3 sequence-region length "
            f"({annotation_length:,}) differ"
        )
    genes = parse_gff3_genes_and_utrs(reference_gff3)
    if genes["end"].max() > reference_length:
        raise ValueError("A GFF3 annotation extends beyond the FASTA sequence")

    summaries = []
    depth_lengths = []
    for method in ("panda", "twist"):
        for strain in strains:
            depth_path = find_depth_file(depth_dirs[method], strain, target)
            raw_length = pd.read_csv(depth_path, header=None, nrows=1).shape[1]
            depth_lengths.append(raw_length)
            depth = load_normalized_depth(depth_path, reference_length)
            for gene in genes.itertuples(index=False):
                within_gene = depth["position"].between(gene.start, gene.end)
                summaries.append(
                    {
                        "method": "PANDA" if method == "panda" else "CVRP",
                        "strain": strain,
                        "gene": gene.gene,
                        "start": gene.start,
                        "end": gene.end,
                        "gene_length_nt": gene.end - gene.start + 1,
                        "mean_normalized_reads_all": depth.loc[
                            within_gene, "normalized_reads_all"
                        ].mean(),
                    }
                )

    summary = pd.DataFrame(summaries)
    output_length = max(depth_lengths)
    print(
        f"Depth output length: {min(depth_lengths):,}-{max(depth_lengths):,} positions; "
        f"FASTA length: {reference_length:,} nt; GFF3 sequence-region length: "
        f"{annotation_length:,} nt; output/reference: "
        f"{output_length / reference_length:.2%}; trailing positions excluded: "
        f"{output_length - reference_length:,}."
    )
    return summary


def combine_nonstructural_genes(summary):
    """Combine 2A-2C and 3A-3D using nucleotide-length-weighted means."""
    gene_groups = {
        "2A": "2",
        "2B": "2",
        "2C": "2",
        "3A": "3",
        "3B": "3",
        "3C": "3",
        "3D": "3",
    }
    grouped = summary.assign(
        gene=summary["gene"].replace(gene_groups),
        weighted_mean=(
            summary["mean_normalized_reads_all"] * summary["gene_length_nt"]
        ),
    )
    grouped = (
        grouped.groupby(["method", "strain", "gene"], as_index=False, sort=False)
        .agg(
            start=("start", "min"),
            end=("end", "max"),
            gene_length_nt=("gene_length_nt", "sum"),
            weighted_mean=("weighted_mean", "sum"),
        )
    )
    grouped["mean_normalized_reads_all"] = (
        grouped.pop("weighted_mean") / grouped["gene_length_nt"]
    )
    return grouped.sort_values(["method", "strain", "start"]).reset_index(drop=True)


def summarize_gene_means_by_condition(summary):
    """Summarize per-sample gene means across strains for each method condition."""
    condition_summary = (
        summary.rename(columns={"method": "condition"})
        .groupby(
            ["condition", "gene", "start", "end", "gene_length_nt"],
            as_index=False,
            sort=False,
        )
        .agg(
            n_samples=("strain", "nunique"),
            mean_normalized_reads_all=("mean_normalized_reads_all", "mean"),
            median_normalized_reads_all=("mean_normalized_reads_all", "median"),
        )
    )
    condition_order = {"PANDA": 0, "CVRP": 1}
    return (
        condition_summary.assign(
            _condition_order=condition_summary["condition"].map(condition_order)
        )
        .sort_values(["start", "_condition_order"])
        .drop(columns="_condition_order")
        .reset_index(drop=True)
    )


def plot_separated_locus_means(summary, output_path):
    """Plot paired methods using the Figure 5 colour assigned to each locus."""
    gene_order = (
        summary[["gene", "start"]]
        .drop_duplicates()
        .sort_values("start")["gene"]
        .tolist()
    )
    missing_colours = set(gene_order) - set(LOCUS_PALETTE)
    if missing_colours:
        raise ValueError(f"No genome-structure colour defined for {sorted(missing_colours)}")

    method_styles = {
        "PANDA": {"offset": -0.18, "hatch": None, "marker": "o"},
        "CVRP": {"offset": 0.18, "hatch": "///", "marker": "D"},
    }
    fig, axis = plt.subplots(figsize=(16, 7))
    for gene_position, gene in enumerate(gene_order):
        locus_colour = LOCUS_PALETTE[gene]
        for method, style in method_styles.items():
            values = (
                summary.loc[
                    summary["gene"].eq(gene) & summary["method"].eq(method),
                    ["strain", "mean_normalized_reads_all"],
                ]
                .sort_values("strain")["mean_normalized_reads_all"]
                .to_numpy()
            )
            box_position = gene_position + style["offset"]
            axis.boxplot(
                values,
                positions=[box_position],
                widths=0.28,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                whis=(0, 100),
                showmeans=True,
                meanline=True,
                boxprops={
                    "facecolor": locus_colour,
                    "edgecolor": "#333333",
                    "hatch": style["hatch"],
                    "linewidth": 1.2,
                },
                medianprops={"visible": False},
                meanprops={"color": "#222222", "linewidth": 1.5, "linestyle": "-"},
                whiskerprops={"color": "#555555", "linewidth": 1.0},
                capprops={"color": "#555555", "linewidth": 1.0},
            )
            jitter = np.linspace(-0.035, 0.035, len(values))
            axis.scatter(
                box_position + jitter,
                values,
                color=locus_colour,
                marker=style["marker"],
                s=32,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    axis.legend(
        handles=[
            Patch(facecolor="#BDBDBD", edgecolor="#333333", label="PANDA"),
            Patch(
                facecolor="#BDBDBD",
                edgecolor="#333333",
                hatch="///",
                label="CVRP",
            ),
        ],
        title="Method",
        loc="upper right",
    )
    axis.set_xticks(range(len(gene_order)), labels=gene_order)
    for tick_label, gene in zip(axis.get_xticklabels(), gene_order):
        tick_label.set_color(LOCUS_PALETTE[gene])
        tick_label.set_fontweight("bold")
    axis.set_xlabel("Annotated gene or 5′ UTR (5′ to 3′)")
    axis.set_ylabel("Mean normalized_reads_all")
    axis.set_ylim(0, 1)
    axis.set_title(
        "Normalized all-read depth by enterovirus annotation\n"
        "Boxes show IQR; whiskers show full range; lines show means; points show samples"
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gene_means(summary, output_path, title):
    """Plot paired method distributions and sample values in GFF3 feature order."""
    gene_order = (
        summary[["gene", "start"]]
        .drop_duplicates()
        .sort_values("start")["gene"]
        .tolist()
    )
    method_order = ["PANDA", "CVRP"]
    palette = {"PANDA": "#a50f15", "CVRP": "#08519c"}
    fig, axis = plt.subplots(figsize=(16, 7))
    sns.boxplot(
        data=summary,
        x="gene",
        y="mean_normalized_reads_all",
        hue="method",
        order=gene_order,
        hue_order=method_order,
        palette=palette,
        width=0.65,
        gap=0.12,
        showfliers=False,
        whis=(0, 100),
        showmeans=True,
        meanline=True,
        medianprops={"visible": False},
        meanprops={"color": "#222222", "linewidth": 1.5, "linestyle": "-"},
        ax=axis,
    )
    sns.stripplot(
        data=summary,
        x="gene",
        y="mean_normalized_reads_all",
        hue="method",
        order=gene_order,
        hue_order=method_order,
        palette=palette,
        dodge=True,
        jitter=0.14,
        size=5,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        ax=axis,
    )
    handles, labels = axis.get_legend_handles_labels()
    legend_handles = {label: handle for handle, label in zip(handles, labels)}
    axis.legend(
        [legend_handles[method] for method in method_order],
        method_order,
        title="Method",
        loc="upper right",
    )
    axis.set_xlabel("Annotated gene or 5′ UTR (5′ to 3′)")
    axis.set_ylabel("Mean normalized_reads_all")
    axis.set_ylim(0, 1)
    axis.set_title(
        f"{title}\n"
        "Boxes show IQR; whiskers show full range; lines show means; points show samples",
        fontsize=14,
        pad=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def add_gene_centered_boxplots(axis, summary, genes):
    """Add paired method-coloured boxplots centred on genomic loci."""
    method_styles = {
        "PANDA": {
            "colour": "#576CB8",
            "direction": -1,
            "hatch": None,
            "marker": "o",
        },
        "CVRP": {
            "colour": "#BD4040",
            "direction": 1,
            "hatch": None,
            "marker": "D",
        },
    }
    locus_boundaries = sorted(
        set(genes["start"].astype(int)).union({int(genes["end"].max())})
    )
    for boundary in locus_boundaries:
        axis.axvline(
            boundary,
            color="#BDBDBD",
            linewidth=0.5,
            zorder=0,
        )
    centres = ((genes["start"] + genes["end"]) / 2).to_numpy(dtype=float)
    for gene_number, gene in enumerate(genes.itertuples(index=False)):
        neighbour_distances = []
        if gene_number > 0:
            neighbour_distances.append(centres[gene_number] - centres[gene_number - 1])
        if gene_number + 1 < len(centres):
            neighbour_distances.append(centres[gene_number + 1] - centres[gene_number])
        if not neighbour_distances:
            neighbour_distances.append(gene.end - gene.start + 1)
        nearest_centre = min(neighbour_distances)
        gene_length = gene.end - gene.start + 1
        spacing_basis = min(nearest_centre, gene_length)
        box_width = spacing_basis * 0.26
        method_offset = spacing_basis * 0.20
        for method, style in method_styles.items():
            values = (
                summary.loc[
                    summary["gene"].eq(gene.gene)
                    & summary["method"].eq(method),
                    ["strain", "mean_normalized_reads_all"],
                ]
                .sort_values("strain")["mean_normalized_reads_all"]
                .to_numpy()
            )
            if len(values) == 0:
                raise ValueError(f"No gene-mean values found for {method}/{gene.gene}")
            box_position = centres[gene_number] + style["direction"] * method_offset
            axis.boxplot(
                values,
                positions=[box_position],
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                whis=(0, 100),
                showmeans=True,
                meanline=True,
                boxprops={
                    "facecolor": style["colour"],
                    "edgecolor": "#333333",
                    "hatch": style["hatch"],
                    "linewidth": 1.0,
                },
                medianprops={"visible": False},
                meanprops={"color": "#222222", "linewidth": 1.5, "linestyle": "-"},
                whiskerprops={"color": "#555555", "linewidth": 0.9},
                capprops={"color": "#555555", "linewidth": 0.9},
            )
            jitter = np.linspace(-0.12 * box_width, 0.12 * box_width, len(values))
            axis.scatter(
                box_position + jitter,
                values,
                color=style["colour"],
                marker=style["marker"],
                s=12,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    axis.legend(
        handles=[
            Patch(facecolor="#576CB8", edgecolor="#333333", label="PANDA"),
            Patch(
                facecolor="#BD4040",
                edgecolor="#333333",
                label="CVRP",
            ),
        ],
        title="Method",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
    )
    axis.set_ylabel("")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="both", labelsize=9)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)


def plot_smoothed_depth_profiles(
    output_path,
    summary,
    depth_dirs,
    reference_fasta,
    reference_gff3,
    strains,
    target,
    window=SMOOTHING_WINDOW,
    plot_width=7.09,
    plot_height=6.2,
):
    """Plot rolling-mean profiles, annotations and aligned locus-mean boxplots.

    Each sample profile is scaled by its maximum after smoothing. ``summary``
    supplies unsmoothed locus means scaled by the raw-depth maximum instead.
    """
    reference_length = fasta_length(reference_fasta)
    genes = parse_gff3_genes_and_utrs(reference_gff3)
    analysis_end = int(genes["end"].max())
    method_profiles = {}
    for method, method_label in (("panda", "PANDA"), ("twist", "CVRP")):
        profiles = pd.DataFrame(index=np.arange(1, analysis_end + 1))
        for strain in strains:
            depth = load_normalized_depth(
                find_depth_file(depth_dirs[method], strain, target), reference_length
            )
            smoothed_reads = depth["reads_all"].rolling(
                window=window, center=True
            ).mean()
            smoothed_maximum = smoothed_reads.max()
            if not np.isfinite(smoothed_maximum) or smoothed_maximum <= 0:
                raise ValueError(
                    f"Cannot normalize smoothed profile for {method_label}/{strain}"
                )
            profiles[strain] = (
                smoothed_reads.iloc[:analysis_end].to_numpy() / smoothed_maximum
            )
        method_profiles[method_label] = profiles

    colours = {
        "PANDA": {"sample": "lightblue", "median": "darkblue"},
        "CVRP": {"sample": "lightcoral", "median": "darkred"},
    }
    legend_fontsize = 9
    if set(summary["gene"]) != set(genes["gene"]):
        raise ValueError("Gene-mean summary and plotted GFF3 loci do not match")
    fig, (axis, gene_axis, box_axis) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(plot_width, plot_height),
        sharex=True,
        gridspec_kw={"height_ratios": [5.8, 0.55, 1.8], "hspace": 0.03},
    )
    for method_label, profiles in method_profiles.items():
        for strain in profiles.columns:
            axis.plot(
                profiles.index,
                profiles[strain],
                color=colours[method_label]["sample"],
                linewidth=0.7,
                alpha=0.8,
            )
        median_profile = profiles.median(axis=1)
        axis.plot(
            profiles.index,
            median_profile,
            color=colours[method_label]["median"],
            linewidth=2.5,
            label=f"{method_label} median profile",
            zorder=4,
        )

    for gene in genes.itertuples(index=False):
        gene_length = gene.end - gene.start + 1
        gene_axis.add_patch(
            Rectangle(
                (gene.start, 0.50),
                gene_length,
                0.38,
                facecolor=LOCUS_PALETTE[gene.gene],
                edgecolor="#222222",
                linewidth=0.8,
            )
        )
        label_inside = gene_length >= 230
        gene_axis.text(
            gene.start + gene_length / 2,
            0.69 if label_inside else 0.18,
            gene.gene,
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
        )

    add_gene_centered_boxplots(box_axis, summary, genes)

    axis.set_ylabel("")
    axis.set_xlim(1, analysis_end)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="upper right", ncols=2, fontsize=legend_fontsize)
    axis.tick_params(axis="both", labelsize=9)
    axis.tick_params(axis="x", labelbottom=False)
    gene_axis.set_xlim(1, analysis_end)
    gene_axis.set_ylim(0, 1)
    gene_axis.set_axis_off()
    box_axis.set_xlim(1, analysis_end)
    box_axis.set_xlabel("Genome position", fontsize=10)
    fig.supylabel("Normalised depth of sequencing", x=0.025, fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.98, hspace=0.03)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--panda-dir", type=Path, required=True,
                        help="Directory containing PANDA CASTANET sample directories.")
    parser.add_argument("--twist-dir", type=Path, required=True,
                        help="Directory containing Twist/CVRP CASTANET sample directories.")
    parser.add_argument("--reference-fasta", type=Path, required=True,
                        help="Single-record mapping reference FASTA.")
    parser.add_argument("--reference-gff3", type=Path, required=True,
                        help="GFF3 annotations in the mapping reference coordinates.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/figure5e"),
                        help="Output directory (default: output/figure5e).")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="Target prefix used in depth filenames (default: %(default)s).")
    parser.add_argument("--strains", nargs="+", default=DEFAULT_STRAINS,
                        help="Primary sample prefixes (default: %(default)s).")
    parser.add_argument("--window", type=int, default=SMOOTHING_WINDOW,
                        help="Centred smoothing window in nt (default: %(default)s).")
    parser.add_argument(
        "--plot-width",
        type=float,
        default=7.09,
        help="Overall three-panel plot width in inches.",
    )
    parser.add_argument(
        "--plot-height",
        type=float,
        default=6.2,
        help="Overall three-panel plot height in inches.",
    )
    args = parser.parse_args()
    if args.plot_width <= 0 or args.plot_height <= 0:
        parser.error("--plot-width and --plot-height must be greater than zero")

    if not np.isfinite([args.plot_width, args.plot_height]).all():
        parser.error("Plot dimensions must be finite")
    if args.window < 1:
        parser.error("--window must be greater than zero")
    if len(set(args.strains)) != len(args.strains):
        parser.error("--strains must not contain duplicate sample prefixes")
    for path in (args.panda_dir, args.twist_dir):
        if not path.is_dir():
            parser.error(f"Input directory does not exist: {path}")
    for path in (args.reference_fasta, args.reference_gff3):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if args.window > fasta_length(args.reference_fasta):
        parser.error("--window exceeds the reference length")

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.size"] = 10
    depth_dirs = {"panda": args.panda_dir, "twist": args.twist_dir}
    table_output_dir = args.output_dir / "data"
    plot_output_dir = args.output_dir / "pdf"
    table_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_normalized_depth_by_gene(
        depth_dirs, args.reference_fasta, args.reference_gff3, args.strains, args.target
    )
    csv_path = table_output_dir / "normalized_reads_all_gene_means.csv"
    plot_path = plot_output_dir / "normalized_reads_all_gene_means.pdf"
    summary.to_csv(csv_path, index=False)
    plot_separated_locus_means(summary, plot_path)

    condition_summary = summarize_gene_means_by_condition(summary)
    condition_csv_path = (
        table_output_dir / "normalized_reads_all_gene_condition_summary.csv"
    )
    condition_summary.to_csv(condition_csv_path, index=False)

    grouped_summary = combine_nonstructural_genes(summary)
    grouped_csv_path = table_output_dir / "normalized_reads_all_grouped_gene_means.csv"
    grouped_plot_path = plot_output_dir / "normalized_reads_all_grouped_gene_means.pdf"
    grouped_summary.to_csv(grouped_csv_path, index=False)
    plot_gene_means(
        grouped_summary,
        grouped_plot_path,
        title="Normalized all-read depth with genes 2 and 3 combined",
    )
    line_plot_path = (
        plot_output_dir
        / f"normalized_reads_all_sm_{args.window}window_median.pdf"
    )
    plot_smoothed_depth_profiles(
        line_plot_path,
        summary,
        depth_dirs,
        args.reference_fasta,
        args.reference_gff3,
        args.strains,
        args.target,
        window=args.window,
        plot_width=args.plot_width,
        plot_height=args.plot_height,
    )
    print(f"Wrote {len(summary):,} gene summaries to {csv_path}")
    print(
        f"Wrote {len(condition_summary):,} gene-condition summaries to "
        f"{condition_csv_path}"
    )
    print(f"Wrote gene-level plot to {plot_path}")
    print(f"Wrote {len(grouped_summary):,} grouped summaries to {grouped_csv_path}")
    print(f"Wrote grouped gene-level plot to {grouped_plot_path}")
    print(f"Wrote smoothed sample and median line plot to {line_plot_path}")


if __name__ == "__main__":
    main()
