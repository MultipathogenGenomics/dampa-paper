#!/usr/bin/env python3
r"""Plot gene annotations above the fraction of aligned genomes covered by probes.

Adapted from get_position_maps() and get_proportion_of_genomes_covered() in
probe_plot_onto_alignment.py. This focused script creates one two-track figure
and its plotted-data CSV, with no local paths, cached results or other panels.
Dependencies: Python 3.10+, numpy, pandas, matplotlib and biopython.

Inputs
------
--alignment-file: nucleotide FASTA multiple-sequence alignment, with equal
sequence lengths and unique IDs. '-' denotes gaps. The annotation reference
selected by --strain-to-copy-annots must be an exact alignment record ID.

--blast-file: one or more headerless, tab-separated BLAST files, each for one
probe set, paired with --blast-labels in the same order. Required column order:
qseqid sseqid pident qstart qend sstart send evalue bitscore
Use BLAST -outfmt "6 qseqid sseqid pident qstart qend sstart send evalue bitscore",
not the standard twelve-column outfmt 6. qseqid identifies a probe, sseqid is
an exact alignment genome ID. Subject coordinates are 1-based inclusive on
the UNGAPPED subject sequence; reverse hits are accepted. Unknown subjects
are reported and excluded. Matching hits outside subject bounds are rejected.

--gff-file: GFF3 annotations, with 1-based inclusive coordinates on the
ungapped annotation reference. Feature types and ordered label attributes
are configurable. A multi-sequence GFF requires --gff-seqid when the selected
feature types occur on more than one sequence. No fixed HIV/enterovirus locus
overrides are applied: supply a GFF matching the chosen reference sequence.

Calculations
------------
At each alignment column, the numerator is the number of DISTINCT subject
genomes with at least one qualifying probe hit. The denominator is ALL genomes
in the alignment, including genomes with a gap or no hit at that column.
Repeated probes, overlapping hits and reverse hits do not double-count a
genome. A gap has no corresponding subject base and is never covered.

By default no identity/length filtering is added, matching the original
get_proportion_of_genomes_covered() helper. Use --min-pident 80 and
--min-hit-length 81 to apply the other source probe panels' usual thresholds.
Missing hits imply zero coverage. This is predicted BLAST coverage, not
experimental read depth or a fraction conditional on non-gap occupancy.

Columns with non-gap occupancy >= --min-non-gap-proportion (default 0.2) are
retained. The x-axis numbers these retained columns in alignment order, starting
at 1. Annotations are projected through the ungapped reference into this same
coordinate system. Coverage is smoothed with a centred rolling mean over
retained columns (default 120), allowing partial windows at the edges. Set
--window-size 1 for unsmoothed proportions. CSVs retain raw counts/proportions
and original alignment positions as well as the smoothed plotted proportions.

Outputs
-------
--output-file sets the figure format (.pdf, .png, .svg, etc.). A sibling
<stem>_data.csv records one row per retained column and probe set. Existing
outputs are overwritten. PDF text stays editable. Optional --title sets the
figure title. The annotation track uses automatic lanes for overlapping genes.

Example::

    python figure_scripts/supp_fig1_probe_plot_onto_alignment.py \
        --alignment-file data/genomes.fasta \
        --blast-file data/reference_probes.tsv --blast-labels Reference \
        --gff-file data/reference.gff3 --strain-to-copy-annots reference_id \
        --annotation-feature-types mature_protein_region_of_CDS five_prime_UTR \
        --annotation-name-prefixes product Name \
        --output-file output/supp_fig1.pdf
"""

import argparse
import csv
from pathlib import Path
from urllib.parse import unquote
import warnings

from Bio import AlignIO
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


DEFAULT_FEATURE_TYPES = (
    "gene", "CDS", "mature_protein_region_of_CDS", "five_prime_UTR", "three_prime_UTR"
)


def load_alignment(path, reference_id, min_occupancy):
    """Return full alignment, ungapped-to-aligned maps, and retained columns."""
    alignment = AlignIO.read(path, "fasta")
    ids = [record.id for record in alignment]
    if len(set(ids)) != len(ids):
        raise ValueError("Alignment record IDs must be unique")
    if reference_id not in ids:
        raise ValueError(f"Annotation reference {reference_id!r} is absent from alignment")
    sequences = np.array([list(str(record.seq)) for record in alignment])
    occupancy = (sequences != "-").mean(axis=0)
    keep = np.flatnonzero(occupancy >= min_occupancy)
    if not len(keep):
        raise ValueError("No alignment columns pass the occupancy threshold")
    maps = {name: np.flatnonzero(row != "-") for name, row in zip(ids, sequences)}
    return alignment, maps, keep, occupancy


def genome_coverage(path, position_maps, alignment_length, min_pident=0, min_hit_length=1):
    """Union subject intervals before counting each genome once per column."""
    intervals = {}
    stats = {"rows": 0, "qualifying_hits": 0, "unmatched_hits": 0}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 9:
                raise ValueError(f"{path}:{line_number}: expected nine BLAST columns")
            stats["rows"] += 1
            try:
                identity = float(row[2])
                start, end = sorted((int(row[5]), int(row[6])))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid identity or subject coordinates") from error
            if not np.isfinite(identity) or not 0 <= identity <= 100 or start < 1:
                raise ValueError(f"{path}:{line_number}: invalid identity or subject coordinates")
            if identity < min_pident or end - start + 1 < min_hit_length:
                continue
            subject = row[1]
            if subject not in position_maps:
                stats["unmatched_hits"] += 1
                continue
            if end > len(position_maps[subject]):
                raise ValueError(f"{path}:{line_number}: hit extends beyond ungapped subject {subject!r}")
            intervals.setdefault(subject, []).append((start - 1, end))
            stats["qualifying_hits"] += 1

    counts = np.zeros(alignment_length, dtype=np.int64)
    for subject, hits in intervals.items():
        merged = []
        for start, end in sorted(hits):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in merged:
            counts[position_maps[subject][start:end]] += 1
    if stats["unmatched_hits"]:
        if not intervals:
            raise ValueError(f"No qualifying BLAST subjects in {path} match the alignment IDs")
        warnings.warn(f"{path}: excluded {stats['unmatched_hits']} hits with unknown subject IDs")
    return counts, stats


def project_annotations(path, reference_map, keep, feature_types, name_prefixes, seqid=None):
    """Read GFF and map 1-based reference intervals into retained columns."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"{path}:{line_number}: expected nine GFF3 columns")
            if fields[2] not in feature_types or (seqid is not None and fields[0] != seqid):
                continue
            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid GFF coordinates") from error
            if not 1 <= start <= end <= len(reference_map):
                raise ValueError(f"{path}:{line_number}: annotation outside the ungapped reference")
            attributes = dict(item.split("=", 1) for item in fields[8].split(";") if "=" in item)
            name = next((unquote(attributes[prefix.rstrip("=")])
                         for prefix in name_prefixes if attributes.get(prefix.rstrip("="))), fields[2])
            records.append((fields[0], start, end, name))
    if len({record[0] for record in records}) > 1:
        raise ValueError("Selected GFF features occur on multiple sequences; specify --gff-seqid")
    annotations = set()
    for _, start, end, name in records:
        retained = np.intersect1d(reference_map[start - 1:end], keep, assume_unique=True)
        if len(retained):
            left = int(np.searchsorted(keep, retained[0])) + 1
            right = int(np.searchsorted(keep, retained[-1])) + 1
            annotations.add((left, right, name))
    if not annotations:
        raise ValueError("No selected annotations overlap retained alignment columns")
    return sorted(annotations)


def draw_annotations(axis, annotations, font_size):
    """Draw coloured intervals, assigning lanes using both box and label extents."""
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    font = mpl.font_manager.FontProperties(size=font_size)
    scale = axis.transData.transform((1, 0))[0] - axis.transData.transform((0, 0))[0]
    lanes = []
    names = list(dict.fromkeys(name for _, _, name in annotations))
    palette = plt.get_cmap("tab20")
    colours = {name: palette(i % 20) for i, name in enumerate(names)}
    for start, end, name in annotations:
        width = end - start + 1
        middle = (start + end) / 2
        text_width = renderer.get_text_width_height_descent(name, font, ismath=False)[0]
        label_width = (text_width + 6) / scale
        left = min(start - 0.5, middle - label_width / 2)
        right = max(end + 0.5, middle + label_width / 2)
        lane = next((i for i, last in enumerate(lanes) if left >= last), len(lanes))
        if lane == len(lanes):
            lanes.append(right)
        else:
            lanes[lane] = right
        outside = label_width > width
        axis.add_patch(Rectangle((start - 0.5, lane), width, 0.6,
                                 facecolor=colours[name], edgecolor="#333333", linewidth=0.6))
        # Keep external labels inside the horizontal figure bounds.
        xmin, xmax = axis.get_xlim()
        label_x = np.clip(middle, xmin + label_width / 2, xmax - label_width / 2)
        axis.text(label_x, lane + (0.65 if outside else 0.3), name,
                  ha="center", va="bottom" if outside else "center", fontsize=font_size)
    axis.set_ylim(-0.08, max(1, len(lanes)))
    axis.set_axis_off()
    return len(lanes)


def save_plot(data, annotations, args):
    """Save the two-track figure with a shared retained-alignment x-axis."""
    labels = args.blast_labels
    with mpl.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": args.font_size}):
        fig, (gene_axis, axis) = plt.subplots(
            2, 1, figsize=(args.plot_width, args.plot_height), sharex=True,
            gridspec_kw={"height_ratios": [1, 4], "hspace": 0.08},
        )
        axis.set_xlim(0.5, int(data["filtered_position"].max()) + 0.5)
        lanes = draw_annotations(gene_axis, annotations, args.font_size)
        # Increase the annotation track's share for overlapping/exon-rich loci.
        gene_axis.get_subplotspec().get_gridspec().set_height_ratios([max(1, lanes), 4])
        colours = ["#666666"] if len(labels) == 1 else plt.get_cmap("tab10").colors
        for i, label in enumerate(labels):
            values = data.loc[data["probe_set"].eq(label)]
            axis.plot(values["filtered_position"], values["smoothed_proportion"],
                      linewidth=1.1, color=colours[i % len(colours)], label=label)
        axis.set_ylim(0, 1.02)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis.set_ylabel("Proportion of genomes covered")
        axis.set_xlabel("Position in filtered alignment")
        axis.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", length=3)
        axis.legend(frameon=False, loc="best")
        if args.title:
            gene_axis.set_title(args.title, loc="left", pad=8)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.90)
        fig.savefig(args.output_file, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alignment-file", "--alignment_file", type=Path, required=True)
    parser.add_argument("--blast-file", "--blast_file", type=Path, nargs="+", required=True)
    parser.add_argument("--blast-labels", nargs="+", required=True)
    parser.add_argument("--gff-file", "--gff_file", type=Path, required=True)
    parser.add_argument("--strain-to-copy-annots", "--strain_to_copy_annots", required=True)
    parser.add_argument("--gff-seqid", help="GFF sequence ID to use when annotations contain multiple sequences.")
    parser.add_argument("--annotation-feature-types", nargs="+", default=DEFAULT_FEATURE_TYPES)
    parser.add_argument("--annotation-name-prefixes", nargs="+", default=("Name", "gene", "product", "ID"))
    parser.add_argument("--min-non-gap-proportion", type=float, default=0.2)
    parser.add_argument("--window-size", "--window_size", type=int, default=120)
    parser.add_argument("--min-pident", type=float, default=0, help="BLAST identity threshold (default: no additional filter).")
    parser.add_argument("--min-hit-length", type=int, default=1, help="Minimum inclusive subject hit length (default: 1).")
    parser.add_argument("--output-file", "--output_file", type=Path, default=Path("output/supp_fig1.pdf"))
    parser.add_argument("--title")
    parser.add_argument("--plot-width", type=float, default=7.5, help="Width in inches (default: 7.5).")
    parser.add_argument("--plot-height", type=float, default=3.2, help="Height in inches (default: 3.2).")
    parser.add_argument("--font-size", type=float, default=8, help="Font size in points (default: 8).")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if len(args.blast_file) != len(args.blast_labels) or len(set(args.blast_labels)) != len(args.blast_labels):
        parser.error("Supply one unique --blast-labels value per BLAST file")
    if not 0 <= args.min_non_gap_proportion <= 1 or not 0 <= args.min_pident <= 100:
        parser.error("Occupancy must be in [0, 1]; identity must be in [0, 100]")
    if args.window_size < 1 or args.min_hit_length < 1 or args.dpi < 1:
        parser.error("Window size, minimum hit length and DPI must be positive")
    sizes = np.array([args.plot_width, args.plot_height, args.font_size])
    if not np.isfinite(sizes).all() or (sizes <= 0).any():
        parser.error("Plot dimensions and font size must be finite and positive")
    if any(not prefix.rstrip("=") for prefix in args.annotation_name_prefixes):
        parser.error("Annotation name prefixes must not be empty")
    inputs = {p.resolve() for p in [args.alignment_file, args.gff_file, *args.blast_file]}
    data_path = args.output_file.with_name(f"{args.output_file.stem}_data.csv")
    if args.output_file.resolve() in inputs or data_path.resolve() in inputs:
        parser.error("Output files must not overwrite input files")
    return args


def main():
    args = parse_args()
    alignment, maps, keep, occupancy = load_alignment(
        args.alignment_file, args.strain_to_copy_annots, args.min_non_gap_proportion
    )
    annotations = project_annotations(
        args.gff_file, maps[args.strain_to_copy_annots], keep,
        args.annotation_feature_types, args.annotation_name_prefixes, args.gff_seqid,
    )
    frames = []
    for path, label in zip(args.blast_file, args.blast_labels):
        counts, stats = genome_coverage(path, maps, alignment.get_alignment_length(),
                                       args.min_pident, args.min_hit_length)
        proportions = counts[keep] / len(alignment)
        frames.append(pd.DataFrame({
            "probe_set": label,
            "filtered_position": np.arange(1, len(keep) + 1),
            "alignment_position": keep + 1,
            "non_gap_proportion": occupancy[keep],
            "genomes_covered": counts[keep],
            "total_genomes": len(alignment),
            "proportion_covered": proportions,
            "smoothed_proportion": pd.Series(proportions).rolling(
                args.window_size, min_periods=1, center=True
            ).mean(),
        }))
        print(f"{label}: {stats['qualifying_hits']} qualifying hits, "
              f"{stats['unmatched_hits']} unmatched hits; denominator {len(alignment)} genomes")
    data = pd.concat(frames, ignore_index=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    save_plot(data, annotations, args)
    data_path = args.output_file.with_name(f"{args.output_file.stem}_data.csv")
    data.to_csv(data_path, index=False)
    print(f"Retained {len(keep)} of {alignment.get_alignment_length()} alignment columns")
    print(f"Wrote {args.output_file} and {data_path}")


if __name__ == "__main__":
    main()
