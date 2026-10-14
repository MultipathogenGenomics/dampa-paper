#!/usr/bin/env python3
r"""Plot genome annotations, nucleotide diversity, and probe coverage.

This is a focused companion to ``probe_plot_onto_alignment.py``. It retains
that script's BLAST-to-alignment coordinate mapping, but produces only four
aligned panels:

1. genome annotations named from an ordered list of GFF attribute prefixes;
2. nucleotide diversity (pi);
3. unique-probe coverage for every input program; and
4. non-DAMPA probe coverage divided by DAMPA coverage.

Inputs and outputs
------------------
Python 3.10+, numpy, pandas, matplotlib, seaborn, scipy, biopython and
scikit-allel (including dependencies imported by the shared helper). Keep
probe_plot_onto_alignment.py and plot_program_colors.py alongside this script.

BLAST inputs are headerless nine-column TSVs in this exact order:
qseqid sseqid pident qstart qend sstart send evalue bitscore (tab-separated),
corresponding to BLAST -outfmt "6 qseqid sseqid pident qstart qend sstart send
evalue bitscore". qseqid identifies a probe; sseqid identifies an ungapped
subject genome whose ID matches a FASTA alignment record. Coordinates are
1-based inclusive and may be reversed. Do not use the standard twelve-column
outfmt 6 layout. Each file represents one probe set, paired with one label.
The alignment is a nucleotide FASTA MSA with unique IDs and equal row lengths;
BLAST subjects must use the ungapped sequences in that alignment.

--gff-file is a nine-column GFF3 in the ungapped coordinates of the alignment
record selected by --strain-to-copy-annots; GFF coordinates are 1-based
inclusive. --annotation-feature-types and --annotation-name-prefixes control
feature selection and labels. The publication's HXB2 reference ID uses the
explicit HXB2_ANNOTATIONS table instead of GFF features; other IDs use GFF.
Legacy Enterovirus A ID aliases are supported by the existing mapping helper.

The DAMPA-labelled probe set supplies the denominator for relative coverage;
zero denominators yield undefined values. Optional pickle caches use this
script's/source helper's Python objects and must come from trusted sources.
Cache checks include source timestamps, array lengths and coordinate mappings. Normal usage
requires only FASTA, BLAST and GFF, with no caches. --output-file saves the
four-panel figure; --save-pickles additionally exports reusable intermediates.
Existing outputs are overwritten.

Example:
    python figure_scripts/figure4d_plot_annotations_pi_probe_coverage.py \
        --alignment-file data/genomes.fasta --gff-file data/reference.gff3 \
        --strain-to-copy-annots reference --blast-file data/probes.tsv \
        --blast-labels DAMPA --output-file output/figure4d.pdf
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import allel
from Bio import SeqIO

from plot_program_colors import distinct_program_colors
from probe_plot_onto_alignment import (
    get_non_gap_proportion,
    get_position_maps,
    get_probedepth_for_plot,
    read_fasta_to_haplotype_array,
)


mpl.rcParams["pdf.fonttype"] = 42

A4_WIDTH_INCHES = 8.27
A4_HEIGHT_INCHES = 11.69
FONT_SIZE = 7.0
MIN_GENOME_PRESENCE = 0.90
HXB2_REFERENCE_ID = "B.FR.83.HXB2_LAI_IIIB_BRU.K03455"
HXB2_ANNOTATIONS = [
    (0, 634, "5'UTR"),
    (790, 2292, "gag"),
    (2085, 5096, "pol"),
    (5041, 5619, "vif"),
    (5559, 5850, "vpr"),
    (5831, 6045, "tat(1)"),
    (8379, 8424, "tat(2)"),
    (5970, 6045, "rev(1)"),
    (8379, 8653, "rev(2)"),
    (6062, 6310, "vpu"),
    (6225, 7758, "env-gp120"),
    (7759, 8795, "env-gp41"),
    (8797, 9417, "nef"),
    (9086, 9718, "3'UTR"),
]


def program_rank(label: str) -> int:
    """Order programs as DAMPA, CATCH, Probetools, then other inputs."""
    normalized = str(label).lower().replace("-", "_").replace(" ", "_")
    if "dampa" in normalized:
        return 0
    if "catch" in normalized:
        return 1
    if "probetools" in normalized:
        return 2
    return 3


def add_enterovirus_a_blast_aliases(
    position_maps: dict[str, dict[int, int]],
    alignment_ids: set[str],
) -> int:
    """Alias legacy alignment IDs to the spelling used in BLAST hits.

    For example, the alignment ID
    ``enterovirus-A~coxsackievirus_ON730871.1`` is also made available as
    ``enterovirus~A~coxsackievirus_ON730871.1``. Existing compatible aliases
    loaded from a position-map pickle are retained.
    """
    alignment_prefix = "enterovirus-A~"
    blast_prefix = "enterovirus~A~"
    aliases: dict[str, dict[int, int]] = {}
    for alignment_id in alignment_ids:
        if not alignment_id.startswith(alignment_prefix):
            continue
        alias = blast_prefix + alignment_id[len(alignment_prefix):]
        if alias in alignment_ids:
            raise ValueError(
                "Cannot add Enterovirus A BLAST ID alias because both forms "
                f"are present in the alignment: {alignment_id!r} and {alias!r}"
            )
        if alias in position_maps:
            if position_maps[alias] != position_maps[alignment_id]:
                raise ValueError(
                    f"Cached Enterovirus A alias {alias!r} does not match "
                    f"alignment ID {alignment_id!r}"
                )
            continue
        aliases[alias] = position_maps[alignment_id]
    position_maps.update(aliases)
    return len(aliases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot GFF annotations, nucleotide diversity, unique-probe coverage, "
            "and probe coverage relative to DAMPA along an alignment."
        )
    )
    parser.add_argument(
        "--blast_file",
        "--blast-file",
        dest="blast_file",
        nargs="+",
        required=True,
        help="One or more tab-separated BLAST result files.",
    )
    parser.add_argument(
        "--blast-labels",
        nargs="+",
        required=True,
        help="Display labels corresponding to --blast_file.",
    )
    parser.add_argument(
        "--dampa-reference-label",
        "--dampa_reference_label",
        dest="dampa_reference_label",
        help=(
            "DAMPA label to use as the denominator in the relative-coverage "
            "panel. When omitted, the first label containing 'DAMPA' is used."
        ),
    )
    parser.add_argument(
        "--alignment_file",
        "--alignment-file",
        dest="alignment_file",
        required=True,
        help="Multiple-sequence alignment in FASTA format.",
    )
    parser.add_argument(
        "--gff_file",
        "--gff-file",
        dest="gff_file",
        required=True,
        help="GFF3 annotation file.",
    )
    parser.add_argument(
        "--strain_to_copy_annots",
        "--strain-to-copy-annots",
        dest="strain_to_copy_annots",
        required=True,
        help="Alignment record whose ungapped coordinates match the GFF.",
    )
    parser.add_argument(
        "--output_file",
        "--output-file",
        dest="output_file",
        default="annotations_pi_probe_coverage.pdf",
        help="Output plot path (default: annotations_pi_probe_coverage.pdf).",
    )
    parser.add_argument(
        "--window_size",
        "--window-size",
        dest="window_size",
        type=int,
        default=120,
        help="Centered smoothing window for unique-probe coverage (default: 120).",
    )
    parser.add_argument(
        "--pi-window-size",
        type=int,
        default=120,
        help=(
            "Step-one sliding-window size in alignment columns used to "
            "calculate pi (default: 120)."
        ),
    )
    parser.add_argument(
        "--x-tick-interval",
        type=int,
        default=1000,
        help=(
            "Interval between x-axis ticks in retained alignment positions "
            "(default: 1000)."
        ),
    )
    parser.add_argument(
        "--min-presence",
        type=float,
        default=MIN_GENOME_PRESENCE,
        help=(
            "Retain columns whose fraction of genomes with a base is strictly "
            "greater than this value; must be at least 0 and less than 1 "
            "(default: 0.9)."
        ),
    )
    parser.add_argument(
        "--annotation-feature-types",
        nargs="+",
        default=[
            "gene",
            "CDS",
            "mature_protein_region_of_CDS",
            "five_prime_UTR",
            "three_prime_UTR",
        ],
        help="GFF feature types eligible for plotting.",
    )
    parser.add_argument(
        "--annotation-name-prefixes",
        nargs="+",
        default=["Name=", "gene=", "product="],
        help=(
            "Ordered GFF attribute prefixes used to obtain locus labels "
            "(default: Name= gene= product=). If none match, GFF column 3 is used."
        ),
    )
    parser.add_argument(
        "--figure-width",
        dest="figure_width",
        type=float,
        default=A4_WIDTH_INCHES,
        help="Figure width in inches, capped at A4 width (default: 8.27).",
    )
    parser.add_argument(
        "--figure-height",
        "--plot-height",
        dest="figure_height",
        type=float,
        default=7.2,
        help="Minimum figure height in inches, capped at A4 height (default: 7.2).",
    )
    parser.add_argument(
        "--text-size",
        type=float,
        default=FONT_SIZE,
        help=(
            "Font size for all plot text; only 7 is supported for the compact "
            "A4 layout (default: 7)."
        ),
    )
    parser.add_argument(
        "--annotation-font-size",
        type=float,
        default=None,
        help="Deprecated compatibility option; if supplied, it must be 7.",
    )
    parser.add_argument(
        "--non-gap-proportion-pickle",
        help="Optional compatible non-gap-proportion pickle from probe_plot_onto_alignment.py.",
    )
    parser.add_argument(
        "--position-maps-pickle",
        help="Optional compatible position-maps pickle from probe_plot_onto_alignment.py.",
    )
    parser.add_argument(
        "--pi-series-pickle",
        "--pi-pickle",
        dest="pi_series_pickle",
        help=(
            "Optional compatible pi-series pickle calculated across retained "
            "columns after applying --min-presence."
        ),
    )
    parser.add_argument(
        "--probe-coverage-pickles",
        nargs="+",
        help=(
            "Optional unsmoothed unique-probe series pickles, one per --blast_file "
            "and in the same order."
        ),
    )
    parser.add_argument(
        "--save-pickles",
        action="store_true",
        help=(
            "Save compatible non-gap, position-map, retained-column pi, and "
            "unsmoothed probe-coverage pickles for reuse."
        ),
    )
    parser.add_argument(
        "--pickle-output-prefix",
        help=(
            "Prefix for files written by --save-pickles. The default is the "
            "complete --output_file path. Supplying this option implies --save-pickles."
        ),
    )
    args = parser.parse_args()

    if len(args.blast_file) != len(args.blast_labels):
        parser.error("--blast_file and --blast-labels must have the same length")
    if len(set(args.blast_labels)) != len(args.blast_labels):
        parser.error("--blast-labels values must be unique")
    if args.dampa_reference_label is not None:
        if args.dampa_reference_label not in args.blast_labels:
            parser.error("--dampa-reference-label must match a --blast-labels value")
        if "dampa" not in args.dampa_reference_label.lower():
            parser.error("--dampa-reference-label must contain 'DAMPA'")
    if args.window_size <= 0 or args.pi_window_size <= 0:
        parser.error("window sizes must be greater than zero")
    if args.x_tick_interval <= 0:
        parser.error("--x-tick-interval must be greater than zero")
    if not 0 <= args.min_presence < 1:
        parser.error("--min-presence must be at least 0 and less than 1")
    if not 0 < args.figure_width <= A4_WIDTH_INCHES:
        parser.error(
            f"--figure-width must be greater than zero and at most {A4_WIDTH_INCHES}"
        )
    if not 0 < args.figure_height <= A4_HEIGHT_INCHES:
        parser.error(
            f"--figure-height must be greater than zero and at most {A4_HEIGHT_INCHES}"
        )
    if args.text_size != FONT_SIZE:
        parser.error("--text-size must be 7 for the A4 layout")
    if args.annotation_font_size not in (None, FONT_SIZE):
        parser.error("--annotation-font-size must be 7 when supplied")
    if any(not prefix for prefix in args.annotation_name_prefixes):
        parser.error("--annotation-name-prefixes cannot contain an empty prefix")
    if args.probe_coverage_pickles is not None and (
        len(args.probe_coverage_pickles) != len(args.blast_file)
    ):
        parser.error(
            "--probe-coverage-pickles must contain one path per --blast_file"
        )
    return args


def load_pickle(path: str, description: str):
    """Load a pickle with a contextual error message."""
    pickle_path = Path(path).expanduser()
    if not pickle_path.is_file():
        raise ValueError(f"{description} pickle does not exist: {pickle_path}")
    try:
        with pickle_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as error:
        raise ValueError(f"Could not load {description} pickle {pickle_path}: {error}") from error


def save_pickle(value, path: Path, description: str) -> None:
    """Write one reusable pickle, creating its parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        raise ValueError(f"Could not save {description} pickle {path}: {error}") from error
    print(f"Saved {description} pickle: {path}")


def safe_pickle_label(label: str) -> str:
    """Convert a display label to a stable filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "probe_set"


def require_pickle_current(
    pickle_path: str, source_paths: list[str], description: str
) -> None:
    """Reject a legacy pickle older than any source from which it was calculated."""
    path = Path(pickle_path).expanduser()
    newest_source = max(Path(source).expanduser().stat().st_mtime_ns for source in source_paths)
    if path.stat().st_mtime_ns < newest_source:
        raise ValueError(
            f"{description} pickle is older than its alignment/BLAST source: {path}. "
            "Recalculate it or provide a newer compatible pickle."
        )


def validate_numeric_series(
    value,
    expected_length: int,
    description: str,
    minimum: float | None = None,
    maximum: float | None = None,
    expected_name: str = "alignment length",
) -> pd.Series:
    """Return a one-dimensional numeric series after structural/range checks."""
    series = pd.Series(value).reset_index(drop=True)
    if len(series) != expected_length:
        raise ValueError(
            f"{description} pickle has length {len(series):,}; expected "
            f"{expected_name} {expected_length:,}"
        )
    numeric = pd.to_numeric(series, errors="coerce")
    newly_missing = numeric.isna() & ~series.isna()
    if newly_missing.any():
        raise ValueError(f"{description} pickle contains non-numeric values")
    finite = numeric[np.isfinite(numeric)]
    if minimum is not None and (finite < minimum).any():
        raise ValueError(f"{description} pickle contains values below {minimum}")
    if maximum is not None and (finite > maximum).any():
        raise ValueError(f"{description} pickle contains values above {maximum}")
    return numeric.astype(float)


def validate_position_maps(
    value,
    alignment,
    retained_positions: list[int],
) -> dict[str, dict[int, int]]:
    """Check that legacy position maps match alignment IDs and retained columns."""
    if not isinstance(value, dict):
        raise ValueError("Position-maps pickle must contain a dictionary")
    alignment_ids = [record.id for record in alignment]
    missing_ids = [record_id for record_id in alignment_ids if record_id not in value]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(f"Position-maps pickle is missing alignment IDs: {preview}")

    retained_set = set(retained_positions)
    alignment_length = len(alignment[0].seq)
    for record in alignment:
        position_map = value[record.id]
        if not isinstance(position_map, dict):
            raise ValueError(f"Position map for {record.id!r} is not a dictionary")
        if any(
            not isinstance(genome_pos, (int, np.integer))
            or not isinstance(alignment_pos, (int, np.integer))
            for genome_pos, alignment_pos in position_map.items()
        ):
            raise ValueError(f"Position map for {record.id!r} contains non-integer coordinates")
        if any(
            alignment_pos < 0
            or alignment_pos >= alignment_length
            or alignment_pos not in retained_set
            for alignment_pos in position_map.values()
        ):
            raise ValueError(
                f"Position map for {record.id!r} contains coordinates incompatible "
                "with the alignment or --min-presence"
            )
        expected_count = sum(
            str(record.seq[position]) != "-" for position in retained_positions
        )
        if len(position_map) != expected_count:
            raise ValueError(
                f"Position map for {record.id!r} has {len(position_map):,} retained "
                f"bases; expected {expected_count:,}. Check the alignment and "
                "--min-presence used to create the pickle."
            )
    return value


def validate_pi_window_from_name(path: str, requested_window: int) -> None:
    """Check that a pi pickle identifies the current window method and size."""
    name = Path(path).name
    if "alignment_sliding" not in name.lower():
        raise ValueError(
            "Pi pickle does not identify the alignment-coordinate sliding-window "
            f"method and may contain legacy retained-site smoothing: {path}. "
            "Recalculate it with --save-pickles."
        )
    match = re.search(r"pi(?:_series)?_window[_-]?(\d+)", name, re.IGNORECASE)
    if match and int(match.group(1)) != requested_window:
        raise ValueError(
            f"Pi pickle filename encodes window {match.group(1)}, but "
            f"--pi-window-size is {requested_window}: {path}"
        )


def annotation_name_from_attributes(
    attribute_text: str,
    prefixes: list[str],
    fallback: str,
) -> str:
    """Return the first attribute matching the user-specified prefix order."""
    attribute_items = [item.strip() for item in attribute_text.split(";")]
    for requested_prefix in prefixes:
        normalized_prefix = (
            requested_prefix if requested_prefix.endswith("=") else requested_prefix + "="
        )
        for item in attribute_items:
            if item.startswith(normalized_prefix):
                value = item[len(normalized_prefix):].strip()
                if value:
                    return value
    return fallback


def read_named_annotations(
    gff_file: str,
    feature_types: list[str],
    name_prefixes: list[str],
) -> list[tuple[int, int, str]]:
    """Read 0-based inclusive annotation intervals with ordered label fallbacks."""
    accepted_types = set(feature_types)
    annotations: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    with open(gff_file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in accepted_types:
                continue
            name = annotation_name_from_attributes(
                fields[8], name_prefixes, fallback=fields[2]
            )
            try:
                # GFF3 coordinates are 1-based and end-inclusive.
                start = int(fields[3]) - 1
                end = int(fields[4]) - 1
            except ValueError as error:
                raise ValueError(
                    f"Invalid coordinates on GFF line {line_number}: {line.rstrip()}"
                ) from error
            annotation = (start, end, name)
            if annotation not in seen:
                annotations.append(annotation)
                seen.add(annotation)
    if not annotations:
        raise ValueError("No annotations were found for --annotation-feature-types")
    return sorted(annotations, key=lambda item: (item[0], item[1], item[2]))


def map_annotations_to_plot(
    annotations: list[tuple[int, int, str]],
    reference_position_map: dict[int, int],
    retained_alignment_positions: list[int],
) -> list[tuple[int, int, str]]:
    """Map ungapped GFF intervals through alignment and retained-column coordinates."""
    retained_to_plot = {
        alignment_position: plot_position
        for plot_position, alignment_position in enumerate(retained_alignment_positions)
    }
    mapped_annotations: list[tuple[int, int, str]] = []
    for start, end, name in annotations:
        alignment_positions = [
            reference_position_map[position]
            for position in range(start, end + 1)
            if position in reference_position_map
            and reference_position_map[position] in retained_to_plot
        ]
        if not alignment_positions:
            print(f"Warning: annotation {name} ({start + 1}-{end + 1}) was not mapped")
            continue
        plot_positions = [retained_to_plot[position] for position in alignment_positions]
        mapped_annotations.append((min(plot_positions), max(plot_positions), name))
    if not mapped_annotations:
        raise ValueError("None of the named GFF annotations mapped to retained alignment columns")
    return mapped_annotations


def assign_annotation_lanes(
    annotations: list[tuple[int, int, str]],
) -> list[tuple[int, int, str, int]]:
    """Assign the first non-overlapping vertical lane to each annotation."""
    lane_ends: list[int] = []
    assigned: list[tuple[int, int, str, int]] = []
    for start, end, name in sorted(annotations, key=lambda item: (item[0], item[1])):
        lane = next(
            (index for index, lane_end in enumerate(lane_ends) if start > lane_end),
            len(lane_ends),
        )
        if lane == len(lane_ends):
            lane_ends.append(end)
        else:
            lane_ends[lane] = end
        assigned.append((start, end, name, lane))
    return assigned


def smooth(values: pd.Series, window_size: int) -> pd.Series:
    """Return a centered rolling mean, with window size 1 meaning no smoothing."""
    return values.rolling(window=window_size, center=True, min_periods=1).mean()


def calculate_pi_on_retained_columns(
    alignment_file: str,
    retained_positions: list[int],
    window_size: int,
) -> pd.Series:
    """Calculate pi in step-one windows spanning alignment coordinates.

    Sites failing the presence filter do not contribute to a window's mean,
    but they retain their physical width within the window. Returned values
    correspond only to the retained alignment positions used on the plot axis.
    """
    haplotypes = read_fasta_to_haplotype_array(alignment_file)
    allele_counts = haplotypes.count_alleles()
    try:
        per_site_pi = np.asarray(
            allel.mean_pairwise_difference(allele_counts), dtype=float
        )
    except Exception:
        per_site_pi = np.full(len(allele_counts), np.nan, dtype=float)

    retained = np.asarray(retained_positions, dtype=int)
    eligible = np.zeros(len(per_site_pi), dtype=bool)
    eligible[retained] = True
    per_site_pi[~eligible] = np.nan
    sliding_pi = smooth(pd.Series(per_site_pi, dtype=float), window_size)
    return sliding_pi.iloc[retained].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "legend.title_fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "figure.labelsize": FONT_SIZE,
        }
    )
    output_path = Path(args.output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alignment = list(SeqIO.parse(args.alignment_file, "fasta"))
    if not alignment:
        raise ValueError("The alignment contains no FASTA records")
    alignment_length = len(alignment[0].seq)
    if any(len(record.seq) != alignment_length for record in alignment):
        raise ValueError("All alignment records must have the same aligned length")

    if args.non_gap_proportion_pickle:
        require_pickle_current(
            args.non_gap_proportion_pickle,
            [args.alignment_file],
            "Non-gap-proportion",
        )
        non_gap_proportion = validate_numeric_series(
            load_pickle(args.non_gap_proportion_pickle, "non-gap-proportion"),
            alignment_length,
            "Non-gap-proportion",
            minimum=0.0,
            maximum=1.0,
        ).to_numpy()
        print(f"Loaded compatible non-gap proportions: {args.non_gap_proportion_pickle}")
    else:
        non_gap_proportion = get_non_gap_proportion(alignment, alignment_length)
    retained_positions = [
        position
        for position, proportion in enumerate(non_gap_proportion)
        if proportion > args.min_presence
    ]
    if not retained_positions:
        raise ValueError(
            "No alignment columns have genome presence strictly greater than "
            f"--min-presence ({args.min_presence:g})"
        )
    print(
        f"Retained {len(retained_positions):,} of {alignment_length:,} alignment "
        f"columns with genome presence > {args.min_presence:.0%}"
    )

    if args.position_maps_pickle:
        require_pickle_current(
            args.position_maps_pickle,
            [args.alignment_file],
            "Position-maps",
        )
        position_maps = validate_position_maps(
            load_pickle(args.position_maps_pickle, "position-maps"),
            alignment,
            retained_positions,
        )
        print(f"Loaded compatible position maps: {args.position_maps_pickle}")
    else:
        position_maps = get_position_maps(alignment, retained_positions)
    alignment_ids = {record.id for record in alignment}
    alias_count = add_enterovirus_a_blast_aliases(position_maps, alignment_ids)
    if alias_count:
        print(
            f"Added {alias_count:,} Enterovirus A BLAST-ID aliases "
            "(enterovirus-A~... -> enterovirus~A~...)"
        )
    if args.strain_to_copy_annots not in position_maps:
        available = ", ".join(record.id for record in alignment[:5])
        raise ValueError(
            f"Annotation strain {args.strain_to_copy_annots!r} is not in the alignment. "
            f"First alignment IDs: {available}"
        )

    if args.strain_to_copy_annots == HXB2_REFERENCE_ID:
        named_annotations = sorted(HXB2_ANNOTATIONS, key=lambda item: item[0])
        print("Used hard-coded HXB2 annotations")
    else:
        named_annotations = read_named_annotations(
            args.gff_file,
            args.annotation_feature_types,
            args.annotation_name_prefixes,
        )
    mapped_annotations = map_annotations_to_plot(
        named_annotations,
        position_maps[args.strain_to_copy_annots],
        retained_positions,
    )

    if args.pi_series_pickle:
        require_pickle_current(
            args.pi_series_pickle,
            [args.alignment_file],
            "Pi-series",
        )
        validate_pi_window_from_name(args.pi_series_pickle, args.pi_window_size)
        pi_values = validate_numeric_series(
            load_pickle(args.pi_series_pickle, "pi-series"),
            len(retained_positions),
            "Pi-series",
            minimum=0.0,
            expected_name="retained-column count",
        )
        print(f"Loaded compatible pi series: {args.pi_series_pickle}")
    else:
        pi_values = calculate_pi_on_retained_columns(
            args.alignment_file,
            retained_positions,
            args.pi_window_size,
        )

    coverage_by_label: dict[str, pd.Series] = {}
    unsmoothed_coverage_by_label: dict[str, pd.Series] = {}
    for blast_index, (blast_file, blast_label) in enumerate(
        zip(args.blast_file, args.blast_labels)
    ):
        if args.probe_coverage_pickles:
            probe_pickle = args.probe_coverage_pickles[blast_index]
            require_pickle_current(
                probe_pickle,
                [args.alignment_file, blast_file],
                f"Probe coverage for {blast_label}",
            )
            unsmoothed = validate_numeric_series(
                load_pickle(probe_pickle, f"probe coverage for {blast_label}"),
                alignment_length,
                f"Probe coverage for {blast_label}",
                minimum=0.0,
            )
            print(f"Loaded compatible probe coverage for {blast_label}: {probe_pickle}")
        else:
            _, unsmoothed = get_probedepth_for_plot(
                blast_file,
                args.window_size,
                alignment_length,
                non_gap_proportion,
                alignment,
                retained_positions,
                position_maps,
                gapthresh=1 - args.min_presence,
            )
        unsmoothed = pd.Series(unsmoothed).reset_index(drop=True)
        unsmoothed_coverage_by_label[blast_label] = unsmoothed
        retained = unsmoothed.iloc[retained_positions].reset_index(drop=True)
        coverage_by_label[blast_label] = smooth(retained, args.window_size)

    if args.save_pickles or args.pickle_output_prefix:
        pickle_prefix = (
            Path(args.pickle_output_prefix).expanduser()
            if args.pickle_output_prefix
            else output_path
        )
        min_presence_label = f"{args.min_presence:g}".replace(".", "p")
        save_pickle(
            np.asarray(non_gap_proportion, dtype=float),
            Path(f"{pickle_prefix}_non_gap_proportion.pkl"),
            "non-gap-proportion",
        )
        save_pickle(
            position_maps,
            Path(f"{pickle_prefix}_position_maps.pkl"),
            "position-maps",
        )
        save_pickle(
            pd.Series(pi_values).reset_index(drop=True),
            Path(
                f"{pickle_prefix}_pi_series_window{args.pi_window_size}_alignment_sliding_"
                f"minpresence{min_presence_label}.pkl"
            ),
            "pi-series",
        )
        for blast_index, label in enumerate(args.blast_labels, start=1):
            save_pickle(
                unsmoothed_coverage_by_label[label],
                Path(
                    f"{pickle_prefix}_blast_{blast_index}_{safe_pickle_label(label)}_"
                    "unique_probes_series_unsmoothed.pkl"
                ),
                f"probe coverage for {label}",
            )

    dampa_labels = [label for label in args.blast_labels if "dampa" in label.lower()]
    if not dampa_labels:
        raise ValueError(
            "At least one --blast-labels value must contain 'DAMPA' to create "
            "the relative panel"
        )
    dampa_label = args.dampa_reference_label or dampa_labels[0]
    if len(dampa_labels) > 1 and args.dampa_reference_label is None:
        print(
            f"Multiple DAMPA labels found; using the first as the relative-coverage "
            f"reference: {dampa_label}"
        )
    dampa_coverage = coverage_by_label[dampa_label].replace(0, np.nan)
    ordered_labels = sorted(
        args.blast_labels,
        key=lambda label: (program_rank(label), args.blast_labels.index(label)),
    )
    colors_by_label = dict(zip(ordered_labels, distinct_program_colors(ordered_labels)))

    lane_annotations = assign_annotation_lanes(mapped_annotations)
    lane_count = max(item[3] for item in lane_annotations) + 1
    annotation_height = max(0.9, 0.28 * lane_count + 0.45)
    figure_height = min(
        A4_HEIGHT_INCHES,
        max(args.figure_height, annotation_height + 4.35),
    )
    fig, (annotation_ax, pi_ax, coverage_ax, relative_ax) = plt.subplots(
        4,
        1,
        figsize=(args.figure_width, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [annotation_height, 1.0, 1.35, 1.35]},
        constrained_layout=True,
    )
    x_values = np.arange(len(retained_positions))
    xmax = len(retained_positions) - 1

    annotation_colors = plt.get_cmap("tab20").colors
    for index, (start, end, name, lane) in enumerate(lane_annotations):
        width = max(1, end - start + 1)
        annotation_ax.broken_barh(
            [(start, width)],
            (lane + 0.30, 0.55),
            facecolors=annotation_colors[index % len(annotation_colors)],
            edgecolors="black",
            linewidth=0.4,
        )
    annotation_ax.set_ylim(0, lane_count + 0.15)
    annotation_ax.set_yticks([])
    annotation_ax.set_ylabel("Genes")
    annotation_ax.set_title("Genome annotations", loc="left")

    pi_ax.plot(x_values, pi_values, color="purple", linewidth=1.0)
    pi_ax.axhline(0.15, color="black", linestyle="--", linewidth=0.7, alpha=0.7)
    pi_ax.set_ylabel("Nucleotide\ndiversity (π)")
    pi_ax.set_title(
        f"Nucleotide diversity ({args.pi_window_size}-bp sliding window)",
        loc="left",
    )
    finite_pi = pi_values[np.isfinite(pi_values)]
    if not finite_pi.empty:
        pi_ax.set_ylim(0, max(float(finite_pi.max()) * 1.08, 0.165))

    for label in ordered_labels:
        coverage_ax.plot(
            x_values,
            coverage_by_label[label],
            label=label,
            color=colors_by_label[label],
            linewidth=1.1,
            alpha=0.9,
        )
    coverage_ax.set_ylabel("Unique probe\ncount")
    coverage_ax.set_title(
        f"Probe coverage ({args.window_size}-position smoothing window)", loc="left"
    )
    coverage_ax.set_ylim(bottom=0)
    coverage_ax.legend(
        frameon=False,
        loc="upper right",
        ncol=min(3, len(ordered_labels)),
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0.25,
    )

    relative_ax.axhline(
        1.0,
        color=colors_by_label[dampa_label],
        linestyle="--",
        linewidth=1.0,
        label=dampa_label,
    )
    for label in ordered_labels:
        if label == dampa_label:
            continue
        relative_values = (coverage_by_label[label] / dampa_coverage).replace(
            [np.inf, -np.inf], np.nan
        )
        relative_ax.plot(
            x_values,
            relative_values,
            label=label,
            color=colors_by_label[label],
            linewidth=1.1,
            alpha=0.9,
        )
    relative_ax.set_ylabel("Coverage relative\nto DAMPA")
    relative_ax.set_title("DAMPA-relative probe coverage", loc="left")
    relative_ax.set_xlabel("Position in retained alignment")
    relative_ax.set_ylim(bottom=0)
    relative_ax.legend(
        frameon=False,
        loc="upper right",
        ncol=min(3, len(ordered_labels)),
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0.25,
    )

    tick_labels_numeric = np.arange(
        args.x_tick_interval,
        len(retained_positions) + 1,
        args.x_tick_interval,
        dtype=int,
    )
    if not len(tick_labels_numeric):
        tick_labels_numeric = np.asarray([1], dtype=int)
    tick_positions = tick_labels_numeric - 1
    tick_labels = [f"{position:,}" for position in tick_labels_numeric]
    relative_ax.set_xticks(tick_positions, tick_labels)

    for axis in (annotation_ax, pi_ax, coverage_ax, relative_ax):
        axis.set_xlim(0, xmax)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.7)
    for axis in (annotation_ax, pi_ax, coverage_ax):
        axis.tick_params(axis="x", labelbottom=False)

    # Compare the actual rendered widths, not character counts: narrow features
    # keep their names readable below the strip, while wider ones retain the
    # more compact centered label.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    font_properties = mpl.font_manager.FontProperties(size=FONT_SIZE)
    for start, end, name, lane in lane_annotations:
        width = max(1, end - start + 1)
        left_px = annotation_ax.transData.transform((start, 0))[0]
        right_px = annotation_ax.transData.transform((start + width, 0))[0]
        strip_width_px = abs(right_px - left_px)
        label_width_px = renderer.get_text_width_height_descent(
            name, font_properties, ismath=False
        )[0]
        fits_inside = label_width_px + 4 <= strip_width_px
        label_x = start + width / 2
        if not fits_inside:
            axes_left_px, axes_right_px = annotation_ax.bbox.x0, annotation_ax.bbox.x1
            label_center_px = annotation_ax.transData.transform((label_x, 0))[0]
            label_center_px = np.clip(
                label_center_px,
                axes_left_px + label_width_px / 2 + 1,
                axes_right_px - label_width_px / 2 - 1,
            )
            label_x = annotation_ax.transData.inverted().transform(
                (label_center_px, annotation_ax.bbox.y0)
            )[0]
        annotation_ax.text(
            label_x,
            lane + (0.575 if fits_inside else 0.25),
            name,
            ha="center",
            va="center" if fits_inside else "top",
            fontsize=FONT_SIZE,
            clip_on=True,
        )

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
