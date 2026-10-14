#!/usr/bin/env python3
r"""Summarise probe counts and plot Figure 4-style normalized columns.

The six-species counts reproduce the data used by the existing Figure 4
workflow.  The dengue comparison therefore uses ``dengue_w_gensig_in_outputs``.
The SPECIES table defines the representative run for each of the six species.
Representative FASTAs listed there are required. Missing main-program FASTAs
are written as blank CSV fields and are not plotted.

Input contract
--------------
Python 3.10+, numpy and matplotlib. --results-root is required; its directory
layout is:

    6_species_4_tools/<species>/<program-specific FASTA>
    dampa_representative/<species>/<representative FASTA>

For each main species directory, PROGRAMS lists the relative FASTA candidates:
dampa/dampa_probes.fasta, catch/probes.fasta, catch_large/catch_large_e0_probes.fasta
(fallback catch_large/probes.fasta), and probetools/probetools_probes.fa.
If both CATCH-large candidates exist, the first is used. Each FASTA record
represents one probe; headers are counted without sequence deduplication.

SPECIES specifies source directory names, display labels, publication genome
counts and representative FASTA paths. Dengue's main directory is
'dengue_w_gensig_in_outputs', while its representative directory is 'dengue'.
Genome counts on axis labels come from SPECIES, not from the probe FASTAs.
Edit the table for a different evaluation cohort. The six representative paths
currently configured in SPECIES are all required; other missing combinations
remain blank. Empty FASTAs are rejected.

Optional --baseline is a headered CSV with species, program and probe_count
columns, matching the keys in SPECIES/PROGRAMS; counts must agree with the
observed FASTAs. No baseline is searched for or loaded implicitly.

Bar height is 100 * probe_count / largest available count for that species,
including the representative count. Bar labels show original integer counts,
not percentages. Outputs under --output-dir are the headered wide count CSV
and matching PDF/PNG with stem six_species_probe_counts_with_dampa_representative.
Existing outputs are overwritten; output defaults are relative to the working
directory. --dpi controls raster resolution.

Example:
    python figure_scripts/supp_fig5a_probe_counts_with_representative.py \
        --results-root data/comparison_runs --output-dir output/supp_fig5a
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DEFAULT_OUTPUT_DIR = Path("output/supp_fig5a")


@dataclass(frozen=True)
class Species:
    key: str
    source_directory: str
    label: str
    genome_count: int
    representative_fasta: str | None = None


@dataclass(frozen=True)
class Program:
    key: str
    label: str
    colour: str
    fasta_candidates: tuple[str, ...]


SPECIES = (
    Species("jev", "jev", "JEV", 398, "jev/jev_rep_consseq_probes.fasta"),
    Species(
        "hiv_groupM",
        "hiv_groupM",
        "HIV Group M",
        72,
        "hiv_groupM/hiv_groupM_rep_consseq_probes.fasta",
    ),
    Species(
        "eva",
        "eva",
        "Enterovirus A",
        768,
        "eva/eva_rep_consseq_probes.fasta",
    ),
    Species("hhv4", "hhv4", "HHV4", 479, "hhv4/hhv4_rep_consseq_probes.fasta"
    ),
    Species("dengue", "dengue_w_gensig_in_outputs", "Dengue", 16_400,"dengue/dengue_rep_consseq_probes.fasta"),
    Species("pertussis_18", "pertussis_18", r"$\it{B.\ pertussis}$", 18, "pertussis_18/pertussis_rep_consseq_probes.fasta"),
)

PROGRAMS = (
    Program("dampa", "DAMPA", "#3279A8", ("dampa/dampa_probes.fasta",)),
    Program("dampa_representative", "DAMPA_rep", "#9467BD", ()),
    Program("catch", "CATCH", "#398D3E", ("catch/probes.fasta",)),
    Program(
        "catch_large",
        "CATCH_large",
        "#C43D3D",
        (
            "catch_large/catch_large_e0_probes.fasta",
            "catch_large/probes.fasta",
        ),
    ),
    Program(
        "probetools",
        "Probetools",
        "#E58025",
        ("probetools/probetools_probes.fa",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Root containing 6_species_4_tools/ and dampa_representative/.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--baseline", type=Path,
                        help="Optional headered species/program/probe_count CSV for count verification.")
    args = parser.parse_args()
    if not args.results_root.is_dir():
        parser.error(f"Results directory does not exist: {args.results_root}")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if args.baseline is not None and not args.baseline.is_file():
        parser.error(f"Baseline file does not exist: {args.baseline}")
    return args


def count_fasta_records(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    if count == 0:
        raise ValueError(f"No FASTA records found in {path}")
    return count


def first_existing(base: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


def collect_counts(results_root: Path) -> dict[str, dict[str, int | None]]:
    main_root = results_root / "6_species_4_tools"
    representative_root = results_root / "dampa_representative"
    counts: dict[str, dict[str, int | None]] = {}

    for species in SPECIES:
        species_root = main_root / species.source_directory
        row: dict[str, int | None] = {}
        for program in PROGRAMS:
            if program.key == "dampa_representative":
                continue
            fasta = first_existing(species_root, program.fasta_candidates)
            row[program.key] = count_fasta_records(fasta) if fasta else None

        if species.representative_fasta:
            representative_fasta = representative_root / species.representative_fasta
            if not representative_fasta.is_file():
                raise FileNotFoundError(representative_fasta)
            row["dampa_representative"] = count_fasta_records(representative_fasta)
        else:
            row["dampa_representative"] = None
        counts[species.key] = row
    return counts


def validate_against_figure4_baseline(
    counts: dict[str, dict[str, int | None]], baseline_path: Path
) -> None:
    if not baseline_path.is_file():
        return
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            species = record["species"]
            program = record["program"]
            expected = int(record["probe_count"])
            observed = counts[species][program]
            if observed != expected:
                raise ValueError(
                    "Count differs from the existing Figure 4 baseline: "
                    f"{species}/{program}: FASTA={observed}, baseline={expected}"
                )


def write_summary_csv(
    path: Path, counts: dict[str, dict[str, int | None]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", *(program.key for program in PROGRAMS)])
        for species in SPECIES:
            writer.writerow(
                [
                    species.key,
                    *(
                        "" if counts[species.key][program.key] is None
                        else counts[species.key][program.key]
                        for program in PROGRAMS
                    ),
                ]
            )


def draw_plot(counts: dict[str, dict[str, int | None]]) -> plt.Figure:
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
    fig, axis = plt.subplots(figsize=(19 * cm, 5.4 * cm))
    species_x = np.arange(len(SPECIES), dtype=float)
    offsets = np.linspace(-0.32, 0.32, len(PROGRAMS))
    bar_width = 0.145

    for species_index, species in enumerate(SPECIES):
        row = counts[species.key]
        available = [value for value in row.values() if value is not None]
        normalizer = max(available)
        for program_index, program in enumerate(PROGRAMS):
            original = row[program.key]
            if original is None:
                continue
            height = 100 * original / normalizer
            centre = species_x[species_index] + offsets[program_index]
            axis.bar(
                centre,
                height,
                width=bar_width,
                color=program.colour,
                edgecolor="none",
                zorder=2,
            )
            inside = height >= 25
            axis.text(
                centre,
                height / 2 if inside else height + 2.5,
                f"{original:,}",
                ha="center",
                va="center" if inside else "bottom",
                rotation=90,
                fontsize=5.8,
                color="white" if inside else "black",
                clip_on=False,
            )

    axis.legend(
        handles=[Patch(facecolor=p.colour, label=p.label) for p in PROGRAMS],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.0,
    )
    axis.set_xticks(
        species_x,
        [f"{species.label}\n({species.genome_count:,})" for species in SPECIES],
    )
    axis.set_xlim(-0.55, len(SPECIES) - 0.45)
    axis.set_ylim(0, 112)
    axis.set_yticks([])
    axis.set_ylabel("Probe number")
    axis.tick_params(axis="x", length=3, pad=3)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.82, bottom=0.25)
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = collect_counts(args.results_root)
    if args.baseline is not None:
        validate_against_figure4_baseline(counts, args.baseline)

    csv_path = args.output_dir / "six_species_probe_counts_with_dampa_representative.csv"
    stem = args.output_dir / "six_species_probe_counts_with_dampa_representative"
    write_summary_csv(csv_path, counts)

    figure = draw_plot(counts)
    figure.savefig(stem.with_suffix(".pdf"), dpi=args.dpi)
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    plt.close(figure)

    print(f"Wrote {csv_path}")
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
