#!/usr/bin/env python3
r"""Plot probe counts and resource use across the JEV input dilutions.

Run folders are named ``clusterN_repN_step_N``. For every run, the script
counts the four probe FASTAs and parses elapsed time and maximum resident set
size from each program's GNU ``time`` file. Points show individual replicates;
lines connect the median at each number of input genomes.

Input contract
--------------
Python 3.10+ and matplotlib. The input directory is a required positional
argument. Each clusterN_repN_step_N directory represents a replicate with N
input genomes (the last N). Under every run, provide these relative files:

    dampa/outprefix_probes.fasta       dampa/timing.txt
    probetools_probes.fa              probetools/timing.txt
    catch/probes.fasta                catch/timing.txt
    catch_large/probes.fasta          catch_large/timing.txt

FASTA headers are counted as probes. Timing files must contain GNU time
elapsed and maxresident fields, e.g. 0:01.50elapsed and 1024maxresident)k.
Memory is interpreted as KiB unless an explicit k/m/g suffix is present, and
converted to GiB. Missing files fail rather than silently dropping replicates.
PROGRAMS defines the publication's file layout and labels.

--ideal-probes is the reference count for probe redundancy (count/reference);
92 reproduces the JEV analysis. Set it explicitly for another reference.
Outputs: a tidy jev_dilutions_data.csv and four plots (probe count, redundancy,
elapsed seconds, memory) in --formats. Existing output names are overwritten.
All output defaults are relative to the working directory.

Example:
    python figure_scripts/figure4b-c-jevsubsets.py data/subsampling \
        --ideal-probes 92 --output-dir output/figure4bc --formats pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter


DEFAULT_OUTPUT_DIR = Path("output/figure4bc")
RUN_RE = re.compile(
    r"^cluster(?P<cluster>\d+)_rep(?P<replicate>\d+)_step_(?P<inputs>\d+)$"
)
ELAPSED_RE = re.compile(r"(?P<elapsed>\d+(?::\d+){0,2}(?:\.\d+)?)elapsed")
MAX_RSS_RE = re.compile(
    r"(?P<rss>\d+(?:\.\d+)?)maxresident\)?(?P<unit>[kKmMgG]?)"
)

# Manuscript colours sampled from Figure 4 panel A; order follows that legend.
PROGRAMS = {
    "dampa": {
        "label": "dampa",
        "colour": "#2278B5",
        "probe": "dampa/outprefix_probes.fasta",
        "timing": "dampa/timing.txt",
    },
    "probetools": {
        "label": "probetools",
        "colour": "#F57F20",
        "probe": "probetools_probes.fa",
        "timing": "probetools/timing.txt",
    },
    "catch": {
        "label": "catch",
        "colour": "#2FA148",
        "probe": "catch/probes.fasta",
        "timing": "catch/timing.txt",
    },
    "catch_large": {
        "label": "catch_large",
        "colour": "#C03D3E",
        "probe": "catch_large/probes.fasta",
        "timing": "catch_large/timing.txt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing cluster/replicate/step folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--ideal-probes",
        type=float,
        default=92,
        help="probe count represented by 1 in the normalised plot (default: 92)",
    )
    parser.add_argument(
        "--jitter-width",
        type=float,
        default=4,
        help="full horizontal jitter width in genome units (default: 4)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png"),
        help="plot formats to write (default: pdf png)",
    )
    return parser.parse_args()


def count_fasta_records(path: Path) -> int:
    """Count FASTA headers without loading sequences into memory."""
    with path.open(encoding="utf-8") as handle:
        count = sum(line.startswith(">") for line in handle)
    if count == 0:
        raise ValueError(f"probe FASTA contains no records: {path}")
    return count


def elapsed_to_seconds(value: str) -> float:
    """Convert GNU-time SS, MM:SS, or HH:MM:SS text to seconds."""
    fields = [float(field) for field in value.split(":")]
    if len(fields) == 1:
        return fields[0]
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600 + fields[1] * 60 + fields[2]
    raise ValueError(f"invalid elapsed time: {value!r}")


def parse_timing(path: Path) -> tuple[float, float]:
    """Return elapsed seconds and maximum resident memory in GiB."""
    text = path.read_text(encoding="utf-8")
    elapsed_match = ELAPSED_RE.search(text)
    memory_match = MAX_RSS_RE.search(text)
    if elapsed_match is None or memory_match is None:
        raise ValueError(f"could not parse elapsed time and max RSS from {path}")

    elapsed_seconds = elapsed_to_seconds(elapsed_match.group("elapsed"))
    memory_value = float(memory_match.group("rss"))
    # These GNU-time files report %M in KiB. Also accept explicit MiB/GiB.
    unit = memory_match.group("unit").lower() or "k"
    gib_factor = {"k": 1 / 1024**2, "m": 1 / 1024, "g": 1}
    return elapsed_seconds, memory_value * gib_factor[unit]


def collect_results(input_dir: Path, ideal_probes: float) -> list[dict]:
    """Return a tidy row for every program in every replicate/step folder."""
    run_dirs = []
    for folder in input_dir.iterdir():
        match = RUN_RE.fullmatch(folder.name)
        if folder.is_dir() and match:
            run_dirs.append((int(match.group("inputs")), int(match.group("replicate")), folder, match))
    if not run_dirs:
        raise ValueError(f"no folders matching {RUN_RE.pattern!r} in {input_dir}")

    rows = []
    missing = []
    for inputs, replicate, folder, match in sorted(run_dirs):
        for program, config in PROGRAMS.items():
            probe_path = folder / config["probe"]
            timing_path = folder / config["timing"]
            absent = [path for path in (probe_path, timing_path) if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            probe_count = count_fasta_records(probe_path)
            elapsed_seconds, max_memory_gib = parse_timing(timing_path)
            rows.append(
                {
                    "cluster": int(match.group("cluster")),
                    "replicate": replicate,
                    "input_genomes": inputs,
                    "program": program,
                    "probe_count": probe_count,
                    "probe_redundancy": probe_count / ideal_probes,
                    "elapsed_seconds": elapsed_seconds,
                    "max_memory_gib": max_memory_gib,
                    "probe_file": str(probe_path.resolve()),
                    "timing_file": str(timing_path.resolve()),
                }
            )

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:12])
        more = "" if len(missing) <= 12 else f"\n  ... and {len(missing) - 12} more"
        raise FileNotFoundError(
            f"missing {len(missing)} required probe/timing files:\n{preview}{more}"
        )
    expected = len(run_dirs) * len(PROGRAMS)
    if len(rows) != expected:
        raise RuntimeError(f"collected {len(rows)} observations; expected {expected}")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_jitter(program: str, inputs: int, replicate: int, width: float) -> float:
    """Return deterministic jitter independent of filesystem/row ordering."""
    key = f"{program}:{inputs}:{replicate}".encode()
    integer = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")
    fraction = integer / (2**32 - 1)
    return (fraction - 0.5) * width


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(direction="out", length=3, width=0.8)
    axis.grid(False)


def plot_metric(
    rows: list[dict],
    metric: str,
    ylabel: str,
    output_stem: Path,
    formats: list[str] | tuple[str, ...],
    ideal_probes: float,
    jitter_width: float,
) -> list[Path]:
    """Draw replicate points plus one median line per program."""
    fig, axis = plt.subplots(figsize=(4.45, 3.35))
    medians: dict[str, dict[int, list[float]]] = {
        program: defaultdict(list) for program in PROGRAMS
    }

    for program, config in PROGRAMS.items():
        subset = [row for row in rows if row["program"] == program]
        axis.scatter(
            [
                row["input_genomes"]
                + stable_jitter(
                    program, row["input_genomes"], row["replicate"], jitter_width
                )
                for row in subset
            ],
            [row[metric] for row in subset],
            color=config["colour"],
            s=15,
            alpha=0.95,
            edgecolors="none",
            zorder=3,
        )
        for row in subset:
            medians[program][row["input_genomes"]].append(row[metric])
        median_x = sorted(medians[program])
        axis.plot(
            median_x,
            [median(medians[program][x]) for x in median_x],
            color=config["colour"],
            linewidth=1.6,
            alpha=0.35,
            zorder=2,
        )

    inputs = sorted({row["input_genomes"] for row in rows})
    axis.set_xticks(inputs)
    axis.set_xticklabels([str(value) for value in inputs])
    axis.set_xlim(min(inputs) - 11, max(inputs) + 11)
    axis.set_xlabel(f"Included genomes (total {max(inputs)})")
    axis.set_ylabel(ylabel)
    style_axis(axis)

    if metric == "probe_redundancy":
        axis.axhline(1, color="black", linestyle=(0, (4, 2)), linewidth=1.1, zorder=1)
        axis.annotate(
            f"- {ideal_probes:g} probes",
            xy=(1, 1),
            xycoords=("axes fraction", "data"),
            xytext=(-1, -1),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
        )
    else:
        axis.set_ylim(bottom=0)
    if metric == "probe_count":
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    handles = [
        Line2D(
            [0],
            [0],
            color=config["colour"],
            linewidth=1.6,
            alpha=0.55,
            label=config["label"],
        )
        for config in PROGRAMS.values()
    ]
    axis.legend(
        handles=handles,
        loc="best",
        frameon=True,
        framealpha=0.65,
        fontsize=7,
        handlelength=2.4,
        borderpad=0.4,
        labelspacing=0.25,
    )
    fig.tight_layout(pad=0.6)

    outputs = []
    for output_format in formats:
        output = output_stem.with_suffix(f".{output_format}")
        kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            kwargs["dpi"] = 300
        fig.savefig(output, **kwargs)
        outputs.append(output)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
    if not math.isfinite(args.ideal_probes) or args.ideal_probes <= 0:
        raise ValueError("--ideal-probes must be finite and greater than zero")
    if not math.isfinite(args.jitter_width) or args.jitter_width < 0:
        raise ValueError("--jitter-width must be finite and non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows = collect_results(args.input_dir, args.ideal_probes)
    csv_path = args.output_dir / "jev_dilutions_data.csv"
    write_csv(rows, csv_path)
    specs = (
        ("probe_count", "Probe number", "jev_probe_counts"),
        ("probe_redundancy", "Probe redundancy", "jev_probe_redundancy"),
        ("elapsed_seconds", "Time (seconds)", "jev_compute_time"),
        ("max_memory_gib", "Maximum memory (GiB)", "jev_max_memory"),
    )
    outputs = []
    for metric, ylabel, stem in specs:
        outputs.extend(
            plot_metric(
                rows,
                metric,
                ylabel,
                args.output_dir / stem,
                args.formats,
                args.ideal_probes,
                args.jitter_width,
            )
        )

    print(f"Collected {len(rows)} observations from {len(rows) // len(PROGRAMS)} runs.")
    print(f"Wrote data: {csv_path}")
    for output in outputs:
        print(f"Wrote plot: {output}")


if __name__ == "__main__":
    main()
