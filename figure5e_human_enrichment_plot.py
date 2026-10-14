#!/usr/bin/env python3
r"""Plot median read depth and interquartile range across samples for one region.

Dependencies: Python 3.9+, numpy, pandas and matplotlib.

Inputs
------
--counts accepts one or more headerless, tab-delimited files, one per sample.
Each row contains exactly three columns: chromosome, position, depth. Positions
are 1-based integers (as in samtools depth); depths must be finite and
non-negative. Chromosome/position pairs must be unique within each file. All
files must use the same reference and chromosome naming scheme. Include zero-depth
positions explicitly: missing positions are not imputed and each position's summary
uses only samples with an observation there. Empty files are rejected.

--chrom selects one chromosome. Optional --start and --end restrict the plot
to a 1-based inclusive interval. No sample names or loci are inferred from
filenames, and no experiment-specific exclusions are applied.

--bed optionally supplies target intervals for grey overlays. The first three
columns are chromosome, start, end, using standard BED coordinates (0-based,
end-exclusive); extra columns are ignored. BED coordinates are converted to
1-based inclusive positions before matching depths. A BED file is required
when --target-quantile is greater than zero.

Calculations
------------
The line is the per-position median of raw depths; the band spans the 25th to
75th percentiles across input samples. No depth normalisation is performed.
By default all samples are included. --target-quantile optionally retains
samples whose median depth across observed target positions is at or above
that quantile of sample medians (0.5 selects the upper half, including ties).
This selection uses targets within the selected chromosome/plot interval;
samples without observed target positions are excluded when filtering is on.
Overlapping BED intervals count each observed position only once.

The y-axis is logarithmic. Values below --plot-floor (including zeros) are
drawn at that floor for display only; summaries and sample selection use the
original depths. Missing genomic positions break the line and shaded band.
Exactly one figure is saved to --output; its extension determines the format
(e.g. .pdf or .png). Existing output files are overwritten. PDF text remains
editable; --dpi controls raster export resolution.

Example::

    python figure_scripts/figure5e_human_enrichment_plot.py \
        --counts data/sample_1.tsv data/sample_2.tsv \
        --chrom chr1 --start 10001 --end 15000 \
        --bed data/targets.bed --output output/enrichment.pdf
"""

import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def read_counts(path, chrom, start=None, end=None):
    """Validate one sample's depth table and select the requested coordinates."""
    try:
        data = pd.read_csv(path, sep="\t", header=None, dtype={0: str})
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Empty depth file: {path}") from error
    if data.shape[1] != 3:
        raise ValueError(f"Expected chromosome, position, depth columns in {path}")
    data.columns = ["chrom", "position", "depth"]
    data[["position", "depth"]] = data[["position", "depth"]].apply(
        pd.to_numeric, errors="raise"
    )
    if data["chrom"].isna().any() or not np.isfinite(
        data[["position", "depth"]].to_numpy()
    ).all():
        raise ValueError(f"Missing or non-finite values in {path}")
    if (
        (data["position"] < 1).any()
        or (data["position"] % 1 != 0).any()
        or (data["depth"] < 0).any()
    ):
        raise ValueError(f"Positions must be positive integers and depths non-negative: {path}")
    if data.duplicated(["chrom", "position"]).any():
        raise ValueError(f"Duplicate chromosome/position pairs in {path}")
    data = data.loc[data["chrom"].eq(chrom)].copy()
    if start is not None:
        data = data.loc[data["position"].between(start, end)].copy()
    if data.empty:
        raise ValueError(f"No depth observations in the selected region: {path}")
    data["position"] = data["position"].astype(int)
    return data.sort_values("position")


def read_targets(path, chrom, start, end):
    """Read BED targets, convert coordinates, and clip to the plotted interval."""
    bed = pd.read_csv(
        path, sep="\t", header=None, comment="#", usecols=[0, 1, 2], dtype={0: str}
    )
    bed.columns = ["chrom", "start", "end"]
    bed[["start", "end"]] = bed[["start", "end"]].apply(pd.to_numeric, errors="raise")
    coordinates = bed[["start", "end"]].to_numpy()
    if (
        bed["chrom"].isna().any()
        or not np.isfinite(coordinates).all()
        or (coordinates % 1 != 0).any()
        or (bed["start"] < 0).any()
        or (bed["end"] <= bed["start"]).any()
    ):
        raise ValueError(f"Invalid BED intervals in {path}")
    bed = bed.loc[bed["chrom"].eq(chrom)].copy()
    bed["start"] = bed["start"].astype(int) + 1
    bed["end"] = bed["end"].astype(int)
    bed = bed.loc[(bed["end"] >= start) & (bed["start"] <= end)].copy()
    bed["start"] = bed["start"].clip(lower=start)
    bed["end"] = bed["end"].clip(upper=end)
    return bed


def select_samples(data, targets, quantile):
    """Optionally select samples using their median depth within target intervals."""
    if quantile == 0:
        return data
    target_mask = np.zeros(len(data), dtype=bool)
    for region in targets.itertuples(index=False):
        target_mask |= data["position"].between(region.start, region.end).to_numpy()
    medians = data.loc[target_mask].groupby("sample")["depth"].median()
    if medians.empty:
        raise ValueError("No observed target positions available for sample selection")
    retained = medians.index[medians >= medians.quantile(quantile)]
    return data.loc[data["sample"].isin(retained)].copy()


def summarize_depth(data):
    """Calculate per-position median and quartiles with equal weight per sample."""
    return data.groupby("position")["depth"].agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).sort_index()


def plot_depth(summary, targets, output, chrom, start, end, title, floor, width, height, dpi):
    """Save a single median/IQR plot, breaking paths across unobserved positions."""
    # Insert a missing point at each coordinate gap without allocating the entire
    # chromosome. Otherwise a line would imply measured depths across that gap.
    positions = summary.index.to_numpy()
    gap_starts = positions[:-1][np.diff(positions) > 1] + 1
    summary = summary.reindex(np.union1d(positions, gap_starts))
    x = summary.index.to_numpy(dtype=float)
    values = summary.clip(lower=floor)
    with mpl.rc_context({"pdf.fonttype": 42, "font.size": 12}):
        fig, axis = plt.subplots(figsize=(width, height))
        axis.plot(x, values["median"].to_numpy(), color="#9e2a2b", label="Median")
        axis.fill_between(
            x, values["q25"].to_numpy(), values["q75"].to_numpy(),
            color="#9e2a2b", alpha=0.3, label="Interquartile range",
        )
        for i, region in enumerate(targets.itertuples(index=False)):
            axis.axvspan(
                region.start - 0.5, region.end + 0.5,
                color="grey", alpha=0.25, linewidth=0,
                label="Target interval" if i == 0 else None,
            )
        axis.set_yscale("log")
        upper = max(float(values["q75"].max()) * 1.2, floor * 10)
        axis.set_ylim(floor, upper)
        axis.set_xlim(start - 0.5, end + 0.5)
        axis.set_xlabel(f"Position on {chrom} (1-based)")
        axis.set_ylabel("Read depth")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        if title:
            axis.set_title(title)
        axis.grid(which="major", axis="y", linewidth=0.6, alpha=0.5)
        axis.grid(which="minor", axis="y", linewidth=0.3, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--counts", nargs="+", type=Path, required=True,
                        help="Headerless chromosome/position/depth TSVs, one per sample.")
    parser.add_argument("--chrom", required=True, help="Chromosome name as written in the inputs.")
    parser.add_argument("--start", type=int, help="First position, 1-based inclusive.")
    parser.add_argument("--end", type=int, help="Last position, 1-based inclusive.")
    parser.add_argument("--bed", type=Path, help="Optional target intervals in BED format.")
    parser.add_argument("--target-quantile", type=float, default=0,
                        help="Target-depth quantile cutoff in [0, 1]; 0 disables filtering.")
    parser.add_argument("--output", type=Path, required=True, help="One output figure path.")
    parser.add_argument("--title", help="Optional plot title.")
    parser.add_argument("--plot-floor", type=float, default=0.9,
                        help="Display floor for log-scale depths (default: 0.9).")
    parser.add_argument("--plot-width", type=float, default=7, help="Width in inches (default: 7).")
    parser.add_argument("--plot-height", type=float, default=5, help="Height in inches (default: 5).")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI (default: 300).")
    args = parser.parse_args()
    if (args.start is None) != (args.end is None):
        parser.error("Supply --start and --end together")
    if args.start is not None and (args.start < 1 or args.end < args.start):
        parser.error("Require 1 <= --start <= --end")
    if not 0 <= args.target_quantile <= 1:
        parser.error("--target-quantile must be in [0, 1]")
    if args.target_quantile > 0 and args.bed is None:
        parser.error("--bed is required for target-depth filtering")
    dimensions = np.array([args.plot_floor, args.plot_width, args.plot_height])
    if not np.isfinite(dimensions).all() or (dimensions <= 0).any() or args.dpi < 1:
        parser.error("Plot dimensions, floor and DPI must be positive and finite")
    if len({path.resolve() for path in args.counts}) != len(args.counts):
        parser.error("Each sample file must be supplied only once")
    if args.output.resolve() in {
        path.resolve() for path in args.counts + ([args.bed] if args.bed else [])
    }:
        parser.error("Output must not overwrite an input file")
    try:
        samples = []
        for sample, path in enumerate(args.counts):
            data = read_counts(path, args.chrom, args.start, args.end)
            data["sample"] = sample
            samples.append(data)
        data = pd.concat(samples, ignore_index=True)
        start = args.start if args.start is not None else int(data["position"].min())
        end = args.end if args.end is not None else int(data["position"].max())
        targets = (
            read_targets(args.bed, args.chrom, start, end)
            if args.bed else pd.DataFrame(columns=["chrom", "start", "end"])
        )
        data = select_samples(data, targets, args.target_quantile)
        summary = summarize_depth(data)
        plot_depth(
            summary, targets, args.output, args.chrom, start, end, args.title,
            args.plot_floor, args.plot_width, args.plot_height, args.dpi,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {args.output} using {data['sample'].nunique()} of {len(args.counts)} samples")


if __name__ == "__main__":
    main()
