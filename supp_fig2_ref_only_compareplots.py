#!/usr/bin/env python3
r"""Compare reference-derived and full-species probe coverage across datasets.

Dependencies: Python 3.10+, numpy, pandas, matplotlib, seaborn >= 0.13 and scipy.
No files are read and no plots are created on import.

Inputs
------
--reference-files and --dampa-files are corresponding lists of headered
per-genome coverage evaluation CSVs, one file per --species-labels value.
The first CSV column contains unique genome IDs (its header may be blank).
The column named '0' is the PERCENTAGE of each genome with zero probe coverage
(values from 0 to 100, not fractions). Other evaluation columns are ignored.
Reference files contain results for probes chopped from a reference genome;
DAMPA files contain results for the full-species probe design. This script
consumes existing evaluations and does not design or evaluate probes.

Optional --genome-ids LABEL FILE restricts BOTH methods for that dataset to
genome IDs listed one per line. Repeat the option for more datasets. LABEL must
match a --species-labels value; no HIV/group-M filter is applied implicitly.
Missing/non-finite percentages and duplicate genome IDs are rejected.

Calculations and outputs
------------------------
The upper panel shows zero-coverage percentages as boxplots and individual
genome points. Lower panels show fractions and counts of genomes in bins
[0, 1], (1, 50] and (50, 100] percent missing coverage, matching the original.
All valid genomes are plotted, even when the methods evaluate different sets.
Mean covered percentage is 100 minus mean zero-coverage percentage.

Paired t-tests compare covered percentages after joining on GENOME ID (the
original script paired by row order when lengths matched). Only shared IDs
enter the test; the exported n_pairs records that sample size. With fewer than
two shared genomes, test statistics are missing. Tests are two-sided, with no
multiple-testing correction. The absolute mean difference is based on those
paired observations. Reordering CSV rows therefore cannot change the test.

--output-prefix writes <prefix>_0cov_vs_ref_combined.pdf plus CSVs with suffixes
_mean_cov_val, _test_cov_val, _combined_alldata and _0covbins_not_cumulative.
The bins CSV is tidy: one row per dataset/method/bin, with count and proportion.
Existing output names are overwritten. --seed makes point jitter reproducible.

Example::

    python figure_scripts/supp_fig2_ref_only_compareplots.py \
        --reference-files data/reference_a.csv data/reference_b.csv \
        --dampa-files data/dampa_a.csv data/dampa_b.csv \
        --species-labels SpeciesA SpeciesB --output-prefix output/supp_fig2
"""

import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns


SOURCES = ("dampa_all", "chopped_ref")
PALETTE = {"chopped_ref": "tab:grey", "dampa_all": "tab:blue"}
BIN_EDGES = (0, 1, 50, 100)
BIN_LABELS = ("0-1", "1-50", "50-100")


def read_coverage(path, species, source, allowed_ids=None):
    """Read the genome-ID index and validate the zero-coverage percentages."""
    frame = pd.read_csv(path, index_col=0, dtype={0: str})
    if "0" not in frame.columns:
        raise ValueError(f"Missing required '0' percentage column: {path}")
    if frame.index.hasnans or frame.index.duplicated().any():
        raise ValueError(f"Genome IDs must be non-missing and unique: {path}")
    frame.index = frame.index.astype(str)
    values = pd.to_numeric(frame["0"], errors="raise")
    if not np.isfinite(values).all() or not values.between(0, 100).all():
        raise ValueError(f"Zero-coverage percentages must be finite and in [0, 100]: {path}")
    data = pd.DataFrame({"genome": frame.index, "0": values.to_numpy()})
    if allowed_ids is not None:
        data = data.loc[data["genome"].isin(allowed_ids)].copy()
    if data.empty:
        raise ValueError(f"No genomes remain for {species}/{source}: {path}")
    return data.assign(species=species, source=source, coverage=100-data["0"])


def paired_tests(data, species_order):
    """Pair by genome ID rather than by the order of CSV rows."""
    results = []
    for species in species_order:
        subset = data.loc[data["species"].eq(species)]
        paired = subset.pivot(index="genome", columns="source", values="coverage").dropna()
        n_pairs = len(paired)
        statistic, p_value = (ttest_rel(paired["dampa_all"], paired["chopped_ref"])
                              if n_pairs >= 2 else (np.nan, np.nan))
        results.append({
            "species": species, "comparison": "dampa_all vs chopped_ref",
            "n_pairs": n_pairs, "t_stat": statistic, "p_value": p_value,
            "abs_diff": abs(paired["dampa_all"].mean()-paired["chopped_ref"].mean()),
        })
    return pd.DataFrame(results)


def coverage_bins(data):
    """Summarise non-cumulative missing-coverage bins with explicit boundaries."""
    rows = []
    for (species, source), group in data.groupby(["species", "source"], sort=False):
        for i, label in enumerate(BIN_LABELS):
            selected = group["0"].le(BIN_EDGES[i+1])
            if i > 0:
                selected &= group["0"].gt(BIN_EDGES[i])
            count = int(selected.sum())
            rows.append({"species": species, "source": source, "bin": label,
                         "count": count, "total": len(group), "proportion": count/len(group)})
    return pd.DataFrame(rows)


def draw_plot(data, bins, args):
    """Save the original box/strip plot above per-dataset bin proportions."""
    species_order = args.species_labels
    np.random.seed(args.seed)
    with mpl.rc_context({"pdf.fonttype": 42, "font.size": 10}):
        fig = plt.figure(figsize=(max(6, len(species_order)*2.5), 12))
        grid = fig.add_gridspec(2, len(species_order), height_ratios=[3, 1], hspace=0.12, wspace=0.12)
        axis = fig.add_subplot(grid[0, :])
        sns.boxplot(data=data, x="species", y="0", hue="source", order=species_order,
                    hue_order=SOURCES, showfliers=False, fill=False, linewidth=1,
                    palette=PALETTE, ax=axis)
        sns.stripplot(data=data, x="species", y="0", hue="source", order=species_order,
                      hue_order=SOURCES, dodge=True, alpha=0.3, jitter=0.32, size=6,
                      palette=PALETTE, legend=False, ax=axis)
        for threshold in (1, 50):
            axis.axhline(threshold, color="black", linestyle="--", alpha=0.5, zorder=0)
        axis.set_ylabel("Percentage of genome with zero probe coverage")
        axis.set_xlabel("")
        axis.legend(title="Method")
        for j, species in enumerate(species_order):
            bottom = fig.add_subplot(grid[1, j])
            subset = bins.loc[bins["species"].eq(species)]
            sns.barplot(data=subset, y="bin", x="proportion", hue="source", order=BIN_LABELS,
                        hue_order=SOURCES, palette=PALETTE, orient="h", errorbar=None,
                        legend=False, ax=bottom)
            for container, source in zip(bottom.containers, SOURCES):
                counts = subset.loc[subset["source"].eq(source)].set_index("bin").loc[list(BIN_LABELS), "count"]
                bottom.bar_label(container, labels=counts, label_type="center", fontsize=8)
            bottom.set_xlim(0, 1.02)
            bottom.set_xlabel("Fraction of genomes")
            bottom.set_ylabel("Missing coverage (%)" if j == 0 else "")
            if j != 0:
                bottom.tick_params(axis="y", labelleft=False)
            bottom.invert_yaxis()
        fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.07)
        output = Path(f"{args.output_prefix}_0cov_vs_ref_combined.pdf")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-files", nargs="+", type=Path, required=True)
    parser.add_argument("--dampa-files", nargs="+", type=Path, required=True)
    parser.add_argument("--species-labels", nargs="+", required=True)
    parser.add_argument("--genome-ids", nargs=2, action="append", default=[], metavar=("LABEL", "FILE"))
    parser.add_argument("--output-prefix", type=Path, default=Path("output/supp_fig2"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not len(args.reference_files) == len(args.dampa_files) == len(args.species_labels):
        parser.error("Both file lists and --species-labels must have the same length")
    if len(set(args.species_labels)) != len(args.species_labels):
        parser.error("Species labels must be unique")
    filters = {}
    for label, path in args.genome_ids:
        if label not in args.species_labels or label in filters:
            parser.error("Each genome-ID filter must specify a unique known species label")
        filters[label] = {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
    parts = []
    for label, reference, dampa in zip(args.species_labels, args.reference_files, args.dampa_files):
        for path, source in ((reference, "chopped_ref"), (dampa, "dampa_all")):
            parts.append(read_coverage(path, label, source, filters.get(label)))
    data = pd.concat(parts, ignore_index=True)
    means = data.groupby(["species", "source"], as_index=False)["coverage"].mean()
    tests = paired_tests(data, args.species_labels)
    bins = coverage_bins(data)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for table, suffix in ((means, "mean_cov_val"), (tests, "test_cov_val"),
                          (data, "combined_alldata"), (bins, "0covbins_not_cumulative")):
        table.to_csv(f"{args.output_prefix}_{suffix}.csv", index=False)
    output = draw_plot(data, bins, args)
    print(f"Wrote {output} and four summary/data CSVs")


if __name__ == "__main__":
    main()
