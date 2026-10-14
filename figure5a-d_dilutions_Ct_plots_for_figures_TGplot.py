#!/usr/bin/env python3
r"""Plot the publication's validation dilution/Ct regression panels.

Dependencies: Python 3.10+, numpy, pandas, matplotlib, statsmodels and openpyxl
(for Excel metadata). No files are read and no plots are created on import.

Inputs
------
--validation-summary: headered CASTANET summary CSV, one row per sample/target,
with sampleid, probetype, Ct and n_reads_dedup columns. The publication selects
enterovirus_a/POS-EV-002-0*, enterococcus_faecium/POS-BACTI-005-0* and
streptococcus_pneumoniae/POS-IPD-006-0* for the dilution panels. Ct values are
measured PCR cycle thresholds, not dilution factors. n_reads_dedup is the raw
non-negative deduplicated read count. Sample IDs may have an underscore and
sequencing suffix; the prefix before the first underscore is the metadata key.

--mixed-summary: CSV with the same columns for the mixed dilution series.
Rows whose probetype contains sarscov2 (case-insensitive) are summed per
sampleid for Ct and n_reads_dedup, matching the original panel calculation.
This assumes Ct contributions are partitioned appropriately across these
rows; do not repeat the same Ct value across rows to be summed. Samples with
MIXED in the sample-ID prefix are selected for this regression.

--metadata: Excel workbook (sheet selected by --metadata-sheet) or CSV, with
Organism, Specimen Name and newCt columns. --ct-column can select a different
Ct field. Rows without Organism or with Ct equal to 'not detected' are removed.
Specimen Name must match summary sample-ID prefixes and be unique among the
selected rows. Metadata Ct replaces summary Ct for CMV and the additional
enterovirus observations. Selection presets are retained to reproduce the
publication; this script does not infer new experimental cohorts.

Calculations and outputs
------------------------
Regressions fit ordinary least squares to log10(n_reads_dedup) versus Ct,
with an intercept, 95% confidence intervals for the fitted mean, and 95%
prediction intervals. Rows with missing/non-numeric/non-finite Ct or count,
or zero counts, are omitted; negative counts are rejected. Regressions require
at least three observations and two distinct Ct values. The y-axis displays
read counts on a log10-transformed scale.

One figure is saved to --output: four dilution regressions and a CMV Ct panel.
Additional non-dilution enterovirus observations are shown in red on the
enterovirus dilution panel, without fitting a second regression. The sixth
subplot is hidden, matching the original layout. Existing output is overwritten.

Example::

    python figure_scripts/figure5a-d_dilutions_Ct_plots_for_figures_TGplot.py \
        --validation-summary data/validation.csv --mixed-summary data/mixed.csv \
        --metadata data/metadata.xlsx --output output/figure5ad.pdf
"""

import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import statsmodels.api as sm


SUMMARY_COLUMNS = {"sampleid", "probetype", "Ct", "n_reads_dedup"}


def read_summary(path):
    """Read a CASTANET export and validate the fields used in this figure."""
    data = pd.read_csv(path, dtype={"sampleid": str, "probetype": str})
    missing = SUMMARY_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for column in ("Ct", "n_reads_dedup"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if (data["n_reads_dedup"] < 0).any():
        raise ValueError(f"Negative deduplicated read counts in {path}")
    return data


def select_cohort(data, target, sample_prefix):
    """Select the original target/cohort and remove sequencing suffixes."""
    selected = data.loc[data["probetype"].eq(target)].copy()
    selected["sampleid"] = selected["sampleid"].str.split("_").str[0]
    return selected.loc[
        selected["sampleid"].str.contains(sample_prefix, regex=False, na=False)
    ].copy()


def prepare_regression(data):
    """Create finite Ct and log-count pairs, omitting undetected observations."""
    values = data[["Ct", "n_reads_dedup"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values).all(axis=1) & values["n_reads_dedup"].gt(0)
    values = values.loc[valid].copy().sort_values("Ct")
    values["log_reads"] = np.log10(values["n_reads_dedup"])
    return values


def plot_regression(data, axis, title):
    """Plot OLS fit, mean confidence band and observation prediction band."""
    values = prepare_regression(data)
    if len(values) < 3 or values["Ct"].nunique() < 2:
        raise ValueError(f"{title}: need at least three finite positive-count observations and two Ct values")
    model = sm.OLS(values["log_reads"], sm.add_constant(values["Ct"])).fit()
    prediction = model.get_prediction().summary_frame(alpha=0.05)
    p = model.pvalues["Ct"]
    p_text = f"p<{p:.0e}" if p < 0.001 else f"p={p:.3f}"
    x = values["Ct"].to_numpy()
    axis.scatter(x, values["log_reads"], color="black", s=25)
    axis.plot(x, prediction["mean"], "--", color="teal",
              label=f"R²={model.rsquared:.2f}, {p_text}")
    axis.fill_between(x, prediction["obs_ci_lower"].to_numpy(),
                      prediction["obs_ci_upper"].to_numpy(), color="#82cbb2",
                      alpha=0.2, linewidth=0, label="95% prediction interval")
    axis.fill_between(x, prediction["mean_ci_lower"].to_numpy(),
                      prediction["mean_ci_upper"].to_numpy(), color="teal",
                      alpha=0.2, linewidth=0, label="95% CI")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{10**y:,.0f}" if y >= 0 else "")
    )
    axis.set_xlabel("Ct")
    axis.set_ylabel("Deduplicated reads")
    axis.legend(title=title, fontsize=7)
    print(f"{title}: fitted {len(values)} of {len(data)} observations")


def merge_metadata(summary, metadata, target, sample_prefix, ct_column, exclude=None):
    """Join measured Ct by sample prefix without duplicating observations."""
    cohort = select_cohort(summary, target, sample_prefix)
    if exclude:
        cohort = cohort.loc[~cohort["sampleid"].str.contains(exclude, regex=False)].copy()
    selected = metadata.loc[
        metadata["Organism"].notna()
        & metadata["Specimen Name"].str.contains(sample_prefix, regex=False, na=False)
        & metadata[ct_column].astype(str).str.lower().ne("not detected")
    ].copy()
    selected[ct_column] = pd.to_numeric(selected[ct_column], errors="coerce")
    selected = selected[["Specimen Name", ct_column]].rename(columns={ct_column: "metadata_ct"})
    merged = cohort.merge(selected, left_on="sampleid", right_on="Specimen Name",
                          how="left", validate="many_to_one")
    merged["Ct"] = merged["metadata_ct"]
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--mixed-summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="Headered CSV or Excel workbook.")
    parser.add_argument("--metadata-sheet", default="Sheet1", help="Excel worksheet (default: Sheet1).")
    parser.add_argument("--ct-column", default="newCt", help="Metadata Ct column (default: newCt).")
    parser.add_argument("--output", type=Path, default=Path("output/figure5ad.pdf"))
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in
                                (args.validation_summary, args.mixed_summary, args.metadata)}:
        parser.error("Output must not overwrite an input")
    validation = read_summary(args.validation_summary)
    mixed = read_summary(args.mixed_summary)
    metadata = (pd.read_csv(args.metadata) if args.metadata.suffix.lower() == ".csv"
                else pd.read_excel(args.metadata, sheet_name=args.metadata_sheet))
    required = {"Organism", "Specimen Name", args.ct_column}
    if not required.issubset(metadata.columns):
        parser.error(f"Metadata is missing columns: {sorted(required - set(metadata.columns))}")

    mask = mixed["probetype"].str.contains("sarscov2", case=False, na=False)
    combined = mixed.loc[mask].groupby("sampleid", as_index=False)[["Ct", "n_reads_dedup"]].sum(min_count=1)
    combined["probetype"] = "sarscov2"
    cohorts = [
        (select_cohort(validation, "enterovirus_a", "POS-EV-002-0"), "Enterovirus A dilution"),
        (select_cohort(combined, "sarscov2", "MIXED"), "SARS-CoV-2 dilution"),
        (select_cohort(validation, "enterococcus_faecium", "POS-BACTI-005-0"), "E. faecium dilution"),
        (select_cohort(validation, "streptococcus_pneumoniae", "POS-IPD-006-0"), "S. pneumoniae dilution"),
        (merge_metadata(validation, metadata, "cytomegallovirus", "POS-CMV", args.ct_column), "CMV Ct"),
    ]
    additional = prepare_regression(merge_metadata(
        validation, metadata, "enterovirus_a", "POS-EV", args.ct_column, exclude="POS-EV-002-0"
    ))
    with mpl.rc_context({"pdf.fonttype": 42, "font.size": 8}):
        fig, axes = plt.subplots(3, 2, figsize=(8, 9))
        for axis, (cohort, title) in zip(axes.flat, cohorts):
            plot_regression(cohort, axis, title)
        axes.flat[0].scatter(additional["Ct"], additional["log_reads"], color="red", alpha=0.8)
        axes.flat[5].set_visible(False)
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
