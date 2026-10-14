# Publication figure scripts

Run scripts from the repository root with Python 3.10 or newer. Input paths
are supplied explicitly; relative input and output paths are resolved from the
working directory. Each script's module documentation and `--help` describe
its file formats, calculations, parameters and example command. Plotting does
not run probe design, alignment, mapping or evaluation pipelines.

## Figure-to-script list

Script filenames retain legacy panel numbering. The current manuscript uses
`figure4d_plot_annotations_pi_probe_coverage.py` for Supplementary Figure 3
and `figure4e_pt2_merge_species_pi_probe_scatter.py` for Figure 4.
Other panel labels below follow the script filenames; only figures with scripts
in this repository are listed.

| Figure or panel | Corresponding script(s) |
| --- | --- |
| Figure 3 | `figure3_plot_heatmap_pi_annotations_probes_pair.py` |
| Figure 4 | `figure4e_pt2_merge_species_pi_probe_scatter.py`; first generate per-species input CSVs with `figure4e_pt1_pi_vs_unique_prove_scatter.py` |
| Figure 4a (legacy panel label) | `figure4a_coverage_and_probe_counts.py` |
| Figure 4b–c (legacy panel labels) | `figure4b-c-jevsubsets.py` |
| Figure 5a–d | `figure5a-d_dilutions_Ct_plots_for_figures_TGplot.py` |
| Figure 5e | `figure5e_human_enrichment_plot.py` |
| Figure 5f | `figure5f_normalised_depth_plots.py` |
| Supplementary Figure 1 | `supp_fig1_probe_plot_onto_alignment.py` |
| Supplementary Figure 2 | `supp_fig2_ref_only_compareplots.py` |
| Supplementary Figure 3 | `figure4d_plot_annotations_pi_probe_coverage.py` |
| Supplementary Figure 5a | `supp_fig5a_probe_counts_with_representative.py` |
| Supplementary Figure 5d | `supp_fig5d_plot_catch_dampa_difference_decomposition.py`; uses per-species CSVs from `figure4e_pt1_pi_vs_unique_prove_scatter.py` |

## Dependencies

The collection uses numpy, pandas, matplotlib, seaborn (0.13 or newer), scipy,
biopython, scikit-allel, statsmodels and openpyxl. Individual scripts list their
own subset. These are dependency names, not a locked environment; record exact
versions with any published reproduction. For headless execution, set
`MPLBACKEND=Agg`.

## Shared helper modules

- `alignment_probe_helpers.py`: FASTA coordinate maps, non-gap occupancy,
  per-site diversity and distinct-probe coverage for Figures 4d and 4e.
- `heatmap_panel_helpers.py`: heatmap/diversity/probe preparation, annotation
  projection and label spacing for Figure 3.
- `plot_program_colors.py`: shared program colours for Figures 4d and 4e.

The helpers contain only calculation/layout functions and constants. They do
not parse command-line arguments, select a plotting backend, load local files
on import, or import any original plotting script. NumPy, pandas and biopython
are required by the calculation helpers; scikit-allel is loaded lazily by the
Figure 4d haplotype conversion. `plot_program_colors.py` uses matplotlib/numpy.

Manuscript-specific HXB2 annotations and legacy sequence-ID handling remain
explicitly documented. The two coordinate-map APIs and the different diversity
window conventions are retained to preserve existing figure calculations.

## Inputs and outputs

| Script | Required inputs | Outputs |
| --- | --- | --- |
| `supp_fig2_ref_only_compareplots.py` | Paired lists of reference/full-species per-genome coverage CSVs and dataset labels; optional genome-ID filters | Coverage distributions and bins, paired-test/mean/data CSVs |
| `supp_fig5a_probe_counts_with_representative.py` | Main and representative probe FASTAs in a results tree; optional explicit count baseline | Normalised probe-count PDF/PNG and raw-count CSV |
| `supp_fig5d_plot_catch_dampa_difference_decomposition.py` | Per-species window CSVs containing CATCH, DAMPA representative and DAMPA consensus | Contribution boxplots, paired-window and binned count/contribution CSVs |
| `supp_fig1_probe_plot_onto_alignment.py` | FASTA MSA, nine-column BLAST TSVs and labels, matching reference GFF3 and reference ID | Gene-annotation track above genome-coverage proportions, plotted-data CSV |
| `figure3_plot_heatmap_pi_annotations_probes_pair.py` | FASTA MSA, nine-column BLAST TSV, GFF3 and annotation reference ID for each of two datasets | Paired heatmap/diversity/annotation/probe figure, PNG and per-side CSVs |
| `figure4a_coverage_and_probe_counts.py` | Evaluation directory tree of per-genome coverage and probe-count summary CSV pairs | Coverage/probe-count PDF and PNG, two data CSVs |
| `figure4b-c-jevsubsets.py` | Replicate/step run folders containing probe FASTAs and GNU time reports | Probe-count, redundancy, runtime and memory figures, tidy CSV |
| `figure4d_plot_annotations_pi_probe_coverage.py` | FASTA MSA, nine-column BLAST TSVs and labels, GFF3 and annotation reference ID | Four-panel annotated coverage figure; optional caches |
| `figure4e_pt1_pi_vs_unique_prove_scatter.py` | FASTA MSA, nine-column BLAST TSVs and labels | Diversity/probe scatterplot, data CSV and regression summary |
| `figure4e_pt2_merge_species_pi_probe_scatter.py` | Figure 4e part 1 data CSVs and species labels | Combined scatterplot or binned plot, data and regression summaries |
| `figure5a-d_dilutions_Ct_plots_for_figures_TGplot.py` | Validation and mixed-series summary CSVs, Ct metadata workbook or CSV | Combined dilution/Ct regression figure |
| `figure5e_human_enrichment_plot.py` | Per-sample chromosome/position/depth TSVs and chromosome; optional BED | One median-depth/IQR figure |
| `figure5f_normalised_depth_plots.py` | PANDA and Twist CASTANET depth directories, reference FASTA and GFF3 | Normalised depth/annotation/boxplot figure and supporting plots/tables |

BLAST inputs use `6 qseqid sseqid pident qstart qend sstart send evalue bitscore`,
not the standard twelve-column outfmt 6. Subjects must match the ungapped
sequences of the supplied alignment. GFF3 uses 1-based inclusive reference
coordinates. The human-enrichment BED input uses 0-based end-exclusive
coordinates. Consult individual script documentation before preparing data.

Output files with existing names are overwritten. Publication species/program
ordering and cohort definitions remain in the relevant scripts to preserve the
original analyses; these are documented separately from input paths.
