#!/usr/bin/env python3
r"""A4-width, one-third-page EV-A / HIV-1 heatmap, pi, annotation and probe figure.

Inputs for both datasets are required and are prefixed with --left- or --right-. Shares calculations with the single-column
script. Writes a PDF, PNG and one data CSV per dataset.

Input contract and dependencies
-------------------------------
Python 3.9+, numpy, pandas, matplotlib and biopython. Keep
plot_heatmap_pi_annotations_probes.py alongside this script; it supplies the
shared calculations. Each side requires --*-alignment-file, --*-blast-file,
--*-gff-file and --*-strain-to-copy-annots. Titles are configurable.

BLAST inputs are headerless nine-column TSVs in this exact order:
qseqid sseqid pident qstart qend sstart send evalue bitscore (tab-separated),
corresponding to BLAST -outfmt "6 qseqid sseqid pident qstart qend sstart send
evalue bitscore". qseqid identifies a probe; sseqid identifies an ungapped
subject genome whose ID matches a FASTA alignment record. Coordinates are
1-based inclusive and may be reversed. Do not use the standard twelve-column
outfmt 6 layout. Each file represents one probe set, paired with one label.
The alignment is a nucleotide FASTA MSA with unique IDs and equal row lengths;
BLAST subjects must use the ungapped sequences in that alignment.

Each GFF3 uses 1-based inclusive coordinates of the selected alignment record's
ungapped sequence. Features/labels are configurable per side. The helper uses
fixed HXB2 intervals when the reference is B.FR.83.HXB2_LAI_IIIB_BRU.K03455;
this manuscript-specific override is retained for reproducibility. Other
references use their supplied GFF annotations.

Plots retain alignment columns meeting the occupancy threshold. The heatmap
shows rolling agreement with a non-gap consensus; pi uses sliding windows of
the full alignment; probe counts represent distinct query IDs with identity
>=80% and inclusive subject length >=81 nt. --window-size controls smoothing.
--output-file writes the figure plus PNG and *_left_data.csv/*_right_data.csv.
Existing files are overwritten. --font-family selects an installed typeface.

Example:
    python figure_scripts/figure3_plot_heatmap_pi_annotations_probes_pair.py \
        --left-alignment-file data/a.fasta --left-blast-file data/a.tsv \
        --left-gff-file data/a.gff3 --left-strain-to-copy-annots reference_a \
        --right-alignment-file data/b.fasta --right-blast-file data/b.tsv \
        --right-gff-file data/b.gff3 --right-strain-to-copy-annots reference_b \
        --output-file output/figure3.pdf
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from plot_heatmap_pi_annotations_probes import (
    HXB2, COLORS, prepare_panel, parse_panel_height_ratios,
    space_annotation_labels, plt, np, pd, Rectangle, MaxNLocator,
)
from matplotlib import font_manager
from matplotlib.text import Text

A4_WIDTH = 210 / 25.4
MAX_HEIGHT = 297 / 25.4 / 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {}
    for side in ('left', 'right'):
        values = dict(blast_file=None, alignment_file=None, gff_file=None,
                      strain_to_copy_annots=None, title=side.title(),
                      annotation_feature_types=['gene', 'CDS', 'mature_protein_region_of_CDS',
                                                'five_prime_UTR', 'three_prime_UTR'],
                      annotation_name_prefixes=['Name=', 'gene=', 'product='])
        defaults[side] = values
        for name, value in values.items():
            options = dict(default=value)
            if value is None:
                options = dict(required=True)
            elif isinstance(value, list):
                options['nargs'] = '+'
            parser.add_argument(f'--{side}-{name.replace("_", "-")}', **options)
    parser.add_argument('--output_file', '--output-file', default='output/figure3.pdf')
    parser.add_argument('--font-family', default='DejaVu Sans', help='Installed font family.')
    parser.add_argument('--width', type=float, default=A4_WIDTH, help='Inches, at most A4 width.')
    parser.add_argument('--height', type=float, default=MAX_HEIGHT, help='Inches, at most one-third A4 height.')
    parser.add_argument('--panel-height-ratios', type=parse_panel_height_ratios, default=(3, 1.7, 1.4, 1.7))
    parser.add_argument('--window_size', '--window-size', type=int, default=120)
    parser.add_argument('--min-non-gap-proportion', type=float, default=.2)
    parser.add_argument('--annotation-label-spacing', type=float, default=3)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if not 6 <= args.width <= A4_WIDTH or not 3 <= args.height <= MAX_HEIGHT:
        parser.error('Width must be 6–8.2677 inches; height must be 3–3.8976 inches.')
    if args.window_size < 1 or not 0 < args.min_non_gap_proportion <= 1 or args.dpi < 1:
        parser.error('Window and DPI must be positive; occupancy must be in (0, 1].')
    if not np.isfinite(args.annotation_label_spacing) or args.annotation_label_spacing < 0:
        parser.error('Annotation label spacing must be finite and nonnegative.')
    for side in defaults:
        if any(not prefix for prefix in getattr(args, f'{side}_annotation_name_prefixes')):
            parser.error('Annotation name prefixes cannot be empty.')
    return args, defaults


def draw_annotations(ax, data, reference):
    lanes, labels = [], []
    palette = plt.get_cmap('tab20').colors
    colors = {name: palette[i % len(palette)] for i, name in enumerate(
        dict.fromkeys(name for _, _, name in data['annotations']))}
    fixed_lanes = {'pol': 1, 'vpr': 1, 'tat(2)': 1, 'rev(1)': 2, 'rev(2)': 2, 'env-gp120': 1, "3'UTR": 1}
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    font = font_manager.FontProperties(family=plt.rcParams['font.family'], size=5)
    for start, end, name in data['annotations']:
        if reference == HXB2:
            lane = fixed_lanes.get(name, 0)
            while len(lanes) <= lane:
                lanes.append(0)
            lanes[lane] = max(lanes[lane], end)
        else:
            lane = next((i for i, last in enumerate(lanes) if start >= last), len(lanes))
            if lane == len(lanes):
                lanes.append(end)
            else:
                lanes[lane] = end
        color = COLORS.get(name.split('(')[0], '#aec7e8') if reference == HXB2 else colors[name]
        ax.add_patch(Rectangle((start - .5, lane), end - start, .6,
                              facecolor=color, edgecolor='black', linewidth=.4))
        label = name.split('(')[0] if reference == HXB2 else name
        width = renderer.get_text_width_height_descent(label, font, ismath=False)[0]
        feature_width = ax.transData.transform((end, 0))[0] - ax.transData.transform((start, 0))[0]
        outside = width + 4 > feature_width
        labels.append(ax.text((start + end - 1) / 2, lane + (.66 if outside else .3),
                              label, ha='center', va='bottom' if outside else 'center', fontsize=5))
    ax.set_ylim(-.08, max(1, len(lanes)) + .25)
    ax.set_axis_off()
    return labels


def main():
    args, defaults = parse_args()
    # Fail explicitly rather than silently substituting a different typeface.
    font_manager.findfont(font_manager.FontProperties(family=args.font_family), fallback_to_default=False)
    plt.rcParams.update({'font.family': args.font_family, 'font.size': 6, 'axes.titlesize': 7,
                         'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .5})
    fig = plt.figure(figsize=(args.width, args.height))
    grid = fig.add_gridspec(4, 2, left=.08, right=.985, bottom=.075, top=.95,
                           wspace=.22, hspace=.07, height_ratios=args.panel_height_ratios)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = None
    for col, side in enumerate(('left', 'right')):
        config = SimpleNamespace(**{name: getattr(args, f'{side}_{name}') for name in defaults[side]},
                                 window_size=args.window_size,
                                 min_non_gap_proportion=args.min_non_gap_proportion)
        data = prepare_panel(config)
        axes = [fig.add_subplot(grid[row, col]) for row in range(4)]
        hm, pa, aa, pc = axes
        keep = data['keep']
        x = np.arange(len(keep))
        for row, ax in enumerate(axes):
            ax.set_xlim(-.5, len(keep) - .5)
            pos = ax.get_position()
            if col == 0:
                fig.text(.01, pos.y1, 'ABCD'[row],
                         fontsize=8, fontweight='bold', va='top')
        image = hm.imshow(data['heat'], aspect='auto', interpolation='nearest', cmap='magma',
                          vmin=0, vmax=1, extent=[-.5, len(keep) - .5, len(data['ids']), 0], rasterized=True)
        hm.set_axis_off()
        hm.set_title(config.title, pad=3)
        if col == 0:
            heatmap_position = hm.get_position()
        pa.plot(x, data['pi'][keep], color='purple', linewidth=1.05)
        pa.axhline(.15, color='black', linestyle='--', linewidth=.75)
        finite = data['pi'][keep][np.isfinite(data['pi'][keep])]
        pa.set_ylim(0, max(.17, finite.max() * 1.1 if len(finite) else .17))
        pa.set_ylabel('Nucleotide\ndiversity (π)', labelpad=3)
        pa.tick_params(axis='x', bottom=False, labelbottom=False)
        labels = draw_annotations(aa, data, config.strain_to_copy_annots)
        space_annotation_labels(aa, labels, args.annotation_label_spacing)
        pc.plot(x, data['smoothed_counts'], color='#6666ff', linewidth=1.05)
        pc.set_ylim(0, max(1, data['smoothed_counts'].max() * 1.12))
        pc.set_ylabel('Unique probes', labelpad=3)
        pc.set_xlabel('Position in filtered alignment', labelpad=2)
        for ax in (pa, pc):
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=5, length=2, pad=1)
        pc.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        pd.DataFrame({'filtered_position': x, 'alignment_position_0based': keep,
                      'non_gap_proportion': data['occupancy'][keep], 'pi': data['pi'][keep],
                      'unique_probes': data['counts'][keep],
                      'unique_probes_smoothed': data['smoothed_counts']}).to_csv(
                          output.with_name(f'{output.stem}_{side}_data.csv'), index=False)
        print(f'{config.title}: {len(data["ids"])} genomes; {len(keep)}/{data["length"]} columns; '
              f'{data["matched"]} qualifying BLAST hits.')
    cb = fig.colorbar(image, cax=fig.add_axes([
        .057, heatmap_position.y0, .009, heatmap_position.height]), orientation='vertical')
    cb.set_ticks([0, .2, .4, .6, .8, 1])
    cb.ax.tick_params(labelsize=5, length=2, pad=1)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.set_label('Identity to consensus', fontsize=5, labelpad=2)
    cb.outline.set_visible(False)
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontsize() < 5:
            raise ValueError('Figure contains text smaller than 5 points.')
    fig.savefig(output, dpi=args.dpi, facecolor='white')
    if output.suffix.lower() != '.png':
        fig.savefig(output.with_suffix('.png'), dpi=args.dpi, facecolor='white')
    plt.close(fig)
    print(f'Saved {output} ({args.width * 25.4:.1f} × {args.height * 25.4:.1f} mm).')


if __name__ == '__main__':
    main()
