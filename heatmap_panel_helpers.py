"""Prepare heatmap, diversity, annotation and probe data for Figure 3.

Focused helpers copied from plot_heatmap_pi_annotations_probes.py, without its
CLI, figure-authoring entry point, plotting imports or backend selection.
Dependencies: numpy, pandas and biopython. Annotation label spacing accepts
existing Matplotlib axes/text objects but does not create figures.

prepare_panel(args) expects attributes alignment_file, blast_file, gff_file,
strain_to_copy_annots, annotation_feature_types, annotation_name_prefixes,
window_size and min_non_gap_proportion. It returns full-alignment arrays plus
retained 0-based column indices; annotations use 0-based half-open coordinates
in the retained alignment. It resolves and updates strain_to_copy_annots.

Inputs are aligned nucleotide FASTA, nine-column BLAST TSV, and a GFF3 matching
the chosen ungapped reference. See Figure 3's --help for the input contract.
The manuscript HXB2 annotation override and sequence-ID alias behaviour are
preserved explicitly for reproducibility. Probe hits require >=80% identity
and inclusive length >80 nt. The pi window is computed on the full alignment;
heatmap and probe-count smoothing use retained columns. Do not interchange
these coordinate/window conventions with the Figure 4e block summaries.
"""

import argparse
import warnings

from Bio import AlignIO
import numpy as np
import pandas as pd

HXB2 = "B.FR.83.HXB2_LAI_IIIB_BRU.K03455"


HXB2_ANNOTATIONS = [
    (0, 634, "5'UTR"), (790, 2292, "gag"), (2085, 5096, "pol"),
    (5041, 5619, "vif"), (5559, 5850, "vpr"),
    (5831, 6045, "tat(1)"), (5970, 6045, "rev(1)"),
    (6062, 6310, "vpu"), (6225, 7758, "env-gp120"),
    (7759, 8795, "env-gp41"), (8379, 8424, "tat(2)"),
    (8379, 8653, "rev(2)"), (8797, 9417, "nef"),
    (9086, 9718, "3'UTR"),
]


COLORS = {"5'UTR": "#1f77b4", "3'UTR": "#1f77b4", "gag": "#aec7e8",
          "pol": "#ff7f0e", "vif": "#ffbb78", "vpr": "#2ca02c",
          "tat": "#98df8a", "rev": "#d62728", "vpu": "#ff9896",
          "env-gp120": "#9467bd", "env-gp41": "#c5b0d5", "nef": "#8c564b"}


def parse_panel_height_ratios(value):
    """Parse four finite positive ratios in plot order."""
    try:
        ratios = tuple(float(part.strip()) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Panel height ratios must be four positive numbers separated by commas.') from error
    if len(ratios) != 4 or any(not np.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise argparse.ArgumentTypeError('Panel height ratios must be four finite positive numbers separated by commas.')
    return ratios


def annotation_name_from_attributes(attribute_text, prefixes, fallback):
    """Return the first attribute matching the requested prefix order."""
    items = [item.strip() for item in attribute_text.split(';')]
    for prefix in prefixes:
        prefix = prefix if prefix.endswith('=') else prefix + '='
        for item in items:
            if item.startswith(prefix):
                value = item[len(prefix):].strip()
                if value:
                    return value
    return fallback


def load_annotations(gff, reference, feature_types, name_prefixes):
    if reference == HXB2:
        # The supplied GFF uses NC_001802 coordinates, which differ from HXB2.
        # Preserve the source script's HXB2-specific annotation override.
        return HXB2_ANNOTATIONS
    result = []
    accepted_types = set(feature_types)
    seen = set()
    with open(gff, encoding='utf-8') as handle:
        for lineno, line in enumerate(handle, 1):
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9 or fields[2] not in accepted_types:
                continue
            name = annotation_name_from_attributes(fields[8], name_prefixes, fields[2])
            try:
                # GFF uses 1-based inclusive coordinates; slices are 0-based half-open.
                annotation = (int(fields[3]) - 1, int(fields[4]), name)
            except ValueError as error:
                raise ValueError(f'Invalid coordinates on GFF line {lineno}: {line.rstrip()}') from error
            if annotation not in seen:
                result.append(annotation)
                seen.add(annotation)
    if not result:
        raise ValueError('No annotations were found for --annotation-feature-types')
    return sorted(result)


def sliding_pi(seqs, window):
    """Same unbiased per-site pairwise difference as allel, then window mean."""
    counts = np.stack([(seqs == base).sum(axis=0) for base in 'ACGT'])
    n = counts.sum(axis=0).astype(float)
    denominator = n * (n - 1)
    site_pi = np.full(n.shape, np.nan)
    np.divide(n * n - (counts.astype(float) ** 2).sum(axis=0), denominator,
              out=site_pi, where=denominator > 0)
    pi = np.full(n.shape, np.nan)
    if window <= len(n):
        valid = np.isfinite(site_pi)
        sums = np.r_[0., np.cumsum(np.where(valid, site_pi, 0.))]
        sizes = np.r_[0, np.cumsum(valid)]
        total = sums[window:] - sums[:-window]
        number = sizes[window:] - sizes[:-window]
        means = np.full(total.shape, np.nan)
        np.divide(total, number, out=means, where=number > 0)
        pi[np.arange(len(means)) + window // 2] = means
    return pi


def normalize_alignment_id(name):
    """Unify separators used in enterovirus FASTA and BLAST names."""
    return name.replace('-', '_').replace('~', '_')


def space_annotation_labels(ax, labels, spacing):
    """Spread labels horizontally where their rendered vertical extents overlap."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    gap = spacing * ax.figure.dpi / 72
    padding = ax.figure.dpi / 72
    left, right = ax.bbox.x0 + padding, ax.bbox.x1 - padding
    boxes = [(label, label.get_window_extent(renderer)) for label in labels]
    # Connected vertical bands must be packed together, including labels above
    # short features that may overlap text in the next annotation lane.
    bands = []
    for label, box in sorted(boxes, key=lambda item: item[1].y0):
        if not bands or box.y0 > bands[-1][0]:
            bands.append([box.y1, [(label, box)]])
        else:
            bands[-1][0] = max(bands[-1][0], box.y1)
            bands[-1][1].append((label, box))
    for _, band in bands:
        band.sort(key=lambda item: item[1].x0 + item[1].width / 2)
        widths = [box.width for _, box in band]
        if sum(widths) + gap * (len(band) - 1) > right - left:
            warnings.warn('Annotation labels exceed panel width; increase --width '
                          'or select fewer features/shorter annotation names.', stacklevel=2)
            continue
        # Forward and backward passes preserve order and move text only as
        # needed to satisfy the gap and panel boundaries.
        positions = []
        edge = left
        for (_, box), width in zip(band, widths):
            position = max(edge, min(box.x0, right - width))
            positions.append(position)
            edge = position + width + gap
        edge = right
        for i in range(len(band) - 1, -1, -1):
            positions[i] = min(positions[i], edge - widths[i])
            edge = positions[i] - gap
        for (label, _), position, width in zip(band, positions, widths):
            x, y = label.get_position()
            label.set_x(ax.transData.inverted().transform(
                (position + width / 2, ax.transData.transform((x, y))[1]))[0])


def resolve_alignment_id(name, ids, aliases):
    """Prefer exact IDs, then tolerate hyphen/underscore/tilde differences."""
    if name in ids:
        return name
    candidates = aliases.get(normalize_alignment_id(name), [])
    if len(candidates) > 1:
        raise ValueError(f'Ambiguous alignment ID {name!r}: {candidates}')
    return candidates[0] if candidates else None


def prepare_panel(args):
    """Calculate panel data shared by the single- and two-column figures."""
    aln = AlignIO.read(args.alignment_file, 'fasta')
    ids = [record.id for record in aln]
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('Alignment requires at least two sequences with unique IDs.')
    id_set = set(ids)
    aliases = {}
    for name in ids:
        aliases.setdefault(normalize_alignment_id(name), []).append(name)
    reference = resolve_alignment_id(args.strain_to_copy_annots, id_set, aliases)
    if reference is None:
        raise ValueError('Annotation reference is absent from alignment.')
    args.strain_to_copy_annots = reference
    seqs = np.array([list(str(record.seq).upper()) for record in aln])
    length = seqs.shape[1]
    occupancy = (seqs != '-').mean(axis=0)
    keep = np.flatnonzero(occupancy >= args.min_non_gap_proportion)
    if not len(keep):
        raise ValueError('No alignment positions passed the occupancy filter.')
    # Ungapped subject coordinate -> original alignment coordinate.
    maps = {name: np.flatnonzero(row != '-') for name, row in zip(ids, seqs)}
    probe_sets = [set() for _ in range(length)]
    matched = 0
    with open(args.blast_file) as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.rstrip().split('\t')
            if len(fields) != 9:
                raise ValueError(f'BLAST line {lineno}: expected nine columns, got {len(fields)}.')
            probe, subject = fields[:2]
            start, end = sorted((int(fields[5]), int(fields[6])))
            if float(fields[2]) < 80 or end - start + 1 <= 80:
                continue
            subject = resolve_alignment_id(subject, id_set, aliases)
            if subject is None:
                continue
            if start < 1 or end > len(maps[subject]):
                raise ValueError(f'BLAST line {lineno}: coordinates exceed ungapped subject length.')
            matched += 1
            for pos in maps[subject][start - 1:end]:
                probe_sets[pos].add(probe)
    if not matched:
        raise ValueError('No qualifying BLAST hits matched the alignment IDs.')
    counts = np.array([len(probes) for probes in probe_sets])
    pi = sliding_pi(seqs, args.window_size)
    # Source heatmap compares each sequence with a consensus base. Use the
    # most frequent non-gap base for reproducibility (source randomly selects).
    bases = np.unique(seqs[seqs != '-'])
    frequencies = np.stack([(seqs == base).sum(axis=0) for base in bases])
    consensus = bases[frequencies.argmax(axis=0)]
    heat = pd.DataFrame((seqs[:, keep] == consensus[keep]).astype(float)).T.rolling(
        args.window_size, min_periods=1, center=True).mean().T.to_numpy()
    smoothed_counts = pd.Series(counts[keep]).rolling(
        args.window_size, min_periods=1, center=True).mean().to_numpy()
    refmap = maps[args.strain_to_copy_annots]
    annotations = []
    for start, end, name in load_annotations(
            args.gff_file, args.strain_to_copy_annots,
            args.annotation_feature_types, args.annotation_name_prefixes):
        original = refmap[max(0, start):min(end, len(refmap))]
        retained = np.intersect1d(original, keep)
        if len(retained):
            annotations.append((np.searchsorted(keep, retained[0]),
                                np.searchsorted(keep, retained[-1]) + 1, name))

    return dict(ids=ids, keep=keep, length=length, occupancy=occupancy,
                counts=counts, pi=pi, heat=heat, smoothed_counts=smoothed_counts,
                annotations=annotations, matched=matched)
