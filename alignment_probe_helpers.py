"""Alignment coordinates, distinct-probe coverage and nucleotide diversity.

Shared calculation functions extracted from the original plotting scripts.
This module has no CLI, plotting side effects, cached files or local paths.
Dependencies: numpy, pandas and biopython; scikit-allel is loaded only by
read_fasta_to_haplotype_array(), used by Figure 4d.

Coordinate conventions
----------------------
BLAST inputs use nine headerless TSV columns in this order:
qseqid sseqid pident qstart qend sstart send evalue bitscore.
Subject coordinates are 1-based inclusive on the ungapped genome. All returned
alignment columns are 0-based. Two map representations are intentionally kept
because the source figures use different filtering and cache conventions:

* load_alignment(): arrays mapping every ungapped position to its original
  alignment column, before occupancy filtering (Figure 4e).
* get_position_maps(): dictionaries mapping only retained positions, still in
  original alignment coordinates (Figure 4d and its legacy caches).

unique_probe_counts_with_positions() unions each query ID across all subject
hits; a probe is counted once per alignment column. It reports/clips subject
bounds as in the source implementation. get_probedepth_for_plot() preserves
Figure 4d's original >=80% identity / >80-nt subject-length filters and smooths
across the full alignment. Its unused arguments remain for compatibility.

Pi uses A/C/G/T only and the finite-sample correction n/(n-1); columns with
fewer than two called bases return NaN. These functions calculate data only;
figure scripts retain control of filtering, windows, export and presentation.
"""

from __future__ import annotations

from collections import defaultdict

from Bio import AlignIO, SeqIO
import numpy as np
import pandas as pd

BLAST_COLUMNS = ["qseqid", "sseqid", "pident", "sstart", "send"]


def load_alignment(
    alignment_file: str, reference_id: str
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Return genome-position maps, non-gap proportions, and alignment length."""
    records = list(SeqIO.parse(alignment_file, "fasta"))
    if not records:
        raise ValueError("The alignment contains no FASTA records")
    alignment_length = len(records[0].seq)
    if any(len(record.seq) != alignment_length for record in records):
        raise ValueError("FASTA records do not all have the same aligned length")

    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("The alignment contains duplicate FASTA record IDs")
    if reference_id not in set(ids):
        examples = ", ".join(ids[:3])
        raise ValueError(
            f"Reference ID {reference_id!r} is not in the alignment "
            f"(first IDs: {examples})"
        )

    non_gap_counts = np.zeros(alignment_length, dtype=np.int32)
    position_maps = {}
    for record in records:
        is_non_gap = np.fromiter(
            (base != "-" for base in str(record.seq)),
            dtype=bool,
            count=alignment_length,
        )
        non_gap_counts += is_non_gap
        # Array index is the zero-based ungapped genome position; value is the
        # corresponding zero-based alignment column.
        position_maps[record.id] = np.flatnonzero(is_non_gap).astype(np.int32)

    return position_maps, non_gap_counts / len(records), alignment_length


def unique_probe_counts_with_positions(
    blast_file: str,
    position_maps: dict[str, np.ndarray],
    alignment_length: int,
    min_pident: float,
    min_hit_length: int,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, set[int]], dict[str, int]]:
    """Count qseqids per alignment column and retain each qseqid's positions."""
    probe_positions: dict[str, set[int]] = defaultdict(set)
    stats = {"rows": 0, "passing": 0, "unmatched_genome": 0, "clipped": 0}

    chunks = pd.read_csv(
        blast_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 5, 6],
        names=BLAST_COLUMNS,
        chunksize=chunk_size,
    )
    for chunk in chunks:
        stats["rows"] += len(chunk)
        numeric = chunk[["pident", "sstart", "send"]].apply(pd.to_numeric, errors="coerce")
        chunk[["pident", "sstart", "send"]] = numeric
        chunk = chunk.dropna(subset=["qseqid", "sseqid", "pident", "sstart", "send"])
        lengths = (chunk["send"] - chunk["sstart"]).abs() + 1
        chunk = chunk[(chunk["pident"] >= min_pident) & (lengths >= min_hit_length)]
        stats["passing"] += len(chunk)

        for row in chunk.itertuples(index=False):
            position_map = position_maps.get(str(row.sseqid))
            if position_map is None:
                stats["unmatched_genome"] += 1
                continue
            start = int(min(row.sstart, row.send)) - 1
            end = int(max(row.sstart, row.send))  # exclusive zero-based bound
            clipped_start = max(0, start)
            clipped_end = min(len(position_map), end)
            if clipped_start != start or clipped_end != end:
                stats["clipped"] += 1
            if clipped_start >= clipped_end:
                continue
            probe_positions[str(row.qseqid)].update(
                position_map[clipped_start:clipped_end].tolist()
            )

    counts = np.zeros(alignment_length, dtype=np.int32)
    for positions in probe_positions.values():
        if positions:
            counts[np.fromiter(positions, dtype=np.int32)] += 1
    stats["unique_probes"] = len(probe_positions)
    return counts, probe_positions, stats


def calculate_site_pi(alignment_file: str) -> np.ndarray:
    """Calculate unbiased nucleotide diversity at every alignment column."""
    records = list(SeqIO.parse(alignment_file, "fasta"))
    if not records:
        raise ValueError("The alignment contains no FASTA records")
    sequences = np.asarray(
        [np.frombuffer(str(record.seq).upper().encode("ascii"), dtype="S1") for record in records]
    )
    if sequences.ndim != 2:
        raise ValueError("FASTA records do not all have the same aligned length")

    allele_counts = np.vstack([(sequences == base).sum(axis=0) for base in (b"A", b"C", b"G", b"T")])
    sample_count = allele_counts.sum(axis=0).astype(float)
    sum_squared_counts = np.square(allele_counts, dtype=float).sum(axis=0)
    pi = np.full(sequences.shape[1], np.nan, dtype=float)
    valid = sample_count >= 2
    # Equivalent to n/(n-1) * (1 - sum(p_k**2)), written to avoid
    # constructing a separate frequency matrix.
    pi[valid] = (
        sample_count[valid] / (sample_count[valid] - 1.0)
        * (1.0 - sum_squared_counts[valid] / np.square(sample_count[valid]))
    )
    return pi


def read_fasta_to_haplotype_array(fasta_file):
    """Encode A/C/G/T as 0..3 and other symbols as -1 in an allel haplotype array."""
    import allel

    # Load alignment
    alignment = AlignIO.read(fasta_file, 'fasta')
    seqs = np.array([list(rec.seq.upper()) for rec in alignment])  # shape: (n_samples, n_sites)

    # Define mapping: ACGT → 0,1,2,3; everything else → -1 (missing)
    base_map = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'N': -1, '-': -1, '?': -1, '.': -1,
        'R': -1, 'Y': -1, 'S': -1, 'W': -1,
        'K': -1, 'M': -1, 'B': -1, 'D': -1,
        'H': -1, 'V': -1, 'X': -1
    }

    def map_base(b): return base_map.get(b, -1)
    geno = np.vectorize(map_base)(seqs)  # shape: (n_samples, n_sites)

    # Transpose to (n_sites, n_samples) for haplotype array
    hap = allel.HaplotypeArray(geno.T)
    return hap


def get_position_maps(alignment,alignment_pos_tokeep):
    """Map zero-based ungapped coordinates to retained original alignment columns."""
    # Create position maps for each genome in the alignment
    position_maps = {}
    for record in alignment:
        genome_seq = str(record.seq)
        alignment_pos = 0
        genome_pos = 0
        position_map = {}
        for base in genome_seq:
            if alignment_pos not in alignment_pos_tokeep:
                if base != '-':  # Non-gap in genome
                    genome_pos += 1
            else:
                if base != '-':  # Non-gap in genome
                    position_map[genome_pos] = alignment_pos
                    genome_pos += 1
            alignment_pos += 1

        position_maps[record.id] = position_map
    return position_maps


def get_non_gap_proportion(alignment,alignment_length):
    """Return the fraction of non-gap genomes at each original alignment column."""
    non_gap_proportion = np.zeros(alignment_length)
    for pos in range(alignment_length):
        column_bases = [record.seq[pos] for record in alignment]
        non_gap_count = sum(1 for base in column_bases if base != '-')
        non_gap_proportion[pos] = non_gap_count / len(column_bases)
    return non_gap_proportion


def get_probedepth_for_plot(
    blast_file, window_size, alignment_length, non_gap_proportion, alignment,
    alignment_pos_tokeep, position_maps, gapthresh=0.1,
):
    """Return centred rolling and raw distinct-probe counts for Figure 4d.

    ``position_maps`` determines retained coordinates; the occupancy/alignment
    arguments are retained for source API compatibility. Subject hits must have
    >=80% identity and inclusive length >80 nt. Unknown subjects or unmapped
    coordinates are ignored, matching the original implementation.
    """
    blast_df = pd.read_csv(
        blast_file, sep="\t", header=None,
        names=["qseqid", "sseqid", "pident", "qstart", "qend", "sstart", "send", "evalue", "bitscore"],
    )
    blast_df = blast_df[blast_df["pident"] >= 80].copy()
    unique_probes_per_position = {position: set() for position in range(alignment_length)}
    for _, row in blast_df.iterrows():
        genome_id = row["sseqid"]
        if genome_id not in position_maps:
            continue
        position_map = position_maps[genome_id]
        start, end = int(row["sstart"]), int(row["send"])
        if abs(end - start) + 1 <= 80:
            continue
        if start > end:
            start, end = end, start
        for position in range(start - 1, end):
            if position in position_map:
                unique_probes_per_position[position_map[position]].add(row["qseqid"])
    counts = pd.Series(np.array([
        len(unique_probes_per_position[position]) for position in range(alignment_length)
    ]))
    smoothed = counts.rolling(window=window_size, min_periods=1, center=True).mean()
    return smoothed, counts
