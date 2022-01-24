# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

from .aa import DNA_TO_AA
from .align import NEXTCLADE_DATA

REFERENCE_SEQ = None  # loaded lazily

# Adapted from https://github.com/nextstrain/ncov/blob/50ceffa/defaults/annotation.gff
# Note these are 1-based positions
annotation_tsv = """\
seqname	source	feature	start	end	score	strand	frame	attribute
.	.	gene	26245	26472	.	+	.	 gene_name "E"
.	.	gene	26523	27191	.	+	.	 gene_name "M"
.	.	gene	28274	29533	.	+	.	 gene_name "N"
.	.	gene	29558	29674	.	+	.	 gene_name "ORF10"
.	.	gene	28734	28955	.	+	.	 gene_name "ORF14"
.	.	gene	266	13468	.	+	.	 gene_name "ORF1a"
.	.	gene	13468	21555	.	+	.	 gene_name "ORF1b"
.	.	gene	25393	26220	.	+	.	 gene_name "ORF3a"
.	.	gene	27202	27387	.	+	.	 gene_name "ORF6"
.	.	gene	27394	27759	.	+	.	 gene_name "ORF7a"
.	.	gene	27756	27887	.	+	.	 gene_name "ORF7b"
.	.	gene	27894	28259	.	+	.	 gene_name "ORF8"
.	.	gene	28284	28577	.	+	.	 gene_name "ORF9b"
.	.	gene	21563	25384	.	+	.	 gene_name "S"
"""


def _():
    genes = []
    rows = annotation_tsv.split("\n")
    header, rows = rows[0].split("\t"), rows[1:]
    for row in rows:
        if row:
            row = dict(zip(header, row.split("\t")))
            gene_name = row["attribute"].split('"')[1]
            start = int(row["start"])
            end = int(row["end"])
            genes.append(((start, end), gene_name))
    genes.sort()
    return OrderedDict((gene_name, pos) for pos, gene_name in genes)


# This maps gene name to the nucleotide position in the genome,
# as measured in the original Wuhan virus.
GENE_TO_POSITION: Dict[str, Tuple[int, int]] = _()

# This maps gene name to a set of regions in that gene.
# These regions may be used in plotting e.g. mutrans.ipynb.
# Each region has a string label and an extent (start, end)
# measured in amino acid positions relative to the start.
GENE_STRUCTURE: Dict[str, Dict[str, Tuple[int, int]]] = {
    # Source: https://www.nature.com/articles/s41401-020-0485-4/figures/2
    "S": {
        "NTD": (13, 305),
        "RBD": (319, 541),
        "FC": (682, 685),
        "FP": (788, 806),
        "HR1": (912, 984),
        "HR2": (1163, 1213),
        "TM": (1213, 1237),
        "CT": (1237, 1273),
    },
    # Source https://www.nature.com/articles/s41467-021-21953-3
    "N": {
        "NTD": (1, 49),
        "RNA binding": (50, 174),
        "SR": (175, 215),
        "dimerization": (246, 365),
        "CTD": (365, 419),
        # "immunogenic": (133, 217),
    },
    # Source: https://www.ncbi.nlm.nih.gov/protein/YP_009725295.1
    "ORF1a": {
        "nsp1": (0, 180),  # leader protein
        "nsp2": (180, 818),
        "nsp3": (818, 2763),
        "nsp4": (2763, 3263),
        "nsp5": (3263, 3569),  # 3C-like proteinase
        "nsp6": (3569, 3859),
        "nsp7": (3859, 3942),
        "nsp8": (3942, 4140),
        "nsp9": (4140, 4253),
        "nsp10": (4253, 4392),
        "nsp11": (4392, 4405),
    },
    # Source: https://www.ncbi.nlm.nih.gov/protein/1796318597
    "ORF1ab": {
        "nsp1": (0, 180),  # leader protein
        "nsp2": (180, 818),
        "nsp3": (818, 2763),
        "nsp4": (2763, 3263),
        "nsp5": (3263, 3569),  # 3C-like proteinase
        "nsp6": (3569, 3859),
        "nsp7": (3859, 3942),
        "nsp8": (3942, 4140),
        "nsp9": (4140, 4253),
        "nsp10": (4253, 4392),
        "nsp12": (4392, 5324),  # RNA-dependent RNA polymerase
        "nsp13": (5324, 5925),  # helicase
        "nsp14": (5925, 6452),  # 3'-to-5' exonuclease
        "nsp15": (6452, 6798),  # endoRNAse
        "nsp16": (6798, 7096),  # 2'-O-ribose methyltransferase
    },
    # Source: see infer_ORF1b_structure() below.
    "ORF1b": {
        "nsp12": (0, 924),  # RNA-dependent RNA polymerase
        "nsp13": (924, 1525),  # helicase
        "nsp14": (1525, 2052),  # 3'-to-5' exonuclease
        "nsp15": (2052, 2398),  # endoRNAse
        "nsp16": (2398, 2696),  # 2'-O-ribose methyltransferase
    },
}


def load_gene_structure(filename=None, gene_name=None):
    """
    Loads structure from a GenPept file for use in GENE_STRUCTURE.
    This is used only when updating the static GENE_STRUCTURE dict.
    """
    from Bio import SeqIO

    if filename is None:
        assert gene_name is not None
        filename = f"data/{gene_name}.gp"
    result = {}
    with open(filename) as f:
        for record in SeqIO.parse(f, format="genbank"):
            for feature in record.features:
                if feature.type == "mat_peptide":
                    product = feature.qualifiers["product"][0]
                    assert isinstance(product, str)
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    result[product] = start, end
    return result


def infer_ORF1b_structure():
    """
    Infers approximate ORF1b structure from ORF1ab.
    This is used only when updating the static GENE_STRUCTURE dict.
    """
    ORF1a_start = GENE_TO_POSITION["ORF1a"][0]
    ORF1b_start = GENE_TO_POSITION["ORF1b"][0]
    shift = (ORF1b_start - ORF1a_start) // 3
    result = {}
    for name, (start, end) in GENE_STRUCTURE["ORF1ab"].items():
        start -= shift
        end -= shift
        if end > 0:
            start = max(start, 0)
            result[name] = start, end
    return result


def aa_mutation_to_position(m: str) -> int:
    """
    E.g. map 'S:N501Y' to 21563 + (501 - 1) * 3 = 23063.
    """
    gene_name, subs = m.split(":")
    start, end = GENE_TO_POSITION[gene_name]
    match = re.search(r"\d+", subs)
    assert match is not None
    aa_offset = int(match.group(0)) - 1
    return start + aa_offset * 3


def nuc_mutations_to_aa_mutations(ms: List[str]) -> List[str]:
    global REFERENCE_SEQ
    if REFERENCE_SEQ is None:
        REFERENCE_SEQ = load_reference_sequence()

    ms_by_aa = defaultdict(list)

    for m in ms:
        # Parse a nucleotide mutation such as "A1234G" -> (1234, "G").
        # Note this uses 1-based indexing.
        if isinstance(m, str):
            position_nuc = int(m[1:-1])
            new_nuc = m[-1]
        else:
            # assert isinstance(m, pyrocov.usher.Mutation)
            position_nuc = m.position
            new_nuc = m.mut

        # Find the first matching gene.
        for gene, (start, end) in GENE_TO_POSITION.items():
            if start <= position_nuc <= end:
                position_aa = (position_nuc - start) // 3
                position_codon = (position_nuc - start) % 3
                ms_by_aa[gene, position_aa].append((position_codon, new_nuc))

    # Format cumulative amino acid changes.
    result = []
    for (gene, position_aa), ms in ms_by_aa.items():
        start, end = GENE_TO_POSITION[gene]

        # Apply mutation to determine new aa.
        pos = start + position_aa * 3
        pos -= 1  # convert from 1-based to 0-based
        old_codon = REFERENCE_SEQ[pos : pos + 3]
        new_codon = list(old_codon)
        for position_codon, new_nuc in ms:
            new_codon[position_codon] = new_nuc
        new_codon = "".join(new_codon)

        # Format.
        old_aa = DNA_TO_AA[old_codon]
        new_aa = DNA_TO_AA[new_codon]
        if new_aa == old_aa:  # ignore synonymous substitutions
            continue
        if old_aa is None:
            old_aa = "STOP"
        if new_aa is None:
            new_aa = "STOP"
        result.append(f"{gene}:{old_aa}{position_aa + 1}{new_aa}")  # 1-based
    return result


def load_reference_sequence():
    with open(os.path.join(NEXTCLADE_DATA, "reference.fasta")) as f:
        ref = "".join(line.strip() for line in f if not line.startswith(">"))
    assert len(ref) == 29903, len(ref)
    return ref
