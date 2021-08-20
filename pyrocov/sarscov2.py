import re
from collections import OrderedDict

# Adapted from https://github.com/nextstrain/ncov/blob/50ceffa/defaults/annotation.gff
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


GENE_TO_POSITION = _()

GENE_STRUCTURE = {
    # https://www.nature.com/articles/s41401-020-0485-4/figures/2
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
    "N": {"immunogenic": (133, 217)},
}


def aa_mutation_to_position(m):
    """
    E.g. map 'S:N501Y' to 21563 + (501 - 1) * 3 = 23063.
    """
    gene_name, subs = m.split(":")
    start, end = GENE_TO_POSITION[gene_name]
    aa_offset = int(re.search(r"\d+", subs).group(0)) - 1
    return start + aa_offset * 3
