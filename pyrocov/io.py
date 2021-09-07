# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import functools
import io
import logging
import math
import re
import sys

import torch
import torch.multiprocessing as mp
from Bio import AlignIO
from Bio.Phylo.NewickIO import Parser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .phylo import Phylogeny

logger = logging.getLogger(__name__)

FILE_FORMATS = {
    "nex": "nexus",
    "nexus": "nexus",
    "fasta": "fasta",
    "xml": "beast",
}


def _print_dot():
    sys.stdout.write(".")
    sys.stdout.flush()


def _handle_translate(lines, context):
    map_lines = [line.rstrip(",").split() for line in lines[1:-1]]
    context["translate"] = {key: value for key, value in map_lines}


def _handle_tree_count(lines, context):
    return 1


def _handle_tree_newick(lines, context):
    assert len(lines) == 1
    tree, name, equal, newick = lines[0].split()
    assert tree == "tree"
    assert equal == "="
    tree = next(Parser.from_string(newick).parse())
    tree.name = name

    # Add translations as .comment attributes
    if "translate" in context:
        translate = context["translate"]
        for leaf in tree.get_terminals():
            leaf.comment = translate[leaf.name]

    return tree


def _handle_tree_torch(lines, context):
    assert len(lines) == 1
    tree, name, equal, newick = lines[0].split()
    assert tree == "tree"
    assert equal == "="
    tree = next(Parser.from_string(newick).parse())
    tree = Phylogeny.from_bio_phylo(tree)
    _print_dot()
    return tree


def _handle_raw(lines, context):
    return lines, context


def _apply(fn, args):
    return fn(*args)


def read_nexus_trees(filename, *, format="newick", max_num_trees=math.inf, processes=0):
    """
    Parse and iterate over newick trees stored in a nexus file.
    This streams the file and thus can handle larger files than
    ``Bio.Phylo.read(..., format="nexus")``.

    Returns an iterator of ``Bio.Phylo`` tree objects.
    """
    if format == "count":
        context = {}
        handlers = {"tree": _handle_tree_count}
    elif format == "newick":
        context = {"translate": {}}
        handlers = {"translate": _handle_translate, "tree": _handle_tree_newick}
    elif format == "_raw_newick":
        context = {"translate": {}}
        handlers = {"translate": _handle_translate, "tree": _handle_raw}
    elif format == "torch":
        context = None
        handlers = {"tree": _handle_tree_torch}
    elif format == "_raw_torch":
        context = None
        handlers = {"tree": _handle_raw}
    else:
        raise ValueError(f"unknown format: {format}")

    if processes != 0:
        trees = read_nexus_trees(
            filename, format="_raw_" + format, max_num_trees=max_num_trees
        )
        with mp.Pool(processes) as pool:
            handler = functools.partial(_apply, handlers["tree"])
            yield from pool.imap(handler, trees)
        return

    with open(filename) as f:
        lines = iter(f)
        for line in lines:
            if line.startswith("Begin trees;"):
                break
        part = []
        for line in lines:
            line = line.strip()
            part.append(line)
            if not line.endswith(";"):
                continue
            type_ = part[0].split()[0].lower()
            handle = handlers.get(type_)
            if handle is not None:
                tree = handle(part, context)
                if tree is not None:
                    yield tree
                    max_num_trees -= 1
                    if max_num_trees <= 0:
                        break
            part = []


def count_nexus_trees(filename):
    """
    Counts the number of trees in a nexus file.
    """
    return sum(read_nexus_trees(filename, format="count"))


def stack_nexus_trees(filename, *, max_num_trees=math.inf, processes=0):
    """
    Loads a batch of trees from a nexus file.
    """
    trees = read_nexus_trees(
        filename, format="torch", max_num_trees=max_num_trees, processes=processes
    )
    return Phylogeny.stack(trees)


def read_newick_tree(filename):
    """
    Parse a single newick tree and convert to a ``Phylogeny``.
    """
    with open(filename) as f:
        line = f.read().strip()
    tree = next(Parser.from_string(line).parse())
    return Phylogeny.from_bio_phylo(tree)


def read_alignment(
    filename, format=None, *, max_taxa=math.inf, max_characters=math.inf
):
    """
    Reads a single alignment file to a torch tensor of probabilites.

    :param str filename: Name of input file.
    :param str format: Optional input format, e.g. "nexus" or "fasta".
    :param int max_taxa: Optional number of taxa for truncation.
    :param int max_characters: Optional number of characters for truncation.
    :rtype: torch.Tensor
    :returns: A float tensor of shape ``(num_sequences, num_characters,
        num_bases)`` that is normalized along its rightmost dimension. Note
        that ``num_bases`` is 5 = 4 + 1, where the final base denots a gap or
        indel.
    """
    # Load a Bio.Align.MultipleSeqAlignment object.
    logger.info(f"Loading data from {filename}")
    if format is None:
        suffix = filename.split(".")[-1].lower()
        format = FILE_FORMATS.get(suffix)
    if format is None:
        raise ValueError("Please specify a file format, e.g. 'nexus' or 'fasta'")
    elif format == "nexus":
        alignment = _read_alignment_nexus(filename)
    elif format == "beast":
        alignment = _read_alignment_beast(filename)
    else:
        alignment = AlignIO.read(filename, format)

    # Convert to a single torch.Tensor.
    num_taxa = min(len(alignment), max_taxa)
    if num_taxa < len(alignment):
        alignment = alignment[:num_taxa]
    num_characters = min(len(alignment[0]), max_characters)
    if num_characters < len(alignment[0]):
        alignment = alignment[:, :num_characters]
    logger.info(f"parsing {num_taxa} taxa x {num_characters} characters")
    codebook = _get_codebook()
    probs = torch.full((num_taxa, num_characters, 5), 1 / 5)
    for i in range(num_taxa):
        seq = alignment[i].seq
        if not VALID_CODES.issuperset(seq):
            raise ValueError(f"Invalid characters: {set(seq) - VALID_CODES}")
        # Replace gaps at ends with missing.
        beg, end = 0, probs.size(1)
        if seq[0] in "-.N":
            seq, old = seq.lstrip(seq[0]), seq
            beg += len(old) - len(seq)
        if seq[-1] in "-.N":
            seq, old = seq.rstrip(seq[-1]), seq
            end -= len(old) - len(seq)
        probs[i, beg:end] = codebook[list(map(ord, seq))]
    assert torch.isfinite(probs).all()
    return probs


def _read_alignment_nexus(filename):
    # Work around bugs in Bio.Nexus reader.
    lines = []
    section = None
    done = set()
    with open(filename) as f:
        for line in f:
            if line.startswith("BEGIN"):
                section = line.split()[-1].strip()[:-1]
            elif line.startswith("END;"):
                done.add(section)
                section = None
                if "TAXA" in done and "CHARACTERS" in done:
                    lines.append(line)
                    break
            elif section == "CHARACTERS":
                if "{" in line:
                    line = re.sub("{([ATCG]+)}", _encode_ambiguity, line)
            lines.append(line)
    f = io.StringIO("".join(lines))
    alignment = AlignIO.read(f, "nexus")
    return alignment


def _read_alignment_beast(filename):
    result = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("<sequence "):
                continue
            id_ = re.search(r'\bid="([^"]*)"', line).group(1)
            seq = re.search(r'\bvalue="([^"]*)"', line).group(1)
            result.append(SeqRecord(Seq(seq), id=id_))
    return result


# See https://www.bioinformatics.org/sms/iupac.html
NUCLEOTIDE_CODES = {
    #    [  A,   C,   G,   T, gap]
    "?": [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],  # missing
    "n": [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],  # missing
    "A": [1 / 1, 0.0, 0.0, 0.0, 0.0],  # adenine
    "C": [0.0, 1 / 1, 0.0, 0.0, 0.0],  # cytosine
    "G": [0.0, 0.0, 1 / 1, 0.0, 0.0],  # guanine
    "T": [0.0, 0.0, 0.0, 1 / 1, 0.0],  # thymine
    "U": [0.0, 0.0, 0.0, 1 / 1, 0.0],  # uracil
    "R": [1 / 2, 0.0, 1 / 2, 0.0, 0.0],
    "Y": [0.0, 1 / 2, 0.0, 1 / 2, 0.0],
    "S": [0.0, 1 / 2, 1 / 2, 0.0, 0.0],
    "W": [1 / 2, 0.0, 0.0, 1 / 2, 0.0],
    "K": [0.0, 0.0, 1 / 2, 1 / 2, 0.0],
    "M": [1 / 2, 1 / 2, 0.0, 0.0, 0.0],
    "B": [0.0, 1 / 3, 1 / 3, 1 / 3, 0.0],
    "D": [1 / 3, 0.0, 1 / 3, 1 / 3, 0.0],
    "H": [1 / 3, 1 / 3, 0.0, 1 / 3, 0.0],
    "V": [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0],
    "N": [1 / 4, 1 / 4, 1 / 4, 1 / 4, 0.0],
    "-": [0.0, 0.0, 0.0, 0.0, 1 / 1],  # gap
    ".": [0.0, 0.0, 0.0, 0.0, 1 / 1],  # gap
}
VALID_CODES = set(NUCLEOTIDE_CODES)

AMBIGUOUS_CODES = {
    frozenset("AG"): "R",
    frozenset("CT"): "Y",
    frozenset("CG"): "S",
    frozenset("AT"): "W",
    frozenset("GT"): "K",
    frozenset("AC"): "M",
    frozenset("CGT"): "B",
    frozenset("AGT"): "D",
    frozenset("ACT"): "H",
    frozenset("ACG"): "V",
    frozenset("ACGT"): "N",
}
assert len(AMBIGUOUS_CODES) == 6 + 4 + 1


def _encode_ambiguity(chars):
    return AMBIGUOUS_CODES[frozenset(chars.group(1))]


def _get_codebook():
    codes = torch.full((256, 5), math.nan)
    keys = list(map(ord, NUCLEOTIDE_CODES.keys()))
    values = torch.tensor(list(NUCLEOTIDE_CODES.values()))
    assert values.sum(-1).sub(1).abs().le(1e-6).all()
    codes[keys] = values
    return codes
