import math
from contextlib import ExitStack

import torch.multiprocessing as mp
from Bio.Phylo.NewickIO import Parser

from .phylo import Phylogeny


def _handle_translate(lines, context):
    map_lines = [line.rstrip(",").split() for line in lines[1:-1]]
    context["translate"] = {key: value for key, value in map_lines}


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
    return Phylogeny.from_bio_phylo(tree)


def _handle_async(pool, fn):
    def handle(lines, context):
        return pool.apply_async(fn, (lines, context))
    return handle


def iter_nexus_trees(filename, *, format="newick", processes=0):
    """
    Parse and iterate over newick trees stored in a nexus file.
    This streams the file and thus can handle larger files than
    ``Bio.Phylo.read(..., format="nexus")``.

    Returns an iterator of ``Bio.Phylo`` tree objects.
    """
    if format == "newick":
        context = {"translate": {}}
        handlers = {"translate": _handle_translate,
                    "tree": _handle_tree_newick}
    elif format == "torch":
        context = None
        handlers = {"tree": _handle_tree_torch}
    else:
        raise ValueError(f"unknown format: {format}")

    with ExitStack() as stack:
        # Optionally generate async results.
        if processes > 0:
            mp.set_start_method("spawn")
            pool = stack.enter_context(mp.Pool(processes))
            handlers["tree"] = _handle_async(pool, handlers.pop("tree"))

        # Skip until trees.
        lines = iter(stack.enter_context(open(filename)))
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
            part = []


def stack_nexus_trees(filename, *, max_num_trees=math.inf, processes=0, timeout=None):
    """
    Loads a batch of trees from a nexus file.
    """
    phylogenies = []
    for tree in iter_nexus_trees(filename, format="torch", processes=processes):
        phylogenies.append(tree)
        if len(phylogenies) >= max_num_trees:
            break
    if processes > 0:
        phylogenies = [async_result.get(timeout) for async_result in phylogenies]
    return Phylogeny.stack(phylogenies)
