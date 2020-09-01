from Bio.Phylo.NewickIO import Parser


def _handle_translate(lines, context):
    map_lines = [line.rstrip(",").split() for line in lines[1:-1]]
    context["translate"] = {key: value for key, value in map_lines}
    return ()


def _handle_tree(lines, context):
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

    yield tree


_HANDLERS = {
    "translate": _handle_translate,
    "tree": _handle_tree,
}


def parse_nexus_trees(filename):
    """
    Parse and iterate over newick trees stored in a nexus file.
    This streams the file and thus can handle larger files than
    ``Bio.Phylo.read(..., format="nexus")``.
    """
    with open(filename) as f:
        lines = iter(f)
        for line in lines:
            if line.startswith("Begin trees;"):
                break
        context = {"translate": {}}
        part = []
        for line in lines:
            line = line.strip()
            part.append(line)
            if line.endswith(";"):
                type_ = part[0].split()[0].lower()
                yield from _HANDLERS[type_](part, context)
                part = []
