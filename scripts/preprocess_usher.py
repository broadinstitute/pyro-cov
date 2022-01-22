# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging
import math
import pickle
import re
from collections import Counter, defaultdict

import pandas as pd
import torch
import tqdm
from Bio.Phylo.NewickIO import Parser

from pyrocov.external.usher import parsimony_pb2
from pyrocov.geo import get_canonical_location_generator
from pyrocov.mutrans import START_DATE
from pyrocov.sarscov2 import nuc_mutations_to_aa_mutations
from pyrocov.usher import (
    FineToMeso,
    load_mutation_tree,
    prune_mutation_tree,
    refine_mutation_tree,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

DATE_FORMATS = {7: "%Y-%m", 10: "%Y-%m-%d"}


def try_parse_date(string):
    fmt = DATE_FORMATS.get(len(string))
    if fmt is not None:
        return datetime.datetime.strptime(string, fmt)


def try_parse_genbank(strain):
    match = re.search(r"([A-Z]+[0-9]+)\.[0-9]", strain)
    if match:
        return match.group(1)


def load_metadata(args):
    # Collect all names appearing in the usher tree.
    logger.info("Loading fine tree")
    with open(args.tree_file_out, "rb") as f:
        proto = parsimony_pb2.data.FromString(f.read())
    tree = next(Parser.from_string(proto.newick).parse())
    skipped = Counter()

    # Collect condensed samples.
    usher_strains = set()
    nodename_to_genbanks = {}
    for node in proto.condensed_nodes:
        usher_strains.update(node.condensed_leaves)
        genbanks = []
        for strain in node.condensed_leaves:
            usher_strains.add(strain)
            genbank = try_parse_genbank(strain)
            if genbank is None:
                skipped["strain re"] += 1
            else:
                genbanks.append(genbank)
        nodename_to_genbanks[node.node_name] = genbanks

    # Collect info from each node in the tree.
    for node in tree.find_clades():
        if node.name and node.name not in nodename_to_genbanks:
            usher_strains.add(node.name)
            genbank = try_parse_genbank(node.name)
            if genbank is None:
                skipped["strain re"] += 1
            else:
                nodename_to_genbanks[node.name] = [genbank]
    logger.info(f"Found {len(usher_strains)} strains in the usher tree")

    # Collate.
    genbank_to_nodename = {k: v for v, ks in nodename_to_genbanks.items() for k in ks}
    logger.info(
        f"Found {len(genbank_to_nodename)} genbank accessions in the usher tree"
    )
    node_to_clade = {}
    for clade, meta in zip(tree.find_clades(), proto.metadata):
        if meta.clade:
            node_to_clade[clade] = meta.clade
        # Propagate down to descendent clones.
        for child in clade.clades:
            node_to_clade[child] = node_to_clade[clade]
    nodename_to_clade = {
        node.name: clade for node, clade in node_to_clade.items() if node.name
    }

    # Read date, location, and stats from nextstrain metadata.
    logger.info("Loading nextstrain metadata")
    get_canonical_location = get_canonical_location_generator(
        args.recover_missing_usa_state
    )
    stats = defaultdict(Counter)
    nextstrain_df = pd.read_csv("results/nextstrain/metadata.tsv", sep="\t", dtype=str)
    columns = defaultdict(list)
    for row in tqdm.tqdm(nextstrain_df.itertuples(), total=len(nextstrain_df)):
        # Collect background mutation statistics for dN/dS etc.
        for key in ["aaSubstitutions", "insertions", "substitutions"]:
            values = getattr(row, key)
            if isinstance(values, str) and values:
                stats[key].update(values.split(","))

        genbank_accession = row.genbank_accession
        nodename = genbank_to_nodename.get(genbank_accession)
        if nodename is None:
            skipped["unknown genbank_accession"] += 1
            continue
        clade = nodename_to_clade[nodename]

        date = row.date
        if not isinstance(date, str) or date == "?":
            skipped["no date"] += 1
            continue
        date = try_parse_date(date)
        if date is None:
            skipped["no date"] += 1
            continue
        if date < args.start_date:
            date = args.start_date  # Clip rows before start date.

        # Create a standard location.
        location = get_canonical_location(
            row.strain, row.region, row.country, row.division, row.location
        )
        if location is None:
            skipped["country"] += 1
            continue

        lineage = row.pango_lineage
        if not isinstance(lineage, str) or not lineage or lineage == "?":
            lineage = None

        # Add a row.
        columns["genbank_accession"].append(genbank_accession)
        columns["day"].append((date - args.start_date).days)
        columns["location"].append(location)
        columns["lineage"].append(lineage)
        columns["nodename"].append(nodename)
        columns["clade"].append(clade)
    columns = dict(columns)
    assert len(set(map(len, columns.values()))) == 1, "columns have unequal length"
    assert sum(skipped.values()) < 2e6, f"suspicious skippage:\n{skipped}"
    logger.info(f"Skipped {sum(skipped.values())} nodes because:\n{skipped}")
    logger.info(f"Kept {len(columns['day'])} rows")

    with open("results/columns.pkl", "wb") as f:
        pickle.dump(columns, f)
    logger.info("Saved results/columns.pkl")

    with open(args.stats_file_out, "wb") as f:
        pickle.dump(stats, f)
    logger.info(f"Saved {args.stats_file_out}")

    return columns, nodename_to_genbanks


def prune_tree(args, coarse_to_fine, nodename_to_genbanks):
    logger.info(f"Loading fine tree {args.tree_file_out}")
    with open(args.tree_file_out, "rb") as f:
        proto = parsimony_pb2.data.FromString(f.read())
    tree = next(Parser.from_string(proto.newick).parse())

    # Add weights for leaves.
    cum_weights = {}
    mrca_weights = {}
    for clade in tree.find_clades():
        genbanks = nodename_to_genbanks.get(clade.name, [])
        count = len(genbanks)
        cum_weights[clade] = count
        mrca_weights[clade] = count ** 2

    # Add weights of MRCA pairs.
    reverse_clades = list(tree.find_clades())
    reverse_clades.reverse()  # from leaves to root
    for parent in reverse_clades:
        for child in parent.clades:
            cum_weights[parent] += cum_weights[child]
        for child in parent.clades:
            mrca_weights[parent] += cum_weights[child] * (
                cum_weights[parent] - cum_weights[child]
            )
    num_samples = sum(map(len, nodename_to_genbanks.values()))
    assert cum_weights[tree.root] == num_samples
    assert sum(mrca_weights.values()) == num_samples ** 2

    # Aggregate among clones to basal representative.
    weights = defaultdict(float)
    for meta, parent in zip(reversed(proto.metadata), reverse_clades):
        for child in parent.clades:
            mrca_weights[parent] += mrca_weights.get(child, 0)
        assert isinstance(meta.clade, str)
        if meta.clade:
            weights[meta.clade] = mrca_weights.pop(parent)
    assert sum(weights.values()) == num_samples ** 2

    # To ensure pango lineages remain distinct, set their weights to infinity.
    for fine in coarse_to_fine.values():
        weights[fine] = math.inf
    assert "" not in weights

    # Prune the tree, minimizing the number of incorrect mutations.
    args.max_num_clades = max(args.max_num_clades, len(coarse_to_fine))
    pruned_tree_filename = f"results/lineageTree.{args.max_num_clades}.pb"
    meso_set = prune_mutation_tree(
        args.tree_file_out, pruned_tree_filename, args.max_num_clades, weights
    )
    assert len(meso_set) == args.max_num_clades
    return FineToMeso(meso_set), pruned_tree_filename


def main(args):
    # Extract mappings between coarse lineages and fine clades.
    coarse_proto = args.tree_file_in
    fine_proto = args.tree_file_out
    fine_to_coarse = refine_mutation_tree(coarse_proto, fine_proto)
    coarse_to_fines = defaultdict(list)
    for fine, coarse in fine_to_coarse.items():
        coarse_to_fines[coarse].append(fine)
    # Choose the basal representative.
    coarse_to_fine = {c: min(fs) for c, fs in coarse_to_fines.items()}

    # Create columns.
    columns, nodename_to_genbanks = load_metadata(args)
    columns["lineage"] = [fine_to_coarse[f] for f in columns["clade"]]

    # Prune tree, updating data structures to use meso-scale clades.
    fine_to_meso, pruned_tree_filename = prune_tree(
        args, coarse_to_fine, nodename_to_genbanks
    )
    fine_to_coarse = {fine_to_meso(f): c for f, c in fine_to_coarse.items()}
    coarse_to_fine = {c: fine_to_meso(f) for c, f in coarse_to_fine.items()}
    columns["clade"] = [fine_to_meso(f) for f in columns["clade"]]
    clade_set = set(columns["clade"])
    assert len(clade_set) <= args.max_num_clades
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(columns, f)
    logger.info(f"Saved {args.columns_file_out}")

    # Convert from nucleotide mutations to amino acid mutations.
    nuc_mutations_by_clade = load_mutation_tree(pruned_tree_filename)
    assert nuc_mutations_by_clade
    aa_mutations_by_clade = {
        clade: nuc_mutations_to_aa_mutations(mutations)
        for clade, mutations in nuc_mutations_by_clade.items()
    }

    # Create dense aa features.
    clades = sorted(nuc_mutations_by_clade)
    clade_ids = {k: i for i, k in enumerate(clades)}
    aa_mutations = sorted(set().union(*aa_mutations_by_clade.values()))
    logger.info(f"Found {len(aa_mutations)} amino acid mutations")
    mutation_ids = {k: i for i, k in enumerate(aa_mutations)}
    aa_features = torch.zeros(len(clade_ids), len(mutation_ids), dtype=torch.bool)
    for clade, ms in aa_mutations_by_clade.items():
        i = clade_ids[clade]
        for m in ms:
            j = mutation_ids.get(m)
            aa_features[i, j] = True

    # Save features.
    features = {
        "clades": clades,
        "clade_to_lineage": fine_to_coarse,
        "lineage_to_clade": coarse_to_fine,
        "aa_mutations": aa_mutations,
        "aa_features": aa_features,
    }
    logger.info(
        f"saving {tuple(aa_features.shape)} aa features to {args.features_file_out}"
    )
    torch.save(features, args.features_file_out)
    logger.info(f"Saved {args.features_file_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pangolin mutations")
    parser.add_argument(
        "--usher-metadata-file-in", default="results/usher/metadata.tsv"
    )
    parser.add_argument(
        "--nextstrain-metadata-file-in", default="results/nextstrain/metadata.tsv"
    )
    parser.add_argument("--tree-file-in", default="results/usher/all.masked.pb")
    parser.add_argument("--tree-file-out", default="results/lineageTree.fine.pb")
    parser.add_argument("--stats-file-out", default="results/stats.pkl")
    parser.add_argument("--columns-file-out", default="")
    parser.add_argument("--features-file-out", default="")
    parser.add_argument("--recover-missing-usa-state", action="store_true")
    parser.add_argument("-c", "--max-num-clades", type=int, default=3000)
    parser.add_argument("--start-date", default=START_DATE)
    args = parser.parse_args()
    args.start_date = try_parse_date(args.start_date)
    if not args.features_file_out:
        args.features_file_out = f"results/features.{args.max_num_clades}.pt"
    if not args.columns_file_out:
        args.columns_file_out = f"results/columns.{args.max_num_clades}.pkl"
    main(args)
