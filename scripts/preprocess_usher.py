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
from pyrocov.mutrans import START_DATE
from pyrocov.sarscov2 import nuc_mutations_to_aa_mutations
from pyrocov.usher import (
    FineToMeso,
    load_mutation_tree,
    prune_mutation_tree,
    refine_mutation_tree,
)

from pyrocov.geo import get_canonical_location_generator

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

DATE_FORMATS = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}


def parse_date(string):
    fmt = DATE_FORMATS.get(len(string))
    if fmt is None:
        # Attempt to fix poorly formated dates like 2020-09-1.
        parts = string.split("-")
        parts = parts[:1] + [f"{int(p):>02d}" for p in parts[1:]]
        string = "-".join(parts)
        fmt = DATE_FORMATS[len(string)]
    return datetime.datetime.strptime(string, fmt)


def load_metadata(args):
    # Collect all names appearing in the usher tree.
    logger.info("Loading usher tree")
    with open(args.tree_file_in, "rb") as f:
        proto = parsimony_pb2.data.FromString(f.read())
    tree = next(Parser.from_string(proto.newick).parse())

    # Collect condensed samples.
    usher_strains = set()
    condensed_counts = {}
    for node in proto.condensed_nodes:
        usher_strains.update(node.condensed_leaves)
        condensed_counts[node.node_name] = len(node.condensed_leaves)

    # Collect info from each node in the tree.
    clade_to_sample_count = []
    for node in tree.find_clades():
        count = 0
        if node.name:
            if "_condensed_" in node.name:
                count += condensed_counts[node.name]
            else:
                count += 1
                usher_strains.add(node.name)
        clade_to_sample_count.append(count)
    logger.info(f"Found {len(usher_strains)} strains in the usher tree")
    usher_genbanks = set()
    skipped = Counter()
    for strain in usher_strains:
        match = re.search(r"([A-Z]+[0-9]+)\.[0-9]", strain)
        if match:
            usher_genbanks.add(match.group(1))
        else:
            skipped["strain re"] += 1
    logger.info(f"Found {len(usher_genbanks)} genbank accessions in the usher tree")

    # Read date, location, and stats from nextstrain metadata.
    logger.info("Loading nextstrain metadata")
    get_canonical_location = get_canonical_location_generator()
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
        if genbank_accession not in usher_genbanks:
            skipped["unknown genbank_accession"] += 1
            continue

        date = row.date
        if not isinstance(date, str) or date == "?":
            skipped["no date"] += 1
            continue
        date = parse_date(date)
        if date < args.start_date:
            date = args.start_date  # Clip rows before start date.

        # Create a standard location.
        strain = row.strain
        region = row.region
        country = row.country
        division = row.division
        location = row.location
        
        location = get_canonical_location(strain, region, country, division, location)
        if not location:
            skipped["country"] += 1
            continue

        # Add a row.
        columns["genbank_accession"] = genbank_accession
        columns["day"].append((date - args.start_date).days)
        columns["location"] = location
    assert sum(skipped.values()) < 2e6, f"suspicious skippage:\n{skipped}"
    logger.info(f"Skipped {sum(skipped.values())} nodes because:\n{skipped}")
    logger.info(f"Kept {len(columns['day'])} rows")

    with open(args.stats_file_out, "wb") as f:
        pickle.dump(stats, f)
    logger.info(f"Saved {args.stats_file_out}")

    return columns, clade_to_sample_count


def prune_tree(args, coarse_to_fine, clade_to_sample_count):
    # To ensure pango lineages remain distinct, set their weights to infinity.
    weights = defaultdict(float, {fine: math.inf for fine in coarse_to_fine.values()})

    # Add weight of 1 for each sampled genome.
    with open(args.tree_file_out, "rb") as f:
        proto = parsimony_pb2.data.FromString(f.read())
    for meta, count in zip(proto.metadata, clade_to_sample_count):
        assert isinstance(meta.clade, str)
        weights[meta.clade] += count
    weights.pop("", None)

    # Prune the tree, minimizing the number of incorrect mutations.
    args.max_num_clades = max(args.max_num_clades, len(coarse_to_fine))
    pruned_tree_filename = f"results/lineageTree.{args.max_num_clades}.pb"
    meso_set = prune_mutation_tree(
        args.tree_file_out, pruned_tree_filename, args.max_num_clades, weights
    )
    assert len(meso_set) == args.max_num_clades
    return FineToMeso(meso_set), pruned_tree_filename


def main(args):
    # Create columns.
    columns, clade_to_sample_count = load_metadata(args)

    # Extract mappings between coarse lineages and fine clades.
    coarse_proto = args.tree_file_in
    fine_proto = args.tree_file_out
    fine_to_coarse = refine_mutation_tree(coarse_proto, fine_proto)
    coarse_to_fines = defaultdict(list)
    for fine, coarse in fine_to_coarse.items():
        coarse_to_fines[coarse].append(fine)
    # Choose the basal representative.
    coarse_to_fine = {c: min(fs) for c, fs in coarse_to_fines.items()}

    # Prune tree, updating data structures to use meso-scale clades.
    fine_to_meso, pruned_tree_filename = prune_tree(
        args, coarse_to_fine, clade_to_sample_count
    )
    fine_to_coarse = {fine_to_meso(f): c for f, c in fine_to_coarse.items()}
    coarse_to_fine = {c: fine_to_meso(f) for c, f in coarse_to_fine.items()}
    columns["clade"] = [fine_to_meso(c) for c in columns["clade"]]
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
    parser.add_argument("-c", "--max-num-clades", type=int, default=5000)
    parser.add_argument("--start-date", default=START_DATE)
    args = parser.parse_args()
    args.start_date = parse_date(args.start_date)
    if not args.features_file_out:
        args.features_file_out = f"results/features.{args.max_num_clades}.pt"
    if not args.columns_file_out:
        args.columns_file_out = f"results/columns.{args.max_num_clades}.pkl"
    main(args)
