# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import random
import tempfile
from collections import defaultdict

from pyrocov.align import PANGOLEARN_DATA
from pyrocov.usher import prune_mutation_tree, refine_mutation_tree


def test_refine_prune():
    filename1 = os.path.join(PANGOLEARN_DATA, "lineageTree.pb")
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename2 = os.path.join(tmpdirname, "refinedTree.pb")
        filename3 = os.path.join(tmpdirname, "prunedTree.pb")

        # Refine the tree.
        fine_to_coarse = refine_mutation_tree(filename1, filename2)

        # Find canonical fine names for each coarse name.
        coarse_to_fine = defaultdict(list)
        for fine, coarse in fine_to_coarse.items():
            coarse_to_fine[coarse].append(fine)
        coarse_to_fine = {k: min(vs) for k, vs in coarse_to_fine.items()}

        # Prune the tree, keeping coarse nodes.
        weights = {fine: random.lognormvariate(0, 1) for fine in fine_to_coarse}
        for fine in coarse_to_fine.values():
            weights[fine] = math.inf
        prune_mutation_tree(filename2, filename3, weights=weights, max_num_nodes=10000)
