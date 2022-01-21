# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

from pyrocov.sarscov2 import nuc_mutations_to_aa_mutations


def test_nuc_to_aa():
    assert nuc_mutations_to_aa_mutations(["A23403G"]) == ["S:D614G"]
