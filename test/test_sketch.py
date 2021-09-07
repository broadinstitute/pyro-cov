# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import re

import pyro.distributions as dist
import pytest
import torch

from pyrocov.sketch import AMSSketcher, ClockSketcher, KmerCounter


def random_string(size):
    probs = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.05])
    probs /= probs.sum()
    string = "".join("ACGTN"[i] for i in dist.Categorical(probs).sample([size]))
    return string


def test_kmer_counter():
    string = random_string(10000)

    results = {}
    for backend in ["python", "cpp"]:
        results[backend] = KmerCounter(backend=backend)
        for part in re.findall("[ACTG]+", string):
            results[backend].update(part)
        results[backend].flush()

    expected = results["python"]
    actual = results["cpp"]
    assert actual == expected


@pytest.mark.parametrize("min_k,max_k", [(2, 2), (2, 4), (3, 12)])
@pytest.mark.parametrize("bits", [16])
def test_string_to_soft_hash(min_k, max_k, bits):
    string = random_string(1000)

    results = {}
    for backend in ["python", "cpp"]:
        results[backend] = torch.empty(64)
        sketcher = AMSSketcher(min_k=min_k, max_k=max_k, bits=bits, backend=backend)
        sketcher.string_to_soft_hash(string, results[backend])

    expected = results["python"]
    actual = results["cpp"]
    tol = expected.std().item() * 1e-6
    assert (actual - expected).abs().max().item() < tol


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8, 16, 32])
def test_string_to_clock_hash(k):
    string = random_string(1000)

    results = {}
    for backend in ["python", "cpp"]:
        sketcher = ClockSketcher(k, backend=backend)
        results[backend] = sketcher.init_sketch()
        sketcher.string_to_hash(string, results[backend])

    expected = results["python"]
    actual = results["cpp"]
    assert (actual.clocks == expected.clocks).all()
    assert (actual.count == expected.count).all()


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8, 16, 32])
@pytest.mark.parametrize("size", [20000])
def test_clock_cdiff(k, size):
    n = 10
    strings = [random_string(size) for _ in range(n)]
    sketcher = ClockSketcher(k)
    sketch = sketcher.init_sketch(n)
    for i, string in enumerate(strings):
        sketcher.string_to_hash(string, sketch[i])

    cdiff = sketcher.cdiff(sketch, sketch)
    assert cdiff.shape == (n, n)
    assert (cdiff.clocks.transpose(0, 1) == -cdiff.clocks).all()
    assert (cdiff.clocks.diagonal(dim1=0, dim2=1) == 0).all()
    assert (cdiff.count.diagonal(dim1=0, dim2=1) == 0).all()
    mask = torch.arange(n) < torch.arange(n).unsqueeze(-1)
    mean = cdiff.clocks[mask].abs().float().mean().item()
    assert mean > 64 - 10, mean
