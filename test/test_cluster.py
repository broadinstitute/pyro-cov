import pyro.distributions as dist
import pytest
import torch

from pyrophylo.cluster import AMSSketcher, ClockSketcher


def random_string(size):
    probs = torch.tensor([1., 1., 1., 1., 0.05])
    probs /= probs.sum()
    string = "".join("ACGTN"[i] for i in dist.Categorical(probs).sample([size]))
    return string


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

    clocks = {}
    count = {}
    for backend in ["python", "cpp"]:
        sketcher = ClockSketcher(k, backend=backend)
        clocks[backend], count[backend] = sketcher.init_hash()
        sketcher.string_to_hash(string, clocks[backend], count[backend])

    for results in [clocks, count]:
        expected = results["python"]
        actual = results["cpp"]
        assert (actual == expected).all()


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8, 16, 32])
@pytest.mark.parametrize("size", [20000])
def test_clock_cdiff(k, size):
    n = 10
    strings = [random_string(size) for _ in range(n)]
    sketcher = ClockSketcher(k)
    clocks, count = sketcher.init_hash(n)
    for i, string in enumerate(strings):
        sketcher.string_to_hash(string, clocks[i], count[i])

    cdiff = sketcher.cdiff(clocks, count, clocks, count)
    assert cdiff.shape == (n, n, 64)
    cc = cdiff.transpose(0, 1) + cdiff
    assert ((cc == 0) | (cc == -512)).all()
    assert (cdiff.diagonal(dim1=0, dim2=1) == 0).all()
    mask = torch.arange(n) < torch.arange(n).unsqueeze(-1)
    mean = cdiff[mask].abs().float().mean().item()
    assert mean > 127 - 20, mean
