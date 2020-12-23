import pyro.distributions as dist
import pytest
import torch

from pyrophylo.cluster import KmerSketcher


@pytest.mark.parametrize("min_k,max_k", [(2, 2), (2, 4), (3, 12)])
@pytest.mark.parametrize("bits", [16])
def test_string_to_soft_hash(min_k, max_k, bits):
    probs = torch.tensor([1., 1., 1., 1., 0.05])
    probs /= probs.sum()
    string = "".join("ACGTN"[i] for i in dist.Categorical(probs).sample([10000]))

    results = {}
    for backend in ["python", "cpp"]:
        results[backend] = torch.empty(bits)
        sketcher = KmerSketcher(min_k=min_k, max_k=max_k, bits=bits, backend=backend)
        sketcher.string_to_soft_hash(string, results[backend])

    expected = results["python"]
    actual = results["cpp"]
    tol = expected.std().item() * 1e-6
    assert (actual - expected).abs().max().item() < tol
