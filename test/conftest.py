import pyro


def pytest_runtest_setup(item):
    pyro.clear_param_store()
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)
