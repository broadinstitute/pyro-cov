# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pyro

level = logging.INFO if "CI" in os.environ else logging.DEBUG
logging.basicConfig(format="%(levelname).1s \t %(message)s", level=level)


def pytest_runtest_setup(item):
    pyro.clear_param_store()
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)
