# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import re
import sys

from setuptools import find_packages, setup

with open("pyrocov/__init__.py") as f:
    for line in f:
        match = re.match('^__version__ = "(.*)"$', line)
        if match:
            __version__ = match.group(1)
            break

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="pyrocov",
    version="0.1.0",
    description="Pyro tools for Sars-CoV-2 analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["pyrocov"]),
    url="http://pyro.ai",
    author="Pyro team at the Broad Institute of MIT and Harvard",
    author_email="fobermey@broadinstitute.org",
    install_requires=[
        "biopython>=1.54",
        "pyro-ppl>=1.7",
        "geopy",
        "gpytorch",
        "scikit-learn",
        "umap-learn",
        "mappy",
        "protobuf",
        "tqdm",
        "colorcet",
    ],
    extras_require={
        "test": [
            "black",
            "isort>=5.0",
            "flake8",
            "pytest>=5.0",
            "mypy>=0.812",
            "types-protobuf",
        ],
    },
    python_requires=">=3.6",
    keywords="pyro pytorch phylogenetic machine learning",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
