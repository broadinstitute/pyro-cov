# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from subprocess import check_call

# This keeps repos organized as ~/github/{user}/{repo}
GITHUB = os.path.expanduser(os.path.join("~", "github"))
if not os.path.exists(GITHUB):
    os.makedirs(GITHUB)

for arg in sys.argv[1:]:
    try:
        user, repo = arg.split("/")
    except Exception:
        raise ValueError(
            f"Expected args of the form username/repo e.g. pyro-ppl/pyro, but got {arg}"
        )

    dirname = os.path.join(GITHUB, user)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)
    dirname = os.path.join(dirname, repo)
    if not os.path.exists(dirname):
        print(f"Cloning {arg}")
        check_call(["git", "clone", f"git@github.com:{user}/{repo}"])
    else:
        print(f"Pulling {arg}")
        os.chdir(dirname)
        check_call(["git", "pull"])
