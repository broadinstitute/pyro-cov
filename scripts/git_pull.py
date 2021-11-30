# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from subprocess import check_call

# This keeps repos organized as ~/github/{user}/{repo}
GITHUB = os.path.expanduser(os.path.join("~", "github"))
if not os.path.exists(GITHUB):
    os.makedirs(GITHUB)

update = True
for arg in sys.argv[1:]:
    if arg == "--no-update":
        update = False
        continue

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
        check_call(["git", "clone", "--depth", "1", f"git@github.com:{user}/{repo}"])
    elif update:
        print(f"Pulling {arg}")
        os.chdir(dirname)
        check_call(["git", "pull"])
