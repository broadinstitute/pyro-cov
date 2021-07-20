## File organization

- The root directory `/` contains scripts and jupyter notebooks.
- All module code lives in the `/pyrocov/` package directory.
  As a rule of thumb, if you're importing a .py file an any other file,
  it should live in the `/pyrocov/` directory; if your .py file has an `if __name__ == "__main__"` check, then it should live in the root directory.
- Notebooks should be cleared of data before committing.
- The `/results/` directory (not under git control) contains large intermediate data
  including preprocessed input data and outputs from models and prediction.
- The `/paper/` directory contains git-controlled output for the paper, namely
  plots, .tsv files, and .fasta files for sharing outside of the dev team.

## Committing code

- The current policy is to allow pushing of code directly, without PRs.
- If you would like code reviewed, please submit a PR and tag a reviewer.
- Please run `make format` and `make lint` before committing code.
- We'd like to resurrect `make test` but it is currently broken.
- Each notebook has an owner (see git history);
  changes to notebooks by non-owners should be through pull request.
  This rule is intended to reduce difficult merge conflicts in jupyter notebooks.

