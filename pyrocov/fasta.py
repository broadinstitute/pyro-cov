# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0


class ShardedFastaWriter:
    """
    Writer that splits into multiple fasta files to avoid nextclade file size
    limit.
    """

    def __init__(self, filepattern, max_count=5000):
        assert filepattern.count("*") == 1
        self.filepattern = filepattern
        self.max_count = max_count
        self._file_count = 0
        self._line_count = 0
        self._file = None

    def _open(self):
        filename = self.filepattern.replace("*", str(self._file_count))
        print(f"writing to {filename}")
        return open(filename, "wt")

    def __enter__(self):
        assert self._file is None
        self._file = self._open()
        self._file_count += 1
        return self

    def __exit__(self, *args, **kwargs):
        self._file.close()
        self._file = None
        self._file_count = 0
        self._line_count = 0

    def write(self, name, sequence):
        if self._line_count == self.max_count:
            self._file.close()
            self._file = self._open()
            self._file_count += 1
            self._line_count = 0
        self._file.write(">")
        self._file.write(name)
        self._file.write("\n")
        self._file.write(sequence)
        self._file.write("\n")
        self._line_count += 1
