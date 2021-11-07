# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import os
import re
import shutil
from collections import Counter, defaultdict
from subprocess import check_call

from pyrocov import pangolin

logger = logging.getLogger(__name__)

NEXTSTRAIN_DATA = os.path.expanduser("~/github/nextstrain/nextclade/data/sars-cov-2")
PANGOLEARN_DATA = os.path.expanduser("~/github/cov-lineages/pangoLEARN/pangoLEARN/data")


def log_call(*args):
    logger.info(" ".join(args))
    return check_call(args)


def hash_sequence(seq):
    hasher = hashlib.sha1()
    hasher.update(seq.replace("\n", "").encode("utf-8"))
    return hasher.hexdigest()


def load_usher_clades(filename):
    with open(filename) as f:
        clades = dict(line.strip().split("\t") for line in f)

    # Disambiguate histograms like B.1.1.161*|B.1.1(2/3),B.1.1.161(1/3)
    # by choosing the most ancestral plurality.
    for name, lineage in list(clades.items()):
        if "|" in lineage:
            weights = {}
            for entry in lineage.split("|")[1].split(","):
                match = re.match(r"(.*)\((\d+)/\d+\)", entry)
                try:
                    lineage = pangolin.decompress(match.group(1))
                except ValueError:  # e.g. XA lineage
                    continue
                weights[lineage] = int(match.group(2))
            if weights:
                lineage = min(weights, key=lambda k: (-weights[k], k.count("."), k))
            else:  # e.g. XA lineage
                lineage = None
        if lineage:
            clades[name] = pangolin.compress(lineage)
        else:
            del clades[name]
    return clades


class NextcladeDB:
    """
    Database to cache results of nextclade and usher, so that only new samples
    need to be aligned.
    """

    def __init__(self, fileprefix="results/nextcladedb", max_fasta_count=4000):
        self.max_fasta_count = max_fasta_count
        fileprefix = os.path.realpath(fileprefix)
        self.fasta_filename = fileprefix + ".temp.fasta"
        self.output_dir = os.path.dirname(self.fasta_filename)
        self.rows_temp_filename = fileprefix + "rows.temp.tsv"
        self.usher_fasta_filename = self.fasta_filename.replace(
            ".fasta", ".usher.fasta"
        )
        self.vcf_filename = fileprefix + ".temp.vcf"
        self.usher_proto = f"{PANGOLEARN_DATA}/lineageTree.pb"
        self.clades_filename = os.path.join(self.output_dir, "clades.txt")
        self.tsv_filename = fileprefix + ".temp.tsv"
        self.header_filename = fileprefix + ".header.tsv"
        self.rows_filename = fileprefix + ".rows.tsv"

        self._fasta_file = open(self.fasta_filename, "wt")
        self._pending = set()
        self._tasks = defaultdict(list)

        # Load hashes of already-aligned sequences.
        self._already_aligned = set()
        if os.path.exists(self.rows_filename):
            with open(self.rows_filename) as f:
                for line in f:
                    key = line.split("\t", 1)[0]
                    self._already_aligned.add(key)

    def schedule(self, sequence, *fn_args):
        """
        Schedule a task for a given input ``sequence``.
        """
        key = hash_sequence(sequence)
        if key not in self._already_aligned:
            self._schedule_alignment(key, sequence)
        self._tasks[key].append(fn_args)

    def maybe_schedule(self, sequence, *fn_args):
        """
        Schedule a task iff no new alignment work is required.
        Tasks requiring new alignment work will be silently dropped.
        """
        key = hash_sequence(sequence)
        if key in self._already_aligned:
            self._tasks[key].append(fn_args)

    def wait(self, log_every=1000):
        """
        Wait for all scheduled or maybe_scheduled tasks to complete.
        """
        self._flush()
        with open(self.header_filename) as f:
            header = f.read().strip().split("\t")
        with open(self.rows_filename) as f:
            for i, row in enumerate(f):
                row = row.strip().split("\t")
                key = row[0]
                assert len(row) == len(header)
                row = dict(zip(header, row))
                for fn_args in self._tasks.pop(key, []):
                    fn, args = fn_args[0], fn_args[1:]
                    fn(*args, row)
                if log_every and i % log_every == 0:
                    print(".", end="", flush=True)

        num_skipped = sum(map(len, self._tasks.values()))
        logger.info(f"Skipped {num_skipped} sequences")
        self._tasks.clear()

    def repair(self, log_every=10000):
        logger.info(f"Repairing {self.rows_filename}")
        with open(self.header_filename) as f:
            header = f.read().strip().split("\t")
        bad = Counter()
        temp_filename = self.rows_filename + ".temp"
        with open(self.rows_filename) as fin, open(temp_filename, "wt") as fout:
            for i, line in enumerate(fin):
                if i % log_every == 0:
                    print(".", end="", flush=True)
                row = line.strip().split("\t")
                if len(row) < len(header):
                    bad[f"invalid length {len(header)} vs {len(row)}"] += 1
                    row = row[:-1] + [""] * (len(header) - len(row)) + row[-1:]
                    line = "\t".join(row) + "\n"
                fout.write(line)
        logger.info(f"Fixed {sum(bad.values())} errors:")
        if bad:
            for k, v in bad.most_common():
                logger.info(f"{v}\t{k}")
            logger.info(f"Next mv {temp_filename} {self.rows_filename}")

    def _schedule_alignment(self, key, sequence):
        if key in self._pending:
            return
        self._fasta_file.write(">")
        self._fasta_file.write(key)
        self._fasta_file.write("\n")
        self._fasta_file.write(sequence)
        self._fasta_file.write("\n")
        self._pending.add(key)
        if len(self._pending) >= self.max_fasta_count:
            self._flush()

    def _flush(self):
        if not self._pending:
            return
        self._fasta_file.close()

        # Align via nextclade.
        log_call(
            "./nextclade",
            f"--input-root-seq={NEXTSTRAIN_DATA}/reference.fasta",
            "--genes=E,M,N,ORF1a,ORF1b,ORF3a,ORF6,ORF7a,ORF7b,ORF8,ORF9b,S",
            f"--input-gene-map={NEXTSTRAIN_DATA}/genemap.gff",
            f"--input-tree={NEXTSTRAIN_DATA}/tree.json",
            f"--input-qc-config={NEXTSTRAIN_DATA}/qc.json",
            f"--input-pcr-primers={NEXTSTRAIN_DATA}/primers.csv",
            f"--input-fasta={self.fasta_filename}",
            f"--output-tsv={self.tsv_filename}",
            f"--output-dir={self.output_dir}",
        )

        # Concatenate reference to aligned sequences.
        with open(self.usher_fasta_filename, "wt") as fout:
            with open(f"{NEXTSTRAIN_DATA}/reference.fasta") as fin:
                shutil.copyfileobj(fin, fout)
            with open(self.fasta_filename.replace(".fasta", ".aligned.fasta")) as fin:
                shutil.copyfileobj(fin, fout)

        # Convert aligned fasta to vcf for usher.
        log_call("faToVcf", self.usher_fasta_filename, self.vcf_filename)

        # Run Usher.
        log_call(
            "usher",
            "-n",
            "-D",
            "-i",
            self.usher_proto,
            "-v",
            self.vcf_filename,
            "-d",
            self.output_dir,
        )
        fingerprint_to_lineage = load_usher_clades(self.clades_filename)

        # Append to a copy to ensure atomicity.
        if os.path.exists(self.rows_filename):
            shutil.copyfile(self.rows_filename, self.rows_temp_filename)
        with open(self.tsv_filename) as f:
            with open(self.rows_temp_filename, "a") as frows:
                num_cols = 0  # defined below
                for i, line in enumerate(f):
                    line = line.rstrip("\n")
                    if i:
                        fingerprint = line.split("\t", 1)[0]
                        assert " " not in fingerprint
                        lineage = fingerprint_to_lineage.get(fingerprint)
                        if lineage is None:
                            continue  # skip row
                        tab = "\t" * (num_cols - line.count("\t"))
                        frows.write(f"{line}{tab}{lineage}\n")
                    else:
                        with open(self.header_filename, "w") as fheader:
                            fheader.write(f"{line}\tlineage\n")
                            num_cols = line.count("\t") + 1
        os.rename(self.rows_temp_filename, self.rows_filename)
        os.remove(self.fasta_filename)
        os.remove(self.tsv_filename)
        self._fasta_file = open(self.fasta_filename, "w")
        self._already_aligned.update(self._pending)
        self._pending.clear()


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
