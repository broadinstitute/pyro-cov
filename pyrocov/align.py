# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import base64
import filecmp
import hashlib
import logging
import math
import os
import pickle
import shutil
import subprocess
import warnings
from collections import Counter, defaultdict

from .usher import load_usher_clades, refine_mutation_tree
from .util import open_tqdm

logger = logging.getLogger(__name__)

# Source: https://samtools.github.io/hts-specs/SAMv1.pdf
CIGAR_CODES = "MIDNSHP=X"  # Note minimap2 uses only "MIDNSH"

ROOT = os.path.dirname(os.path.dirname(__file__))
NEXTCLADE_DATA = os.path.join(ROOT, "results", "nextclade_data")
PANGOLEARN_DATA = os.path.expanduser("~/github/cov-lineages/pangoLEARN/pangoLEARN/data")


def _log_call(*args):
    log_filename = os.path.join("results/aligndb", os.path.basename(args[0]) + ".log")
    logger.info(" ".join(args) + " > " + log_filename)
    with open(log_filename, "at") as f:
        subprocess.run(args, stdout=f, stderr=f, check=True)


def fingerprint_sequence(seq: str) -> str:
    """
    Create a 12-character (60-bit) fingerprint, safe for up to 1 billion sequences.
    """
    hasher = hashlib.sha1()
    hasher.update(seq.replace("\n", "").encode("utf-8"))
    return base64.b32encode(hasher.digest())[:12].decode("utf-8")


class AlignDB:
    """
    Database to cache results of nextclade and usher, so that only new samples
    need to be aligned.
    """

    def __init__(
        self,
        dirname="results/aligndb",
        usher_proto=f"{PANGOLEARN_DATA}/lineageTree.pb",
    ):
        self.dirname = os.path.realpath(dirname)

        # Initialize with a refined version of lineageTrees.pb, supporting both
        # standard pango lineage use and finer downstream use.
        self.coarse_proto = os.path.join(self.dirname, "lineageTree.coarse.pb")
        self.fine_proto = os.path.join(self.dirname, "lineageTree.fine.pb")
        self.refine_filename = os.path.join(self.dirname, "refine.pkl")
        if not os.path.exists(self.dirname):
            logger.info("Initializing AlignDB")
            os.makedirs(self.dirname)
            shutil.copy2(usher_proto, self.coarse_proto)
            self.fine_to_coarse = refine_mutation_tree(
                self.coarse_proto, self.fine_proto
            )
            with open(self.refine_filename, "wb") as f:
                pickle.dump(self.fine_to_coarse, f)
        else:
            if not filecmp.cmp(usher_proto, self.coarse_proto):
                warnings.warn("AlignDB is using an old lineageTree.pb")
            with open(self.refine_filename, "rb") as f:
                self.fine_to_coarse = pickle.load(f)

        self.fasta_filename = os.path.join(self.dirname, "temp.fasta")
        self.rows_temp_filename = os.path.join(self.dirname, "temp.rows.tsv")
        self.bad_temp_filename = os.path.join(self.dirname, "temp.bad.tsv")
        self.usher_fasta_filename = self.fasta_filename.replace(
            ".fasta", ".usher.fasta"
        )
        self.vcf_filename = os.path.join(self.dirname, "temp.vcf")
        self.clades_filename = os.path.join(self.dirname, "clades.txt")
        self.tsv_filename = os.path.join(self.dirname, "temp.tsv")
        self.header_filename = os.path.join(self.dirname, "header.tsv")
        self.rows_filename = os.path.join(self.dirname, "rows.tsv")
        self.bad_filename = os.path.join(self.dirname, "bad.tsv")

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
        if os.path.exists(self.bad_filename):
            with open(self.bad_filename) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._already_aligned.add(line)

    def schedule(self, sequence, *fn_args):
        """
        Schedule a task for a given input ``sequence``.
        This asynchronously calls::

            fn_args[0](*fn_args[1:], row)

        where ``row`` is a dictionary with nextclade columns plus the
        additional columns:

        - ``lineage`` a coarse pango lineage.
        - ``lineages`` a comma-deliminted string of ambiguous pango lineages.
        - ``clade`` a fine clade name e.g. ``fine.1...3.2``.
        - ``clades`` a comma-deliminted string of ambiguous clades.
        """
        key = fingerprint_sequence(sequence)
        if key not in self._already_aligned:
            self._schedule_alignment(key, sequence)
        self._tasks[key].append(fn_args)

    def maybe_schedule(self, sequence, *fn_args):
        """
        Schedule a task iff no new alignment work is required.
        Tasks requiring new alignment work will be silently dropped.
        """
        key = fingerprint_sequence(sequence)
        if key in self._already_aligned:
            self._tasks[key].append(fn_args)

    def wait(self):
        """
        Wait for all scheduled or maybe_scheduled tasks to complete.
        """
        self._flush()
        with open(self.header_filename) as f:
            header = f.read().strip().split("\t")
        logger.info(f"Processing {len(self._tasks)} sequences")
        for row in open_tqdm(self.rows_filename):
            row = row.strip().split("\t")
            key = row[0]
            assert len(row) == len(header)
            row = dict(zip(header, row))
            for fn_args in self._tasks.pop(key, []):
                fn, args = fn_args[0], fn_args[1:]
                fn(*args, row)

        num_skipped = sum(map(len, self._tasks.values()))
        logger.info(f"Skipped {num_skipped} sequences")
        self._tasks.clear()

    def repair(self):
        logger.info(f"Repairing {self.rows_filename}")
        with open(self.header_filename) as f:
            header = f.read().strip().split("\t")
        bad = Counter()
        temp_filename = self.rows_filename + ".temp"
        with open(temp_filename, "wt") as fout:
            for line in open_tqdm(self.rows_filename):
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
        max_fasta_count = 4000  # Avoid nextclade file size limit.
        if len(self._pending) >= max_fasta_count:
            self._flush()

    def _flush(self):
        if not self._pending:
            return
        self._fasta_file.close()

        # Align via nextclade.
        _log_call(
            "./nextclade",
            f"--input-root-seq={NEXTCLADE_DATA}/reference.fasta",
            "--genes=E,M,N,ORF1a,ORF1b,ORF3a,ORF6,ORF7a,ORF7b,ORF8,ORF9b,S",
            f"--input-gene-map={NEXTCLADE_DATA}/genemap.gff",
            f"--input-tree={NEXTCLADE_DATA}/tree.json",
            f"--input-qc-config={NEXTCLADE_DATA}/qc.json",
            f"--input-pcr-primers={NEXTCLADE_DATA}/primers.csv",
            f"--input-fasta={self.fasta_filename}",
            f"--output-tsv={self.tsv_filename}",
            f"--output-dir={self.dirname}",
        )

        # Classify pango lineages.
        aligned_filename = self.fasta_filename.replace(".fasta", ".aligned.fasta")
        if os.stat(aligned_filename).st_size == 0:
            # No sequences could be aligned; skip.
            fingerprint_to_lineage = {}
        else:
            # Concatenate reference to aligned sequences.
            with open(self.usher_fasta_filename, "wt") as fout:
                with open(f"{NEXTCLADE_DATA}/reference.fasta") as fin:
                    shutil.copyfileobj(fin, fout)
                with open(aligned_filename) as fin:
                    shutil.copyfileobj(fin, fout)

            # Convert aligned fasta to vcf for usher.
            _log_call("faToVcf", self.usher_fasta_filename, self.vcf_filename)

            # Run Usher.
            _log_call(
                "usher",
                "-n",
                "-D",
                "-i",
                self.fine_proto,
                "-v",
                self.vcf_filename,
                "-d",
                self.dirname,
            )
            fingerprint_to_lineage = load_usher_clades(self.clades_filename)

        # Append to a copy to ensure atomicity.
        self._already_aligned.update(self._pending)
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
                        cc = fingerprint_to_lineage.get(fingerprint)
                        if cc is None:
                            continue  # skip row
                        clade, clades = cc
                        lineage = self.fine_to_coarse[clade]
                        lineages = ",".join(
                            sorted(
                                set(self.fine_to_coarse[c] for c in clades.split(","))
                            )
                        )
                        tab = "\t" * (num_cols - 4 - line.count("\t"))
                        line = f"{line}{tab}{lineage}\t{lineages}\t{clade}\t{clades}\n"
                        assert line.count("\t") + 1 == num_cols
                        frows.write(line)
                        self._pending.remove(fingerprint)
                    else:
                        with open(self.header_filename, "w") as fheader:
                            line = f"{line}\tlineage\tlineages\tclade\tclades\n"
                            fheader.write(line)
                            num_cols = line.count("\t") + 1
        os.rename(self.rows_temp_filename, self.rows_filename)  # atomic
        if self._pending:
            logger.info(f"Failed to align {len(self._pending)} sequences")
            if os.path.exists(self.bad_filename):
                shutil.copyfile(self.bad_filename, self.bad_temp_filename)
            with open(self.bad_temp_filename, "a") as f:
                for fingerprint in self._pending:
                    f.write(fingerprint + "\n")
            os.rename(self.bad_temp_filename, self.bad_filename)  # atomic
        os.remove(self.fasta_filename)
        os.remove(self.tsv_filename)
        self._fasta_file = open(self.fasta_filename, "w")
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


class Differ:
    """
    Genetic sequence differ based on mappy.
    """

    def __init__(self, ref, lb=0, ub=math.inf, **kwargs):
        import mappy

        self.ref = ref
        self.lb = lb
        self.ub = ub
        self.aligner = mappy.Aligner(seq=ref, **kwargs)

    def diff(self, seq):
        ref = self.ref
        lb = self.lb
        ub = self.ub
        diff = []
        for hit in self.aligner.map(seq):
            ref_pos = hit.r_st
            if ref_pos < lb:
                continue
            if ref_pos >= ub:
                break

            seq_pos = hit.q_st
            for size, code in hit.cigar:
                if code == 0:  # M
                    if seq[seq_pos : seq_pos + size] != ref[ref_pos : ref_pos + size]:
                        for i in range(min(size, ub - ref_pos)):
                            s = seq[seq_pos + i]
                            if s != "N" and s != ref[ref_pos + i]:
                                diff.append((ref_pos + i, "X", s))
                elif code == 1:  # I
                    diff.append((ref_pos, "I", seq[seq_pos : seq_pos + size]))
                elif code == 2:  # D
                    diff.append((ref_pos, "D", size))
                ref_pos += size
                seq_pos += size

        return diff
