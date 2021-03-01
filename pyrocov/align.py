import math

import mappy

# Source: https://samtools.github.io/hts-specs/SAMv1.pdf
CIGAR_CODES = "MIDNSHP=X"  # Note minimap2 uses only "MIDNSH"


class Differ:
    def __init__(self, ref, lb=0, ub=math.inf, **kwargs):
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
