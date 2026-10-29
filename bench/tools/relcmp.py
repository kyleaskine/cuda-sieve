"""Compare two relation files (a,b:side0:side1, hex primes; .gz or plain).

Usage: relcmp.py A B [lim0 lim1]
Prints raw/unique counts and the intersection, and for each exclusive set the
mean log2 of its p<96 part per side and its large-prime (p > lim) mix."""
import gzip
import math
import sys
from collections import Counter

SMALL = 96


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    rels, raw = {}, 0
    with op(path, "rt") as f:
        for line in f:
            if not line.strip() or line[0] == "#":
                continue
            ab, s0, s1 = line.strip().split(":")
            a, b = ab.split(",")
            raw += 1
            rels.setdefault((int(a), int(b)), (s0, s1))
    return raw, rels


def show(name, rels, keys, lim):
    n = len(keys)
    if not n:
        print(f"{name}: empty")
        return
    small, large = [0.0, 0.0], [Counter(), Counter()]
    for k in keys:
        for side, fl in enumerate(rels[k]):
            ps = [int(x, 16) for x in fl.split(",")] if fl else []
            small[side] += sum(math.log2(p) for p in ps if p < SMALL)
            large[side][sum(1 for p in ps if p > lim[side])] += 1
    print(f"{name}: n={n}")
    for side in (0, 1):
        dist = " ".join(f"{k}LP:{v / n:.3f}" for k, v in sorted(large[side].items()))
        print(f"  side {side}: mean log2(p<{SMALL} part) {small[side] / n:.2f}   {dist}")


lim = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (14000000, 18000000)
ra, A = load(sys.argv[1])
rb, B = load(sys.argv[2])
both, aonly, bonly = A.keys() & B.keys(), A.keys() - B.keys(), B.keys() - A.keys()
print(f"A: raw {ra} unique {len(A)}   ({sys.argv[1]})")
print(f"B: raw {rb} unique {len(B)}   ({sys.argv[2]})")
print(f"shared {len(both)}  A-only {len(aonly)}  B-only {len(bonly)}")
show("shared (A's lists)", A, both, lim)
show("A-only", A, aonly, lim)
show("B-only", B, bonly, lim)
