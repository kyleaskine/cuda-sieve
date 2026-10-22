# Medians and per-round paired changes for timing.sh's output (timing.txt).
import re, statistics as st, sys
d = {}
for l in open(sys.argv[1] if len(sys.argv) > 1 else 'timing.txt'):
    r = l.split()
    if len(r) < 8 or r[1] not in ('sq', 'pl') or not r[3].startswith('r'): continue
    job, mode, arm, rd = r[0], r[1], r[2], r[3]
    d.setdefault((job, mode, arm), {})[rd] = (float(r[5]), float(r[7]), int(r[9]))
def med(job, mode, arm, k): return st.median(v[k] for v in d[(job, mode, arm)].values())
def pct(a, b): return 100 * (a / b - 1)
print("median over rounds, ms/q. base = installed HEAD fat build (f37fe64d); new = 38bb3d30 at its default;")
print("slices = 38bb3d30 --no-tiny-fold (finding 128 alone). Relations identical in every arm (timing.sh checks).\n")
print("single-queued apply")
for job in ('C135', 'I16', 'C1147'):
    if (job, 'sq', 'base') not in d: continue
    b, n, s = (med(job, 'sq', a, 1) for a in ('base', 'new', 'slices'))
    print(f"  {job:7s} base {b:8.3f}  slices {s:8.3f} ({pct(s,b):+5.1f}%)  new {n:8.3f} ({pct(n,b):+5.1f}%)")
print("\npipelined wall (COMPLETE)                                   new vs base by round")
for job in ('C135', 'C143', 'C183', 'I16', 'C194', 'AS276', 'SNFS301', 'C1147'):
    if (job, 'pl', 'base') not in d: continue
    b, n, s = (med(job, 'pl', a, 0) for a in ('base', 'new', 'slices'))
    rounds = sorted(d[(job, 'pl', 'base')])
    pr = " ".join(f"{pct(d[(job,'pl','new')][k][0], d[(job,'pl','base')][k][0]):+.1f}" for k in rounds if k in d[(job,'pl','new')])
    print(f"  {job:7s} base {b:8.2f}  slices {s:8.2f} ({pct(s,b):+5.1f}%)  new {n:8.2f} ({pct(n,b):+5.1f}%) | {pr}")
