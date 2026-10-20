#!/usr/bin/env python3
"""For idle gaps charged to (A -> B), list the host CUDA runtime calls made
inside the gap, as offsets from the gap start, for a few sample gaps, and the
mean share of the gap spent inside CUDA calls vs outside any call.
usage: gapdetail.py file.sqlite A B [nsamples]"""
import sqlite3, sys, bisect, collections
p, A, B = sys.argv[1:4]; ns = int(sys.argv[4]) if len(sys.argv) > 4 else 3
db = sqlite3.connect(p)
names = dict(db.execute("SELECT id, value FROM StringIds"))
ks = sorted((s, e, names.get(n, str(n))) for s, e, n in
            db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
api = sorted((s, e, names.get(n, str(n))) for s, e, n in
             db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
starts = [a[0] for a in api]
n = len(ks); lo, hi = ks[n // 10][0], ks[(9 * n) // 10][0]
gaps = []; end_max = None; last = None
for s, e, nm in ks:
    if lo <= s <= hi and end_max is not None and s > end_max and last == A and nm == B:
        gaps.append((end_max, s))
    if end_max is None or e > end_max: end_max, last = e, nm
tot = sum(b - a for a, b in gaps); incall = 0
bycall = collections.Counter()
for g0, g1 in gaps:
    i = max(bisect.bisect_left(starts, g0) - 50, 0)
    for s, e, nm in api[i:bisect.bisect_left(starts, g1)]:
        ov = min(e, g1) - max(s, g0)
        if ov > 0: incall += ov; bycall[nm.split('_v')[0]] += ov
print(f"{A} -> {B}: {len(gaps)} gaps, mean {tot/len(gaps)/1e3:.1f} us; "
      f"inside CUDA calls {100*incall/tot:.0f}%, outside {100-100*incall/tot:.0f}%")
for nm, t in bycall.most_common(8):
    print(f"  {nm:32s} {t/len(gaps)/1e3:7.1f} us/gap")
step = max(len(gaps) // ns, 1)
for g0, g1 in gaps[::step][:ns]:
    print(f"-- sample gap {(g1-g0)/1e3:.0f} us")
    i = max(bisect.bisect_left(starts, g0) - 50, 0)
    for s, e, nm in api[i:bisect.bisect_left(starts, g1)]:
        if e > g0:
            print(f"   +{(s-g0)/1e3:7.1f} us  {(e-s)/1e3:6.1f} us  {nm.split('_v')[0]}")
