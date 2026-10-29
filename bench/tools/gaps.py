#!/usr/bin/env python3
"""Where the GPU idles: every gap with no kernel running (steady-state
window as analyze.py), charged to the (last kernel to finish, next kernel
to start) pair, plus the CUDA runtime calls in progress on the host when
the gap ends (what the host was doing while the card waited)."""
import sqlite3, sys, collections
p = sys.argv[1]
db = sqlite3.connect(p)
names = dict(db.execute("SELECT id, value FROM StringIds"))
ks = sorted((s, e, names.get(n, str(n))) for s, e, n in
            db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
api = sorted((s, e, names.get(n, str(n))) for s, e, n in
             db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
n = len(ks); lo, hi = ks[n // 10][0], ks[(9 * n) // 10][0]
win = hi - lo
gaps = collections.Counter(); gcnt = collections.Counter(); during = collections.Counter()
end_max = None; last_name = None
import bisect
api_starts = [a[0] for a in api]
for s, e, nm in ks:
    if s < lo or s > hi: 
        if end_max is None or e > end_max: end_max, last_name = e, nm
        continue
    if end_max is not None and s > end_max:
        g = s - end_max
        key = (last_name, nm); gaps[key] += g; gcnt[key] += 1
        # host API calls overlapping the gap, charged by their overlap
        i = bisect.bisect_left(api_starts, end_max) - 200
        for a_s, a_e, a_n in api[max(i, 0):bisect.bisect_left(api_starts, s)]:
            ov = min(a_e, s) - max(a_s, end_max)
            if ov > 0: during[a_n] += ov
    if end_max is None or e > end_max: end_max, last_name = e, nm
tot = sum(gaps.values())
print(f"== {p}: idle {tot/1e9:.3f} s of {win/1e9:.3f} s window ({100*tot/win:.1f}%)")
print(f"  {'last kernel -> next kernel':60s} {'idle':>7s} {'gaps':>6s} {'mean us':>8s}")
for (a, b), g in gaps.most_common(12):
    print(f"  {(a+' -> '+b)[:60]:60s} {100*g/win:6.1f}% {gcnt[(a,b)]:6d} {g/gcnt[(a,b)]/1e3:8.1f}")
print("  host CUDA calls in progress during the idle time (share of idle):")
for a, g in during.most_common(8):
    print(f"    {a[:50]:50s} {100*g/tot:6.1f}%")
