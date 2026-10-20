#!/usr/bin/env python3
"""Finding 120, review: how much of each arm's idle change the change itself
earned. Same window and gap rule as gaps.py, but every (last kernel, next
kernel) pair is summed rather than the top 12, and the pairs are split into
the two the change acts on and everything else. q in the window counts
k_resieve_scatter launches (one per slab) over the arm's slabs per q.

usage: gapsplit.py SLABS_PER_Q before.sqlite after.sqlite"""
import sqlite3, sys, collections

TOUCHED = {("k_td_record_warp", "k_cand_stats"), ("k_record_scan", "k_fill_rows")}

def gaps(p, spq):
    db = sqlite3.connect(p)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = sorted((s, e, names.get(n, str(n))) for s, e, n in
                db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    n = len(ks); lo, hi = ks[n // 10][0], ks[(9 * n) // 10][0]
    g = collections.Counter(); end_max = None; last = None; nslab = 0
    for s, e, nm in ks:
        if lo <= s <= hi:
            if nm == "k_resieve_scatter": nslab += 1
            if end_max is not None and s > end_max: g[(last, nm)] += s - end_max
        if end_max is None or e > end_max: end_max, last = e, nm
    nq = nslab / spq
    t = sum(v for k, v in g.items() if k in TOUCHED)
    return dict(win=hi - lo, nq=nq, idle=sum(g.values()), touched=t,
                pairs={k: g[k] for k in TOUCHED})

spq = float(sys.argv[1])
b, a = gaps(sys.argv[2], spq), gaps(sys.argv[3], spq)
for lbl, x in (("before", b), ("after", a)):
    print(f"  {lbl:6s} window {x['win']/1e9:.3f} s, {x['nq']:.0f} q: idle "
          f"{x['idle']/1e6:7.1f} ms ({100*x['idle']/x['win']:.2f}%), touched "
          f"{x['touched']/1e6:6.1f} ms ({100*x['touched']/x['win']:.2f}%)  " +
          ", ".join(f"{p}->{q} {v/1e6:.1f} ms" for (p, q), v in x['pairs'].items()))
d_idle = b['idle'] / b['nq'] - a['idle'] / a['nq']
d_t = b['touched'] / b['nq'] - a['touched'] / a['nq']
wall = a['win'] / a['nq']
print(f"  idle fell {d_idle/1e3:.0f} us/q; the two gaps the change acts on "
      f"{d_t/1e3:.0f} us/q ({100*d_t/wall:.2f}% of the after-trace's "
      f"{wall/1e6:.2f} ms/q in window), the rest {(d_idle-d_t)/1e3:.0f} us/q")
