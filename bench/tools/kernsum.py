#!/usr/bin/env python3
"""Per-kernel GPU time per special-q from an nsys sqlite export, over the
same steady-state window as analyze.py (middle 80% of kernel launches).
q in the window = transform launches / 2 (one per side per q in both
programs). Prints ms/q per kernel, the summed kernel time, and the
kernel-busy wall per q."""
import sqlite3, sys, collections
def main(p, xform):
    db = sqlite3.connect(p)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = sorted((s, e, names.get(n, str(n))) for s, e, n in
                db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    n = len(ks); lo, hi = ks[n // 10][0], ks[(9 * n) // 10][0]
    tot = collections.Counter(); cnt = collections.Counter()
    for s, e, nm in ks:
        if lo <= s < hi:
            tot[nm] += min(e, hi) - s; cnt[nm] += 1
    nq = cnt[xform] / 2
    print(f"== {p}: window {(hi-lo)/1e9:.3f} s, {nq:.0f} q, wall {(hi-lo)/1e6/nq:.1f} ms/q, "
          f"summed kernels {sum(tot.values())/1e6/nq:.1f} ms/q")
    for nm, t in tot.most_common():
        print(f"  {nm:<28} {t/1e6/nq:8.2f} ms/q  {cnt[nm]/nq:6.2f} launches/q")
for p, x in (("cuda-single.sqlite", "k_transform"), ("cuda-qp.sqlite", "k_transform"),
             ("cado-slots8.sqlite", "transform_kernel")):
    main(p, x)
