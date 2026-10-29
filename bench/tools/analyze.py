#!/usr/bin/env python3
"""Kernel-concurrency profile of an nsys sqlite export (CUDA trace).
Window: the middle 80% of kernel launches by count (steady state).
Reports busy/idle, time at each concurrency depth, compression (summed
kernel time / time with >= 1 kernel running), idle gaps by length, and per
kernel: total time, launches, mean duration, and the fraction of its run
time overlapped by at least one other kernel."""
import sqlite3, sys, collections

def load(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = [(s, e, names.get(n, str(n))) for s, e, n in
          db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    other = []
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try: other += [(s, e) for s, e in db.execute(f"SELECT start, end FROM {t}")]
        except sqlite3.OperationalError: pass
    return sorted(ks), other

def union(iv, lo, hi):
    tot = 0; cur_s = cur_e = None
    for s, e in sorted((max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi):
        if cur_e is None or s > cur_e:
            if cur_e is not None: tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else: cur_e = max(cur_e, e)
    if cur_e is not None: tot += cur_e - cur_s
    return tot

def main(path, label):
    ks, other = load(path)
    n = len(ks)
    lo, hi = ks[n // 10][0], ks[(9 * n) // 10][0]
    win = hi - lo
    ev = []
    for i, (s, e, nm) in enumerate(ks):
        s, e = max(s, lo), min(e, hi)
        if e > s: ev += [(s, 1, i), (e, -1, i)]
    ev.sort(key=lambda x: (x[0], x[1]))
    depth_t = collections.Counter(); run = set(); t_prev = lo
    ovl = collections.Counter(); tot = collections.Counter(); cnt = collections.Counter()
    gaps = []; gap_start = lo
    for t, d, i in ev:
        dt = t - t_prev
        if dt > 0:
            depth_t[min(len(run), 3)] += dt
            if len(run) >= 2:
                for j in run: ovl[ks[j][2]] += dt
        t_prev = t
        if d > 0:
            if not run and t > gap_start: gaps.append(t - gap_start)
            run.add(i)
        else:
            run.discard(i)
            if not run: gap_start = t
    if hi > t_prev: depth_t[min(len(run), 3)] += hi - t_prev
    for s, e, nm in ks:
        c = min(e, hi) - max(s, lo)
        if c > 0: tot[nm] += c; cnt[nm] += 1
    busy = win - depth_t[0]
    ksum = sum(tot.values())
    busy_any = union([(s, e) for s, e, _ in ks] + other, lo, hi)
    print(f"== {label}: window {win/1e9:.3f} s, {sum(cnt.values())} kernels")
    print(f"  kernel busy {100*busy/win:.1f}%  (with copies/memsets {100*busy_any/win:.1f}%)"
          f"   depth 0/1/2/3+: " + " / ".join(f"{100*depth_t[d]/win:.1f}%" for d in range(4)))
    print(f"  compression (summed kernel time / kernel-busy time): {ksum/busy:.2f}x")
    b = collections.Counter()
    for g in gaps:
        b['<10us' if g < 1e4 else '10-100us' if g < 1e5 else '0.1-1ms' if g < 1e6 else '>1ms'] += g
    print("  idle by gap length: " + "  ".join(f"{k} {100*b[k]/win:.1f}%" for k in ('<10us','10-100us','0.1-1ms','>1ms')))
    print(f"  {'kernel':38s} {'share':>6s} {'launches':>8s} {'mean us':>8s} {'overlapped':>10s}")
    for nm, t in tot.most_common(16):
        print(f"  {nm[:38]:38s} {100*t/ksum:5.1f}% {cnt[nm]:8d} {t/cnt[nm]/1e3:8.1f} {100*ovl[nm]/t:9.1f}%")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p, p.rsplit('/', 1)[-1].replace('.sqlite', ''))
