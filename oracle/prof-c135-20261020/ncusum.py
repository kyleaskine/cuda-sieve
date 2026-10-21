#!/usr/bin/env python3
"""Per-kernel means from an ncu --page raw CSV: isolated duration, SM and
memory throughput (% of peak), achieved occupancy (warps active % of peak),
grid, block, registers, waves per SM, L2 hit rate. Usage: ncusum.py CSV
[launches-per-q kernel=count ...] to add an isolated ms/q column."""
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
hdr, data = rows[0], rows[2:]
col = {h: i for i, h in enumerate(hdr)}
per = dict(a.split('=') for a in sys.argv[2:])
acc = collections.defaultdict(list)
for r in data:
    acc[r[col["Kernel Name"]]].append(r)
def m(rs, k):
    v = [float(r[col[k]].replace(',', '')) for r in rs if r[col[k]] not in ('', 'n/a')]
    return sum(v) / len(v) if v else float('nan')
print(f"{'kernel':<26}{'n':>3}{'us':>9}{'SM%':>6}{'mem%':>6}{'occ%':>6}{'grid':>8}{'blk':>5}"
      f"{'reg':>5}{'waves':>7}{'L2hit':>6}{'ms/q':>7}")
tot = 0.0
for k, rs in sorted(acc.items(), key=lambda kv: -m(kv[1], "gpu__time_duration.sum")):
    us = m(rs, "gpu__time_duration.sum")
    mq = us * float(per[k]) / 1000 if k in per else float('nan')
    if k in per: tot += mq
    print(f"{k[:25]:<26}{len(rs):>3}{us:>9.1f}"
          f"{m(rs, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):>6.1f}"
          f"{m(rs, 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'):>6.1f}"
          f"{m(rs, 'sm__warps_active.avg.pct_of_peak_sustained_active'):>6.1f}"
          f"{m(rs, 'launch__grid_size'):>8.0f}{m(rs, 'launch__block_size'):>5.0f}"
          f"{m(rs, 'launch__registers_per_thread'):>5.0f}"
          f"{m(rs, 'launch__waves_per_multiprocessor'):>7.2f}"
          f"{m(rs, 'lts__t_sector_hit_rate.pct'):>6.1f}{mq:>7.3f}")
if per: print(f"isolated sum over the kernels given: {tot:.3f} ms/q")
