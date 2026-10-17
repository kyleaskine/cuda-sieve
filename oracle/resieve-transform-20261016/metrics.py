#!/usr/bin/env python3
"""Headline ncu metrics per launch from a report (raw page, CSV)."""
import csv, subprocess, sys
M = ["gpu__time_duration.sum", "launch__grid_size", "launch__block_size",
     "launch__registers_per_thread", "sm__warps_active.avg.pct_of_peak_sustained_active",
     "smsp__issue_active.avg.pct_of_peak_sustained_active", "smsp__inst_executed.sum",
     "sm__throughput.avg.pct_of_peak_sustained_elapsed",
     "lts__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_sectors.sum",
     "lts__t_sector_hit_rate.pct", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
     "l1tex__t_sector_hit_rate.pct", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
     "smsp__thread_inst_executed_per_inst_executed.ratio"]
out = subprocess.run(["ncu", "--import", sys.argv[1], "--page", "raw", "--csv"],
                     capture_output=True, text=True).stdout
r = list(csv.reader(out.splitlines()))
h = r[0]; rows = [x for x in r[2:] if x and x[0].isdigit()]
for m in M:
    if m in h:
        i = h.index(m); print(f"{m:<58}", "  ".join(x[i] for x in rows))
    else: print(f"{m:<58} (absent)")
