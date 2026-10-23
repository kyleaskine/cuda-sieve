# Medians over rounds for stageb/timing.sh: single-queued apply and pipelined
# wall per job, each arm against base, plus the per-round change of `new`.
import statistics as st, sys
d = {}
for l in open(sys.argv[1] if len(sys.argv) > 1 else 'timing/timing.txt'):
    r = l.split()
    if len(r) < 8 or r[1] not in ('sq', 'pl') or not r[3].startswith('r'): continue
    d.setdefault((r[0], r[1], r[2]), {})[r[3]] = (float(r[5]), float(r[7]))
arms = ('base', 'logrec', 'notable', 'new')
def med(j, m, a, k): return st.median(v[k] for v in d[(j, m, a)].values())
for mode, k, title in (('sq', 1, 'single-queued apply'), ('pl', 0, 'pipelined wall (COMPLETE)')):
    print(f"\n{title}, median over rounds (ms/q); change against base")
    for job in ('C135', 'C183', 'I16', 'C194', 'C1147'):
        if (job, mode, 'base') not in d: continue
        b = med(job, mode, 'base', k)
        cells = [f"{a} {med(job, mode, a, k):8.3f} ({100*(med(job, mode, a, k)/b-1):+5.1f}%)" for a in arms[1:]]
        rds = sorted(d[(job, mode, 'base')])
        pr = " ".join(f"{100*(d[(job,mode,'new')][x][k]/d[(job,mode,'base')][x][k]-1):+.1f}" for x in rds if x in d[(job, mode, 'new')])
        print(f"  {job:6s} base {b:8.3f} | " + " | ".join(cells) + f" | new by round {pr}")
