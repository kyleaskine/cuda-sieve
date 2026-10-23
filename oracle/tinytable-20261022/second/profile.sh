#!/bin/sh
# stageb2 (every warp shares the table build): did the barrier stall go? ncu --set full with source counters on two
# k_apply launches of the NFS@Home I16 job (q index 3, slab 0: side 1, then
# side 0; single-queued so the launch order is fixed, as in finding 110) for
# logrec (records carry log p), notable (stage B binary, --no-tiny-table) and
# new (stage B at its default). Then a summary of duration, instructions,
# issue-slot use and the barrier stall per launch.
R=/home/kylea/code/cuda-sieve; W=$R/work/stageb2/prof
N=$R/work/stageb2/src/bench/bench
A="--pipeline --cofactor --poly $R/work/applyprof/nfs.job --fb1 $R/work/applyprof/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 5 --no-q-pipeline"
cd $R/work/stageb2/src/bench || exit 1
fail=0
for arm in notable new; do
  case $arm in notable) bin=$N; ax=--no-tiny-table ;; new) bin=$N; ax= ;; esac
  ncu --kernel-name regex:k_apply --launch-skip 24 --launch-count 2 --set full --import-source yes \
      -f -o $W/$arm $bin $A $ax > $W/$arm.log 2>&1 || fail=$((fail+1))
  ncu --import $W/$arm.ncu-repz --page details --csv > $W/$arm.details.csv 2>/dev/null || fail=$((fail+1))
  for k in 0 1; do
    ncu --import $W/$arm.ncu-repz --page source --csv --print-source cuda,sass --launch-skip $k --launch-count 1 \
        > $W/$arm.L$k.csv 2>/dev/null || fail=$((fail+1))
  done
done
python3 - "$W" <<'PY'
import csv, sys, collections
W = sys.argv[1]
want = ['Duration', 'Executed Instructions', 'Issue Slots Busy', 'Achieved Occupancy',
        'Registers Per Thread', 'Dynamic Shared Memory Per Block', 'Block Limit Shared Mem']
for arm in ('notable', 'new'):
    rows = list(csv.DictReader(open(f'{W}/{arm}.details.csv')))
    per = collections.OrderedDict()
    for r in rows:
        per.setdefault(r['ID'], {})[r['Metric Name']] = (r['Metric Value'], r['Metric Unit'])
    for i, (lid, m) in enumerate(per.items()):
        print(f"{arm:8s} launch {i}: " + "; ".join(f"{k} {m[k][0]} {m[k][1]}".strip() for k in want if k in m))
PY
echo "done: $fail failed"
[ $fail -eq 0 ]
