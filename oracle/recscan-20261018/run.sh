#!/bin/sh
# Side-0 record-scan measurement (test-only recscan.diff build): per side the
# bucket records per slab and per-region fill against cap, and, single-queued
# (where side 0's records are still in the buckets at TD), side 0's records on
# two-sided survivors against the side-0 resieve's pairs, survivor by survivor.
R=/home/kylea/code/cuda-sieve; W=$R/work/recscan; X=${X:-$W/stat/bench/bench}
P=$R/work/applyprof
NFS="--poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 20"
C194="--poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 20"
C183="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 50"
C183S0="$C183 --sq-side 0"
C143="--poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 300"
cd $R/bench || exit 1
fails=0
for job in NFS C194 C183 C183S0 C143; do
  eval "A=\$$job"
  o=$W/$job.single.out
  $X --pipeline --no-q-pipeline $A > $o 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  echo "== $job single-queued (rc=$rc)"; grep -h "^RECSCAN\|j-slabbing\|OVERFLOW" $o
done
for job in NFS C143; do
  eval "A=\$$job"
  o=$W/$job.pipe.out
  $X --pipeline $A > $o 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  echo "== $job pipelined (rc=$rc)"; grep -h "^RECSCAN" $o
done
echo "done: $fails failed run(s)"
[ "$fails" -eq 0 ]
