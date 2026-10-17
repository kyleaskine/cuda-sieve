#!/bin/sh
# Final source (sm_120 snapshot) against fef825c, pipelined (the default),
# interleaved, two rounds, SM clock and power logged; NFS also single-queued
# for the stage times.
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench; B=$W/bench.fef825c
P=$R/work/applyprof
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 80"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 60"
cd $R/bench || exit 1
run() {  # job arm extra binary
  job=$1; arm=$2; extra=$3; bin=$4
  eval "A=\$$job"
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!
  $bin $A $extra > $W/ft.$job.$arm.out 2>&1
  kill $SMI
  o=$W/ft.$job.$arm.out
  g() { grep -h "$1" $o | head -1 | awk '{print $NF=="ms" ? $(NF-1) : $NF}'; }
  printf "%-4s %-11s wall %8s | xform %6s resieve %6s | %s load=%s\n" $job $arm \
    "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" "$(g 'transform + plattice')" "$(g 'resieve + scatter')" \
    "$(awk -F', ' 'NR>4 {c+=$1; p+=$2; n++} END {printf "sm=%.0f p=%.0fW", c/n, p/n}' $W/smi.tmp)" "$(cut -d' ' -f1 /proc/loadavg)"
}
for round in 1 2; do
  for job in NFS C194 C183 C143; do
    run $job base "" $B; run $job new "" $X
  done
  run NFS base.single --no-q-pipeline $B; run NFS new.single --no-q-pipeline $X
done
rm -f $W/smi.tmp
