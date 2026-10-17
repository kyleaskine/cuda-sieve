#!/bin/sh
# The other geometries: C143 (I13, one slab), c183 (I15), C194 (I16, four
# slabs). base = fef825c, rs0 = fast transform only, t512 = + coarse filter at
# 512 threads. Pipelined (the default) for wall; C143 also single-queued.
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench; B=$W/bench.fef825c
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 1000"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 60"
cd $R/bench || exit 1
run() {  # job label extra env... binary
  job=$1; label=$2; extra=$3; shift 3
  eval "A=\$$job"
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!
  env "$@" $A $extra > $W/ab2.$job.$label.out 2>&1
  kill $SMI
  o=$W/ab2.$job.$label.out
  g() { grep -h "$1" $o | head -1 | awk '{print $NF=="ms" ? $(NF-1) : $NF}'; }
  printf "%-4s %-12s wall %7s | xform %6s resieve %6s | sm=%s load=%s\n" $job $label \
    "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" "$(g 'transform + plattice')" \
    "$(g 'resieve + scatter')" "$(awk 'NR>4 {c+=$1; n++} END {printf "%.0f", c/n}' $W/smi.tmp)" "$(cut -d' ' -f1 /proc/loadavg)"
}
for round in 1 2; do
  for job in C143 C183 C194; do
    run $job base "" $B
    run $job rs0 "" BENCH_TEST_RS=0 $X
    run $job t512 "" BENCH_TEST_RS_THREADS=512 $X
  done
  run C143 base.single --no-q-pipeline $B
  run C143 rs0.single --no-q-pipeline BENCH_TEST_RS=0 $X
  run C143 t512.single --no-q-pipeline BENCH_TEST_RS_THREADS=512 $X
done
rm -f $W/smi.tmp
