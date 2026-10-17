#!/bin/sh
# Timing A/B at I16 (NFS@Home job, h2h band's first 80 q), interleaved, two
# rounds. base = fef825c; rs0 = fast transform, old resieve; rs1 = + coarse
# filter (256 threads); t512 = coarse filter at 512 threads. Single-queued
# for stage times, pipelined for wall. SM clock and power averaged per run.
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench; B=$W/bench.fef825c
P=$R/work/applyprof
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 80"
cd $R/bench || exit 1
run() {  # label mode env... binary
  label=$1; mode=$2; shift 2
  extra=; [ $mode = single ] && extra=--no-q-pipeline
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!
  env "$@" $NFS $extra > $W/ab.$label.$mode.out 2>&1
  kill $SMI
  st=$(awk -F', ' 'NR>10 {c+=$1; p+=$2; n++} END {printf "sm=%.0f p=%.0fW", c/n, p/n}' $W/smi.tmp)
  o=$W/ab.$label.$mode.out
  g() { grep -h "$1" $o | head -1 | awk '{print $NF=="ms" ? $(NF-1) : $NF}'; }
  printf "%-5s %-6s wall %7s | xform %6s fill %6s apply %6s | summary %5s resieve %6s td %6s | %s load=%s\n" \
    $label $mode "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" \
    "$(g 'transform + plattice')" "$(g '^      fill ')" "$(g '^      apply ')" \
    "$(g 'survivor filter')" "$(g 'resieve + scatter')" "$(g 'norms + trial division')" "$st" "$(cut -d' ' -f1 /proc/loadavg)"
}
for round in 1 2; do
  for mode in single qp; do
    run base $mode $B
    run rs0 $mode BENCH_TEST_RS=0 $X
    run rs1 $mode $X
    run t512 $mode BENCH_TEST_RS_THREADS=512 $X
  done
done
rm -f $W/smi.tmp
