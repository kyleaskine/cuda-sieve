#!/bin/sh
# Per-launch k_apply duration and warp instructions, base vs experiment, on the
# NFS@Home I16 job, q index 3, single-queued (8 launches: 4 slabs x side 1, 0).
R=/home/kylea/code/cuda-sieve; W=$R/work/scan-i16; P=$R/work/applyprof
cd $R/bench || exit 1
NFS="--poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 296847000: --nq 5"
M=gpu__time_duration.sum,smsp__inst_executed.sum,sm__inst_executed_pipe_lsu.sum,smsp__issue_active.avg.pct_of_peak_sustained_active
for arm in base exp; do
  [ $arm = base ] && X=$W/bench.f116 || X=$W/exp/bench/bench
  ncu --metrics $M --kernel-name k_apply --launch-skip 24 --launch-count 8 --csv \
      $X --pipeline --cofactor $NFS --no-q-pipeline > $W/prof.$arm.csv 2> $W/prof.$arm.log
  echo "$arm ncu exit=$?"
done
