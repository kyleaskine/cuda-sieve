#!/bin/sh
# As qpcheck.sh, with --relations (file + per-q checkpoint), same 80 q as the trace,
# interleaved, with SM clock and board power sampled every 0.5 s.
R=/home/kylea/code/cuda-sieve; W=$R/work/nsys-i16-20261016; P=$R/work/applyprof
cd $R/bench || exit 1
for arm in qp single qp single; do
  extra=; [ $arm = single ] && extra=--no-q-pipeline
  rm -f "${W:?}"/qprel-$arm.rel "${W:?}"/qprel-$arm.rel.part "${W:?}"/qprel-$arm.rel.part.ckpt "${W:?}"/qprel-$arm.rel.lock
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 500 > $W/smi-$arm.tmp &
  SMI=$!
  ./bench --pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 \
      --logI 16 --J 32768 --qrange 297000000: --nq 80 $extra --relations $W/qprel-$arm.rel > $W/qprel-$arm.out 2>&1
  kill $SMI
  st=$(awk -F', ' 'NR>10 {c+=$1; p+=$2; n++} END {printf "sm=%.0f MHz power=%.0f W", c/n, p/n}' $W/smi-$arm.tmp)
  echo "$arm: $(grep -h 'wall clock per q, COMPLETE' $W/qprel-$arm.out | tr -s ' ') | $st | load=$(cut -d' ' -f1 /proc/loadavg)"
done
rm -f $W/smi-*.tmp
