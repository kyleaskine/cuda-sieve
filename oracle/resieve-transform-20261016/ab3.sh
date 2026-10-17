#!/bin/sh
# Where the coarse filter starts to pay: C143 at intermediate slab areas.
# I14 J8192 = 2^27 positions, I14 J16384 = 2^28. rs0 (old resieve) vs t512,
# single-queued (stage times) and pipelined, two rounds.
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench
cd $R/bench || exit 1
for round in 1 2; do
 for geo in "27|--logI 14 --J 8192" "28|--logI 14 --J 16384"; do
  a=${geo%%|*}; g=${geo#*|}
  for mode in single qp; do
   extra=; [ $mode = single ] && extra=--no-q-pipeline
   for arm in rs0 t512; do
    if [ $arm = rs0 ]; then e="BENCH_TEST_RS=0"; else e="BENCH_TEST_RS_THREADS=512"; fi
    o=$W/ab3.$a.$mode.$arm.out
    env $e $X --pipeline --cofactor --poly ../oracle/c143.job $g --region 13 --maxbits 13 --sq-side 1 \
      --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 600 $extra > $o 2>&1
    printf "2^%s %-6s %-4s wall %7s resieve %6s rel %s\n" $a $mode $arm \
      "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" \
      "$(grep -h 'resieve + scatter' $o | head -1 | awk '{print $(NF-1)}')" \
      "$(grep -h '^  total relations' $o | tail -1 | awk '{print $NF}')"
   done
  done
 done
done
