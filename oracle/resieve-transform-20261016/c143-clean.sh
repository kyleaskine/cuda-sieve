#!/bin/sh
# C143 recheck ON A FREE CARD (client stopped). final-timing.txt showed +1.8% with the card time-sliced against
# the NFS@Home client; the clean A/B (ab2.txt) had -1.0%. Four more
# interleaved rounds, alternating which arm runs first, pipelined and single.
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench; B=$W/bench.fef825c
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
cd $R/bench || exit 1
one() { arm=$1; bin=$2; extra=$3
  $bin $C143 $extra > $W/rc2.out 2>&1
  printf "%-4s %-6s wall %6s xform %s | client=%s\n" $arm "${extra:-qp}" "$(grep -h 'wall clock per q, COMPLETE' $W/rc2.out | awk '{print $(NF-1)}')" \
    "$(grep -h 'transform + plattice' $W/rc2.out | head -1 | awk '{print $(NF-1)}')" "$(pgrep -fc 'ggnfs-client/w0' )"; }
for r in 1 2 3 4; do
  for extra in "" --no-q-pipeline; do
    if [ $((r % 2)) = 1 ]; then one base $B "$extra"; one new $X "$extra"; else one new $X "$extra"; one base $B "$extra"; fi
  done
done
