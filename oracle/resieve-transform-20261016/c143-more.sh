#!/bin/sh
# C143 pipelined on a free card, six more rounds alternating which arm runs
# first (c143-clean.txt's four rounds were +2.8, -1.3, +1.2, +1.1%).
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench; B=$W/bench.fef825c
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
cd $R/bench || exit 1
one() { $2 $C143 > $W/rc3.out 2>&1; printf "%s %s " $1 "$(grep -h 'wall clock per q, COMPLETE' $W/rc3.out | awk '{print $(NF-1)}')"; }
for r in 1 2 3 4 5 6; do
  if [ $((r % 2)) = 1 ]; then one base $B; one new $X; else one new $X; one base $B; fi
  echo "| sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits) load=$(cut -d' ' -f1 /proc/loadavg)"
done
