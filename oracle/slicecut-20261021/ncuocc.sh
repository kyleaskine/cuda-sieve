#!/bin/sh
# MORNING: confirm the block limit directly. One k_apply launch of 2,1147+
# under ncu (launch stats + occupancy) for the installed HEAD build and the new
# build: finding 128 predicts Block Limit Shared Mem 2 -> 3 and dynamic shared
# memory 33,280 -> 32,832 B. Fails unless both readings match.
R=/home/kylea/code/cuda-sieve; W=$R/work/land127
A="--pipeline --cofactor --poly $R/work/c1147/c1147.job --fb1 $R/work/c1147/c1147.roots1.m16 --logI 16 --J 32768 --maxbits 16 --qrange 300000000: --nq 1 --no-q-pipeline"
fail=0
cd $W/fat/bench || exit 1
for arm in base new; do
  case $arm in base) bin=$W/bench.f37fe64d; want=2; smem=33.28 ;; new) bin=$W/fat/bench/bench; want=3; smem=32.83 ;; esac
  o=$W/ncu.$arm.txt
  ncu --kernel-name regex:k_apply --launch-count 1 --section LaunchStats --section Occupancy $bin $A > $o 2>&1
  bl=$(grep 'Block Limit Shared Mem' $o | awk '{print $NF}' | head -1)
  ds=$(grep 'Dynamic Shared Memory Per Block' $o | awk '{print $NF}' | head -1)
  th=$(grep 'Theoretical Occupancy' $o | awk '{print $NF}' | head -1)
  r=PASS; [ "$bl" = "$want" ] || r=FAIL; [ $r = FAIL ] && fail=$((fail+1))
  echo "   $r $arm: Block Limit Shared Mem ${bl:-?} (want $want), dynamic smem/block ${ds:-?} KB (expect ~$smem), theoretical occupancy ${th:-?}%"
done
echo "done: $fail failed"
[ $fail -eq 0 ]
