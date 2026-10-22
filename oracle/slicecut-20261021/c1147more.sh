#!/bin/sh
# timing.sh's 2,1147+ pipelined wall at 10 q was inside its own noise (the final
# flush is ~50 ms/q of a 10 q band; in-loop 177-210) although apply fell ~18% in
# every round. Four rotated rounds of 40 q. Same checks as timing.sh:
# every run exits 0 and relations match each other within the arm set.
R=/home/kylea/code/cuda-sieve; W=$R/work/land127
B=$W/bench.f37fe64d; F=$W/fat/bench/bench
A="--pipeline --cofactor --poly $R/work/c1147/c1147.job --fb1 $R/work/c1147/c1147.roots1.m16 --logI 16 --J 32768 --maxbits 16 --qrange 300000000: --nq 40"
cd $W/fat/bench || exit 1
fails=0
for rd in 4 5 6 7; do
  case $((rd % 3)) in 1) arms="base new slices";; 2) arms="slices base new";; *) arms="new slices base";; esac
  for arm in $arms; do
    case $arm in base) bin=$B; ax= ;; new) bin=$F; ax= ;; slices) bin=$F; ax=--no-tiny-fold ;; esac
    if ps -eo args | grep -v grep | grep -qE 'ggnfs-sieve-client|run-cuda-client'; then echo "   client running"; fails=$((fails+1)); fi
    sleep 2; o=$W/m40.C1147.$arm.$rd
    $bin $A $ax --relations $o.rel > $o.out 2>&1; rc=$?; [ $rc -ne 0 ] && fails=$((fails+1))
    ref=$W/m40.C1147.base.4.rel
    if [ -s $ref ] && ! cmp -s $o.rel $ref; then echo "   relations of $arm r$rd differ"; fails=$((fails+1)); fi
    printf "C1147   pl %-6s r%s  wall %8s  apply %8s  rels %7s | load=%s rc=%s\n" $arm $rd \
      "$(grep -h 'wall clock per q, COMPLETE' $o.out | awk '{print $(NF-1)}')" \
      "$(grep -h '^      apply ' $o.out | awk '{print $2}')" "$(wc -l < $o.rel)" "$(cut -d' ' -f1 /proc/loadavg)" $rc
  done
done
echo "done: $fails failed"
[ $fails -eq 0 ]
