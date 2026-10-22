#!/bin/sh
# timing.sh's I16 pipelined wall was inside its own noise (three rounds of
# 30 q: base 146-151, new 145-152 ms/q in-loop) although apply fell ~6% in
# every round. Six more rotated rounds of 60 q. Same checks as timing.sh:
# every run exits 0 and relations match each other within the arm set.
R=/home/kylea/code/cuda-sieve; W=$R/work/land127
B=$W/bench.f37fe64d; F=$W/fat/bench/bench
A="--pipeline --cofactor --poly $R/work/applyprof/nfs.job --fb1 $R/work/applyprof/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 60"
cd $W/fat/bench || exit 1
fails=0
for rd in 4 5 6 7 8 9; do
  case $((rd % 3)) in 1) arms="base new slices";; 2) arms="slices base new";; *) arms="new slices base";; esac
  for arm in $arms; do
    case $arm in base) bin=$B; ax= ;; new) bin=$F; ax= ;; slices) bin=$F; ax=--no-tiny-fold ;; esac
    if ps -eo args | grep -v grep | grep -qE 'ggnfs-sieve-client|run-cuda-client'; then echo "   client running"; fails=$((fails+1)); fi
    sleep 2; o=$W/m60.I16.$arm.$rd
    $bin $A $ax --relations $o.rel > $o.out 2>&1; rc=$?; [ $rc -ne 0 ] && fails=$((fails+1))
    ref=$W/m60.I16.base.4.rel
    if [ -s $ref ] && ! cmp -s $o.rel $ref; then echo "   relations of $arm r$rd differ"; fails=$((fails+1)); fi
    printf "I16     pl %-6s r%s  wall %8s  apply %8s  rels %7s | load=%s rc=%s\n" $arm $rd \
      "$(grep -h 'wall clock per q, COMPLETE' $o.out | awk '{print $(NF-1)}')" \
      "$(grep -h '^      apply ' $o.out | awk '{print $2}')" "$(wc -l < $o.rel)" "$(cut -d' ' -f1 /proc/loadavg)" $rc
  done
done
echo "done: $fails failed"
[ $fails -eq 0 ]
