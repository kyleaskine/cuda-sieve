#!/bin/sh
# Finding 120 gates, second part: growfail.sh and tdskip, rerun after their
# test builds finished (gates.txt ran them before the binaries existed: exit
# 127, counted as failures). Same checks as in gates.sh.
R=${R:-/home/kylea/code/cuda-sieve}; W=${W:-$R/work/tdwait}; X=${X:-$W/exp/bench/bench}
B=$R/work/abcheck/bench.2128138
P=$R/work/applyprof
cd $R/bench || exit 1
fail=0
echo "base $(sha256sum $B | cut -c1-16)  growfail $(sha256sum $W/growfail/bench/bench | cut -c1-16)  pk8 $(sha256sum $W/pk8/bench/bench | cut -c1-16)"
echo "== growfail.sh: the pair list fails to grow, the scan frees its index (drained first now)"
G=$W/growfail/bench/bench; tail -1 $W/growfail.build.log | sed 's/^/   /'
W=$W/gfrun B=$B G=$G sh $W/growfail.sh > $W/growfail.txt 2>&1; rc=$?
sed 's/^/   /' $W/growfail.txt
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))

echo "== tdskip (PIPE_K=8): truncated lists skip slabs, single-queued vs pipelined, c183 in five slabs"
K=$W/pk8/bench/bench; tail -1 $W/pk8.build.log | sed 's/^/   /'
A="--pipeline --cofactor --poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 120000000: --nq 150"
$K $A --no-q-pipeline --relations $W/tdskip.single.rel > $W/tdskip.single.out 2>&1; r1=$?
$K $A --relations $W/tdskip.qp.rel > $W/tdskip.qp.out 2>&1; r2=$?
sk=$(grep -h 'slab(s) skipped or abandoned' $W/tdskip.qp.out | tr -s ' ' | cut -c1-40)
# exit 5 (degraded) is a legitimate end for a band that skips this much;
# both arms must end the same way, 0 or 5
if [ $r1 -eq $r2 ] && { [ $r1 -eq 0 ] || [ $r1 -eq 5 ]; } && [ -s $W/tdskip.qp.rel ] && cmp -s $W/tdskip.single.rel $W/tdskip.qp.rel; then
  echo "   IDENTICAL ($(wc -l < $W/tdskip.qp.rel)), both exit $r1 |$sk | $(grep -h 'issued ahead' $W/tdskip.qp.out | tr -s ' ')"
else
  echo "   FAIL: exits $r1/$r2, $(wc -l < $W/tdskip.single.rel 2>/dev/null) vs $(wc -l < $W/tdskip.qp.rel 2>/dev/null) |$sk"
  fail=$((fail+1))
fi
[ -n "$sk" ] || { echo "   FAIL: no skip line -- vacuous"; fail=$((fail+1)); }
echo "done: $fail failed"
[ "$fail" -eq 0 ]
