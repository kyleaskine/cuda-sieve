#!/bin/sh
# Gates for finding 120's two host-wait fixes (TD's record wait paid after
# k_cof_enqueue is queued; the --q-pipeline issue point moved ahead of the
# intersection's wait). NEW is the sm_120 snapshot (exp/), BASE the committed
# 2128138 fat bench/bench (77f1abab). Neither change may alter what the siever
# finds, so every check is byte identity against BASE or against the
# single-queued arm of the same binary. Every check counts toward the exit
# status.
R=${R:-/home/kylea/code/cuda-sieve}; W=${W:-$R/work/tdwait}; X=${X:-$W/exp/bench/bench}
B=$R/work/abcheck/bench.2128138
P=$R/work/applyprof
cd $R/bench || exit 1
fail=0
echo "base $(sha256sum $B | cut -c1-16)  new $(sha256sum $X | cut -c1-16)  load=$(cut -d' ' -f1 /proc/loadavg)"
ab() {  # label, then env assignments for abcheck
  label=$1; shift
  echo "== abcheck: $label"
  rm -rf "$R/work/abcheck/run"
  env "$@" ./abcheck.sh "$B" "$X" > "$W/ab.tmp" 2>&1
  rc=$?
  sed 's/^/   /' "$W/ab.tmp"
  echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
}
ab "default, every case"
# The resieve-scan arms whose branches sit after the moved issue point: the
# pair list grows (cudaFree with the next sieve in flight), every slab walks.
ab "pair list from 1: grows, then scans"   SETS="cand cof stop" QUICK=1 EXCEPT="*.concurrent" NEWARGS="--resieve-scan-pairs 1"
ab "index cap 1: every slab falls back"    SETS="cand cof stop" QUICK=1 EXCEPT="*.concurrent" NEWARGS="--resieve-scan-cap 1"
ab "--no-resieve-scan"                     SETS="cand cof stop" QUICK=1 NEWARGS="--no-resieve-scan"

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
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ -s $W/tdskip.qp.rel ] && cmp -s $W/tdskip.single.rel $W/tdskip.qp.rel; then
  echo "   IDENTICAL ($(wc -l < $W/tdskip.qp.rel)) |$sk | $(grep -h 'issued ahead' $W/tdskip.qp.out | tr -s ' ')"
else
  echo "   FAIL: exits $r1/$r2, $(wc -l < $W/tdskip.single.rel 2>/dev/null) vs $(wc -l < $W/tdskip.qp.rel 2>/dev/null) |$sk"
  fail=$((fail+1))
fi
[ -n "$sk" ] || { echo "   FAIL: no skip line -- vacuous"; fail=$((fail+1)); }

echo "== make check (snapshot)"
(cd "$W/exp/bench" && nice make GPU_ARCH=120 check > "$W/check.log" 2>&1); rc=$?
echo "   $(grep -c '^PASS' $W/check.log) PASS, $(grep -c '^FAIL' $W/check.log) FAIL; $(tail -1 $W/check.log)"
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "== fillcheck.sh (snapshot)"
(cd "$W/exp/bench" && ./fillcheck.sh > "$W/fillcheck.log" 2>&1); rc=$?
tail -2 "$W/fillcheck.log" | sed 's/^/   /'
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "done: $fail failed"
[ "$fail" -eq 0 ]
