#!/bin/sh
# Gates for --resieve-scan: the sm_120 snapshot (exp/) against the committed
# 9b9f314 build, with BASE's outputs cached by abcheck.sh. Every abcheck run
# keeps NEW's outputs and prints which path each case's side 0 actually took
# ("side 0 from its records" line), so a pass on the fallback arms shows the
# fallback ran.
R=${R:-/home/kylea/code/cuda-sieve}; W=${W:-$R/work/rscan}; X=${X:-$W/exp/bench/bench}
B=$R/work/abcheck/bench.9b9f314
cd $R/bench || exit 1
fail=0
paths() {  # what each kept NEW run did with side 0
  for d in $R/work/abcheck/run/*/; do
    l=$(grep -h "side 0 from its records\|--resieve-scan:" "$d/out" | tr -s ' ' | tr '\n' ';')
    printf "      %-18s %s\n" "$(basename "$d")" "${l:-no scan line}"
  done
}
ab() {  # label, then env assignments for abcheck
  label=$1; shift
  echo "== abcheck: $label"
  rm -rf "$R/work/abcheck/run"
  env KEEP=1 "$@" ./abcheck.sh "$B" "$X" > "$W/ab.tmp" 2>&1
  rc=$?
  sed 's/^/   /' "$W/ab.tmp"
  paths
  echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
}
ab "default (scan on wherever the gate is)"
# Caps that split a band's slabs between the two paths, found by trying caps
# on each job: at I16 the four slabs of one q split, at c183 and C143 whole q.
# c183.concurrent is left out of every knob arm: the gate is off there, so
# the scan is off and the binary refuses its knobs (exit 1), as it should.
ab "index cap 1536 at I16: slabs split"    SETS="cand cof" ONLY="[nc][f1][s9]*" NEWARGS="--resieve-scan-cap 1536"
ab "index cap 3072 at c183: q split"       SETS="cand cof" ONLY="c183*" EXCEPT="*.concurrent" NEWARGS="--resieve-scan-cap 3072"
ab "index cap 768 at C143: q split"        SETS="cand cof stop" ONLY="c143*" NEWARGS="--resieve-scan-cap 768"
ab "index cap 1: every slab falls back"    SETS="cand cof stop" QUICK=1 EXCEPT="*.concurrent" NEWARGS="--resieve-scan-cap 1"
ab "pair list from 1: grows, then scans"   SETS="cand cof stop" QUICK=1 EXCEPT="*.concurrent" NEWARGS="--resieve-scan-pairs 1"
ab "--no-resieve-scan"                     SETS="cand cof stop" QUICK=1 NEWARGS="--no-resieve-scan"
ab "does not fit: walked"                  SETS="cand" QUICK=1 ONLY="c183*" EXCEPT="*.concurrent" NEWARGS="--resieve-scan-pairs 4000000000"

echo "== refusals (each must exit nonzero with a message)"
J="--poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --qrange 2200000: --nq 2"
for a in "--pipeline --resieve-scan --no-fill-gate" "--pipeline --resieve-scan --fill-concurrent" \
         "--pipeline --no-resieve-scan --resieve-scan-cap 8" "--resieve-scan" "--resieve-scan-pairs 8"; do
  m=$($X $J $a 2>&1 | grep -i "resieve-scan" | head -1); rc=$?
  $X $J $a > /dev/null 2>&1; rc=$?
  printf "   %-50s exit %s: %s\n" "$a" "$rc" "$m"
  [ $rc -ne 0 ] || fail=$((fail+1))
done
echo "== make check (snapshot)"
(cd "$W/exp/bench" && nice make GPU_ARCH=120 check > "$W/check.log" 2>&1); rc=$?
grep -h "PASS, \|all gates passed\|FAIL" "$W/check.log" | tail -3 | sed 's/^/   /'
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "== fillcheck.sh (snapshot)"
(cd "$W/exp/bench" && ./fillcheck.sh > "$W/fillcheck.log" 2>&1); rc=$?
tail -3 "$W/fillcheck.log" | sed 's/^/   /'
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "done: $fail failed"
[ "$fail" -eq 0 ]
