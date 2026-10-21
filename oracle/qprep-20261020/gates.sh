#!/bin/sh
# Finding 121 gates: --q-prep (the next special-q prepared and issued during
# this q's trial division when its sieve has already finished) and the next
# q's TD small-prime tables built when it is prepared. Neither may change what
# the siever finds, so every check is byte identity: against the committed
# 2128138 fat build (BASE, outputs cached) under each policy, or between the
# single-queued and pipelined arms of one binary. skipcheck and degradecheck
# run on the snapshot as built (auto) and through a wrapper that adds
# --q-prep td, so the norm-width skip and the degradation ceiling are also
# met with the preparation in TD. Every check counts toward the exit status.
R=/home/kylea/code/cuda-sieve; W=$R/work/qprep; X=$W/exp/bench/bench
B=$R/work/abcheck/bench.2128138
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
# --q-prep is refused without the pipeline: leave out the .single and
# .concurrent cases, which turn it off.
NOQP='*.[sc][io]n[gc]*'
ab "default (--q-prep auto), every case"
ab "--q-prep td"   SETS="cand cof stop" EXCEPT="$NOQP" NEWARGS="--q-prep td"
ab "--q-prep join" SETS="cand cof stop" EXCEPT="$NOQP" NEWARGS="--q-prep join"
ab "--q-prep td, pair list from 1 (grows)" SETS="cand cof stop" QUICK=1 EXCEPT="$NOQP" \
   NEWARGS="--q-prep td --resieve-scan-pairs 1"
ab "--q-prep td, index cap 1 (every slab walks)" SETS="cand cof stop" QUICK=1 EXCEPT="$NOQP" \
   NEWARGS="--q-prep td --resieve-scan-cap 1"

echo "== growfail.sh: the pair list fails to grow (auto)"
G=$W/growfail/bench/bench; tail -1 $W/growfail.build.log | sed 's/^/   /'
W=$W/gfrun B=$B G=$G sh $R/work/tdwait/growfail.sh > $W/growfail.txt 2>&1; rc=$?
sed 's/^/   /' $W/growfail.txt
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))

echo "== tdskip (PIPE_K=8): truncated lists skip slabs; single-queued against auto and td, c183 in five slabs"
K=$W/pk8/bench/bench; tail -1 $W/pk8.build.log | sed 's/^/   /'
A="--pipeline --cofactor --poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 120000000: --nq 150"
$K $A --no-q-pipeline --relations $W/tdskip.single.rel > $W/tdskip.single.out 2>&1; r0=$?
for p in auto td; do
  $K $A --q-prep $p --relations $W/tdskip.$p.rel > $W/tdskip.$p.out 2>&1; r=$?
  sk=$(grep -h 'slab(s) skipped or abandoned' $W/tdskip.$p.out | tr -s ' ' | cut -c1-40)
  td=$(grep -h 'prepared during trial division' $W/tdskip.$p.out | tr -s ' ')
  if [ $r0 -eq 0 ] && [ $r -eq 0 ] && [ -s $W/tdskip.$p.rel ] && cmp -s $W/tdskip.single.rel $W/tdskip.$p.rel; then
    echo "   $p: IDENTICAL ($(wc -l < $W/tdskip.$p.rel)) |$sk |$td"
  else
    echo "   $p: FAIL: exits $r0/$r, $(wc -l < $W/tdskip.single.rel 2>/dev/null) vs $(wc -l < $W/tdskip.$p.rel 2>/dev/null) |$sk"
    fail=$((fail+1))
  fi
  [ -n "$sk" ] || { echo "   $p: FAIL: no skip line -- vacuous"; fail=$((fail+1)); }
done

echo "== make check (snapshot)"
(cd "$W/exp/bench" && nice make GPU_ARCH=120 check > "$W/check.log" 2>&1); rc=$?
echo "   $(grep -c '^PASS' $W/check.log) PASS, $(grep -c '^FAIL' $W/check.log) FAIL; $(tail -1 $W/check.log)"
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "== fillcheck.sh (snapshot)"
(cd "$W/exp/bench" && ./fillcheck.sh > "$W/fillcheck.log" 2>&1); rc=$?
tail -2 "$W/fillcheck.log" | sed 's/^/   /'
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))

# skipcheck/degradecheck run ./bench from their own directory. The td arm is
# a directory whose ./bench appends --q-prep td (every call they make is
# --pipeline), with the snapshot's normscan and oracle.
T=$W/td
mkdir -p $T/bench
ln -sfn $W/exp/oracle $T/oracle
ln -sf $W/exp/bench/normscan $T/bench/normscan
cp $W/exp/bench/skipcheck.sh $W/exp/bench/degradecheck.sh $T/bench/
printf '#!/bin/sh\nexec %s "$@" --q-prep td\n' "$X" > $T/bench/bench; chmod +x $T/bench/bench
for d in exp td; do
  for g in skipcheck degradecheck; do
    echo "== $g.sh ($d)"
    (cd $W/$d/bench && sh ./$g.sh > $W/$g.$d.log 2>&1); rc=$?
    tail -2 $W/$g.$d.log | sed 's/^/   /'
    echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
  done
done
echo "done: $fail failed"
[ "$fail" -eq 0 ]
