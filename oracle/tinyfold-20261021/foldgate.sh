#!/bin/sh
# Gates for --tiny-fold stage A (fold/), against base/ (the same source,
# 2e05b9d + the k_apply comment fix, without the fold), both sm_120 builds.
# Only output-identical behaviour is acceptable: the fold must reproduce the
# walk's cells exactly. A FAIL line, a missing PASS or a nonzero exit fails a
# check; every check counts toward the exit status.
R=/home/kylea/code/cuda-sieve; W=$R/work/tinytab
B=$W/base/bench/bench; F=$W/fold/bench/bench
fail=0
echo "base $(sha256sum $B | cut -c1-16)  fold $(sha256sum $F | cut -c1-16)  load=$(cut -d' ' -f1 /proc/loadavg)"
if ps -eo args | grep -v grep | grep -qE 'ggnfs-sieve-client|run-cuda-client'; then echo "   client running"; fail=$((fail+1)); fi

echo "== build: exit 0, no warnings, every k_apply instance <= 40 registers and no spill"
nk=$(grep -c "Compiling entry function '_Z7k_apply" $W/fold.build.log)
nsp=$(grep -A4 "Compiling entry function '_Z7k_apply" $W/fold.build.log | grep 'spill stores' | grep -vc ' 0 bytes spill stores, 0 bytes spill loads')
nreg=$(grep -A4 "Compiling entry function '_Z7k_apply" $W/fold.build.log | grep -o 'Used [0-9]* registers' | awk '$2 > 40' | wc -l)
warns=$(grep -i warning $W/fold.build.log | grep -vc minnctapersm)
echo "   $(cat $W/fold.done) | k_apply instances $nk, spilling $nsp, over 40 registers $nreg | warnings $warns"
grep -q 'exit=0' $W/fold.done && [ "$nk" -ge 11 ] && [ "$nsp" -eq 0 ] && [ "$nreg" -eq 0 ] && [ "$warns" -eq 0 ] || fail=$((fail+1))

cd $W/fold/bench || exit 1
echo "== --verify-only"
v=$($F --verify-only 2>&1 | grep -c '\[verify\] OK')
echo "   OK lines: $v (3 expected)"; [ "$v" -eq 3 ] || fail=$((fail+1))

echo "== apply cell parity (--verify: one region replayed on the CPU, every small-sieve entry)"
C183="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1"
vf() {  # name want-fold(y|n) args...
  name=$1; wantf=$2; shift 2; o=$W/vf.$name.out
  $F "$@" --verify > $o 2>&1; rc=$?
  ok0=$(grep -c ' 0 cells differ' $o); nd=$(grep -c 'cells differ' $o)
  nf=$(grep -h -- '--tiny-fold:' $o | awk '{print $2}' | sort -n | head -1)
  r=PASS
  [ $rc -eq 0 ] && [ "$nd" -ge 1 ] && [ "$ok0" -eq "$nd" ] || r=FAIL
  if [ $wantf = y ]; then [ "${nf:-0}" -gt 0 ] || r=FAIL; else [ -z "$nf" ] || r=FAIL; fi
  [ $r = FAIL ] && fail=$((fail+1))
  echo "   $r $name: exit $rc, regions checked $nd, with 0 cells differing $ok0, folded entries ${nf:-none}"
}
vf s1.I15        y $C183 --logI 15 --side 1
vf s0.I15        y $C183 --logI 15 --side 0
vf s1.I15.r10    y $C183 --logI 15 --side 1 --region 10
vf s1.I13.r13    y $C183 --logI 13 --J 4096 --side 1 --region 13
vf s0.I16        y $C183 --logI 16 --J 2048 --side 0
vf s1.I16.t96    y $C183 --logI 16 --J 2048 --side 1 --apply-threads 96
vf s0.I14.t160   y $C183 --logI 14 --J 8192 --side 0 --apply-threads 160
vf s1.I15.nofold n $C183 --logI 15 --side 1 --no-tiny-fold

cd $R/bench || exit 1
ab() {
  label=$1; shift
  echo "== abcheck: $label"
  rm -rf "$W/abcheck/run"
  env ABDIR=$W/abcheck "$@" ./abcheck.sh "$B" "$F" > "$W/ab.tmp" 2>&1
  rc=$?
  sed 's/^/   /' "$W/ab.tmp"
  echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
}
ab "fold on (default), every case"
ab "--no-tiny-fold on NEW's pipeline runs (harness cases still fold)" NEWARGS="--no-tiny-fold"

echo "== make check (fold build)"
(cd $W/fold/bench && nice make GPU_ARCH=120 check > $W/check.log 2>&1); rc=$?
echo "   $(grep -c '^PASS' $W/check.log) PASS, $(grep -c '^FAIL' $W/check.log) FAIL; $(tail -1 $W/check.log)"
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))

echo "done: $fail failed"
[ "$fail" -eq 0 ]
