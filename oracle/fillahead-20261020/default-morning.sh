#!/bin/sh
# --fill-ahead made the default wherever --q-pipeline runs (finding 123's
# follow-up, Kyle 2026-10-20). Gates for the fat build in fat/bench, which is
# the binary to install as bench/bench once they pass. Byte identity against
# the committed 2128138 build (BASE outputs cached).
#
# Only the option's default changed (bench_main resolves -1 to --q-pipeline's
# setting), so the code it now selects is the code finding 123's gates ran
# under --fill-ahead: degradecheck and skipcheck passed with it on their
# PIPE_K=8 and BN_LIMBS=4 builds, and are not rebuilt here. What is new is
# which runs get it: abcheck's every case now runs it wherever the plan is one
# slab, the old default path is kept as a --no-fill-ahead arm, and the
# resolution is checked case by case below.
#
# A SKIP line, a FAIL line, no PASS line or a nonzero exit fails a check;
# every check counts toward the exit status.
R=/home/kylea/code/cuda-sieve; W=$R/work/fadefault
F=$W/fat/bench/bench; B=$R/work/abcheck/bench.2128138
cd $R/bench || exit 1
fail=0
# Held until the card is free: no process of the gpuecm2 session, and the
# card under 10% busy for six samples in a row, ten seconds apart.
echo "hold from $(date '+%T'): waiting for an idle card"
quiet=0
while [ $quiet -lt 6 ]; do
  if ps -eo args | grep -v grep | grep -q gpuecm2; then
    quiet=0
  else
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "${u:-100}" -lt 10 ]; then quiet=$((quiet+1)); else quiet=0; fi
  fi
  [ $quiet -lt 6 ] && sleep 10
done
echo "hold released $(date '+%T')"
echo "base $(sha256sum $B | cut -c1-16)  new $(sha256sum $F | cut -c1-16)  load=$(cut -d' ' -f1 /proc/loadavg)"

echo "== build: the tree's sources (docs aside), six archs, no warnings"
nd=0
for f in $(git -C $R ls-files bench | grep -v '^bench/attic/' | grep -v '\.md$'); do
  cmp -s $R/$f $W/fat/$f || { echo "   differs from the tree: $f"; nd=$((nd+1)); }
done
archs=$(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $W/fat.build.log | sort -u | wc -l)
warns=$(grep -i warning $W/fat.build.log | grep -vc minnctapersm)
echo "   $(cat $W/fat.done) | sources differing $nd | archs $archs | warnings $warns"
grep -q 'exit=0' $W/fat.done && [ $nd -eq 0 ] && [ "$archs" -eq 6 ] && [ "$warns" -eq 0 ] \
  || fail=$((fail+1))

echo "== --verify-only and fbtest"
v=$($F --verify-only 2>&1 | grep -c '\[verify\] OK')
echo "   verify-only OK lines: $v (3 expected)"; [ "$v" -eq 3 ] || fail=$((fail+1))
(cd $W/fat/bench && ./fbtest --fb1 ../oracle/c183.fb1 > $W/fbtest.log 2>&1); rc=$?
echo "   fbtest exit=$rc $(tail -1 $W/fbtest.log)"; [ $rc -eq 0 ] || fail=$((fail+1))

ab() {
  label=$1; shift
  echo "== abcheck: $label"
  rm -rf "$R/work/abcheck/run"
  env "$@" ./abcheck.sh "$B" "$F" > "$W/ab.tmp" 2>&1
  rc=$?
  sed 's/^/   /' "$W/ab.tmp"
  echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
}
ab "default (fill ahead on wherever the plan is one slab), every case"
ab "--no-fill-ahead (finding 121's default path), every case" NEWARGS="--no-fill-ahead"

echo "== resolution: where the default turns it on, off or is refused"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 300"
C183S="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 130000000: --nq 30"
res() {  # name want(on|off-slabs|absent|refused) args...
  name=$1; want=$2; shift 2; o=$W/res.$name.out
  $F "$@" --relations $W/res.$name.rel > $o 2>&1; rc=$?
  on=$(grep -c -- "--fill-ahead: the q after next" $o)
  sl=$(grep -c -- "--fill-ahead: off, because this plan has several slabs" $o)
  any=$(grep -c -- "--fill-ahead" $o)
  ref=$(grep -c -- "--fill-ahead fills the next special-q" $o)
  fa=$(grep -h 'side-1 fill ahead (--fill-ahead)' $o | awk '{print $(NF-3)}')
  ta=$(grep -h 'prepared two ahead (--fill-ahead)' $o | awk '{print $(NF-1)}')
  case $want in
    on)        ok=$([ $rc -eq 0 ] && [ $on -eq 1 ] && [ "${fa:-0}" -gt 0 ] && [ "${ta:-0}" -gt 0 ] && echo y) ;;
    off-slabs) ok=$([ $rc -eq 0 ] && [ $sl -eq 1 ] && [ $on -eq 0 ] && echo y) ;;
    absent)    ok=$([ $rc -eq 0 ] && [ $any -eq 0 ] && echo y) ;;
    refused)   ok=$([ $rc -ne 0 ] && [ $ref -eq 1 ] && echo y) ;;
  esac
  if [ "$ok" = y ]; then r=PASS; else r=FAIL; fail=$((fail+1)); fi
  echo "   $r $name: want $want | exit $rc, on $on, slabs-off $sl, mentions $any, refused $ref, fill ahead ${fa:--} q, two ahead ${ta:--} q"
}
res c143-default on $C143
res c143-no-fill-ahead absent $C143 --no-fill-ahead
res c183-slabbed-default off-slabs $C183S
res c143-no-q-pipeline absent $C143 --no-q-pipeline
res c143-fill-concurrent absent $C143 --fill-concurrent
res c143-qspan absent $C143 --qspan
res c143-no-q-pipeline-fill-ahead refused $C143 --no-q-pipeline --fill-ahead
if [ -s $W/res.c143-default.rel ] && cmp -s $W/res.c143-default.rel $W/res.c143-no-fill-ahead.rel \
   && cmp -s $W/res.c143-default.rel $W/res.c143-no-q-pipeline.rel; then
  echo "   PASS c143 relations identical, default / --no-fill-ahead / --no-q-pipeline ($(wc -l < $W/res.c143-default.rel))"
else
  echo "   FAIL c143 relations differ between default, --no-fill-ahead and --no-q-pipeline"; fail=$((fail+1))
fi

echo "== make check (fat build)"
(cd $W/fat/bench && nice make check > $W/check.log 2>&1); rc=$?
echo "   $(grep -c '^PASS' $W/check.log) PASS, $(grep -c '^FAIL' $W/check.log) FAIL; $(tail -1 $W/check.log)"
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "== fillcheck.sh (fat build)"
(cd $W/fat/bench && ./fillcheck.sh > $W/fillcheck.log 2>&1); rc=$?
tail -2 $W/fillcheck.log | sed 's/^/   /'
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))

echo "done: $fail failed"
[ "$fail" -eq 0 ]
