#!/bin/sh
# Finding 117's review fixes, sm_120 snapshot of the final source: fbtest's
# reworked transform gate, --verify-only, --verify (the CPU replay now runs
# pl_transform_fast like the device), make check, fillcheck, survivor bitmaps
# against finding 116's binary (the harness's host small-sieve transform
# changed), relation identity on four jobs.
R=/home/kylea/code/cuda-sieve; W=$R/work/scan-i16; X=$W/exp/bench/bench
cd $W/exp/bench || exit 1
echo "build: $(cat $W/build3.status) | warnings $(grep -i warning $W/build3.log | grep -vc minnctapersm)"
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
( cd $R/bench && $X --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --verify > $W/verify.c183.out 2>&1; echo "--verify c183 exit=$?: $(grep -h '\[verify\]' $W/verify.c183.out | tr -s ' ' | tr '\n' '|')" )
nice -n 5 make check GPU_ARCH=120 > $W/check3.log 2>&1; echo "make check exit=$? ($(grep -c '^PASS' $W/check3.log) PASS, $(grep -c '^FAIL' $W/check3.log) FAIL) $(tail -1 $W/check3.log)"
grep -E "fast path|compared" $W/check3.log | tr -s ' ' | sed 's/^/  /'
./fillcheck.sh > $W/fillcheck3.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $W/fillcheck3.log) ok) $(tail -1 $W/fillcheck3.log)"
echo "== survivor bitmaps against finding 116's binary"
sh $W/sbgate.sh > $W/sbgate3.txt 2>&1; tail -1 $W/sbgate3.txt
echo "== relation identity"
sh $W/ident.sh 2>&1 | sed 's/^/  /'
