#!/bin/sh
# The rebuilt fat bench/bench (the client's binary): archs, warnings,
# --verify-only, fbtest, and the full abcheck.sh against the committed
# 9b9f314 build (BASE outputs cached), so the binary the client will run is
# the one gated, not only the sm_120 snapshot. Every check counts toward the
# exit status, not only the last.
R=${R:-/home/kylea/code/cuda-sieve}; W=${W:-$R/work/rscan}
cd $R/bench || exit 1
fails=0
archs=$(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $W/fat.build.log | sort -u | wc -l)
warns=$(grep -i warning $W/fat.build.log | grep -vc minnctapersm)
echo "fat: $(cat $W/fat.done) | archs $archs | warnings $warns | bench/bench $(sha256sum bench | cut -c1-16) built $(stat -c %y bench | cut -c1-19)"
[ "$archs" -eq 6 ] && [ "$warns" -eq 0 ] || fails=$((fails+1))
v=$(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')
echo "verify-only OK lines: $v (3 expected)"; [ "$v" -eq 3 ] || fails=$((fails+1))
./fbtest --fb1 ../oracle/c183.fb1 > $W/fat-fbtest.log 2>&1; rc=$?
echo "fbtest exit=$rc $(tail -1 $W/fat-fbtest.log)"; [ $rc -eq 0 ] || fails=$((fails+1))
./abcheck.sh $R/work/abcheck/bench.9b9f314 ./bench > $W/fat-abcheck.txt 2>&1; rc=$?
tail -1 $W/fat-abcheck.txt; echo "abcheck exit $rc"; [ $rc -eq 0 ] || fails=$((fails+1))
echo "fat-check: $fails failed"
[ $fails -eq 0 ]
