#!/bin/sh
# The branch a real job never reaches: the --resieve-scan pair list fails to
# grow, so the scan turns off for the rest of the band and frees its index.
# A test build (-DPIPE_RS_TEST_GROW_FAIL=1, sm_120) fails every growth;
# --resieve-scan-pairs 1 makes the first slab need one. A DEFS build refuses
# --relations (bench_main.cu, a pricing build's relations would pass for
# production), so both binaries write the candidate stream alone, which is
# what this branch could change. Each case must be byte-identical to the
# committed build, print the warning, and report the scan off.
R=${R:-/home/kylea/code/cuda-sieve}; W=${W:-$R/work/rscan}
G=${G:-$W/growfail/bench/bench}; B=$R/work/abcheck/bench.9b9f314; D=$W/gf
P=$R/work/applyprof
cd $R/bench || exit 1
mkdir -p "${D:?}"
fail=0
case_() {  # name args...
  name=$1; shift
  rm -f "${D:?}/$name".*
  $B --pipeline "$@" --candidates $D/$name.base.cand > $D/$name.base.out 2>&1; rb=$?
  $G --pipeline "$@" --resieve-scan-pairs 1 --candidates $D/$name.test.cand > $D/$name.test.out 2>&1; rt=$?
  why=""
  [ $rb -eq 0 ] || why="base exited $rb"
  [ $rt -eq 0 ] || why="${why:+$why, }test exited $rt"
  [ -s $D/$name.base.cand ] || why="${why:+$why, }base has no candidates"
  cmp -s $D/$name.base.cand $D/$name.test.cand || why="${why:+$why, }candidates differ"
  grep -q "growing the pair list .* failed" $D/$name.test.out || why="${why:+$why, }no growth-failure warning"
  grep -q "side 0 from its records *0 slab(s); walked: 0 index full, 1 pair list grown, then off" $D/$name.test.out || why="${why:+$why, }summary not 'grown, then off'"
  if [ -n "$why" ]; then fail=$((fail+1)); r="FAIL: $why"; else r="IDENTICAL"; fi
  printf "  %-10s %s, %s candidates\n" "$name" "$r" "$(wc -l < $D/$name.test.cand 2>/dev/null)"
  printf "             %s\n" "$(grep -h 'side 0 from its records' $D/$name.test.out | tr -s ' ')"
}
case_ nfs   --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 296847000: --nq 10
case_ c194  --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 6
case_ c183  --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 30
case_ c183s --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 130000000: --nq 10
case_ c143  --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 300
grep -h "growing the pair list" $D/nfs.test.out | head -1 | sed 's/^ */  warning: /'
echo "done: $fail failed"
[ "$fail" -eq 0 ]
