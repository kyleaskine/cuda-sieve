#!/bin/sh
# Finding 123: the three sm_120 test builds of the two-ahead source --
# growfail (-DPIPE_RS_TEST_GROW_FAIL=1), pk8 (PIPE_K=8, for tdskip and
# degradecheck), bn4 (BN_LIMBS=4, bench and normscan, for skipcheck) -- from
# the tree's sources, in copies of finding 121's test directories. Every
# build counts.
R=/home/kylea/code/cuda-sieve; W=$R/work/fillahead; Q=$R/work/qprep
fail=0
for d in growfail pk8 bn4; do
  [ -d $W/$d ] || cp -a $Q/$d $W/$d || fail=$((fail+1))
  cp $R/bench/*.cu $R/bench/*.cuh $R/bench/*.h $R/bench/*.c $R/bench/*.cpp $R/bench/*.sh \
     $R/bench/Makefile $W/$d/bench/ || fail=$((fail+1))
done
b() {  # dir, targets, then make variables
  d=$1; t=$2; shift 2
  (cd $W/$d/bench && nice make GPU_ARCH=120 "$@" $t > $W/$d.build.log 2>&1); rc=$?
  [ $rc -eq 0 ] || fail=$((fail+1))
  echo "$d: exit $rc, $(grep -i warning $W/$d.build.log | grep -vc minnctapersm) warning line(s), sha256 $(sha256sum $W/$d/bench/bench | cut -c1-16)"
}
b growfail bench DEFS=-DPIPE_RS_TEST_GROW_FAIL=1
b pk8 bench PIPE_K=8
b bn4 "bench normscan" BN_LIMBS=4
for d in exp growfail pk8 bn4; do
  for f in pipeline.cuh bench.h bench_main.cu cofac.cuh; do
    cmp -s $R/bench/$f $W/$d/bench/$f || { echo "$d/$f differs from the tree"; fail=$((fail+1)); }
  done
done
echo "build failures: $fail"
[ $fail -eq 0 ]
