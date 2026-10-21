#!/bin/sh
# Finding 121: the two sm_120 test builds of the q-prep source -- growfail
# (-DPIPE_RS_TEST_GROW_FAIL=1, the pair list cannot grow) and pk8 (PIPE_K=8,
# truncated lists skip slabs) -- from the tree's sources. Every build counts.
R=/home/kylea/code/cuda-sieve; W=$R/work/qprep
fail=0
for d in growfail pk8; do
  [ -d $W/$d ] || cp -a $R/work/tdwait/$d $W/$d || fail=$((fail+1))
  cp $R/bench/*.cu $R/bench/*.cuh $R/bench/*.h $R/bench/*.c $R/bench/*.sh $R/bench/Makefile \
     $W/$d/bench/ || fail=$((fail+1))
done
b() {  # dir, then make variables
  d=$1; shift
  (cd $W/$d/bench && nice make GPU_ARCH=120 "$@" bench > $W/$d.build.log 2>&1); rc=$?
  [ $rc -eq 0 ] || fail=$((fail+1))
  echo "$d: exit $rc, $(grep -ci 'warning' $W/$d.build.log) warning line(s), sha256 $(sha256sum $W/$d/bench/bench | cut -c1-16)"
}
b growfail DEFS=-DPIPE_RS_TEST_GROW_FAIL=1
b pk8 PIPE_K=8
for d in exp growfail pk8; do
  for f in pipeline.cuh bench.h bench_main.cu cofac.cuh; do
    cmp -s $R/bench/$f $W/$d/bench/$f || { echo "$d/$f differs from the tree"; fail=$((fail+1)); }
  done
done
echo "build failures: $fail"
[ $fail -eq 0 ]
