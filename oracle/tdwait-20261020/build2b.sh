#!/bin/sh
# Rerun of build2.sh after a whitespace-only realignment of cofac.cuh (CF_ENQ's
# backslashes), made while the fat build ran, to compare hashes with build2.txt.
# That premise was wrong: nvcc builds here differ by a few bytes each time, even
# from the same source (repro.txt), so the differing hashes say nothing.
# Review fixes (finding 120): rebuild the three sm_120 test snapshots from the
# tree's pipeline.cuh, cofac.cuh and abcheck.sh. Every build counts.
R=/home/kylea/code/cuda-sieve; W=$R/work/tdwait
fail=0
for d in exp growfail pk8; do
  cp $R/bench/pipeline.cuh $R/bench/cofac.cuh $R/bench/abcheck.sh $W/$d/bench/ || fail=$((fail+1))
done
b() {  # dir, then make variables
  d=$1; shift
  (cd $W/$d/bench && nice make GPU_ARCH=120 "$@" bench > $W/$d.build2b.log 2>&1); rc=$?
  [ $rc -eq 0 ] || fail=$((fail+1))
  echo "$d: exit $rc, $(grep -ci 'warning' $W/$d.build2b.log) warning line(s), sha256 $(sha256sum $W/$d/bench/bench | cut -c1-16)"
}
b exp
b growfail DEFS=-DPIPE_RS_TEST_GROW_FAIL=1
b pk8 PIPE_K=8
for d in exp growfail pk8; do
  for f in pipeline.cuh cofac.cuh; do cmp -s $R/bench/$f $W/$d/bench/$f || { echo "$d/$f differs from the tree"; fail=$((fail+1)); }; done
done
echo "build failures: $fail"
[ $fail -eq 0 ]
