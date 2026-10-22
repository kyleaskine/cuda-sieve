#!/bin/sh
# Small-sieve hit census by effective modulus (hitcount.c), host only. Job
# arguments and scales as the bench runs them (scales from each job's own
# "params from q=" line in work/h2h-20261021 and work/fadefault outputs).
R=/home/kylea/code/cuda-sieve; O=$R/oracle; W=$R/work/tinytab
cd $W || exit 1
fail=0
run() {  # name args...
  name=$1; shift
  echo "== $name"
  nice ./hitcount "$@" > $W/census.$name.txt 2>&1; rc=$?
  grep -v 'factor base' $W/census.$name.txt
  [ $rc -eq 0 ] && grep -q 'special-q from' $W/census.$name.txt || { echo "   FAIL $name rc=$rc"; fail=$((fail+1)); }
}
run c135  --poly $R/work/c135/c135.job --fb1 $R/work/c135/c135.roots1.m13 --logI 13 --J 4096 --region 13 --maxbits 13 --qmin 7000000 --nq 30 --scale1 1.600 --scale0 2.450
run c143  --poly $O/c143.job --fb1 $W/c143.roots1.m13 --logI 13 --J 8192 --region 13 --maxbits 13 --qmin 2200000 --nq 30 --scale1 1.550 --scale0 2.375
run c183  --poly $O/input.job --fb1 $O/c183.fb1 --logI 15 --qmin 130000000 --nq 12 --scale1 1.275 --scale0 1.925
run i16   --poly $R/work/applyprof/nfs.job --fb1 $R/work/applyprof/nfs.roots1.m16 --logI 16 --J 32768 --qmin 297000000 --nq 8 --scale1 1.175 --scale0 1.750
run c194  --poly $O/c194.job --fb1 $O/c194.roots1.m16 --logI 16 --J 32768 --qmin 80000023 --nq 8 --scale1 1.225 --scale0 1.800
run as276 --poly $O/AS276.job --fb1 $R/work/lpbsweep/as276.roots1.m16 --logI 16 --J 32768 --maxbits 16 --qmin 80000023 --nq 8 --scale1 1.175 --scale0 1.700
echo "done: $fail failed"
[ $fail -eq 0 ]
