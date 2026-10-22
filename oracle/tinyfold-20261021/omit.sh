#!/bin/sh
# Upper bounds for folding tiny moduli (STATUS 8b): the same source built twice
# for sm_120 (base/, omit/), omit/ dropping line-sieved entries by effective
# modulus under TINY_OMIT (pow2: m in {1,2,4,8,16,32}; t420: m in {1,2,3,4,5,7}).
# Removing them outright prices the ceiling, not a replacement: wrong output,
# so only exit status and the arm's TINY_OMIT line are checked. Single-queued
# (--no-q-pipeline) runs give a clean apply timer; pipelined runs give wall.
R=/home/kylea/code/cuda-sieve; W=$R/work/tinytab
C135="--pipeline --cofactor --poly $R/work/c135/c135.job --fb1 $R/work/c135/c135.roots1.m13 --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000: --nq 1500"
C183="--pipeline --cofactor --poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 100"
I16="--pipeline --cofactor --poly $R/work/applyprof/nfs.job --fb1 $R/work/applyprof/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 30"
cd $W/base/bench || exit 1
fails=0
run() {  # job mode arm round
  job=$1; mode=$2; arm=$3; rd=$4
  eval "A=\$$job"
  case $mode in sq) ex=--no-q-pipeline ;; *) ex= ;; esac
  case $arm in base) bin=$W/base/bench/bench; env= ;; *) bin=$W/omit/bench/bench; env="TINY_OMIT=$arm" ;; esac
  if ps -eo args | grep -v grep | grep -qE 'ggnfs-sieve-client|run-cuda-client'; then
    echo "   client running before $job $mode $arm r$rd"; fails=$((fails+1))
  fi
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  o=$W/om.$job.$mode.$arm.$rd.out
  env $env $bin $A $ex > $o 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  want=off; [ $arm != base ] && want=$arm
  if [ $arm != base ] && ! grep -q "TINY_OMIT: $want" $o; then echo "   $job $mode $arm r$rd: no TINY_OMIT $want line"; fails=$((fails+1)); fi
  printf "%-4s %-2s %-4s r%s  wall %8s  apply %8s  surv %10s | load=%s card=%s%% rc=%s\n" $job $mode $arm $rd \
    "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" \
    "$(grep -h '^      apply ' $o | awk '{print $2}')" \
    "$(grep -h 'two-sided primitive survivors/q' $o | awk '{print $NF}')" \
    "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
echo "base $(sha256sum $W/base/bench/bench | cut -c1-16)  omit $(sha256sum $W/omit/bench/bench | cut -c1-16)"
for rd in ${ROUNDS:-1 2 3}; do
  case $((rd % 3)) in 1) arms="base pow2 t420";; 2) arms="t420 base pow2";; *) arms="pow2 t420 base";; esac
  for job in C135 C183 I16; do for mode in sq pl; do for arm in $arms; do run $job $mode $arm $rd; done; done; done
done
echo "done: $fails failed"
[ $fails -eq 0 ]
