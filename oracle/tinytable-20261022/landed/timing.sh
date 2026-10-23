#!/bin/sh
# Timing for findings 130-131 as landed: the fat build of the tree (land130)
# against the installed HEAD fat build (47f20cc2), so fat against fat. Arms:
# base = HEAD; notable = new --no-tiny-table (finding 130 alone: records carry
# log p); new = the default (with --tiny-table). Single-queued apply on five
# jobs, pipelined wall on four (I16 60 q, 2,1147+ 40 q). Rotated order. Not a
# gate on speed: every run must exit 0 and relations must match base r1 for
# that job and mode. ROUNDS overrides "1 2 3 4".
R=/home/kylea/code/cuda-sieve; W=$R/work/land130/timing; O=$R/oracle
B=$W/bench.47f20cc2; N=$R/work/land130/fat/bench/bench
C135="--pipeline --cofactor --poly $R/work/c135/c135.job --fb1 $R/work/c135/c135.roots1.m13 --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000: --nq 1500"
C183="--pipeline --cofactor --poly $O/input.job --fb1 $O/c183.fb1 --logI 15 --qrange 130000000: --nq 100"
I16="--pipeline --cofactor --poly $R/work/applyprof/nfs.job --fb1 $R/work/applyprof/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 60"
C194="--pipeline --cofactor --poly $O/c194.job --fb1 $O/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 30"
C1147="--pipeline --cofactor --poly $R/work/c1147/c1147.job --fb1 $R/work/c1147/c1147.roots1.m16 --logI 16 --J 32768 --maxbits 16 --qrange 300000000: --nq 40"
cd $R/work/land130/fat/bench || exit 1
fails=0
run() {  # job mode arm round
  job=$1; mode=$2; arm=$3; rd=$4
  eval "A=\$$job"
  case $mode in sq) ex=--no-q-pipeline ;; *) ex= ;; esac
  case $arm in base) bin=$B; ax= ;; notable) bin=$N; ax=--no-tiny-table ;; new) bin=$N; ax= ;; esac
  if ps -eo args | grep -v grep | grep -qE 'ggnfs-sieve-client|run-cuda-client'; then
    echo "   client running before $job $mode $arm r$rd"; fails=$((fails+1))
  fi
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  o=$W/tm.$job.$mode.$arm.$rd
  $bin $A $ex $ax --relations $o.rel > $o.out 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  ref=$W/tm.$job.$mode.base.1.rel
  if [ -s $ref ] && ! cmp -s $o.rel $ref; then echo "   relations of $job $mode $arm r$rd differ from base r1"; fails=$((fails+1)); fi
  printf "%-6s %-2s %-7s r%s  wall %8s  apply %8s  rels %7s | load=%s card=%s%% rc=%s\n" $job $mode $arm $rd \
    "$(grep -h 'wall clock per q, COMPLETE' $o.out | awk '{print $(NF-1)}')" \
    "$(grep -h '^      apply ' $o.out | awk '{print $2}')" \
    "$(wc -l < $o.rel)" "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
echo "base $(sha256sum $B | cut -c1-16)  new $(sha256sum $N | cut -c1-16)"
for rd in ${ROUNDS:-1 2 3 4}; do
  case $((rd % 3)) in 1) arms="base notable new";; 2) arms="new base notable";; *) arms="notable new base";; esac
  for job in C135 C183 I16 C194 C1147; do for arm in $arms; do run $job sq $arm $rd; done; done
  for job in C183 I16 C194 C1147; do for arm in $arms; do run $job pl $arm $rd; done; done
done
echo "done: $fails failed"
[ $fails -eq 0 ]
