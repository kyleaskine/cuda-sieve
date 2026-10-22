#!/bin/sh
# The 128-slice cliff: 2,1147+ (alim = rlim = 600M) cuts into 138/137 slices,
# so apply's log table pads to 256 and only two blocks fit per SM. nocut/ cuts
# slices only where the log changes (fb_build_slices' 262,144-entry cut off),
# ~20 slices on any factor base. base/ is the same source as it stands. The
# NFS@Home I16 job (76/63 slices, three blocks either way) is the control.
# Not a gate on speed: every run must exit 0 and relations must match base r1.
R=/home/kylea/code/cuda-sieve; W=$R/work/tinytab
B=$W/base/bench/bench; N=$W/nocut/bench/bench
C1147="--pipeline --cofactor --poly $R/work/c1147/c1147.job --fb1 $R/work/c1147/c1147.roots1.m16 --logI 16 --J 32768 --maxbits 16 --qrange 300000000: --nq 10"
I16="--pipeline --cofactor --poly $R/work/applyprof/nfs.job --fb1 $R/work/applyprof/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 30"
cd $W/base/bench || exit 1
fails=0
run() {  # job mode arm round
  job=$1; mode=$2; arm=$3; rd=$4
  eval "A=\$$job"
  case $mode in sq) ex=--no-q-pipeline ;; *) ex= ;; esac
  case $arm in base) bin=$B ;; nocut) bin=$N ;; esac
  if ps -eo args | grep -v grep | grep -qE 'ggnfs-sieve-client|run-cuda-client'; then
    echo "   client running before $job $mode $arm r$rd"; fails=$((fails+1))
  fi
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  o=$W/cl.$job.$mode.$arm.$rd
  $bin $A $ex --relations $o.rel > $o.out 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  ref=$W/cl.$job.$mode.base.1.rel
  if [ -s $ref ] && ! cmp -s $o.rel $ref; then echo "   relations of $job $mode $arm r$rd differ from base r1"; fails=$((fails+1)); fi
  printf "%-5s %-2s %-5s r%s  wall %8s  apply %8s  fill %8s  rels %6s | load=%s card=%s%% rc=%s\n" $job $mode $arm $rd \
    "$(grep -h 'wall clock per q, COMPLETE' $o.out | awk '{print $(NF-1)}')" \
    "$(grep -h '^      apply ' $o.out | awk '{print $2}')" \
    "$(grep -h '^      fill ' $o.out | awk '{print $2}')" \
    "$(wc -l < $o.rel)" "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
echo "base $(sha256sum $B | cut -c1-16)  nocut $(sha256sum $N | cut -c1-16)"
for rd in 1 2 3; do
  case $rd in 2) arms="nocut base";; *) arms="base nocut";; esac
  for job in C1147 I16; do for mode in sq pl; do for arm in $arms; do run $job $mode $arm $rd; done; done; done
done
echo "done: $fails failed"
[ $fails -eq 0 ]
