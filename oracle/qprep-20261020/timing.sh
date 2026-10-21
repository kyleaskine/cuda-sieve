#!/bin/sh
# Finding 121 timing: the finding-120 snapshot (BASE: the tree before this
# change, sm_120) against the q-prep snapshot under each --q-prep policy,
# pipelined, order rotated per round, card clocks and power logged. Each row
# carries the run's exit status, the card's utilization sampled just before
# it, and how many q were prepared during trial division.
R=/home/kylea/code/cuda-sieve; W=$R/work/qprep
B=${B:-$R/work/tdwait/exp/bench/bench}; X=${X:-$W/exp/bench/bench}
P=$R/work/applyprof; T=${T:-t}
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 80"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 60"
cd $R/bench || exit 1
fails=0
run() {  # job arm round
  job=$1; arm=$2; rd=$3
  eval "A=\$$job"
  case $arm in base) bin=$B; extra= ;; *) bin=$X; extra="--q-prep $arm" ;; esac
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!
  o=$W/$T.$job.$arm.$rd.out
  $bin $A $extra > $o 2>&1; rc=$?
  kill $SMI
  [ $rc -ne 0 ] && fails=$((fails+1))
  printf "%-4s %-4s r%s wall %8s | td-prep %5s | %s load=%s idle=%s%% rc=%s\n" $job $arm $rd \
    "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" \
    "$(grep -h 'prepared during trial division' $o | awk '{print $5}')" \
    "$(awk -F', ' 'NR>4 {c+=$1; p+=$2; n++} END {printf "sm=%.0f p=%.0fW", c/n, p/n}' $W/smi.tmp)" \
    "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
rot() {  # rotate a list by $1 places
  n=$1; shift
  i=0; while [ $i -lt $n ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done
  echo "$@"
}
for rd in 1 2 3 4 5; do
  for arm in $(rot $((rd - 1)) base join auto td); do run C143 $arm $rd; done
done
for rd in 1 2 3; do
  for job in NFS C194 C183; do
    for arm in $(rot $((rd - 1)) base auto td); do run $job $arm $rd; done
  done
done
rm -f $W/smi.tmp
echo "done: $fails failed run(s)"
[ $fails -eq 0 ]
