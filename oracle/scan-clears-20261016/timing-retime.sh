#!/bin/sh
# Finding 117's review fixes, retimed on a free card: the pre-review fat
# bench/bench (work/scan-i16/bench.f117) against the committed fat build
# (9b9f314, bench/bench 9b691952), pipelined, alternating order per round, SM
# clock and power logged; NFS also single-queued. Each row also carries the
# run's exit status and the card's utilization sampled just before it starts,
# so a foreign GPU job shows up in the log instead of only in the walls.
R=/home/kylea/code/cuda-sieve; W=$R/work/review-retime
B=${B:-$R/work/scan-i16/bench.f117}; X=${X:-$W/bench.new}
P=$R/work/applyprof; T=${T:-t}; ROUNDS=${ROUNDS:-3}
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 80"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 60"
cd $R/bench || exit 1
fails=0
run() {  # job arm round extra binary
  job=$1; arm=$2; rd=$3; extra=$4; bin=$5
  eval "A=\$$job"
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!
  o=$W/$T.$job.$arm.$rd.out
  $bin $A $extra > $o 2>&1; rc=$?
  kill $SMI
  [ $rc -ne 0 ] && fails=$((fails+1))
  g() { grep -h "^ *$1" $o | head -1 | awk '{print $NF=="ms" ? $(NF-1) : $NF}'; }
  printf "%-4s %-11s wall %8s | fill %7s apply %7s isect %6s | %s load=%s idle=%s%% rc=%s\n" $job $arm \
    "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" "$(g 'fill ')" "$(g 'apply ')" "$(g 'intersect')" \
    "$(awk -F', ' 'NR>4 {c+=$1; p+=$2; n++} END {printf "sm=%.0f p=%.0fW", c/n, p/n}' $W/smi.tmp)" "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
rd=1
while [ $rd -le $ROUNDS ]; do
  if [ $((rd % 2)) -eq 1 ]; then o1=base; b1=$B; o2=new; b2=$X; else o1=new; b1=$X; o2=base; b2=$B; fi
  for job in NFS C194 C183 C143; do
    run $job $o1 $rd "" $b1; run $job $o2 $rd "" $b2
  done
  run NFS $o1.single $rd --no-q-pipeline $b1; run NFS $o2.single $rd --no-q-pipeline $b2
  rd=$((rd+1))
done
rm -f $W/smi.tmp
echo "done: $fails failed run(s)"
[ $fails -eq 0 ]
