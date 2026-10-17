#!/bin/sh
# Packed threshold scan + skipped clears (sm_120 snapshot) against finding
# 116's fat bench/bench, free card, pipelined (the default), alternating order
# per round, SM clock and power logged; NFS also single-queued for stage times.
R=/home/kylea/code/cuda-sieve; W=$R/work/scan-i16; X=${X:-$W/exp/bench/bench}; B=${B:-$W/bench.f116}
P=$R/work/applyprof; T=${T:-t}; ROUNDS=${ROUNDS:-3}
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 80"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 60"
cd $R/bench || exit 1
run() {  # job arm round extra binary
  job=$1; arm=$2; rd=$3; extra=$4; bin=$5
  eval "A=\$$job"
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!
  o=$W/$T.$job.$arm.$rd.out
  $bin $A $extra > $o 2>&1
  kill $SMI
  g() { grep -h "^ *$1" $o | head -1 | awk '{print $NF=="ms" ? $(NF-1) : $NF}'; }
  printf "%-4s %-11s wall %8s | fill %7s apply %7s isect %6s | %s load=%s\n" $job $arm \
    "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" "$(g 'fill ')" "$(g 'apply ')" "$(g 'intersect')" \
    "$(awk -F', ' 'NR>4 {c+=$1; p+=$2; n++} END {printf "sm=%.0f p=%.0fW", c/n, p/n}' $W/smi.tmp)" "$(cut -d' ' -f1 /proc/loadavg)"
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
