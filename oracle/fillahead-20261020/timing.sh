#!/bin/sh
# Finding 123 timing, quiet host: the two-ahead snapshot with and without
# --fill-ahead, and finding 121's installed bench/bench (5e975e96) for the
# default path, order rotated per round. c135: CADO's workunit
# [7000000,7100000); C143 I13: 3000 q from 2200000; c183 I15: 200 q. Each row
# carries exit status, the card's utilization just before it, and whether
# two-ahead engaged.
R=/home/kylea/code/cuda-sieve; W=$R/work/fillahead
B=${B:-$R/work/qprep/bench.5e975e96}; X=${X:-$W/exp/bench/bench}
C135="--pipeline --cofactor --poly $R/work/c135/c135.job --fb1 $R/work/c135/c135.roots1.m13 --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000:7100000"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200"
cd $R/bench || exit 1
fails=0
run() {  # job arm round
  job=$1; arm=$2; rd=$3
  eval "A=\$$job"
  case $arm in base) bin=$B; ex= ;; off) bin=$X; ex= ;; fa) bin=$X; ex=--fill-ahead ;; esac
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  o=$W/t.$job.$arm.$rd.out
  $bin $A $ex > $o 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  printf "%-4s %-4s r%s %7s ms | %s | load=%s idle=%s%% rc=%s\n" $job $arm $rd \
    "$(grep -h 'wall clock per q, COMPLETE' $o | awk '{print $(NF-1)}')" \
    "$(grep -h 'two ahead\|--fill-ahead: off' $o | tr -s ' ' | cut -c1-70)" \
    "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
for rd in 1 2 3 4; do
  case $rd in 1) arms="base off fa";; 2) arms="fa base off";; 3) arms="off fa base";; *) arms="fa off base";; esac
  for job in C135 C143; do for arm in $arms; do run $job $arm $rd; done; done
done
for rd in 1 2; do for arm in off fa; do run C183 $arm $rd; done; done
echo "done: $fails failed run(s)"
[ $fails -eq 0 ]
