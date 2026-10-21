#!/bin/sh
# --fill-ahead as the default: the fat build (fat/bench/bench) at its default
# and with --no-fill-ahead, against finding 121's installed bench/bench
# (5e975e96), order rotated per round. Jobs as finding 123's timing.sh. The
# host carries Kyle's CPU ECM load; each row records it. Not a gate: every
# run must exit 0, and the relations of the three arms must match.
R=/home/kylea/code/cuda-sieve; W=$R/work/fadefault
B=$R/work/qprep/bench.5e975e96; X=$W/fat/bench/bench
C135="--pipeline --cofactor --poly $R/work/c135/c135.job --fb1 $R/work/c135/c135.roots1.m13 --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000:7100000"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 3000"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200"
cd $R/bench || exit 1
fails=0
run() {  # job arm round
  job=$1; arm=$2; rd=$3
  eval "A=\$$job"
  case $arm in base) bin=$B; ex= ;; off) bin=$X; ex=--no-fill-ahead ;; def) bin=$X; ex= ;; esac
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  o=$W/t.$job.$arm.$rd
  $bin $A $ex --relations $o.rel > $o.out 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  if ! cmp -s $o.rel $W/t.$job.base.1.rel 2>/dev/null && [ -s $W/t.$job.base.1.rel ]; then
    echo "   relations of $job $arm r$rd differ from base r1"; fails=$((fails+1))
  fi
  printf "%-4s %-4s r%s %7s ms | %s | load=%s card=%s%% rc=%s\n" $job $arm $rd \
    "$(grep -h 'wall clock per q, COMPLETE' $o.out | awk '{print $(NF-1)}')" \
    "$(grep -h 'two ahead\|--fill-ahead: off' $o.out | tr -s ' ' | cut -c1-60)" \
    "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
echo "base $(sha256sum $B | cut -c1-16)  new $(sha256sum $X | cut -c1-16)"
# ROUNDS: extra rounds (4 5 6 7) skip C183, which needs base r1 to compare
for rd in ${ROUNDS:-1 2 3}; do
  case $((rd % 3)) in 1) arms="base off def";; 2) arms="def base off";; *) arms="off def base";; esac
  for job in C135 C143; do for arm in $arms; do run $job $arm $rd; done; done
done
[ -z "$ROUNDS" ] && for arm in base off def; do run C183 $arm 1; done
echo "done: $fails failed"
[ $fails -eq 0 ]
