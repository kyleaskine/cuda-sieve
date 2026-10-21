#!/bin/sh
# Finding 121 traces: C143 I13 (800 q from 2200000, finding 120's arm) and
# c135 (2000 q from 7000000, CADO's workunit range), pipelined, on the
# finding-120 snapshot (base) and the q-prep snapshot under --q-prep auto.
# CUDA trace only. Each trace goes through finding 113's analyze.py and
# gaps.py, and finding 120's gapsplit-style full-gap sum (gapall.py).
R=/home/kylea/code/cuda-sieve; W=$R/work/qprep/nsys
B=$R/work/tdwait/exp/bench/bench; X=$R/work/qprep/exp/bench/bench
O=$R/oracle/nsys-c143-20261014
NS="nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true"
load() { cut -d' ' -f1 /proc/loadavg; }
mkdir -p $W
fail=0
C143="--pipeline --cofactor --poly $R/oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --qrange 2200000: --nq 800 --lambda0 1.905 --lambda1 1.94"
C135="--pipeline --cofactor --poly c135.job --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000: --nq 2000"
cd $R/work/c135 || exit 1
for job in c143 c135; do
  for arm in base auto; do
    case $arm in base) bin=$B; extra= ;; *) bin=$X; extra="--q-prep $arm" ;; esac
    eval "A=\$$(echo $job | tr a-z A-Z)"
    $NS -o $W/$job-$arm $bin $A $extra > $W/$job-$arm.out 2>&1
    rc=$?; [ $rc -eq 0 ] || fail=$((fail + 1))
    echo "$job $arm: exit=$rc $(grep -h 'wall clock per q, COMPLETE' $W/$job-$arm.out | tr -s ' ') load=$(load)"
  done
done
for f in $W/*.nsys-rep; do
  nsys export --type sqlite --force-overwrite=true -o ${f%.nsys-rep}.sqlite $f > /dev/null 2>&1
  rc=$?; [ $rc -eq 0 ] || fail=$((fail + 1))
done
for f in $W/*.sqlite; do
  python3 -I $O/analyze.py $f > ${f%.sqlite}.analyze.txt 2>&1 || fail=$((fail + 1))
  python3 -I $O/gaps.py $f > ${f%.sqlite}.gaps.txt 2>&1 || fail=$((fail + 1))
done
echo "failures: $fail"
[ $fail -eq 0 ]
