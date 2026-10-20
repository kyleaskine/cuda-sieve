#!/bin/sh
# After-trace for finding 120: the same arms on the change (BIN), outputs prefixed new-.
# Same arms as findings 113 (C143 I13, 800 q from 2200000) and 115 (NFS@Home
# I16, 80 q from 297000000), cuda-sieve only, pipelined and single-queued.
# CUDA trace only: no CPU sampling, no context switches.
R=/home/kylea/code/cuda-sieve; W=$R/work/nsys-20261020; P=$R/work/applyprof
BIN=${BIN:-$R/work/tdwait/exp/bench/bench}
O=$R/oracle/nsys-c143-20261014; K=$R/oracle/i16-h2h-20261016
NS="nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true"
load() { cut -d' ' -f1 /proc/loadavg; }
fail=0
cd $R || exit 1
echo "bin sha256 $(sha256sum $BIN | cut -c1-16)  HEAD $(git rev-parse --short HEAD)  load=$(load)"
for arm in "qp|" "single|--no-q-pipeline"; do
  name=${arm%%|*}; extra=${arm#*|}
  $NS -o $W/new-c143-$name $BIN --pipeline --cofactor --poly oracle/c143.job --logI 13 --J 8192 \
      --region 13 --maxbits 13 --sq-side 1 --qrange 2200000: --nq 800 --lambda0 1.905 --lambda1 1.94 $extra \
      > $W/new-c143-$name.out 2>&1
  rc=$?; [ $rc -eq 0 ] || fail=$((fail + 1))
  echo "c143 $name: exit=$rc $(grep -h 'wall clock per q, COMPLETE' $W/new-c143-$name.out | tr -s ' ') load=$(load)"
done
cd $R/bench || exit 1
for arm in "qp|" "single|--no-q-pipeline"; do
  name=${arm%%|*}; extra=${arm#*|}
  $NS -o $W/new-i16-$name $BIN --pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 \
      --logI 16 --J 32768 --qrange 297000000: --nq 80 $extra > $W/new-i16-$name.out 2>&1
  rc=$?; [ $rc -eq 0 ] || fail=$((fail + 1))
  echo "i16 $name: exit=$rc $(grep -h 'wall clock per q, COMPLETE' $W/new-i16-$name.out | tr -s ' ') load=$(load)"
done
for f in $W/new-*.nsys-rep; do
  nsys export --type sqlite --force-overwrite=true -o ${f%.nsys-rep}.sqlite $f > /dev/null 2>&1
  rc=$?; [ $rc -eq 0 ] || fail=$((fail + 1))
  echo "exported $(basename $f): $rc"
done
for f in $W/new-*.sqlite; do
  python3 $O/analyze.py $f > ${f%.sqlite}.analyze.txt 2>&1 || fail=$((fail + 1))
  python3 $O/gaps.py $f > ${f%.sqlite}.gaps.txt 2>&1 || fail=$((fail + 1))
  python3 -c "import sys; exec(open('$K/kernsum.py').read().split('for p, x in')[0]); main(sys.argv[1], 'k_transform')" \
      $f > ${f%.sqlite}.kernsum.txt 2>&1 || fail=$((fail + 1))
done
echo "failures: $fail"
[ $fail -eq 0 ]
