#!/bin/sh
# Nsight Systems kernel timelines at I16, NFS@Home lpba-34 job, equal yield.
# Where are cuda-sieve's ~18 ms/q of non-kernel time, and why does the
# q-pipeline not help at I16? cuda-sieve pipelined and single-queued, 80 q
# from 297000000; CADO skip-off 8 slots, q in [297000000, 297001600).
R=/home/kylea/code/cuda-sieve; W=$R/work/nsys-i16-20261016; P=$R/work/applyprof; C=$R/work/cado-nfs-i16
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
NS="nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true"
load() { cut -d' ' -f1 /proc/loadavg; }
cd $R/bench || exit 1
for arm in "qp|" "single|--no-q-pipeline"; do
  name=${arm%%|*}; extra=${arm#*|}
  $NS -o $W/cuda-$name ./bench --pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 \
      --logI 16 --J 32768 --qrange 297000000: --nq 80 $extra > $W/cuda-$name.out 2>&1
  echo "cuda $name: exit=$? $(grep -h 'wall clock per q, COMPLETE' $W/cuda-$name.out | tr -s ' ') load=$(load)"
done
GLAS_SKIP_BOUND=0 $NS -o $W/cado-slots8 "$L" -poly $C/nfs.poly -q0 297000000 -q1 297001600 -A 31 \
    -lim0 200000000 -lim1 300000000 -lpb0 33 -lpb1 34 -mfb0 64 -mfb1 96 -fb1 $C/nfs.roots1.gz \
    -out $W/cado-slots8.rel.gz -t 16 -stats-stderr > $W/cado-slots8.out 2> $W/cado-slots8.err
echo "cado slots8: exit=$? $(grep -h '^# Total' $W/cado-slots8.err) load=$(load)"
for f in $W/*.nsys-rep; do nsys export --type sqlite --force-overwrite=true -o ${f%.nsys-rep}.sqlite $f > /dev/null 2>&1; echo "exported $(basename $f): $?"; done
echo done
