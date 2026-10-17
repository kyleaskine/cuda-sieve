#!/bin/sh
# cado-gpu vs cuda-sieve on the NFS@Home lpba-34 I16 job (Kyle's production
# geometry), q in [297000000, 297006000). CADO: las-glas, -A 31, the job's
# lim/lpb/mfb (no lambda: CADO's threshold is then mfb, nearest to ours),
# skip off for equal yield; cuda-sieve 3e79628 + finding 114, defaults.
R=/home/kylea/code/cuda-sieve; W=$R/work/cado-nfs-i16; P=$R/work/applyprof
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
load() { cut -d' ' -f1 /proc/loadavg; }
cado() {   # name env...
    name=$1; shift
    env "$@" /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S' -o $W/cado-$name.time \
      $L -poly $W/nfs.poly -q0 297000000 -q1 297006000 -A 31 -lim0 200000000 -lim1 300000000 \
      -lpb0 33 -lpb1 34 -mfb0 64 -mfb1 96 -fb1 $W/nfs.roots1.gz -out $W/cado-$name.rel.gz \
      -t 16 -stats-stderr > $W/cado-$name.out 2> $W/cado-$name.err
    echo "cado $name: $(grep -h '^# Total' $W/cado-$name.err | head -1 | sed 's/^# //') | $(cat $W/cado-$name.time) load=$(load)"
    grep -E '^# GPU (transform|fill|region|tiny|resieve_s|filter|ecm|chunk span)' $W/cado-$name.err | sed 's/^/    /'
}
cuda() {   # name extra...
    name=$1; shift
    ( cd $R/bench && /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S' -o $W/cuda-$name.time \
      ./bench --pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 \
        --qrange 297000000:297005999 "$@" --relations $W/cuda-$name.rel > $W/cuda-$name.out 2>&1 )
    echo "cuda $name: rel=$(wc -l < $W/cuda-$name.rel) q=$(grep -oE '[0-9]+ q +[0-9]+ rel' $W/cuda-$name.out | tail -1) | $(cat $W/cuda-$name.time) | $(grep -h 'wall clock per q, COMPLETE' $W/cuda-$name.out | tr -s ' ') load=$(load)"
}
echo "== end to end"
cado skip0 GLAS_SKIP_BOUND=0
cuda qp
cado skip0b GLAS_SKIP_BOUND=0
cuda qpb
cado default
echo "== per kernel"
cado slot1-skip0 GLAS_SKIP_BOUND=0 GLAS_GPU_SLOTS=1 GLAS_GPU_STATS=1
cuda single --no-q-pipeline
echo "== relation sets"
python3 $R/oracle/c143-fill-20261013/relcmp.py $W/cado-skip0.rel.gz $W/cuda-qp.rel 200000000 300000000 | head -14
echo done
