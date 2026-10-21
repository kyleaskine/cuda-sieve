#!/bin/sh
# c135 kernel timelines, quiet host: cuda-sieve (bench/bench, --q-prep auto)
# and cado-gpu (las-glas as its client ran it), q in [7000000,7020000), about
# 1,290 special-q each. CUDA trace only. Per-kernel ms per special-q over the
# middle 80% of launches (finding 115's kernsum.py), finding 113's analyze.py
# and gaps.py on both.
R=/home/kylea/code/cuda-sieve; W=$R/work/prof-c135
D=/home/kylea/cado-gpu/c136-934760359018/client/download
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
O=$R/oracle/nsys-c143-20261014; K=$R/oracle/i16-h2h-20261016
NS="nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true"
load() { cut -d' ' -f1 /proc/loadavg; }
cd $W || exit 1
fail=0
$NS -o $W/ours $R/bench/bench --pipeline --cofactor --poly $R/work/c135/c135.job --logI 13 --J 4096 \
    --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000:7020000 \
    > $W/ours.trace.out 2>&1 || fail=$((fail + 1))
echo "ours: $(grep -h 'wall clock per q, COMPLETE' $W/ours.trace.out | tr -s ' ') load=$(load)"
$NS -o $W/cado $L -poly $D/c135.poly -q0 7000000 -I 13 -q1 7020000 \
    -lim0 9000000 -lim1 12000000 -lpb0 29 -lpb1 30 -mfb0 55 -mfb1 58 -lambda0 1.89 -lambda1 1.94 \
    -ncurves0 12 -ncurves1 14 -fb1 $D/c135.roots1.gz -out $W/cado.trace.rel.gz -t 16 -stats-stderr \
    > $W/cado.trace.out 2> $W/cado.trace.err || fail=$((fail + 1))
echo "cado: $(grep -h '^# Total' $W/cado.trace.err) load=$(load)"
for f in ours cado; do
  nsys export --type sqlite --force-overwrite=true -o $W/$f.sqlite $W/$f.nsys-rep > /dev/null 2>&1 || fail=$((fail + 1))
  python3 -I $O/analyze.py $W/$f.sqlite > $W/$f.analyze.txt 2>&1 || fail=$((fail + 1))
  python3 -I $O/gaps.py $W/$f.sqlite > $W/$f.gaps.txt 2>&1 || fail=$((fail + 1))
done
python3 -I -c "import sys; exec(open('$K/kernsum.py').read().split('for p, x in')[0]); main(sys.argv[1], 'k_transform')" \
    $W/ours.sqlite > $W/ours.kernsum.txt 2>&1 || fail=$((fail + 1))
python3 -I -c "import sys; exec(open('$K/kernsum.py').read().split('for p, x in')[0]); main(sys.argv[1], 'transform_kernel')" \
    $W/cado.sqlite > $W/cado.kernsum.txt 2>&1 || fail=$((fail + 1))
echo "failures: $fail"
[ $fail -eq 0 ]
