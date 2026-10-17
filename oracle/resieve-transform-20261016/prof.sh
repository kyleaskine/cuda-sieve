#!/bin/sh
# ncu full-set profile of k_resieve_scatter and k_transform on the NFS@Home
# lpba-34 I16 job, q index 3 of the h2h band (finding 115). Single-queued so
# launch order is fixed: per slab, side 1 then side 0 -> 8 resieve launches
# per q (skip 24); k_transform has one n=0 warm-up launch then 2 per q (skip 7).
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/resieve-i16; P=/home/kylea/code/cuda-sieve/work/applyprof
NFS="--poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 5"
git describe --always --dirty > $W/build.txt
ncu --set full --import-source yes --kernel-name k_resieve_scatter --launch-skip 24 --launch-count 8 \
    -f -o $W/resieve ./bench --pipeline --cofactor $NFS --no-q-pipeline > $W/resieve.log 2>&1
echo "resieve ncu exit=$?"
ncu --set full --import-source yes --kernel-name k_transform --launch-skip 7 --launch-count 2 \
    -f -o $W/transform ./bench --pipeline --cofactor $NFS --no-q-pipeline > $W/transform.log 2>&1
echo "transform ncu exit=$?"
echo done
