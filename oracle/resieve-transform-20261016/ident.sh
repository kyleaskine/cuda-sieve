#!/bin/sh
# Identity of the experiment build (fast transform + coarse resieve filter)
# against the reference relation files of findings 110/114 (byte-identical
# across 3e79628 and fef825c), on the four jobs.
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench
H=$R/work/hostwait; P=$R/work/applyprof
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768"
cd $R/bench || exit 1
id() {  # name ref env... -- args...
  name=$1; ref=$2; shift 2
  rm -f "${W:?}"/id.$name.rel "${W:?}"/id.$name.rel.part "${W:?}"/id.$name.rel.part.ckpt "${W:?}"/id.$name.rel.lock
  env "$@" --relations $W/id.$name.rel > $W/id.$name.out 2>&1
  cmp -s $ref $W/id.$name.rel && echo "$name OK($(wc -l < $W/id.$name.rel))" || echo "$name DIFFER ($(wc -l < $W/id.$name.rel) vs $(wc -l < $ref))"
}
id nfs $P/id.nfs.base.rel $X $NFS --qrange 296847000: --nq 40
id nfs.rs0 $P/id.nfs.base.rel BENCH_TEST_RS=0 $X $NFS --qrange 296847000: --nq 40
id nfs.t512 $P/id.nfs.base.rel BENCH_TEST_RS_THREADS=512 $X $NFS --qrange 296847000: --nq 40
id c183 $P/id.c183.base.rel $X $C183 --qrange 130000000: --nq 200
id c143 $H/id.c143.base.rel $X $C143 --qrange 2200000: --nq 1000
id c194 $H/id.c194.base.rel $X $C194 --qrange 80000023: --nq 60
echo done
