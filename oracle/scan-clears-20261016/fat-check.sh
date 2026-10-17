#!/bin/sh
# The rebuilt fat bench/bench (the client's binary): archs, warnings,
# --verify-only, fbtest, and identity on the four jobs.
R=/home/kylea/code/cuda-sieve; W=$R/work/scan-i16; H=$R/work/hostwait; P=$R/work/applyprof
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768"
cd $R/bench || exit 1
echo "fat: $(cat $W/fat.done) | archs $(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $W/fat.build.log | sort -u | wc -l) | warnings $(grep -i warning $W/fat.build.log | grep -vc minnctapersm) | bench/bench $(sha256sum bench | cut -c1-16) built $(stat -c %y bench | cut -c1-19)"
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
./fbtest --fb1 ../oracle/c183.fb1 > $W/fat-fbtest.log 2>&1; echo "fbtest exit=$? $(tail -1 $W/fat-fbtest.log)"
id() { name=$1; ref=$2; shift 2
  rm -f "${W:?}"/f.$name.rel "${W:?}"/f.$name.rel.part "${W:?}"/f.$name.rel.part.ckpt "${W:?}"/f.$name.rel.lock
  ./bench "$@" --relations $W/f.$name.rel > $W/f.$name.out 2>&1
  cmp -s $ref $W/f.$name.rel && printf "%s OK(%s) " $name "$(wc -l < $W/f.$name.rel)" || printf "%s DIFFER " $name; }
id nfs $P/id.nfs.base.rel $NFS --qrange 296847000: --nq 40
id c183 $P/id.c183.base.rel $C183 --qrange 130000000: --nq 200
id c143 $H/id.c143.base.rel $C143 --qrange 2200000: --nq 1000
id c194 $H/id.c194.base.rel $C194 --qrange 80000023: --nq 60
echo
