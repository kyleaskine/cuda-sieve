#!/bin/sh
# Gates for landing findings 130-131 (work/land130/fat, the fat build of the
# tree): bucket records carry log p; --tiny-table adds moduli 3, 5, 7, 9 at
# apply's init from per-region tables every warp helps build, wherever they
# carry >= 1 hit per position. Output must stay byte-identical. Base:
# work/tinytab/base (abcheck outputs cached). A FAIL line, a missing PASS or a
# nonzero exit fails a check; every check counts toward the exit status.
R=/home/kylea/code/cuda-sieve; W=$R/work/land130/gate; T=$R/work/tinytab
F=$R/work/land130/fat/bench/bench; B=$T/base/bench/bench
fail=0
echo "base $(sha256sum $B | cut -c1-16)  new $(sha256sum $F | cut -c1-16)  load=$(cut -d' ' -f1 /proc/loadavg)  card $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
if ps -eo args | grep -v grep | grep -qE 'ggnfs-sieve-client|run-cuda-client'; then echo "   client running"; fail=$((fail+1)); fi

echo "== build: the tree's sources (docs aside), six archs, no warnings"
nd=0
for f in $(git -C $R ls-files bench | grep -v '^bench/attic/' | grep -v '\.md$'); do
  cmp -s $R/$f $R/work/land130/fat/$f || { echo "   differs from the tree: $f"; nd=$((nd+1)); }
done
archs=$(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $R/work/land130/fat.build.log | sort -u | wc -l)
warns=$(grep -i warning $R/work/land130/fat.build.log | grep -vc minnctapersm)
echo "   $(cat $R/work/land130/fat.done) | sources differing $nd | archs $archs | warnings $warns"
grep -q 'exit=0' $R/work/land130/fat.done && [ $nd -eq 0 ] && [ "$archs" -eq 6 ] && [ "$warns" -eq 0 ] || fail=$((fail+1))

echo "== k_apply on every arch: <= 40 registers, no stack (no spill)"
cuobjdump -res-usage $F 2>/dev/null | awk '/^ *Function _Z7k_apply/ {f=1; next} f && /REG:/ {print; f=0}' > $W/resusage.txt
nk=$(wc -l < $W/resusage.txt)
nbad=$(awk '{for(i=1;i<=NF;i++){if($i ~ /^REG:/){split($i,a,":"); if(a[2]>40) b=1} if($i ~ /^STACK:/){split($i,a,":"); if(a[2]>0) b=1}} if(b) n++; b=0} END {print n+0}' $W/resusage.txt)
echo "   k_apply entries $nk, over 40 registers or with stack $nbad"
[ "$nk" -ge 30 ] && [ "$nbad" -eq 0 ] || fail=$((fail+1))

cd $R/work/land130/fat/bench || exit 1
echo "== --verify-only and fbtest"
v=$($F --verify-only 2>&1 | grep -c '\[verify\] OK')
echo "   verify-only OK lines: $v (3 expected)"; [ "$v" -eq 3 ] || fail=$((fail+1))
./fbtest --fb1 ../oracle/c183.fb1 > $W/fbtest.log 2>&1; rc=$?
echo "   fbtest exit=$rc $(tail -1 $W/fbtest.log)"; [ $rc -eq 0 ] || fail=$((fail+1))

echo "== apply cell parity (--verify: one region replayed on the CPU from every small-sieve entry)"
C183="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1"
C1147="--poly $R/work/c1147/c1147.job --fb1 $R/work/c1147/c1147.roots1.m16 --maxbits 16"
vf() {  # name want-fold(y|n) args...
  name=$1; wantf=$2; shift 2; o=$W/vf.$name.out
  $F "$@" --verify > $o 2>&1; rc=$?
  ok0=$(grep -c ' 0 cells differ' $o); nd=$(grep -c 'cells differ' $o)
  nf=$(grep -h -- '--tiny-fold:' $o | awk '{print $2}' | sort -n | head -1)
  r=PASS
  [ $rc -eq 0 ] && [ "$nd" -ge 1 ] && [ "$ok0" -eq "$nd" ] || r=FAIL
  if [ $wantf = y ]; then [ "${nf:-0}" -gt 0 ] || r=FAIL; else [ -z "$nf" ] || r=FAIL; fi
  [ $r = FAIL ] && fail=$((fail+1))
  echo "   $r $name: exit $rc, regions checked $nd, with 0 cells differing $ok0, folded entries ${nf:-none}"
}
vf s1.I15        y $C183 --logI 15 --side 1
vf s0.I15        y $C183 --logI 15 --side 0
vf s1.I15.r10    y $C183 --logI 15 --side 1 --region 10
vf s1.I13.r13    y $C183 --logI 13 --J 4096 --side 1 --region 13
vf s0.I16        y $C183 --logI 16 --J 2048 --side 0
vf s1.I16.t96    y $C183 --logI 16 --J 2048 --side 1 --apply-threads 96
vf s0.I14.t160   y $C183 --logI 14 --J 8192 --side 0 --apply-threads 160
vf s1.I15.nofold n $C183 --logI 15 --side 1 --no-tiny-fold
vf c1147.s1.I16  y $C1147 --logI 16 --J 2048 --side 1
vf c1147.s0.I16  y $C1147 --logI 16 --J 2048 --side 0
vf c1147.s1.nofold n $C1147 --logI 16 --J 2048 --side 1 --no-tiny-fold
vf s1.I15.notable y $C183 --logI 15 --side 1 --no-tiny-table
vf s0.I15.notable y $C183 --logI 15 --side 0 --no-tiny-table
vf c1147.s1.notable y $C1147 --logI 16 --J 2048 --side 1 --no-tiny-table

echo "== 2,1147+: relations byte-identical to base (10 q at 300M; base from work/tinytab/cliff.sh)"
A="--pipeline --cofactor --poly $R/work/c1147/c1147.job --fb1 $R/work/c1147/c1147.roots1.m16 --logI 16 --J 32768 --maxbits 16 --qrange 300000000: --nq 10"
for mode in sq pl; do for arm in def off; do
  ex=; [ $mode = sq ] && ex=--no-q-pipeline; ax=; [ $arm = off ] && ax="--no-tiny-fold --no-tiny-table"
  o=$W/id.c1147.$mode.$arm
  $F $A $ex $ax --relations $o.rel > $o.out 2>&1; rc=$?
  ref=$T/cl.C1147.$mode.base.1.rel
  if [ $rc -eq 0 ] && [ -s $o.rel ] && cmp -s $o.rel $ref; then r=PASS; else r=FAIL; fail=$((fail+1)); fi
  echo "   $r $mode $arm: exit $rc, $(wc -l < $o.rel) relations against $(wc -l < $ref) | $(grep -h -- '--q-pipeline:' $o.out | cut -c1-70)"
done; done

cd $R/bench || exit 1
ab() {
  label=$1; shift
  echo "== abcheck: $label"
  rm -rf "$T/abcheck/run"
  env ABDIR=$T/abcheck "$@" ./abcheck.sh "$B" "$F" > "$W/ab.tmp" 2>&1
  rc=$?
  sed 's/^/   /' "$W/ab.tmp"
  echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
}
ab "default (fold and tables on), every case"
ab "--no-tiny-table on NEW's pipeline runs (harness cases keep the tables)" NEWARGS="--no-tiny-table"

echo "== make check (fat build)"
(cd $R/work/land130/fat/bench && nice make check > $W/check.log 2>&1); rc=$?
echo "   $(grep -c '^PASS' $W/check.log) PASS, $(grep -c '^FAIL' $W/check.log) FAIL; $(tail -1 $W/check.log)"
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "== fillcheck.sh (fill writes the logs into the records)"
(cd $R/work/land130/fat/bench && ./fillcheck.sh > $W/fillcheck.log 2>&1); rc=$?
tail -2 $W/fillcheck.log | sed 's/^/   /'
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))


echo "== review additions"
cd $R/work/land130/fat/bench || exit 1
for f in $W/id.c1147.pl.def.out $W/id.c1147.sq.def.out; do
  b=$(grep -h '  apply: ' $f)
  case "$b" in *"apply: 3 blocks of 512 threads per SM, 32976 B"*) r=PASS ;; *) r=FAIL; fail=$((fail+1)) ;; esac
  case "$b" in *"shared memory is the limit"*) r=FAIL; fail=$((fail+1)) ;; esac
  echo "   $r banner $(basename $f): $b"
done
$F --pipeline --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --no-smallsieve --no-tiny-fold --nq 1 > $W/refuse.out 2>&1; rc=$?
if [ $rc -ne 0 ] && grep -q -- '--no-tiny-fold changes nothing under --no-smallsieve' $W/refuse.out; then r=PASS; else r=FAIL; fail=$((fail+1)); fi
echo "   $r --no-smallsieve --no-tiny-fold refused (exit $rc)"
$F --pipeline --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --no-smallsieve --no-tiny-table --nq 1 > $W/refuse2.out 2>&1; rc=$?
if [ $rc -ne 0 ] && grep -q -- '--no-tiny-table changes nothing under --no-smallsieve' $W/refuse2.out; then r=PASS; else r=FAIL; fail=$((fail+1)); fi
echo "   $r --no-smallsieve --no-tiny-table refused (exit $rc)"
$F --pipeline --cofactor --poly $R/work/applyprof/nfs.job --fb1 $R/work/applyprof/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 2 > $W/banner.i16.out 2>&1; rc=$?
b=$(grep -h '  apply: ' $W/banner.i16.out)
case "$b" in *"apply: 3 blocks of 512 threads per SM"*) r=PASS ;; *) r=FAIL; fail=$((fail+1)) ;; esac
[ $rc -eq 0 ] || { r=FAIL; fail=$((fail+1)); }
echo "   $r banner I16: $b"

echo "done: $fail failed"
[ "$fail" -eq 0 ]
