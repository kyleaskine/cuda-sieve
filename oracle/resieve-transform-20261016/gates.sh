#!/bin/sh
# Gates for the fast transform + resieve coarse filter, final source, sm_120
# snapshot builds (bench/bench, the client's binary, untouched):
# verify-only, make check (fbtest's new exact transform gates), fillcheck,
# identity on four jobs, stop/resume (C194 four slabs, C143), skipslab
# (BN_LIMBS=4), TD-list truncation (PIPE_K=8) below and above the coarse
# threshold, single-queued vs pipelined.
R=/home/kylea/code/cuda-sieve; W=$R/work/resieve-i16; X=$W/exp/bench/bench
H=$R/work/hostwait; P=$R/work/applyprof
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768"
cd $W/exp/bench || exit 1
echo "build: $(cat $W/build2.status 2>/dev/null) | warnings $(grep -ci warning $W/exp/build2.log)"
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
nice -n 5 make check GPU_ARCH=120 > $W/check.log 2>&1; echo "make check exit=$? ($(grep -c '^PASS' $W/check.log) PASS, $(grep -c '^FAIL' $W/check.log) FAIL) $(tail -1 $W/check.log)"
grep -E "fast path" $W/check.log | tr -s ' ' | sed 's/^/  /'
./fillcheck.sh > $W/fillcheck.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $W/fillcheck.log) ok) $(tail -1 $W/fillcheck.log)"
cd $R/bench || exit 1
echo "== identity vs the reference relation files"
id() { name=$1; ref=$2; shift 2
  rm -f "${W:?}"/g.$name.rel "${W:?}"/g.$name.rel.part "${W:?}"/g.$name.rel.part.ckpt "${W:?}"/g.$name.rel.lock
  $X "$@" --relations $W/g.$name.rel > $W/g.$name.out 2>&1
  cmp -s $ref $W/g.$name.rel && printf "%s OK(%s) " $name "$(wc -l < $W/g.$name.rel)" || printf "%s DIFFER " $name; }
id nfs $P/id.nfs.base.rel $NFS --qrange 296847000: --nq 40
id c183 $P/id.c183.base.rel $C183 --qrange 130000000: --nq 200
id c143 $H/id.c143.base.rel $C143 --qrange 2200000: --nq 1000
id c194 $H/id.c194.base.rel $C194 --qrange 80000023: --nq 60
echo
echo "== stop/resume"
stoprun() { name=$1; ref=$2; delay=$3; shift 3
  D=$W/stop-$name; mkdir -p "${D:?}"; rm -f "${D:?}/STOP" "${D:?}/qp.rel" "${D:?}/qp.rel.part" "${D:?}/qp.rel.part.ckpt" "${D:?}/qp.rel.lock"
  ( sleep $delay; touch "$D/STOP" ) &
  $X "$@" --stop-file "$D/STOP" --relations "$D/qp.rel" > "$D/qp1.out" 2>&1
  wait
  if grep -q "stopping cleanly" "$D/qp1.out"; then st="$(grep -h 'stopped after' "$D/qp1.out" | tr -s ' ' | cut -c1-60)"; else st="NOT STOPPED -- vacuous"; fi
  rm -f "${D:?}/STOP"
  $X "$@" --stop-file "$D/STOP" --relations "$D/qp.rel" > "$D/qp2.out" 2>&1
  cmp -s "$ref" "$D/qp.rel" && echo "  $name: $st | resumed IDENTICAL ($(wc -l < "$D/qp.rel"))" || echo "  $name: $st | resumed DIFFER"; }
stoprun c194 $R/work/slabpipe/stop2/base.rel 45 $C194 --qrange 80000023: --nq 400
stoprun c143 $H/gates/c143-base.rel 12 $C143 --qrange 2200000: --nq 4000
echo "== skipslab (BN_LIMBS=4)"
cat $W/l4.done
( cd $W/l4/bench && sh $R/work/slabpipe/skipslab.sh > $W/skipslab.log 2>&1; echo "skipslab exit=$?" ); grep -E "^PASS|^FAIL|all cases|FAILED" $W/skipslab.log | tr -s ' '
echo "== TD-list truncation (PIPE_K=8), single-queued vs pipelined"
cat $W/pk8.done
for v in "slab4095|--slab-j 4095" "oneslab|"; do name=${v%%|*}; ex=${v#*|}
  A="--pipeline --cofactor --poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 $ex --qrange 120000000: --nq 150"
  $W/pk8/bench/bench $A --no-q-pipeline --relations $W/tdskip.$name.single.rel > $W/tdskip.$name.single.out 2>&1
  $W/pk8/bench/bench $A --relations $W/tdskip.$name.qp.rel > $W/tdskip.$name.qp.out 2>&1
  s1=$(grep -h 'slab(s) skipped or abandoned' $W/tdskip.$name.single.out | grep -oE '^ *[0-9]+' | tr -d ' '); s2=$(grep -h 'slab(s) skipped or abandoned' $W/tdskip.$name.qp.out | grep -oE '^ *[0-9]+' | tr -d ' ')
  cmp -s $W/tdskip.$name.single.rel $W/tdskip.$name.qp.rel && r="IDENTICAL ($(wc -l < $W/tdskip.$name.qp.rel))" || r="DIFFER"
  echo "  $name: $r | skipped single ${s1:-0} qp ${s2:-0}"
done
echo done
