#!/bin/sh
# cado-gpu with its small-prime skip off (GLAS_SKIP_BOUND=0, equal yield)
# against cuda-sieve, quiet host, card free, alternating.
#
# c135: CADO's workunit [7000000,7100000), las-glas as its client ran it
#   (finding 122) with the skip off, three rounds; once with the skip on for
#   reference. Ours: the two-ahead snapshot with --fill-ahead (finding 123,
#   not yet installed), and the installed bench/bench (finding 121).
# NFS@Home I16: finding 115's head-to-head, q in [297000000,297006000), CADO
#   skip off, two rounds, against the installed bench/bench (two-ahead does
#   not engage on its four-slab plan).
# Wall by /usr/bin/time; relations counted (raw, not deduplicated).
R=/home/kylea/code/cuda-sieve; W=$R/work/skipoff
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
D=/home/kylea/cado-gpu/c136-934760359018/client/download
N=$R/work/cado-nfs-i16; P=$R/work/applyprof
FA=$R/work/fillahead/exp/bench/bench; INST=$R/bench/bench
cd $W || exit 1
fails=0
load() { cut -d' ' -f1 /proc/loadavg; }
c135_cado() {  # name env...
  name=$1; shift; o=$W/c135.cado-$name
  env "$@" /usr/bin/time -f %e -o $o.time $L -poly $D/c135.poly -q0 7000000 -I 13 -q1 7100000 \
      -lim0 9000000 -lim1 12000000 -lpb0 29 -lpb1 30 -mfb0 55 -mfb1 58 -lambda0 1.89 -lambda1 1.94 \
      -ncurves0 12 -ncurves1 14 -fb1 $D/c135.roots1.gz -out $o.rel.gz -t 16 -stats-stderr \
      > $o.out 2> $o.err; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  echo "c135 cado $name: $(cat $o.time) s | $(zcat $o.rel.gz | grep -vc '^#') rels | $(grep -h '^# Total' $o.err | sed 's/^# //') load=$(load) rc=$rc"
}
c135_ours() {  # name binary extra...
  name=$1; bin=$2; shift 2; o=$W/c135.ours-$name
  /usr/bin/time -f %e -o $o.time $bin --pipeline --cofactor --poly $R/work/c135/c135.job \
      --fb1 $R/work/c135/c135.roots1.m13 --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 \
      --lambda0 1.89 --lambda1 1.94 --qrange 7000000:7100000 "$@" --relations $o.rel > $o.out 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  echo "c135 ours $name: $(cat $o.time) s | $(wc -l < $o.rel) rels | $(grep -h 'wall clock per q, COMPLETE' $o.out | tr -s ' ') load=$(load) rc=$rc"
}
i16_cado() {  # name env...
  name=$1; shift; o=$W/i16.cado-$name
  env "$@" /usr/bin/time -f %e -o $o.time $L -poly $N/nfs.poly -q0 297000000 -q1 297006000 -A 31 \
      -lim0 200000000 -lim1 300000000 -lpb0 33 -lpb1 34 -mfb0 64 -mfb1 96 -fb1 $N/nfs.roots1.gz \
      -out $o.rel.gz -t 16 -stats-stderr > $o.out 2> $o.err; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  echo "i16 cado $name: $(cat $o.time) s | $(zcat $o.rel.gz | grep -vc '^#') rels | $(grep -h '^# Total' $o.err | sed 's/^# //') load=$(load) rc=$rc"
}
i16_ours() {  # name
  name=$1; o=$W/i16.ours-$name
  (cd $R/bench && /usr/bin/time -f %e -o $o.time $INST --pipeline --cofactor --poly $P/nfs.job \
      --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 297000000:297005999 \
      --relations $o.rel > $o.out 2>&1); rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  echo "i16 ours $name: $(cat $o.time) s | $(wc -l < $o.rel) rels | $(grep -h 'wall clock per q, COMPLETE' $o.out | tr -s ' ') load=$(load) rc=$rc"
}
echo "bench/bench $(sha256sum $INST | cut -c1-16), two-ahead snapshot $(sha256sum $FA | cut -c1-16), load=$(load)"
c135_cado skip0.1 GLAS_SKIP_BOUND=0
c135_ours fa.1 $FA --fill-ahead
c135_ours inst.1 $INST
c135_cado skip0.2 GLAS_SKIP_BOUND=0
c135_ours fa.2 $FA --fill-ahead
c135_cado default
c135_ours fa.3 $FA --fill-ahead
c135_cado skip0.3 GLAS_SKIP_BOUND=0
i16_cado skip0.1 GLAS_SKIP_BOUND=0
i16_ours 1
i16_ours 2
i16_cado skip0.2 GLAS_SKIP_BOUND=0
echo "done: $fails failed"
[ $fails -eq 0 ]
