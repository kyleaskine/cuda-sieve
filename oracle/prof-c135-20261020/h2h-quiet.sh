#!/bin/sh
# c135, CADO's workunit [7000000,7100000), both sievers on a quiet host,
# alternating, three rounds. CADO with its client's command from the run's
# localhost.log (las-glas from the gpu-siever tree, beside glas-gpu, on the
# run's poly and roots); ours with c135.job and
# the default --q-prep auto. Wall by /usr/bin/time, relations counted,
# card clock/power logged per run.
R=/home/kylea/code/cuda-sieve; W=$R/work/prof-c135
D=/home/kylea/cado-gpu/c136-934760359018/client/download
X=$R/bench/bench; L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
cd $W || exit 1
fails=0
smi() { nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!; }
smistat() { kill $SMI; awk -F', ' 'NR>2 {c+=$1; p+=$2; n++} END {if (n) printf "sm=%.0f p=%.0fW", c/n, p/n}' $W/smi.tmp; }
cado() {
  rd=$1; o=$W/cado.$rd
  smi
  /usr/bin/time -f %e -o $o.time $L -poly $D/c135.poly -q0 7000000 -I 13 -q1 7100000 \
      -lim0 9000000 -lim1 12000000 -lpb0 29 -lpb1 30 -mfb0 55 -mfb1 58 -lambda0 1.89 -lambda1 1.94 \
      -ncurves0 12 -ncurves1 14 -fb1 $D/c135.roots1.gz -out $o.rel.gz -t 16 -stats-stderr \
      > $o.out 2> $o.err; rc=$?
  st=$(smistat); [ $rc -ne 0 ] && fails=$((fails+1))
  echo "cado r$rd total $(cat $o.time)s | $(zcat $o.rel.gz | grep -vc '^#') rels | $(grep -h '^# Total' $o.err) | $st load=$(cut -d' ' -f1 /proc/loadavg) rc=$rc"
}
ours() {
  rd=$1; o=$W/ours.$rd
  smi
  /usr/bin/time -f %e -o $o.time $X --pipeline --cofactor --poly $R/work/c135/c135.job --logI 13 --J 4096 \
      --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000:7100000 \
      --relations $o.rel > $o.out 2>&1; rc=$?
  st=$(smistat); [ $rc -ne 0 ] && fails=$((fails+1))
  echo "ours r$rd total $(cat $o.time)s | $(wc -l < $o.rel) rels | $(grep -h 'wall clock per q, COMPLETE' $o.out | tr -s ' ') | $st load=$(cut -d' ' -f1 /proc/loadavg) rc=$rc"
}
for rd in 1 2 3; do
  if [ $((rd % 2)) -eq 1 ]; then cado $rd; ours $rd; else ours $rd; cado $rd; fi
done
rm -f $W/smi.tmp
echo "done: $fails failed"
[ $fails -eq 0 ]
