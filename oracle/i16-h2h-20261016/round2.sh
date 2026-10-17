#!/bin/sh
# Second end-to-end round of h2h.sh's protocol (external clock, relations
# written), interleaved, with SM clock and board power sampled every 0.5 s.
# Run because h2h.sh's two pipelined cuda-sieve arms (231.4, 232.3 ms/q)
# measured ~5% slower than six later runs of the same binary.
R=/home/kylea/code/cuda-sieve; W=$R/work/cado-nfs-i16; P=$R/work/applyprof
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
smi_on() { nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 > $W/smi.tmp & SMI=$!; }
smi_off() { kill $SMI; awk -F', ' 'NR>10 {c+=$1; p+=$2; n++} END {printf "sm=%.0f MHz power=%.0f W", c/n, p/n}' $W/smi.tmp; }
for name in c1 q1 c2 q2; do
  smi_on
  case $name in
  c*) GLAS_SKIP_BOUND=0 /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S' -o $W/r2-$name.time \
        $L -poly $W/nfs.poly -q0 297000000 -q1 297006000 -A 31 -lim0 200000000 -lim1 300000000 \
        -lpb0 33 -lpb1 34 -mfb0 64 -mfb1 96 -fb1 $W/nfs.roots1.gz -out $W/r2-$name.rel.gz \
        -t 16 -stats-stderr > $W/r2-$name.out 2> $W/r2-$name.err
      res="$(grep -h '^# Total' $W/r2-$name.err | sed 's/^# //')" ;;
  q*) rm -f "${W:?}"/r2-$name.rel "${W:?}"/r2-$name.rel.part "${W:?}"/r2-$name.rel.part.ckpt "${W:?}"/r2-$name.rel.lock
      ( cd $R/bench && /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S' -o $W/r2-$name.time \
        ./bench --pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 \
          --qrange 297000000:297005999 --relations $W/r2-$name.rel > $W/r2-$name.out 2>&1 )
      res="rel=$(wc -l < $W/r2-$name.rel) $(grep -h 'wall clock per q, COMPLETE' $W/r2-$name.out | tr -s ' ')" ;;
  esac
  echo "$name: $res | $(cat $W/r2-$name.time) | $(smi_off) | load=$(cut -d' ' -f1 /proc/loadavg)"
done
rm -f $W/smi.tmp
