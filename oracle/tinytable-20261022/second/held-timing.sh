#!/bin/sh
# Holds timing.sh until the card is free: no msieve process (Kyle's GPU
# polynomial selection) and the card under 10% busy for six samples in a row,
# ten seconds apart. Then runs it.
W=/home/kylea/code/cuda-sieve/work/stageb2
echo "hold from $(date '+%F %T'): waiting for msieve to finish and an idle card"
quiet=0
while [ $quiet -lt 6 ]; do
  if ps -eo args | grep -v grep | grep -q 'msieve'; then
    quiet=0
  else
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "${u:-100}" -lt 10 ]; then quiet=$((quiet+1)); else quiet=0; fi
  fi
  [ $quiet -lt 6 ] && sleep 10
done
echo "hold released $(date '+%F %T'), load $(cut -d' ' -f1 /proc/loadavg)"
$W/timing.sh
