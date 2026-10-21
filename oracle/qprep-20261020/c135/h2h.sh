#!/bin/sh
# c135 against cado-gpu on one of its own workunits: q in [7000000,7100000),
# side-1 special-q, I=13 J=4096, CADO's bounds and lambdas (c135.job). CADO
# sieved that range in 13.99 s (0.7 s of it startup) for 462,878 relations
# from 6,456 special-q (cado-wu.txt, from c135.log). Each row: our wall for
# the whole run including startup and the final flush (/usr/bin/time), the
# band's own per-q figures, and relations. BASE is the finding-120 snapshot,
# the others the q-prep snapshot under each --q-prep policy. Card logged.
R=/home/kylea/code/cuda-sieve; W=$R/work/c135
B=$R/work/tdwait/exp/bench/bench; X=$R/work/qprep/exp/bench/bench
A="--pipeline --cofactor --poly c135.job --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000:7100000"
cd $W || exit 1
fails=0
run() {  # arm round
  arm=$1; rd=$2
  case $arm in base) bin=$B; extra= ;; *) bin=$X; extra="--q-prep $arm" ;; esac
  sleep 2; idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  o=$W/h2h.$arm.$rd
  /usr/bin/time -f %e -o $o.time $bin $A $extra --relations $o.rel > $o.out 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  nq=$(grep -h 'issued ahead' $o.out | awk '{print $(NF-1)}')
  printf "%-4s r%s total %6ss | %5s q | per q %s ms, COMPLETE %s ms | %s rels (%s/q) | load=%s idle=%s%% rc=%s\n" \
    $arm $rd "$(cat $o.time)" "$((nq + 1))" \
    "$(grep -h '^  wall clock per q  ' $o.out | awk '{print $(NF-1)}')" \
    "$(grep -h 'wall clock per q, COMPLETE' $o.out | awk '{print $(NF-1)}')" \
    "$(wc -l < $o.rel)" "$(grep -h 'ALL RELATIONS/q' $o.out | awk '{print $NF}')" \
    "$(cut -d' ' -f1 /proc/loadavg)" "$idle" "$rc"
}
for rd in ${ROUNDS:-1 2}; do
  case $rd in 1) arms="base join auto td";; 2) arms="td auto join base";; 3) arms="join base td auto";; *) arms="auto td base join";; esac
  for arm in $arms; do run $arm $rd; done
done
for f in h2h.*.rel; do cmp -s h2h.base.1.rel $f || { echo "$f DIFFERS from h2h.base.1.rel"; fails=$((fails+1)); }; done
echo "done: $fails failed"
[ $fails -eq 0 ]
