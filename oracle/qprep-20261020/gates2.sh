#!/bin/sh
# Finding 121: skipcheck and degradecheck on the builds they need. gates.sh
# ran both on the default build, where each prints SKIP and exits 0 -- a
# vacuous pass that gates.sh counted. Here degradecheck runs on the PIPE_K=8
# snapshot and skipcheck on a BN_LIMBS=4 one, each as built (--q-prep auto)
# and through a wrapper that adds --q-prep td. A SKIP line, a FAIL line, no
# PASS line or a nonzero exit each fail the check; every check counts.
W=/home/kylea/code/cuda-sieve/work/qprep
fail=0
wrap() {  # source dir, wrapper dir, gate script
  mkdir -p $W/$2/bench
  ln -sfn $W/$1/oracle $W/$2/oracle
  [ -x $W/$1/bench/normscan ] && ln -sf $W/$1/bench/normscan $W/$2/bench/normscan
  cp $W/$1/bench/$3 $W/$2/bench/
  printf '#!/bin/sh\nexec %s "$@" --q-prep td\n' "$W/$1/bench/bench" > $W/$2/bench/bench
  chmod +x $W/$2/bench/bench
}
wrap pk8 pk8td degradecheck.sh
wrap bn4 bn4td skipcheck.sh
for c in "pk8 degradecheck" "pk8td degradecheck" "bn4 skipcheck" "bn4td skipcheck"; do
  set -- $c
  echo "== $2.sh ($1, sha256 $(sha256sum $W/${1%td}/bench/bench | cut -c1-16)$(case $1 in *td) echo ', --q-prep td';; esac))"
  (cd $W/$1/bench && sh ./$2.sh > $W/$2.$1.log 2>&1); rc=$?
  np=$(grep -c '^ *PASS' $W/$2.$1.log); nf=$(grep -c '^ *FAIL' $W/$2.$1.log); ns=$(grep -c 'SKIP:' $W/$2.$1.log)
  echo "   exit $rc, $np PASS, $nf FAIL, $ns SKIP | $(tail -1 $W/$2.$1.log)"
  [ $rc -eq 0 ] && [ $np -gt 0 ] && [ $nf -eq 0 ] && [ $ns -eq 0 ] || fail=$((fail+1))
done
echo "done: $fail failed"
[ "$fail" -eq 0 ]
