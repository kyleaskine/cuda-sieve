#!/bin/sh
# Finding 123 gates, two-ahead form: --fill-ahead (opt-in) and the shared
# paths it touched. Byte identity against the committed 2128138 build (BASE,
# outputs cached), or between arms of one binary; skipcheck and degradecheck
# on the builds they need, as built and through wrappers adding --fill-ahead
# (and --q-prep td), so the norm-width skip, the TD skip and the degradation
# ceiling are met with the q after next prepared. A SKIP line, a FAIL line, no
# PASS line or a nonzero exit fails a check; every check counts.
R=/home/kylea/code/cuda-sieve; W=$R/work/fillahead; X=$W/exp/bench/bench
B=$R/work/abcheck/bench.2128138
cd $R/bench || exit 1
fail=0
# Held until the card is free of the other session's ECM run (Kyle,
# 2026-10-20): no process of the gpuecm2 session left, and the card under 10%
# busy for six samples in a row, ten seconds apart.
echo "hold from $(date '+%T'): waiting for the gpuecm2 session's run and an idle card"
quiet=0
while [ $quiet -lt 6 ]; do
  if ps -eo args | grep -v grep | grep -q gpuecm2; then
    quiet=0
  else
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "${u:-100}" -lt 10 ]; then quiet=$((quiet+1)); else quiet=0; fi
  fi
  [ $quiet -lt 6 ] && sleep 10
done
echo "hold released $(date '+%T')"
echo "base $(sha256sum $B | cut -c1-16)  new $(sha256sum $X | cut -c1-16)  load=$(cut -d' ' -f1 /proc/loadavg)"
ab() {
  label=$1; shift
  echo "== abcheck: $label"
  rm -rf "$R/work/abcheck/run"
  env "$@" ./abcheck.sh "$B" "$X" > "$W/ab.tmp" 2>&1
  rc=$?
  sed 's/^/   /' "$W/ab.tmp"
  echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
}
NOQP='*.[sc][io]n[gc]*'
ab "default (fill ahead off), every case"
ab "--fill-ahead" SETS="cand cof stop" EXCEPT="$NOQP" NEWARGS="--fill-ahead"
ab "--fill-ahead --q-prep join" SETS="cand cof stop" EXCEPT="$NOQP" NEWARGS="--fill-ahead --q-prep join"
ab "--fill-ahead --q-prep td" SETS="cand cof stop" EXCEPT="$NOQP" NEWARGS="--fill-ahead --q-prep td"
ab "--fill-ahead, pair list from 1 (grows)" SETS="cand cof stop" QUICK=1 EXCEPT="$NOQP" \
   NEWARGS="--fill-ahead --resieve-scan-pairs 1"
ab "--fill-ahead, index cap 1 (every slab walks)" SETS="cand cof stop" QUICK=1 EXCEPT="$NOQP" \
   NEWARGS="--fill-ahead --resieve-scan-cap 1"
ab "--fill-ahead --no-resieve-scan" SETS="cand cof stop" QUICK=1 EXCEPT="$NOQP" \
   NEWARGS="--fill-ahead --no-resieve-scan"

echo "== growfail.sh: the pair list fails to grow (default path)"
G=$W/growfail/bench/bench; tail -1 $W/growfail.build.log | sed 's/^/   /'
W=$W/gfrun B=$B G=$G sh $R/work/tdwait/growfail.sh > $W/growfail.txt 2>&1; rc=$?
sed 's/^/   /' $W/growfail.txt
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))

echo "== tdskip (PIPE_K=8): truncated lists skip slabs; single-queued against pipelined arms"
K=$W/pk8/bench/bench; tail -1 $W/pk8.build.log | sed 's/^/   /'
for job in "c183s|--poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 120000000: --nq 150" \
           "c183|--poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 --qrange 120000000: --nq 150"; do
  jn=${job%%|*}; A="--pipeline --cofactor ${job#*|}"
  $K $A --no-q-pipeline --relations $W/tdskip.$jn.single.rel > $W/tdskip.$jn.single.out 2>&1; r0=$?
  for arm in "auto|" "fa|--fill-ahead" "fatd|--fill-ahead --q-prep td"; do
    an=${arm%%|*}; ex=${arm#*|}
    $K $A $ex --relations $W/tdskip.$jn.$an.rel > $W/tdskip.$jn.$an.out 2>&1; r=$?
    sk=$(grep -h 'slab(s) skipped or abandoned' $W/tdskip.$jn.$an.out | tr -s ' ' | cut -c1-40)
    ta=$(grep -h 'two ahead\|--fill-ahead: off' $W/tdskip.$jn.$an.out | tr -s ' ' | cut -c1-60)
    if [ $r0 -eq 0 ] && [ $r -eq 0 ] && [ -s $W/tdskip.$jn.$an.rel ] && cmp -s $W/tdskip.$jn.single.rel $W/tdskip.$jn.$an.rel; then
      echo "   $jn $an: IDENTICAL ($(wc -l < $W/tdskip.$jn.$an.rel)) |$sk |$ta"
    else
      echo "   $jn $an: FAIL: exits $r0/$r |$sk |$ta"; fail=$((fail+1))
    fi
    [ -n "$sk" ] || { echo "   $jn $an: FAIL: no skip line -- vacuous"; fail=$((fail+1)); }
  done
done

echo "== make check (snapshot)"
(cd "$W/exp/bench" && nice make GPU_ARCH=120 check > "$W/check.log" 2>&1); rc=$?
echo "   $(grep -c '^PASS' $W/check.log) PASS, $(grep -c '^FAIL' $W/check.log) FAIL; $(tail -1 $W/check.log)"
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))
echo "== fillcheck.sh (snapshot)"
(cd "$W/exp/bench" && ./fillcheck.sh > "$W/fillcheck.log" 2>&1); rc=$?
tail -2 "$W/fillcheck.log" | sed 's/^/   /'
echo "   exit $rc"; [ $rc -eq 0 ] || fail=$((fail+1))

wrap() {  # source dir, wrapper dir, gate script, extra args
  mkdir -p $W/$2/bench
  ln -sfn $W/$1/oracle $W/$2/oracle
  [ -x $W/$1/bench/normscan ] && ln -sf $W/$1/bench/normscan $W/$2/bench/normscan
  cp $W/$1/bench/$3 $W/$2/bench/
  printf '#!/bin/sh\nexec %s "$@" %s\n' "$W/$1/bench/bench" "$4" > $W/$2/bench/bench
  chmod +x $W/$2/bench/bench
}
wrap pk8 pk8fa degradecheck.sh "--fill-ahead"
wrap pk8 pk8fatd degradecheck.sh "--fill-ahead --q-prep td"
wrap bn4 bn4fa skipcheck.sh "--fill-ahead"
wrap bn4 bn4fatd skipcheck.sh "--fill-ahead --q-prep td"
for c in "pk8 degradecheck" "pk8fa degradecheck" "pk8fatd degradecheck" \
         "bn4 skipcheck" "bn4fa skipcheck" "bn4fatd skipcheck"; do
  set -- $c
  echo "== $2.sh ($1)"
  (cd $W/$1/bench && sh ./$2.sh > $W/$2.$1.log 2>&1); rc=$?
  np=$(grep -c '^ *PASS' $W/$2.$1.log); nf=$(grep -c '^ *FAIL' $W/$2.$1.log); ns=$(grep -c 'SKIP:' $W/$2.$1.log)
  echo "   exit $rc, $np PASS, $nf FAIL, $ns SKIP | $(tail -1 $W/$2.$1.log)"
  [ $rc -eq 0 ] && [ $np -gt 0 ] && [ $nf -eq 0 ] && [ $ns -eq 0 ] || fail=$((fail+1))
done
echo "done: $fail failed"
[ "$fail" -eq 0 ]
