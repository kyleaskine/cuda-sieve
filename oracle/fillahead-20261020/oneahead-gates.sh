#!/bin/sh
# Finding 123 gates: --fill-ahead (opt-in) and the shared paths it touched
# (pipe_side_sieve_issue split into parts, the transform on its own stream,
# ev_up). Byte identity against the committed 2128138 build, default (fill
# ahead off) on every case, and with --fill-ahead on the cases that keep the
# pipeline (on slabbed plans it reports itself off). Every check counts.
R=/home/kylea/code/cuda-sieve; W=$R/work/fillahead; X=$W/exp/bench/bench
B=$R/work/abcheck/bench.2128138
cd $R/bench || exit 1
fail=0
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
echo "done: $fail failed"
[ "$fail" -eq 0 ]
