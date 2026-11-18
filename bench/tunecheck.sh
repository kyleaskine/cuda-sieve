#!/bin/sh
# tunecheck.sh -- tune.sh's saving rules, card-free, against a fake bench.
#
#   sh tunecheck.sh            (from bench/; make tunecheck)
#
# The fake bench (written below into a temp dir) prints what the real one
# prints for tune.sh: the grid and plan lines, bench's tuned:/tuned-run:/
# tuned-plan: lines, and a wall clock per q that is a product of per-flag
# factors (FAKE_F, and FAKE_F3 in the confirmation stage), with 0.1% noise.
# Its "judge" is a stand-in for bench's: the last entry line at logI 16 in the
# file it is pointed at is the incumbent (policy 1 applies, any other is
# declined), minus --no-fill-merge (this I16 plan has no merge), so a test can
# see that the saved arm is timed from what bench would apply, not from the
# file. It never touches a card; nvidia-smi is faked empty and TUNE_BUSY is
# cleared. FAKE_OLD plays a bench from before saved tuning (no --no-tuned in
# its --help, no tuned lines, and --tuned/--no-tuned refused); without
# --pipeline it refuses them as bench does; FAKE_SFA_OFF has
# --slab-fill-ahead turn itself off (the run is then the defaults' speed).
# Exits nonzero if any check fails.
set -u
B=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/tunecheck.XXXXXX") || exit 2
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/home"
fails=0; checks=0
ok()   { checks=$((checks + 1)); echo "PASS $*"; }
bad()  { checks=$((checks + 1)); fails=$((fails + 1)); echo "FAIL $*"; }
check() { if eval "$1"; then ok "$2"; else bad "$2 [$1]"; fi; }

cat > "$T/bin/nvidia-smi" <<'EOF'
#!/bin/sh
exit 0
EOF
cat > "$T/bin/bench" <<'EOF'
#!/bin/sh
case " $* " in *" --help "*) [ -n "${FAKE_OLD:-}" ] || echo "  --no-tuned       apply none of it"; exit 1 ;; esac
rel=; log=; tp=; nt=; flags=; over=; pipe=; tg=
while [ $# -gt 0 ]; do
  case $1 in
    --relations) rel=$2; shift ;;
    --log) log=$2; shift ;;
    --pipeline) pipe=1 ;;
    --tuned|--no-tuned) [ -n "${FAKE_OLD:-}" ] && { echo "unknown option $1"; exit 1; }
      tg=1; if [ "$1" = --tuned ]; then tp=$2; shift; else nt=1; fi ;;
    --fill-blocks|--ss-block-cut|--ss-warp-cut|--fill-gate-shift|--slab-fill-blocks) flags="$flags $1=$2"; shift ;;
    --no-fill-merge|--slab-fill-ahead|--no-slab-fill-ahead) flags="$flags $1" ;;
    --sieve-skip) over="$over --sieve-skip"; shift ;;
    --poly|--logI|--qrange|--nq) shift ;;
  esac
  shift
done
case " ${FAKE_FAIL:-} " in *" $(basename "$rel" .rel) "*) echo "fake failure"; exit 1 ;; esac
[ -n "$tg" ] && [ -z "$pipe" ] && { echo "--tuned and --no-tuned apply to --pipeline only"; exit 2; }
[ -n "$tp" ] || tp=$FAKE_HOME_TUNED
st=nofile; would=
if [ -f "$tp" ]; then
  if [ "$(sed -n '/[^ ]/{p;q}' "$tp")" != "# cuda-sieve tuned v1" ]; then st=badfile
  else
    e=$(awk -F'|' 'NF >= 13 && $1 !~ /^#/ && $4 == 16 { l = $0 } END { print l }' "$tp")
    if [ -z "$e" ]; then st=none
    elif [ "$(echo "$e" | awk -F'|' '{ print $11 }')" != 1 ]; then st=declined
    else st=apply; would=$(echo "$e" | awk -F'|' '{ print $12 }' | sed 's/--no-fill-merge//; s/  */ /g; s/^ //; s/ $//'); fi
  fi
fi
printf '1,2:3:4\n5,6:7:8\n' > "$rel"
printf '# commit fake0123-dirty\n# device fake 3060\n' > "$log"
echo "args:$flags"
echo "grid: 28 SMs x 6 = 168 blocks (dev 0: FAKE RTX 3060, 2 MB L2)"
if [ -z "${FAKE_OLD:-}" ] && [ -n "$pipe" ]; then
  echo "tuned: fake verdict $st ($would)"
  echo "tuned-run: FAKE RTX 3060|8.6|28|16|32768|14|1|300000000|200000000|1|2147483647|16|$st|$would|$(echo $over)|$tp"
  echo "tuned-plan: 4 slabs per q"
fi
case "$flags " in *" --slab-fill-ahead "*) if [ -n "${FAKE_SFA_OFF:-}" ]; then
  echo "  --slab-fill-ahead: off, because this plan has one slab per q"
  flags=$(echo $flags | tr ' ' '\n' | grep -v -e '^--slab-fill-ahead$' -e '^--slab-fill-blocks=' | tr '\n' ' ')
fi ;; esac
echo "  --q-pipeline: each slab's sieve under the previous slab's trial division"
echo "--- band of 10 q"
tbl=${FAKE_F:-}; case $rel in */s3.*) tbl=${FAKE_F3:-$tbl} ;; esac
f=1
for x in $flags; do
  for y in $tbl; do [ "${y%:*}" = "$x" ] && f=$(awk -v a=$f -v b=${y##*:} 'BEGIN { print a * b }'); done
done
awk -v f=$f -v s=$$ 'BEGIN { srand(s); printf "  wall clock per q, COMPLETE  %.2f ms\n", 700 * f * (1 + (rand() - 0.5) * 0.002) }'
EOF
chmod +x "$T/bin/bench" "$T/bin/nvidia-smi"

export PATH="$T/bin:$PATH" TUNE_BUSY= TUNE_POLL=0.02 FAKE_HOME_TUNED="$T/home/tuned.txt"
HDR='# cuda-sieve tuned v1'
OTHER='OTHER CARD|8.9|46|16|32768|14|4|1|300000000|200000000|1|--fill-blocks 9216|keep me'
entry() { echo "FAKE RTX 3060|8.6|28|16|32768|14|4|1|300000000|200000000|$1|$2|$3"; }
n=0
# tunerun EXPECTED-EXIT ARGS... : tune.sh with the fake, -r 3, into $T/r.$n
tunerun() {
  want=$1; shift; n=$((n + 1)); R=$T/r.$n
  sh "$B/tune.sh" -b "$T/bin/bench" -r 3 -o "$R" "$@" > "$R.txt" 2>&1; rc=$?
  [ "$rc" = "$want" ] || { bad "run $n exited $rc, want $want (see below)"; sed 's/^/    /' "$R.txt" | tail -15; }
}
JOB="--pipeline --cofactor --poly x.job --logI 16 --qrange 50000000: --nq 10"
W=$T/w.txt

echo "== 1. a fresh file: the winner is written, with the header"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912 B48=--ss-block-cut,48" -- $JOB
check '[ "$(head -1 "$W")" = "$HDR" ]' "the header line"
check 'grep -q "^FAKE RTX 3060|8.6|28|16|32768|14|4|1|300000000|200000000|1|--fill-blocks 294912|tune.sh [0-9-]* fake0123-dirty x.job q 50000000: nq 10 -2[0-9].[0-9]*% against the defaults$" "$W"' "the entry: key, nslab 4 from tuned-plan, policy, flags, provenance"
check '[ $(grep -c "^FAKE" "$W") = 1 ]' "one entry"
check '! ls "$T" | grep -q "w.txt.tune"' "no temp file left"
check 'grep -q "^== confirmation: default cand" "$R.txt"' "a confirmation stage ran"

echo "== 2. a faster saved entry: kept, the file unchanged"
{ echo "$HDR"; entry 1 "--fill-blocks 589824 --no-fill-merge" "saved earlier"; } > "$W"; cp "$W" "$T/w.before"
FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=589824:0.75" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check 'cmp -s "$W" "$T/w.before"' "the file is unchanged"
check 'grep -q "unchanged" "$R.txt"' "the report says so"
check 'grep -q "^args: --fill-blocks=589824$" "$R"/s3.saved.1.out' "saved is timed from bench's would-apply part (no --no-fill-merge)"

echo "== 3. the candidate beats a slower saved entry: replaced, other cards kept"
{ echo "$HDR"; echo "$OTHER"; entry 1 "--fill-blocks 147456" "saved earlier"; } > "$W"
FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=147456:0.85" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check 'grep -q "|--fill-blocks 294912|" "$W" && ! grep -q "|--fill-blocks 147456|" "$W"' "the entry replaced"
check 'grep -qxF "$OTHER" "$W"' "another card's entry kept"

echo "== 4. nothing wins and the defaults clearly beat a stale saved entry: an empty entry"
{ echo "$HDR"; entry 1 "--fill-blocks 147456" "stale"; } > "$W"
FAKE_F="--fill-blocks=147456:1.05" tunerun 0 -w "$W" -a "F1152=--fill-blocks,1152" -- $JOB
check 'grep -q "^FAKE RTX 3060|.*|1||tune.sh .*the defaults beat the saved --fill-blocks 147456$" "$W"' "an empty entry replaces it"

echo "== 5. the candidate fades in confirmation, the defaults beat saved: empty, not the candidate"
{ echo "$HDR"; entry 1 "--fill-blocks 147456" "stale"; } > "$W"
FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=147456:1.05" FAKE_F3="--fill-blocks=294912:0.998 --fill-blocks=147456:1.05" \
  tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check 'grep -q "^FAKE RTX 3060|.*|1||" "$W" && ! grep -q "294912" "$W"' "the candidate is not written; the defaults are"

echo "== 6. the candidate fades and saved is within noise of the defaults: unchanged"
{ echo "$HDR"; entry 1 "--fill-blocks 147456" "close"; } > "$W"; cp "$W" "$T/w.before"
FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=147456:1.002" FAKE_F3="--fill-blocks=294912:0.998 --fill-blocks=147456:1.002" \
  tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check 'cmp -s "$W" "$T/w.before"' "the file is unchanged"

echo "== 7. a declined entry (another policy) is no incumbent, and is replaced"
{ echo "$HDR"; entry 0 "--fill-blocks 589824" "old policy"; } > "$W"
FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=589824:0.5" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check '! ls "$R" | grep -q "^s3.saved"' "no saved arm timed"
check 'grep -q "|1|--fill-blocks 294912|" "$W" && ! grep -q "|0|--fill-blocks 589824|" "$W"' "replaced"

echo "== 8. -w FILE: the incumbent comes from FILE; the home file is neither timed nor touched"
{ echo "$HDR"; entry 1 "--fill-blocks 589824" "home"; } > "$FAKE_HOME_TUNED"; cp "$FAKE_HOME_TUNED" "$T/home.before"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=589824:0.5" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check '! ls "$R" | grep -q "^s3.saved"' "the home entry is not timed"
check 'grep -q "|--fill-blocks 294912|" "$W"' "written to FILE"
check 'cmp -s "$FAKE_HOME_TUNED" "$T/home.before"' "the home file is untouched"
check 'grep -q "fake verdict nofile" "$R"/s1.default.1.out' "bench was pointed at FILE from the first run"

echo "== 9. refusals: tune.sh's own flags and the knobs in the job line"
for a in "--tuned $W" "--no-tuned" "--fill-blocks 9216" "--slab-fill-ahead" "--no-fill-merge"; do
  tunerun 2 -w "$W" -- $JOB $a
done
ok "--tuned, --no-tuned and knobs in the job line refused (exit 2; failures above if not)"

echo "== 10. without -w: the path bench names; an -a experiment does not write"
rm -f "$FAKE_HOME_TUNED"
FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -a "F294912=--fill-blocks,294912" -- $JOB
check '[ ! -e "$FAKE_HOME_TUNED" ] && grep -q "not saving: an -a experiment" "$R.txt"' "an -a experiment writes nothing"

echo "== 11. -n writes nothing"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -n -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check '[ ! -e "$W" ] && grep -q "not saving: -n" "$R.txt"' "-n"

echo "== 12. a failed run: nothing written"
rm -f "$W"
FAKE_FAIL="s1.F294912.2" FAKE_F="--fill-blocks=294912:0.78" tunerun 1 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check '[ ! -e "$W" ] && grep -q "not saving: runs failed" "$R.txt"' "nothing written"

echo "== 13. a job line outside the plain base: nothing written"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB --sieve-skip 1
check '[ ! -e "$W" ] && grep -q "not saving: the job line sets --sieve-skip" "$R.txt"' "overrides block saving"

echo "== 14. nothing wins, no saved entry: an empty entry records the tune, no confirmation"
rm -f "$W"
tunerun 0 -w "$W" -a "F1152=--fill-blocks,1152" -- $JOB
check 'grep -q "^FAKE RTX 3060|.*|1||tune.sh [0-9-]* fake0123-dirty x.job q 50000000: nq 10, the defaults won$" "$W" && ! ls "$R" | grep -q "^s3"' "an empty entry, no stage 3"

echo "== 15. a confirmation cut short by a failure: unchanged"
{ echo "$HDR"; entry 1 "--fill-blocks 147456" "kept"; } > "$W"; cp "$W" "$T/w.before"
FAKE_FAIL="s3.cand.2" FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=147456:0.9" tunerun 1 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check 'cmp -s "$W" "$T/w.before"' "the file is unchanged"

echo "== 16. a winning --slab-fill-blocks alone is saved with --slab-fill-ahead, as bench reads it"
rm -f "$W"
FAKE_F="--slab-fill-blocks=384:0.8" tunerun 0 -w "$W" -a "S384=--slab-fill-blocks,384" -- $JOB
check 'grep -q "^FAKE RTX 3060|.*|1|--slab-fill-ahead --slab-fill-blocks 384|" "$W"' "written as --slab-fill-ahead --slab-fill-blocks 384"

echo "== 17. arms whose entry bench could not read: refused"
tunerun 2 -w "$W" -a "F0=--fill-blocks,0" -- $JOB
check 'grep -q "automatic grid" "$R.txt"' "--fill-blocks 0"
tunerun 2 -w "$W" -a "FF=--fill-blocks,9216,--fill-blocks,18432" -- $JOB
check 'grep -q "gives --fill-blocks twice" "$R.txt"' "a flag twice in one arm"

echo "== 18. a saved --slab-fill-ahead that turns itself off is still timed, and replaced"
{ echo "$HDR"; entry 1 "--slab-fill-ahead --slab-fill-blocks 384" "sfa"; } > "$W"
FAKE_SFA_OFF=1 FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check '[ $(ls "$R" | grep -c "^s3.saved.[0-9].out$") = 3 ] && grep -q "|--fill-blocks 294912|" "$W"' "saved timed in every round; the candidate replaces it"

echo "== 19. a bench from before saved tuning: no --tuned or --no-tuned, nothing written"
rm -f "$W"
FAKE_OLD=1 FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check '[ ! -e "$W" ] && grep -q "not saving: this bench predates" "$R.txt" && grep -q "recommended for this card and job: --fill-blocks 294912" "$R.txt"' "it still tunes and reports"

echo "== 20. a job line without --pipeline: no --tuned or --no-tuned, nothing written"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912" -- --poly x.job --logI 16 --qrange 50000000: --nq 10
check '[ ! -e "$W" ] && grep -q "not saving: the job line has no --pipeline" "$R.txt" && grep -q "recommended for this card and job: --fill-blocks 294912" "$R.txt"' "the harness tunes and reports"

echo "tunecheck: $((checks - fails)) of $checks checks passed"
[ $fails -eq 0 ]
