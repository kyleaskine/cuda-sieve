#!/bin/sh
# tunecheck.sh -- tune.sh's search (the ladders, stage 2, -q and -t, arms
# out early and -f) and its saving rules, card-free, against a fake bench.
#
#   sh tunecheck.sh            (from bench/; make tunecheck)
#
# The fake bench (written below into a temp dir) prints what the real one
# prints for tune.sh: the grid and plan lines, bench's tuned:/tuned-run:/
# tuned-plan: lines, and a wall clock per q that is a product of per-flag
# factors (FAKE_F, and FAKE_F3 in the confirmation stage), with 0.1% noise
# (none under FAKE_NONOISE, so an arm with no factor ties the defaults).
# Its "judge" is a stand-in for bench's: the last entry line at logI 16 in the
# file it is pointed at is the incumbent (policy 1 applies, any other is
# declined), minus --no-fill-merge (this I16 plan has no merge), so a test can
# see that the saved arm is timed from what bench would apply, not from the
# file. It never touches a card; nvidia-smi is faked empty and TUNE_BUSY is
# cleared. FAKE_OLD plays a bench from before saved tuning (no --no-tuned in
# its --help, no tuned lines, and --tuned/--no-tuned refused); without
# --pipeline it refuses them as bench does; FAKE_SFA_OFF has
# --slab-fill-ahead turn itself off (the run is then the defaults' speed), and
# FAKE_SFA_ON has it on by default, as on the 5070 (else it is off unless an
# arm turns it on, and --slab-fill-blocks without it is refused, as bench
# refuses it).
# FAKE_C gives a factor for two flags together, for knobs that interact. It
# refuses what bench refuses of the ladders' values (bench.h's bounds, a warp
# cut not above the block cut), so a ladder step past a bound fails the run.
# Under TUNE_FAKE_CLOCK (a file tune.sh reads as its clock) each run adds
# FAKE_SECS to it, FAKE_SECS3 in the confirmation stage, for the -t budget.
# TUNECHECK_KEEP=1 keeps the temp dir with every run's report, and prints it.
# Exits nonzero if any check fails.
set -u
B=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/tunecheck.XXXXXX") || exit 2
if [ -n "${TUNECHECK_KEEP:-}" ]; then echo "tunecheck: keeping $T"; else trap 'rm -rf "$T"' EXIT; fi
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
# bench's own refusals of the knobs' values (bench.h's bounds, the pair rule)
bc=36; wc=256; fb=
for x in $flags; do case $x in --ss-block-cut=*) bc=${x#*=} ;; --ss-warp-cut=*) wc=${x#*=} ;; --fill-blocks=*) fb=${x#*=} ;; esac; done
[ "$bc" -ge 33 ] && [ "$bc" -le 128 ] || { echo "--ss-block-cut: $bc is outside 33..128"; exit 1; }
[ "$wc" -ge 34 ] && [ "$wc" -le 32768 ] || { echo "--ss-warp-cut: $wc is outside 34..32768"; exit 1; }
[ "$wc" -gt "$bc" ] || { echo "the warp cut ($wc) must be above the block cut ($bc)"; exit 1; }
[ -z "$fb" ] || { [ "$fb" -ge 1 ] && [ "$fb" -le 1048576 ]; } || { echo "--fill-blocks: $fb is outside 1..1048576"; exit 1; }
case " $flags " in *" --slab-fill-blocks="*)
  case " $flags " in
    *" --no-slab-fill-ahead "*) echo "--slab-fill-blocks sets --slab-fill-ahead's grid; add --slab-fill-ahead"; exit 1 ;;
    *" --slab-fill-ahead "*) ;;
    *) [ -n "${FAKE_SFA_ON:-}" ] || { echo "--slab-fill-blocks sets --slab-fill-ahead's grid, and it is off by default on this card; add --slab-fill-ahead"; exit 1; } ;;
  esac ;;
esac
# the pretend clock tune.sh reads under TUNE_FAKE_CLOCK: FAKE_SECS a run
# (FAKE_SECS3 in the confirmation stage)
if [ -n "${TUNE_FAKE_CLOCK:-}" ]; then
  secs=${FAKE_SECS:-30}; case $rel in */s3.*) secs=${FAKE_SECS3:-$secs} ;; esac
  echo $(( $(cat "$TUNE_FAKE_CLOCK") + secs )) > "$TUNE_FAKE_CLOCK"
fi
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
if [ -n "${FAKE_SFA_ON:-}" ]; then case " $flags " in *" --no-slab-fill-ahead "*) ;; *)
  g=512; for x in $flags; do case $x in --slab-fill-blocks=*) g=${x#*=} ;; esac; done
  echo "  --slab-fill-ahead: each later slab's side-1 fill beside the previous slab's side-1 apply, $g x 32 at the greatest priority [--slab-fill-blocks]" ;; esac
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
# FAKE_C: "A+B:factor", a factor for two flags together (knobs that interact)
for y in ${FAKE_C:-}; do
  p=${y%:*}
  case " $flags " in *" ${p%%+*} "*) case " $flags " in *" ${p#*+} "*)
    f=$(awk -v a=$f -v b=${y##*:} 'BEGIN { print a * b }') ;; esac ;; esac
done
awk -v f=$f -v s=$$ -v nz="${FAKE_NONOISE:-}" 'BEGIN { srand(s); printf "  wall clock per q, COMPLETE  %.2f ms\n", 700 * f * (1 + (nz != "" ? 0 : (rand() - 0.5) * 0.002)) }'
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
# every grid arm that the fake card times, 1% slower than the defaults: out
# after round 1 whatever the noise (a scenario's own factors come after)
GRIDLOSE="--ss-block-cut=40:1.01 --ss-block-cut=48:1.01 --ss-block-cut=64:1.01 --ss-warp-cut=128:1.01
  --ss-warp-cut=384:1.01 --ss-warp-cut=512:1.01 --ss-warp-cut=1024:1.01 --fill-blocks=1152:1.01
  --fill-blocks=2304:1.01 --fill-blocks=9216:1.01 --fill-blocks=18432:1.01 --fill-gate-shift=1:1.01
  --fill-gate-shift=2:1.01 --fill-gate-shift=4:1.01 --slab-fill-blocks=256:1.01 --slab-fill-blocks=384:1.01"
# flags_in FILE: the flags field of the fake card's entry
flags_in() { grep '^FAKE' "$1" | cut -d'|' -f12; }
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
tunerun 2 -t "" -w "$W" -- $JOB
check 'grep -q "needs a whole number of minutes" "$R.txt"' "-t with an empty value refused, not taken as no budget"

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

echo "== 16. a winning --slab-fill-blocks alone (the slab fill ahead on by default) is saved with --slab-fill-ahead, as bench reads it"
rm -f "$W"
FAKE_SFA_ON=1 FAKE_F="--slab-fill-blocks=384:0.8" tunerun 0 -w "$W" -a "S384=--slab-fill-blocks,384" -- $JOB
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

echo "== 21. tune.sh's copies of bench.h's bounds and defaults match bench.h"
hdef() { sed -n "s/^#define $1 *\([0-9][0-9]*\)u* *\$/\1/p" "$B/bench.h" | head -1; }
tdef() { sed -n "s/^$1=\([0-9][0-9]*\).*/\1/p" "$B/tune.sh" | head -1; }
for p in SS_BLOCK_CUT_MIN:SS_BLOCK_MIN SS_BLOCK_CUT_MAX:SS_BLOCK_MAX SS_WARP_CUT_MAX:SS_WARP_MAX \
         SS_BLOCK_CUT_DEFAULT:SS_BLOCK_DEF SS_WARP_CUT_DEFAULT:SS_WARP_DEF FILL_BLOCKS_DEFAULT:FILL_DEF \
         SLAB_FILL_BLOCKS_DEFAULT:SFA_BLOCKS_DEF SLAB_FILL_DEFAULT_SMS:SFA_DEF_SMS; do
  h=$(hdef ${p%:*}); t=$(tdef ${p#*:})
  check '[ -n "$h" ] && [ "$h" = "$t" ]' "${p#*:} ($t) is bench.h's ${p%:*} ($h)"
done
h=$(sed -n 's/^#define BENCH_BLOCKS_MAX *(1 << \([0-9][0-9]*\)).*/\1/p' "$B/bench.h"); t=$(tdef BLOCKS_MAX)
check '[ -n "$h" ] && [ $((1 << h)) = "$t" ]' "BLOCKS_MAX ($t) is bench.h's BENCH_BLOCKS_MAX (1 << $h)"
h=$(sed -n 's/.*--fill-blocks N .*0 = auto (\([0-9][0-9]*\),.*/\1/p' "$B/bench_main.cu" | head -1); t=$(tdef FILL_DEF)
check '[ -n "$h" ] && [ "$h" = "$t" ]' "FILL_DEF ($t) is bench --help's automatic --fill-blocks ($h)"

echo "== 22. the --fill-blocks ladder steps past the grid's top, two at a time, and stops on a clear loss"
rm -f "$W"
FAKE_F="--fill-blocks=36864:0.92 --fill-blocks=73728:0.88 --fill-blocks=147456:0.84 --fill-blocks=294912:0.80 --fill-blocks=589824:0.76 --fill-blocks=1048576:0.84" \
  tunerun 0 -w "$W" -- $JOB
check '[ $(grep -c "^== ladder pass" "$R.txt") = 3 ]' "three ladder passes"
check 'grep -q "F1048576 is clearly slower than F589824" "$R.txt"' "it stops on the loss at 2^20, the clamped last step"
check 'grep -q "|--fill-blocks 589824|" "$W"' "589824 written: the grid's top, 36864, times 16"
check '! ls "$R" | grep -q "F2097152"' "nothing past bench's bound, 2^20"
check '! ls "$R" | grep -q "SFA512" && ls "$R" | grep -q "^s1.SFA384.1.out$"' "the slab fill ahead's grid stops at 16 resident blocks per SM (448 on 28 SMs)"
f22=$(flags_in "$W"); rm -f "$W"
FAKE_F="--fill-blocks=36864:0.92 --fill-blocks=73728:0.88 --fill-blocks=147456:0.84 --fill-blocks=294912:0.80 --fill-blocks=589824:0.76 --fill-blocks=1048576:0.84" \
  tunerun 0 -f -w "$W" -- $JOB
check '[ -n "$f22" ] && [ "$(flags_in "$W")" = "$f22" ]' "-f, every arm in every round, writes the same ($f22)"

echo "== 23. the --ss-warp-cut ladder runs to its bound, 32768, and no further"
rm -f "$W"
FAKE_F="--ss-warp-cut=1024:0.96 --ss-warp-cut=2048:0.94 --ss-warp-cut=4096:0.92 --ss-warp-cut=8192:0.90 --ss-warp-cut=16384:0.88 --ss-warp-cut=32768:0.86" \
  tunerun 0 -w "$W" -- $JOB
check 'grep -q "W32768 is the ladder.s bound" "$R.txt"' "it stops at the bound"
check 'grep -q "|--ss-warp-cut 32768|" "$W"' "32768 written"

echo "== 24. --ss-block-cut steps up past 64; --ss-warp-cut steps down to 37, not to the block cut; a pair bench refuses is not timed"
rm -f "$W"
FAKE_F="--ss-block-cut=64:0.90 --ss-block-cut=96:0.95 --ss-block-cut=128:0.97 --ss-warp-cut=128:0.97 --ss-warp-cut=64:0.95 --ss-warp-cut=37:0.92" \
  tunerun 0 -w "$W" -- $JOB
check 'grep -q "B128 is clearly slower than B64" "$R.txt"' "the block cut's ladder stops on the loss at 128"
check 'grep -q "W37 is the ladder.s bound" "$R.txt" && ! ls "$R" | grep -qE "[.]W([0-9]|[12][0-9]|3[0-6])[.]"' "the warp cut's stops at 37, above the block cut 36"
check 'grep -q "not timing W37 on top of B64: bench would refuse the pair" "$R.txt"' "warp cut 37 is not timed on top of block cut 64"
check 'grep -q "|--ss-block-cut 64 --ss-warp-cut 128|" "$W"' "stage 2 adds the warp cut that fits beside it"

echo "== 25. stage 2, knobs that undo each other (finding 161): the best single setting stands"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78 --slab-fill-blocks=384:0.95" FAKE_C="--fill-blocks=294912+--slab-fill-ahead:1.15" \
  tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912 SFA384=--slab-fill-ahead,--slab-fill-blocks,384" -- $JOB
check 'grep -q "nothing beat F294912 clearly on top of it" "$R.txt"' "SFA384 on top of F294912 does not beat it"
check '[ -e "$R/s2.F294912+SFA384.1.out" ] && [ ! -e "$R/s2.F294912+SFA384.2.out" ] && [ ! -e "$R/s2.default.2.out" ]' "it is out after round 1 (slower than the base, though faster than the defaults), and stage 2 stops"
check 'grep -q "|--fill-blocks 294912|" "$W"' "the grid alone is written"
check '[ $(grep -c "^== ladder pass" "$R.txt") = 0 ]' "no ladder past -a arms"

echo "== 26. stage 2, knobs that add: each is timed on top of the best so far, and kept"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check 'ls "$R" | grep -q "^s2a2.F294912+W1024+G2.1.out$"' "F294912+W1024+G2 timed against F294912+W1024"
check 'grep -q "|--fill-blocks 294912 --ss-warp-cut 1024 --fill-gate-shift 2|" "$W"' "all three written"
f26=$(flags_in "$W"); rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 0 -f -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check '[ "$(flags_in "$W")" = "$f26" ]' "-f writes the same"
check '! grep -q "not confirmed" "$R.txt" && grep -q "^== confirmation" "$R.txt"' "a tune that saves confirms the combination, and does not call it unconfirmed"

echo "== 27. stage 2: an addition that loses on top of the best so far is left out"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" FAKE_C="--ss-warp-cut=1024+--fill-gate-shift=2:1.03" \
  tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check 'grep -q "G2 did not beat F294912+W1024 clearly; left out" "$R.txt"' "G2 left out"
check '[ -e "$R/s2a2.F294912+W1024+G2.1.out" ] && [ ! -e "$R/s2a2.F294912+W1024+G2.2.out" ]' "it is out after round 1 against F294912+W1024 (though faster than the defaults)"
check 'grep -q "|--fill-blocks 294912 --ss-warp-cut 1024|" "$W"' "the pair is written"

echo "== 28. -q: no stage 2; the best single setting is the recommendation"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 0 -q -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check '! ls "$R" | grep -q "^s2" && grep -q "|--fill-blocks 294912|" "$W"' "no stage 2; F294912 written"

# The -t cases run on the fake's clock, 30 s a run: stage 1 of 4 arms x 3
# rounds is 6 min, and the confirmation (defaults, cand) holds 3 min back.
echo "== 29. -t: a stage-2 knob that would eat into the confirmation's time is left out, and named"
rm -f "$W"; echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 0 -t 14 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check 'grep -q "stage 2: --fill-gate-shift not timed on top of F294912, for the -t budget" "$R.txt"' "G2's knob is left out of stage 2 (smaller gain)"
check 'grep -q "not searched, for the -t budget: --fill-gate-shift$" "$R.txt"' "the result names it"
check '! ls "$R" | grep -q "^s2[.]F294912+G2" && grep -q "|--fill-blocks 294912 --ss-warp-cut 1024|" "$W"' "W1024 is still timed on top and written"
check '[ "$(cat "$T/clock")" -le 840 ]' "the whole tune ends within 14 min of pretend time ($(cat "$T/clock") s)"

echo "== 30. -t too short for stage 1 and the confirmation: refused"
rm -f "$W"; echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_F="--fill-blocks=294912:0.78" \
  tunerun 2 -t 8 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check '[ ! -e "$W" ] && grep -q "is too short" "$R.txt"' "refused, nothing written"

echo "== 31. -t: a confirmation that runs out of time is cut short, and the file is unchanged"
{ echo "$HDR"; entry 1 "--fill-blocks 147456" "kept"; } > "$W"; cp "$W" "$T/w.before"; echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_SECS3=200 FAKE_F="--fill-blocks=294912:0.78 --fill-blocks=147456:0.9" \
  tunerun 4 -t 8 -w "$W" -a "F294912=--fill-blocks,294912" -- $JOB
check 'cmp -s "$W" "$T/w.before" && grep -q "budget ran out in the confirmation stage" "$R.txt"' "exit 4, the file unchanged"

echo "== 32. -t, with -f (every arm every round): a ladder pass that does not fit is left out, and named"
rm -f "$W"; echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_F="--fill-blocks=36864:0.9" tunerun 0 -f -t 31 -w "$W" -- $JOB
check 'grep -q -- "--fill-blocks past F36864: not timed, for the -t budget" "$R.txt" && [ $(grep -c "^== ladder pass" "$R.txt") = 0 ]' "no ladder pass"
check 'grep -q "|--fill-blocks 36864|" "$W"' "stage 1's winner is confirmed and written"

echo "== 32b. -t, the losing arms out after round 1: the time they no longer take lets the ladder pass fit"
rm -f "$W"; echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_F="$GRIDLOSE --fill-blocks=36864:0.9" tunerun 0 -t 31 -w "$W" -- $JOB
check '[ $(grep -c "^== ladder pass" "$R.txt") = 1 ] && ! grep -q -- "--fill-blocks past F36864: not timed" "$R.txt"' "the ladder pass runs"
check 'grep -q "|--fill-blocks 36864|" "$W" && [ "$(cat "$T/clock")" -le 1860 ]' "36864 written, within 31 min of pretend time ($(cat "$T/clock") s)"

echo "== 33. a flat ladder runs to its bound, and the step nearest the defaults within noise of the best wins"
rm -f "$W"
FAKE_F="--fill-blocks=36864:0.900 --fill-blocks=73728:0.897 --fill-blocks=147456:0.895 --fill-blocks=294912:0.896 --fill-blocks=589824:0.895 --fill-blocks=1048576:0.897" \
  tunerun 0 -w "$W" -- $JOB
check 'grep -q "F1048576 is the ladder.s bound" "$R.txt"' "no clear loss: it runs to 2^20"
check 'grep -q "|--fill-blocks 36864|" "$W"' "36864 written: the others gain at most 0.5% more"
rm -f "$W"
FAKE_F="--fill-blocks=36864:0.900 --fill-blocks=73728:0.897 --fill-blocks=147456:0.895 --fill-blocks=294912:0.896 --fill-blocks=589824:0.895 --fill-blocks=1048576:0.897" \
  tunerun 0 -f -w "$W" -- $JOB
check 'grep -q "|--fill-blocks 36864|" "$W"' "-f writes the same"

echo "== 34. out early: an arm slower than the defaults in a round is not timed again"
rm -f "$W"
FAKE_F="--fill-blocks=294912:0.78 --ss-block-cut=48:1.02" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912 B48=--ss-block-cut,48" -- $JOB
check '[ -e "$R/s1.B48.1.out" ] && [ ! -e "$R/s1.B48.2.out" ] && grep -q "B48 .* out after round 1: slower than the defaults in it" "$R.txt"' "B48 is out after round 1, and the report says so"
check '[ -e "$R/s1.F294912.3.out" ] && [ -e "$R/s1.default.3.out" ] && grep -q "stage 1 is at most" "$R.txt"' "the winner and the defaults run every round; stage 1's time is an upper bound"
check 'grep -q "|--fill-blocks 294912|" "$W"' "the winner is written"

echo "== 35. every arm out: the stage stops, and the defaults' empty entry is written"
rm -f "$W"
FAKE_F="--ss-block-cut=48:1.02 --ss-warp-cut=1024:1.02" tunerun 0 -w "$W" -a "B48=--ss-block-cut,48 W1024=--ss-warp-cut,1024" -- $JOB
check '[ ! -e "$R/s1.default.2.out" ] && grep -q "every arm is out after round 1" "$R.txt" && grep -q "W1024 .* out after round 1" "$R.txt"' "stage 1 stops after round 1, and still reports each arm"
check 'grep -q "|1||tune.sh .*, the defaults won$" "$W" && ! ls "$R" | grep -q "^s3"' "an empty entry, no confirmation"

echo "== 36. -f: every arm in every round"
rm -f "$W"
FAKE_F="--ss-block-cut=48:1.02 --ss-warp-cut=1024:1.02" tunerun 0 -f -w "$W" -a "B48=--ss-block-cut,48 W1024=--ss-warp-cut,1024" -- $JOB
check '[ -e "$R/s1.B48.3.out" ] && [ -e "$R/s1.default.3.out" ] && ! grep -q "out after" "$R.txt" && grep -q "B48 .* faster in 0 of 3 rounds" "$R.txt" && ! grep -q "at most" "$R.txt"' "B48 is timed in all 3 rounds, with its full median"
check 'grep -q "|1||tune.sh .*, the defaults won$" "$W"' "the same empty entry"

echo "== 37. a ladder pass times every arm in every round: a step slower than the defaults can still stop it"
rm -f "$W"
FAKE_F="$GRIDLOSE --fill-blocks=36864:0.92 --fill-blocks=73728:0.95 --fill-blocks=147456:1.05" tunerun 0 -w "$W" -- $JOB
check '[ -e "$R/sx1.F147456.3.out" ] && grep -q "F147456 is clearly slower than F36864" "$R.txt" && [ $(grep -c "^== ladder pass" "$R.txt") = 1 ]' "F147456 runs all 3 rounds and stops the ladder"
check 'grep -q "|--fill-blocks 36864|" "$W"' "36864 written"

echo "== 38. a stage with nothing left to time: it stops, with and without -f, and the report has no shell error"
for fl in "" -f; do
  rm -f "$W"
  FAKE_SFA_OFF=1 tunerun 0 $fl -w "$W" -a "SFA384=--slab-fill-ahead,--slab-fill-blocks,384" -- $JOB
  check '! grep -qiE "cannot open|no such file" "$R.txt" && grep -q "|1||tune.sh .*, the defaults won$" "$W"' "${fl:-no -f}: the defaults' empty entry, no error"
done
check 'grep -q "no arm is left to time after round 1" "$T/r.$((n - 1)).txt"' "without -f the stage stops after round 1"

echo "== 39. a tie with the defaults is not a loss: the arm stays in"
rm -f "$W"
FAKE_NONOISE=1 FAKE_F="--fill-blocks=294912:0.78" tunerun 0 -w "$W" -a "F294912=--fill-blocks,294912 B48=--ss-block-cut,48" -- $JOB
check '[ -e "$R/s1.B48.3.out" ] && grep -q "B48 .* faster in 0 of 3 rounds, 3 tied" "$R.txt"' "B48, tied in every round, is timed in all 3"

echo "== 40. a bare --slab-fill-blocks arm and --no-slab-fill-ahead are one knob: stage 2 never pairs them (bench refuses the pair)"
rm -f "$W"
FAKE_SFA_ON=1 FAKE_F="--slab-fill-blocks=384:0.8" tunerun 0 -w "$W" -a "S384=--slab-fill-blocks,384 SFA0=--no-slab-fill-ahead" -- $JOB
check '[ -e "$R/s1.SFA0.1.out" ] && ! ls "$R" | grep -q "^s2[.].*[.]out$"' "SFA0 is timed in stage 1, and not on top of S384"
check 'grep -q "^FAKE RTX 3060|.*|1|--slab-fill-ahead --slab-fill-blocks 384|" "$W"' "S384 written"

echo "== 41. a ladder pass with a failed run: the search stops there"
rm -f "$W"
FAKE_FAIL="sx1.F73728.2" FAKE_F="--fill-blocks=36864:0.92 --fill-blocks=73728:0.88 --fill-blocks=147456:0.84 --fill-blocks=294912:0.80" \
  tunerun 1 -w "$W" -- $JOB
check '[ $(grep -c "^== ladder pass" "$R.txt") = 1 ] && grep -q "a run failed or differed in this pass; the search stops here" "$R.txt"' "one ladder pass, and it says why it stops"
check '! ls "$R" | grep -qE "^s(x2|2)[.].*[.]out$" && [ ! -e "$W" ]' "no second pass, no stage 2, nothing written"

echo "== 42. stage 2 or a step after it with a failed run: no further step"
rm -f "$W"
FAKE_FAIL="s2.F294912+B48.1" FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 1 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2 B48=--ss-block-cut,48" -- $JOB
check 'grep -q "a run failed or differed in stage 2; the search stops here" "$R.txt" && ! ls "$R" | grep -q "^s2a"' "a failure in stage 2: no step after it"
FAKE_FAIL="s2a2.F294912+W1024+G2.1" FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98 --ss-block-cut=64:0.985" \
  tunerun 1 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2 B64=--ss-block-cut,64" -- $JOB
check 'ls "$R" | grep -q "^s2a2[.]" && ! ls "$R" | grep -q "^s2a3[.]" && grep -q "a run failed or differed; the search stops here" "$R.txt"' "a failure in step 2.2: no step 2.3"
check '[ ! -e "$W" ]' "nothing written"

echo "== 43. -t on a tune that cannot save (-n): no time is held back for a confirmation it never runs"
rm -f "$W"; echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 0 -n -t 7 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check '! grep -q "is too short" "$R.txt" && grep -q "nothing held back" "$R.txt" && [ "$(cat "$T/clock")" -le 420 ]' "-t 7 is not refused (stage 1 is 6 min), and it ends within 7 ($(cat "$T/clock") s)"
echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 0 -n -t 11 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check '[ -e "$R/s2.F294912+W1024.1.out" ] && [ "$(cat "$T/clock")" -le 660 ]' "-t 11: W1024 is timed on top of F294912 in the time a confirmation would have had ($(cat "$T/clock") s)"
check 'grep -q "recommended for this card and job: --fill-blocks 294912 --ss-warp-cut 1024" "$R.txt" && grep -q "not confirmed against the defaults as a whole" "$R.txt" && [ ! -e "$W" ]' "the combination is recommended, marked unconfirmed, and not written"

echo "== 44. stage 2 stops after round 1 with the defaults' run failed: no awk error"
rm -f "$W"
FAKE_FAIL="s2.default.1" FAKE_F="--fill-blocks=294912:0.78 --slab-fill-blocks=384:0.95" FAKE_C="--fill-blocks=294912+--slab-fill-ahead:1.15" \
  tunerun 1 -w "$W" -a "F294912=--fill-blocks,294912 SFA384=--slab-fill-ahead,--slab-fill-blocks,384" -- $JOB
check '! ls "$R" | grep -q "^s2[.]default[.]2" && ! grep -qi "division" "$R.txt" && grep -q "too few default runs to judge" "$R.txt"' "the stage stops, and the report says it cannot judge"

echo "== 45. -t: a step the budget leaves out is named by its knob, as the others are"
rm -f "$W"; echo 0 > "$T/clock"
TUNE_FAKE_CLOCK="$T/clock" FAKE_F="--fill-blocks=294912:0.78 --ss-warp-cut=1024:0.96 --fill-gate-shift=2:0.98" \
  tunerun 0 -t 16 -w "$W" -a "F294912=--fill-blocks,294912 W1024=--ss-warp-cut,1024 G2=--fill-gate-shift,2" -- $JOB
check 'grep -q "not adding G2 to F294912+W1024, for the -t budget" "$R.txt" && grep -q "not searched, for the -t budget: --fill-gate-shift$" "$R.txt"' "G2 is named as --fill-gate-shift"
check 'grep -q "|--fill-blocks 294912 --ss-warp-cut 1024|" "$W" && [ "$(cat "$T/clock")" -le 960 ]' "the pair is confirmed and written within 16 min ($(cat "$T/clock") s)"

echo "== 46. the default busy pattern: the CUDA client sieving or benchmarking, not its tune subcommand or the CPU client"
BD=$(sed -n "s/^BUSY_DEFAULT='\(.*\)'\$/\1/p" "$B/tune.sh")
bm() { echo "$1" | grep -qE "$BD"; }
check '[ -n "$BD" ]' "tune.sh's BUSY_DEFAULT is found"
check 'bm "./ggnfs-sieve-client --server-url=http://h:8080 --engine=cuda --cuda-bench=/c/bench/bench --device=0"' "the CUDA client sieving matches"
check 'bm "/bin/bash ./run-cuda-client.sh"' "run-cuda-client.sh before its exec matches"
check 'bm "./ggnfs-sieve-client benchmark --engine=cuda --cuda-bench=/c/bench/bench"' "the GPU benchmark matches"
check '! bm "./ggnfs-sieve-client tune --server-url=http://h:8080 --token=x --cuda-bench=/c/bench/bench --device=0"' "the client's tune subcommand does not match"
check '! bm "./ggnfs-sieve-client --server-url=http://h:8080 --siever=/g/sievers --workers=12"' "the CPU client does not match"
check '! bm "./ggnfs-sieve-client benchmark --siever=/g/sievers --workers=12"' "the CPU benchmark does not match"

echo "tunecheck: $((checks - fails)) of $checks checks passed"
[ $fails -eq 0 ]
