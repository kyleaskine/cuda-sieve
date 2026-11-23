#!/bin/sh
# tune.sh -- time the speed-only knobs on THIS card for ONE job, and report
# which settings beat the defaults. Every knob it tries changes no output, and
# every run's relations are compared byte for byte with the defaults' run.
#
#   sh tune.sh [-b BENCH] [-r ROUNDS] [-o DIR] [-a "ARMS"] [-q] [-t MINUTES]
#              [-f] [-n] [-w FILE] -- BENCH-ARGS...
#
# BENCH-ARGS is a short band of the job to tune for, as bench takes it,
# without --relations, --log, --tuned or --no-tuned (tune.sh adds its own)
# and without the knobs it tunes (they belong in -a arms), for example
#   sh tune.sh -- --pipeline --cofactor --poly c148.job --logI 14 \
#       --qrange 10000000: --nq 2000
# Pick --nq so that one run takes 10-60 s. A much shorter run is mostly
# startup, and it is rounds, not run length, that beat the noise.
#
#   -b PATH   the bench binary                              [./bench]
#   -r N      rounds per stage                              [6]
#   -o DIR    where runs and the report go                  [tune.HOST.DATE]
#   -a ARMS   arms to time against the defaults instead of the grid below:
#             NAME=FLAGS, space-separated, commas for spaces in FLAGS,
#             e.g. "B48=--ss-block-cut,48 F9216=--fill-blocks,9216";
#             the ladders (below) do not step past -a arms
#   -q        quick: stage 1, the ladders and the confirmation; no stage 2
#   -t MIN    a time budget, in minutes. The confirmation stage's time is
#             held back from the start (none when the tune cannot save:
#             -n, an -a experiment without -w, or a job line bench will not
#             save for); ladder steps and stage-2 knobs that would eat into
#             it are left out, smallest stage-1 gain first, and the report
#             names them. Refused if stage 1 cannot fit.
#   -f        full: time every arm in every round, as for a table of every
#             arm's median (an arm out early, below, has none)
#   -n        report only; never write the saved-tuning file
#   -w FILE   the saved-tuning file to compare against and write [bench's
#             default, ~/.config/cuda-sieve/tuned.txt; bench --help, --tuned]
#
# SAVING (bench --help, --tuned; bench/tuned.h). bench applies the file's
# entry for this card at this logI to every --pipeline run, so a result
# saved here needs no flags afterwards. tune.sh times every arm with
# --no-tuned, against the built-in defaults, and saves only after a final
# confirmation stage that times the defaults, the entry already saved for
# this card and logI (as bench itself would apply it: "saved") and the
# recommendation ("cand") in the same rounds, judged by the adoption rule
# below:
#   - cand beats the defaults, and saved if there is one: cand is written;
#   - else, if the defaults clearly beat the saved entry: an empty entry
#     (the defaults won) replaces it;
#   - else, with no saved entry: an empty entry records that the card was
#     tuned and the defaults won;
#   - else the file is left as it is.
# saved is timed as bench runs it, so an entry whose --slab-fill-ahead turns
# itself off on this job is timed as the defaults it then is.
# Nothing is written when a run failed or its relations differed, when the
# -t budget ran out before the confirmation stage finished, when the
# card got busy, when the job line carries flags outside bench's plain base
# (its tuned-run line names them) or has no --pipeline (only pipeline runs
# read the file), with -n, or for an -a experiment without -w. The write
# replaces this card's entry at this logI and nothing else.
#
# The grid, one knob at a time away from the defaults (bench --help):
#   --ss-block-cut 40 48 64              default 36; 33 is the floor, and
#                                        33-35 split as 36 does
#   --ss-warp-cut 128 384 512 1024       default 256
#   --fill-blocks 1152 2304 9216 18432 36864   default 4608
#   --fill-gate-shift 1 2 4              default 3
#   --no-fill-merge                      default on with the row tier, at
#                                        one slab per q; not timed elsewhere
#   --slab-fill-ahead at --slab-fill-blocks 256 384 512 768 1024 1536 2048
#     3072, and --no-slab-fill-ahead     on by default (512 blocks) only on
#                                        sm_120 with 48 SMs and no row tier,
#                                        the RTX 5070 it was timed on (finding
#                                        160); elsewhere off, and its default
#                                        grid there is 512 per 48 SMs
#   The --slab-fill-ahead arms run only where it can: a plan of several slabs
#   per q under --q-pipeline (bench's first run says which). Its right grid is
#   a count of block slots per SM, so of that ladder only the grids of 6
#   blocks per SM up to the most this card holds resident (bench's tuned-run
#   line; 24 if it prints none) are timed (512 is 10.7 on 48 SMs), and an arm
#   equal to this card's default is not timed. A recommendation of
#   --slab-fill-ahead here is worth reporting: the default is off on every
#   card but the 5070, and the one other card measured, an RTX 3060, loses
#   with it beside its own best --fill-blocks (RESULTS finding 161).
#
# Method (STATUS item 2, RESULTS findings 85, 145, 146 and 149). Every arm sieves the
# same q, and the arms are interleaved within a round. The order is rotated
# every two rounds and reversed in the second of each pair, so a drift within
# a round favours an arm in one round of the pair and the defaults in the
# other. A run on a busy card is not a measurement: tune.sh refuses to start
# with another process on any card or an NFS@Home CUDA client running, checks
# again every 3 s while bench runs, and stops if either appears. An arm is
# ADOPTED only if it was slower than the defaults in no round (bench prints
# the wall to 0.01 ms, so a tie is not a loss) and its median gain exceeds
# both 1% and the defaults' own spread between rounds. Rankings between
# nearby settings reproduce; their margins do not. For each knob the adopted
# arm with the largest gain wins; on a ladder knob, the arm nearest the
# defaults whose gain is within max(1%, the defaults' spread) of that.
#
# Out early. An arm slower than the defaults in one round can never be
# adopted, so in stage 1 it is not timed again after that round; in stage 2
# the same holds against the base (or the best so far). A stage whose arms
# are all out stops there. Every decision is the one timing every round
# would give from the same runs, but for two under -t: the time saved can
# let more of the search fit, and stage-2 knobs are left out in order of
# their stage-1 gains, where an out arm's gain is over the rounds it ran.
# The ladder passes and the confirmation stage time every arm in every
# round: their rules compare two arms that may both lose to the defaults.
# -f turns this off.
#
# Ladders (the grid's four numeric knobs; finding 161's 3060 wanted
# --fill-blocks 589824, 16 times the grid's top). When a ladder knob's best
# adopted arm is the grid's top or bottom one, a ladder pass times the next
# two steps past it, with the defaults and that edge, and the next pass goes
# on from the farther step, until the newest step is clearly slower than the
# edge (the adoption rule) or the ladder reaches its bound:
#   --fill-blocks        doubling, within [SMs, min(2^20, the card's grid limit)]
#   --ss-warp-cut        doubling, above the block cut (36), up to 32768
#   --ss-block-cut       40 48 64 96 128, below the warp cut (256)
#   --slab-fill-blocks   x1.5 and x4/3 in turn (256 384 512 768 ...), within 6
#                        to the most resident blocks per SM
# A doubling past a bound is clamped to it (bench refuses a value past
# bench.h's bound, and tunecheck.sh checks the copies below against bench.h).
#
# Stage 2, coordinate descent (knobs can undo each other: finding 161, where
# --slab-fill-ahead beside the 3060's big fill grid was slower than the grid
# alone). The best single setting is the base. Every other knob's arms, the
# grid's and each ladder's winner, are timed on top of it, with the defaults
# and the base in each round. Those that beat the base are added one at a
# time, largest gain first, each timed against the best so far (base+A+B
# against base+A) and kept only if it beats it by the adoption rule. -q
# leaves stage 2 out: the best single setting is the recommendation.
#
# The report goes to stdout and to DIR/report.txt, with the commit and the
# card taken from the defaults' first run log (s1.default.1.log). Exits 0
# when done, 1 if a run failed or its relations differed, 2 if it refused to
# start, 3 if the card became busy partway through, and 4 if the -t budget
# ran out in stage 1 or in the confirmation stage (nothing is then written).
BIN=./bench; ROUNDS=6; DIR=; ARMS=; NOWRITE=; TPATH=; WGIVEN=; NOSAVE=; QUICK=; TMIN=; TGIVEN=; FULL=
while [ $# -gt 0 ]; do
  case $1 in
    -b) BIN=$2; shift 2 ;;
    -r) ROUNDS=$2; shift 2 ;;
    -o) DIR=$2; shift 2 ;;
    -a) ARMS=$2; shift 2 ;;
    -q) QUICK=1; shift ;;
    -t) TMIN=$2; TGIVEN=1; shift 2 ;;
    -f) FULL=1; shift ;;
    -n) NOWRITE=1; shift ;;
    -w) TPATH=$2; WGIVEN=1; [ -n "$TPATH" ] || { echo "tune: -w needs a file"; exit 2; }; shift 2 ;;
    --) shift; break ;;
    *) echo "tune: unknown option $1 (bench's arguments go after --)"; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "tune: no bench arguments after --"; exit 2; }
for a in "$@"; do
  case $a in
    --relations|--log|--tuned|--no-tuned) echo "tune: leave $a out; tune.sh sets it"; exit 2 ;;
    # the knobs it tunes: a base that sets one times every arm on top of it,
    # and a saved entry would leave it out
    --fill-blocks|--ss-block-cut|--ss-warp-cut|--fill-gate-shift|--no-fill-merge|--slab-fill-ahead|--no-slab-fill-ahead|--slab-fill-blocks)
      echo "tune: $a is a knob tune.sh times; give it as an -a arm instead"; exit 2 ;;
  esac
done
[ -x "$BIN" ] || { echo "tune: $BIN is not an executable"; exit 2; }
# A bench older than saved tuning has no --no-tuned, and would refuse it.
if "$BIN" --help 2>/dev/null | grep -q -- '--no-tuned'; then NOTUNED=--no-tuned
else NOTUNED=; NOSAVE="this bench predates saved tuning (no --no-tuned in its --help)"; fi
# Only --pipeline runs read saved tuning, and bench refuses --tuned and
# --no-tuned on any other run, as a bench that predates them refuses both.
case " $* " in
  *" --pipeline "*) ;;
  *) NOTUNED=; NOSAVE="the job line has no --pipeline, and only pipeline runs read saved tuning" ;;
esac
[ -n "$NOTUNED" ] || TPATH=
case $ROUNDS in ''|*[!0-9]*) echo "tune: -r needs a number"; exit 2 ;; esac
[ "$ROUNDS" -ge 3 ] || { echo "tune: -r must be at least 3"; exit 2; }
BUDGET=
if [ -n "$TGIVEN" ]; then
  case $TMIN in ''|*[!0-9]*) echo "tune: -t needs a whole number of minutes"; exit 2 ;; esac
  [ "$TMIN" -gt 0 ] || { echo "tune: -t needs a whole number of minutes"; exit 2; }
  BUDGET=$((TMIN * 60))
fi
# bench.h's bounds and defaults for the ladder knobs, mirrored; tunecheck.sh
# checks each against bench.h (and FILL_DEF against bench --help's text too)
BLOCKS_MAX=1048576   # BENCH_BLOCKS_MAX, 1 << 20
SS_BLOCK_MIN=33      # SS_BLOCK_CUT_MIN
SS_BLOCK_MAX=128     # SS_BLOCK_CUT_MAX
SS_WARP_MAX=32768    # SS_WARP_CUT_MAX
SS_BLOCK_DEF=36      # SS_BLOCK_CUT_DEFAULT
SS_WARP_DEF=256      # SS_WARP_CUT_DEFAULT
FILL_DEF=4608        # FILL_BLOCKS_DEFAULT, --fill-blocks 0's automatic grid
SFA_BLOCKS_DEF=512   # SLAB_FILL_BLOCKS_DEFAULT, the slab fill ahead's grid
SFA_DEF_SMS=48       # SLAB_FILL_DEFAULT_SMS, ... per this many SMs
BLOCK_LADDER="40 48 64 96 128"
ARMS_GIVEN=$ARMS
[ -n "$ARMS" ] || ARMS="B40=--ss-block-cut,40 B48=--ss-block-cut,48 B64=--ss-block-cut,64
  W128=--ss-warp-cut,128 W384=--ss-warp-cut,384 W512=--ss-warp-cut,512 W1024=--ss-warp-cut,1024
  F1152=--fill-blocks,1152 F2304=--fill-blocks,2304 F9216=--fill-blocks,9216 F18432=--fill-blocks,18432
  F36864=--fill-blocks,36864 G1=--fill-gate-shift,1 G2=--fill-gate-shift,2 G4=--fill-gate-shift,4 M0=--no-fill-merge
  SFA0=--no-slab-fill-ahead SFA256=--slab-fill-ahead,--slab-fill-blocks,256 SFA384=--slab-fill-ahead,--slab-fill-blocks,384
  SFA512=--slab-fill-ahead,--slab-fill-blocks,512 SFA768=--slab-fill-ahead,--slab-fill-blocks,768
  SFA1024=--slab-fill-ahead,--slab-fill-blocks,1024 SFA1536=--slab-fill-ahead,--slab-fill-blocks,1536
  SFA2048=--slab-fill-ahead,--slab-fill-blocks,2048 SFA3072=--slab-fill-ahead,--slab-fill-blocks,3072"
ARMS=$(echo $ARMS)
# Leave out of the grid whatever bench would refuse on this job line: one
# refused arm fails in every round and voids the whole result.
if [ -z "$ARMS_GIVEN" ]; then
  keep=
  for a in $ARMS; do
    case "$a $* " in
      *=--ss-*" --no-smallsieve "*) ;;
      *=--fill-gate-shift*" --no-fill-gate "*|*=--fill-gate-shift*" --fill-concurrent "*) ;;
      *=--no-fill-merge*" --fill-merge "*|*=--no-fill-merge*" --fill-merge-cut "*) ;;
      # --slab-fill-ahead needs --q-pipeline, which these turn off (bench
      # refuses it), and --pipeline itself
      *slab-fill-ahead*" --no-q-pipeline "*|*slab-fill-ahead*" --fill-concurrent "*|*slab-fill-ahead*" --qspan "*) ;;
      *slab-fill-ahead*) case " $* " in *" --pipeline "*) keep="$keep $a" ;; esac ;;
      *) keep="$keep $a" ;;
    esac
  done
  ARMS=$(echo $keep)
fi
seen=" "
for a in $ARMS; do
  case $a in *=--*) ;; *) echo "tune: arm '$a' is not NAME=FLAGS"; exit 2 ;; esac
  nm=${a%%=*}
  case $nm in default|combined|cand|saved|*[!A-Za-z0-9_]*|'') echo "tune: arm name '$nm' must be letters, digits or _, and not default, combined, cand or saved"; exit 2 ;; esac
  case $seen in *" $nm "*) echo "tune: arm name '$nm' is used twice"; exit 2 ;; esac
  # what bench takes on its command line but the saved file would not
  case ",${a#*=}," in *,--fill-blocks,0,*)
    echo "tune: arm $nm: --fill-blocks 0 is bench's automatic grid, which a saved entry cannot hold; give a number"; exit 2 ;;
  esac
  dup=$(echo "${a#*=}" | tr , '\n' | grep -- '^--' | sort | uniq -d | head -1)
  [ -z "$dup" ] || { echo "tune: arm $nm gives $dup twice"; exit 2; }
  seen="$seen$nm "
done
# The default DIR lands in the working directory; .gitignore has tune.*/ so
# a run inside the checkout does not mark the next build -dirty.
[ -n "$DIR" ] || DIR=tune.$(hostname -s 2>/dev/null || echo host).$(date +%Y%m%d-%H%M%S)
mkdir -p "$DIR" || exit 2
REP=$DIR/report.txt; : > "$REP"
say() { echo "$*"; echo "$*" >> "$REP"; }

# Busy: a process on any card (nvidia-smi lists every card's, and which card
# bench will pick is not known before it starts, so on a host with several
# cards all must be idle), or a local NFS@Home CUDA client (TUNE_BUSY, a
# pattern for ps; empty to skip). The CPU client has the same name and never
# touches the card, so only a CUDA one counts. The default matches the
# client sieving (a flag straight after its name, as run-cuda-client.sh
# starts it) or running its GPU `benchmark`, but not its `tune` subcommand:
# that is the client running this script, with --cuda-bench= on its own
# command line, and matching it refused every tune the client started. A
# new client subcommand that sieves on the card would need adding here.
# tunecheck.sh tests this pattern against each form. busy PID leaves out
# bench's own process.
BUSY_DEFAULT='run-cuda-client|ggnfs-sieve-client +(--|benchmark ).*(engine=cuda|cuda-bench)'
BUSY=${TUNE_BUSY-$BUSY_DEFAULT}
# Seconds between checks on a running bench (busy every third). TUNE_POLL is
# for tunecheck.sh's fake bench, whose runs take milliseconds.
POLL=${TUNE_POLL:-1}
busy() {
  if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
      tr -d ' ' | grep -v -x -e '' -e "${1:-none}" | grep -q . && return 0
  fi
  [ -n "$BUSY" ] && ps -eo args | grep -v grep | grep -qE "$BUSY"
}
# A hit is checked again 3 s later: nvidia-smi can still list a process that
# has just exited (work/tunecut/run.sh, 2026-11-01).
busy_now() { busy "$@" && { sleep 3; busy "$@"; }; }
if busy_now; then say "tune: another process is on the card; not starting"; exit 2; fi

# The clock, in seconds. TUNE_FAKE_CLOCK names a file holding it instead, for
# tunecheck.sh's fake bench, which adds its pretend run time to it.
now() { if [ -n "${TUNE_FAKE_CLOCK:-}" ]; then cat "$TUNE_FAKE_CLOCK"; else date +%s; fi; }
START=$(now); RUN_N=0; RUN_T=0; CUT=
avg_run() { if [ $RUN_N -gt 0 ]; then echo $(( (RUN_T + RUN_N - 1) / RUN_N )); else echo 0; fi; }
# reserve: seconds held back for the confirmation stage (the defaults, the
# recommendation and, when bench applies one, the saved entry). None when
# the tune cannot save, as far as the defaults' first run tells (REASON's
# cases below but the failed runs and the flags of the recommendation):
# the confirmation runs only before a save.
reserve() {
  if [ -n "$NOWRITE" ] || { [ -n "$ARMS_GIVEN" ] && [ -z "$WGIVEN" ]; } || [ -n "$NOSAVE" ]; then echo 0; return 0; fi
  _c=2; [ "${TR_STATUS:-}" = apply ] && _c=3; echo $((_c * ROUNDS * $(avg_run)))
}
# fits N: whether a stage of N arms ends before the reserve (always, with no -t)
fits() {
  [ -n "$BUDGET" ] || return 0
  [ $(( $(now) - START + $1 * ROUNDS * $(avg_run) + $(reserve) )) -le "$BUDGET" ]
}
# trim_budget BASE FILE: of FILE's "gain count name" lines, largest gain
# first, the names whose arms (count each) fit in one stage beside BASE arms
# into KEEP; from the first that does not fit on, into SKIP
trim_budget() {
  KEEP=; SKIP=; _tot=$1; _full=
  while read -r _g _c _n; do
    [ -n "$_n" ] || continue
    if [ -z "$_full" ] && fits $((_tot + _c)); then KEEP="$KEEP $_n"; _tot=$((_tot + _c))
    else _full=1; SKIP="$SKIP $_n"; fi
  done <<EOF
$(sort -k1,1gr "$2")
EOF
}

wall_of() {  # the band's wall clock per q, from a bench stdout file
  awk '/wall clock per q, COMPLETE/ { c = $(NF-1) } /wall clock per q *[0-9]/ { w = $(NF-1) }
       END { if (c != "") print c; else if (w != "") print w }' "$1"
}
flags_of() {  # the flags of arm $1 (empty for default), commas to spaces
  [ "$1" = default ] && return 0
  for a in $ARMS; do [ "${a%%=*}" = "$1" ] && echo "${a#*=}" | tr , ' '; done
}
fails=0
NOOP=
# tuned_first OUT: from the defaults' first run, bench's own view of saved
# tuning (bench_main.cu, tuned_step): the key and applicability record to
# write, the incumbent (what bench would apply), the plan's slabs per q, any
# flags outside bench's plain base, and the file's path when -w gave none.
#   tuned-run: card|cc|SMs|logI|J|region|side|alim|rlim|policy|maxgrid|
#              maxblocks|status|would|overrides|path
TR=; TR_STATUS=; SAVED=; TR_OVER=; TR_NSLAB=; TR_MAXGRID=; TR_MAXBLK=; NOSAVE=${NOSAVE-}
tuned_first() {
  [ -n "$NOTUNED" ] || return 0
  TR=$(sed -n 's/^tuned-run: //p' "$1" | head -1)
  if [ -z "$TR" ]; then NOSAVE="bench printed no tuned-run line"; return 0; fi
  TR_STATUS=$(printf '%s\n' "$TR" | awk -F'|' '{ print $13 }')
  SAVED=$(printf '%s\n' "$TR" | awk -F'|' '{ print $14 }')
  TR_OVER=$(printf '%s\n' "$TR" | awk -F'|' '{ print $15 }')
  p=$(printf '%s\n' "$TR" | awk -F'|' '{ s = $16; for (i = 17; i <= NF; i++) s = s "|" $i; print s }')
  TR_MAXGRID=$(printf '%s\n' "$TR" | awk -F'|' '{ print $11 }')
  TR_MAXBLK=$(printf '%s\n' "$TR" | awk -F'|' '{ print $12 }')
  TR_NSLAB=$(sed -n 's/^tuned-plan: \([0-9][0-9]*\) slab.*/\1/p' "$1" | head -1)
  [ -n "$TPATH" ] || TPATH=$p
  say "   saved tuning: $(sed -n 's/^tuned: //p' "$1" | head -1)"
  [ -n "$TR_OVER" ] && NOSAVE="the job line sets $TR_OVER, outside the plain base saved entries are tuned on"
  [ -n "$TPATH" ] || NOSAVE="bench names no saved-tuning file (no HOME?); give -w FILE"
  [ "$TR_STATUS" = badfile ] && NOSAVE="bench cannot use $TPATH (see its tuned: line); fix or move it"
  [ -n "$TR_NSLAB" ] || NOSAVE="bench printed no tuned-plan line"
  return 0
}
# run_stage STAGE "NAMES": every arm in NAMES (default first) for ROUNDS
# rounds, rotated every two rounds and reversed in the second; appends
# "stage arm round wall" lines to DIR/walls.txt. Under -t, a run that would
# end past the budget (less the confirmation's reserve, outside stage 3) is
# not started: CUT is set to STAGE and the stage returns early. With DROP set
# (an arm in NAMES, and no -f), an arm slower than DROP in a round is out:
# not timed again ("STAGE ARM ROUND" in DIR/out.txt), and once every arm but
# the defaults and DROP is out the stage stops (STOPPED, the round).
run_stage() {
  st=$1; names=$2; shift 2; n=$(echo $names | wc -w); OUT=; STOPPED=
  dref=; [ -n "$FULL" ] || dref=${DROP:-}
  for rd in $(seq 1 "$ROUNDS"); do
    k=$(( ((rd + 1) / 2 - 1) % n ))
    order="$(echo $names | tr ' ' '\n' | tail -n +$((k + 1)) | tr '\n' ' ')$(echo $names | tr ' ' '\n' | head -n $k | tr '\n' ' ')"
    [ $((rd % 2)) -eq 0 ] && order=$(echo $order | awk '{ for (i = NF; i > 0; i--) printf "%s ", $i }')
    for arm in $order; do
      case " $NOOP $OUT " in *" $arm "*) continue ;; esac
      if [ -n "$BUDGET" ]; then
        lim=$BUDGET; [ "$st" = 3 ] || lim=$((BUDGET - $(reserve)))
        if [ $(( $(now) - START + $(avg_run) )) -gt $lim ]; then CUT=$st; return 0; fi
      fi
      if busy_now; then say "tune: another process came onto the card before stage $st $arm round $rd; stopping"; exit 3; fi
      o=$DIR/s$st.$arm.$rd
      # A reused DIR: an interrupted run's .rel.part and checkpoint would
      # make bench resume it and time only the rest of the band.
      rm -f "$o".*
      t0=$(now)
      "$BIN" "$@" $(flags_of $arm) ${TPATH:+--tuned "$TPATH"} $NOTUNED --relations "$o.rel" --log "$o.log" > "$o.out" 2>&1 &
      pid=$!; i=0
      while kill -0 $pid 2>/dev/null; do
        sleep "$POLL"; i=$((i + 1))
        if [ $((i % 3)) -eq 0 ] && kill -0 $pid 2>/dev/null && busy_now $pid; then
          kill $pid 2>/dev/null; wait $pid 2>/dev/null
          say "tune: another process came onto the card during stage $st $arm round $rd; stopping"; exit 3
        fi
      done
      wait $pid; rc=$?
      t1=$(now); RUN_N=$((RUN_N + 1)); RUN_T=$((RUN_T + t1 - t0))
      w=$(wall_of "$o.out")
      ref=$DIR/s1.default.1.rel
      if [ "$st $arm $rd" = "1 default 1" ]; then
        # Everything is compared with this run, so without it nothing can be.
        if [ $rc -ne 0 ] || [ -z "$w" ]; then say "tune: the defaults' first run failed (bench exited $rc, see $o.out); nothing to compare against"; exit 1; fi
        [ -s "$ref" ] || { say "tune: the defaults' first run found no relations; pick a band that finds some"; exit 1; }
        say "   commit $(sed -n 's/^# commit *//p' "$o.log" | head -1)"
        say "   card $(sed -n 's/^# device *//p' "$o.log" | head -1)"
        tuned_first "$o.out"
        # --no-fill-merge on a plan with no merge would time the defaults
        # against themselves, and noise could adopt a flag that does nothing.
        if ! grep -q -- '--fill-merge: .*one launch' "$o.out"; then
          nm=; for a in $ARMS; do [ "${a#*=}" = --no-fill-merge ] && nm="$nm ${a%%=*}"; done
          [ -n "$nm" ] && say "   not timing$nm: this plan has no merge to turn off"
          NOOP="$NOOP$nm"
        fi
        # --slab-fill-ahead: where it can run, this card's default (on at
        # grid G, or off), and the ladder's grids of 6 blocks per SM up to
        # the most this card holds resident (bench's tuned-run line; 24 if
        # it prints none)
        sms=$(sed -n 's/^grid: \([0-9][0-9]*\) SMs.*/\1/p; s/^grid: [0-9][0-9]* blocks on \([0-9][0-9]*\) SMs.*/\1/p' "$o.out" | head -1)
        sg=$(sed -n "s/.*--slab-fill-ahead: each later slab.* \([0-9][0-9]*\) x [0-9][0-9]* at the greatest priority.*/\1/p" "$o.out" | head -1)
        [ -n "$sms" ] || sms=$(printf '%s\n' "$TR" | awk -F'|' '{ print $3 }')
        case $sms in *[!0-9]*) sms= ;; esac
        # the ladders' card limits: SMs, the grid limit, resident blocks per SM
        CARD_SMS=$sms; SFA_DEF=$sg
        mb=$TR_MAXBLK; case $mb in ''|*[!0-9]*|0) mb=24 ;; esac; MAXBLK=$mb
        FILL_HI=$BLOCKS_MAX
        case $TR_MAXGRID in ''|*[!0-9]*) ;; *) [ "$TR_MAXGRID" -lt $FILL_HI ] && FILL_HI=$TR_MAXGRID ;; esac
        FILL_LO=${sms:-1}
        nm=; why=
        if ! grep -q -- "--q-pipeline: each slab's sieve" "$o.out"; then why="it needs a plan of several slabs per q under --q-pipeline"; fi
        for a in $ARMS; do
          m=${a%%=*}; f=${a#*=}
          case $f in
            --no-slab-fill-ahead) [ -n "$why" ] || [ -z "$sg" ] && nm="$nm $m" ;;
            --slab-fill-ahead,--slab-fill-blocks,*)
              b=${f##*,}
              if [ -n "$why" ] || [ "$b" = "$sg" ] || [ -z "$sms" ] ||
                 [ "$b" -lt $((6 * sms)) ] || [ "$b" -gt $((mb * sms)) ]; then nm="$nm $m"; fi ;;
          esac
        done
        if [ -n "$nm" ]; then
          if [ -n "$why" ]; then say "   not timing$nm: $why"
          else
            if [ -n "$sg" ]; then st_sfa="on at $sg blocks"; else st_sfa=off; fi
            say "   --slab-fill-ahead is $st_sfa in the defaults' run on this card${sms:+ ($sms SMs)}; not timing$nm (the defaults' own setting, or outside 6-$mb blocks per SM)"
          fi
          NOOP="$NOOP$nm"
        fi
        nt=$n; for m in $NOOP; do nt=$((nt - 1)); done
        say "   one run takes about $((t1 - t0)) s; stage 1 is $([ -n "$FULL" ] || echo 'at most ')$(( ((t1 - t0) * nt * ROUNDS + 59) / 60 )) min ($nt arms timed)"
        if [ -n "$BUDGET" ]; then
          s1t=$(( (t1 - t0) * nt * ROUNDS )); rst=$(reserve)
          if [ $((s1t + rst)) -gt "$BUDGET" ]; then
            say "tune: -t $TMIN is too short: stage 1 takes $([ -n "$FULL" ] && echo about || echo 'up to') $(( (s1t + 59) / 60 )) min$([ "$rst" -gt 0 ] && echo " and the confirmation stage about $(( (rst + 59) / 60 )) more"); give -t $(( (s1t + rst + 59) / 60 + 1 )) or more, or fewer arms (-a)"
            exit 2
          fi
          if [ "$rst" -gt 0 ]; then say "   -t $TMIN: $(( (rst + 59) / 60 )) min held back for the confirmation stage"
          else say "   -t $TMIN: nothing held back; this tune does not save, so it has no confirmation stage"; fi
        fi
      else
        if [ $rc -ne 0 ] || [ -z "$w" ]; then
          say "   stage $st $arm round $rd: bench exited $rc (see $o.out)"; fails=$((fails+1)); continue
        fi
        # --slab-fill-ahead turns itself off when its memory does not fit:
        # such an arm would time the defaults against themselves
        # Not the saved entry: what bench does with it, off included, is what
        # its runs would get, and what a re-tune has to beat.
        if [ "$arm" != saved ] && [ "$(flags_of $arm | grep -c -- '--slab-fill-ahead')" -gt 0 ] &&
           grep -q -- '--slab-fill-ahead: off, because' "$o.out"; then
          say "   not timing $arm further: $(grep -h -- '--slab-fill-ahead: off, because' "$o.out" | head -1 | sed 's/^ *//')"
          NOOP="$NOOP $arm"; rm -f "$o.rel" "$o.rel".* "$o.log"; continue
        fi
        if cmp -s "$o.rel" "$ref"; then rm -f "$o.rel" "$o.rel".* "$o.log"
        else say "   stage $st $arm round $rd: RELATIONS DIFFER from the defaults' (a bug: keep $o.rel)"; echo "$st $arm DIFFERS" >> "$DIR/differs.txt"; fails=$((fails+1)); fi
      fi
      echo "$st $arm $rd $w" >> "$DIR/walls.txt"
    done
    # (after the last round there is nothing left to save)
    [ -n "$dref" ] && [ $rd -lt "$ROUNDS" ] || continue
    # slower than DROP in this round: it can no longer be adopted (the rule
    # needs it slower in no round), so it is not timed again
    left=
    for arm in $names; do
      case " default $dref $NOOP $OUT " in *" $arm "*) continue ;; esac
      if awk -v st="$st" -v a="$arm" -v b="$dref" -v r=$rd '$1 == st && $3 == r { w[$2] = $4 }
           END { exit !((a in w) && (b in w) && w[a] + 0 > w[b] + 0) }' "$DIR/walls.txt"; then
        OUT="$OUT $arm"; echo "$st $arm $rd" >> "$DIR/out.txt"
      else left=1; fi
    done
    if [ -z "$left" ]; then
      STOPPED=$rd
      if [ -n "$OUT" ]; then say "   every arm is out after round $rd (each was slower than $([ "$dref" = default ] && echo the defaults || echo $dref) in a round); the stage stops"
      else say "   no arm is left to time after round $rd; the stage stops"; fi
      return 0
    fi
  done
}
# summarise STAGE [REF]: per arm against REF (the defaults if none; REF itself
# against the defaults), medians over rounds of the wall ratio, rounds faster
# and tied, the default's spread, and ADOPTED or not. Prints "arm gain%
# adopted(0/1) spread%" to DIR/s$STAGE.pick, for every arm but REF. An arm
# out early (run_stage) is shown with the round it went out in; it is never
# adopted.
summarise() {
  st=$1; sref=${2:-default}
  awk -v st=$st -v ref="$sref" -v R="$ROUNDS" -v differs="$DIR/differs.txt" -v pick="$DIR/s$st.pick" \
      -v outs="$DIR/out.txt" -v stopped="${STOPPED:-}" '
    function med(v, n,   i, j, t) { for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
                                    return n % 2 ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }
    BEGIN { printf "" > pick
            while ((getline l < differs) > 0) { split(l, x, " "); if (x[1] == st) bad[x[2]] = 1 }
            while ((getline l < outs) > 0) { split(l, x, " "); if (x[1] == st) out[x[2]] = x[3] } }
    $1 == st { w[$2, $3] = $4; arms[$2] = 1; if (!($2 in first)) { first[$2] = ++na; ord[na] = $2 } }
    END {
      n = 0; for (r = 1; r <= R; r++) if (("default", r) in w) d[++n] = w["default", r]
      if (n == 0 || (n < 3 && stopped == "")) { print "   too few default runs to judge"; exit }
      lo = d[1]; hi = d[1]; for (i = 1; i <= n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i] }
      dm = med(d, n); spread = (hi - lo) / dm * 100
      printf "   %-8s %9.3f ms/q (median of %d round%s)%s\n", "default", dm, n, (n == 1) ? "" : "s", (n > 1) ? sprintf(", spread between rounds %.2f%%", spread) : ""
      for (a = 1; a <= na; a++) { m = ord[a]; if (m == "default") continue
        b = (m == ref) ? "default" : ref
        k = 0; f = 0; s = 0; delete x
        for (r = 1; r <= R; r++) if ((b, r) in w && (m, r) in w) { x[++k] = w[m, r] / w[b, r]; f += (x[k] < 1); s += (x[k] > 1) }
        if (!k) continue
        g = (1 - med(x, k)) * 100
        ok = (n >= 3 && k == n && s == 0 && g > 1 && g > spread && !(m in bad) && !(m in out))
        if (m in out)
          printf "   %-8s %+8.2f%% wall%s (median of %d), out after round %d: slower than %s in it%s\n", m, g ? -g : 0, b == "default" ? "" : " against " b, k, out[m], b == "default" ? "the defaults" : b, (m in bad) ? "  RELATIONS DIFFER" : ""
        else
          printf "   %-8s %+8.2f%% wall%s (median), faster in %d of %d rounds%s%s\n", m, g ? -g : 0, b == "default" ? "" : " against " b, f, k, (k - f - s) ? sprintf(", %d tied", k - f - s) : "", (m in bad) ? "  RELATIONS DIFFER" : ok ? "  ADOPTED" : ""
        if (m != ref) print m, g, ok, spread > pick
      } }' "$DIR/walls.txt" | tee -a "$REP"
}
# beats STAGE A B: one report line for arm A against arm B over STAGE's rounds,
# by the rule summarise applies against the defaults (slower in no round, a
# median gain over 1% and over the defaults' spread), ending in ADOPTED if A
# wins.
beats() {
  awk -v st=$1 -v A=$2 -v B=$3 -v R="$ROUNDS" '
    function med(v, n,   i, j, t) { for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
                                    return n % 2 ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }
    $1 == st { w[$2, $3] = $4 }
    END {
      n = 0; for (r = 1; r <= R; r++) if (("default", r) in w) d[++n] = w["default", r]
      if (n < 3) { printf "   %s against %s: too few default runs to judge\n", A, B; exit }
      lo = d[1]; hi = d[1]; for (i = 1; i <= n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i] }
      spread = (hi - lo) / med(d, n) * 100
      k = 0; f = 0; s = 0
      for (r = 1; r <= R; r++) if ((A, r) in w && (B, r) in w) { x[++k] = w[A, r] / w[B, r]; f += (x[k] < 1); s += (x[k] > 1) }
      if (!k) { printf "   %s against %s: no rounds with both\n", A, B; exit }
      g = (1 - med(x, k)) * 100
      ok = (k == n && s == 0 && g > 1 && g > spread)
      printf "   %-8s %+8.2f%% wall against %s (median), faster in %d of %d rounds%s\n", A, g ? -g : 0, B, f, k, ok ? "  ADOPTED" : ""
    }' "$DIR/walls.txt"
}

say "tune.sh on $(hostname 2>/dev/null), $(date '+%Y-%m-%d %H:%M')"
say "   bench $BIN, sha256 $(sha256sum "$BIN" 2>/dev/null | cut -c1-8)"
say "   job: $*"
say "   $ROUNDS rounds; load $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
[ -n "$WSL_DISTRO_NAME" ] && say "   NOTE: under WSL nvidia-smi does not show Windows programs using the card; close them (a browser tab cost 6-8%)"
: > "$DIR/walls.txt"; : > "$DIR/differs.txt"; : > "$DIR/out.txt"; rm -f "$DIR"/s*.pick
NAMES="default $(for a in $ARMS; do printf '%s ' "${a%%=*}"; done)"
say "== stage 1: $(echo $NAMES | wc -w) arms, $(date +%T)"
DROP=default; run_stage 1 "$NAMES" "$@"
if [ "$CUT" = 1 ]; then say "== stopping: the -t budget ran out in stage 1; nothing is judged or written"; exit 4; fi
summarise 1

# knobs_of FLAGS: the knobs FLAGS set, each once, values dropped and --no-X
# as --X (arms that share one are one knob, and an -a arm may set several).
# --slab-fill-blocks is --slab-fill-ahead's: bench refuses its grid with the
# slab fill ahead off, so it never pairs with --no-slab-fill-ahead.
knobs_of() {
  for _f in $1; do
    case $_f in --slab-fill-blocks) echo --slab-fill-ahead ;; --no-*) echo "--${_f#--no-}" ;; --*) echo "$_f" ;; esac
  done | awk '!s[$0]++ { printf "%s ", $0 }'
}
shares() { for _x in $(knobs_of "$1"); do case " $(knobs_of "$2") " in *" $_x "*) return 0 ;; esac; done; return 1; }
# valid_flags FLAGS: what bench would accept of the knobs' bounds and pairing
valid_flags() {
  _fb=; _bc=$SS_BLOCK_DEF; _wc=$SS_WARP_DEF; _prev=
  for _f in $1; do
    case $_prev in --fill-blocks) _fb=$_f ;; --ss-block-cut) _bc=$_f ;; --ss-warp-cut) _wc=$_f ;; esac
    _prev=$_f
  done
  [ "$_wc" -gt "$_bc" ] && [ "$_bc" -ge $SS_BLOCK_MIN ] && [ "$_bc" -le $SS_BLOCK_MAX ] &&
    [ "$_wc" -le $SS_WARP_MAX ] && { [ -z "$_fb" ] || [ "$_fb" -le "$FILL_HI" ]; }
}
# ladder_of NAME: "KNOB VALUE" when the arm sets one ladder knob alone
ladder_of() {
  _l=$(flags_of "$1")
  case "$_l" in
    "--fill-blocks "*[0-9]) _k=fill ;; "--ss-warp-cut "*[0-9]) _k=warp ;; "--ss-block-cut "*[0-9]) _k=block ;;
    "--slab-fill-ahead --slab-fill-blocks "*[0-9]) _k=sfa ;; *) return 0 ;;
  esac
  _v=${_l##* }; case $_v in *[!0-9]*) return 0 ;; esac
  [ "$(echo $_l | wc -w)" -eq $([ $_k = sfa ] && echo 3 || echo 2) ] && echo "$_k $_v"
}
knob_flag() { case $1 in fill) echo --fill-blocks ;; warp) echo --ss-warp-cut ;; block) echo --ss-block-cut ;; sfa) echo --slab-fill-blocks ;; esac; }
knob_def() {
  case $1 in
    fill) echo $FILL_DEF ;; warp) echo $SS_WARP_DEF ;; block) echo $SS_BLOCK_DEF ;;
    sfa) if [ -n "$SFA_DEF" ]; then echo "$SFA_DEF"; elif [ -n "$CARD_SMS" ]; then echo $((SFA_BLOCKS_DEF * CARD_SMS / SFA_DEF_SMS)); else echo $SFA_BLOCKS_DEF; fi ;;
  esac
}
# ladder_next KNOB VALUE up|down: the ladder's next step, clamped to its
# bound; nothing past the bound
ladder_next() {
  _v=$2; _n=
  case $1 in
    fill)
      if [ $3 = up ]; then _n=$((_v * 2)); [ $_n -le $FILL_HI ] || { if [ $_v -lt $FILL_HI ]; then _n=$FILL_HI; else _n=; fi; }
      else _n=$((_v / 2)); [ $_n -ge $FILL_LO ] || { if [ $_v -gt $FILL_LO ]; then _n=$FILL_LO; else _n=; fi; }; fi ;;
    warp)
      _lo=$((SS_BLOCK_DEF + 1))
      if [ $3 = up ]; then _n=$((_v * 2)); [ $_n -le $SS_WARP_MAX ] || { if [ $_v -lt $SS_WARP_MAX ]; then _n=$SS_WARP_MAX; else _n=; fi; }
      else _n=$((_v / 2)); [ $_n -ge $_lo ] || { if [ $_v -gt $_lo ]; then _n=$_lo; else _n=; fi; }; fi ;;
    block)
      if [ $3 = up ]; then _n=$(for _x in $BLOCK_LADDER; do [ $_x -gt $_v ] && echo $_x; done | head -1)
      else _n=$(for _x in $BLOCK_LADDER; do [ $_x -lt $_v ] && echo $_x; done | tail -1); fi
      [ -n "$_n" ] && { [ $_n -lt $SS_BLOCK_MIN ] || [ $_n -gt $SS_BLOCK_MAX ] || [ $_n -ge $SS_WARP_DEF ] || [ $_n -eq $SS_BLOCK_DEF ]; } && _n= ;;
    sfa)
      [ -n "$CARD_SMS" ] || return 0
      _n=$_v
      while :; do
        if [ $(( _n & (_n - 1) )) -eq 0 ]; then
          if [ $3 = up ]; then _n=$((_n * 3 / 2)); else _n=$((_n * 3 / 4)); fi
        else
          if [ $3 = up ]; then _n=$((_n * 4 / 3)); else _n=$((_n * 2 / 3)); fi
        fi
        [ "$_n" = "$SFA_DEF" ] || break
      done
      { [ $_n -lt $((6 * CARD_SMS)) ] || [ $_n -gt $((MAXBLK * CARD_SMS)) ]; } && _n= ;;
  esac
  echo $_n
}
# ladder_arm KNOB VALUE: the arm for that step (named as the grid names its
# arms), added to ARMS if new; its name in LA. Not in $(...): it sets ARMS.
ladder_arm() {
  case $1 in
    fill) LA=F$2; _f=--fill-blocks,$2 ;; warp) LA=W$2; _f=--ss-warp-cut,$2 ;;
    block) LA=B$2; _f=--ss-block-cut,$2 ;; sfa) LA=SFA$2; _f=--slab-fill-ahead,--slab-fill-blocks,$2 ;;
  esac
  case " $ARMS " in *" $LA="*) ;; *) ARMS="$ARMS $LA=$_f" ;; esac
}
# winners PICK: PICK's rows as "arm gain adopted", where on each ladder knob
# only one adopted arm stays adopted: the nearest the defaults (in ratio)
# among those within max(1%, the defaults' spread) of the knob's best gain
winners() {
  while read -r _m _g _ok _sp; do echo "$_m $(ladder_of "$_m")"; done < "$1" > "$DIR/ladder.kv"
  awk -v fd=$FILL_DEF -v wdf=$SS_WARP_DEF -v bd=$SS_BLOCK_DEF -v sd="$(knob_def sfa)" '
    NR == FNR { if (NF == 3) { kn[$1] = $2; val[$1] = $3 }; next }
    { n++; arm[n] = $1; g[n] = $2; ok[n] = $3; sp[n] = $4 }
    END {
      def["fill"] = fd; def["warp"] = wdf; def["block"] = bd; def["sfa"] = sd
      for (i = 1; i <= n; i++) if (ok[i] && (arm[i] in kn)) { k = kn[arm[i]]
        if (!(k in bg) || g[i] > bg[k]) { bg[k] = g[i]; bs[k] = sp[i] } }
      for (i = 1; i <= n; i++) if (ok[i] && (arm[i] in kn)) { k = kn[arm[i]]
        tol = bs[k] > 1 ? bs[k] : 1
        if (g[i] < bg[k] - tol) continue
        v = val[arm[i]]; d = def[k] > 0 ? log(v / def[k]) : 0; if (d < 0) d = -d
        if (!(k in win) || d < wd[k] || (d == wd[k] && v < wv[k])) { win[k] = arm[i]; wd[k] = d; wv[k] = v } }
      for (i = 1; i <= n; i++) { o = ok[i]; if (o && (arm[i] in kn) && win[kn[arm[i]]] != arm[i]) o = 0; print arm[i], g[i], o } }' "$DIR/ladder.kv" "$1"
}
# pick_wins PICK: each knob's winner, largest gain first, each kept only if
# it sets no knob a larger winner sets
pick_wins() {
  [ -s "$1" ] || return 0
  winners "$1" | awk '$3 == 1' | sort -k2,2gr | while read -r _m _g _ok; do echo "$_m $(knobs_of "$(flags_of "$_m")")"; done |
    awk '{ ok = 1; for (i = 2; i <= NF; i++) if ($i in used) ok = 0
           if (ok) { for (i = 2; i <= NF; i++) used[$i] = 1; print $1 } }'
}
gain_of() { awk -v a="$2" '$1 == a { printf "%.2f", $2 }' "$1"; }
SKIPPED=

# Ladders: each ladder knob whose best adopted arm is the grid's top or
# bottom one steps on past it. Not for -a arms, which are an experiment.
XPASS=0; EXTS=
if [ -z "$ARMS_GIVEN" ] && [ $fails -eq 0 ]; then
  while read -r _m _g _ok _sp; do
    kv=$(ladder_of "$_m"); [ -n "$kv" ] && echo "$_m $kv $_g $_ok"
  done < "$DIR/s1.pick" > "$DIR/s1.ladder"
  for k in fill warp block sfa; do
    e=$(awk -v k=$k '$2 == k && $5 == 1' "$DIR/s1.ladder" | sort -k4,4gr | awk 'NR == 1 { print $1, $3 }')
    [ -n "$e" ] || continue
    v=${e#* }; lo=$(awk -v k=$k '$2 == k { print $3 }' "$DIR/s1.ladder" | sort -n | head -1)
    hi=$(awk -v k=$k '$2 == k { print $3 }' "$DIR/s1.ladder" | sort -n | tail -1)
    if [ "$lo" = "$hi" ]; then if [ "$v" -lt "$(knob_def $k)" ]; then dir=down; else dir=up; fi
    elif [ "$v" = "$hi" ]; then dir=up
    elif [ "$v" = "$lo" ]; then dir=down
    else continue; fi
    EXTS="$EXTS $k:$dir:${e%% *}"
  done
fi
while [ -n "$EXTS" ]; do
  XPASS=$((XPASS + 1)); : > "$DIR/x.items"
  for e in $EXTS; do
    k=${e%%:*}; r=${e#*:}; dir=${r%%:*}; edge=${r#*:}
    ev=$(ladder_of "$edge"); ev=${ev#* }
    n1=$(ladder_next $k $ev $dir); n2=
    if [ -z "$n1" ]; then say "   $(knob_flag $k): $edge is the ladder's bound; it stops there"; continue; fi
    n2=$(ladder_next $k $n1 $dir)
    c=2; [ -n "$n2" ] && c=3
    echo "$(awk -v k=$k '$2 == k && $5 == 1 { print $4 }' "$DIR/s1.ladder" | sort -gr | head -1) $c $k:$dir:$edge:$n1:$n2" >> "$DIR/x.items"
  done
  EXTS=
  [ -s "$DIR/x.items" ] || break
  trim_budget 1 "$DIR/x.items"
  for e in $SKIP; do
    k=${e%%:*}; say "   $(knob_flag $k) past $(echo $e | cut -d: -f3): not timed, for the -t budget"; SKIPPED="$SKIPPED $(knob_flag $k)"
  done
  [ -n "$KEEP" ] || break
  xn=default; xplan=; xsay=
  for e in $KEEP; do
    k=$(echo $e | cut -d: -f1); dir=$(echo $e | cut -d: -f2); edge=$(echo $e | cut -d: -f3)
    n1=$(echo $e | cut -d: -f4); n2=$(echo $e | cut -d: -f5)
    ladder_arm $k $n1; a1=$LA; a2=
    [ -n "$n2" ] && { ladder_arm $k $n2; a2=$LA; }
    xn="$xn $edge $a1 $a2"; xplan="$xplan $k:$dir:$edge:$a1:$a2"
    xsay="$xsay, $(knob_flag $k) $n1${n2:+ and $n2} past $edge"
  done
  say "== ladder pass $XPASS: ${xsay#, }, $(date +%T)"
  DROP=; run_stage x$XPASS "$(echo $xn)" "$@"
  if [ "$CUT" = "x$XPASS" ]; then
    say "   the -t budget ran out in this pass; its runs are not used"
    for e in $xplan; do SKIPPED="$SKIPPED $(knob_flag ${e%%:*})"; done
    XPASS=$((XPASS - 1)); break
  fi
  summarise x$XPASS
  if [ $fails -ne 0 ]; then say "   a run failed or differed in this pass; the search stops here"; break; fi
  for e in $xplan; do
    k=$(echo $e | cut -d: -f1); dir=$(echo $e | cut -d: -f2); edge=$(echo $e | cut -d: -f3)
    a1=$(echo $e | cut -d: -f4); a2=$(echo $e | cut -d: -f5); newest=${a2:-$a1}
    case " $NOOP " in *" $newest "*) say "   $(knob_flag $k): $newest turns itself off; the ladder stops"; continue ;; esac
    vs=$(beats x$XPASS "$edge" "$newest"); say "$vs"
    case $vs in *ADOPTED) say "   $(knob_flag $k): $newest is clearly slower than $edge; the ladder stops"; continue ;; esac
    nv=$(ladder_of "$newest"); nv=${nv#* }
    if [ -z "$(ladder_next $k $nv $dir)" ]; then say "   $(knob_flag $k): $newest is the ladder's bound; it stops there"; continue; fi
    EXTS="$EXTS $k:$dir:$newest"
  done
done
# every arm's latest measurement against the defaults: stage 1's, or a
# ladder pass's for the edges it timed again
for p in 1 $(i=1; while [ $i -le $XPASS ]; do echo x$i; i=$((i + 1)); done); do
  [ -f "$DIR/s$p.pick" ] && cat "$DIR/s$p.pick"
done | awk '{ if (!($1 in r)) o[++n] = $1; r[$1] = $0 } END { for (i = 1; i <= n; i++) print r[o[i]] }' > "$DIR/ladder.pick"
WIN=$(pick_wins "$DIR/ladder.pick")
BEST=$(echo $WIN | awk '{ print $1 }')
[ -n "$WIN" ] && say "   each knob's winner: $(for m in $WIN; do printf '%s %s%%, ' "$m" "-$(gain_of "$DIR/ladder.pick" $m)"; done | sed 's/, $//')"

# Stage 2: coordinate descent from the best single setting.
REC=; CUR=$BEST
if [ -n "$BEST" ] && [ -n "$QUICK" ]; then say "   -q: no stage 2; $BEST stands"
elif [ -n "$BEST" ] && [ $fails -eq 0 ]; then
  bf=$(flags_of $BEST); : > "$DIR/s2.items"
  # every other knob's arms that ran: the grid's, and the ladders' winners
  for x in $(echo $NAMES $WIN | tr ' ' '\n' | awk '!s[$0]++'); do
    [ "$x" = default ] || [ "$x" = "$BEST" ] && continue
    awk -v a="$x" '$1 == a { f = 1 } END { exit !f }' "$DIR/ladder.pick" || continue
    shares "$bf" "$(flags_of $x)" && continue
    if ! valid_flags "$bf $(flags_of $x)"; then say "   not timing $x on top of $BEST: bench would refuse the pair"; continue; fi
    echo "$x $(knobs_of "$(flags_of $x)" | sed 's/ $//; s/ /,/g') $(awk -v a="$x" '$1 == a { print $2 }' "$DIR/ladder.pick")" >> "$DIR/s2.items"
  done
  # one line per knob, with its best gain anywhere so far, for the budget
  awk '{ c[$2]++; if (!($2 in g) || $3 > g[$2]) g[$2] = $3 } END { for (k in c) print g[k], c[k], k }' "$DIR/s2.items" > "$DIR/s2.knobs"
  trim_budget 2 "$DIR/s2.knobs"
  for x in $SKIP; do x=$(echo $x | tr , /); say "   stage 2: $x not timed on top of $BEST, for the -t budget"; SKIPPED="$SKIPPED $x"; done
  XS=$(awk -v keep=" $KEEP " 'index(keep, " " $2 " ") { print $1 }' "$DIR/s2.items")
  if [ -n "$XS" ]; then
    names2="default $BEST"
    for x in $XS; do ARMS="$ARMS $BEST+$x=$(echo $bf $(flags_of $x) | tr ' ' ,)"; names2="$names2 $BEST+$x"; done
    nx=$(echo $XS | wc -w); say "== stage 2: $nx arm$([ $nx -eq 1 ] || echo s) on top of $BEST ($bf), $(date +%T)"
    DROP=$BEST; run_stage 2 "$names2" "$@"
    if [ "$CUT" = 2 ]; then say "   the -t budget ran out in stage 2; its runs are not used, and $BEST stands"
    else
      summarise 2 "$BEST"
      sed "s/^$BEST+//" "$DIR/s2.pick" > "$DIR/s2.x.pick"
      W2=; [ $fails -eq 0 ] && W2=$(pick_wins "$DIR/s2.x.pick")
      if [ $fails -ne 0 ]; then say "   a run failed or differed in stage 2; the search stops here"
      elif [ -z "$W2" ]; then say "   nothing beat $BEST clearly on top of it; $BEST stands"
      else
        first=$(echo $W2 | awk '{ print $1 }'); CUR="$BEST+$first"
        say "   on top of $BEST, $(echo $W2 | sed 's/ /, /g') beat it; $CUR is the best so far"
        j=1; dcut=
        for y in $(echo $W2 | awk '{ for (i = 2; i <= NF; i++) print $i }'); do
          j=$((j + 1)); cf="$(flags_of $CUR) $(flags_of $y)"
          # named by its knobs, as the other cuts name theirs
          yk=$(knobs_of "$(flags_of $y)" | sed 's/ $//; s/ /\//g')
          if ! valid_flags "$cf"; then say "   not adding $y to $CUR: bench would refuse the pair"; continue; fi
          if [ -n "$dcut" ] || ! fits 3; then say "   not adding $y to $CUR, for the -t budget"; SKIPPED="$SKIPPED $yk"; continue; fi
          ARMS="$ARMS $CUR+$y=$(echo $cf | tr ' ' ,)"
          say "== stage 2.$j: adding $y to $CUR, $(date +%T)"
          DROP=$CUR; run_stage 2a$j "default $CUR $CUR+$y" "$@"
          if [ "$CUT" = "2a$j" ]; then say "   the -t budget ran out; $CUR stands"; SKIPPED="$SKIPPED $yk"; dcut=1; continue; fi
          summarise 2a$j "$CUR"
          if [ $fails -ne 0 ]; then say "   a run failed or differed; the search stops here"; break; fi
          if awk -v a="$CUR+$y" '$1 == a && $3 == 1 { f = 1 } END { exit !f }' "$DIR/s2a$j.pick"; then
            CUR="$CUR+$y"; say "   $y kept: $CUR is the best so far"
          else say "   $y did not beat $CUR clearly; left out"; fi
        done
      fi
    fi
  fi
fi
[ -n "$CUR" ] && REC=$(flags_of "$CUR")
# --slab-fill-blocks alone ran with the slab fill ahead on by this card's
# default (bench refuses it where that is off); a saved entry must say so
case " $REC " in
  *" --slab-fill-blocks "*) case " $REC " in *" --slab-fill-ahead "*) ;; *) REC="--slab-fill-ahead $REC" ;; esac ;;
esac
# Saving. Only knobs the file may carry (bench/tuned.h's allowlist) can be
# written; an -a arm may set others.
knobs_only() {
  for f in $1; do
    case $f in
      --fill-blocks|--ss-block-cut|--ss-warp-cut|--fill-gate-shift|--slab-fill-blocks|--no-fill-merge|--slab-fill-ahead|--no-slab-fill-ahead) ;;
      --*) return 1 ;;
    esac
  done
}
# why nothing is saved (empty: the confirmation stage runs, then the rules
# in the header decide); reserve() above holds no time back in the cases a
# first run settles
REASON=
if [ -n "$NOWRITE" ]; then REASON="-n"
elif [ -n "$ARMS_GIVEN" ] && [ -z "$WGIVEN" ]; then REASON="an -a experiment saves only with -w FILE"
elif [ $fails -ne 0 ]; then REASON="runs failed or differed"
elif [ -n "$NOSAVE" ]; then REASON=$NOSAVE
elif [ -n "$REC" ] && ! knobs_only "$REC"; then REASON="$REC sets a flag the saved-tuning file may not carry"
fi

say "== result $(date +%T)"
if [ $fails -ne 0 ]; then say "   $fails runs failed or had different relations; do not use this result (see above)"
elif [ -n "$REC" ]; then
  say "   recommended for this card and job: $REC"
  # stage 2 judged each part against the setting before it; only the
  # confirmation stage times the whole against the defaults
  case $CUR in *+*) [ -n "$REASON" ] && say "   (not confirmed against the defaults as a whole: there is no confirmation stage, since this tune does not save)" ;; esac
else say "   keep the defaults: no setting beat them clearly on this card and job"; fi
[ -n "$SKIPPED" ] && say "   not searched, for the -t budget:$(echo " $SKIPPED" | tr ' ' '\n' | awk 'NF && !s[$0]++ { printf " %s", $0 }')"

# canon FLAGS: flag/value pairs one per line, sorted, so two orders compare equal
canon() { printf '%s\n' "$1" | awk '{ for (i = 1; i <= NF; i++) if (i < NF && $(i + 1) !~ /^--/) { print $i, $(i + 1); i++ } else print $i }' | sort | tr '\n' ' '; }
# write_entry FLAGS NOTE: replace this card's entry at this logI in $TPATH,
# atomically (a temp file beside it, then mv)
write_entry() {
  line="$(printf '%s\n' "$TR" | awk -F'|' -v ns="$TR_NSLAB" -v fl="$1" -v nt="$2" '{ printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n", $1, $2, $3, $4, $5, $6, ns, $7, $8, $9, $10, fl, nt }')"
  d=$(dirname "$TPATH")
  mkdir -p "$d" || return 1
  tmp="$TPATH.tune.$$"
  {
    if [ -s "$TPATH" ]; then
      # every line but this key's (fields 1-4, trimmed as bench trims them)
      awk -F'|' -v key="$(printf '%s\n' "$TR" | awk -F'|' '{ print $1 "|" $2 "|" $3 "|" $4 }')" '
        function t(s) { gsub(/^[ \t]+|[ \t\r]+$/, "", s); return s }
        NF >= 13 && $0 !~ /^[ \t]*#/ && t($1) "|" t($2) "|" t($3) "|" t($4) == key { next }
        { print }' "$TPATH"
    else
      echo "# cuda-sieve tuned v1"
      echo "# Saved tuning, written by bench/tune.sh and read by bench (bench --help: --tuned)."
      echo "# card|cc|SMs|logI|J|region|nslab|side|alim|rlim|policy|flags|note"
    fi
    printf '%s\n' "$line"
  } > "$tmp" && mv "$tmp" "$TPATH" || { rm -f "$tmp"; return 1; }
  say "   wrote to $TPATH:"
  say "   $line"
}
if [ -n "$REASON" ]; then
  say "== not saving: $REASON"
else
  # saved: bench applies an entry (TR_STATUS apply) for this card and logI;
  # SAVED is what of it would apply, empty when that is the defaults.
  HAVE_SAVED=; [ "$TR_STATUS" = apply ] && HAVE_SAVED=1
  [ -n "$REC" ] && [ -n "$SAVED" ] && [ "$(canon "$REC")" = "$(canon "$SAVED")" ] && SAVED_IS_CAND=1 || SAVED_IS_CAND=
  names="default"
  [ -n "$REC" ] && { ARMS="$ARMS cand=$(echo $REC | tr ' ' ,)"; names="$names cand"; }
  [ -n "$SAVED" ] && [ -z "$SAVED_IS_CAND" ] && { ARMS="$ARMS saved=$(echo $SAVED | tr ' ' ,)"; names="$names saved"; }
  # the entry's note: date, commit (whole, as "57e9d88d-dirty"), job, band
  band=$(printf '%s\n' "$*" | sed -n 's/.*--qrange \([^ ]*\).*/q \1/p')$(printf '%s\n' "$*" | sed -n 's/.*--nq \([^ ]*\).*/ nq \1/p')
  note="tune.sh $(date +%Y-%m-%d) $(sed -n 's/^# commit *//p' "$DIR/s1.default.1.log" | head -1 | awk '{ print $1 }') $(basename "$(printf '%s\n' "$*" | sed -n 's/.*--poly \([^ ]*\).*/\1/p')") $band"
  if [ "$names" = default ]; then
    # nothing to time: no recommendation, and saved is the defaults or absent
    if [ -z "$HAVE_SAVED" ]; then
      write_entry "" "$note, the defaults won" || { say "   writing $TPATH failed"; exit 1; }
    else say "== saving: nothing to change; the saved entry already says the defaults"; fi
  else
    say "== confirmation: $names, $(date +%T)"
    DROP=; run_stage 3 "$names" "$@"
    if [ "$CUT" = 3 ]; then say "== not saving: the -t budget ran out in the confirmation stage; ${TPATH:-the file} is unchanged"; exit 4; fi
    [ $fails -eq 0 ] || { say "== not saving: the confirmation stage had failed runs"; exit 1; }
    summarise 3
    vc=; vs=; vd=
    [ -n "$REC" ] && { vc=$(beats 3 cand default); say "$vc"; }
    if [ -n "$REC" ] && [ -n "$SAVED" ] && [ -z "$SAVED_IS_CAND" ]; then vs=$(beats 3 cand saved); say "$vs"; fi
    # the defaults against saved; where saved is cand, against cand
    if [ -n "$SAVED" ]; then
      if [ -n "$SAVED_IS_CAND" ]; then vd=$(beats 3 default cand); else vd=$(beats 3 default saved); fi
      say "$vd"
    fi
    case $vc in
      *ADOPTED)
        case $vs in
          ''|*ADOPTED) g=$(printf '%s\n' "$vc" | awk '{ print $2 }')
                       write_entry "$REC" "$note $g against the defaults" || { say "   writing $TPATH failed"; exit 1; } ;;
          *) vc=no ;;
        esac ;;
      *) vc=no ;;
    esac
    if [ "$vc" = no ]; then
      case $vd in
        *ADOPTED) write_entry "" "$note, the defaults beat the saved $SAVED" || { say "   writing $TPATH failed"; exit 1; } ;;
        *) if [ -z "$HAVE_SAVED" ]; then write_entry "" "$note, the defaults won" || { say "   writing $TPATH failed"; exit 1; }
           else say "== not saving: nothing beat both the defaults and the saved entry clearly; $TPATH is unchanged"; fi ;;
      esac
    fi
  fi
fi
[ $fails -eq 0 ]
