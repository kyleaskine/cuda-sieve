#!/bin/sh
# tune.sh -- time the speed-only knobs on THIS card for ONE job, and report
# which settings beat the defaults. Every knob it tries changes no output, and
# every run's relations are compared byte for byte with the defaults' run.
#
#   sh tune.sh [-b BENCH] [-r ROUNDS] [-o DIR] [-a "ARMS"] [-n] [-w FILE] -- BENCH-ARGS...
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
#             e.g. "B48=--ss-block-cut,48 F9216=--fill-blocks,9216"
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
#   a count of block slots per SM, so of that ladder only the grids of 6 to 24
#   blocks per SM of this card are timed (512 is 10.7 on 48 SMs), and an arm
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
# arm with the largest gain wins. If more than one knob has a winner, stage 2
# times the combination and the single winner with the largest gain beside
# the defaults, in the same rounds. The combination is recommended only if,
# under the same rule, it beats the defaults and also that single winner;
# otherwise the single winner is. (Knobs can undo each other: finding 161.)
#
# The report goes to stdout and to DIR/report.txt, with the commit and the
# card taken from the defaults' first run log (s1.default.1.log). Exits 0
# when done, 1 if a run failed or its relations differed, 2 if it refused to
# start, and 3 if the card became busy partway through.
BIN=./bench; ROUNDS=6; DIR=; ARMS=; NOWRITE=; TPATH=; WGIVEN=; NOSAVE=
while [ $# -gt 0 ]; do
  case $1 in
    -b) BIN=$2; shift 2 ;;
    -r) ROUNDS=$2; shift 2 ;;
    -o) DIR=$2; shift 2 ;;
    -a) ARMS=$2; shift 2 ;;
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
# touches the card, so only a CUDA one counts. busy PID leaves out bench's
# own process.
BUSY_DEFAULT='run-cuda-client|ggnfs-sieve-client.*(engine=cuda|cuda-bench)'
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
TR=; TR_STATUS=; SAVED=; TR_OVER=; TR_NSLAB=; NOSAVE=${NOSAVE-}
tuned_first() {
  [ -n "$NOTUNED" ] || return 0
  TR=$(sed -n 's/^tuned-run: //p' "$1" | head -1)
  if [ -z "$TR" ]; then NOSAVE="bench printed no tuned-run line"; return 0; fi
  TR_STATUS=$(printf '%s\n' "$TR" | awk -F'|' '{ print $13 }')
  SAVED=$(printf '%s\n' "$TR" | awk -F'|' '{ print $14 }')
  TR_OVER=$(printf '%s\n' "$TR" | awk -F'|' '{ print $15 }')
  p=$(printf '%s\n' "$TR" | awk -F'|' '{ s = $16; for (i = 17; i <= NF; i++) s = s "|" $i; print s }')
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
# "stage arm round wall" lines to DIR/walls.txt.
run_stage() {
  st=$1; names=$2; shift 2; n=$(echo $names | wc -w)
  for rd in $(seq 1 "$ROUNDS"); do
    k=$(( ((rd + 1) / 2 - 1) % n ))
    order="$(echo $names | tr ' ' '\n' | tail -n +$((k + 1)) | tr '\n' ' ')$(echo $names | tr ' ' '\n' | head -n $k | tr '\n' ' ')"
    [ $((rd % 2)) -eq 0 ] && order=$(echo $order | awk '{ for (i = NF; i > 0; i--) printf "%s ", $i }')
    for arm in $order; do
      case " $NOOP " in *" $arm "*) continue ;; esac
      if busy_now; then say "tune: another process came onto the card before stage $st $arm round $rd; stopping"; exit 3; fi
      o=$DIR/s$st.$arm.$rd
      # A reused DIR: an interrupted run's .rel.part and checkpoint would
      # make bench resume it and time only the rest of the band.
      rm -f "$o".*
      t0=$(date +%s)
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
      t1=$(date +%s)
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
        # grid G, or off), and the ladder's grids of 6-24 blocks per SM
        sms=$(sed -n 's/^grid: \([0-9][0-9]*\) SMs.*/\1/p; s/^grid: [0-9][0-9]* blocks on \([0-9][0-9]*\) SMs.*/\1/p' "$o.out" | head -1)
        sg=$(sed -n "s/.*--slab-fill-ahead: each later slab.* \([0-9][0-9]*\) x [0-9][0-9]* at the greatest priority.*/\1/p" "$o.out" | head -1)
        nm=; why=
        if ! grep -q -- "--q-pipeline: each slab's sieve" "$o.out"; then why="it needs a plan of several slabs per q under --q-pipeline"; fi
        for a in $ARMS; do
          m=${a%%=*}; f=${a#*=}
          case $f in
            --no-slab-fill-ahead) [ -n "$why" ] || [ -z "$sg" ] && nm="$nm $m" ;;
            --slab-fill-ahead,--slab-fill-blocks,*)
              b=${f##*,}
              if [ -n "$why" ] || [ "$b" = "$sg" ] || [ -z "$sms" ] ||
                 [ "$b" -lt $((6 * sms)) ] || [ "$b" -gt $((24 * sms)) ]; then nm="$nm $m"; fi ;;
          esac
        done
        if [ -n "$nm" ]; then
          if [ -n "$why" ]; then say "   not timing$nm: $why"
          else
            if [ -n "$sg" ]; then st_sfa="on at $sg blocks"; else st_sfa=off; fi
            say "   --slab-fill-ahead is $st_sfa in the defaults' run on this card${sms:+ ($sms SMs)}; not timing$nm (the defaults' own setting, or outside 6-24 blocks per SM)"
          fi
          NOOP="$NOOP$nm"
        fi
        nt=$n; for m in $NOOP; do nt=$((nt - 1)); done
        say "   one run takes about $((t1 - t0)) s; stage 1 is $(( (t1 - t0) * nt * ROUNDS / 60 + 1 )) min ($nt arms timed)"
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
  done
}
# summarise STAGE: per arm against default, medians over rounds of the wall
# ratio, rounds faster and tied, the default's spread, and ADOPTED or not.
# Prints "arm gain% adopted(0/1)" to DIR/s$STAGE.pick.
summarise() {
  st=$1
  awk -v st=$st -v R="$ROUNDS" -v differs="$DIR/differs.txt" -v pick="$DIR/s$st.pick" '
    function med(v, n,   i, j, t) { for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
                                    return n % 2 ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }
    BEGIN { while ((getline l < differs) > 0) { split(l, x, " "); if (x[1] == st) bad[x[2]] = 1 } }
    $1 == st { w[$2, $3] = $4; arms[$2] = 1; if (!($2 in first)) { first[$2] = ++na; ord[na] = $2 } }
    END {
      n = 0; for (r = 1; r <= R; r++) if (("default", r) in w) d[++n] = w["default", r]
      if (n < 3) { print "   too few default runs to judge"; exit }
      lo = d[1]; hi = d[1]; for (i = 1; i <= n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i] }
      dm = med(d, n); spread = (hi - lo) / dm * 100
      printf "   %-8s %9.3f ms/q (median of %d rounds), spread between rounds %.2f%%\n", "default", dm, n, spread
      for (a = 1; a <= na; a++) { m = ord[a]; if (m == "default") continue
        k = 0; f = 0; s = 0; delete x
        for (r = 1; r <= R; r++) if (("default", r) in w && (m, r) in w) { x[++k] = w[m, r] / w["default", r]; f += (x[k] < 1); s += (x[k] > 1) }
        if (!k) continue
        g = (1 - med(x, k)) * 100
        ok = (k == n && s == 0 && g > 1 && g > spread && !(m in bad))
        printf "   %-8s %+8.2f%% wall (median), faster in %d of %d rounds%s%s\n", m, g ? -g : 0, f, k, (k - f - s) ? sprintf(", %d tied", k - f - s) : "", (m in bad) ? "  RELATIONS DIFFER" : ok ? "  ADOPTED" : ""
        print m, g, ok > pick
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
: > "$DIR/walls.txt"; : > "$DIR/differs.txt"; rm -f "$DIR/s1.pick" "$DIR/s2.pick"
NAMES="default $(for a in $ARMS; do printf '%s ' "${a%%=*}"; done)"
say "== stage 1: $(echo $NAMES | wc -w) arms, $(date +%T)"
run_stage 1 "$NAMES" "$@"
summarise 1

# Each knob's winner: the adopted arms by gain, largest first, each kept only
# if it sets no flag a larger winner sets (arms that share a flag are one knob,
# and an -a arm may set several); BEST is the largest gain of all.
WIN=$(awk '$3 == 1 { print }' "$DIR/s1.pick" 2>/dev/null | sort -k2,2gr | while read m g ok; do
        printf '%s %s\n' "$m" "$(for f in $(flags_of $m); do case $f in --no-*) printf '%s ' "--${f#--no-}" ;; --*) printf '%s ' $f ;; esac; done)"; done |
      awk '{ ok = 1; for (i = 2; i <= NF; i++) if ($i in used) ok = 0
             if (ok) { for (i = 2; i <= NF; i++) used[$i] = 1; print $1 } }')
BEST=$(awk '$3 == 1 { print }' "$DIR/s1.pick" 2>/dev/null | sort -k2,2gr | awk 'NR == 1 { print $1 }')
COMBO=""; for m in $WIN; do COMBO="$COMBO $(flags_of $m)"; done; COMBO=$(echo $COMBO)
nwin=$(echo $WIN | wc -w)
if [ "$nwin" -ge 2 ]; then
  ARMS="$ARMS combined=$(echo $COMBO | tr ' ' ,)"
  # The combination must beat the defaults AND stage 1's best single setting,
  # timed in the same rounds: knobs can undo each other (finding 161: on an RTX
  # 3060, --fill-blocks 294912 alone -21.8%, with --slab-fill-ahead -14.1%).
  say "== stage 2: the combination ($COMBO) against the defaults and $BEST, $(date +%T)"
  run_stage 2 "default combined $BEST" "$@"
  summarise 2
  if grep -q '^combined .* 1$' "$DIR/s2.pick" 2>/dev/null; then
    vs=$(beats 2 combined "$BEST"); say "$vs"
    case $vs in
      *ADOPTED) REC=$COMBO ;;
      *) REC=$(flags_of $BEST); say "   the combination did not beat $BEST clearly; $BEST stands" ;;
    esac
  else REC=$(flags_of $BEST); say "   the combination did not hold up; stage 1's best single setting stands"; fi
elif [ "$nwin" -eq 1 ]; then REC=$COMBO
else REC=""; fi
# --slab-fill-blocks alone ran with the slab fill ahead on by this card's
# default (bench refuses it where that is off); a saved entry must say so
case " $REC " in
  *" --slab-fill-blocks "*) case " $REC " in *" --slab-fill-ahead "*) ;; *) REC="--slab-fill-ahead $REC" ;; esac ;;
esac
say "== result $(date +%T)"
if [ $fails -ne 0 ]; then say "   $fails runs failed or had different relations; do not use this result (see above)"
elif [ -n "$REC" ]; then say "   recommended for this card and job: $REC"
else say "   keep the defaults: no setting beat them clearly on this card and job"; fi

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
REASON=
if [ -n "$NOWRITE" ]; then REASON="-n"
elif [ -n "$ARMS_GIVEN" ] && [ -z "$WGIVEN" ]; then REASON="an -a experiment saves only with -w FILE"
elif [ $fails -ne 0 ]; then REASON="runs failed or differed"
elif [ -n "$NOSAVE" ]; then REASON=$NOSAVE
elif [ -n "$REC" ] && ! knobs_only "$REC"; then REASON="$REC sets a flag the saved-tuning file may not carry"
fi
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
    run_stage 3 "$names" "$@"
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
