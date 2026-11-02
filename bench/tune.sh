#!/bin/sh
# tune.sh -- time the speed-only knobs on THIS card for ONE job, and report
# which settings beat the defaults. Every knob it tries changes no output, and
# every run's relations are compared byte for byte with the defaults' run.
#
#   sh tune.sh [-b BENCH] [-r ROUNDS] [-o DIR] [-a "ARMS"] -- BENCH-ARGS...
#
# BENCH-ARGS is a short band of the job to tune for, as bench takes it,
# without --relations or --log (tune.sh adds its own), for example
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
#
# The grid, one knob at a time away from the defaults (bench --help):
#   --ss-block-cut 40 48 64              default 36; 33 is the floor, and
#                                        33-35 split as 36 does
#   --ss-warp-cut 128 384 512 1024       default 256
#   --fill-blocks 1152 2304 9216 18432 36864   default 4608
#   --fill-gate-shift 1 2 4              default 3
#   --no-fill-merge                      default on with the row tier, at
#                                        one slab per q; not timed elsewhere
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
# times the combination against the defaults under the same rule; if the
# combination does not hold up, the single winner with the largest gain is
# recommended instead.
#
# The report goes to stdout and to DIR/report.txt, with the commit and the
# card taken from the defaults' first run log (s1.default.1.log). Exits 0
# when done, 1 if a run failed or its relations differed, 2 if it refused to
# start, and 3 if the card became busy partway through.
BIN=./bench; ROUNDS=6; DIR=; ARMS=
while [ $# -gt 0 ]; do
  case $1 in
    -b) BIN=$2; shift 2 ;;
    -r) ROUNDS=$2; shift 2 ;;
    -o) DIR=$2; shift 2 ;;
    -a) ARMS=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "tune: unknown option $1 (bench's arguments go after --)"; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "tune: no bench arguments after --"; exit 2; }
for a in "$@"; do
  case $a in --relations|--log) echo "tune: leave $a out; tune.sh sets it"; exit 2 ;; esac
done
[ -x "$BIN" ] || { echo "tune: $BIN is not an executable"; exit 2; }
case $ROUNDS in ''|*[!0-9]*) echo "tune: -r needs a number"; exit 2 ;; esac
[ "$ROUNDS" -ge 3 ] || { echo "tune: -r must be at least 3"; exit 2; }
ARMS_GIVEN=$ARMS
[ -n "$ARMS" ] || ARMS="B40=--ss-block-cut,40 B48=--ss-block-cut,48 B64=--ss-block-cut,64
  W128=--ss-warp-cut,128 W384=--ss-warp-cut,384 W512=--ss-warp-cut,512 W1024=--ss-warp-cut,1024
  F1152=--fill-blocks,1152 F2304=--fill-blocks,2304 F9216=--fill-blocks,9216 F18432=--fill-blocks,18432
  F36864=--fill-blocks,36864 G1=--fill-gate-shift,1 G2=--fill-gate-shift,2 G4=--fill-gate-shift,4 M0=--no-fill-merge"
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
      *) keep="$keep $a" ;;
    esac
  done
  ARMS=$(echo $keep)
fi
seen=" "
for a in $ARMS; do
  case $a in *=--*) ;; *) echo "tune: arm '$a' is not NAME=FLAGS"; exit 2 ;; esac
  nm=${a%%=*}
  case $nm in default|combined|*[!A-Za-z0-9_]*|'') echo "tune: arm name '$nm' must be letters, digits or _, and not 'default' or 'combined'"; exit 2 ;; esac
  case $seen in *" $nm "*) echo "tune: arm name '$nm' is used twice"; exit 2 ;; esac
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
      "$BIN" "$@" $(flags_of $arm) --relations "$o.rel" --log "$o.log" > "$o.out" 2>&1 &
      pid=$!; i=0
      while kill -0 $pid 2>/dev/null; do
        sleep 1; i=$((i + 1))
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
        say "   one run takes about $((t1 - t0)) s; stage 1 is $(( (t1 - t0) * n * ROUNDS / 60 + 1 )) min"
        # --no-fill-merge on a plan with no merge would time the defaults
        # against themselves, and noise could adopt a flag that does nothing.
        if ! grep -q -- '--fill-merge: .*one launch' "$o.out"; then
          for a in $ARMS; do [ "${a#*=}" = --no-fill-merge ] && NOOP="$NOOP ${a%%=*}"; done
          [ -n "$NOOP" ] && say "   not timing$NOOP: this plan has no merge to turn off"
        fi
      else
        if [ $rc -ne 0 ] || [ -z "$w" ]; then
          say "   stage $st $arm round $rd: bench exited $rc (see $o.out)"; fails=$((fails+1)); continue
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

# Each knob's winner: the adopted arm with the largest gain, by flag name;
# BEST is the largest gain of all.
WIN=$(awk '$3 == 1 { print }' "$DIR/s1.pick" 2>/dev/null | while read m g ok; do
        printf '%s %s %s\n' "$(flags_of $m | cut -d' ' -f1)" "$g" "$m"; done |
      sort -k1,1 -k2,2gr | awk '!seen[$1]++ { print $3 }')
BEST=$(awk '$3 == 1 { print }' "$DIR/s1.pick" 2>/dev/null | sort -k2,2gr | awk 'NR == 1 { print $1 }')
COMBO=""; for m in $WIN; do COMBO="$COMBO $(flags_of $m)"; done; COMBO=$(echo $COMBO)
nwin=$(echo $WIN | wc -w)
if [ "$nwin" -ge 2 ]; then
  ARMS="$ARMS combined=$(echo $COMBO | tr ' ' ,)"
  say "== stage 2: the defaults against the combination ($COMBO), $(date +%T)"
  run_stage 2 "default combined" "$@"
  summarise 2
  if grep -q '^combined .* 1$' "$DIR/s2.pick" 2>/dev/null; then REC=$COMBO
  else REC=$(flags_of $BEST); say "   the combination did not hold up; stage 1's best single setting stands"; fi
elif [ "$nwin" -eq 1 ]; then REC=$COMBO
else REC=""; fi
say "== result $(date +%T)"
if [ $fails -ne 0 ]; then say "   $fails runs failed or had different relations; do not use this result (see above)"
elif [ -n "$REC" ]; then say "   recommended for this card and job: $REC"
else say "   keep the defaults: no setting beat them clearly on this card and job"; fi
[ $fails -eq 0 ]
