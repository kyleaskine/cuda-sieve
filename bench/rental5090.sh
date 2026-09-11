#!/usr/bin/env bash
# rental5090.sh -- the whole wide-card protocol for item 1 / finding 94, in one run.
#
# Written to be handed to a rented box: clone, stage ../oracle/input.job, run this.
# It builds, gates output identity, then measures. Every phase writes its own
# log under OUTDIR and the summary at the end greps the numbers back out, so a
# session that gets cut short still leaves the earlier phases usable.
#
#   usage:  bench/rental5090.sh [OUTDIR] [phase ...]
#           phases: build fb ident band wide c147 streams  (refuse: opt-in)
#
# Run it from anywhere; it cd's to bench/. Expects ../oracle/{input,c147}.job.
# Card must be IDLE: a foreign process flatters the concurrent arm (finding 94).

set -u
cd "$(dirname "$0")" || exit 1   # the Makefile and the binaries live here

OUT=${1:-rental-$(date +%Y%m%d-%H%M%S)}
shift 2>/dev/null
PHASES=${*:-build fb ident band wide c147 streams}
mkdir -p "$OUT" || exit 1
echo "logs -> $OUT"

# Reusing an OUTDIR is SUPPORTED -- `rental5090.sh out build fb ident` then
# `rental5090.sh out band` is the documented way to survive a cut-short session.
# What is not supported is a summary that silently mixes two invocations, which
# is what globbing by name did on the 5090, 2026-09-10: the previous run's single
# 16e pair printed beside the new three with nothing to tell them apart. So each
# invocation records the arms IT ran, and the summary reads that instead of the
# directory. An earlier fix refused the reuse; that traded a documented
# capability for a constraint the real fix does not need.
MANIFEST="$OUT/.arms.$$"
: > "$MANIFEST"

# NQ is an override for a dry run (NQ=20 bench/rental5090.sh out band) -- the
# timing bands want the full 2000 or the boost-clock ramp dominates.
NQ=${NQ:-2000}

# Every timed arm runs unattended for minutes. A GPU that stops responding does
# NOT look like a hang from outside: CUDA's default sync policy is spin-wait, so
# the process sits in R at 100% user CPU with zero syscall time while the card
# reads idle -- which is indistinguishable from healthy compute unless you are
# watching wall-clock progress. On a 5070, 2026-09-10, that cost 57 minutes and
# a whole band arm before anyone noticed. The watchdog is OFF by default; arm it
# here, and let --watchdog-kill (600 s default) end the arm instead of the run.
# Threshold is generous against the legitimate stalls: a 16e q is ~430 ms on a
# 3090 and the end-of-band cofactor flush can run into hundreds of ms.
WD="--watchdog 120 --watchdog-log"


want() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }
run()  { local n=$1; shift; echo "== $n"; echo "\$ $*" > "$OUT/$n.log"
         "$@" >> "$OUT/$n.log" 2>&1; local rc=$?
         # rc and membership both recorded. A --watchdog-kill exit (4) leaves a
         # TRUNCATED log with no `band of` summary and a truncated .rels, so a
         # killed arm would otherwise drop out of the summary, out of the spread
         # check's count, and print a short relation count with no marking --
         # silently turning three pairs into two.
         printf '%s %d\n' "$n" "$rc" >> "$MANIFEST"
         echo "   rc=$rc  ($OUT/$n.log)"; return $rc; }

# Same, but with board power sampled at 5 Hz for the WHOLE arm and averaged.
# The runlog's own `board=` is ONE instantaneous reading per log tick, and four
# of them cannot carry an energy comparison: two identical 16e pairs on a 5090
# disagreed by 6 points of rel/J and straddled zero, purely on that sampling
# (finding 94). rel/J is the metric this project is graded on, so the timed arms
# get a real mean. Costs nothing -- nvidia-smi runs on the host.
runp() { local n=$1
         nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits \
             -lms 200 > "$OUT/$n.pw" 2>/dev/null &
         local pw=$!
         shift; run "$n" "$@"; local rc=$?
         kill $pw 2>/dev/null; wait $pw 2>/dev/null
         awk '{s+=$1;n++} END{if(n)printf "   board %.1f W mean, %d samples\n",s/n,n}' \
             "$OUT/$n.pw"
         return $rc; }

# The band arms. Both write relations so the band itself is an identity gate as
# well as a timing run -- the 500-q gate below is only the cheap early abort.
BAND="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 \
      --logI 15 --J 16384 --qrange 190000000: --nq $NQ --restart"

# ---------------------------------------------------------------- environment
{ date -u; echo; git -C .. rev-parse HEAD; git -C .. status --short; echo
  nvidia-smi; echo; nvcc --version; echo; nproc; free -g; } > "$OUT/00-env.log" 2>&1
echo "== 00-env  ($OUT/00-env.log)"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader

# --------------------------------------------------------------------- build
# GPU_ARCH=native is sm_120 on a 5090 and takes ~5 min; that is expected.
# CF_LMAX=3 is valid for c183 (mfba 92 < 96) and cannot touch k_fill_atomic.
if want build; then
    run 01-build make GPU_ARCH=native CF_LMAX=3 -j"$(nproc)" bench || exit 1
    run 02-fbgen make fbgen || exit 1
fi

# ------------------------------------------------------------- factor base
# Generated rather than copied: ../oracle/c183.fb1 is 115 MB and not in git. Same
# --maxbits 15 as finding 84, so the fill work matches the existing 5090 row.
if want fb && [ ! -s ../oracle/c183.fb1 ]; then
    run 03-fb ./fbgen --poly ../oracle/input.job --maxbits 15 \
        --threads "$(nproc)" --out ../oracle/c183.fb1 || exit 1
fi

# ------------------------------------------------------------ identity gate
# 500 q, both arms, byte-compare. THIS IS THE ABORT: if the arms differ, every
# number after it is meaningless and the run stops here.
if want ident; then
    # The gate runs FIRST and unattended, and everything after it depends on it.
    # A card that wedges here hangs the session with no report at all.
    G="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 \
       --logI 15 --J 16384 --qrange 120000000:120000500 --restart $WD $OUT/id.wd"
    run 10-ident-serial     ./bench $G --relations "$OUT/id.s.rels"
    run 11-ident-concurrent ./bench $G --relations "$OUT/id.c.rels" --fill-concurrent
    # And the slabbed path, where walk-continuation state is advanced per slab.
    S="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 \
       --logI 15 --J 32768 --qrange 120000000:120000200 --restart $WD $OUT/sl.wd"
    run 12-slab-serial      ./bench $S --relations "$OUT/sl.s.rels"
    run 13-slab-concurrent  ./bench $S --relations "$OUT/sl.c.rels" --fill-concurrent
    # Host-only reconstruction gate: every factor divides, is prime, is under
    # lpb, gcd(a,b)==1, and both norms rebuild to exactly 1.
    run 14-checkrel ./bench --poly ../oracle/input.job \
        --check-relations "$OUT/id.c.rels"
    # Two empty files have the SAME md5, so a bench that creates the relations
    # file and then dies would have reported IDENTITY OK and sent the session on
    # to ~35 minutes of timing arms with no correctness gate behind them. Require
    # a nonzero count on both sides, and the reference count where we have one.
    ok=1
    set -- id 1591 sl 937
    while [ $# -ge 2 ]; do
        p=$1; ref=$2; shift 2
        a=$(md5sum < "$OUT/$p.s.rels" 2>/dev/null | cut -d' ' -f1)
        b=$(md5sum < "$OUT/$p.c.rels" 2>/dev/null | cut -d' ' -f1)
        na=$(wc -l < "$OUT/$p.s.rels" 2>/dev/null || echo 0)
        nb=$(wc -l < "$OUT/$p.c.rels" 2>/dev/null || echo 0)
        if [ "${na:-0}" -lt 1 ] || [ "${nb:-0}" -lt 1 ]; then
            echo "   IDENTITY $p ***FAIL*** empty or missing ($na / $nb relations)"; ok=0
        elif [ "$a" != "$b" ]; then
            echo "   IDENTITY $p ***FAIL***  $a vs $b"; ok=0
        else
            echo "   IDENTITY $p OK  $a  ($na relations, 5070 gave $ref)"
            [ "$na" = "$ref" ] || echo "      note: count differs from the 5070 reference"
        fi
    done
    # Cross-card md5, RTX 5070, these exact commands, 2026-09-10. A mismatch is
    # NOT a failure -- cross-card relation identity has never been gated and this
    # is the first run that would show it -- but it is worth knowing which it is.
    echo "   5070 md5s: id 6e33c6b84bce325ddce1cffa2eebc53b"
    echo "              sl 1604756a33f030a6d3e48c98511cf8af"
    [ "$ok" = 1 ] || { echo "STOPPING: arms disagree"; exit 1; }
fi

# ------------------------------------------------------------- the headline
# Three INTERLEAVED pairs, each arm best-of-3 by pair. Interleaving is not
# optional: finding 89 showed un-interleaved A/B doubled an apparent win, and
# finding 84 lost a full point to boost-clock decay across a fixed arm order.
if want band; then
    # Within a pair the arms alternate too: the first arm of a pass runs at the
    # highest boost clocks, which is worth about a point (finding 84).
    for p in 1 2 3; do
        S_ARM=(runp "20-band-serial-$p"     ./bench $BAND --relations "$OUT/b.s.$p.rels"
               --log "$OUT/b.s.$p.log" --log-every 20 $WD "$OUT/b.s.$p.wd")
        C_ARM=(runp "21-band-concurrent-$p" ./bench $BAND --relations "$OUT/b.c.$p.rels"
               --log "$OUT/b.c.$p.log" --log-every 20 $WD "$OUT/b.c.$p.wd" --fill-concurrent)
        if [ $((p % 2)) = 1 ]; then "${S_ARM[@]}"; "${C_ARM[@]}"
        else                        "${C_ARM[@]}"; "${S_ARM[@]}"; fi
    done
fi

# ------------------------------------------------ the production geometry (16e)
# c183 at I16/J32768 is the shape a real job runs and the one where the flag's
# memory cost is largest -- roughly 2.7 GB of bucket array, so 5.4 GB for the
# pair. A 12 GB card cannot hold that, which is why this arm needs the rented
# one; it is also where the "second bucket array" line in the by-stage memory
# table should be read off at production size. Fewer q: they are ~4x slower.
if want wide; then
    NQW=$(( NQ/4 > 0 ? NQ/4 : 1 ))     # --nq 0 is refused by the parser
    W="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 \
       --logI 16 --J 32768 --qrange 190000000: --nq $NQW --restart"
    # THREE interleaved pairs, not one. The 2026-09-10 5090 session ran a single
    # pair here and it returned the only NEGATIVE rel/J in the whole run (-1.4%,
    # board +7.3% against wall -5.5%) -- the row that decides deployment, on the
    # production geometry, measured once. One pair cannot carry that.
    for p in 1 2 3; do
        S_ARM=(runp "25-wide-serial-$p"     ./bench $W --relations "$OUT/w.s.$p.rels"
               --log "$OUT/w.s.$p.log" --log-every 5 $WD "$OUT/w.s.$p.wd")
        C_ARM=(runp "26-wide-concurrent-$p" ./bench $W --relations "$OUT/w.c.$p.rels"
               --log "$OUT/w.c.$p.log" --log-every 5 $WD "$OUT/w.c.$p.wd" --fill-concurrent)
        if [ $((p % 2)) = 1 ]; then "${S_ARM[@]}"; "${C_ARM[@]}"
        else                        "${C_ARM[@]}"; "${S_ARM[@]}"; fi
    done
    grep -h "second bucket array\|device memory, steady state\|REFUS\|refus" \
        "$OUT/26-wide-concurrent-1.log" | sed 's/^/   /'
fi

# ------------------------------------------- the geometry that should win most
# c147 at I14/J8192 is where the 5090 gave 39.7% off fill against 27.4% at
# c183 I15e: fewer regions per kernel, so one kernel underfeeds the card worse.
# If the pipeline gain tracks the synthetic gain anywhere, it is here.
if want c147; then
    C="--pipeline --cofactor --poly ../oracle/c147.job \
       --logI 14 --J 8192 --qrange 120000000: --nq $NQ --restart"
    for p in 1 2; do
        S_ARM=(runp "30-c147-serial-$p"     ./bench $C --relations "$OUT/c.s.$p.rels"
               --log "$OUT/c.s.$p.log" --log-every 5 $WD "$OUT/c.s.$p.wd")
        C_ARM=(runp "31-c147-concurrent-$p" ./bench $C --relations "$OUT/c.c.$p.rels"
               --log "$OUT/c.c.$p.log" --log-every 5 $WD "$OUT/c.c.$p.wd" --fill-concurrent)
        if [ $((p % 2)) = 1 ]; then "${S_ARM[@]}"; "${C_ARM[@]}"
        else                        "${C_ARM[@]}"; "${S_ARM[@]}"; fi
    done
fi

# ------------------------------------------------- confirm the synthetic rows
# This 5090 vs the 2026-09-01 5090: single 8.42, N=2 0.7654, N=4 0.6959.
# Cheap, and it tells us whether the box we rented matches the box we modelled.
if want streams; then
    # TWO passes. The 5070 gave concurrent/serial 0.8502 in one session and
    # 1.0392 in another on the same binary and geometry -- the ratio is sensitive
    # to boost state and to anything else on the card, and a single reading of it
    # is not a measurement (finding 84 took best-of-3 for the same reason).
    # Disagreement between the passes is the signal that the box was not idle.
    for pass in a b; do
        for N in 1 2 4 8; do
            run "40-streams-$N$pass" ./bench --poly ../oracle/input.job \
                --fb1 ../oracle/c183.fb1 --watchdog 120 \
                --logI 15 --J 16384 --reps 20 --fill-streams $N
        done
    done
fi

# ------------------------------------ the refusal branch (DISCHARGED, opt-in)
# NOT in the default phase list: this was fired on a 12 GB RTX 5070 on
# 2026-09-10 and needs no card-hours. Kept because it is the reproducer.
#
#   ./bench --pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 \
#           --logI 16 --J 32768 --region 15 --slab-j 32768 --qrange 120000000: --nq 1
#     serial     -> runs, steady state 10.50 GB of 11.91 GB
#     concurrent -> "needs a second bucket array + cursors of 4.87 GB and only
#                    1.43 GB is free after setup"  <-- the branch, at startup
#
# --slab-j is the knob, NOT --logI: the auto-slabber targets ~2^29 positions per
# slab, so raising logI SHRINKS the slab and the array with it (2.43 -> 2.16 ->
# 1.90 GB across logI 16/17/18, measured). And --region is capped by shared
# memory, not VRAM -- see the region-16 note in finding 94.
#
# On a BIG card this cannot fire on c183: the array scales with slab area, area
# is capped at 2^31 positions by the uint32 offsets, and that ceiling puts the
# array at ~4.9 GB. Two of those is nothing to a 32 GB card. Firing it there
# needs a job with a much larger factor base, not a larger geometry -- which is
# why this phase is off by default and why the 12 GB box was the right one.
if want refuse; then
    for SJ in 16384 24576 32768; do
        R_BASE="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 \
                --logI 16 --J 32768 --region 15 --slab-j $SJ --qrange 120000000: --nq 1 --restart"
        run "50-refuse-sj$SJ-serial"     ./bench $R_BASE
        run "51-refuse-sj$SJ-concurrent" ./bench $R_BASE --fill-concurrent
    done
    echo "   --- refusal ladder ---"
    for SJ in 16384 24576 32768; do
        sl="$OUT/50-refuse-sj$SJ-serial.log"; cl="$OUT/51-refuse-sj$SJ-concurrent.log"
        arr=$(grep -m1 -o "= [0-9.]* GB, shared" "$sl" 2>/dev/null | grep -o "[0-9.]*")
        sr=$(grep -q "band of" "$sl" && echo ran || echo no)
        cr=$(grep -q "band of" "$cl" && echo ran || echo no)
        # Matched separately on purpose: "second bucket array" alone also matches
        # the SUCCESS banner, "does not fit" also matches the FIRST array's
        # refusal -- the very failure this ladder must tell apart -- and the
        # shared-memory refusal is neither and would read as "no message".
        msg=$(grep -m1 "fill-concurrent needs a second bucket array" "$cl" 2>/dev/null)
        [ -n "$msg" ] || msg=$(grep -m1 "bucket array does not fit" "$cl" 2>/dev/null)
        [ -n "$msg" ] || msg=$(grep -m1 "shared memory" "$cl" 2>/dev/null)
        [ -n "$msg" ] || msg="(concurrent ran)"
        printf '   slab-j %-6s array %-6s GB  serial=%-3s concurrent=%-3s\n      %s\n' \
            "$SJ" "${arr:-?}" "$sr" "$cr" "$msg"
    done
    echo "   PASS = a rung with serial=ran, concurrent=no, and the second-array"
    echo "   refusal message naming both figures."
fi

# ------------------------------------------------------------------- summary
echo
echo "=============================== SUMMARY ==============================="
# ONE parser, feeding both the per-arm rows and the spread check below. The two
# used to carry independent copies of the same `wall clock per q` pattern, so a
# change to that label would have broken one and left the other quietly matching.
TAB="$OUT/.summary.$$"; : > "$TAB"
while read -r n rc; do
    case "$n" in 2*|3*) ;; *) continue;; esac
    f="$OUT/$n.log"
    pw=$(awk '{s+=$1;c++} END{if(c)printf "%.1f",s/c}' "$OUT/$n.pw" 2>/dev/null)
    printf '%-26s ' "$n"
    if [ "$rc" != 0 ] || ! grep -q "band of" "$f" 2>/dev/null; then
        # A killed or crashed arm is REPORTED, not skipped. rc 4 is the watchdog
        # kill; it exits via a bare _exit() from the watchdog thread with no
        # stdio flush, so the log is truncated with no `band of` summary and the
        # .rels is short. Globbing for results made such an arm vanish -- three
        # pairs quietly became two, and the relation-count list below printed a
        # short count and a different md5 with nothing to mark it.
        extra=""
        [ "$rc" = 4 ] && extra=", WATCHDOG KILL -- see $n.wd"
        printf '*** NO RESULT (rc=%s%s), excluded from the spread check\n' "$rc" "$extra"
        continue
    fi
    awk -v pw="${pw:-}" -v nm="$n" -v tab="$TAB" \
        'function v(  i){for(i=1;i<=NF;i++) if($i=="ms") return $(i-1); return ""}
         /^  wall clock per q  /            {w=v()}
         /^  wall clock per q, COMPLETE/    {W=v()}
         /^    sieve, both sides/           {s=v()}
         /^      fill /                     {f=v()}
         /^      apply /                    {a=v()}
         /^      less: sides overlapped/    {o=v()}
         /^  GPU-accounted . wall/          {g=$NF}
         /^  ALL RELATIONS.q/               {r=$NF}
         END{printf "wall %8s cmplt %8s sieve %8s fill %8s apply %8s ovl %9s acc %5s rel/q %6s",
                    w,W,s,f,a,(o==""?"-":o),g,r
             if(pw!="" && w!="" && r!="")
                 printf "  board %6.1fW J/q %7.3f rel/J %6.3f", pw, w/1000*pw,
                        r/(w/1000*pw)
             printf "\n"
             printf "%s %s %s %s\n", nm, w, (g==""?"-":g), (pw==""?"-":pw) >> tab}' "$f"
done < "$MANIFEST"

echo
# Spread WITHIN one arm type. Interleaving cancels a monotonic drift such as
# boost decay; it does not cancel a burst of load landing on one arm.
#
# There is deliberately NO pass/fail verdict. The clean runs measured on this
# project span 0.03% (3090) to 2.62% (a 5070 concurrent group whose outlier arm
# had a HIGHER acc than its siblings -- a quieter host, not a worse one), so no
# single constant separates clean from dirty; pipeline.cuh refuses to hardcode a
# comparable "good" constant for exactly that reason. Printed instead is the
# spread beside the two columns that identify the MECHANISM, which wall clock
# alone cannot: acc falls when host time appears with the GPU idle, and board
# falls when the device is starved rather than throttled. Wall up with watts DOWN
# is a starved GPU; wall up with watts at the limit is thermal. Compare against
# your own idle baseline on the same card, job and band length (finding 53).
echo "spread within each arm type, beside the columns that identify the cause:"
awk '{g=$1; sub(/-[0-9]+$/,"",g)
      if($2+0>0){ if(!(g in lo)||$2+0<lo[g])lo[g]=$2+0
                  if($2+0>hi[g])hi[g]=$2+0; n[g]++ }
      else bad[g]++
      if($3!="-")ac[g]=ac[g]" "$3
      if($4!="-")bd[g]=bd[g]" "$4}
     END{for(g in n){
           if(n[g]<2){printf "  %-22s %d usable arm(s), no spread\n",g,n[g];continue}
           if(lo[g]<=0){printf "  %-22s unparseable wall figures\n",g;continue}
           printf "  %-22s %7.2f -%7.2f ms  spread %5.2f%%\n      acc%s\n      board%s\n",
                  g,lo[g],hi[g],100*(hi[g]/lo[g]-1),ac[g],bd[g]}
         for(g in bad) if(!(g in n)) printf "  %-22s no arm produced a summary\n",g}' "$TAB"
rm -f "$TAB"

echo "power, from the --log sidecars (board= is a SPOT SAMPLE, not an"
echo "integrated measurement -- treat rel/J here as indicative):"
for f in "$OUT"/b.?.?.log "$OUT"/w.?.?.log "$OUT"/c.?.?.log; do
    [ -e "$f" ] || continue
    printf '  %-16s ' "$(basename "$f" .log)"
    awk '{for(i=1;i<=NF;i++){if($i~/^rel\/s=/){r=substr($i,7)}
                             if($i~/^board=/){b=substr($i,7);sub(/W$/,"",b)}}
          if(r+0>0&&b+0>0){R+=r;B+=b;n++}}
         END{if(n)printf "rel/s %7.1f  board %6.1fW  rel/J %6.2f  (%d samples)\n",
             R/n,B/n,(R/n)/(B/n),n; else print "no samples"}' "$f"
done

echo
echo "relation counts -- every arm of one geometry must agree:"
# Marked, not merely listed. A watchdog-killed arm leaves a TRUNCATED .rels, so
# its count is short and its md5 differs; printing that in a bare list next to
# five correct ones is exactly how a poisoned run gets quoted.
for pfx in b w c; do
    set -- "$OUT/$pfx".*.rels; [ -e "$1" ] || continue
    ref=""
    for f in "$@"; do
        h=$(md5sum < "$f" | cut -c1-12); nl=$(wc -l < "$f")
        [ -n "$ref" ] || { ref=$h; refn=$nl; }
        if [ "$h" = "$ref" ]; then mark="   "; else mark="***"; fi
        printf '  %s %-24s %8s  %s\n' "$mark" "$(basename "$f")" "$nl" "$h"
    done
    for f in "$@"; do
        [ "$(md5sum < "$f" | cut -c1-12)" = "$ref" ] || {
            echo "     *** ARMS DISAGREE for '$pfx' -- this run is not usable"; break; }
    done
done
echo "======================================================================="
