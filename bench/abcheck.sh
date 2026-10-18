#!/bin/sh
# abcheck.sh -- base-versus-new identity gate for changes that must not change
# what the siever finds.
#
#   ./abcheck.sh BASE [NEW]      NEW defaults to ./bench
#
# BASE is a binary built from the source before the change (keep a copy of
# bench/bench before rebuilding), NEW the one after. Both run the same cases
# and every output a case names must match byte for byte:
#
#   sb    the standalone harness's survivor bitmap (--survbits) and its
#         survivor count, across k_apply's scan paths: the packed 16-bit
#         store path, the atomicOr fallback (region 4/5, logI 5), the 8-bit
#         loop, odd block sizes, both-even dropping, regions spanning rows.
#   cand  the pipeline WITHOUT --cofactor, writing --candidates and
#         --relations: every joint candidate in emission order with both
#         cofactors, (a,b) and both factor lists, plus the survivors trial
#         division already factored completely. This is the stream entering
#         cofactorisation. A change that loses a candidate the cofactoriser
#         would not have split leaves the final relations identical and still
#         fails here.
#   cof   the pipeline with --cofactor: the relation file a client returns.
#   stop  the same, stopped by --target-rels with the next q already issued
#         under --q-pipeline, so a q in flight has to be discarded. Under
#         --cofactor the count advances only when a flush of cofactorisation
#         results returns (pipeline.cuh, the --target-rels comment), so the
#         stop lands on a flush: on C143 the first returns 86,426 relations
#         at q 556, and the full-size target of 100,000 stops at a later one.
#
# The cases cover special-q on either side, one slab and several (c183
# --slab-j 4095 is five slabs, the last four rows long), I13 to I16, the
# pipeline off, --fill-concurrent (each side its own bucket array), and the
# fill and resieve row tiers off. Survivor and candidate arrays start empty
# and grow at the first slab and whenever a slab exceeds them, so every case
# exercises growth.
#
# A case FAILS when either run exits nonzero, a named output is missing (or
# empty, for the primary one), the outputs differ, or NEW reports an overflow
# or truncation BASE did not. Overflow in both is printed as a warning: the
# outputs can match while both lose records. The script exits 1 if any case
# failed, or if no case ran at all (every one skipped, or SETS/ONLY matched
# nothing), 0 otherwise. Cases whose inputs are absent (the NFS@Home job lives
# in the git-ignored work/) are SKIPPED and counted, never passed silently.
#
# BASE's outputs are cached under $ABDIR/cache/<sha256 of BASE>/, keyed by
# case and arguments, and reused only from a run that passed, so iterating on
# NEW against one BASE costs NEW's runs alone. `./abcheck.sh X X` runs one
# binary twice: a determinism check of the cases themselves.
#
# Environment:
#   SETS="sb cand cof stop"    which sets to run (default all four)
#   ONLY='c183*'               only the cases whose name matches the pattern
#   EXCEPT='*.concurrent'      leave out the cases whose name matches it
#   QUICK=1                    about a quarter of the special-q per case
#   ABDIR=../work/abcheck      outputs and the BASE cache
#   NFS_JOB, NFS_FB1           the NFS@Home I16 job (default ../work/applyprof)
#   KEEP=1                     keep NEW's outputs of passing cases
#   NEWARGS="--opt ..."        extra arguments for NEW's pipeline runs only
#                              (cand, cof, stop): drive an option BASE lacks,
#                              such as a test knob, against BASE's cached
#                              outputs
#
# NEEDS A CARD, so this is NOT in `check`. Identity does not depend on timing,
# so a busy card only makes it slower. On a free RTX 5070 the full set takes
# ~5 minutes for both binaries, about half that with BASE cached.

[ $# -ge 1 ] && [ $# -le 2 ] || { echo "usage: $0 BASE [NEW]"; exit 2; }
here=$(pwd)
BASE=$1; NEW=${2:-}
cd "$(dirname "$0")" || exit 2
case $BASE in /*) ;; *) BASE=$here/$BASE ;; esac
case $NEW in "") NEW=$(pwd)/bench ;; /*) ;; *) NEW=$here/$NEW ;; esac
for b in "$BASE" "$NEW"; do
    [ -f "$b" ] && [ -x "$b" ] || { echo "abcheck: $b is not an executable file"; exit 2; }
done

ABDIR=${ABDIR:-../work/abcheck}
SETS=${SETS:-sb cand cof stop}
ONLY=${ONLY:-*}
EXCEPT=${EXCEPT:-}
QUICK=${QUICK:-0}
NFS_JOB=${NFS_JOB:-../work/applyprof/nfs.job}
NFS_FB1=${NFS_FB1:-../work/applyprof/nfs.roots1.m16}

BSHA=$(sha256sum "$BASE" | cut -c1-16)
NSHA=$(sha256sum "$NEW" | cut -c1-16)
CACHE=$ABDIR/cache/$BSHA
RUN=$ABDIR/run
mkdir -p "${CACHE:?}" "${RUN:?}" || exit 2

echo "abcheck: BASE $BASE ($BSHA)"
echo "         NEW  $NEW ($NSHA)"
[ "$BSHA" = "$NSHA" ] && echo "         same binary: checking that the cases are deterministic"
[ -n "${NEWARGS:-}" ] && echo "         NEW's pipeline runs add: $NEWARGS"
sleep 2   # let a run that just ended leave the utilization sample
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
case $util in ''|*[!0-9]*) util=0 ;; esac
[ "$util" -gt 5 ] && echo "         note: card ${util}% busy before the first case (identity holds; it is just slower)"

ncase=0; nfail=0; nskip=0; nwarn=0

q() {  # full-size quick-size -> the special-q count for this mode
    if [ "$QUICK" = 1 ]; then echo "$2"; else echo "$1"; fi
}

# One run of one binary: run BIN KIND DIR EXTRA ARGS... -> outputs in DIR, rc
# in DIR/rc. EXTRA is word-split onto pipeline runs (NEWARGS for NEW).
run_one() {
    bin=$1; kind=$2; dir=$3; extra=$4; shift 4
    rm -rf "${dir:?}"; mkdir -p "$dir"
    # shellcheck disable=SC2086
    case $kind in
        sb)   "$bin" "$@" --reps 1 --survbits "$dir/sb.bin" > "$dir/out" 2>&1 ;;
        cand) "$bin" --pipeline "$@" $extra --candidates "$dir/cand" --relations "$dir/rel" > "$dir/out" 2>&1 ;;
        cof|stop) "$bin" --pipeline --cofactor "$@" $extra --relations "$dir/rel" > "$dir/out" 2>&1 ;;
    esac
    echo $? > "$dir/rc"
    if [ "$kind" = sb ]; then
        grep -h "survivors:" "$dir/out" | awk '{print $2}' > "$dir/surv"
    fi
}

# The outputs a kind must produce; a trailing ! means it must be non-empty
# too (a band can legitimately factor no survivor completely in TD).
outputs() {
    case $1 in
        sb)   echo "sb.bin! surv!" ;;
        cand) echo "cand! rel" ;;
        cof|stop) echo "rel!" ;;
    esac
}

# base_good DIR KIND: the run exited 0 and wrote what its kind must write
base_good() {
    [ "$(cat "$1/rc" 2>/dev/null)" = 0 ] || return 1
    for f in $(outputs "$2"); do
        case $f in *!) [ -s "$1/${f%!}" ] || return 1 ;; *) [ -f "$1/$f" ] || return 1 ;; esac
    done
    if [ "$2" = stop ]; then grep -q "target-rels .* reached" "$1/out" || return 1; fi
}

ovf_count() {
    n=$(grep -ci "overflow\|truncated\|raise TD_FMAX" "$1" 2>/dev/null)
    echo "${n:-0}"
}

# ab NAME KIND ARGS...
ab() {
    name=$1; kind=$2; shift 2
    case " $SETS " in *" $kind "*) ;; *) return ;; esac
    # shellcheck disable=SC2254
    case $name in $ONLY) ;; *) return ;; esac
    # shellcheck disable=SC2254
    [ -n "$EXCEPT" ] && case $name in $EXCEPT) return ;; esac
    ncase=$((ncase + 1))
    key=$(echo "$kind $*" | sha256sum | cut -c1-12)
    bdir=$CACHE/$name.$key
    ndir=$RUN/$name.$kind
    cached=""
    if [ -f "$bdir/ok" ]; then cached=" (base cached)"
    else
        run_one "$BASE" "$kind" "$bdir" "" "$@"
        base_good "$bdir" "$kind" && touch "$bdir/ok"
    fi
    run_one "$NEW" "$kind" "$ndir" "${NEWARGS:-}" "$@"

    why=""
    rb=$(cat "$bdir/rc"); rn=$(cat "$ndir/rc")
    [ "$rb" = 0 ] || why="base exited $rb"
    [ "$rn" = 0 ] || why="${why:+$why, }new exited $rn"
    sizes=""
    for f in $(outputs "$kind"); do
        need=0; case $f in *!) need=1; f=${f%!} ;; esac
        for d in "$bdir" "$ndir"; do
            if [ ! -f "$d/$f" ] || { [ $need = 1 ] && [ ! -s "$d/$f" ]; }; then
                case $d in "$bdir") w=base ;; *) w=new ;; esac
                why="${why:+$why, }$w has no $f"
            fi
        done
        if [ -f "$bdir/$f" ] && [ -f "$ndir/$f" ] && ! cmp -s "$bdir/$f" "$ndir/$f"; then
            why="${why:+$why, }$f differs"
        fi
        case $f in
            sb.bin) ;;
            surv) sizes="$sizes survivors $(cat "$ndir/surv" 2>/dev/null)" ;;
            *) [ -f "$ndir/$f" ] && sizes="$sizes $f $(wc -l < "$ndir/$f")" ;;
        esac
    done
    if [ "$kind" = stop ]; then
        grep -q "target-rels .* reached" "$ndir/out" || why="${why:+$why, }new did not stop at --target-rels"
    fi
    ob=$(ovf_count "$bdir/out"); on=$(ovf_count "$ndir/out")
    if [ "$on" -gt 0 ] && [ "$ob" -eq 0 ]; then
        why="${why:+$why, }new reports overflow/truncation, base does not"
    fi

    if [ -n "$why" ]; then
        nfail=$((nfail + 1))
        printf "  %-18s %-4s FAIL: %s\n" "$name" "$kind" "$why"
        echo "      outputs: base $bdir, new $ndir"
        [ -f "$bdir/ok" ] || echo "      (base not cached: it failed too)"
    else
        printf "  %-18s %-4s IDENTICAL%s%s\n" "$name" "$kind" "$sizes" "$cached"
        if [ "$on" -gt 0 ]; then
            nwarn=$((nwarn + 1))
            echo "      warning: both runs report overflow/truncation ($ob / $on lines)"
        fi
        [ "${KEEP:-0}" = 1 ] || rm -rf "${ndir:?}"
    fi
}

skip() {  # name kind reason
    case " $SETS " in *" $2 "*) ;; *) return ;; esac
    # shellcheck disable=SC2254
    case $1 in $ONLY) ;; *) return ;; esac
    # shellcheck disable=SC2254
    [ -n "$EXCEPT" ] && case $1 in $EXCEPT) return ;; esac
    ncase=$((ncase + 1)); nskip=$((nskip + 1))
    printf "  %-18s %-4s SKIPPED: %s\n" "$1" "$2" "$3"
}

C183="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1"
C143="--poly ../oracle/c143.job --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000:"
C194="--poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023:"
NFS="--poly $NFS_JOB --fb1 $NFS_FB1 --sq-side 1 --logI 16 --J 32768 --qrange 296847000:"
have_nfs=1; [ -f "$NFS_JOB" ] && [ -f "$NFS_FB1" ] || have_nfs=0

echo "survivor bitmaps (harness, c183)"
ab s1.default      sb $C183 --logI 15 --side 1
ab s0.default      sb $C183 --logI 15 --side 0
ab s1.nbe          sb $C183 --logI 15 --side 1 --not-both-even
ab s0.nbe          sb $C183 --logI 15 --side 0 --not-both-even
ab s1.region6      sb $C183 --logI 15 --side 1 --region 6 --not-both-even
ab s1.thr96        sb $C183 --logI 15 --side 1 --apply-threads 96 --not-both-even
ab s1.thr32        sb $C183 --logI 15 --side 1 --apply-threads 32
ab s1.cells8       sb $C183 --logI 15 --side 1 --cells 8 --not-both-even
ab s1.I12.r12      sb $C183 --logI 12 --J 2048 --side 1 --region 12 --not-both-even
ab s1.I12.r6       sb $C183 --logI 12 --J 2048 --side 1 --region 6 --not-both-even
ab s0.I12.r6       sb $C183 --logI 12 --J 2048 --side 0 --region 6 --not-both-even
ab s1.I12.r5       sb $C183 --logI 12 --J 2048 --side 1 --region 5 --not-both-even
ab s1.I12.r4       sb $C183 --logI 12 --J 2048 --side 1 --region 4
ab s1.I6.r6        sb $C183 --logI 6 --J 8192 --side 1 --region 6 --not-both-even
ab s1.I7.r9.nss    sb $C183 --logI 7 --J 4096 --side 1 --region 9 --no-smallsieve --not-both-even
ab s1.I6.r8.nss    sb $C183 --logI 6 --J 8192 --side 1 --region 8 --no-smallsieve --not-both-even
ab s0.I6.r8.nss    sb $C183 --logI 6 --J 8192 --side 0 --region 8 --no-smallsieve --not-both-even
ab s1.I5.r8.nss    sb $C183 --logI 5 --J 16384 --side 1 --region 8 --no-smallsieve --not-both-even

echo "candidates entering cofactorisation (pipeline, no --cofactor)"
ab c143.i13        cand $C143 --logI 13 --J 8192 --nq "$(q 1000 300)"
ab c143.i14        cand $C143 --logI 14 --J 4096 --nq "$(q 300 100)"
ab c183            cand $C183 --logI 15 --qrange 130000000: --nq "$(q 100 30)"
ab c183.slab       cand $C183 --logI 15 --slab-j 4095 --qrange 130000000: --nq "$(q 30 10)"
ab c183.sq0        cand $C183 --logI 15 --sq-side 0 --qrange 130000000: --nq "$(q 50 15)"
ab c183.single     cand $C183 --logI 15 --no-q-pipeline --qrange 130000000: --nq "$(q 30 10)"
ab c183.concurrent cand $C183 --logI 15 --fill-concurrent --qrange 130000000: --nq "$(q 30 10)"
ab c183.norows     cand $C183 --logI 15 --fill-rows 0 --no-resieve-rows --qrange 130000000: --nq "$(q 30 10)"
ab c194            cand $C194 --nq "$(q 20 6)"
if [ $have_nfs = 1 ]; then
    ab nfs         cand $NFS --nq "$(q 40 10)"
    ab nfs.single  cand $NFS --no-q-pipeline --nq "$(q 10 4)"
else
    skip nfs cand "no $NFS_JOB / $NFS_FB1"; skip nfs.single cand "no NFS job"
fi

echo "relations (pipeline --cofactor)"
ab c143            cof $C143 --logI 13 --J 8192 --nq "$(q 1000 300)"
ab c183            cof $C183 --logI 15 --qrange 130000000: --nq "$(q 200 50)"
ab c194            cof $C194 --nq "$(q 20 6)"
if [ $have_nfs = 1 ]; then ab nfs cof $NFS --nq "$(q 40 10)"
else skip nfs cof "no NFS job"; fi
ab c143.target     stop $C143 --logI 13 --J 8192 --nq 3000 --target-rels "$(q 100000 40000)"

echo "$ncase cases: $((ncase - nfail - nskip)) identical, $nfail failed, $nskip skipped, $nwarn with overflow warnings"
[ $((ncase - nskip)) -gt 0 ] || { echo "abcheck: no case ran"; exit 1; }
[ $nfail -eq 0 ]
