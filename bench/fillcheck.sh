#!/bin/sh
# fillcheck.sh -- identity gate for --fill-rows, --resieve-rows, --fill-gate
# (with --norm-gate) and --q-pipeline.
#
# None of these changes which records the sieve needs. --fill-rows moves
# the smallest bucketed moduli to a row-banded kernel (same lattice points,
# another order inside each bucket); --fill-gate drops side-0 records only at
# positions side 1 already rejected, which k_intersect_compact discards
# anyway, and --norm-gate, on wherever the fill gate is, skips side 0's norm
# at those positions, so every arm with the gate runs both and `gate3-nonorm`
# is the fill gate alone; --resieve-rows finds the same (survivor, prime)
# pairs in another order; --q-pipeline only moves the next slab's or q's sieve
# under this one's trial division. So every arm must reproduce the old
# configuration's relation file (all of them off) byte for byte. The explicit
# arms run with the pipeline at its default (on, where VRAM allows), `single`
# turns it off, and `concurrent` excludes it.
#
# The jobs cover the shapes the kernels branch on: C143 at I13 (the row tier
# starts at q = I, one region per row) and I14 (two regions per row), c183 I15
# in one slab (the pipeline on a large factor base) and in five slabs (the row
# tier's next-slab handoff, which resieve replays, with each slab issued under
# the previous slab's TD), and c194 I16/J32768 (four slabs by default). One
# exception is legitimate and reported rather than failed: an arm that changes how many
# records are written (the gate, or another --sieve-skip level) can overflow a
# bucket array where the base did not, or fit where it did, and a skipped slab
# changes the relations. Only those arms get the exemption, and only when a
# run of the pair printed OVERFLOW; the row tier writes the same records and
# must match regardless. --fill-concurrent is skipped, not failed, on a job
# whose second bucket array does not fit.
#
# NEEDS A CARD, so this is NOT in `check`.
set -e

BENCH=./bench
[ -x "$BENCH" ] || { echo "fillcheck: no $BENCH -- run make first"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0

# name|arguments
JOBS='c143-i13|--poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 300
c143-i14|--poly ../oracle/c143.job --logI 14 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 300
c183-i15|--poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 30
c183-i15-slab|--poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 130000000: --nq 30
c194-i16|--poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 10'

# The fill arms state all three fill options, so the fill defaults cannot
# change what they test. They leave --q-pipeline at its default ON on
# purpose, and every arm's output is checked for what it actually got: the
# pipeline granted everywhere except the `single` and `concurrent` arms, and
# then really used -- at least one q issued ahead, and on a slabbed plan at
# least one slab (the band summary's `issued ahead` line). A run that silently
# fell back (VRAM, a changed default) or stopped overlapping slabs fails here
# rather than passing without testing the pipelined path. On a card too small
# for the pipeline on these jobs this gate fails by design; that card cannot
# certify it. `default` is the configuration of record (findings 107-109).
OFF='--fill-rows 0 --no-fill-gate --no-q-pipeline'
ARMS='rows4|--fill-rows 4 --no-resieve-rows --no-fill-gate
rows16|--fill-rows 16 --no-resieve-rows --no-fill-gate
rows16-bands7|--fill-rows 16 --fill-row-bands 7 --no-resieve-rows --no-fill-gate
gate3|--fill-rows 0 --fill-gate
gate3-nonorm|--fill-rows 0 --fill-gate --no-norm-gate
gate0|--fill-rows 0 --fill-gate --fill-gate-shift 0
rows16-gate3|--fill-rows 16 --no-resieve-rows --fill-gate
rows4-rs|--fill-rows 4 --resieve-rows --no-fill-gate
rows16-rs-gate3|--fill-rows 16 --resieve-rows --fill-gate
concurrent|--fill-concurrent --fill-rows 4 --resieve-rows
skip1|--sieve-skip 1 --fill-rows 4 --resieve-rows --fill-gate
single|--no-q-pipeline
default|'

echo "$JOBS" | while IFS='|' read -r job jargs; do
    # shellcheck disable=SC2086
    $BENCH --pipeline --cofactor $jargs $OFF --relations "$TMP/$job.base.rel" \
        > "$TMP/$job.base.out" 2>&1 || { echo "FAIL $job base: exit $?"; tail -5 "$TMP/$job.base.out"; exit 1; }
    nbase=$(wc -l < "$TMP/$job.base.rel")
    echo "$job: base $nbase relations"
    echo "$ARMS" | while IFS='|' read -r arm aargs; do
        # shellcheck disable=SC2086
        if ! $BENCH --pipeline --cofactor $jargs $aargs --relations "$TMP/$job.$arm.rel" \
            > "$TMP/$job.$arm.out" 2>&1; then
            if grep -q "needs a second bucket array" "$TMP/$job.$arm.out"; then
                echo "  skip $arm: the second bucket array does not fit"
                continue
            fi
            echo "FAIL $job $arm: nonzero exit"; tail -5 "$TMP/$job.$arm.out"; exit 1
        fi
        # whole tokens: --no-fill-gate must not count as the gate
        case " $aargs " in
            *" --fill-gate "*|*" --sieve-skip "*) counts=1 ;;
            *) [ "$arm" = default ] && counts=1 || counts=0 ;;
        esac
        want=1
        case "$arm" in single|concurrent) want=0 ;; esac
        if grep -q -- "--q-pipeline: " "$TMP/$job.$arm.out" &&
           ! grep -q "staying single-queued" "$TMP/$job.$arm.out"; then got=1; else got=0; fi
        if [ $got -ne $want ]; then
            echo "  FAIL $arm: q-pipeline $([ $got -eq 1 ] && echo granted || echo not granted), expected $([ $want -eq 1 ] && echo granted || echo single-queued)"
            grep -h "q-pipeline:" "$TMP/$job.$arm.out"
            exit 1
        fi
        if [ $want -eq 1 ]; then
            ah=$(grep -h "issued ahead (--q-pipeline)" "$TMP/$job.$arm.out")
            nsl=$(echo "$ah" | grep -oE '[0-9]+ slab' | grep -oE '[0-9]+')
            nqa=$(echo "$ah" | grep -oE '[0-9]+ q$' | grep -oE '[0-9]+')
            if [ "${nqa:-0}" -eq 0 ]; then
                echo "  FAIL $arm: the pipeline was granted but no q was issued ahead ($ah)"
                exit 1
            fi
            if grep -q "j-slabbing: [0-9]* slabs" "$TMP/$job.$arm.out" &&
               [ "${nsl:-0}" -eq 0 ]; then
                echo "  FAIL $arm: slabbed plan, but no slab was issued ahead ($ah)"
                exit 1
            fi
        fi
        if cmp -s "$TMP/$job.base.rel" "$TMP/$job.$arm.rel"; then
            echo "  ok   $arm"
        elif [ $counts -eq 1 ] && grep -q OVERFLOW "$TMP/$job.base.out" "$TMP/$job.$arm.out"; then
            echo "  NOTE $arm differs, and a run of the pair overflowed a bucket array:"
            grep -h OVERFLOW "$TMP/$job.base.out" "$TMP/$job.$arm.out" | head -3
        else
            echo "  FAIL $arm: $(wc -l < "$TMP/$job.$arm.rel") relations, not byte-identical"
            exit 1
        fi
    done || exit 1
done || fail=1

[ $fail -eq 0 ] && echo "fillcheck: all arms byte-identical" || echo "fillcheck: FAILED"
exit $fail
