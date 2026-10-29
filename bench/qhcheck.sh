#!/bin/sh
# qhcheck.sh -- gate for the cofactor queue's CONTENTS under --q-pipeline
# (findings 135 and 136).
#
# WHAT IS GUARDED. Under --q-pipeline, with inline cofactorisation and no
# candidate file, TD's record pass and k_cof_enqueue are queued ahead of the
# counter readback, on a count k_scatter_sel gates on device. Writing a
# candidate file turns that off, so abcheck's candidate cases never run it,
# and its relation cases cannot see a candidate lost or left stale that
# splits into no relation. Here COFQ_TEST_HASH makes every cofq_flush print a
# hash of every queued slot, and the band end print how many slabs were
# enqueued in TD and how many after. Each case runs pipelined (in TD wherever
# the queue has room for the slab's bound) and --no-q-pipeline (after the
# host's count, every slab), and the flush lines must match in order.
#
# FIVE TEST BUILDS of this directory's sources, made in $QHDIR, each a
# pricing build (DEFS), so never installed:
#   qh  COFQ_TEST_HASH            the cases on full-size queues
#   qf  qh + CQ_FLUSH=16384u      the queue fills mid-band, so some slabs have
#                                 the record pass queued ahead and the
#                                 enqueue after the host's count and a flush
#   cc  qh + PIPE_TD_TEST_CCAP=16 a slab with more than 16 candidates drains
#                                 the pass ahead, grows the list and records
#                                 again, and says so; run --no-td-verify,
#                                 since the first q's validation turns the
#                                 pass ahead off for its slab
#   qe  qh, k_cof_enqueue edited to drop the last of the device's count:
#       the hash must change, or this gate could not see a lost candidate
#   qm  qh, k_scatter_sel's device count edited one short: the run must
#       fail on pipe_td_perq's check of that count against the host's
# A pricing build refuses --relations, so no stop, checkpoint or
# limb-widening flush can fire: the queue's capacity and band end are the
# flush points in both runs.
#
# One more case runs qh with --no-td-ahead: identical, none enqueued in TD.
#
# The sources are what git lists here (tracked and untracked, not ignored),
# or, outside a checkout, this directory without fbase caches and build
# products. Cases whose inputs are absent (c135 and the NFS@Home I16 job live
# in the git-ignored work/) are SKIPPED and counted. Exits 1 if any check
# failed or no case ran.
#
# Environment:
#   GPU_ARCH=120             the builds' architecture
#   QHDIR=../work/qhcheck    the builds and the runs' outputs
#   QH_BUILD=0               reuse the builds already in $QHDIR
#   C135_JOB, C135_FB1       c135 (default ../work/c135)
#   NFS_JOB, NFS_FB1         the NFS@Home I16 job (default ../work/applyprof)
#
# NEEDS A CARD, so this is NOT in `check`. Identity only: a busy card makes it
# slower. On this host the five sm_120 builds take ~25 minutes, the runs ~10.

cd "$(dirname "$0")" || exit 2
ARCH=${GPU_ARCH:-120}
QHDIR=${QHDIR:-../work/qhcheck}
C135_JOB=${C135_JOB:-../work/c135/c135.job}
C135_FB1=${C135_FB1:-../work/c135/c135.roots1.m13}
NFS_JOB=${NFS_JOB:-../work/applyprof/nfs.job}
NFS_FB1=${NFS_FB1:-../work/applyprof/nfs.roots1.m16}
mkdir -p "$QHDIR/out" || exit 2
QHDIR=$(cd "$QHDIR" && pwd)
O=$QHDIR/out
ncase=0; nskip=0; nfail=0
fail() { echo "   FAIL $*"; nfail=$((nfail+1)); }

build() {  # name DEFS
    d=$QHDIR/$1/bench
    rm -rf "${QHDIR:?}/$1" && mkdir -p "$d" || { fail "$1: could not make $d"; return; }
    if [ -n "$(git ls-files -co --exclude-standard 2>/dev/null | head -1)" ]; then
        git ls-files -z -co --exclude-standard | tar --null -T - --ignore-failed-read -cf - | tar -xf - -C "$d"
    else
        tar --exclude='./fbase*' --exclude='*.o' --exclude='*.d' -cf - . | tar -xf - -C "$d"
    fi || { fail "$1: copying the sources"; return; }
    case $1 in
    qm) sed -i 's/^\(                \*spec = ((\*tdflags & TDF_REFUSE) || na > cap) ? 0u : \)na;$/\1(na ? na - 1u : 0u);/' "$d/td.cuh"
        [ "$(grep -c '? 0u : (na ? na - 1u : 0u);' "$d/td.cuh")" -eq 1 ] ||
            { fail "qm: the one-short edit did not land in td.cuh"; return; } ;;
    qe) sed -i 's/^    if (dn) n = min(n, \*dn);$/    if (dn) n = min(n, *dn) ? min(n, *dn) - 1u : 0u;/' "$d/cofac.cuh"
        [ "$(grep -c 'if (dn) n = min(n, \*dn) ? min(n, \*dn) - 1u : 0u;' "$d/cofac.cuh")" -eq 1 ] ||
            { fail "qe: the one-short edit did not land in cofac.cuh"; return; } ;;
    esac
    echo "building $1 (DEFS=$2)"
    # never a make outside $d: a failed cd must not rebuild this directory
    (cd "$d" || exit 1
     make clean > /dev/null 2>&1
     make -j8 GPU_ARCH="$ARCH" DEFS="$2" bench > ../build.log 2>&1) ||
        fail "$1: build failed, see $QHDIR/$1/build.log"
}

if [ "${QH_BUILD:-1}" != 0 ]; then
    build qh "-DCOFQ_TEST_HASH=1"
    build qf "-DCOFQ_TEST_HASH=1 -DCQ_FLUSH=16384u"
    build cc "-DCOFQ_TEST_HASH=1 -DPIPE_TD_TEST_CCAP=16"
    build qe "-DCOFQ_TEST_HASH=1"
    build qm "-DCOFQ_TEST_HASH=1"
fi
for b in qh qf cc qe qm; do
    [ -x "$QHDIR/$b/bench/bench" ] || fail "no $b build in $QHDIR"
done
[ $nfail -eq 0 ] || { echo "qhcheck: $nfail failed"; exit 1; }

# One case: name build least-after drained args... Least-after is how many
# slabs the pipelined run must have enqueued after the host's count (the
# first q's, which validation keeps off the pass ahead, is one); drained=1
# requires the growth branch's line.
pair() {
    name=$1; B=$QHDIR/$2/bench/bench; ma=$3; dr=$4; shift 4
    ncase=$((ncase+1))
    "$B" --pipeline --cofactor "$@" > "$O/$name.pl.out" 2>&1 || fail "$name: the pipelined run exited $?"
    "$B" --pipeline --cofactor --no-q-pipeline "$@" > "$O/$name.sq.out" 2>&1 ||
        fail "$name: the --no-q-pipeline run exited $?"
    grep 'tb_qh: flush' "$O/$name.pl.out" > "$O/$name.pl.qh"
    grep 'tb_qh: flush' "$O/$name.sq.out" > "$O/$name.sq.qh"
    nf=$(grep -c . "$O/$name.pl.qh")
    if [ "$nf" -ge 1 ] && cmp -s "$O/$name.pl.qh" "$O/$name.sq.qh"; then
        echo "   $name: $nf flush(es) identical," \
             "$(awk -F'n=' '{split($2, x, " "); s += x[1]} END {print s}' "$O/$name.pl.qh") candidates"
    else
        fail "$name: flush lines differ or are missing ($nf pipelined," \
             "$(grep -c . "$O/$name.sq.qh") --no-q-pipeline)"
    fi
    set -- $(grep -h 'tb_qh:.*enqueued in TD' "$O/$name.pl.out"); pin=${2:-0}; paf=${7:-0}
    set -- $(grep -h 'tb_qh:.*enqueued in TD' "$O/$name.sq.out"); sin=${2:-none}
    echo "     pipelined: $pin slab(s) enqueued in TD, $paf after"
    [ "$pin" -ge 1 ] 2>/dev/null || fail "$name: the pipelined run enqueued no slab in TD"
    [ "$paf" -ge "$ma" ] 2>/dev/null ||
        fail "$name: the pipelined run enqueued $paf slab(s) after the count, want >= $ma"
    [ "$sin" = 0 ] || fail "$name: the --no-q-pipeline run enqueued $sin slab(s) in TD"
    if [ "$dr" = 1 ]; then
        k=$(grep -c 'tb_cc: speculative record pass drained' "$O/$name.pl.out")
        if [ "$k" -ge 1 ]; then echo "     drained for growth $k time(s)"
        else fail "$name: the pass ahead was never drained for growth"; fi
    fi
}
skip() { echo "   SKIP $1: $2"; nskip=$((nskip+1)); }

C143="--poly ../oracle/c143.job --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --logI 13 --J 8192"
C183="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 20"
C194="--poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --slab-j 4095 --nq 3"
C135="--poly $C135_JOB --fb1 $C135_FB1 --logI 13 --J 4096 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000: --nq 300"
NFS="--poly $NFS_JOB --fb1 $NFS_FB1 --sq-side 1 --logI 16 --J 32768 --qrange 297000000: --nq 10"
have_c135=0; [ -f "$C135_JOB" ] && [ -f "$C135_FB1" ] && have_c135=1
have_nfs=0;  [ -f "$NFS_JOB" ] && [ -f "$NFS_FB1" ] && have_nfs=1

echo "full-size queue"
pair c143 qh 1 0 $C143 --nq 100
pair c183 qh 1 0 $C183
pair c194 qh 1 0 $C194
if [ $have_c135 = 1 ]; then pair c135 qh 1 0 $C135; else skip c135 "no $C135_JOB / $C135_FB1"; fi
if [ $have_nfs = 1 ];  then pair nfs qh 1 0 $NFS;   else skip nfs "no $NFS_JOB / $NFS_FB1"; fi

echo "--no-td-ahead"
ncase=$((ncase+1))
"$QHDIR/qh/bench/bench" --pipeline --cofactor --no-td-ahead $C143 --nq 100 > "$O/c143.nta.out" 2>&1 ||
    fail "c143.nta: the run exited $?"
grep 'tb_qh: flush' "$O/c143.nta.out" > "$O/c143.nta.qh"
set -- $(grep -h 'tb_qh:.*enqueued in TD' "$O/c143.nta.out")
if [ -s "$O/c143.nta.qh" ] && cmp -s "$O/c143.nta.qh" "$O/c143.sq.qh" && [ "${2:-x}" = 0 ] &&
   grep -q -- '--no-td-ahead: trial division' "$O/c143.nta.out"; then
    echo "   c143.nta: identical, none enqueued in TD"
else
    fail "c143.nta: flush lines differ, a slab went in TD, or no --no-td-ahead line"
fi

echo "16,384-slot queue"
pair c143.qf qf 2 0 $C143 --nq 100
pair c183.qf qf 2 0 $C183
if [ $have_c135 = 1 ]; then pair c135.qf qf 2 0 $C135; else skip c135.qf "no c135 job"; fi

echo "candidate list grown under the pass ahead"
pair c143.cc cc 1 1 $C143 --nq 100 --no-td-verify
if [ $have_c135 = 1 ]; then pair c135.cc cc 1 1 $C135 --no-td-verify; else skip c135.cc "no c135 job"; fi
ncase=$((ncase+1))
compute-sanitizer --tool memcheck --error-exitcode 9 "$QHDIR/cc/bench/bench" --pipeline --cofactor \
    $C143 --nq 4 --no-td-verify > "$O/c143.cc.mc.out" 2>&1 || fail "c143.cc.mc: memcheck run exited $?"
if grep -q "ERROR SUMMARY: 0 errors" "$O/c143.cc.mc.out" &&
   grep -q 'tb_cc: speculative record pass drained' "$O/c143.cc.mc.out"; then
    echo "   c143.cc.mc: 0 errors under memcheck, drained for growth"
else
    fail "c143.cc.mc: memcheck errors, or the pass ahead was never drained"
fi

echo "enqueue one short (the hash must differ)"
ncase=$((ncase+1))
"$QHDIR/qe/bench/bench" --pipeline --cofactor $C143 --nq 100 > "$O/c143.qe.pl.out" 2>&1 ||
    fail "c143.qe: the pipelined run exited $?"
grep 'tb_qh: flush' "$O/c143.qe.pl.out" > "$O/c143.qe.pl.qh"
if [ -s "$O/c143.qe.pl.qh" ] && [ -s "$O/c143.sq.qh" ] && ! cmp -s "$O/c143.qe.pl.qh" "$O/c143.sq.qh"; then
    echo "   c143.qe: seen ($(awk '{print $NF}' "$O/c143.qe.pl.qh" | head -1)" \
         "against $(awk '{print $NF}' "$O/c143.sq.qh" | head -1))"
else
    fail "c143.qe: the one-short enqueue went unseen"
fi

echo "device count one short (the run must fail)"
ncase=$((ncase+1))
"$QHDIR/qm/bench/bench" --pipeline --cofactor $C143 --nq 100 > "$O/c143.qm.pl.out" 2>&1; r=$?
if [ $r -ne 0 ] && grep -q 'have drifted apart' "$O/c143.qm.pl.out"; then
    echo "   c143.qm: refused (exit $r, the device's count against the host's)"
else
    fail "c143.qm: exit $r without the drift message"
fi

echo "qhcheck: $ncase case(s), $nskip skipped, $nfail failed"
[ $nfail -eq 0 ] && [ $ncase -gt 0 ]
