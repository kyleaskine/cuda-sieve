#!/bin/sh
# Golden test for the cofactor path, on the parity special-q q = 120000053.
#
# Every case pins an exact relation count rather than a range. The bugs this
# guards against -- a method switch that never takes effect, a factor base
# missing p = 2, a parser dropping its last record, a bound that truncates a
# factor -- all present as a *plausible* smaller number, never as a crash, so
# only an exact expectation catches them.
#
# las finds 37 relations at this q; 7 of them need no cofactorisation.
set -e

FB=../oracle/c183.fb1
POLY=../oracle/c183.poly
# Two words on purpose: the parser takes `--qrange VALUE`, not `--qrange=VALUE`,
# and the latter hits the unknown-option branch. Used unquoted so it splits.
Q="--qrange 120000053:120000053"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0

# The relation-count cases pin the survivor bound EXPLICITLY, to the values the
# derivation produced before it stopped importing lambda conventions. They are
# here to test the cofactoriser -- a method that silently falls back, a factor
# base missing p = 2, a truncated parse -- and that job is only done if the
# count is held still by everything else. Letting the allowance policy move
# these numbers would mean re-baselining 37 every time the policy is tuned, and
# a re-baselined golden number stops catching the bugs it was written for.
#
# The policy itself is pinned separately, in its own case below.
PIN="--allowance 101.6 --allowance0 68.1"
run() { ./bench --pipeline --cadofb $FB --poly $POLY --qrange 120000053:120000053 $PIN "$@" 2>&1; }

expect_rel() {   # name expected <bench args...>
    name=$1; want=$2; shift 2
    got=$(run "$@" --relations $TMP/r.txt | grep 'total relations' | tail -1 | awk '{print $NF}')
    if [ "$got" = "$want" ]; then
        printf 'PASS   %-34s %s relations\n' "$name" "$got"
    else
        printf 'FAIL   %-34s expected %s, got %s\n' "$name" "$want" "$got"
        fail=1
    fi
}

expect_refused() {  # name <bench args...>
    name=$1; shift
    if ./bench --pipeline --cadofb $FB --poly $POLY --qrange 120000053:120000053 \
               "$@" --relations $TMP/r.txt >$TMP/o 2>&1; then
        printf 'FAIL   %-34s accepted, should have been refused\n' "$name"; fail=1
    else
        printf 'PASS   %-34s refused\n' "$name"
    fi
}

echo
echo "[cofactor] golden test, q = 120000053"

expect_rel "trial division only"            7
# --cof-rho on every case that names rho. Since 2026-08-19 the method is chosen
# per side and this job's algebraic side is 3LP, so without the flag these run
# ECM there -- a rho regression would leave the pinned count intact and the
# suite green, which is the opposite of what a golden case is for.
expect_rel "rho, inline queue"             37 --cofactor --cof-rho --cof-rounds 2 --cof-budget 65536
expect_rel "ECM, inline queue"             37 --cofactor --cof-ecm --ecm-b1 2000 --ecm-curves 48
# --cof-chunk, both methods. WITHOUT these the multi-launch path is untested by
# this entire suite: one q queues ~2000 records against an auto floor of
# blocks*threads (73,728 on a 5070, 196,608 on a 4090), so `step` collapses to
# a single slice and every case above runs the pre-chunking code path. 256
# forces ~8 launches per round, so a wrong slice bound, a record dropped at a
# boundary, or an off-by-one in k_cofac's `hi` shows up as a changed count here
# rather than only on a slow volunteer card. The expectation is the SAME 37:
# chunking splits the record list and cannot change a result, which is the
# entire claim the flag rests on.
expect_rel "rho, chunked launches"         37 --cofactor --cof-rho --cof-rounds 2 --cof-budget 65536 --cof-chunk 256
expect_rel "ECM, chunked launches"         37 --cofactor --cof-ecm --ecm-b1 2000 --ecm-curves 48 --cof-chunk 256
expect_refused "chunk below one block"     --cofactor --cof-rho --cof-chunk 1
# --cof-rounds 2 pinned: the default became 4 on 2026-08-19, and at 4 rounds
# stage 1 alone (64 curves) finds the relation this control exists to show
# stage 2 finding. The case tests stage 2, so it must hold the curve budget
# still.
expect_rel "ECM stage 2 control, disabled" 36 --cofactor --cof-ecm --cof-rounds 2 --ecm-b1 1000 --ecm-b2 0 --ecm-curves 16
expect_rel "ECM stage 2, inline queue"     37 --cofactor --cof-ecm --cof-rounds 2 --ecm-b1 1000 --ecm-b2 10000 --ecm-curves 16
# The method switch must actually REACH the queue. B1 = 2 with one curve can
# split essentially nothing, so this must fall back to the trial-division count;
# 37 here would mean rho ran despite --cof-ecm, which is exactly the defect the
# review found (the queue's ECM fields were left at their memset zero).
expect_rel "ECM honoured, not silently rho"  7 --cofactor --cof-ecm --ecm-b1 2 --ecm-curves 1
expect_refused "zero ECM curves"               --cofactor --cof-ecm --ecm-curves 0
# B2 is DERIVED (30*B1) when --ecm-b2 is absent, so --ecm-b1 alone now runs
# stage 2 where it used to run stage 1 only. Deliberate, and pinned here
# because it silently changes the cost and yield of any script passing B1 alone.
got=$(run --cofactor --cof-ecm --ecm-b1 2000 --relations $TMP/b2d.txt \
      | grep -c 'ECM B1 = 2000, .*B2 = 60000' || true)
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s B2 = 30*B1\n' "B2 derived from B1"
else
    printf 'FAIL   %-34s no derived B2 line\n' "B2 derived from B1"; fail=1
fi
# ...and a B1 near the top of its own documented range must not self-refuse by
# deriving a B2 past the B2 ceiling. --ecm-b1 accepts up to 1000000.
if run --cofactor --cof-ecm --ecm-b1 400000 --nq 1 --relations $TMP/b1hi.txt \
   >/dev/null 2>&1; then
    printf 'PASS   %-34s accepted\n' "large B1 with derived B2"
else
    printf 'FAIL   %-34s refused itself\n' "large B1 with derived B2"; fail=1
fi
# --ecm-b2 without --cof-ecm is NO LONGER an error: since 2026-08-19 the method
# is chosen per side, and this job's algebraic side is 3LP, so ECM is selected
# automatically and the flag is live. Forcing rho is what makes it a no-op.
expect_refused "stage 2 with rho forced"        --cofactor --cof-rho --ecm-b2 10000
expect_refused "stage 2 below B1"               --cofactor --cof-ecm --ecm-b1 1000 --ecm-b2 1000

# ---- job-file parameters -------------------------------------------------
# ../oracle/input.job is the SAME polynomial as c183.poly, plus the sieve keys.
# So the two must agree once the one thing that differs -- the lambdas, which
# c183.poly does not carry -- is stated explicitly on both.
#
# The counts are compared to EACH OTHER rather than pinned, deliberately. What
# is being tested is that reading the job file changes nothing it should not;
# pinning a number here would also pin the derived allowance, which is a
# separate thing with its own case above.
JOB=../oracle/input.job
# `|| true` on every grep -c: it exits 1 when the count is 0, and `set -e` at
# the top of this script turns that into an abort partway through the suite
# rather than a FAIL line. That cost a confusing debugging round once.
got=$(./bench --pipeline --cadofb $FB --poly $JOB $Q --relations $TMP/j.txt 2>&1 \
      | grep -c 'job file: rlim 67100000, alim 134200000, lpbr 31, lpba 32, mfbr 60, mfba 92' || true)
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s all six keys\n' "job file parsed"
else
    printf 'FAIL   %-34s banner line not found\n' "job file parsed"; fail=1
fi

# The job file's lambda is REPORTED and NOT APPLIED. Two things to pin: that
# the GGNFS-unit conversion is still right (3.5 * log2(134200000) = 94.5 bits,
# not the 112 a CADO reading would give), and that the number does not reach
# the survivor bound -- it is calibrated to GGNFS's gate, which is measurably
# tighter than ours at the same nominal bits.
# `|| true` for the same reason as the greps: a bare command substitution under
# `set -e` takes bench's exit status, so a failing case would abort the suite
# instead of printing FAIL and continuing.
out=$(./bench --pipeline --cadofb $FB --poly $JOB $Q --relations $TMP/j.txt 2>&1 || true)
got=$(printf '%s' "$out" | grep -c 'alambda 3.5 -> 94.5[0-9]* bits (side 1), reported only' || true)
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s 3.5 -> 94.5 bits, reported not applied\n' "GGNFS lambda conversion"
else
    printf 'FAIL   %-34s wrong or missing conversion\n' "GGNFS lambda conversion"; fail=1
fi
# ...and the allowance actually in force is the DERIVED one, not 94.50.
got=$(printf '%s' "$out" | grep -c 'side 1 log2(maxnorm)=.* allowance=94.50' || true)
if [ "$got" = "0" ]; then
    printf 'PASS   %-34s job lambda did not reach the bound\n' "lambda not applied"
else
    printf 'FAIL   %-34s job lambda leaked into the allowance\n' "lambda not applied"; fail=1
fi

# Explicit flags outrank the job file. With both allowances stated, the job
# file and the bare poly must produce the identical relation set.
a=$(run --cofactor --cof-rounds 2 --cof-budget 65536 --allowance 112 --allowance0 68.1 --relations $TMP/p1.txt | grep 'total relations' | tail -1 | awk '{print $NF}')
b=$(./bench --pipeline --cadofb $FB --poly $JOB --qrange 120000053:120000053 \
        --cofactor --cof-rounds 2 --cof-budget 65536 --allowance 112 --allowance0 68.1 \
        --relations $TMP/p2.txt 2>&1 | grep 'total relations' | tail -1 | awk '{print $NF}')
if [ -n "$a" ] && [ "$a" = "$b" ] && cmp -s $TMP/p1.txt $TMP/p2.txt; then
    printf 'PASS   %-34s %s each, byte-identical\n' "explicit flags outrank job file" "$a"
else
    printf 'FAIL   %-34s poly %s vs job %s\n' "explicit flags outrank job file" "$a" "$b"; fail=1
fi

# ---- cofactor width -------------------------------------------------------
# Side 0 is 3 limbs like side 1, so an SNFS job with mfbr 88 on the rational
# side is representable. 97 is not, and 96 against lpb0 31 asks for 4 parts,
# which mz_split's stack cannot hold -- it would present as CF_OVERFLOW on the
# records needing it, i.e. a silent partial yield loss, so it is refused.
expect_refused "side-0 mfb above 3 limbs"     --cofactor --mfb0 97
expect_refused "side-0 4LP refused"           --cofactor --mfb0 96 --lpb0 31
expect_refused "side-1 4LP refused"           --cofactor --mfb 96 --lpb 24
# $PIN so this measures the cofactor WIDTH and not the allowance policy: with
# --mfb0 88 and no pin, the derived allowance0 follows mfb to 89.5 bits and the
# case would be exercising the survivor bound instead of the thing it is named
# for.
if ./bench --pipeline --cadofb $FB --poly $POLY $Q $PIN --cofactor --mfb0 88 --lpb0 31 \
           --relations $TMP/w.txt >$TMP/o 2>&1; then
    printf 'PASS   %-34s accepted\n' "side-0 3LP width (mfb0 88)"
else
    printf 'FAIL   %-34s refused, should be legal\n' "side-0 3LP width (mfb0 88)"; fail=1
fi

expect_refused "sq-side out of range"         --cofactor --sq-side 2

# ---- survivor allowance policy -------------------------------------------
# The DERIVED default (mfb + ~1.5 bits, from our own quantisation) is tighter
# than the old CADO-rule value, and on this single q that costs one relation:
# 36 rather than the 37 the pinned cases above get. That is a deliberate trade,
# measured over a real band rather than this one q -- c183, 120 special-q:
#
#     allowance0 61.5 (mfb+1.5)   82,969 survivors/q   46.48 rel/q
#     allowance0 68.1 (mfb+8.1)  143,760 survivors/q   46.57 rel/q
#
# 42% of the trial-division input for 0.19% of the relations. The single-q
# figure of 1-in-37 is an unlucky sample, not the rate.
#
# Pinned so that a change to the allowance policy has to be deliberate: if this
# moves, the derivation moved, and the band measurement above needs redoing.
got=$(./bench --pipeline --cadofb $FB --poly $POLY $Q --cofactor --cof-rounds 2 \
      --cof-budget 65536 --relations $TMP/pol.txt 2>&1 \
      | grep 'total relations' | tail -1 | awk '{print $NF}')
if [ "$got" = "36" ]; then
    printf 'PASS   %-34s 36 relations\n' "derived allowance policy"
else
    printf 'FAIL   %-34s expected 36, got %s\n' "derived allowance policy" "$got"; fail=1
fi

# ---- large primes above 2^32 ----------------------------------------------
# lpb 33 is a REAL configuration, not an edge: an NFS@Home C194 asks for
# lpba 33 with mfba 95. Split factors, unsplit prime residuals, the emitter and
# the reconstruction gate are all 64-bit, so the limit is the 64-bit word and
# nothing below it.
expect_refused "lpb above the 64-bit word"    --cofactor --lpb 65
# --lpb 65 is rejected by the ARGUMENT PARSER, which is not the check that
# matters. resolve_and_check_cofactor_config() exists for the values that arrive from a
# .job FILE, where nothing range-checks them -- its own comment says so. That
# branch needs a job file to reach it.
{ cat $POLY; printf 'rlim: 134200000\nalim: 134200000\nlpbr: 31\nlpba: 65\nmfbr: 60\nmfba: 92\n'; } > $TMP/lpb65.job
if ./bench --pipeline --cadofb $FB --poly $TMP/lpb65.job $Q --cofactor \
           --relations $TMP/r.txt >$TMP/o 2>&1; then
    printf 'FAIL   %-34s accepted, should have been refused\n' "job-file lpba above 64"; fail=1
else
    printf 'PASS   %-34s refused\n' "job-file lpba above 64"
fi
# The positive half, and it is the one that matters: refusal cases alone would
# still pass with the whole 64-bit path broken. Pinned on an exact count AND on
# a factor actually exceeding 2^32 -- without the second assertion this case
# would silently stop exercising the widening if the gate or the job ever
# moved. Through `run`, so it carries $PIN like every other count-pinned case:
# 59 has to be a function of the widening, not of the allowance policy.
# --cof-rho: this case pins a count, and at lpb 33 / mfb 92 the automatic
# method would pick ECM (3 parts), which finds 60 rather than rho's 59. Both
# are correct -- ECM is a strict superset here (finding 70) -- but a golden
# count has to name its method or it moves the next time a default does.
got=$(run --cofactor --cof-rho --cof-rounds 2 --cof-budget 65536 --lpb 33 \
          --relations $TMP/l33.txt | grep 'total relations' | tail -1 | awk '{print $NF}')
# length > 8 hex digits is the whole test: any 8-digit value is below 2^32 by
# definition. `|| echo 0` because a failed run above leaves no file, and a bare
# command substitution would take `set -e` down with it before the FAIL prints.
big=$(awk -F: '!/^#/ && NF>=3 {
          n = split($2","$3, f, ",");
          for (i = 1; i <= n; i++) if (length(f[i]) > 8) c++
      } END {print c+0}' $TMP/l33.txt 2>/dev/null || echo 0)
if [ "$got" = "59" ] && [ "$big" -gt 0 ] && \
   ./bench --check-relations $TMP/l33.txt --poly $POLY --lpb 33 --lpb0 31 2>&1 \
   | grep -q 'PASS'; then
    printf 'PASS   %-34s 59 relations, %s factors > 2^32, all reconstruct\n' \
           "lpb 33 emits and verifies" "$big"
else
    printf 'FAIL   %-34s expected 59 relations with >2^32 factors, got %s / %s\n' \
           "lpb 33 emits and verifies" "$got" "$big"; fail=1
fi
# The gate's lpb bound must be load-bearing, not decorative: the same file read
# back at lpb 32 has to be REFUSED, or "verifies at lpb 33" proves nothing.
if ./bench --check-relations $TMP/l33.txt --poly $POLY --lpb 32 --lpb0 31 2>&1 \
   | grep -q 'FAIL'; then
    printf 'PASS   %-34s same file refused at lpb 32\n' "lpb bound is enforced"
else
    printf 'FAIL   %-34s lpb 32 accepted a 33-bit factor\n' "lpb bound is enforced"; fail=1
fi

# ---- automatic per-side METHOD --------------------------------------------
# rho for a 2LP side, ECM for a 3LP one. This job is mfbr 60 / lpbr 31 (2 parts)
# and mfba 92 / lpba 32 (3 parts), so it must resolve to rho / ECM -- the same
# mixed shape a real job has, and the reason the choice is per side at all.
got=$(run --cofactor --relations $TMP/m.txt | grep -c 'cofactor method: side 0 rho, side 1 ECM' || true)
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s rho / ECM\n' "auto method, 2LP and 3LP sides"
else
    printf 'FAIL   %-34s did not resolve to rho / ECM\n' "auto method, 2LP and 3LP sides"; fail=1
fi
# Forcing must override the automatic choice in both directions.
got=$(run --cofactor --cof-rho --relations $TMP/m.txt | grep -c 'side 0 rho, side 1 rho' || true)
got2=$(run --cofactor --cof-ecm --relations $TMP/m.txt | grep -c 'side 0 ECM, side 1 ECM' || true)
if [ "$got" = "1" ] && [ "$got2" = "1" ]; then
    printf 'PASS   %-34s both directions\n' "--cof-rho / --cof-ecm override"
else
    printf 'FAIL   %-34s rho=%s ecm=%s\n' "--cof-rho / --cof-ecm override" "$got" "$got2"; fail=1
fi

# ---- cofactor WIDTH (3 limbs vs 4) ----------------------------------------
# The width is now a per-side run-time choice: 3 limbs (96 bits) covers this
# job and a C194, 4 limbs (128) is what a C208's algebraic mfb needs. Which
# widths a build carries is a compile-time knob (CF_LMAX), so the cases below
# ask the binary what it has rather than assuming, and the boundary case is
# computed from the answer instead of being pinned at 97.
LMAX=$(run --cofactor --cof-rounds 1 --cof-budget 4096 --relations $TMP/w.txt \
       | sed -n 's/.*this build carries [0-9]*\.\.\([0-9]*\).*/\1/p' | head -1)
case "$LMAX" in
    3|4) printf 'PASS   %-34s %s limbs\n' "build reports its widest cofactor" "$LMAX" ;;
    *)   printf 'FAIL   %-34s no width line in the output\n' "build reports its widest cofactor"
         fail=1; LMAX=3 ;;
esac
expect_refused "mfb above the widest cofactor" --cofactor --mfb $((32 * LMAX + 1))

if [ "$LMAX" -ge 4 ]; then
    # THE case for the 4-limb work, and the only one that can be run without a
    # C208 in hand: rho and ECM are Montgomery-domain algorithms whose sequence
    # is y <- y^2 + c in the TRUE domain regardless of R = 2^(32L), so widening
    # a side must change the cost and nothing else. Byte-identical output, not
    # an equal count -- an equal count would still pass if the widening
    # reordered or substituted factors. Same argument, and the same evidence,
    # as the 2 -> 3 limb widening of the rational side.
    run --cofactor --cof-rho --cof-rounds 2 --cof-budget 65536 --relations $TMP/w3.txt >/dev/null
    run --cofactor --cof-rho --cof-rounds 2 --cof-budget 65536 --cof-limbs 4 --cof-limbs0 4 \
        --relations $TMP/w4.txt >/dev/null
    if [ -s $TMP/w3.txt ] && cmp -s $TMP/w3.txt $TMP/w4.txt; then
        printf 'PASS   %-34s identical to the 3-limb run\n' "rho at 4 limbs"
    else
        printf 'FAIL   %-34s output differs from the 3-limb run\n' "rho at 4 limbs"; fail=1
    fi
    # 4/3, the asymmetric shape a C208 actually asks for: only the hard side
    # widens. It exercises the mixed instantiation of k_cof_enqueue, which the
    # symmetric case above does not.
    run --cofactor --cof-rho --cof-rounds 2 --cof-budget 65536 --cof-limbs 4 \
        --relations $TMP/w43.txt >/dev/null
    if [ -s $TMP/w43.txt ] && cmp -s $TMP/w3.txt $TMP/w43.txt; then
        printf 'PASS   %-34s identical to the 3-limb run\n' "4/3 split width"
    else
        printf 'FAIL   %-34s output differs from the 3-limb run\n' "4/3 split width"; fail=1
    fi
    run --cofactor --cof-ecm --ecm-b1 1000 --ecm-b2 10000 --ecm-curves 16 \
        --relations $TMP/e3.txt >/dev/null
    run --cofactor --cof-ecm --ecm-b1 1000 --ecm-b2 10000 --ecm-curves 16 \
        --cof-limbs 4 --relations $TMP/e4.txt >/dev/null
    if [ -s $TMP/e3.txt ] && cmp -s $TMP/e3.txt $TMP/e4.txt; then
        printf 'PASS   %-34s identical to the 3-limb run\n' "ECM at 4 limbs"
    else
        printf 'FAIL   %-34s output differs from the 3-limb run\n' "ECM at 4 limbs"; fail=1
    fi

    # The cases above force the width; this one DERIVES it, which is the path
    # every real job takes. `lpb 33, mfb 99` is the smallest shape on this
    # polynomial that genuinely needs 4 limbs -- CF_MAXFAC caps the split at 3
    # parts, so mfb cannot exceed 3*lpb, and at lpb 32 that is exactly the
    # 96 bits 3 limbs already hold. It is the c183 standing in for a C208's
    # asymmetry: side 1 wide, side 0 narrow, chosen by nobody.
    out=$(run --cofactor --cof-rounds 2 --cof-budget 65536 --lpb 33 --mfb 99 \
              --relations $TMP/w99.txt || true)
    got=$(printf '%s' "$out" | grep -c 'side 0 3 limbs (96 bits), side 1 4 limbs (128 bits)' || true)
    n99=$(printf '%s' "$out" | grep 'total relations' | tail -1 | awk '{print $NF}')
    # Not pinned to a number: mfb 99 admits candidates the pinned lpb-33 case
    # (mfb 92) never saw, so the count is a property of the wider gate and not
    # of the width. What must hold is that the width was DERIVED as 4/3, that
    # relations came out, and that every one of them reconstructs its norm --
    # which is the assertion a truncated high limb would fail.
    if [ "$got" = "1" ] && [ "${n99:-0}" -ge 59 ] && \
       ./bench --check-relations $TMP/w99.txt --poly $POLY --lpb 33 --lpb0 31 2>&1 \
       | grep -q 'PASS'; then
        printf 'PASS   %-34s 4/3 derived, %s relations, all reconstruct\n' \
               "lpb 33 / mfb 99 needs 4 limbs" "$n99"
    else
        printf 'FAIL   %-34s width line %s, %s relations\n' \
               "lpb 33 / mfb 99 needs 4 limbs" "$got" "$n99"; fail=1
    fi
    # ...and the same shape with the width forced back down must be REFUSED,
    # or "derived 4" above proves only that a default was printed.
    expect_refused "cof-limbs below what mfb needs" --cofactor --lpb 33 --mfb 99 --cof-limbs 3
else
    printf 'SKIP   %-34s build is 3-limb only\n' "4-limb width cases"
fi

expect_refused "zero rho budget"              --cofactor --cof-budget 0
expect_refused "rho rounds overflowing shift" --cofactor --cof-rounds 40
expect_refused "lambda in the typo window"    --cofactor --lambda1 0.01
expect_refused "lambda absurdly large"        --cofactor --lambda1 99
# 0 is the documented "use CADO's automatic" sentinel and must stay legal.
expect_rel     "lambda 0 == automatic"     37 --cofactor --cof-rounds 2 --cof-budget 65536 --lambda1 0

# A factor base missing p = 2 leaves algebraic cofactors even and breaks
# mz_n0inv's odd-modulus contract.  This used to rely on omitting --fb1 and
# thereby selecting the incomplete legacy GGNFS factor base.  Omitted --fb1
# now deliberately means "generate a complete factor base on the GPU", so make
# the broken input explicit: remove the q = 2 rows from the known-good native
# C183 factor base and verify that relation production refuses it.  Powers of 2
# may remain; the invariant being tested is specifically the missing p = 2 row.
grep -v '^2:' "$FB" > "$TMP/no_p2.fb1"
if ./bench --pipeline --fb1 "$TMP/no_p2.fb1" --poly $POLY \
           --qrange 120000053:120000053 --cofactor \
           --relations $TMP/r.txt >$TMP/o 2>&1; then
    printf 'FAIL   %-34s accepted an even-cofactor factor base\n' "even cofactors refused"; fail=1
else
    printf 'PASS   %-34s refused\n' "even cofactors refused"
fi

# Standalone splitter: the candidate file must parse identically with and
# without a trailing newline, and must agree with the inline queue.
run --candidates $TMP/c.txt --relations $TMP/td.txt >/dev/null
head -c -1 $TMP/c.txt > $TMP/c_nonl.txt
a=$(./bench --cofac $TMP/c.txt      --poly $POLY --relations $TMP/s1.txt --cof-rounds 2 --cof-budget 65536 2>&1 | grep 'RELATIONS' | awk '{print $2}')
b=$(./bench --cofac $TMP/c_nonl.txt --poly $POLY --relations $TMP/s2.txt --cof-rounds 2 --cof-budget 65536 2>&1 | grep 'RELATIONS' | awk '{print $2}')
if [ "$a" = "$b" ] && [ "$a" = "30" ]; then
    printf 'PASS   %-34s %s each, terminated and not\n' "unterminated candidate file" "$a"
else
    printf 'FAIL   %-34s expected 30/30, got %s/%s\n' "unterminated candidate file" "$a" "$b"
    fail=1
fi

# The inline queue and the two-pass path must emit the same relation set.
run --cofactor --cof-rounds 2 --cof-budget 65536 --relations $TMP/inline.txt >/dev/null
sort $TMP/inline.txt > $TMP/x
sort $TMP/td.txt $TMP/s1.txt > $TMP/y
if cmp -s $TMP/x $TMP/y; then
    printf 'PASS   %-34s byte-identical\n' "inline == two-pass"
else
    printf 'FAIL   %-34s differ\n' "inline == two-pass"; fail=1
fi

# The candidate RECORDING pass, one warp per candidate against one thread per
# candidate (RESULTS finding 77). What this compares is the FACTOR MULTISET and
# the relation total, not the division order -- an earlier version of this
# comment claimed the byte compare caught a reordering, and it does not:
# td_divide_out consumes the whole power of a prime per call so distinct
# entries commute, and both emitters sort before writing (cf_emit_sorted,
# std::sort). Reversing the ballot walk still passes here, by design and not by
# oversight. What WOULD fail is a wrong predicate, a dropped entry, a missed
# multiplicity, or a slab-indexing bug -- which is what this is for.
#
# The count is pinned as well as compared. Both arms share the survivor bound,
# the intersect, the classifier and the cofactoriser, so a defect in any of
# them moves both files together and an equality-only test would still pass.
#
# Both slab shapes, because the two kernels index j differently: unslabbed uses
# the local row directly, slabbed adds j_base, and the recording pass is the one
# place that runs once per slab per side.
#
# Counts come out of command substitution and every branch is guarded, because
# `set -e` at the top turns a bare failing run into an abort partway through the
# suite rather than a FAIL line -- see the note above.
for sj in 0 4096; do
    case $sj in
        0) sjarg=""            ; shape="unslabbed" ;;
        *) sjarg="--slab-j $sj"; shape="4 slabs"   ;;
    esac
    nw=$(run --cofactor $sjarg --relations $TMP/rw.txt \
         | grep 'total relations' | tail -1 | awk '{print $NF}') || true
    ns=$(run --cofactor $sjarg --td-record-scalar --relations $TMP/rs.txt \
         | grep 'total relations' | tail -1 | awk '{print $NF}') || true
    if [ "$nw" = "37" ] && [ "$ns" = "37" ] && cmp -s $TMP/rw.txt $TMP/rs.txt; then
        printf 'PASS   %-34s %s, 37 relations, identical\n' "warp vs thread recording" "$shape"
    else
        printf 'FAIL   %-34s %s, expected 37/37 identical, got %s/%s\n' \
               "warp vs thread recording" "$shape" "${nw:-none}" "${ns:-none}"; fail=1
    fi
done

# Post-cofactor reconstruction: what was EMITTED must rebuild both norms. The
# pre-split gate cannot see this, and a corrupted factor is the negative control.
for f in inline s1; do
    if ./bench --check-relations $TMP/$f.txt --poly $POLY 2>&1 | grep -q PASS; then
        printf 'PASS   %-34s %s\n' "emitted factors rebuild norms" "$f"
    else
        printf 'FAIL   %-34s %s\n' "emitted factors rebuild norms" "$f"; fail=1
    fi
done
sed '1s/:\([0-9a-f]*\),/:\1f,/' $TMP/inline.txt > $TMP/corrupt.txt
if ./bench --check-relations $TMP/corrupt.txt --poly $POLY 2>&1 | grep -q FAIL; then
    printf 'PASS   %-34s corruption detected\n' "reconstruction negative control"
else
    printf 'FAIL   %-34s corruption NOT detected\n' "reconstruction negative control"; fail=1
fi

# The corruption control above only proves the gate notices a factor that does
# not DIVIDE. It says nothing about the failure that is actually likely here: a
# cofactor split that stops early and emits p*q as one "prime". That divides
# perfectly, rebuilds the norm to 1, and passed every check this gate had until
# the primality test was added -- so it needs its own control, built by merging
# a relation's first two side-0 factors into their product.
made=0
while IFS= read -r ln; do
    if [ $made = 0 ]; then
        case ${ln%%:*} in \#*) printf '%s\n' "$ln"; continue;; esac
        ab=${ln%%:*}; rest=${ln#*:}; s0=${rest%%:*}; s1=${rest#*:}
        case $s0 in
        *,*) f1=${s0%%,*}; tl=${s0#*,}; f2=${tl%%,*}; tl2=${tl#*,}
             p=$(( 0x$f1 * 0x$f2 ))
             # Must stay under 2^32 or the range check rejects it first and the
             # control passes without ever exercising primality.
             if [ $p -le 4294967295 ]; then
                 if [ "$tl" = "$f2" ]; then s0=$(printf '%x' $p)
                 else s0="$(printf '%x' $p),$tl2"; fi
                 ln="$ab:$s0:$s1"; made=1
             fi ;;
        esac
    fi
    printf '%s\n' "$ln"
done < $TMP/inline.txt > $TMP/composite.txt
if [ $made = 0 ]; then
    printf 'SKIP   %-34s no relation with two small side-0 factors\n' \
        "primality negative control"
elif ./bench --check-relations $TMP/composite.txt --poly $POLY 2>&1 | grep -q FAIL; then
    printf 'PASS   %-34s composite factor detected\n' "primality negative control"
else
    printf 'FAIL   %-34s composite NOT detected\n' "primality negative control"; fail=1
fi

# ---- generated-band control ---------------------------------------------
# Three ideals exercise all generator handoffs: q0 is cached for norm setup,
# q1 is the first pull inside run_pipeline, and q2 is another root from the
# stream. Compare with the exact oracle pairs, then make the count cap compete
# with an unreachable relation target so its termination message is gated too.
cat >$TMP/q3 <<EOF
120000007 30589397
120000053 112625526
120000103 21725368
EOF
gout=$(./bench --pipeline --cadofb $FB --poly $POLY \
       --qrange 120000000:120000200 --nq 3 --target-rels 999999 $PIN \
       --relations $TMP/qgen.txt 2>&1 || true)
lout=$(./bench --pipeline --cadofb $FB --poly $POLY --qlist $TMP/q3 --nq 10 $PIN \
       --relations $TMP/qlist.txt 2>&1 || true)
if printf '%s' "$gout" | grep -q '\[--nq 3 reached\]' &&
   printf '%s' "$gout" | grep -q 'note: --nq 3 reached.*--target-rels 999999 was not reached' &&
   printf '%s' "$gout" | grep -q -- '--- band of 3 special-q ---' &&
   printf '%s' "$lout" | grep -q -- '--- band of 3 special-q ---' &&
   printf '%s' "$lout" | grep -q '100.0%' &&
   cmp -s $TMP/qgen.txt $TMP/qlist.txt; then
    printf 'PASS   %-34s cached + streamed q, byte-identical\n' \
        "multi-q generated band"
else
    printf 'FAIL   %-34s generator/list mismatch or bad terminal status\n' \
        "multi-q generated band"; fail=1
fi

# qmin == 0 is a valid spelling that normalises to the first prime. It used to
# double as the parser's "--qrange absent" sentinel, silently dropping 0:MAX
# and running the unrelated single-q default instead.
zero=$(./bench --pipeline --cadofb $FB --poly $POLY --qrange 0:1 $PIN 2>&1 || true)
if printf '%s' "$zero" | grep -q 'no affine special-q roots in \[0, 1\]'; then
    printf 'PASS   %-34s parsed as an empty generated band\n' "zero qrange lower bound"
else
    printf 'FAIL   %-34s qrange was dropped or misreported\n' "zero qrange lower bound"; fail=1
fi

# A partial warp has no lane 31 for k_intersect_compact to broadcast its atomic
# base from. This used to be caught only by the survivor/rank cross-check, i.e.
# after a full band had run.
if ./bench --threads 33 --poly $POLY 2>&1 | grep -q 'multiple of 32'; then
    printf 'PASS   %-34s --threads 33 rejected\n' "warp-multiple guard"
else
    printf 'FAIL   %-34s --threads 33 accepted\n' "warp-multiple guard"; fail=1
fi

[ $fail = 0 ] && echo "cofactor golden test passed" || { echo "cofactor golden test FAILED"; exit 1; }
