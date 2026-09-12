/* Cofactorisation: split the residual norms into large primes, on device.
 *
 * This is the stage that stands between a survivor list and a relation count.
 * At the parity q trial division alone yields 7 relations of las's 37; the
 * other 30 are inside the cofactors, and over a 67-q band the split is 556 of
 * 3,162 -- 83% of the yield is here.
 *
 * WHAT THE WORK ACTUALLY IS, measured from 1,958,036 emitted candidates rather
 * than assumed from the mfb settings (see prototype.md):
 *
 *   rational  (lpb 31, mfb 60): 1,007,630 splits, ALL 53-60 bits, ALL 2LP.
 *                               Both factors in (2^26, 2^31].
 *   algebraic (lpb 32, mfb 92): 1,949,125 splits, bimodal --
 *                               2.6% at 55-64 bits (2LP),
 *                               97.4% at 81-92 bits (3LP, ~30-bit factors).
 *                               Nothing at 65-80 bits: that is COF_REJECT_GAP.
 *
 * So this is not a general-purpose cofactorizer. It is a few narrow shapes,
 * and both the arithmetic WIDTH and the METHOD are sized to them PER SIDE,
 * chosen on the host from that side's mfb and lpb (2026-08-18/19):
 *
 *     width   3 limbs (96 bits) up to mfb 96, 4 (128) above    -- CF_LMAX
 *     method  rho at 2LP, ECM at 3LP                           -- cof_meth[]
 *
 * Both rules are in bench.h, both are printed by every run, and both exist
 * because a real job is one shape on one side and another on the other: AS276
 * is 2LP/3-limb rational and 3LP/4-limb algebraic. A per-JOB choice of either
 * is wrong on one side of nearly every job.
 *
 * The rational side was 2 limbs while the special-q always sat on the
 * algebraic side. An SNFS job whose hard side is the rational one (mfbr 88)
 * needs the width there instead. Measured cost of widening it, c183 band of 66
 * special-q, identical parameters, output byte-identical (same md5):
 *
 *     rational queue    2.15 -> 3.94 ms/q   (+83%)
 *     wall clock        172.65 -> 173.75    (+0.64%)
 *
 * So the stage itself nearly doubles and it still does not matter, because the
 * rational queue is ~1% of a special-q. That is the trade being made -- not
 * "the cost is nil", but "the cost is 0.6% of wall to stop assuming which side
 * is hard". A per-side dispatch would buy that 0.6% back at the price of a
 * second instantiation to keep in step, which is what task #15 removed.
 * (Per-side dispatch did return on 2026-08-18, but to WIDEN the hard side
 * rather than narrow the easy one -- the argument above only ever said that
 * narrowing side 0 was not worth it.)
 *
 * WHY POLLARD-BRENT RHO AND NOT ECM. **PARTLY SUPERSEDED 2026-08-19 -- see
 * RESULTS finding 70. With BOTH methods swept from below to saturated yield,
 * tuned ECM is ~2x cheaper than rho on the c183 and ~2.7x on AS276, and the
 * discriminant is the large-prime count: at 2LP rho still wins narrowly, at
 * 3LP ECM wins 2-4x. Output is byte-identical either way. The
 * reasoning below is still a correct account of rho's SHAPE -- its cost really
 * is data-dependent and ECM's really is not -- but the conclusion drawn from
 * it was wrong, because the comparison that supported it ran ECM at B1=1000
 * (~4x above optimal for this job's ~30-bit factors) and rho at a budget below
 * full yield. Both errors favoured rho. The bounded-budget/requeue structure
 * described below is method-agnostic and is what ECM uses too.**
 *
 * The original argument, kept because the requeue design rests on it: ECM's appeal on a GPU is that its cost is
 * data-independent, which rho's emphatically is not -- rho's iteration count is
 * geometric, and a warp runs at its slowest lane. The answer here is not to
 * abandon rho but to BOUND it: every lane gets the same iteration budget per
 * launch, and whatever has not split is requeued for another launch with a
 * different polynomial constant. Divergence inside a launch is then bounded by
 * the budget rather than by the tail of a geometric distribution, and the tail
 * becomes a scheduling question -- which is what the "reseed, requeue,
 * escalate" note in prototype.md was about. A fixed cap that DROPS work would
 * be wrong: rho is Monte Carlo, so a cofactor that resists one constant is not
 * a cofactor that cannot be split.
 *
 * PRIMALITY IS MOSTLY FREE HERE, and that is worth stating because it removes
 * most of the sprp2 calls a general cofactorizer would make. Trial division
 * already removed every factor below `lim`, so any part below lim^2 is prime
 * BY SIZE -- no test. On this job lim^2 is 2^54 on the algebraic side, so the
 * only parts that need a real primality test are those of 55 bits or more,
 * i.e. the middle of a 3LP split.
 */
#ifndef CUDA_SIEVE_COFAC_CUH
#define CUDA_SIEVE_COFAC_CUH

#include "watchdog.h"
#include <stdint.h>
#include <errno.h>          /* strtoull ERANGE in the relation gate */
/* Directly, not by luck of translation-unit ordering: this header calls
 * bench_mul_u64_wide, bench_u128_take32, bench_seek_end and friends, and it
 * compiled only because bench_kernels.cu happens to include platform.h first.
 * Any new .cu that includes this one would fail on MSVC without it. */
#include "platform.h"

#define CF_OK         0   /* fully split, every prime within lpb */
#define CF_DEAD       1   /* a prime factor exceeds 2^lpb: never a relation */
#define CF_INCOMPLETE 2   /* out of rho budget: requeue, do not drop */
#define CF_OVERFLOW   3   /* more parts than the caller made room for */

#define CF_MAXFAC 4       /* ceil(mfb/lpb) is at most 3 on EITHER side -- 3LP
                           * is 88/31 on the rational side of an SNFS job and
                           * 92/32 on the algebraic side of a GNFS one -- so 4
                           * leaves one slot of headroom and an unexpected
                           * shape is reported (CF_OVERFLOW) rather than
                           * silently truncated. mz_split's stack peaks at 3
                           * for a 3-way split, which its `sp + 2 > CF_MAXFAC`
                           * guard admits exactly. Raising this to allow 4LP
                           * means revisiting that guard, not just this line.
                           *
                           * NOTE this is a count of PARTS, not of limbs, and
                           * it is unchanged by the 4-limb work below: a 4-limb
                           * cofactor is still split into at most 3 large
                           * primes (128/34 = 3.8, and no job asks for 4LP).
                           * The two knobs are independent -- CF_LMAX widens
                           * the residual, CF_MAXFAC counts what comes out. */

/* The cofactor WIDTH machinery -- CF_LMAX, CF_LMIN and cf_limbs_for_mfb --
 * lives in bench.h, because bench_main.cu has to refuse an out-of-range mfb
 * and resolve --cof-limbs without pulling in a header full of device code.
 * Everything that USES the width is here; only the three names are there. */

#if defined(__CUDACC__)
#define CF_FN __device__ __forceinline__
#define CF_NOINLINE __device__ __noinline__
#define CF_HD static __host__ __device__ __forceinline__
#else
#define CF_FN static inline
#define CF_NOINLINE static
#define CF_HD static inline
#endif

/* ---- L-limb arithmetic ------------------------------------------------- *
 *
 * Templated on the limb count so the rational side pays for 64 bits and the
 * algebraic side for 96, rather than both paying prp.cuh's fixed 128. At three
 * limbs the CIOS inner loop is 9 multiply-adds against 16 -- the modmul is the
 * whole cost of this stage, so the limb count is not a detail.
 */

template <int L> struct mz { uint32_t v[L]; };

template <int L> CF_FN int mz_cmp(const mz<L> *a, const mz<L> *b)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
    return 0;
}

template <int L> CF_FN int mz_is_zero(const mz<L> *a)
{
    uint32_t o = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> CF_FN int mz_is_one(const mz<L> *a)
{
    uint32_t o = a->v[0] ^ 1u;
    #pragma unroll
    for (int i = 1; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> CF_FN uint32_t mz_sub(mz<L> *a, const mz<L> *b)
{
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t d = (uint64_t)a->v[i] - b->v[i] - borrow;
        a->v[i] = (uint32_t)d;
        borrow = (d >> 32) & 1u;
    }
    return (uint32_t)borrow;
}

/* a = a - b mod n, given a, b < n */
template <int L> CF_FN void mz_sub_mod(mz<L> *a, const mz<L> *b, const mz<L> *n)
{
    if (mz_sub<L>(a, b)) {
        uint64_t c = 0;
        #pragma unroll
        for (int i = 0; i < L; i++) {
            uint64_t s = (uint64_t)a->v[i] + n->v[i] + c;
            a->v[i] = (uint32_t)s;
            c = s >> 32;
        }
    }
}

/* a = a + b mod n, given a, b < n */
template <int L> CF_FN void mz_add_mod(mz<L> *a, const mz<L> *b, const mz<L> *n)
{
    uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t s = (uint64_t)a->v[i] + b->v[i] + c;
        a->v[i] = (uint32_t)s;
        c = s >> 32;
    }
    if (c || mz_cmp<L>(a, n) >= 0) mz_sub<L>(a, n);
}

/* a = 2a mod n */
template <int L> CF_FN void mz_dbl_mod(mz<L> *a, const mz<L> *n)
{
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint32_t hi = a->v[i] >> 31;
        a->v[i] = (a->v[i] << 1) | carry;
        carry = hi;
    }
    if (carry || mz_cmp<L>(a, n) >= 0) mz_sub<L>(a, n);
}

template <int L> CF_FN void mz_rshift1(mz<L> *a)
{
    #pragma unroll
    for (int i = 0; i < L - 1; i++)
        a->v[i] = (a->v[i] >> 1) | (a->v[i + 1] << 31);
    a->v[L - 1] >>= 1;
}

template <int L> CF_FN int mz_bits(const mz<L> *a)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i]) return i * 32 + (32 - __clz(a->v[i]));
    return 0;
}

/* -n^{-1} mod 2^32, Newton. n odd. */
CF_FN uint32_t mz_n0inv(uint32_t n0)
{
    uint32_t inv = n0;
    for (int k = 0; k < 5; k++) inv *= 2u - n0 * inv;
    return (uint32_t)(0u - inv);
}

/* r = a*b*R^-1 mod n, R = 2^(32L). CIOS. */
template <int L>
CF_FN void mz_mul(mz<L> *r, const mz<L> *a, const mz<L> *b,
                  const mz<L> *n, uint32_t n0inv)
{
    uint32_t t[L + 2];
    #pragma unroll
    for (int i = 0; i < L + 2; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t C = 0;
        #pragma unroll
        for (int j = 0; j < L; j++) {
            uint64_t p = (uint64_t)a->v[j] * b->v[i] + t[j] + C;
            t[j] = (uint32_t)p;
            C = p >> 32;
        }
        {
            uint64_t p = (uint64_t)t[L] + C;
            t[L] = (uint32_t)p;
            t[L + 1] = (uint32_t)(p >> 32);
        }
        {
            uint32_t m = t[0] * n0inv;
            uint64_t p = (uint64_t)m * n->v[0] + t[0];
            C = p >> 32;
            #pragma unroll
            for (int j = 1; j < L; j++) {
                p = (uint64_t)m * n->v[j] + t[j] + C;
                t[j - 1] = (uint32_t)p;
                C = p >> 32;
            }
            p = (uint64_t)t[L] + C;
            t[L - 1] = (uint32_t)p;
            t[L] = t[L + 1] + (uint32_t)(p >> 32);
        }
    }
    #pragma unroll
    for (int i = 0; i < L; i++) r->v[i] = t[i];
    if (t[L] || mz_cmp<L>(r, n) >= 0) mz_sub<L>(r, n);
}

/* Binary GCD. Both arguments are destroyed. n is odd, which removes half the
 * usual bookkeeping: the common power of two is always 1. */
template <int L> CF_FN void mz_gcd(mz<L> *g, mz<L> a, mz<L> b)
{
    if (mz_is_zero<L>(&a)) { *g = b; return; }
    while ((a.v[0] & 1u) == 0u) mz_rshift1<L>(&a);
    for (;;) {
        while ((b.v[0] & 1u) == 0u) mz_rshift1<L>(&b);
        if (mz_cmp<L>(&a, &b) > 0) { mz<L> t = a; a = b; b = t; }
        mz_sub<L>(&b, &a);                       /* b >= a, so no borrow */
        if (mz_is_zero<L>(&b)) { *g = a; return; }
    }
}

/* q = n / d, EXACT. Hensel: d is odd, so d has an inverse mod 2^(32L), and for
 * an exact quotient the low 32L bits of n * d^-1 are the whole answer. This is
 * the only division in the stage, and it is a few multiplies. */
template <int L> CF_FN void mz_divexact(mz<L> *q, const mz<L> *n, const mz<L> *d)
{
    mz<L> inv, t;
    uint32_t i0 = d->v[0];
    uint32_t x = i0;
    for (int k = 0; k < 5; k++) x *= 2u - i0 * x;   /* d^-1 mod 2^32 */
    #pragma unroll
    for (int i = 0; i < L; i++) inv.v[i] = 0;
    inv.v[0] = x;
    /* lift to 2^(32L): inv <- inv * (2 - d*inv), low L limbs only */
    for (int step = 1; step < L; step <<= 1) {
        mz<L> p;
        #pragma unroll
        for (int i = 0; i < L; i++) p.v[i] = 0;
        #pragma unroll
        for (int i = 0; i < L; i++) {           /* p = d*inv mod 2^(32L) */
            uint64_t C = 0;
            #pragma unroll
            for (int j = 0; i + j < L; j++) {
                uint64_t s = (uint64_t)d->v[j] * inv.v[i] + p.v[i + j] + C;
                p.v[i + j] = (uint32_t)s;
                C = s >> 32;
            }
        }
        {   /* p = 2 - p, mod 2^(32L) */
            uint64_t borrow = 0;
            mz<L> two;
            #pragma unroll
            for (int i = 0; i < L; i++) two.v[i] = 0;
            two.v[0] = 2u;
            #pragma unroll
            for (int i = 0; i < L; i++) {
                uint64_t s = (uint64_t)two.v[i] - p.v[i] - borrow;
                p.v[i] = (uint32_t)s;
                borrow = (s >> 32) & 1u;
            }
        }
        #pragma unroll
        for (int i = 0; i < L; i++) t.v[i] = 0;
        #pragma unroll
        for (int i = 0; i < L; i++) {           /* t = inv*p mod 2^(32L) */
            uint64_t C = 0;
            #pragma unroll
            for (int j = 0; i + j < L; j++) {
                uint64_t s = (uint64_t)inv.v[j] * p.v[i] + t.v[i + j] + C;
                t.v[i + j] = (uint32_t)s;
                C = s >> 32;
            }
        }
        inv = t;
    }
    #pragma unroll
    for (int i = 0; i < L; i++) q->v[i] = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {               /* q = n*inv mod 2^(32L) */
        uint64_t C = 0;
        #pragma unroll
        for (int j = 0; i + j < L; j++) {
            uint64_t s = (uint64_t)n->v[j] * inv.v[i] + q->v[i + j] + C;
            q->v[i + j] = (uint32_t)s;
            C = s >> 32;
        }
    }
}

/* Strong probable prime test, base 2, entirely in the Montgomery domain.
 * Same argument as prp.cuh: a composite called prime loses a relation at a rate
 * around 2^-40, it never emits one with a composite ideal. */
template <int L> CF_FN int mz_sprp2(const mz<L> *n)
{
    mz<L> one, two, nm1, x, base, d;
    uint32_t n0inv = mz_n0inv(n->v[0]);
    int r = 0, dbits;

    #pragma unroll
    for (int i = 0; i < L; i++) one.v[i] = 0;
    one.v[0] = 1;
    for (int i = 0; i < 32 * L; i++) mz_dbl_mod<L>(&one, n);   /* R mod n */
    two = one; mz_dbl_mod<L>(&two, n);

    nm1 = *n; { mz<L> t = one; mz_sub<L>(&nm1, &t); }

    d = *n; d.v[0] -= 1u;
    while ((d.v[0] & 1u) == 0u) { mz_rshift1<L>(&d); r++; }
    dbits = mz_bits<L>(&d);

    x = one; base = two;
    for (int i = dbits - 1; i >= 0; i--) {
        mz<L> t;
        mz_mul<L>(&t, &x, &x, n, n0inv); x = t;
        if ((d.v[i >> 5] >> (i & 31)) & 1u) { mz_mul<L>(&t, &x, &base, n, n0inv); x = t; }
    }
    if (mz_cmp<L>(&x, &one) == 0 || mz_cmp<L>(&x, &nm1) == 0) return 1;
    for (int i = 1; i < r; i++) {
        mz<L> t;
        mz_mul<L>(&t, &x, &x, n, n0inv); x = t;
        if (mz_cmp<L>(&x, &nm1) == 0) return 1;
    }
    return 0;
}

/* ---- Pollard-Brent rho, with a hard iteration budget ------------------- *
 *
 * The whole loop runs in the Montgomery domain and never leaves it. f(y) =
 * y^2 + c maps Montgomery representatives to Montgomery representatives, and
 * the accumulated product carries a power of R that gcd cannot see: gcd(qR^k,
 * n) = gcd(q, n) because n is odd and R is a power of two. So there is no
 * conversion in or out anywhere in the hot loop.
 *
 * `budget` bounds the iteration count so a warp's cost is bounded by the
 * budget rather than by its unluckiest lane. Returning 0 means "not yet",
 * never "cannot".
 */
#define CF_BATCH 128       /* modmuls between gcds; the gcd is ~L*192 ops, so
                            * batching it under the modmuls keeps it off the
                            * critical path without delaying detection much */

/* `acc` accumulates iterations actually consumed, for the instrumentation that
 * asks where rho's time goes (dead records vs splitting ones). The re-walk in
 * the `found` tail is not counted: it is at most CF_BATCH steps against a
 * budget of tens of thousands. */
template <int L>
CF_FN int mz_rho(mz<L> *fac, const mz<L> *n, uint32_t c0, uint32_t budget,
                 uint64_t *acc)
{
    const uint32_t n0inv = mz_n0inv(n->v[0]);
    mz<L> one, y, x, ys, q, g, c;
    uint32_t steps = 0, r = 1;

    #pragma unroll
    for (int i = 0; i < L; i++) one.v[i] = 0;
    one.v[0] = 1;
    for (int i = 0; i < 32 * L; i++) mz_dbl_mod<L>(&one, n);   /* R mod n */

    c = one;
    for (uint32_t k = 1; k < c0; k++) mz_add_mod<L>(&c, &one, n);  /* c0 * R */
    y = one; mz_add_mod<L>(&y, &one, n);                            /* 2 * R */
    q = one;
    x = y; ys = y;

    for (;;) {
        uint32_t k = 0;
        x = y;
        for (uint32_t i = 0; i < r; i++) {
            mz<L> t;
            mz_mul<L>(&t, &y, &y, n, n0inv);
            mz_add_mod<L>(&t, &c, n);
            y = t;
        }
        steps += r;
        while (k < r) {
            uint32_t lim = r - k; if (lim > CF_BATCH) lim = CF_BATCH;
            ys = y;
            for (uint32_t i = 0; i < lim; i++) {
                mz<L> t, d;
                mz_mul<L>(&t, &y, &y, n, n0inv);
                mz_add_mod<L>(&t, &c, n);
                y = t;
                d = x; mz_sub_mod<L>(&d, &y, n);
                if (mz_is_zero<L>(&d)) { d = one; }   /* cycle closed on x */
                mz_mul<L>(&t, &q, &d, n, n0inv);
                q = t;
            }
            k += lim; steps += lim;
            mz_gcd<L>(&g, q, *n);
            if (!mz_is_one<L>(&g)) goto found;
            if (steps >= budget) { *acc += steps; return 0; }
        }
        r <<= 1;
        if (steps >= budget) { *acc += steps; return 0; }
    }

found:
    if (mz_cmp<L>(&g, n) == 0) {
        /* the batch swallowed the factor: re-walk it one step at a time */
        y = ys;
        for (uint32_t i = 0; i < CF_BATCH; i++) {
            mz<L> t, d;
            mz_mul<L>(&t, &y, &y, n, n0inv);
            mz_add_mod<L>(&t, &c, n);
            y = t;
            d = x; mz_sub_mod<L>(&d, &y, n);
            if (mz_is_zero<L>(&d)) { *acc += steps; return 0; }  /* degenerate */
            mz_gcd<L>(&g, d, *n);
            if (!mz_is_one<L>(&g)) break;
        }
        if (mz_is_one<L>(&g) || mz_cmp<L>(&g, n) == 0) { *acc += steps; return 0; }
    }
    *fac = g;
    *acc += steps;
    return 1;
}

/* ---- ECM, stages 1 and 2 ------------------------------------------------ *
 *
 * Montgomery curves By^2 = x^3 + Ax^2 + x in XZ coordinates, which is the form
 * whose differential addition needs no y and no inversion.
 *
 * The curve constant is kept PROJECTIVE, A24 = (an : ad). Suyama's
 * parameterisation puts A24 = (v-u)^3 (3u+v) / (16 u^3 v), and carrying that as
 * a fraction costs one extra multiply per doubling -- 12 modmuls per ladder bit
 * against 11 -- while removing a modular inversion that would otherwise be paid
 * once per curve per record. At tens of curves per record that trade is clearly
 * the right way round, and it keeps a binary extended GCD out of the file.
 *
 * Stage 2 is the small-table Montgomery-XZ continuation below. It deliberately
 * uses D=30: the four unsigned baby residues {1,7,11,13} keep one curve's
 * private table small enough for this first GPU experiment. The host supplies
 * one four-bit mask per giant step; bit u is set when vD-u or vD+u is prime in
 * (B1,B2]. Thus every lane follows the same giant-step walk and only the cheap
 * product-of-differences loop is masked.
 */

#define CF_ECM_D       30u
#define CF_ECM_NBABY    4u
#define CF_ECM_MAX_B1   1000000u
#define CF_ECM_MAX_B2  10000000u

/* One definition supplies both the host mask-bit mapping and the device baby
 * scalar. A duplicated host/device table can compile and test cleanly while
 * silently assigning different meanings to the same mask bit. */
CF_HD uint8_t cf_ecm_baby(uint32_t k)
{
    return k == 0 ? 1u : k == 1 ? 7u : k == 2 ? 11u : 13u;
}

/* Shared composite sieve for both ECM plans. Bounds live here as well as in
 * the CLI: these helpers are also called by tests and must be safe on their
 * own. */
static uint8_t *cf_ecm_sieve(uint32_t limit)
{
    uint8_t *composite;
    if (limit < 2 || limit > CF_ECM_MAX_B2) return NULL;
    composite = (uint8_t *)calloc((size_t)limit + 1, 1);
    if (!composite) return NULL;
    for (uint32_t p = 2; (uint64_t)p * p <= limit; p++)
        if (!composite[p])
            for (uint64_t q = (uint64_t)p * p; q <= limit; q += p)
                composite[q] = 1;
    return composite;
}

/* The stage-1 exponent lcm(1..B1), held as its prime powers rather than as one
 * big integer: each is at most B1, so the ladder for it is a dozen bits and the
 * whole schedule is an array of uint32 instead of a 1.4*B1-bit number the device
 * would have to store and shift. Returns the count. */
static uint32_t cf_ecm_plan(uint32_t b1, uint32_t **out)
{
    uint32_t n = 0, cap = 0, *s = NULL;
    uint8_t *sieve;
    if (b1 < 2 || b1 > CF_ECM_MAX_B1) { *out = NULL; return 0; }
    sieve = cf_ecm_sieve(b1);
    if (!sieve) { *out = NULL; return 0; }
    for (uint32_t p = 2; p <= b1; p++) {
        if (sieve[p]) continue;
        {   /* the largest power of p that still fits under B1 */
            uint64_t pe = p;
            while (pe * p <= b1) pe *= p;
            if (n == cap) {
                uint32_t *grown;
                cap = cap ? cap * 2 : 256;
                grown = (uint32_t *)realloc(s, (size_t)cap * sizeof(*s));
                if (!grown) { free(s); free(sieve); *out = NULL; return 0; }
                s = grown;
            }
            s[n++] = (uint32_t)pe;
        }
    }
    free(sieve);
    *out = s;
    return n;
}

/* Build the stage-2 prime-pair schedule. For every prime q in (B1,B2], write
 * q = vD +/- u with 0 < u < D/2. Since q > B1 >= 30 in the stage-2 path, q is
 * coprime to D and u is one of the four residues below. One x-coordinate
 * comparison covers both signs. Empty giant steps remain in the schedule so
 * the device can advance with one differential addition rather than restart a
 * scalar multiplication at every populated v. */
static uint32_t cf_ecm_stage2_plan(uint32_t b1, uint32_t b2, uint32_t *vmin_out,
                                   uint8_t **mask_out)
{
    uint8_t *composite = NULL, *mask = NULL;
    uint32_t qlo, qhi, vmin, vmax, nv;
    if (b2 <= b1 || b1 < CF_ECM_D || b1 > CF_ECM_MAX_B1 ||
        b2 > CF_ECM_MAX_B2) {
        *vmin_out = 0; *mask_out = NULL; return 0;
    }
    composite = cf_ecm_sieve(b2);
    if (!composite) { *vmin_out = 0; *mask_out = NULL; return 0; }
    for (qlo = b1 + 1; qlo <= b2 && composite[qlo]; qlo++) {}
    if (qlo > b2) {
        free(composite); *vmin_out = 0; *mask_out = NULL; return 0;
    }
    for (qhi = b2; qhi > qlo && composite[qhi]; qhi--) {}
    vmin = (qlo + CF_ECM_D / 2) / CF_ECM_D;
    vmax = (qhi + CF_ECM_D / 2) / CF_ECM_D;
    nv = vmax - vmin + 1;
    mask = (uint8_t *)calloc(nv, 1);
    if (!mask) { free(composite); *vmin_out = 0; *mask_out = NULL; return 0; }
    for (uint32_t q = b1 + 1; q <= b2; q++) if (!composite[q]) {
        const uint32_t v = (q + CF_ECM_D / 2) / CF_ECM_D;
        const uint32_t vd = v * CF_ECM_D;
        const uint32_t u = q > vd ? q - vd : vd - q;
        for (uint32_t k = 0; k < CF_ECM_NBABY; k++)
            if (u == cf_ecm_baby(k)) {
                mask[v - vmin] |= (uint8_t)(1u << k); break;
            }
    }
    free(composite);
    *vmin_out = vmin; *mask_out = mask;
    return nv;
}

template <int L> struct mpt { mz<L> X, Z; };

/* r = k * one, for a small k, by double-and-add on the Montgomery form of 1. */
template <int L>
CF_FN void mz_small(mz<L> *r, uint32_t k, const mz<L> *one, const mz<L> *n)
{
    int b = 31;
    #pragma unroll
    for (int i = 0; i < L; i++) r->v[i] = 0;
    if (!k) return;
    while (b > 0 && !((k >> b) & 1u)) b--;
    *r = *one;
    for (b--; b >= 0; b--) {
        mz_dbl_mod<L>(r, n);
        if ((k >> b) & 1u) mz_add_mod<L>(r, one, n);
    }
}

/* 2P. r must not alias p. */
template <int L>
CF_FN void x_dbl(mpt<L> *r, const mpt<L> *p, const mz<L> *an, const mz<L> *ad,
                 const mz<L> *n, uint32_t n0inv)
{
    mz<L> t1, t2, s1, s2, d, w, z;
    t1 = p->X; mz_add_mod<L>(&t1, &p->Z, n);
    t2 = p->X; mz_sub_mod<L>(&t2, &p->Z, n);
    mz_mul<L>(&s1, &t1, &t1, n, n0inv);            /* (X+Z)^2 */
    mz_mul<L>(&s2, &t2, &t2, n, n0inv);            /* (X-Z)^2 */
    d = s1; mz_sub_mod<L>(&d, &s2, n);             /* 4XZ     */
    mz_mul<L>(&t1, ad, &s1, n, n0inv);
    mz_mul<L>(&r->X, &t1, &s2, n, n0inv);          /* ad*(X+Z)^2*(X-Z)^2 */
    mz_mul<L>(&t2, ad, &s2, n, n0inv);
    mz_mul<L>(&w, an, &d, n, n0inv);
    z = t2; mz_add_mod<L>(&z, &w, n);
    mz_mul<L>(&r->Z, &d, &z, n, n0inv);
}

/* P+Q given the difference D = P-Q. r must not alias p, q or d. */
template <int L>
CF_FN void x_add(mpt<L> *r, const mpt<L> *p, const mpt<L> *q, const mpt<L> *dif,
                 const mz<L> *n, uint32_t n0inv)
{
    mz<L> a, b, c, e, t1, t2, s, t;
    a = p->X; mz_add_mod<L>(&a, &p->Z, n);
    b = p->X; mz_sub_mod<L>(&b, &p->Z, n);
    c = q->X; mz_add_mod<L>(&c, &q->Z, n);
    e = q->X; mz_sub_mod<L>(&e, &q->Z, n);
    mz_mul<L>(&t1, &b, &c, n, n0inv);              /* (XP-ZP)(XQ+ZQ) */
    mz_mul<L>(&t2, &a, &e, n, n0inv);              /* (XP+ZP)(XQ-ZQ) */
    s = t1; mz_add_mod<L>(&s, &t2, n);
    t = t1; mz_sub_mod<L>(&t, &t2, n);
    mz_mul<L>(&a, &s, &s, n, n0inv);
    mz_mul<L>(&b, &t, &t, n, n0inv);
    mz_mul<L>(&r->X, &dif->Z, &a, n, n0inv);
    mz_mul<L>(&r->Z, &dif->X, &b, n, n0inv);
}

/* P = [k]P by the Montgomery ladder. k is a prime power <= B1, so this is a
 * dozen bits at most and the ladder's constant-time shape costs nothing. */
template <int L>
CF_FN void x_mul(mpt<L> *p, uint32_t k, const mz<L> *an, const mz<L> *ad,
                 const mz<L> *n, uint32_t n0inv, uint64_t *work)
{
    mpt<L> r0 = *p, r1, t;
    int b = 31;
    if (k < 2) { if (!k) { for (int i = 0; i < L; i++) { p->X.v[i] = 0; p->Z.v[i] = 0; } } return; }
    while (b > 0 && !((k >> b) & 1u)) b--;
    x_dbl<L>(&r1, p, an, ad, n, n0inv);
    for (b--; b >= 0; b--) {
        if ((k >> b) & 1u) {
            x_add<L>(&t, &r1, &r0, p, n, n0inv); r0 = t;
            x_dbl<L>(&t, &r1, an, ad, n, n0inv);  r1 = t;
        } else {
            x_add<L>(&t, &r1, &r0, p, n, n0inv); r1 = t;
            x_dbl<L>(&t, &r0, an, ad, n, n0inv);  r0 = t;
        }
        (*work)++;
    }
    *p = r0;
}

/* Keep one copy of the ladder behind a call boundary. Force-inlining x_mul at
 * every baby/giant setup site makes nvcc expand seven copies of the bigint
 * arithmetic into this already-large kernel and markedly increases both
 * compile time and register pressure. */
template <int L>
CF_NOINLINE void x_mul_s2(mpt<L> *p, uint32_t k, const mz<L> *an,
                          const mz<L> *ad, const mz<L> *n, uint32_t n0inv,
                          uint64_t *work)
{
    x_mul<L>(p, k, an, ad, n, n0inv, work);
}

/* One pass over the D=30 baby-step/giant-step continuation. All selected
 * cross-products are accumulated and normally only one gcd is paid. If hits
 * for different prime factors make that gcd equal n, replay the same curve and
 * gcd each selected difference. The replay is rare, preserves the fast common
 * path, and prevents a batched product from swallowing every factor it found. */
template <int L>
CF_NOINLINE int mz_ecm_stage2_pass(mz<L> *fac, const mpt<L> *Q,
                                    const mz<L> *an, const mz<L> *ad,
                                    const mz<L> *one, const mz<L> *n,
                                    uint32_t n0inv,
                                    const uint8_t *__restrict masks,
                                    uint32_t vmin, uint32_t nv,
                                    uint64_t *work)
{
    /* The baby steps are held on a SHARED DENOMINATOR: bx[k] = X_k *
     * prod_{j!=k} Z_j against one bz = prod_j Z_j, so (bx[k] : bz) is the
     * same projective point (X_k : Z_k) was. Every cross product below is
     * then the old one scaled by prod_{j!=k} Z_j.
     *
     * That scaling leaves gcd(d, n) alone only when the scale factor is a UNIT
     * mod n. It normally is -- cofcheck.sh's pinned counts match and a 148-q
     * A/B produced identical relations either way -- but that is empirical,
     * NOT a proof, and an earlier version of this comment wrongly claimed the
     * factors were provably identical.
     *
     * Note the BLAST RADIUS when it is not. Every bx[k] except k's own carries
     * Z_k as a factor, so ONE baby step whose Z shares a factor f with n
     * contaminates the cross product of every OTHER k. All of those gcds
     * become multiples of f, which may surface f earlier than the per-point
     * form would, or may grow to exactly n and be discarded by the g != n
     * guard -- losing chances the per-point form kept independent. Z_k == 0
     * mod n is the extreme: every other k's d collapses to 0 and is rejected
     * outright, while k's own term survives. The per-point form has no such
     * coupling between k.
     *
     * It needs a baby-step Z sharing a factor with n (the lucky-hit case
     * stage 1 usually catches), and it can NEVER return a wrong factor: any
     * gcd with n is a true divisor, and g == n is rejected. That is what
     * makes this safe -- "never returns a non-factor", not "same factors".
     *
     * The point is the inner loop: with one denominator for every baby step,
     * the X_G * Z term is common to all k and hoists out, so a selected pair
     * costs ONE multiply instead of two. At B1=200/B2=6000 that is ~540 fewer
     * mz_mul per curve against the rescale's one-off cost -- 4*NBABY mz_mul
     * issued, two per k in each of the forward and backward passes. Three of
     * those are provably redundant (both passes open by multiplying into
     * acc == one, which is the identity in Montgomery form, and the backward
     * pass's final acc update is never read), so 4*NBABY - 3 do real work.
     * Peeling them is left alone deliberately: 3 multiplies per replay is
     * ~1% of the ~540 this change saves, and it is not worth restructuring
     * delicate modular arithmetic for. Measured
     * A/B on gfx1103 (oracle/c183, 148 q, only this function differing):
     * algebraic queue 52.62/52.18 -> 50.19/49.75 ms, ~4.6%, with the rational
     * (rho) queue unchanged as the control and 596 relations either way.
     *
     * Chosen over per-point affine normalisation deliberately: affine needs a
     * modular inverse, there is no mz_inv in this file, and a binary extended
     * GCD in device code is new delicate arithmetic for no extra gain here.
     * The shared denominator needs no inversion at all.
     *
     * NOT a register-pressure fix, and not why RDNA1/2 lost stage 2 -- every
     * stage-2 kernel is already at vgpr=128, so trimming live values relocates
     * spill rather than freeing registers (measured: scratch 336 -> 384 on
     * gfx1030 L=3). That was the gfx10 MUBUF stack lowering; see CLAUDE.md. */
    mz<L> bx[CF_ECM_NBABY], bz;
    mpt<L> step, prev, cur, next;
    mz<L> prod, g;

    for (int replay = 0; replay < 2; replay++) {
        prod = *one;
        {
            /* Each baby step is built in ONE scratch point, then split into
             * bx[k] and bzs[k]; bx[] doubles as the running accumulator, so
             * no separate prefix array is needed.
             *
             * This does NOT reduce peak storage, and an earlier version of
             * this comment wrongly implied it did: an mpt<L> is (X, Z), so
             * the baby[CF_ECM_NBABY] this replaced was already 2*NBABY mz
             * values -- exactly what bx[] + bzs[] is. With p, acc, t and gx
             * live on top, the peak is slightly HIGHER, which is what the
             * measured scratch 336 -> 384 below is. Deliberately NOT unrolled --
             * x_mul_s2 is __noinline__, and unrolling its call sites widens
             * the live range of every argument across all NBABY calls. */
            mz<L> bzs[CF_ECM_NBABY], acc, t;
            for (uint32_t k = 0; k < CF_ECM_NBABY; k++) {
                const uint32_t scalar = cf_ecm_baby(k);
                mpt<L> p = *Q;
                if (scalar != 1)
                    x_mul_s2<L>(&p, scalar, an, ad, n, n0inv, work);
                bx[k] = p.X; bzs[k] = p.Z;
            }
            /* Forward: bx[k] = X_k * prod_{j<k} Z_j, acc ends at prod_j Z_j. */
            acc = *one;
            for (uint32_t k = 0; k < CF_ECM_NBABY; k++) {
                mz_mul<L>(&t, &bx[k], &acc, n, n0inv); bx[k] = t;
                mz_mul<L>(&t, &acc, &bzs[k], n, n0inv); acc = t;
            }
            bz = acc;
            /* Backward: acc holds prod_{j>k} Z_j, so this completes each
             * bx[k] to X_k * prod_{j!=k} Z_j. */
            acc = *one;
            for (int k = (int)CF_ECM_NBABY - 1; k >= 0; k--) {
                mz_mul<L>(&t, &bx[k], &acc, n, n0inv); bx[k] = t;
                mz_mul<L>(&t, &acc, &bzs[k], n, n0inv); acc = t;
            }
        }
        step = *Q; x_mul_s2<L>(&step, CF_ECM_D, an, ad, n, n0inv, work);
        prev = step; x_mul_s2<L>(&prev, vmin, an, ad, n, n0inv, work);
        if (nv > 1) {
            cur = step;
            x_mul_s2<L>(&cur, vmin + 1, an, ad, n, n0inv, work);
        }

        for (uint32_t vi = 0; vi < nv; vi++) {
            const mpt<L> *G = vi ? &cur : &prev;
            const uint32_t mask = masks[vi];
            if (mask) {
                /* One denominator for every baby step, so this term is the
                 * same for all k -- hoisted out of the loop below, and not
                 * computed at all on a giant step that selects nothing. */
                mz<L> gx;
                mz_mul<L>(&gx, &G->X, &bz, n, n0inv);
                #pragma unroll
                for (uint32_t k = 0; k < CF_ECM_NBABY; k++)
                    if (mask & (1u << k)) {
                        mz<L> b, d;
                        mz_mul<L>(&b, &bx[k], &G->Z, n, n0inv);
                        d = gx; mz_sub_mod<L>(&d, &b, n);
                        (*work)++;              /* one selected stage-2 pair */
                        if (replay) {
                            mz_gcd<L>(&g, d, *n);
                            if (!mz_is_one<L>(&g) && mz_cmp<L>(&g, n) != 0) {
                                *fac = g; return 1;
                            }
                        } else {
                            mz_mul<L>(&b, &prod, &d, n, n0inv); prod = b;
                        }
                    }
            }
            if (vi + 1 < nv && vi) {
                x_add<L>(&next, &cur, &step, &prev, n, n0inv);
                (*work)++;                       /* one stage-2 giant step */
                prev = cur; cur = next;
            }
        }
        if (replay) return 0;
        mz_gcd<L>(&g, prod, *n);
        if (mz_is_one<L>(&g)) return 0;
        if (mz_cmp<L>(&g, n) != 0) { *fac = g; return 1; }
    }
    return 0;
}

/* One curve of stage 1. `s` holds the prime powers p^e <= B1, ascending.
 * Returns 1 and a nontrivial factor when gcd(Z, n) splits n. */
template <int L, int STAGE2>
CF_FN int mz_ecm(mz<L> *fac, const mz<L> *n, uint32_t sigma,
                 const uint32_t *__restrict s, uint32_t ns,
                 const uint8_t *__restrict s2mask, uint32_t s2vmin,
                 uint32_t s2nv, uint64_t *work)
{
    const uint32_t n0inv = mz_n0inv(n->v[0]);
    mz<L> one, sig, u, v, t, w, an, ad, g;
    mpt<L> P;

    #pragma unroll
    for (int i = 0; i < L; i++) one.v[i] = 0;
    one.v[0] = 1;
    for (int i = 0; i < 32 * L; i++) mz_dbl_mod<L>(&one, n);      /* R mod n */

    mz_small<L>(&sig, sigma, &one, n);
    mz_mul<L>(&u, &sig, &sig, n, n0inv);                          /* sigma^2 */
    mz_small<L>(&t, 5, &one, n);
    mz_sub_mod<L>(&u, &t, n);                                     /* u = s^2-5 */
    v = sig; mz_dbl_mod<L>(&v, n); mz_dbl_mod<L>(&v, n);          /* v = 4s   */
    if (mz_is_zero<L>(&u) || mz_is_zero<L>(&v)) return 0;

    mz_mul<L>(&t, &u, &u, n, n0inv);
    mz_mul<L>(&P.X, &t, &u, n, n0inv);                            /* x0 = u^3 */
    mz_mul<L>(&t, &v, &v, n, n0inv);
    mz_mul<L>(&P.Z, &t, &v, n, n0inv);                            /* z0 = v^3 */

    t = v; mz_sub_mod<L>(&t, &u, n);                              /* v-u      */
    mz_mul<L>(&w, &t, &t, n, n0inv);
    mz_mul<L>(&an, &w, &t, n, n0inv);                             /* (v-u)^3  */
    mz_small<L>(&t, 3, &one, n);
    mz_mul<L>(&w, &t, &u, n, n0inv);
    mz_add_mod<L>(&w, &v, n);                                     /* 3u+v     */
    mz_mul<L>(&t, &an, &w, n, n0inv); an = t;                     /* numerator*/

    mz_small<L>(&t, 16, &one, n);
    mz_mul<L>(&w, &t, &P.X, n, n0inv);                            /* 16*u^3   */
    mz_mul<L>(&ad, &w, &v, n, n0inv);                             /* *v       */
    if (mz_is_zero<L>(&ad)) return 0;

    for (uint32_t i = 0; i < ns; i++)
        x_mul<L>(&P, s[i], &an, &ad, n, n0inv, work);

    mz_gcd<L>(&g, P.Z, *n);
    if (mz_is_one<L>(&g)) {
        if constexpr (STAGE2) {
            if (mz_ecm_stage2_pass<L>(fac, &P, &an, &ad, &one, n, n0inv,
                                      s2mask, s2vmin, s2nv, work)) return 1;
        }
        return 0;
    }
    if (mz_cmp<L>(&g, n) == 0) return 0;
    *fac = g;
    return 1;
}

/* ---- full split of one cofactor ---------------------------------------- *
 *
 * `lim2` is lim^2 for the side. Anything below it is PRIME by construction --
 * trial division removed every factor below lim, so a composite needs two of
 * them -- which is why this makes far fewer primality calls than a general
 * cofactorizer would. On this job that threshold is 2^54 on the algebraic side,
 * so only the middle part of a 3LP split is ever tested.
 */
/* METHOD 0 is Pollard-Brent rho and `budget` is an iteration count; METHOD 1 is
 * ECM and `budget` is a CURVE count. Both are bounded per launch and
 * neither turns an exhausted budget into a proof, so the requeue structure
 * around them is unchanged. */
template <int L, int METHOD, int STAGE2>
CF_FN int mz_split(const mz<L> *n0, const mz<L> *lim2, uint32_t lpb,
                   uint32_t c0, uint32_t budget, uint64_t *out, int *nout,
                   uint64_t *acc, const uint32_t *__restrict s, uint32_t ns,
                   const uint8_t *__restrict s2mask, uint32_t s2vmin,
                   uint32_t s2nv)
{
    mz<L> st[CF_MAXFAC];
    int sp = 0, nf = 0;
    st[sp++] = *n0;
    while (sp) {
        mz<L> m = st[--sp], g, h;
        if (mz_is_one<L>(&m)) continue;
        if (mz_cmp<L>(&m, lim2) < 0) {              /* prime by size */
            if ((uint32_t)mz_bits<L>(&m) > lpb) return CF_DEAD;
            if (nf >= CF_MAXFAC) return CF_OVERFLOW;
            /* Two limbs. The bound just checked is lpb, not 32, so at
             * lpb > 32 this factor legitimately needs 64 bits; m.v[0] alone
             * used to emit its low half. L is 2 or 3 at every instantiation,
             * but the guard costs nothing and makes the assumption explicit. */
            if constexpr (L >= 2) out[nf++] = ((uint64_t)m.v[1] << 32) | m.v[0];
            else                  out[nf++] = (uint64_t)m.v[0];
            continue;
        }
        if (mz_sprp2<L>(&m)) return CF_DEAD;        /* prime, and >= lim2 > 2^lpb */
        if constexpr (METHOD == 0) {
            if (!mz_rho<L>(&g, &m, c0, budget, acc)) return CF_INCOMPLETE;
        } else {
            uint32_t cv = 0;
            for (; cv < budget; cv++)
                if (mz_ecm<L, STAGE2>(&g, &m, c0 * 1000u + cv + 6u, s, ns,
                                      s2mask, s2vmin, s2nv, acc)) break;
            if (cv == budget) return CF_INCOMPLETE;
        }
        mz_divexact<L>(&h, &m, &g);
        if (sp + 2 > CF_MAXFAC) return CF_OVERFLOW;
        st[sp++] = g; st[sp++] = h;
    }
    *nout = nf;
    return CF_OK;
}

#if defined(__CUDACC__)

/* ---- the kernel -------------------------------------------------------- *
 *
 * One thread per job, and a job that runs out of budget keeps its slot: the
 * host relaunches with a different rho constant and a larger budget, and a
 * thread whose status is already resolved falls straight through. Compacting
 * between rounds would save the fall-through, but the rounds after the first
 * hold a few percent of the jobs, so the scan is cheap and the compaction is
 * not yet worth its own code. That is a measurement to revisit, not a
 * principle.
 */
/* The launch always runs over a COMPACTED index list rather than every slot.
 * It matters more than a fall-through normally would: a lane that is already
 * resolved costs nothing itself, but its warp still runs for as long as its
 * slowest LIVE lane, so a warp holding one live lane and 31 dead ones pays a
 * full rho budget for one job. Packing the live lanes together is what turns
 * that back into 32 jobs per budget -- measured at 1.32x on the stage.
 *
 * `njp` is read on the device so the grid never has to be sized from a
 * host-visible count, which keeps the round loop free of synchronisation. */
template <int L, int METHOD, int STAGE2>
__global__ void k_cofac(const mz<L> *__restrict n, mz<L> lim2, uint32_t lpb,
                        uint32_t c0, uint32_t budget,
                        const uint32_t *__restrict sel,
                        const uint32_t *__restrict njp,
                        uint8_t *__restrict status,
                        uint64_t *__restrict fac, uint8_t *__restrict nfac,
                        unsigned long long *__restrict iters,
                        const uint32_t *__restrict s, uint32_t ns,
                        const uint8_t *__restrict s2mask, uint32_t s2vmin,
                        uint32_t s2nv, uint32_t ibeg, uint32_t iend)
{
    /* [ibeg, iend) is this launch's slice of the compacted list; cf_sched_t's
     * `chunk` comment says why the caller may split a round this way. The
     * count is still read on the device, so the host never has to know it and
     * the round loop stays free of synchronisation: a slice that starts past
     * the live count simply runs an empty loop, which is what the later rounds
     * (a few percent of the jobs) mostly do. */
    const uint32_t cnt = *njp;
    const uint32_t hi = (iend < cnt) ? iend : cnt;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t ii = ibeg + bench_grid_thread_x(); ii < hi; ii += stride) {
        const uint32_t i = (uint32_t)ii;
        const uint32_t t = sel[i];
        uint64_t o[CF_MAXFAC];
        uint64_t acc = 0;
        int k = 0, st;
        mz<L> v;
        if (status[t] != CF_INCOMPLETE) continue;
        v = n[t];
        st = mz_split<L, METHOD, STAGE2>(&v, &lim2, lpb, c0, budget, o, &k,
                                         &acc, s, ns, s2mask, s2vmin, s2nv);
        if (iters) iters[t] += (unsigned long long)acc;   /* summed over rounds */
        status[t] = (uint8_t)st;
        if (st == CF_OK) {
            nfac[t] = (uint8_t)k;
            for (int i = 0; i < k; i++) fac[(size_t)t * CF_MAXFAC + i] = o[i];
        }
    }
}




/* Flag the jobs a round still has to run, and pack their indices. Rebuilt
 * before every round, so round r+1 sees only what round r left unresolved --
 * which on the algebraic side is where the requeue cost actually lives. */
__global__ void k_cof_selflags(uint32_t n, const uint8_t *__restrict st,
                               uint32_t *__restrict flag)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        flag[t] = (st[t] == CF_INCOMPLETE) ? 1u : 0u;
    }
}

__global__ void k_cof_selscatter(uint32_t n, const uint32_t *__restrict flag,
                                 const uint32_t *__restrict off,
                                 uint32_t *__restrict sel, uint32_t *__restrict nsel)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (flag[t]) sel[off[t]] = t;
        if (t == n - 1) *nsel = off[t] + flag[t];
    }
}

/* ---- the one batch executor -------------------------------------------- *
 *
 * There used to be two of these: `cf_run_side` for the standalone `--cofac`
 * batch and the loop inside `cofq_flush` for the inline cross-q queue. They
 * diverged in exactly the way duplicated schedulers do -- the standalone one
 * grew ECM support and never got per-round compaction, the inline one grew
 * compaction and never received the ECM configuration, so `--cof-ecm` silently
 * ran rho on the production path for as long as both existed. Neither defect
 * was visible from inside its own path.
 *
 * So the schedule lives in one place now, parameterised by the two things that
 * actually differ between callers: which method, and how much of it.
 */
typedef struct {
    int      method;        /* 0 = Pollard-Brent rho, 1 = ECM                */
    int      rounds;
    uint32_t budget;        /* rho iterations in round 0; doubles each round */
    uint32_t curves;        /* ECM curves attempted per round                */
    const uint32_t *d_s;    /* ECM stage-1 prime powers, on device           */
    uint32_t ns;
    const uint8_t *d_s2mask;/* D=30 stage-2 prime-pair mask per giant step   */
    uint32_t s2vmin, s2nv;
    /* Records per launch. ALWAYS A CONCRETE COUNT, never a sentinel: a value
     * at or above the batch means one launch (the original behaviour,
     * bit-for-bit), and callers with nothing to say pass UINT32_MAX rather
     * than 0. cfg.cof_chunk's 0 means AUTO and is resolved by cofq_flush
     * before it ever reaches this struct, so the two zeros are not the same
     * zero and this one has no special meaning at all.
     *
     * WHY THIS EXISTS: a round's launch duration is set by its parameters, not
     * by any clock -- rho's round r runs `budget << r` iterations and ECM runs
     * `curves` curves, both over every selected record. A slow device takes
     * proportionally longer for the identical launch and eventually crosses the
     * host's GPU watchdog, which kills it: cudaErrorLaunchTimeout ("the launch
     * timed out and was terminated") on Windows/NVIDIA, the vaguer
     * "unspecified launch failure" on AMD. Both were seen in the field on
     * slower volunteer hardware, reported at cofq_flush's sync because that is
     * where an async launch failure first surfaces, not where it happened.
     *
     * WHY THE RECORD AXIS AND NOT THE WORK AXIS: every record's mz_split call
     * is independent and is left completely untouched by this, so splitting the
     * compacted list cannot change a single result -- the golden test's exact
     * relation counts hold by construction. Chunking the OTHER axes cannot say
     * that: sigma is `c0*1000 + cv + 6` and mz_split restarts its factor stack
     * from the original cofactor on every call, so a curve sub-range makes a
     * later chunk restart the top composite with sigmas that cannot split it,
     * and capping rho's budget moves the rho constant. Both change which
     * sequences run.
     *
     * WHY IT IS FREE WHERE IT IS NEEDED: the grid is multiProcessorCount*6
     * blocks. On a small device (6 CUs -> 9,216 threads) a 131,072-record batch
     * is ~14 records per thread, so quartering it still leaves every thread
     * loaded. On a large one (128 SMs -> 196,608 threads) the grid already
     * exceeds the batch, and chunking would only idle threads -- which is why
     * the default is one launch and a caller opts in. */
    uint32_t chunk;
} cf_sched_t;

/* Scratch for the per-round compaction. The caller owns it because the inline
 * queue already has these buffers allocated once per band and should not
 * reallocate them per flush. */
typedef struct {
    uint32_t *d_flag, *d_off, *d_bsum, *d_sel, *d_nsel;
} cf_work_t;

/* Run one side of one batch to exhaustion of its round budget. Records that
 * remain CF_INCOMPLETE are requeued into the next round with a different rho
 * constant (or a fresh band of ECM sigmas) and, for rho, twice the budget --
 * so an exhausted budget is never turned into a proof of anything. */
template <int L>
static void cf_run_rounds(const mz<L> *d_n, mz<L> lim2, uint32_t lpb, uint32_t n,
                          uint8_t *d_status, uint64_t *d_fac, uint8_t *d_nfac,
                          const cf_sched_t *S, const cf_work_t *W,
                          unsigned long long *d_iters, int blocks, int threads)
{
    const uint32_t nb = (n + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    if (!n) return;
    for (int r = 0; r < S->rounds; r++) {
        /* Ordered prefix scan, not an atomic append: the job order is then a
         * function of the status array alone and the output stays reproducible
         * byte for byte. Three small kernels over the batch -- microseconds
         * against a stage that runs for tens of milliseconds. */
        k_cof_selflags<<<blocks, threads>>>(n, d_status, W->d_flag);
        k_scan_pass1<<<nb, TD_SCAN_BLK>>>(W->d_flag, n, W->d_off, W->d_bsum);
        k_scan_pass2<<<1, 1024>>>(W->d_bsum, nb);
        k_scan_pass3<<<nb, TD_SCAN_BLK>>>(W->d_off, n, W->d_bsum);
        k_cof_selscatter<<<blocks, threads>>>(n, W->d_flag, W->d_off,
                                              W->d_sel, W->d_nsel);
        /* One launch per record slice. chunk 0 (or any value covering the
         * batch) gives exactly one launch over [0, n) -- the original single
         * launch, same grid, same arguments -- so the default path is
         * unchanged. Slices are cut against n rather than the live count
         * because the count lives on the device; see k_cofac. */
        uint32_t step = (S->chunk < n) ? S->chunk : n;
        if (!step) step = n;   /* b += 0 would never terminate */
        for (uint32_t b = 0; b < n; b += step) {
            const uint32_t e = (n - b > step) ? b + step : n;
            if (S->method) {
                if (S->s2nv)
                    k_cofac<L, 1, 1><<<blocks, threads>>>(
                        d_n, lim2, lpb, (uint32_t)(r + 1), S->curves,
                        W->d_sel, W->d_nsel, d_status, d_fac, d_nfac, d_iters,
                        S->d_s, S->ns, S->d_s2mask, S->s2vmin, S->s2nv, b, e);
                else
                    k_cofac<L, 1, 0><<<blocks, threads>>>(
                        d_n, lim2, lpb, (uint32_t)(r + 1), S->curves,
                        W->d_sel, W->d_nsel, d_status, d_fac, d_nfac, d_iters,
                        S->d_s, S->ns, NULL, 0, 0, b, e);
            } else {
                k_cofac<L, 0, 0><<<blocks, threads>>>(
                    d_n, lim2, lpb, (uint32_t)(r + 1), S->budget << r,
                    W->d_sel, W->d_nsel, d_status, d_fac, d_nfac, d_iters,
                    NULL, 0, NULL, 0, 0, b, e);
            }
        }
    }
}

/* lim^2 at a given width, or -1 if it does not fit.
 *
 * ONE copy, called by both the inline queue and the standalone --cofac path.
 * It was written out twice, and the comment below argued that keeping it next
 * to the instantiation is what stops it drifting from the splitter -- an
 * argument that only holds while there is one such place. lim2 is the
 * "prime by size" threshold and a wrong one accepts composites as prime, so
 * the two paths disagreeing about it is exactly the failure to design out. */
template <int L> static int cf_lim2(mz<L> *out, uint64_t lim)
{
    bench_u128_t v = bench_mul_u64_wide(lim, lim);

    for (int i = 0; i < L; i++)
        out->v[i] = bench_u128_take32(&v);

    if (bench_u128_nonzero(v)) {
        fprintf(stderr, "  cofac: lim^2 does not fit %d limbs\n", L);
        return -1;
    }
    return 0;
}

/* Runtime-width entry to the above.
 *
 * The queue stores its cofactors as RAW LIMBS, because the stride is a
 * per-side run-time choice and `mz<3>` and `mz<4>` are different types with
 * different strides; this is the one place that turns raw limbs back into a
 * typed `mz<L>*`, so the reinterpret_cast lives here and nowhere else. lim^2
 * is computed at the same width for the same reason -- a 3-limb lim2 handed to
 * a 4-limb splitter reads one limb of garbage as the top of the "prime by
 * size" threshold, and a wrong threshold accepts composites as primes.
 *
 * Returns -1 for a width this build does not contain, which is a configuration
 * error the caller must report rather than silently narrow: narrowing is
 * exactly the truncation the width machinery exists to prevent. */
static int cf_run_rounds_dyn(int L, const uint32_t *d_n, uint64_t lim,
                             uint32_t lpb, uint32_t n, uint8_t *d_status,
                             uint64_t *d_fac, uint8_t *d_nfac,
                             const cf_sched_t *S, const cf_work_t *W,
                             unsigned long long *d_iters,
                             int blocks, int threads)
{
#define CF_RR(LL) do {                                                        \
        mz<LL> l2;                                                            \
        if (cf_lim2<LL>(&l2, lim)) return -1;                                 \
        cf_run_rounds<LL>((const mz<LL> *)d_n, l2, lpb, n, d_status, d_fac,   \
                          d_nfac, S, W, d_iters, blocks, threads);            \
        return 0;                                                             \
    } while (0)
    switch (L) {
    case 3: CF_RR(3);
#if CF_LMAX >= 4
    case 4: CF_RR(4);
#endif
    default: break;
    }
#undef CF_RR
    fprintf(stderr, "  cofac: this build has no %d-limb splitter"
            " (CF_LMAX = %d)\n", L, CF_LMAX);
    return -1;
}

/* ---- the cross-q device queue ------------------------------------------ *
 *
 * A special-q produces about 1,956 joint candidates. That is nothing for this
 * GPU -- 73,728 threads at the pipeline's launch geometry -- so cofactorising
 * per special-q would run the rho kernel at 3% occupancy and pay the requeue
 * rounds on an almost empty grid. The queue exists to fix that: candidates
 * accumulate across special-q and are flushed as one batch of ~131,072, which
 * is ~67 q worth and fills the machine.
 *
 * It is also what removes the file round trip. `--candidates` writes 295 MB
 * over a 1,000-q band and the host emission of it costs ~3 ms/q; with the queue
 * resident, only the ~2% of records that become relations are ever read back.
 */

#define CQ_FLUSH  131072u        /* candidates per flush, ~67 special-q */

/* ---- auto chunk sizing -------------------------------------------------- *
 *
 * Per-launch budget. Windows' display watchdog (TDR) kills a kernel at ~2 s by
 * default and AMD's behaves comparably, so this leaves ~8x margin. The budget
 * is compared against a whole SIDE's device time, which is the sum over its
 * rounds and slices and therefore an upper bound on any single launch in it --
 * deliberately conservative, and free to be, because over-chunking costs
 * nothing on the devices that ever reach this path (see the floor below).
 */
#define COF_CHUNK_TARGET_MS   250.0f

/* Giant steps per curve above which cofq_init warns that record chunking can
 * no longer bound a launch. ~40x the derived default's ~500 and ~16x below the
 * ~320,000 of the one configuration known to trip a watchdog; see the warning
 * site in cofq_init for why this is a judgement rather than a measurement. */
#define COF_S2NV_WATCHDOG_WARN  20000u

/* The floor is `blocks * threads`: one record per thread, and never less.
 *
 * MEASURED on an RTX 3090 (82 SMs -> 492 blocks x 256 = 125,952 threads),
 * oracle/c183, 300 q, --ecm-b1 5000 --ecm-curves 32, n=3, algebraic queue:
 *
 *      chunk   131072   65536   32768   16384
 *      ms       27.97   33.86   41.93   69.73
 *      vs off   +0.5%  +21.7%  +50.7% +150.7%
 *
 * A slice at or above the thread count costs nothing (131072 and 262144 both
 * land inside the +-1.5% spread of the unchunked arm itself). Below it the
 * cost is occupancy, not launch overhead: 1 -> 8 launches is ~35 us of launch
 * latency against a measured +42 ms, three orders of magnitude apart, so what
 * is being paid for is threads with no record to work on.
 *
 * That is also why this floor makes chunking free exactly where it is needed.
 * A 780M has 6 CUs -> 36 blocks x 256 = 9,216 threads, so slices of 9,216 are
 * still one record per thread -- fully loaded -- while cutting a 131,072-record
 * launch by 14x. The small devices that trip the watchdog subdivide for free;
 * the large ones (a 4090's 196,608 threads exceed CQ_FLUSH outright) get one
 * slice and the original code path. */
static uint32_t cof_chunk_floor(int blocks, int threads)
{
    const uint64_t f = (uint64_t)(blocks > 0 ? blocks : 1)
                     * (uint64_t)(threads > 0 ? threads : 1);
    return f > 0xffffffffull ? 0xffffffffu : (uint32_t)f;
}

/* Report the slice in effect, on first choice and on every later change.
 *
 * STDERR, unlike the j-slabbing line's stdout, and for the reason bench_main
 * already gives for the grid line: a BOINC client discards stdout with the
 * slot directory, so anything a volunteer's failure report needs to carry has
 * to be on stderr, which is uploaded. This exists to answer "what did it pick
 * on the host that died?", which is unanswerable from source alone.
 *
 * Only on change, never per flush: a long band flushes hundreds of times and
 * the value converges within a few of them. Unbounded per-event logging has
 * already produced hundreds of duplicate lines once in this project. */
static void cof_report_chunk(uint32_t chunk, uint32_t n, int pinned)
{
    const uint32_t step = (chunk && chunk < n) ? chunk : n;
    const uint32_t nl = step ? (n + step - 1) / step : 1u;
    fprintf(stderr, "  cofactor chunk: %u records/launch, %u launch%s per"
            " round over %u records (%s)\n", step, nl, nl == 1 ? "" : "es", n,
            pinned ? "pinned by --cof-chunk" : "auto");
}

typedef struct {
    uint32_t cap, n, rcap;
    /* The two cofactors, narrowed from the 256-bit norm residual to L0/L1
     * limbs. RAW LIMBS, not mz<L>*: the width is a per-side run-time choice
     * (see cf_limbs_for_mfb) and mz<3> and mz<4> are distinct types with
     * distinct strides, so the array cannot carry one of them in its type.
     * cf_run_rounds_dyn is the only place that casts, and k_cof_enqueue<L0,L1>
     * is the only writer.
     *
     * History worth keeping, because both defects were silent truncations of
     * exactly this array. Side 0 used to be mz<2>, which assumed the rational
     * cofactor always fit in 64 bits -- true while the special-q sat on the
     * algebraic side and the rational mfb was ~60, false for an SNFS job whose
     * hard side is the rational one (mfbr 88), and it truncated lim0^2 for any
     * rlim above 2^32. Then both sides were pinned at mz<3>, which is 96 bits
     * and covers a C194's mfba 95 but not a C208's algebraic side.
     *
     * The uniform-width argument that justified pinning both at 3 still holds
     * for the CHEAP direction -- the rational queue was 0.04 ms of a 24.49 ms
     * special-q on the c151, so narrowing side 0 buys nothing -- but it never
     * justified refusing to WIDEN side 1. Hence per-side widths, defaulting to
     * the narrowest that mfb needs. */
    uint32_t *d_c0;   uint32_t *d_c1;
    int      L0, L1;                      /* limbs per side, 3 or 4        */
    uint8_t  *d_st0,  *d_st1;             /* CF_* per side                */
    /* 64-BIT, NOT 32. These two and d_sp0/d_sp1 below hold *resulting primes*,
     * which are bounded by lpb rather than by lim -- so at lpb 33 (a C194 asks
     * for lpba 33) a uint32 stores the low 32 bits of a 33-bit prime and the
     * relation stops reconstructing its own norm. The trial-division lists
     * d_f0/d_f1 stay 32-bit on purpose: those are FACTOR-BASE primes, bounded
     * by lim, and lim is well under 2^32 even for a C208. */
    uint64_t *d_sm0,  *d_sm1;             /* residual when no split needed */
    int64_t  *d_a,    *d_b;
    uint32_t *d_f0,   *d_f1;              /* cap * TD_FMAX, FB primes < lim */
    uint8_t  *d_fn0,  *d_fn1;
    uint64_t *d_sp0,  *d_sp1;             /* cap * CF_MAXFAC split primes */
    uint8_t  *d_nsp0, *d_nsp1;
    uint32_t *d_flag, *d_off, *d_bsum, *d_idx, *d_nrel;
    uint32_t *d_sel, *d_nsel;             /* compacted job list per round  */
    /* Records whose residual did not fit L0/L1 limbs. Unreachable while
     * resolve_and_check_cofactor_config holds -- cof_classify has already rejected
     * anything above mfb, and mfb is refused above 32*L -- so this is an
     * assertion on that chain, in the same spirit as the rcap check in
     * cofq_flush. It is one predicated OR per record and it converts the one
     * failure mode this whole width machinery exists to prevent, a silently
     * truncated cofactor, from a wrong relation into a loud stop. */
    uint32_t *d_ovf;
    /* Records per cofactor launch that auto mode settled on, carried across
     * flushes so the search is done once per band and not once per flush.
     * 0 means "not chosen yet"; cofq_init memsets the struct, so the first
     * flush picks the opening value. Ignored entirely when the operator gave
     * an explicit --cof-chunk. */
    uint32_t chunk_cur;
    uint32_t *d_s, ns, ecm_curves;
    /* Method PER SIDE, 0 = Pollard-Brent rho, 1 = ECM. Per side and not per
     * job because the two sides of a real job are usually different shapes:
     * AS276 is mfbr 64 / lpbr 33 on the rational side (2LP) and
     * mfba 101 / lpba 35 on the algebraic (3LP), and the measured winner
     * differs between exactly those two cases -- rho by ~1.1x at 2LP, ECM by
     * 2-4x at 3LP (RESULTS finding 70). A per-job choice would get one of the
     * two sides wrong on nearly every job. */
    int meth[2];
    uint8_t  *d_s2mask; uint32_t s2vmin, s2nv;/* ECM stage-2 schedule      */
    double ms_rat, ms_alg, ms_host;
    unsigned long long nseen, nrel;
    /* Per side, per CF_* status, accumulated over flushes. `ndead`/`nstuck`
     * used to be two scalars here that NOTHING ever wrote -- which is the
     * whole reason a 13.7% yield loss on AS276 (finding 69) and a 0.33% one on
     * the c183 (finding 70) were invisible for as long as they were: an
     * exhausted rho budget leaves CF_INCOMPLETE, the record is silently not a
     * relation, and no counter moved. A stuck count is the ONLY signal that
     * distinguishes "this job has no more relations" from "the splitter gave
     * up early", and those look identical in every other number the run
     * prints. */
    unsigned long long nst[2][4];
    unsigned long long *d_nst;            /* 8 counters, device side       */
} cofq_t;

/* Append one special-q's joint candidates. Everything the relation will need
 * is copied in, because the per-q arrays are overwritten by the next q. */
/* Templated on BOTH sides' widths because one launch writes both arrays and
 * they are independent choices. Four instantiations at CF_LMAX=4, none of them
 * expensive -- this kernel is a copy, not arithmetic. */
template <int L0, int L1>
__global__ void k_cof_enqueue(const bn_t *__restrict cof0, const uint8_t *__restrict bits0,
                              const bn_t *__restrict cof1, const uint8_t *__restrict bits1,
                              const int64_t *__restrict a, const int64_t *__restrict b,
                              const uint32_t *__restrict f0, const uint32_t *__restrict fn0,
                              const uint32_t *__restrict f1, const uint32_t *__restrict fn1,
                              uint32_t n, uint32_t base, uint32_t lpb0, uint32_t lpb1,
                              cofq_t Q)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        const uint32_t d = base + t;
        Q.d_a[d] = a[t]; Q.d_b[d] = b[t];
        Q.d_nsp0[d] = 0; Q.d_nsp1[d] = 0;
        {
            uint32_t c = fn0[t]; if (c > TD_FMAX) c = TD_FMAX;
            Q.d_fn0[d] = (uint8_t)c;
            for (uint32_t z = 0; z < c; z++)
                Q.d_f0[(size_t)d * TD_FMAX + z] = f0[(size_t)t * TD_FMAX + z];
            c = fn1[t]; if (c > TD_FMAX) c = TD_FMAX;
            Q.d_fn1[d] = (uint8_t)c;
            for (uint32_t z = 0; z < c; z++)
                Q.d_f1[(size_t)d * TD_FMAX + z] = f1[(size_t)t * TD_FMAX + z];
        }
        /* A side within lpb needs no splitting, but its residual is still a
         * prime of the relation whenever it is not 1. */
        {
            /* bn_fits_u64, not a hand-rolled two-limb pack: this branch is
             * taken when the residual is already within lpb, and at lpb > 32
             * that residual is a prime needing 64 bits. The helper also
             * REJECTS a value with limbs above the low two set, so a future
             * change to lpb or the bits accounting cannot silently reinstate
             * the truncation this whole path exists to remove. A 0 here is a
             * residual that did not fit, which cannot happen while
             * bits <= lpb <= 64 and is emitted as "no residual" if it ever
             * does -- visible as a failed reconstruction, not a wrong one. */
            uint64_t sm = 0;
            bn_t v = cof0[t];
            if ((uint32_t)bits0[t] > lpb0) {
                uint32_t hi = 0;
                #pragma unroll
                for (int i = 0; i < L0; i++) Q.d_c0[(size_t)d * L0 + i] = v.v[i];
                #pragma unroll
                for (int i = L0; i < BN_LIMBS; i++) hi |= v.v[i];
                Q.d_sm0[d] = 0;
                if (hi) { Q.d_st0[d] = CF_OVERFLOW; atomicAdd(Q.d_ovf, 1u); }
                else      Q.d_st0[d] = CF_INCOMPLETE;
            } else {
                Q.d_sm0[d] = (bits0[t] > 1 && bn_fits_u64(&v, &sm)) ? sm : 0ull;
                Q.d_st0[d] = CF_OK;
            }
            v = cof1[t];
            if ((uint32_t)bits1[t] > lpb1) {
                uint32_t hi = 0;
                #pragma unroll
                for (int i = 0; i < L1; i++) Q.d_c1[(size_t)d * L1 + i] = v.v[i];
                #pragma unroll
                for (int i = L1; i < BN_LIMBS; i++) hi |= v.v[i];
                Q.d_sm1[d] = 0;
                if (hi) { Q.d_st1[d] = CF_OVERFLOW; atomicAdd(Q.d_ovf, 1u); }
                else      Q.d_st1[d] = CF_INCOMPLETE;
            } else {
                Q.d_sm1[d] = (bits1[t] > 1 && bn_fits_u64(&v, &sm)) ? sm : 0ull;
                Q.d_st1[d] = CF_OK;
            }
        }
    }
}

/* One switch over the (L0, L1) pairs so the pipeline's call site keeps a plain
 * function call and does not have to know the widths are template arguments.
 * The pairs this build does not contain are compiled out, which is what makes
 * `make CF_LMAX=3` a genuinely narrower binary rather than a wider one
 * with a disabled branch. */
static int cof_enqueue(int blocks, int threads,
                       const bn_t *cof0, const uint8_t *bits0,
                       const bn_t *cof1, const uint8_t *bits1,
                       const int64_t *a, const int64_t *b,
                       const uint32_t *f0, const uint32_t *fn0,
                       const uint32_t *f1, const uint32_t *fn1,
                       uint32_t n, uint32_t base, uint32_t lpb0, uint32_t lpb1,
                       const cofq_t *Q)
{
#define CF_ENQ(A, B) do {                                                     \
        k_cof_enqueue<A, B><<<blocks, threads>>>(cof0, bits0, cof1, bits1,    \
                                                 a, b, f0, fn0, f1, fn1,      \
                                                 n, base, lpb0, lpb1, *Q);    \
        return 0;                                                             \
    } while (0)
    switch (Q->L0 * 8 + Q->L1) {
    case 3 * 8 + 3: CF_ENQ(3, 3);
#if CF_LMAX >= 4
    case 3 * 8 + 4: CF_ENQ(3, 4);
    case 4 * 8 + 3: CF_ENQ(4, 3);
    case 4 * 8 + 4: CF_ENQ(4, 4);
#endif
    default: break;
    }
#undef CF_ENQ
    fprintf(stderr, "  cofac queue: this build has no %d/%d-limb enqueue"
            " (CF_LMAX = %d)\n", Q->L0, Q->L1, CF_LMAX);
    return -1;
}

/* The class-aware gate, on device. A record whose rational side did not split
 * has its algebraic status forced away from CF_INCOMPLETE, so k_cofac falls
 * straight through it instead of spending a full rho budget on a record that
 * can never be a relation. */
__global__ void k_cof_gate(uint32_t n, uint8_t *__restrict st0, uint8_t *__restrict st1)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (st0[t] != CF_OK && st1[t] == CF_INCOMPLETE) st1[t] = CF_DEAD;
    }
}
/* Per-flush status histogram, both sides at once.
 *
 * Shared-memory bins first, one global atomic per bin per block: 131,072
 * records against 8 addresses would otherwise serialise on those 8 lines, and
 * this is pure instrumentation -- it must not show up in the stage it measures.
 *
 * Side 1 is counted AFTER k_cof_gate, so its CF_DEAD includes records the gate
 * killed because side 0 failed, not only primes above lpb. CF_INCOMPLETE is
 * unaffected: the gate only ever writes DEAD, so a stuck count is exactly the
 * budget signal it looks like. */
__global__ void k_cof_status_hist(uint32_t n, const uint8_t *__restrict st0,
                                  const uint8_t *__restrict st1,
                                  unsigned long long *__restrict out)
{
    /* Relies on blockDim.x >= 8 for both the zeroing and the flush below.
     * Guaranteed: --threads is refused outside [32, 1024] and must be a
     * multiple of 32. Stated because nothing else in this file needs a block
     * floor, so a future launch geometry could break it silently -- bins 4..7
     * are side 1, and they would read uninitialised shared memory and never
     * be flushed, i.e. the instrumentation would lie rather than fail. */
    __shared__ unsigned int bin[8];
    if (threadIdx.x < 8) bin[threadIdx.x] = 0u;
    __syncthreads();
    {
        const uint64_t stride = bench_grid_stride_x();
        for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
            const uint32_t t = (uint32_t)tt;
            uint32_t a = st0[t], b = st1[t];
            atomicAdd(&bin[(a < 4u ? a : 3u)], 1u);
            atomicAdd(&bin[4u + (b < 4u ? b : 3u)], 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x < 8 && bin[threadIdx.x])
        atomicAdd(&out[threadIdx.x], (unsigned long long)bin[threadIdx.x]);
}

__global__ void k_rel_flags(uint32_t n, const uint8_t *__restrict st0,
                            const uint8_t *__restrict st1, uint32_t *__restrict flag)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        flag[t] = (st0[t] == CF_OK && st1[t] == CF_OK) ? 1u : 0u;
    }
}

/* Gather the relations into the front of the queue's own arrays. Only these
 * rows are read back -- about 2% of the batch. */
__global__ void k_rel_gather(uint32_t n, const uint32_t *__restrict flag,
                             const uint32_t *__restrict off, uint32_t cap,
                             uint32_t *__restrict idx, uint32_t *__restrict nrel)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (flag[t] && off[t] < cap) idx[off[t]] = t;
        if (t == n - 1) *nrel = off[t] + flag[t];
    }
}

__global__ void k_rel_pack(uint32_t nr, const uint32_t *__restrict idx, cofq_t Q,
                           int64_t *__restrict oa, int64_t *__restrict ob,
                           uint32_t *__restrict of0, uint8_t *__restrict ofn0,
                           uint32_t *__restrict of1, uint8_t *__restrict ofn1,
                           uint64_t *__restrict osp0, uint8_t *__restrict onsp0,
                           uint64_t *__restrict osp1, uint8_t *__restrict onsp1)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < nr; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        const uint32_t r = idx[t];
        oa[t] = Q.d_a[r]; ob[t] = Q.d_b[r];
        ofn0[t] = Q.d_fn0[r]; ofn1[t] = Q.d_fn1[r];
        for (uint32_t z = 0; z < Q.d_fn0[r]; z++)
            of0[(size_t)t * TD_FMAX + z] = Q.d_f0[(size_t)r * TD_FMAX + z];
        for (uint32_t z = 0; z < Q.d_fn1[r]; z++)
            of1[(size_t)t * TD_FMAX + z] = Q.d_f1[(size_t)r * TD_FMAX + z];
        /* the split primes, plus the residual of a side that needed no split */
        {
            uint32_t k = Q.d_nsp0[r];
            for (uint32_t z = 0; z < k; z++)
                osp0[(size_t)t * CF_MAXFAC + z] = Q.d_sp0[(size_t)r * CF_MAXFAC + z];
            if (!k && Q.d_sm0[r]) { osp0[(size_t)t * CF_MAXFAC] = Q.d_sm0[r]; k = 1; }
            onsp0[t] = (uint8_t)k;
            k = Q.d_nsp1[r];
            for (uint32_t z = 0; z < k; z++)
                osp1[(size_t)t * CF_MAXFAC + z] = Q.d_sp1[(size_t)r * CF_MAXFAC + z];
            if (!k && Q.d_sm1[r]) { osp1[(size_t)t * CF_MAXFAC] = Q.d_sm1[r]; k = 1; }
            onsp1[t] = (uint8_t)k;
        }
    }
}

/* ---- the queue, host side ---------------------------------------------- */

static int cf_u64cmp(const void *x, const void *y)
{
    uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return a < b ? -1 : (a > b);
}


typedef struct {
    int64_t  *a, *b;
    uint32_t *f0, *f1;
    uint64_t *sp0, *sp1;              /* split primes: bounded by lpb, not lim */
    uint8_t  *fn0, *fn1, *nsp0, *nsp1;
    int64_t  *d_a, *d_b;
    uint32_t *d_f0, *d_f1;
    uint64_t *d_sp0, *d_sp1;
    uint8_t  *d_fn0, *d_fn1, *d_nsp0, *d_nsp1;
} cofq_out_t;

static void cofq_free(cofq_t *Q, cofq_out_t *O)
{
    cudaFree(Q->d_c0); cudaFree(Q->d_c1); cudaFree(Q->d_st0); cudaFree(Q->d_st1);
    cudaFree(Q->d_sm0); cudaFree(Q->d_sm1); cudaFree(Q->d_a); cudaFree(Q->d_b);
    cudaFree(Q->d_f0); cudaFree(Q->d_f1); cudaFree(Q->d_fn0); cudaFree(Q->d_fn1);
    cudaFree(Q->d_sp0); cudaFree(Q->d_sp1); cudaFree(Q->d_nsp0); cudaFree(Q->d_nsp1);
    cudaFree(Q->d_flag); cudaFree(Q->d_off); cudaFree(Q->d_bsum);
    cudaFree(Q->d_idx); cudaFree(Q->d_nrel);
    cudaFree(Q->d_sel); cudaFree(Q->d_nsel); cudaFree(Q->d_ovf); cudaFree(Q->d_nst);
    cudaFree(Q->d_s);
    cudaFree(Q->d_s2mask);
    cudaFree(O->d_a); cudaFree(O->d_b); cudaFree(O->d_f0); cudaFree(O->d_f1);
    cudaFree(O->d_fn0); cudaFree(O->d_fn1); cudaFree(O->d_sp0); cudaFree(O->d_sp1);
    cudaFree(O->d_nsp0); cudaFree(O->d_nsp1);
    if (O->a) cudaFreeHost(O->a);
    if (O->b) cudaFreeHost(O->b);
    if (O->f0) cudaFreeHost(O->f0);
    if (O->f1) cudaFreeHost(O->f1);
    if (O->fn0) cudaFreeHost(O->fn0);
    if (O->fn1) cudaFreeHost(O->fn1);
    if (O->sp0) cudaFreeHost(O->sp0);
    if (O->sp1) cudaFreeHost(O->sp1);
    if (O->nsp0) cudaFreeHost(O->nsp0);
    if (O->nsp1) cudaFreeHost(O->nsp1);
    memset(Q, 0, sizeof(*Q)); memset(O, 0, sizeof(*O));
}

/* `ecm`/`ecm_b1`/`ecm_curves` come from the caller's configuration. They used
 * to be left at the memset's zero, which silently pinned the inline path to rho
 * no matter what the CLI asked for -- the two schedulers had diverged so that
 * only the standalone one could run ECM at all. */
static int cofq_init(cofq_t *Q, cofq_out_t *O, uint32_t cap,
                     int meth0, int meth1, uint32_t ecm_b1, uint32_t ecm_b2,
                     uint32_t ecm_curves, int limbs0, int limbs1)
{
    const uint32_t nb = (cap + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    uint32_t *h_s = NULL;
    uint8_t *h_s2mask = NULL;
    int rc = -1;

#define COF_INIT_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    memset(Q, 0, sizeof(*Q)); memset(O, 0, sizeof(*O));
    Q->cap = cap;
    Q->meth[0] = meth0; Q->meth[1] = meth1; Q->ecm_curves = ecm_curves;
    Q->L0 = limbs0; Q->L1 = limbs1;
    /* Refused here rather than clamped: every caller derives these from mfb,
     * so a width outside the build is a configuration the run cannot honour,
     * and honouring it approximately means truncating a cofactor. */
    if (Q->L0 < CF_LMIN || Q->L0 > CF_LMAX || Q->L1 < CF_LMIN || Q->L1 > CF_LMAX) {
        fprintf(stderr, "  cofac queue: cofactor width %d/%d limbs is outside"
                " this build's %d..%d\n", Q->L0, Q->L1, CF_LMIN, CF_LMAX);
        goto done;
    }
    /* The build range is on the line too, not just the choice. A run log that
     * says "3 limbs" is ambiguous between "this job needs 3" and "this binary
     * only has 3", and those are different facts when the question is whether
     * a wider job could have run at all. It is also what lets cofcheck.sh
     * discover which cases this build can even attempt. */
    printf("  cofactor width: side 0 %d limbs (%d bits), side 1 %d limbs"
           " (%d bits); this build carries %d..%d\n",
           Q->L0, 32 * Q->L0, Q->L1, 32 * Q->L1, CF_LMIN, CF_LMAX);
    if (meth0 || meth1) {          /* one plan serves whichever sides use ECM */
        Q->ns = cf_ecm_plan(ecm_b1, &h_s);
        if (!Q->ns) {
            fprintf(stderr, "  cofac queue: empty ECM plan for B1=%u\n", ecm_b1);
            goto done;
        }
        COF_INIT_CK(cudaMalloc(&Q->d_s, (size_t)Q->ns * 4));
        COF_INIT_CK(cudaMemcpy(Q->d_s, h_s, (size_t)Q->ns * 4,
                               cudaMemcpyHostToDevice));
        free(h_s); h_s = NULL;
        if (ecm_b2) {
            Q->s2nv = cf_ecm_stage2_plan(ecm_b1, ecm_b2, &Q->s2vmin,
                                         &h_s2mask);
            if (!Q->s2nv) {
                fprintf(stderr, "  cofac queue: empty ECM stage-2 plan for"
                        " B1=%u B2=%u\n", ecm_b1, ecm_b2);
                goto done;
            }
            COF_INIT_CK(cudaMalloc(&Q->d_s2mask, Q->s2nv));
            COF_INIT_CK(cudaMemcpy(Q->d_s2mask, h_s2mask, Q->s2nv,
                                   cudaMemcpyHostToDevice));
            free(h_s2mask); h_s2mask = NULL;
        }
        printf("  cofactor queue: ECM B1 = %u, %u prime powers, B2 = %u"
               " (%u giant steps), %u curves per round\n", ecm_b1, Q->ns,
               ecm_b2, Q->s2nv, ecm_curves);
        /* The one thing --cof-chunk cannot bound.
         *
         * Chunking splits the RECORD list, so the smallest launch it can make
         * is one record per thread. A launch therefore never gets shorter than
         * ONE record's own splitting work, and stage 2 is where that can grow
         * without limit: the giant-step count is B2/30, so a large B2 buys a
         * per-record cost no slice size can divide.
         *
         * The auto-derived configuration is nowhere near this -- cof_auto_b1
         * returns 200..500, so B2 is 6,000..15,000 and s2nv is ~194..500 giant
         * steps -- which is why the default path needs no warning at all. The
         * one configuration observed to actually trip a device watchdog is
         * --ecm-b1 400000 (B2 clamped to 10^7, ~320,000 giant steps), which
         * reproducibly kills the stage-2 kernel on gfx1103 and is why
         * cofcheck.sh skips that case on HIP.
         *
         * THE THRESHOLD IS A JUDGEMENT, NOT A MEASUREMENT. The real boundary
         * is per-device and nobody has measured it; this sits well above the
         * derived default and well below the one value known to fail, and it
         * warns rather than refuses because a fast card runs these settings
         * perfectly well. Do not restate it as a supported limit. */
        if (Q->s2nv > COF_S2NV_WATCHDOG_WARN)
            fprintf(stderr, "  cofactor queue: WARNING: %u giant steps/curve x"
                    " %u curves is far above the derived default (~194-500"
                    " steps at B1 200-500). --cof-chunk bounds a launch by"
                    " RECORDS and cannot go below one record, so this much"
                    " stage-2 work per record can still exceed a slow device's"
                    " GPU watchdog and be killed mid-launch"
                    " (cudaErrorLaunchTimeout, or \"unspecified launch"
                    " failure\" on AMD). Lower --ecm-b2 if tasks fail that"
                    " way.\n", Q->s2nv, ecm_curves);
    }
    {
        static const char *nm[2] = { "rho", "ECM" };
        printf("  cofactor method: side 0 %s, side 1 %s\n",
               nm[Q->meth[0] ? 1 : 0], nm[Q->meth[1] ? 1 : 0]);
    }
    /* Every enqueued record can become a relation, so the readback slots have
     * to allow for that. This was cap/4, sized from c183's ~2% relation rate --
     * which is a property of THAT job, not of the queue. The c123 turns 56% of
     * its enqueued records into relations and overflowed it at the second
     * flush; the run failed loudly and correctly, but it should not have been
     * possible to hit at all. cap costs ~74 MB device and the same pinned,
     * against a 12 GB card, and removes the failure mode instead of retuning
     * the threshold for one more job. */
    Q->rcap = cap;
    COF_INIT_CK(cudaMalloc(&Q->d_c0, (size_t)cap * Q->L0 * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_c1, (size_t)cap * Q->L1 * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_st0, cap));
    COF_INIT_CK(cudaMalloc(&Q->d_st1, cap));
    COF_INIT_CK(cudaMalloc(&Q->d_sm0, (size_t)cap * 8));
    COF_INIT_CK(cudaMalloc(&Q->d_sm1, (size_t)cap * 8));
    COF_INIT_CK(cudaMalloc(&Q->d_a, (size_t)cap * 8));
    COF_INIT_CK(cudaMalloc(&Q->d_b, (size_t)cap * 8));
    COF_INIT_CK(cudaMalloc(&Q->d_f0, (size_t)cap * TD_FMAX * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_f1, (size_t)cap * TD_FMAX * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_fn0, cap));
    COF_INIT_CK(cudaMalloc(&Q->d_fn1, cap));
    COF_INIT_CK(cudaMalloc(&Q->d_sp0, (size_t)cap * CF_MAXFAC * 8));
    COF_INIT_CK(cudaMalloc(&Q->d_sp1, (size_t)cap * CF_MAXFAC * 8));
    COF_INIT_CK(cudaMalloc(&Q->d_nsp0, cap));
    COF_INIT_CK(cudaMalloc(&Q->d_nsp1, cap));
    COF_INIT_CK(cudaMalloc(&Q->d_flag, (size_t)cap * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_off, (size_t)cap * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_bsum, (size_t)nb * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_idx, (size_t)Q->rcap * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_nrel, 4));
    COF_INIT_CK(cudaMalloc(&Q->d_sel, (size_t)cap * 4));
    COF_INIT_CK(cudaMalloc(&Q->d_nsel, 4));
    COF_INIT_CK(cudaMalloc(&Q->d_ovf, 4));
    COF_INIT_CK(cudaMemset(Q->d_ovf, 0, 4));
    COF_INIT_CK(cudaMalloc(&Q->d_nst, 8 * sizeof(unsigned long long)));
    COF_INIT_CK(cudaMemset(Q->d_nst, 0, 8 * sizeof(unsigned long long)));
    COF_INIT_CK(cudaMalloc(&O->d_a, (size_t)Q->rcap * 8));
    COF_INIT_CK(cudaMalloc(&O->d_b, (size_t)Q->rcap * 8));
    COF_INIT_CK(cudaMalloc(&O->d_f0, (size_t)Q->rcap * TD_FMAX * 4));
    COF_INIT_CK(cudaMalloc(&O->d_f1, (size_t)Q->rcap * TD_FMAX * 4));
    COF_INIT_CK(cudaMalloc(&O->d_fn0, Q->rcap));
    COF_INIT_CK(cudaMalloc(&O->d_fn1, Q->rcap));
    COF_INIT_CK(cudaMalloc(&O->d_sp0, (size_t)Q->rcap * CF_MAXFAC * 8));
    COF_INIT_CK(cudaMalloc(&O->d_sp1, (size_t)Q->rcap * CF_MAXFAC * 8));
    COF_INIT_CK(cudaMalloc(&O->d_nsp0, Q->rcap));
    COF_INIT_CK(cudaMalloc(&O->d_nsp1, Q->rcap));
    COF_INIT_CK(cudaHostAlloc((void **)&O->a, (size_t)Q->rcap * 8,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->b, (size_t)Q->rcap * 8,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->f0,
                              (size_t)Q->rcap * TD_FMAX * 4,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->f1,
                              (size_t)Q->rcap * TD_FMAX * 4,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->fn0, Q->rcap,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->fn1, Q->rcap,
                              cudaHostAllocDefault));
    /* x8, matching d_sp0/d_sp1 and the readback: split primes are 64-bit (see
     * cofq_t). Sizing these two at 4 while the copy asked for 8 is not a
     * silent overrun -- cudaMemcpy rejects it outright with "invalid argument"
     * on a pinned buffer that short -- but it is exactly the pair of numbers a
     * width change can leave disagreeing, so they are commented together. */
    COF_INIT_CK(cudaHostAlloc((void **)&O->sp0,
                              (size_t)Q->rcap * CF_MAXFAC * 8,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->sp1,
                              (size_t)Q->rcap * CF_MAXFAC * 8,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->nsp0, Q->rcap,
                              cudaHostAllocDefault));
    COF_INIT_CK(cudaHostAlloc((void **)&O->nsp1, Q->rcap,
                              cudaHostAllocDefault));
    rc = 0;

done:
    free(h_s);
    free(h_s2mask);
    if (rc) cofq_free(Q, O);
#undef COF_INIT_CK
    return rc;
}

/* The relation format for one side: ascending hex, comma separated.
 *
 * The list is 64-bit even though the trial-division half is not -- a relation's
 * primes are bounded by lpb, and mixing widths in one sorted array is how a
 * 33-bit split prime would get truncated on its way to the file.
 *
 * Shared by both emitters. They differ only in how the trial-division half
 * arrives (an array from the inline queue, a hex string from a --candidates
 * file), and keeping the sort and the format in one place is what stops the
 * two output paths drifting: the 64-bit widening had to touch the array type,
 * the comparator and the format string, and would have had to do it twice. */
static void cf_emit_sorted(FILE *o, uint64_t *v, int n)
{
    qsort(v, n, sizeof v[0], cf_u64cmp);
    for (int i = 0; i < n; i++)
        fprintf(o, "%s%llx", i ? "," : "", (unsigned long long)v[i]);
}

static void cq_emit_side(FILE *o, const uint32_t *f, int nf,
                         const uint64_t *sp, int nsp)
{
    uint64_t v[TD_FMAX + CF_MAXFAC];
    int n = 0;
    for (int i = 0; i < nf && n < TD_FMAX + CF_MAXFAC; i++) v[n++] = f[i];
    for (int i = 0; i < nsp && n < TD_FMAX + CF_MAXFAC; i++) v[n++] = sp[i];
    cf_emit_sorted(o, v, n);
}

/* Split everything queued, then write only the relations. */
static int cofq_flush(cofq_t *Q, cofq_out_t *O, uint64_t lim0, uint32_t lpb0,
                      uint64_t lim1, uint32_t lpb1, int rounds, uint32_t budget,
                      uint32_t chunk, int blocks, int threads, FILE *fo)
{
    const uint32_t n = Q->n;
    const uint32_t nb = (n + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    uint32_t nr = 0, novf = 0;
    cudaEvent_t e0 = NULL, e1 = NULL, e2 = NULL;
    float t0 = 0, t1 = 0;
    double h0;
    int rc = -1;
    if (!n) return 0;
#define COF_FLUSH_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    /* lim^2 is computed inside cf_run_rounds_dyn now, at the side's own width.
     * It used to be built here at a fixed 3 limbs, which was already the fix
     * for a 2-limb version that truncated lim0^2 for any rlim above 2^32 --
     * the same bug one width up. Keeping it next to the instantiation is what
     * stops it drifting from the splitter again: lim2 is the "prime by size"
     * threshold, and a wrong one accepts composites as prime, which surfaces
     * only as relations that fail to reconstruct. */

    /* Compact before every round. The scan is three small kernels over 131,072
     * slots -- microseconds against a stage that runs for tens of milliseconds --
     * and it is the ordered prefix scan, not an atomic append, so the job order
     * is a function of the status array alone and the output stays reproducible
     * byte for byte. */
    cf_work_t W;
    cf_sched_t S, S1;
    W.d_flag = Q->d_flag; W.d_off = Q->d_off; W.d_bsum = Q->d_bsum;
    W.d_sel = Q->d_sel;   W.d_nsel = Q->d_nsel;
    S.method = Q->meth[0]; S.rounds = rounds; S.budget = budget;
    S.curves = Q->ecm_curves; S.d_s = Q->d_s; S.ns = Q->ns;
    S.d_s2mask = Q->d_s2mask; S.s2vmin = Q->s2vmin; S.s2nv = Q->s2nv;
    /* chunk 0 is AUTO, not off: a volunteer cannot be asked to pick a number
     * for hardware nobody has measured, and picking none is what the field
     * failures were. An explicit value is honoured as given and never steered,
     * which is what the A/B above needs to hold one arm still. "Off" is
     * expressible as any value >= the batch -- one slice, measured
     * indistinguishable from the unchunked path. */
    {
        /* Q->chunk_cur always holds the slice currently in effect, for both
         * modes, so one place reports it and one place compares against it.
         *
         * The auto opening value is the floor and is NOT clamped to this
         * flush's n: a band whose first flush is a small partial one would
         * otherwise store that small n and then need several flushes of
         * doubling to climb back, running the full batches in between at a
         * slice far below the thread count -- the +150% regime. A stored value
         * above n is harmless, since cf_run_rounds treats any chunk >= n as
         * the single-slice case. */
        const uint32_t floor_ch = cof_chunk_floor(blocks, threads);
        const uint32_t want = chunk ? chunk
                            : (Q->chunk_cur ? Q->chunk_cur : floor_ch);
        if (Q->chunk_cur != want) {
            Q->chunk_cur = want;
            cof_report_chunk(Q->chunk_cur, n, chunk != 0);
        }
        S.chunk = Q->chunk_cur;
    }
    S1 = S; S1.method = Q->meth[1];   /* differs only in method */

    COF_FLUSH_CK(cudaEventCreate(&e0));
    COF_FLUSH_CK(cudaEventCreate(&e1));
    COF_FLUSH_CK(cudaEventCreate(&e2));
    COF_FLUSH_CK(cudaEventRecord(e0));
    if (cf_run_rounds_dyn(Q->L0, Q->d_c0, lim0, lpb0, n, Q->d_st0,
                          Q->d_sp0, Q->d_nsp0, &S, &W, NULL, blocks, threads))
        goto done;
    /* The class-aware gate: a record whose rational side did not split is dead
     * whatever the algebraic side does, so its side-1 job is never started. */
    k_cof_gate<<<blocks, threads>>>(n, Q->d_st0, Q->d_st1);
    COF_FLUSH_CK(cudaEventRecord(e1));
    if (cf_run_rounds_dyn(Q->L1, Q->d_c1, lim1, lpb1, n, Q->d_st1,
                          Q->d_sp1, Q->d_nsp1, &S1, &W, NULL, blocks, threads))
        goto done;
    COF_FLUSH_CK(cudaEventRecord(e2));

    /* Instrumentation, not accounting: recorded after e2 so it cannot land in
     * the per-side timings this same flush reports. */
    k_cof_status_hist<<<blocks, threads>>>(n, Q->d_st0, Q->d_st1, Q->d_nst);

    k_rel_flags<<<blocks, threads>>>(n, Q->d_st0, Q->d_st1, Q->d_flag);
    k_scan_pass1<<<nb, TD_SCAN_BLK>>>(Q->d_flag, n, Q->d_off, Q->d_bsum);
    k_scan_pass2<<<1, 1024>>>(Q->d_bsum, nb);
    k_scan_pass3<<<nb, TD_SCAN_BLK>>>(Q->d_off, n, Q->d_bsum);
    k_rel_gather<<<blocks, threads>>>(n, Q->d_flag, Q->d_off, Q->rcap,
                                      Q->d_idx, Q->d_nrel);
    COF_FLUSH_CK(cudaDeviceSynchronize()); COF_FLUSH_CK(cudaGetLastError());
    COF_FLUSH_CK(cudaEventElapsedTime(&t0, e0, e1));
    COF_FLUSH_CK(cudaEventElapsedTime(&t1, e1, e2));
    Q->ms_rat += t0; Q->ms_alg += t1;

    /* Steer the NEXT flush. Only in auto mode, and only ever between flushes,
     * so the slice is fixed for the whole of the one just measured.
     *
     * Halving on an overrun and doubling on a large margin (rather than
     * scaling by the ratio) keeps this from chasing a single noisy flush; a
     * band runs many of them. The floor is the real protection -- a device slow
     * enough to stay over budget simply parks there, which is the free point
     * for it, and a device fast enough grows to one slice and stops. */
    if (!chunk) {
        const uint32_t floor_ch = cof_chunk_floor(blocks, threads);
        const uint32_t was = Q->chunk_cur;
        const float stage = t0 + t1;
        if (stage > COF_CHUNK_TARGET_MS) {
            const uint32_t half = Q->chunk_cur / 2;
            Q->chunk_cur = (half > floor_ch) ? half : floor_ch;
        } else if (stage < COF_CHUNK_TARGET_MS / 4.0f && Q->chunk_cur < Q->cap) {
            /* Capped at the QUEUE CAPACITY, not at this flush's n. Several of
             * pipeline.cuh's cofq_flush call sites are checkpoint- or
             * stop-driven, so a band's first flush can be far smaller than a
             * full one; capping at that n would store the small value and then
             * need one flush per doubling to climb back, running the full
             * flushes in between at a slice far below the grid width -- the
             * +50.7% (32768) to +150.7% (16384) regime measured on the 3090.
             * That is the same trap the opening value above is unclamped to
             * avoid. cap is the real ceiling: n <= Q->cap always, and a stored
             * value above n costs nothing because cf_run_rounds treats any
             * chunk >= n as the single-slice case. */
            Q->chunk_cur = (Q->chunk_cur > Q->cap / 2) ? Q->cap
                                                       : Q->chunk_cur * 2;
        }
        if (Q->chunk_cur != was) cof_report_chunk(Q->chunk_cur, n, 0);
    }

    COF_FLUSH_CK(cudaMemcpy(&nr, Q->d_nrel, 4, cudaMemcpyDeviceToHost));
    /* The width invariant, asserted rather than trusted: cof_classify rejects
     * a residual above mfb and resolve_and_check_cofactor_config refuses an mfb above
     * 32*L, so a record that did not fit its side's limbs means one of those
     * two is wrong. Stop -- the alternative is a relation built from a
     * truncated cofactor, which reconstructs to the wrong norm and is far
     * harder to trace back here. */
    COF_FLUSH_CK(cudaMemcpy(&novf, Q->d_ovf, 4, cudaMemcpyDeviceToHost));
    if (novf) {
        fprintf(stderr, "  cofac queue: %u residual(s) exceeded the %d/%d-limb"
                " cofactor width; mfb and the width are out of step\n",
                novf, Q->L0, Q->L1);
        goto done;
    }
    /* Unreachable while rcap == cap, which it now is: every relation comes from
     * a distinct queued record, so nr <= n <= cap. It was reachable under the
     * old rcap = cap/4, which assumed relations were at most a quarter of
     * candidates -- true at the c183's 2%, false at the c123's 56%, and it
     * failed a whole band. Kept as an assertion on that invariant rather than
     * deleted, because it is one comparison per flush and the sizing is the
     * kind of thing a later change quietly revisits. */
    if (nr > Q->rcap) {
        fprintf(stderr, "  cofac queue: %u relations in a flush of %u exceeds"
                " the %u readback slots (rcap invariant broken)\n",
                nr, n, Q->rcap);
        goto done;
    }
    Q->nseen += n; Q->nrel += nr;
    {   /* cumulative on the device, so this is a snapshot of the total rather
         * than a per-flush delta -- assign, do not add. */
        unsigned long long h[8];
        if (!CUDA_CHECKED(cudaMemcpy(h, Q->d_nst, sizeof h, cudaMemcpyDeviceToHost)))
            for (int sd = 0; sd < 2; sd++)
                for (int k = 0; k < 4; k++) Q->nst[sd][k] = h[sd * 4 + k];
    }
    if (!nr) { Q->n = 0; rc = 0; goto done; }

    k_rel_pack<<<blocks, threads>>>(nr, Q->d_idx, *Q, O->d_a, O->d_b,
                                    O->d_f0, O->d_fn0, O->d_f1, O->d_fn1,
                                    O->d_sp0, O->d_nsp0, O->d_sp1, O->d_nsp1);
    COF_FLUSH_CK(cudaDeviceSynchronize()); COF_FLUSH_CK(cudaGetLastError());
    h0 = host_ms();
    COF_FLUSH_CK(cudaMemcpy(O->a, O->d_a, (size_t)nr * 8, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->b, O->d_b, (size_t)nr * 8, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->f0, O->d_f0, (size_t)nr * TD_FMAX * 4, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->f1, O->d_f1, (size_t)nr * TD_FMAX * 4, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->fn0, O->d_fn0, nr, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->fn1, O->d_fn1, nr, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->sp0, O->d_sp0, (size_t)nr * CF_MAXFAC * 8, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->sp1, O->d_sp1, (size_t)nr * CF_MAXFAC * 8, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->nsp0, O->d_nsp0, nr, cudaMemcpyDeviceToHost));
    COF_FLUSH_CK(cudaMemcpy(O->nsp1, O->d_nsp1, nr, cudaMemcpyDeviceToHost));
    if (fo)
        for (uint32_t k = 0; k < nr; k++) {
            int64_t a = O->a[k], b = O->b[k];
            if (b < 0) { a = -a; b = -b; }
            fprintf(fo, "%lld,%lld:", (long long)a, (long long)b);
            cq_emit_side(fo, O->f0 + (size_t)k * TD_FMAX, O->fn0[k],
                         O->sp0 + (size_t)k * CF_MAXFAC, O->nsp0[k]);
            fputc(':', fo);
            cq_emit_side(fo, O->f1 + (size_t)k * TD_FMAX, O->fn1[k],
                         O->sp1 + (size_t)k * CF_MAXFAC, O->nsp1[k]);
            fputc('\n', fo);
        }
    Q->ms_host += host_ms() - h0;
    Q->n = 0;
    rc = 0;

done:
    if (e0 && CUDA_CHECKED(cudaEventDestroy(e0)) && rc == 0) rc = -1;
    if (e1 && CUDA_CHECKED(cudaEventDestroy(e1)) && rc == 0) rc = -1;
    if (e2 && CUDA_CHECKED(cudaEventDestroy(e2)) && rc == 0) rc = -1;
#undef COF_FLUSH_CK
    return rc;
}

/* ---- host driver: cofactorise an emitted candidate batch ---------------- *
 *
 * Reads the file `--candidates` writes -- `res0,res1:a,b:f0:f1`, res negative
 * when that side still needs splitting -- and emits the relations that come out
 * of it. Deliberately a separate entry point from the pipeline for now: the
 * corpus is 1,958,036 real candidates from a 1,000-q band, which is a far
 * better test of the splitter than one q's worth would be, and it keeps the
 * algorithm measurable before it is wired into the band loop.
 */

/* OR of every limb from HI upward -- "is there anything above the low HI
 * limbs". Written as a loop over L rather than as a named limb so that raising
 * CF_LMAX cannot leave a limb unexamined by a fit test; the two places that
 * asked `v[2]` directly were both correct only at exactly 3 limbs. */
template <int L, int HI> CF_HD uint32_t cf_hi_limbs(const mz<L> *x)
{
    uint32_t o = 0;
    for (int i = HI; i < L; i++) o |= x->v[i];   /* L - HI is 1 or 2 */
    return o;
}

/* Decimal to L limbs. Returns 0 if it does not fit. */
template <int L>
static int cf_from_dec(mz<L> *r, const char *s, int *needs_split)
{
    *needs_split = 0;
    if (*s == '-') { *needs_split = 1; s++; }
    for (int i = 0; i < L; i++) r->v[i] = 0;
    for (; *s >= '0' && *s <= '9'; s++) {
        uint64_t c = (uint64_t)(*s - '0');
        for (int i = 0; i < L; i++) {
            uint64_t t = (uint64_t)r->v[i] * 10u + c;
            r->v[i] = (uint32_t)t;
            c = t >> 32;
        }
        if (c) return 0;
    }
    return 1;
}

typedef struct {
    uint32_t nrec, njob[2], nsplit[2], ndead[2], nstuck[2], nrel;
    double ms[2];
} cof_stat_t;

/* One side's queue: launch, then relaunch whatever is still CF_INCOMPLETE with
 * a different rho constant and a bigger budget. The escalation is the point --
 * a cofactor that resists c=1 is not a cofactor that cannot be split, so the
 * only correct way to stop is to run out of ROUNDS, and even then the record is
 * reported as stuck rather than counted as dead. */
template <int L>
static int cf_run_side(mz<L> *h_n, uint32_t nj, uint64_t lim, uint32_t lpb,
                       uint8_t *h_status, uint64_t *h_fac, uint8_t *h_nfac,
                       int blocks, int threads, int verbose, double *ms_out,
                       const cf_sched_t *S, const char *side_name)
{
    mz<L> *d_n = NULL; uint8_t *d_status = NULL, *d_nfac = NULL;
    uint64_t *d_fac = NULL;
    unsigned long long *d_iters = NULL, *h_iters = NULL;
    cf_work_t W;
    const uint32_t nb = (nj + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    mz<L> lim2;
    cudaEvent_t e0, e1;
    float ms = 0;
    if (!nj) { *ms_out = 0; return 0; }
    memset(&W, 0, sizeof(W));

    if (cf_lim2<L>(&lim2, lim)) return -1;
    CK(cudaMalloc(&d_n, (size_t)nj * sizeof(mz<L>)));
    CK(cudaMalloc(&d_status, nj));
    CK(cudaMalloc(&d_nfac, nj));
    CK(cudaMalloc(&d_fac, (size_t)nj * CF_MAXFAC * 8));
    CK(cudaMalloc(&d_iters, (size_t)nj * 8));
    /* The standalone path used to run every slot every round, because the
     * compaction lived only in the inline queue. Same executor now, so it gets
     * the same scheduling -- and the same output, which the golden test pins. */
    CK(cudaMalloc(&W.d_flag, (size_t)nj * 4));
    CK(cudaMalloc(&W.d_off,  (size_t)nj * 4));
    CK(cudaMalloc(&W.d_bsum, (size_t)nb * 4));
    CK(cudaMalloc(&W.d_sel,  (size_t)nj * 4));
    CK(cudaMalloc(&W.d_nsel, 4));
    CK(cudaMemcpy(d_n, h_n, (size_t)nj * sizeof(mz<L>), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_status, CF_INCOMPLETE, nj));
    CK(cudaMemset(d_nfac, 0, nj));
    CK(cudaMemset(d_iters, 0, (size_t)nj * 8));
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    cudaEventRecord(e0);
    cf_run_rounds<L>(d_n, lim2, lpb, nj, d_status, d_fac, d_nfac,
                     S, &W, d_iters, blocks, threads);
    /* Release before returning. run_cofac calls this twice -- once per side,
     * possibly at different widths -- on the same device, so a bare `return -1`
     * here left the second call short by everything the first had allocated: on
     * the 1.96M-record corpus, hundreds of MB, and the second side then fails
     * for a reason that has nothing to do with itself. */
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "  cofac: launch failed\n");
        cudaFree(d_n); cudaFree(d_status); cudaFree(d_nfac); cudaFree(d_fac);
        cudaFree(d_iters);
        cudaFree(W.d_flag); cudaFree(W.d_off); cudaFree(W.d_bsum);
        cudaFree(W.d_sel); cudaFree(W.d_nsel);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return -1;
    }
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    cudaEventElapsedTime(&ms, e0, e1);
    *ms_out = ms;
    /* Named by SIDE, not keyed on L. Keyed on L it said "side 3-limb queue"
     * twice as soon as both sides became 3 limbs -- the same information loss
     * the summary block below was fixed for, and it would come straight back
     * now that 4/4 is a legal shape. run_cofac calls rational first. */
    if (verbose) printf("  %s queue: %u jobs, %d rounds, %.1f ms\n",
                        side_name ? side_name : "cofactor", nj, S->rounds, ms);

    CK(cudaMemcpy(h_status, d_status, nj, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_nfac, d_nfac, nj, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_fac, d_fac, (size_t)nj * CF_MAXFAC * 8, cudaMemcpyDeviceToHost));

    /* Where does the bounded splitting work actually go? Rho reports loop
     * iterations. ECM reports ladder bits plus every stage-2 giant step and
     * selected pair, including a swallowed-factor replay; gcd work is excluded.
     * The units are meaningful within a method, while device-event time compares
     * methods. Keeping stage 1's original counter matters: observability must
     * not itself increase that kernel's register footprint. */
    h_iters = (unsigned long long *)malloc((size_t)nj * 8);
    if (h_iters) {
        unsigned long long it[6] = {0,0,0,0,0,0}, tot = 0;
        uint32_t cnt[6] = {0,0,0,0,0,0};
        CK(cudaMemcpy(h_iters, d_iters, (size_t)nj * 8, cudaMemcpyDeviceToHost));
        for (uint32_t k = 0; k < nj; k++) {
            int s = h_status[k]; if (s < 0 || s > 5) s = 5;
            it[s] += h_iters[k]; cnt[s]++; tot += h_iters[k];
        }
        if (verbose && tot) {
            static const char *nm[6] = {"split ok", "dead", "stuck", "overflow", "?", "?"};
            printf("  %d-limb %s, by outcome (total %.3g):\n", L,
                   S->method ? "ECM work units (ladder bits + stage-2 walk)"
                             : "rho iterations",
                   (double)tot);
            for (int s = 0; s < 4; s++)
                if (cnt[s]) printf("    %-9s %8u jobs  %6.2f%% of work   %9.0f mean\n",
                                   nm[s], cnt[s], 100.0 * (double)it[s] / (double)tot,
                                   (double)it[s] / cnt[s]);
        }
        free(h_iters);
    }

    cudaFree(d_n); cudaFree(d_status); cudaFree(d_nfac); cudaFree(d_fac);
    cudaFree(d_iters);
    cudaFree(W.d_flag); cudaFree(W.d_off); cudaFree(W.d_bsum);
    cudaFree(W.d_sel); cudaFree(W.d_nsel);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}


/* Run one side at a runtime width, narrowing from the parse's CF_LMAX-wide
 * array. The parse cannot know how wide a decimal cofactor is until it has
 * read it, so it always runs at the widest width this build has and the
 * narrowing happens here -- one pass over a host array on a diagnostic path,
 * against duplicating the whole parser per width.
 *
 * The high-limb check is not decoration. Narrowing is precisely the silent
 * truncation the width machinery exists to remove, so a value that does not
 * fit the requested width fails the batch instead of being cut down to it. */
template <int L>
static int cf_run_side_at(const mz<CF_LMAX> *h_n, uint32_t nj, uint64_t lim,
                          uint32_t lpb, uint8_t *h_status, uint64_t *h_fac,
                          uint8_t *h_nfac, int blocks, int threads, int verbose,
                          double *ms_out, const cf_sched_t *S,
                          const char *side_name)
{
    mz<L> *nn = (mz<L> *)malloc((size_t)(nj ? nj : 1) * sizeof(mz<L>));
    int r;
    if (!nn) {
        fprintf(stderr, "  cofac: out of memory narrowing to %d limbs\n", L);
        return -1;
    }
    for (uint32_t k = 0; k < nj; k++) {
        /* cf_hi_limbs, not a hand-rolled loop: the helper exists precisely so
         * that a fit test is written once as a loop over the width, and a
         * second copy here would be the one place a new limb went unexamined
         * if CF_LMAX ever moves again. */
        if (cf_hi_limbs<CF_LMAX, L>(&h_n[k])) {
            fprintf(stderr, "  cofac: %s record %u does not fit %d limbs\n",
                    side_name ? side_name : "cofactor", k, L);
            free(nn); return -1;
        }
        for (int i = 0; i < L; i++) nn[k].v[i] = h_n[k].v[i];
    }
    r = cf_run_side<L>(nn, nj, lim, lpb, h_status, h_fac, h_nfac,
                       blocks, threads, verbose, ms_out, S, side_name);
    free(nn);
    return r;
}

static int cf_run_side_dyn(int L, const mz<CF_LMAX> *h_n, uint32_t nj,
                           uint64_t lim, uint32_t lpb, uint8_t *h_status,
                           uint64_t *h_fac, uint8_t *h_nfac, int blocks,
                           int threads, int verbose, double *ms_out,
                           const cf_sched_t *S, const char *side_name)
{
    switch (L) {
    case 3: return cf_run_side_at<3>(h_n, nj, lim, lpb, h_status, h_fac,
                                     h_nfac, blocks, threads, verbose, ms_out,
                                     S, side_name);
#if CF_LMAX >= 4
    case 4: return cf_run_side_at<4>(h_n, nj, lim, lpb, h_status, h_fac,
                                     h_nfac, blocks, threads, verbose, ms_out,
                                     S, side_name);
#endif
    default: break;
    }
    fprintf(stderr, "  cofac: this build has no %d-limb splitter"
            " (CF_LMAX = %d)\n", L, CF_LMAX);
    return -1;
}

/* ---- post-cofactor reconstruction gate --------------------------------- *
 *
 * The existing gate (pipe_td_verify) checks trial division's factors against
 * the norm BEFORE splitting, so it says nothing about the primes the
 * cofactoriser emits. This one runs over the emitted relation file, which both
 * the inline queue and the standalone splitter write in the same format, so one
 * implementation gates both paths.
 *
 * For each relation it recomputes F(a,b) and G(a,b) exactly from the
 * polynomial, divides out every recorded factor, and requires the remainder to
 * be exactly 1 with every prime within its side's lpb. That is the property a
 * relation actually has to have, and nothing upstream establishes it: a
 * truncated 33-bit factor, a dropped power of two, or a split that emitted a
 * composite all reconstruct wrongly here and nowhere else.
 */
static int cf_norm_exact(bn_t *out, const tdpoly_t *P, int64_t a, int64_t b)
{
    bns_t acc; bns_zero(&acc);
    const uint64_t ua = (uint64_t)(a < 0 ? -a : a), ub = (uint64_t)(b < 0 ? -b : b);
    const int sa = a < 0 ? -1 : 1, sb = b < 0 ? -1 : 1;
    int ovf = 0;
    for (int k = 0; k <= P->deg; k++) {
        bn_t term = P->c[k];
        int s = P->sign[k];
        if (bn_is_zero(&term)) continue;
        for (int e = 0; e < k; e++) { if (bn_mul_u64(&term, ua)) ovf = 1; s *= sa; }
        for (int e = 0; e < P->deg - k; e++) { if (bn_mul_u64(&term, ub)) ovf = 1; s *= sb; }
        if (bns_addmag(&acc, &term, s)) ovf = 1;
    }
    *out = acc.m;
    return ovf;
}

/* ---- 64-bit helpers for the gate --------------------------------------- *
 *
 * HOST ONLY, and deliberately so: `__uint128_t` is a compiler extension, and
 * these run once per emitted factor in a verification pass, never in a kernel.
 * They exist because raising lpb past 32 moved the gate's own arithmetic out
 * of range -- a 33-bit prime cannot be primality-tested by bench_is_prime32
 * nor divided out by bn_divmod_u32_pre, so without these the gate would have
 * had to skip exactly the factors the widening introduces. */

/* x /= d, returning the remainder. Steps in 64-BIT limbs, not 32: with a
 * divisor above 2^32 a 32-bit-limb long division produces quotient digits that
 * do not fit the limb they are stored in. */
static uint64_t cf_bn_divmod_u64(bn_t *x, uint64_t d)
{
    uint64_t rem = 0;
    for (int i = BN_LIMBS - 2; i >= 0; i -= 2) {
        const uint64_t lo = ((uint64_t)x->v[i + 1] << 32) | x->v[i];
        const uint64_t q = bench_div_u128_u64(rem, lo, d, &rem);
        x->v[i] = (uint32_t)q;
        x->v[i + 1] = (uint32_t)(q >> 32);
    }
    return rem;
}

/* REQUIRES a < m and b < m. That was a free-standing nicety while this was
 * __int128 arithmetic, which just computes the remainder whatever the inputs.
 * It is load-bearing now: bench_div_u128_u64 is _udiv128 on MSVC x64, and
 * _udiv128 raises a hardware divide-overflow (#DE) when the quotient does not
 * fit 64 bits. a, b < m keeps a*b < m^2 and so the quotient below m, which is
 * the only thing standing between this and a Windows-only crash in the
 * primality test. cf_is_prime64 reduces its witnesses (W[i] % n) for exactly
 * this reason; a future caller that does not must reduce first. */
static uint64_t cf_mulmod64(uint64_t a, uint64_t b, uint64_t m)
{
    const bench_u128_t v = bench_mul_u64_wide(a, b);
    uint64_t rem = 0;
    (void)bench_div_u128_u64(v.hi, v.lo, m, &rem);
    return rem;
}

/* Miller-Rabin over the seven bases that are DETERMINISTIC for every n < 2^64
 * (Jaeschke / Sinclair). Not probabilistic: the gate's whole purpose is to
 * catch a composite emitted as a prime, so a test that can be fooled by one
 * would be no test at all. */
static int cf_is_prime64(uint64_t n)
{
    static const uint64_t W[7] = { 2, 325, 9375, 28178, 450775, 9780504,
                                   1795265022 };
    uint64_t d = n - 1;
    int s = 0;
    if (n < 2) return 0;
    if (!(n & 1)) return n == 2;
    while (!(d & 1)) { d >>= 1; s++; }
    for (int i = 0; i < 7; i++) {
        uint64_t a = W[i] % n, x = 1, p = a;
        uint64_t e = d;
        int r, composite = 1;
        if (!a) continue;                 /* n divides the witness */
        while (e) {
            if (e & 1) x = cf_mulmod64(x, p, n);
            p = cf_mulmod64(p, p, n);
            e >>= 1;
        }
        if (x == 1 || x == n - 1) continue;
        for (r = 1; r < s; r++) {
            x = cf_mulmod64(x, x, n);
            if (x == n - 1) { composite = 0; break; }
        }
        if (composite) return 0;
    }
    return 1;
}

/* Returns 0 if every relation reconstructs, else the number that did not. */
/* One line, verified in place (the parse writes NULs over the separators).
 * Returns 1 if it rebuilds both norms exactly, 0 if it does not, and -1 for a
 * comment or blank that is not a relation at all.
 *
 * Split out of cf_check_relations so resume can spot-check a handful of lines
 * from the head and tail of a multi-gigabyte .part instead of rescanning it. */
static int cf_check_one(char *line, tdpoly_t *tp, uint32_t lpb0, uint32_t lpb1,
                        uint32_t *nprime, uint32_t *ncomp, uint32_t *nonprim)
{
    int64_t a, b;
    char *p = line, *e;
    if (*p == '#' || *p == '\n' || !*p) return -1;
    a = strtoll(p, &e, 10); if (*e != ',') return 0;
    b = strtoll(e + 1, &e, 10); if (*e != ':') return 0;
    {
            bn_t N[2];
            char *fld[2];
            int side, bad = 0;
            /* fields: "a,b" ":" side-0 primes ":" side-1 primes */
            fld[0] = e + 1;
            fld[1] = strchr(fld[0], ':');
            if (!fld[1]) return 0;
            *fld[1]++ = 0;
            { char *nlp = strchr(fld[1], '\n'); if (nlp) *nlp = 0; }
            if (cf_norm_exact(&N[1], &tp[1], a, b) ||
                cf_norm_exact(&N[0], &tp[0], a, b)) return 0;
            /* PRIMITIVITY. Both norms can rebuild to exactly 1 and the
             * relation still be worthless: if gcd(a,b) = g > 1 then (a,b) is
             * g times a smaller pair and carries no new information, and
             * msieve refuses it outright ("error -6", relation.c). This gate
             * checked every factor and both norms but not this, which is why a
             * whole band of non-primitive relations shipped before anyone
             * noticed downstream. See the primitivity note on
             * k_intersect_compact for how they arise. */
            {
                int64_t u = a < 0 ? -a : a, v = b < 0 ? -b : b;
                while (v) { int64_t t = u % v; u = v; v = t; }
                /* Return 2, NOT 0. A non-primitive relation reconstructs
                 * perfectly -- both norms rebuild, every factor prime and in
                 * range -- so folding it into the "does not reconstruct" count
                 * makes the RESUME spot-check refuse a sound .part with a
                 * false diagnosis, sending the operator after file corruption
                 * or a wrong polynomial. The two callers want opposite things:
                 * the --check-relations GATE must fail on it, while resume must
                 * not, because a .part written by a pre-finding-68 binary
                 * legitimately holds ~0.1% of them on a small-q job and that is
                 * no reason to strand a multi-day band. Resume tests `if (r)`
                 * and so accepts 2; the gate tests `r == 1`. */
                if (u != 1) { (*nonprim)++; return 2; }
            }
            for (side = 0; side < 2 && !bad; side++) {
                const uint32_t lpb = side ? lpb1 : lpb0;
                char *q = fld[side];
                /* The largest prime this side may carry. Computed rather than
                 * written as `1ull << lpb` because lpb may now be 64, where
                 * that shift is undefined. */
                const uint64_t maxp = lpb >= 64 ? 0xFFFFFFFFFFFFFFFFull
                                                : ((1ull << lpb) - 1u);
                while (*q) {
                    unsigned long long v;
                    errno = 0;
                    v = strtoull(q, &e, 16);
                    if (e == q) { bad = 1; break; }
                    /* ERANGE, not just the lpb bound. Dropping the old
                     * `v > 0xffffffff` test also dropped the only rejection of
                     * strtoull's saturation, and at lpb 64 the bound below
                     * cannot catch it: a field wider than 64 bits returns
                     * ULLONG_MAX, which is <= maxp. It would then be reported
                     * as a composite factor rather than as a malformed line. */
                    if (errno == ERANGE) { (*nprime)++; bad = 1; break; }
                    /* 64-bit throughout. This used to also reject anything
                     * above 2^32 outright, which was correct while the
                     * cofactoriser could not represent such a factor and is
                     * now exactly wrong: at lpb 33 those are the factors the
                     * widening exists to emit. The lpb bound is the real one.  */
                    if (!v || v > maxp) {
                        (*nprime)++; bad = 1; break;
                    }
                    /* THE GATE'S BLIND SPOT UNTIL NOW. Exact division cannot
                     * see a composite: replace a relation's `p,q` with the
                     * single value p*q and the norm still divides down to 1,
                     * so every other check here passes. That is precisely the
                     * shape an incompletely split cofactor has -- the most
                     * likely real defect in the ECM/rho path, and the one
                     * failure this gate exists to catch. */
                    if (!(v <= 0xffffffffull ? bench_is_prime32((uint32_t)v)
                                             : cf_is_prime64(v))) {
                        (*ncomp)++; bad = 1; break;
                    }
                    {   /* exact division: a factor that does not divide is a
                         * reconstruction failure, not a rounding question.
                         * The 32-bit Barrett path stays for the common case --
                         * every trial-division factor is an FB prime below
                         * lim -- and only a large prime pays for 128-bit. */
                        bn_t t = N[side];
                        int top = bn_top(&t);
                        if (top < 0) { bad = 1; break; }
                        if (v <= 0xffffffffull) {
                            if (bn_divmod_u32_pre(&t, (uint32_t)v,
                                                  bn_recip_u32((uint32_t)v), top)) {
                                bad = 1; break;
                            }
                        } else if (cf_bn_divmod_u64(&t, v)) {
                            bad = 1; break;
                        }
                        N[side] = t;
                    }
                    q = e;
                    if (*q == ',') q++; else break;
                }
            }
        return (!bad && bn_is_one(&N[0]) && bn_is_one(&N[1])) ? 1 : 0;
    }
}

static int cf_check_relations(const char *path, const poly_t *poly,
                              uint32_t lpb0, uint32_t lpb1)
{
    tdpoly_t tp[2];
    FILE *f = fopen(path, "rb");
    char *line = NULL;
    size_t cap = 0;
    uint32_t nok = 0, nbad = 0, nprime = 0, ncomp = 0, nonprim = 0;
    if (!f) { perror(path); return -1; }
    if (td_build_poly(&tp[1], poly, 1) || td_build_poly(&tp[0], poly, 0)) {
        fprintf(stderr, "check-relations: polynomial does not fit\n");
        fclose(f); return -1;
    }
    while (bench_getline(&line, &cap, f) > 0) {
        /* A full gate over a multi-gigabyte .part runs for minutes. Without a
         * heartbeat the watchdog reports it as a stall, which trains the
         * operator to ignore the one message that matters. */
        wd_phase("resume.check_relations");
        const int r = cf_check_one(line, tp, lpb0, lpb1, &nprime, &ncomp, &nonprim);
        /* r == 1 only: r == 2 is "reconstructs but is not primitive", which
         * this gate must fail. */
        if (r == 1) nok++; else if (r >= 0) nbad++;
    }
    free(line); fclose(f);
    printf("  relation reconstruction gate: %u of %u rebuild both norms exactly",
           nok, nok + nbad);
    if (nprime) printf(", %u carried a factor outside [2, 2^lpb)", nprime);
    if (ncomp)  printf(", %u carried a COMPOSITE factor", ncomp);
    /* Named separately from the composite count. A non-primitive relation is
     * perfectly factored -- both norms rebuild -- so reporting it as a
     * composite factor sends the reader after the cofactoriser instead of
     * after the survivor filter. */
    if (nonprim) printf(", %u NON-PRIMITIVE (gcd(a,b) != 1; msieve error -6)",
                        nonprim);
    printf("  %s\n", nbad ? "<-- FAIL" : "PASS");
    return (int)nbad;
}

extern "C" int check_relations(const char *path, const poly_t *poly,
                               uint32_t lpb0, uint32_t lpb1)
{
    return cf_check_relations(path, poly, lpb0, lpb1);
}

/* Spot-check a relation file without reading all of it: the first `n` relation
 * lines and the last `n` that fall inside the final 64 KiB. Resume uses this to
 * refuse a .part that does not belong to the job being run.
 *
 * It proves the POLYNOMIAL, and nothing else. logI, J, lambda, mfb, lpb,
 * sq-side and the factor-base convention can all differ while every line still
 * reconstructs exactly -- that is what the checkpoint fingerprint is for.
 * Returns the number of bad lines, or -1 if the file could not be read. */
extern "C" int check_relations_sample(const char *path, const poly_t *poly,
                                      uint32_t lpb0, uint32_t lpb1,
                                      uint32_t n, unsigned long long limit,
                                      uint32_t *checked)
{
    enum { TAILWIN = 65536 };
    tdpoly_t tp[2];
    FILE *f = fopen(path, "rb");
    char *line = NULL, **keep = NULL;
    size_t cap = 0;
    uint32_t nok = 0, nbad = 0, nprime = 0, ncomp = 0, nonprim = 0, seen = 0, nkeep = 0;
    int64_t size, end, head_end = 0;
    if (checked) *checked = 0;
    if (!f) { perror(path); return -1; }
    if (td_build_poly(&tp[1], poly, 1) || td_build_poly(&tp[0], poly, 0)) {
        fprintf(stderr, "resume: polynomial does not fit\n");
        fclose(f); return -1;
    }
    /* 64-bit throughout: `long` is 32 bits on Windows, and this gate exists
     * for multi-gigabyte .part files. A 32-bit ftell there returns -1 and the
     * <= 0 test below would send us straight to done:, reporting "spot-checked
     * 0 relations" as a PASS and admitting a .part from any other job. */
    if (bench_seek_end(f) != 0 || (size = bench_tell(f)) <= 0) goto done;
    /* `limit` is the checkpointed prefix, and everything past it is about to be
     * truncated away. Stopping here is not an optimisation: a kill -9 leaves a
     * torn final line beyond the checkpoint, and sampling it would report a
     * corrupt file and refuse a resume that is in fact perfectly sound. */
    end = (limit && (int64_t)limit < size) ? (int64_t)limit : size;
    rewind(f);
    while (seen < n && bench_getline(&line, &cap, f) > 0) {
        int r;
        if (bench_tell(f) > end) break; /* the line crosses the prefix end */
        r = cf_check_one(line, tp, lpb0, lpb1, &nprime, &ncomp, &nonprim);
        if (r < 0) continue;
        seen++;
        if (r) nok++; else nbad++;
        head_end = bench_tell(f);
    }
    /* The tail matters more than the head: a truncated or mis-joined write
     * lands there, and the head of a resumed file was already checked by
     * whatever session wrote it. */
    {
        /* Never re-read what the head pass already covered. On a prefix under
         * 64 KiB the window would otherwise start at 0, so the same lines get
         * checked twice and `seen`, `nok` and `nbad` all double -- a 6-relation
         * file reporting "spot-checked 12 relations", and one bad line
         * reported as two. */
        int64_t off = end > TAILWIN ? end - TAILWIN : 0;
        if (off < head_end) off = head_end;
        if (off < end && bench_seek(f, (uint64_t)off) == 0) {
            /* Only resync to a line boundary when the window was cut mid-file;
             * head_end is already one. */
            if (off && off != head_end) {
                if (bench_getline(&line, &cap, f) <= 0) goto done;
            }
            keep = (char **)calloc(n, sizeof *keep);
            if (!keep) goto done;
            while (bench_getline(&line, &cap, f) > 0) {
                if (bench_tell(f) > end) break;
                if (line[0] == '#' || line[0] == '\n') continue;
                free(keep[nkeep % n]);
                keep[nkeep % n] = bench_strdup(line);
                nkeep++;
            }
            {   /* replay in file order; the ring holds the last min(n, nkeep) */
                const uint32_t have = nkeep < n ? nkeep : n;
                const uint32_t first = nkeep < n ? 0 : nkeep % n;
                for (uint32_t i = 0; i < have; i++) {
                    char *s = keep[(first + i) % n];
                    int r;
                    if (!s) continue;
                    r = cf_check_one(s, tp, lpb0, lpb1, &nprime, &ncomp, &nonprim);
                    if (r < 0) continue;
                    seen++;
                    if (r) nok++; else nbad++;
                }
            }
            for (uint32_t i = 0; i < n; i++) free(keep[i]);
            free(keep);
        }
    }
done:
    free(line); fclose(f);
    if (checked) *checked = seen;
    printf("  resume: spot-checked %u relations from %s -- %u rebuild both"
           " norms exactly", seen, path, nok);
    if (nprime) printf(", %u out of lpb range", nprime);
    if (ncomp)  printf(", %u composite", ncomp);
    /* Reported, not fatal: see cf_check_one. A pre-finding-68 .part carries
     * these legitimately, and refusing the resume over them would strand the
     * band for a population msieve merely skips. */
    if (nonprim)
        printf(", %u NON-PRIMITIVE (harmless here; msieve skips them, and a"
               " rebuilt binary no longer emits them)", nonprim);
    printf("  %s\n", nbad ? "<-- FAIL" : "PASS");
    return (int)nbad;
}

typedef struct { char *ab, *f0, *f1; } cf_rest_t;

/* Fill `f` with a side's trial-division list plus its split primes, in the
 * relation format's terms. Returns the count, or -1 if they do not all fit.
 *
 * THE TWO BOUNDS ARE DIFFERENT ON PURPOSE. The parse stops at TD_FMAX, which
 * is what trial division can emit; the append then runs to TD_FMAX+CF_MAXFAC,
 * the space reserved for split primes. Bounding BOTH loops at the combined
 * size -- which is what this did -- let a long trial-division list eat the
 * reservation, after which every split prime was dropped and the relation was
 * written anyway, with a factor product that is not the norm. It then failed
 * reconstruction somewhere else entirely, with nothing pointing back here.
 *
 * This path parses an externally supplied --candidates file, so a malformed or
 * truncated record is a real input, not a can't-happen. The inline queue
 * refuses the same condition outright (pipeline.cuh, `c0 > TD_FMAX`). */
static int cf_fill_side(uint64_t *f, const char *tdfac, const uint64_t *extra,
                        int nextra)
{
    int n = 0;
    const char *p = tdfac;
    while (*p) {
        char *end;
        unsigned long long v;
        if (n >= TD_FMAX) return -1;      /* more TD factors than TD can make */
        v = strtoull(p, &end, 16);
        if (end == p) break;
        f[n++] = (uint64_t)v;
        p = end; if (*p == ',') p++;
    }
    if (nextra < 0 || n + nextra > TD_FMAX + CF_MAXFAC) return -1;
    for (int i = 0; i < nextra; i++) f[n++] = extra[i];
    return n;
}

/* Both sides must be built BEFORE either is written: the relation is one line,
 * so discovering the overflow half way through would leave a truncated line in
 * the output, which is worse than the bug it guards.
 *
 * Fills both sides into caller-owned buffers and hands back the counts, rather
 * than filling them to answer "does it fit" and throwing them away -- the
 * emit then re-parsed the same hex lists, four cf_fill_side calls per relation
 * instead of two. */
typedef struct {
    uint64_t f0[TD_FMAX + CF_MAXFAC + 2];
    uint64_t f1[TD_FMAX + CF_MAXFAC + 2];
    int n0, n1;
} cf_rec_t;

static int cf_rec_build(cf_rec_t *r, const char *f0, const uint64_t *e0, int n0,
                        const char *f1, const uint64_t *e1, int n1)
{
    r->n0 = cf_fill_side(r->f0, f0, e0, n0);
    r->n1 = cf_fill_side(r->f1, f1, e1, n1);
    return r->n0 >= 0 && r->n1 >= 0;
}

/* Cofactorise a `--candidates` batch and write the relations it yields.
 *
 * CLASS-AWARE DISPATCH: the rational side runs FIRST and alone. It is one
 * 60-bit semiprime split against the algebraic side's 92-bit 3LP one, and 51%
 * of the corpus needs both. A record whose rational side has a prime factor
 * above 2^lpb is dead whatever the algebraic side does, so its algebraic job is
 * never enqueued at all. */
extern "C" int run_cofac(const char *path, const char *out, uint32_t lim0,
                         uint32_t lpb0, uint32_t lim1, uint32_t lpb1,
                         int rounds, uint32_t budget, int blocks, int threads,
                         int meth0, int meth1, uint32_t ecm_b1, uint32_t ecm_b2,
                         uint32_t ecm_curves, int limbs0, int limbs1)
{
    FILE *f = fopen(path, "rb");
    uint32_t *h_s = NULL, *d_s = NULL, ns = 0;
    uint8_t *h_s2mask = NULL, *d_s2mask = NULL;
    uint32_t s2vmin = 0, s2nv = 0;
    cf_sched_t sched;
    char *buf = NULL;
    size_t sz = 0;
    uint32_t nrec = 0, n0 = 0, n1 = 0;
    cf_rest_t *rest = NULL;
    /* Parsed at the build's widest width; cf_run_side_dyn narrows to the
     * width each side actually runs at. */
    mz<CF_LMAX> *j0 = NULL; mz<CF_LMAX> *j1 = NULL;
    uint32_t *i0 = NULL, *i1 = NULL;
    uint64_t *fac0 = NULL, *fac1 = NULL;   /* resulting primes: lpb-bounded */
    uint8_t *st0 = NULL, *st1 = NULL, *nf0 = NULL, *nf1 = NULL;
    uint8_t *side0ok = NULL;
    uint64_t *small0 = NULL, *small1 = NULL;
    double ms0 = 0, ms1 = 0, thost;
    uint32_t nrel = 0, nrel2 = 0, dead0 = 0, dead1 = 0, stuck0 = 0, stuck1 = 0, novf = 0;
    /* Distinct from novf, which counts records whose SPLITTER produced more
     * than CF_MAXFAC factors. This counts records whose trial-division list
     * plus split primes will not fit the emit buffer -- a property of the
     * input file, not of the split. */
    uint32_t nover = 0;
    cf_rec_t rb;
    uint8_t *alg_job = NULL;
    FILE *fo = NULL;
    char tmp[2048];

    if (!f) { perror(path); return -1; }
    /* 64-bit sizing, then an explicit range check before the cast to size_t.
     * A 32-bit ftell on Windows wraps for anything past 2 GB, and a wrapped
     * positive value would be read back as a complete file -- fread returns
     * exactly the short count asked for, so the truncation passes the check
     * below and the batch silently splits a prefix of its input. */
    {
        int64_t fsz;
        if (bench_seek_end(f) || (fsz = bench_tell(f)) < 0 || bench_seek(f, 0)) {
            perror(path); fclose(f); return -1;
        }
        if ((uint64_t)fsz > (uint64_t)(SIZE_MAX - 1)) {
            fprintf(stderr, "cofac: %s is too large to read into memory\n", path);
            fclose(f); return -1;
        }
        sz = (size_t)fsz;
    }
    buf = (char *)malloc(sz + 1);
    if (!buf || fread(buf, 1, sz, f) != sz) { fprintf(stderr, "cofac: read failed\n"); fclose(f); free(buf); return -1; }
    buf[sz] = 0; fclose(f);
    for (size_t k = 0; k < sz; k++) if (buf[k] == '\n') nrec++;
    /* A final record with no trailing newline is still a record. Counting only
     * newlines silently dropped it, which is the kind of off-by-one that shows
     * up as one missing relation and gets blamed on the splitter. */
    if (sz && buf[sz - 1] != '\n') nrec++;
    printf("\n=== cofactorisation: %u candidate records from %s ===\n", nrec, path);
    if (!nrec) { free(buf); return -1; }

    rest = (cf_rest_t *)malloc((size_t)nrec * sizeof(cf_rest_t));
    j0 = (mz<CF_LMAX> *)malloc((size_t)nrec * sizeof(mz<CF_LMAX>));
    j1 = (mz<CF_LMAX> *)malloc((size_t)nrec * sizeof(mz<CF_LMAX>));
    i0 = (uint32_t *)malloc((size_t)nrec * 4);
    i1 = (uint32_t *)malloc((size_t)nrec * 4);
    small0 = (uint64_t *)malloc((size_t)nrec * 8);
    small1 = (uint64_t *)malloc((size_t)nrec * 8);
    side0ok = (uint8_t *)malloc(nrec);
    alg_job = (uint8_t *)calloc(nrec, 1);
    if (!rest || !j0 || !j1 || !i0 || !i1 || !small0 || !small1 || !side0ok || !alg_job) {
        fprintf(stderr, "cofac: out of memory\n"); return -1;
    }

    thost = host_ms();
    {   /* parse; keep the trial-division part of each line in place */
        char *p = buf;
        for (uint32_t r = 0; r < nrec; r++) {
            char *nl = strchr(p, '\n');
            if (!nl) {                       /* unterminated final line */
                if (!*p) { nrec = r; break; }
                nl = p + strlen(p);
            }
            *nl = 0;
            {
                char *c1 = strchr(p, ',');
                char *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
                char *c3 = c2 ? strchr(c2 + 1, ':') : NULL;
                char *c4 = c3 ? strchr(c3 + 1, ':') : NULL;
                mz<CF_LMAX> v0, v1; int ns0 = 0, ns1 = 0;
                if (!c1 || !c2 || !c3 || !c4) { fprintf(stderr, "cofac: line %u malformed\n", r); return -1; }
                *c1 = 0; *c2 = 0; *c3 = 0; *c4 = 0;
                if (!cf_from_dec<CF_LMAX>(&v0, p, &ns0) ||
                    !cf_from_dec<CF_LMAX>(&v1, c1 + 1, &ns1)) {
                    fprintf(stderr, "cofac: line %u cofactor does not fit\n", r); return -1;
                }
                rest[r].ab = c2 + 1; rest[r].f0 = c3 + 1; rest[r].f1 = c4 + 1;
                side0ok[r] = 1;
                /* A side that does NOT need splitting still carries a residual
                 * prime, and it is part of the relation. Dropping it emits a
                 * factorisation whose product is not the norm. */
                /* Two limbs: an unsplit residual within lpb is a prime of
                 * the relation, and at lpb > 32 it does not fit in one. Every
                 * limb above the low two is checked rather than assumed zero
                 * -- they are zero only because the residual is within
                 * lpb <= 64, which is an invariant of the caller and not of
                 * this line. cf_hi_limbs walks all of them, so widening
                 * CF_LMAX cannot leave a limb unexamined here. */
                small0[r] = (ns0 || cf_hi_limbs<CF_LMAX, 2>(&v0)
                             || (v0.v[0] <= 1u && !v0.v[1]))
                    ? 0ull : (((uint64_t)v0.v[1] << 32) | v0.v[0]);
                small1[r] = (ns1 || cf_hi_limbs<CF_LMAX, 2>(&v1)
                             || (v1.v[0] <= 1u && !v1.v[1]))
                    ? 0ull : (((uint64_t)v1.v[1] << 32) | v1.v[0]);
                if (ns0) { j0[n0] = v0; i0[n0] = r; n0++; }
                if (ns1) { j1[n1] = v1; i1[n1] = r; n1++; alg_job[r] = 1; }
            }
            p = nl + 1;
        }
    }
    printf("  rational jobs %u (%.1f%%), algebraic jobs %u (%.1f%%)\n",
           n0, 100.0 * n0 / nrec, n1, 100.0 * n1 / nrec);
    thost = host_ms() - thost;

    st0 = (uint8_t *)malloc(n0 ? n0 : 1); nf0 = (uint8_t *)calloc(n0 ? n0 : 1, 1);
    fac0 = (uint64_t *)malloc((size_t)(n0 ? n0 : 1) * CF_MAXFAC * 8);
    if (meth0 || meth1) {
        ns = cf_ecm_plan(ecm_b1, &h_s);
        if (!ns) { fprintf(stderr, "  cofac: empty ECM plan for B1=%u\n", ecm_b1); return -1; }
        CK(cudaMalloc(&d_s, (size_t)ns * 4));
        CK(cudaMemcpy(d_s, h_s, (size_t)ns * 4, cudaMemcpyHostToDevice));
        free(h_s);
        if (ecm_b2) {
            s2nv = cf_ecm_stage2_plan(ecm_b1, ecm_b2, &s2vmin, &h_s2mask);
            if (!s2nv) {
                fprintf(stderr, "  cofac: empty ECM stage-2 plan for B1=%u"
                        " B2=%u\n", ecm_b1, ecm_b2);
                free(h_s2mask); cudaFree(d_s);
                free(buf); free(rest); free(j0); free(j1); free(i0); free(i1);
                free(small0); free(small1); free(side0ok); free(alg_job);
                free(st0); free(nf0); free(fac0);
                return -1;
            }
            CK(cudaMalloc(&d_s2mask, s2nv));
            CK(cudaMemcpy(d_s2mask, h_s2mask, s2nv, cudaMemcpyHostToDevice));
            free(h_s2mask);
        }
        printf("  ECM: B1 = %u, %u prime powers, B2 = %u (%u giant steps),"
               " %u curves per round, %d rounds\n", ecm_b1, ns, ecm_b2,
               s2nv, ecm_curves, rounds);
    }
    sched.method = meth0; sched.rounds = rounds; sched.budget = budget;
    sched.curves = ecm_curves; sched.d_s = d_s; sched.ns = ns;
    sched.d_s2mask = d_s2mask; sched.s2vmin = s2vmin; sched.s2nv = s2nv;
    /* One launch per round here -- UINT32_MAX, i.e. "larger than any batch",
     * not 0: cf_sched_t's chunk is a concrete count with no sentinel value.
     * This path is a host-driven batch tool, not the BOINC pipeline, so it is
     * not what a volunteer's watchdog sees. */
    sched.chunk = 0xffffffffu;
    if (cf_run_side_dyn(limbs0, j0, n0, lim0, lpb0, st0, fac0, nf0,
                        blocks, threads, 1, &ms0, &sched, "rational")) return -1;
    for (uint32_t k = 0; k < n0; k++) {
        if (st0[k] == CF_OK) continue;
        side0ok[i0[k]] = 0;
        if (st0[k] == CF_DEAD) dead0++; else if (st0[k] == CF_INCOMPLETE) stuck0++;
        else novf++;
    }
    printf("  rational: %u split, %u dead (a prime > 2^%u), %u still stuck\n",
           n0 - dead0 - stuck0 - novf, dead0, lpb0, stuck0);

    /* class-aware: only records whose rational side survived are enqueued */
    {
        uint32_t keep = 0;
        for (uint32_t k = 0; k < n1; k++)
            if (side0ok[i1[k]]) { j1[keep] = j1[k]; i1[keep] = i1[k]; keep++; }
        printf("  algebraic queue after the rational gate: %u of %u (%.1f%% suppressed)\n",
               keep, n1, n1 ? 100.0 * (n1 - keep) / n1 : 0.0);
        n1 = keep;
    }
    st1 = (uint8_t *)malloc(n1 ? n1 : 1); nf1 = (uint8_t *)calloc(n1 ? n1 : 1, 1);
    fac1 = (uint64_t *)malloc((size_t)(n1 ? n1 : 1) * CF_MAXFAC * 8);
    sched.method = meth1;                 /* per side; see cofq_t.meth */
    if (cf_run_side_dyn(limbs1, j1, n1, lim1, lpb1, st1, fac1, nf1,
                        blocks, threads, 1, &ms1, &sched, "algebraic")) return -1;
    for (uint32_t k = 0; k < n1; k++) {
        if (st1[k] == CF_DEAD) dead1++;
        else if (st1[k] == CF_INCOMPLETE) stuck1++;
        else if (st1[k] != CF_OK) novf++;
    }
    printf("  algebraic: %u split, %u dead (a prime > 2^%u), %u still stuck\n",
           n1 - dead1 - stuck1, dead1, lpb1, stuck1);

    if (out) {
        snprintf(tmp, sizeof tmp, "%s.part", out);
        fo = fopen(tmp, "wb");   /* wire format: see run_pipeline */
        if (!fo) { perror(tmp); return -1; }
        setvbuf(fo, NULL, _IOFBF, 1 << 22);
    }
    {
        uint64_t *ex0 = (uint64_t *)calloc(nrec, CF_MAXFAC * 8);
        uint8_t  *ne0 = (uint8_t *)calloc(nrec, 1);
        uint8_t  *done1 = (uint8_t *)calloc(nrec, 1);
        for (uint32_t r = 0; r < nrec; r++)
            if (small0[r]) { ex0[(size_t)r * CF_MAXFAC] = small0[r]; ne0[r] = 1; }
        for (uint32_t k = 0; k < n0; k++)
            if (st0[k] == CF_OK) {
                ne0[i0[k]] = nf0[k];
                for (int z = 0; z < nf0[k]; z++)
                    ex0[(size_t)i0[k] * CF_MAXFAC + z] = fac0[(size_t)k * CF_MAXFAC + z];
            }
        for (uint32_t k = 0; k < n1; k++) {
            uint32_t r = i1[k];
            if (st1[k] != CF_OK || !side0ok[r]) continue;
            done1[r] = 1;
            if (!cf_rec_build(&rb, rest[r].f0, ex0 + (size_t)r * CF_MAXFAC, ne0[r],
                              rest[r].f1, fac1 + (size_t)k * CF_MAXFAC, nf1[k])) {
                nover++;
                continue;          /* not counted in nrel: it is not one */
            }
            nrel++;
            if (fo) {
                fprintf(fo, "%s:", rest[r].ab);
                cf_emit_sorted(fo, rb.f0, rb.n0);
                fputc(':', fo);
                cf_emit_sorted(fo, rb.f1, rb.n1);
                fputc('\n', fo);
            }
        }
        /* The records whose algebraic side needed no splitting: they are
         * relations as soon as the rational side splits, and they have no
         * algebraic job to come back from. `small1` is 0 both when the side
         * was queued AND when its cofactor was already 1, so the queued case
         * has to be excluded by `done1`/membership, not by `small1` -- testing
         * `small1` alone silently dropped every record whose algebraic
         * cofactor trial division had already reduced to 1. */
        for (uint32_t r = 0; r < nrec; r++) {
            uint64_t one1 = small1[r];
            if (done1[r] || alg_job[r] || !side0ok[r]) continue;
            if (!cf_rec_build(&rb, rest[r].f0, ex0 + (size_t)r * CF_MAXFAC, ne0[r],
                              rest[r].f1, &one1, one1 ? 1 : 0)) {
                nover++;
                continue;
            }
            nrel2++;
            if (fo) {
                fprintf(fo, "%s:", rest[r].ab);
                cf_emit_sorted(fo, rb.f0, rb.n0);
                fputc(':', fo);
                cf_emit_sorted(fo, rb.f1, rb.n1);
                fputc('\n', fo);
            }
        }
        free(ex0); free(ne0); free(done1);
        printf("  relations: %u from the algebraic queue + %u whose algebraic"
               " side needed none\n", nrel, nrel2);
        /* Loud, because the alternative this replaces was a wrong relation
         * written silently. A nonzero count means the --candidates file holds
         * records with more trial-division factors than TD_FMAX. */
        if (nover)
            fprintf(stderr, "  warning: %u record(s) had more factors than"
                    " TD_FMAX+CF_MAXFAC and were SKIPPED rather than emitted"
                    " with factors dropped\n", nover);
        nrel += nrel2;
    }
    if (fo) {
        int bad = ferror(fo);
        if (fclose(fo) || bad) { fprintf(stderr, "cofac: write failed\n"); remove(tmp); return -1; }
        if (bench_atomic_replace(tmp, out)) { perror("rename"); remove(tmp); return -1; }
    }
    if (novf) fprintf(stderr, "  ** %u records produced more than %d factors\n", novf, CF_MAXFAC);

    printf("\n  --- cofactorisation ---\n");
    printf("  %-34s %8u\n", "candidate records", nrec);
    printf("  %-34s %8.1f ms\n", "rational queue", ms0);
    printf("  %-34s %8.1f ms\n", "algebraic queue", ms1);
    printf("  %-34s %8.1f ms\n", "host parse", thost);
    printf("  %-34s %8u   (%.2f%% of candidates)\n", "RELATIONS", nrel,
           100.0 * nrel / nrec);
    printf("  %-34s %8u\n", "still stuck (requeue, not dead)", stuck0 + stuck1);

    free(buf); free(rest); free(j0); free(j1); free(i0); free(i1);
    free(small0); free(small1); free(side0ok); free(alg_job);
    free(st0); free(st1); free(nf0); free(nf1); free(fac0); free(fac1);
    cudaFree(d_s); cudaFree(d_s2mask);
    /* Same discipline as novf: a batch that silently dropped records is not a
     * clean batch, and the caller must not read it as one. */
    return (novf || nover) ? -1 : 0;
}

#endif  /* __CUDACC__ */
#endif  /* CUDA_SIEVE_COFAC_CUH */
