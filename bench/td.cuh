/* Trial division: exact norms, large primes from the resieve, small primes by
 * direct test.
 *
 * The split is the one prototype.md argues for, and it inverts the sieve's own
 * reasoning:
 *
 *   p >= bkthresh (bucketed) -- RESIEVE. Hits are rare and survivors sparse,
 *       so re-walking the factor base and filtering against the survivor
 *       bitmap is cheap. Layout A, measured at 8.55 ms/q.
 *   p <  bkthresh (line-sieved) -- DIRECT TEST per survivor. The small sieve
 *       carries 6.84e9 updates per q; resieving that is ~40 ms, worse than
 *       everything else in the post-sieve budget combined. Testing instead
 *       costs one congruence per (survivor, entry).
 *
 * The direct test is the hot loop -- ~3,500 entries against ~841K survivors is
 * 3e9 tests -- so it is written to avoid integer division entirely. See
 * td_magic_build: each entry carries a multiply-shift reciprocal valid over
 * the range this kernel actually produces, which turns the modulo into
 * __umulhi + shift + mad.
 */
#ifndef CUDA_SIEVE_TD_CUH
#define CUDA_SIEVE_TD_CUH

#include "bigint.cuh"
#include "prp.cuh"
#include "slab.h"

/* ---- exact homogeneous form ------------------------------------------- */

/* F(a,b) = sum_k c[k] a^k b^(deg-k), magnitude and sign kept apart because
 * bn_t is a magnitude type. The RATIONAL side is just deg == 1 with
 * c[1] = Y1, c[0] = Y0 -- G(a,b) = Y1*a + Y0*b is the same homogeneous form,
 * so one kernel covers both sides. */
typedef struct {
    bn_t    c[BENCH_NCOEFF];
    int32_t sign[BENCH_NCOEFF];
    int32_t deg;
} tdpoly_t;

/* ---- small-prime direct-test entry ------------------------------------- */

/* The predicate is the one verify_cpu.c calls hits_pred: solutions exist only
 * on rows j == 0 (mod g), and there i == rt*(j/g) (mod m), with m = p/g.
 *
 * Rewritten to keep the reduced value NONNEGATIVE so a single unsigned
 * multiply-shift reciprocal suffices:
 *
 *     w = rt*j' + (Ihalf - i)          in [1, 2^31), since i < Ihalf
 *     hit  <=>  w mod m == Ihalf mod m
 *
 * `cst` holds Ihalf mod m. m == 1 ("every i, on every g-th row") has no valid
 * 32-bit magic and is flagged by magic == 0; it is rare and the branch is
 * uniform across the warp, since every thread walks the same entry list. */
typedef struct {
    uint64_t recip;    /* Barrett reciprocal of p = m*g, for the division   */
    uint32_t m;        /* reduced modulus p/g                        */
    uint32_t rt;       /* transformed root mod m                     */
    uint32_t g;        /* row divisor; 1 in the ordinary affine case */
    uint32_t cst;      /* Ihalf mod m                                */
    uint32_t magic;    /* multiply-shift reciprocal of m; 0 if m == 1 */
    uint32_t sh;
} tdsmall_t;

/* Multiply-shift reciprocal, exact for dividends below 2^31. The slab
 * planner enforces this bound for the local j range used by the hot TD loop.
 *
 *   m a power of two 2^k, k >= 1 : magic = 2^(32-k), sh = 0
 *   otherwise                    : sh = floor(log2 m), magic = 2^(32+sh)/m + 1
 *
 * The general case needs magic < 2^32 and (magic*m - 2^(32+sh)) * 2^31 <
 * 2^(32+sh). Both follow from 2^sh < m < 2^(sh+1): the first because
 * 2^(32+sh)/m < 2^32. For w < 2^31, the reciprocal overestimate contributes
 * less than 1/2^(sh+1), while the next quotient boundary is at least 1/m away
 * and m < 2^(sh+1), so the truncated quotient is exact. */
static inline void td_magic_build(uint32_t m, uint32_t *magic, uint32_t *sh)
{
    if (m == 1) { *magic = 0; *sh = 0; return; }
    if ((m & (m - 1)) == 0) {                 /* power of two */
        int k = 0; while ((1u << k) != m) k++;
        *magic = 1u << (32 - k);
        *sh = 0;
        return;
    }
    {
        int s = 0;
        while ((1u << (s + 1)) <= m) s++;      /* s = floor(log2 m) */
        *magic = (uint32_t)(((uint64_t)1 << (32 + s)) / m + 1);
        *sh = (uint32_t)s;
    }
}

/* One implementation for both the CPU arithmetic gate and the device hot
 * loop. The host half spells out umulhi with a 64-bit product; the CUDA device
 * half uses the single __umulhi instruction. Keeping the quotient/remainder
 * expression here prevents the regression test from validating a copied
 * formula instead of the code the kernel actually executes. */
#ifdef __CUDACC__
#define TD_MOD_HD static __host__ __device__ __forceinline__
#else
#define TD_MOD_HD static inline
#endif
TD_MOD_HD uint32_t td_mod_magic(uint32_t w, uint32_t m,
                                uint32_t magic, uint32_t sh)
{
    uint32_t q;
#ifdef __CUDA_ARCH__
    q = __umulhi(w, magic) >> sh;
#else
    q = (uint32_t)(((uint64_t)w * magic) >> 32) >> sh;
#endif
    return w - q * m;
}
#undef TD_MOD_HD

/* ---- ss_first's reciprocals (small-prime line sieve) -------------------- *
 *
 * Shared rather than forked: this is pure host arithmetic with no platform
 * API in it, and it calls td_magic_build directly above. It lives here so
 * bench_kernels.cu and bench_kernels.hip use ONE copy, and so pipeline.cuh /
 * pipeline_hip.cuh can call it without depending on their includer having
 * defined it first.
 *
 * SS_KSHIFT is the single source of truth for the bias shift. ss_first forms
 * `rt*j + (p << SS_KSHIFT(logI)) - ilo` on the device and ss_magic_build
 * proves that expression stays below 2^31 on the host; if the two ever
 * disagree the residues are silently wrong, so both sides derive it here
 * rather than open-coding `logI - 2`.
 *
 * The bias must be (a) a multiple of m, so it does not move the residue, and
 * (b) >= ihalf == 1<<(logI-1), so the numerator cannot go negative. `m << s`
 * satisfies (a) for ANY s, and logI-2 is the smallest shift that satisfies
 * (b) for the WORST case m == 2.
 *
 * A PER-M SHIFT WAS TRIED AND REJECTED, on measurement. `(logI-1) -
 * floor(log2 m)` is the smallest shift that works for each individual m, and
 * since the bias is part of the very numerator this function has to prove
 * below 2^31, shrinking it buys magic coverage: over the primes below
 * bkthresh at the default geometry, the flat shift refuses 0% at logI 15 but
 * 30.4% at logI 16 and 80.0% at logI 17, while the per-m shift refuses none
 * at logI 16. floor(log2 m) is even free on the device -- ss_first computes
 * `31 - __clz(p)` anyway as td_mod_magic's shift.
 *
 * It still lost. The refused entries are the LARGEST small primes, which hit
 * rarest (~1.4% of updates at logI 16 by Mertens, against 30% of table
 * entries), so the coverage recovered is worth little -- while the two extra
 * ALU ops to derive the shift run on EVERY small-prime hit. Measured on an
 * RTX 5070, apply stage, oracle/c183, n=4 interleaved, arms non-overlapping
 * in both directions:
 *
 *   logI 15   flat 28.09 ms   per-m 28.52 ms    per-m is 1.5% SLOWER
 *   logI 16   flat 117.65 ms  per-m 116.96 ms   per-m is 0.6% faster
 *
 * logI 14/15 is the operating point, so the flat shift wins where it counts.
 * Having both would need either a per-hit select (which costs about what it
 * saves) or sieve_small's tier loops duplicated under a kernel-uniform
 * branch, which is real code bloat in a kernel whose occupancy is already
 * smem-bound. If logI 16 ever becomes the default geometry, revisit: the
 * change is confined to this macro, ss_magic_build's kshift, and ss_first.
 *
 * REVISITED when logI 16 became the NFS@Home geometry, and answered without
 * touching the shift: the refused entries are a contiguous tail of the sorted
 * table, which the thread tier reduces in two steps instead (ss_bound_ok,
 * finding 111). The bloat argument above did land, in a different form: the
 * tail is a k_apply template parameter (SSTAIL), so each pipeline
 * specialisation ships twice -- 11 k_apply instances instead of 9 in every
 * arch of a fat build -- and the source writes the magic path twice
 * (ss_first's one- and two-step forms). That was the price of keeping one
 * thread-tier loop per instance: two loops in one kernel lost reconvergence
 * and cost apply +24% at I16. */
#define SS_KSHIFT(logI)  ((uint32_t)((logI) - 2))

/* Returns 0 into *magic to mean "no magic, use ss_first's 64-bit fallback" --
 * for m == 1, and, more importantly, whenever the numerator cannot be PROVEN
 * below 2^31, which is the exactness bound on td_mod_magic above.
 *
 * TWO FORMS, one proof each (ss_bound_ok). The ONE-STEP form reduces
 * `rt*j + bias - ilo` in one go. The TWO-STEP form (two_step != 0, for the
 * thread-tier entries at or past ss_two_step_from) reduces `rt*j` first and
 * then `that residue + bias - ilo`, so each dividend only has to be below
 * 2^31 on its own. At I16/J32768 the one-step proof fails for every m above
 * ~43,690, about 30% of the thread tier, which then took the 64-bit fallback
 * in every region (findings 110-111); the two-step proof holds for all of them.
 * The device picks the form by the entry's INDEX, and the magic built here
 * is proved for the form that index gets.
 *
 * THE COUPLING IS ONE-WAY. A one-step proof implies the two-step one, so an
 * entry the device runs two-step when the host built it one-step is still
 * exact. The converse is NOT: a magic proved only two-step, run through the
 * one-step reduction, can give a wrong residue silently (I16/J32768,
 * m = 60000: rt*j alone reaches ~1.97e9, and the bias takes the numerator
 * past 2^31). So the one-step-only k_apply instance (SSTAIL = false) must
 * never be launched on a table with a tail. Every launch site decides with
 * ss_needs_tail, and that instance traps if it sees one.
 *
 * Takes logI rather than a precomputed shift ON PURPOSE: the shift is per-m
 * now, and a caller that derived it even slightly differently from the device
 * would produce silently wrong residues. There is nothing here for a caller
 * to get wrong.
 *
 * jmax is the largest j the DEVICE will pass for this entry, which is not
 * always cfg->J: an entry with a row divisor g > 1 hits only on rows g | j
 * and is handed j/g, so its jmax is J/g. Passing the smaller bound is not an
 * optimisation of the proof, it is the honest value, and it widens coverage.
 *
 * Refusing rather than assuming matters: --bkthresh and --J are both operator
 * knobs, so m*jmax is not bounded by anything this header controls, and
 * neither is the bias. A silent wrong residue here would corrupt relations,
 * not crash. The bound below is computed in 64-bit against the exact
 * expression ss_first evaluates, so it stays honest if either knob is pushed.
 *
 * Powers of two keep a non-zero magic even though ss_first ignores it for
 * them and uses an AND instead: it is the "take the fast path" flag, and they
 * need no bound for it (ss_bound_ok). m == 1 is refused here and answered by
 * the fallback (t % 1 == 0). */
static inline int ss_bound_ok(uint32_t m, uint32_t jmax, uint32_t logI,
                              int two_step);

/* The one-step numerator bound alone, (m-1)*jmax + (m << kshift) + ihalf <
 * 2^31: monotone in m, so the split search and the device gate route a
 * modulus with exactly the predicate ss_bound_ok proves. Caller guarantees
 * logI in [2,20] and m >= 1. */
static inline int ss_one_step_fits(uint64_t m, uint32_t jmax, uint32_t logI)
{
    return (m - 1) * jmax + (m << SS_KSHIFT(logI)) + (1u << (logI - 1))
        < (1ull << 31);
}

static inline void ss_magic_build(uint32_t m, uint32_t jmax, uint32_t logI,
                                  int two_step, uint32_t *magic)
{
    uint32_t mg, sh;
    /* sh is filled by td_magic_build and deliberately DISCARDED: ss_first
     * recomputes the same value on device as 31 - __clz(p), which it needs
     * anyway. That is only correct because td_magic_build's general case sets
     * sh == floor(log2 m), and its power-of-two case (sh == 0) is diverted to
     * ss_first's AND branch and never reaches td_mod_magic. Assert the first
     * half here so a change to that convention cannot silently desynchronise
     * the device. */

    *magic = 0;
    /* The header above claims there is nothing here for a caller to get
     * wrong, and logI is the one input that could still do it: logI < 2 makes
     * both `1u << (logI - 1)` and SS_KSHIFT's `logI - 2` enormous shift
     * counts, which is undefined here AND in the `p << kshift` ss_first would
     * then evaluate. bench_main.cu pins logI to [2,20] and is the ONLY place
     * that does -- run_pipeline validates the slab plan but not logI -- so
     * this refusal is the whole guard, not a second one. Refusing costs only
     * the 64-bit fallback.
     *
     * It protects the MAGIC path only. ss_first's fallback hardcodes its bias
     * as `p << 20`, which needs p<<20 >= 1<<(logI-1), i.e. logI <= 21 for the
     * worst case p == 2; beyond that the fallback's numerator goes negative
     * and wraps. Nothing reaches that today because main() caps logI at 20,
     * but a second entry point that did not would get silently wrong offsets
     * from the fallback, not from here. (ss_bound_ok holds both guards.)
     *
     * THE BOUND IS TESTED BEFORE td_magic_build, not after, because it does
     * not need the reciprocal: it depends only on m, jmax and logI. Building
     * first meant every refused entry still paid td_magic_build's 64-bit
     * divide to have its answer thrown away. */
    if (!ss_bound_ok(m, jmax, logI, two_step)) return;

    td_magic_build(m, &mg, &sh);
    if (!mg) return;
    if ((m & (m - 1)) != 0) {           /* general case: device uses 31-clz */
        uint32_t lg = 0;
        while ((1u << (lg + 1)) <= m) lg++;
        if (sh != lg) return;           /* convention changed: refuse, fall back */
    }
    *magic = mg;
}

/* The exactness proof for one form of ss_first, against the exact expression
 * the device evaluates: rt <= m-1, j <= jmax, and the -ilo term contributes at
 * most +ihalf. Computed in 64-bit, so it stays honest whatever --bkthresh and
 * --J are set to.
 *
 *   one-step   rt*j + bias - ilo                     < 2^31
 *   two-step   rt*j                                  < 2^31, and
 *              (rt*j mod m) + bias - ilo  <=  m-1 + bias + ihalf  < 2^31
 *
 * One-step implies two-step whenever jmax >= 1, so moving such an entry from
 * the first form to the second can never refuse it. (An entry with a row
 * divisor g > J has jmax = J/g = 0 and can pass one and fail the other; that
 * costs it the fallback, nothing else, and only a raised --bkthresh makes one.)
 *
 * A POWER OF TWO NEEDS NO BOUND in either form. ss_first answers it with an
 * AND on the 32-bit sum, and p divides 2^32, so the residue is exact even if
 * the sum wraps -- p < 2^30 because bkthresh is capped there. */
static inline int ss_bound_ok(uint32_t m, uint32_t jmax, uint32_t logI,
                              int two_step)
{
    uint32_t kshift, ihalf;
    uint64_t bias;

    if (logI < 2 || logI > 20) return 0;
    if (m < 2) return 0;                   /* every position hits; fallback */
    ihalf  = 1u << (logI - 1);
    kshift = SS_KSHIFT(logI);
    bias = (uint64_t)m << kshift;          /* a multiple of m, and >= ihalf */
    /* Guaranteed for m >= 2 (2 << (logI-2) == 1 << (logI-1)), and checked
     * anyway: it is the property the whole unsigned-numerator argument rests
     * on, and the m >= 2 guard above is what makes it true. */
    if (bias < ihalf) return 0;
    if ((m & (m - 1)) == 0) return 1;      /* the AND path: exact regardless */
    if (!two_step)
        return ss_one_step_fits(m, jmax, logI);
    return (uint64_t)(m - 1) * jmax < (1ull << 31)
        && (uint64_t)(m - 1) + bias + ihalf < (1ull << 31);
}

/* First index of the thread tier's TWO-STEP range: the first entry at or past
 * nwrp whose one-step numerator bound fails at the largest row bound, J.
 * That bound, (m-1)*J + (m << kshift) + ihalf < 2^31, grows with m and hsp is
 * sorted by modulus, so it is a binary search, not a scan of every entry per
 * q per side. Every entry before the split passes ss_bound_ok one-step at any
 * jmax <= J (a power of two passes regardless), and the range from it is the
 * tail the one-step form refuses. Equal to n (no tail) at logI <= 15 and the
 * default J. An invalid logI puts everything in the tail, where ss_bound_ok
 * refuses it all: the fallback, as before. */
static inline uint32_t ss_two_step_from(const uint32_t *hsp, uint32_t n,
                                        uint32_t nwrp, uint32_t J,
                                        uint32_t logI)
{
    uint32_t lo = nwrp, hi = n;
    if (logI < 2 || logI > 20) return nwrp;
    while (lo < hi) {                      /* first index failing the bound */
        const uint32_t mid = lo + (hi - lo) / 2;
        if (ss_one_step_fits(hsp[mid], J, logI)) lo = mid + 1;
        else                                     hi = mid;
    }
    return lo;
}

/* Which k_apply instance a launch needs: the one with the two-step tail
 * whenever this table has entries at or past nsplit. The one-step-only
 * instance on such a table would run two-step-proved magics through the
 * one-step reduction -- silently wrong residues (see ss_magic_build). Every
 * launch site decides with this, so the predicate exists once. */
static inline int ss_needs_tail(uint32_t nsplit, uint32_t nsmall)
{
    return nsplit < nsmall;
}

#if defined(__CUDACC__)

/* Move the direct-test congruence origin forward by delta_j global rows. This
 * is deliberately a separate tiny kernel: it performs one 64-bit modulo per
 * small-prime ENTRY per slab so the billions-of-tests k_td loop stays on its
 * original 32-bit multiply/high-multiply sequence. */
__global__ void k_tdsmall_advance(tdsmall_t *__restrict sm, uint32_t n,
                                  uint32_t delta_j)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        tdsmall_t v = sm[t];
        if (v.magic) v.cst = slab_phase_cst(v.cst, v.rt, v.m, delta_j);
        sm[t] = v;
    }
}

/* Divide p out of N to full multiplicity.
 *
 * One pass produces the quotient AND the remainder, so a factor costs two
 * passes -- one that succeeds, one that discovers it is done -- rather than
 * the three a separate mod-then-divide would take. The pass walks only the
 * limbs still in use, which matters: the norm loses ~110 bits over a
 * factorisation and the tail limbs go to zero early.
 *
 * `fac`/`nf` optionally record the factor once per division, so multiplicity
 * comes out as repetition -- the form a relation record wants. Recording is a
 * template parameter rather than a null check so the measured path carries no
 * stores at all. */
template <int RECORD>
__device__ __forceinline__ void td_divide_out(bn_t *N, uint32_t p, uint64_t recip,
                                              uint32_t *fac, uint32_t *nf,
                                              uint32_t fmax)
{
    for (;;) {
        bn_t q = *N;
        int top = bn_top(&q);
        if (top < 0) return;
        if (bn_divmod_u32_pre(&q, p, recip, top)) return;
        *N = q;
        if (RECORD && *nf < fmax) fac[(*nf)++] = p;
        else if (RECORD) (*nf)++;          /* count past the cap so it shows */
    }
}

/* Per-q candidate bookkeeping, on device.
 *
 * Under inline cofactorisation the host used to read back every candidate's
 * coordinates, cofactors, factor counts and two 64-entry factor arrays -- about
 * 618 bytes each -- purely to count them and check for overflow, because the
 * queue had already taken everything it needed straight from device memory.
 * Three counters replace all of it.
 *
 * out[0] = records that are already relations (both sides within lpb)
 * out[1] = records still needing cofactorisation
 * out[2] = records whose factor list overflowed TD_FMAX -- these must FAIL the
 *          run, since the enqueue clamps the count and the missing factors
 *          would leave the norm undivided with nothing downstream to notice.
 */
__global__ void k_cand_stats(uint32_t n,
                             const uint8_t *__restrict bits0,
                             const uint8_t *__restrict bits1,
                             const uint32_t *__restrict fn0,
                             const uint32_t *__restrict fn1,
                             uint32_t lpb0, uint32_t lpb1, uint32_t fmax,
                             uint32_t *__restrict out,
                             const uint32_t *__restrict dn = nullptr)
{
    const uint64_t stride = bench_grid_stride_x();
    if (dn) n = min(n, *dn);
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (fn0[t] > fmax || fn1[t] > fmax) { atomicAdd(&out[2], 1u); continue; }
        if ((uint32_t)bits0[t] <= lpb0 && (uint32_t)bits1[t] <= lpb1)
            atomicAdd(&out[0], 1u);
        else
            atomicAdd(&out[1], 1u);
    }
}

/* ---- survivor rank ----------------------------------------------------- */

/* The compacted survivor list is written in rank order over the two-sided
 * bitmap, so a position's index in that list is a function of the bitmap
 * alone. That is what lets the resieve scatter a recovered prime straight
 * into its survivor's slot without sorting 1.9M (x,p) pairs or building a
 * 2 GB position-indexed map.
 *
 * gbase[] holds the exclusive prefix over groups of TD_GROUP_W words, so a
 * lookup reads at most TD_GROUP_W-1 extra words. 8 words = 256 positions
 * keeps that bounded while holding the table to 8 MB on the full area. */
#define TD_GROUP_W 8
#define TD_GROUP_X (TD_GROUP_W * 32)

__device__ __forceinline__ uint32_t td_rank(const uint32_t *__restrict bits,
                                            const uint32_t *__restrict gbase,
                                            uint32_t x)
{
    const uint32_t wx = x >> 5;
    const uint32_t w0 = (x / TD_GROUP_X) * TD_GROUP_W;
    uint32_t r = gbase[x / TD_GROUP_X];
    for (uint32_t w = w0; w < wx; w++) r += __popc(bits[w]);
    return r + __popc(bits[wx] & ((1u << (x & 31u)) - 1u));
}

__global__ void k_group_counts(const uint32_t *__restrict bits,
                               uint32_t ngroup, uint32_t *__restrict cnt)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t gg = bench_grid_thread_x(); gg < ngroup; gg += stride) {
        const uint32_t g = (uint32_t)gg;
        uint32_t s = 0;
        for (uint32_t k = 0; k < TD_GROUP_W; k++) s += __popc(bits[g * TD_GROUP_W + k]);
        cnt[g] = s;
    }
}

/* ---- exclusive scan ---------------------------------------------------- */
/* Hand-written rather than CUB: this project builds with nothing but nvcc and
 * libm, and the scan is 40 lines. Three passes -- block-local scan, scan of
 * the block totals, then add the offsets back. */

#define TD_SCAN_BLK 256

__global__ void k_scan_pass1(const uint32_t *__restrict in, uint32_t n,
                             uint32_t *__restrict out, uint32_t *__restrict bsum)
{
    __shared__ uint32_t s[TD_SCAN_BLK];
    const uint64_t ii = (uint64_t)blockIdx.x * TD_SCAN_BLK + threadIdx.x;
    const uint32_t i = (uint32_t)ii;
    s[threadIdx.x] = (ii < n) ? in[i] : 0u;
    __syncthreads();
    for (uint32_t off = 1; off < TD_SCAN_BLK; off <<= 1) {
        uint32_t v = (threadIdx.x >= off) ? s[threadIdx.x - off] : 0u;
        __syncthreads();
        s[threadIdx.x] += v;
        __syncthreads();
    }
    if (ii < n) out[i] = s[threadIdx.x] - in[i];   /* exclusive */
    if (threadIdx.x == TD_SCAN_BLK - 1) bsum[blockIdx.x] = s[threadIdx.x];
}

/* One block scans the block totals in place. nb is at most a few thousand. */
__global__ void k_scan_pass2(uint32_t *__restrict bsum, uint32_t nb)
{
    __shared__ uint32_t s[1024];
    __shared__ uint32_t running;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();
    for (uint32_t base = 0; base < nb; base += 1024) {
        uint32_t i = base + threadIdx.x;
        uint32_t v = (i < nb) ? bsum[i] : 0u;
        s[threadIdx.x] = v;
        __syncthreads();
        for (uint32_t off = 1; off < 1024; off <<= 1) {
            uint32_t t = (threadIdx.x >= off) ? s[threadIdx.x - off] : 0u;
            __syncthreads();
            s[threadIdx.x] += t;
            __syncthreads();
        }
        if (i < nb) bsum[i] = running + s[threadIdx.x] - v;         /* exclusive */
        __syncthreads();
        if (threadIdx.x == 1023) running += s[1023];
        __syncthreads();
    }
}

__global__ void k_scan_pass3(uint32_t *__restrict out, uint32_t n,
                             const uint32_t *__restrict bsum)
{
    const uint64_t ii = (uint64_t)blockIdx.x * TD_SCAN_BLK + threadIdx.x;
    if (ii < n) out[(uint32_t)ii] += bsum[blockIdx.x];
}

/* ---- rank-ordered compaction ------------------------------------------- */

/* Same output as k_intersect_compact's atomic path, but at a deterministic
 * index: survivor s is the s'th set bit of the two-sided bitmap. Two things
 * come from that. The resieve can find a position's slot with td_rank instead
 * of a sort, and the emitted list is reproducible run to run, which is what
 * makes a bit-identical regression test on it mean anything. */
template <bool SLABBED = false>
__global__ void k_emit_ranked(const uint32_t *__restrict bits,
                              const uint32_t *__restrict gbase,
                              uint32_t nword, uint32_t logI,
                              int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                              uint32_t *__restrict out_x,
                              int64_t *__restrict out_a,
                              int64_t *__restrict out_b,
                              uint32_t cap, uint32_t j_base)
{
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint64_t stride = bench_grid_stride_x();

    for (uint64_t ww = bench_grid_thread_x(); ww < nword; ww += stride) {
        const uint32_t w = (uint32_t)ww;
        uint32_t m = bits[w];
        if (!m) continue;
        /* rank of this word's first position: the group base plus the words
         * of the group that precede it. */
        uint32_t slot = gbase[w / TD_GROUP_W];
        for (uint32_t k = (w / TD_GROUP_W) * TD_GROUP_W; k < w; k++)
            slot += __popc(bits[k]);
        while (m) {
            uint32_t k = __ffs(m) - 1;
            m &= m - 1;
            uint32_t x = (w << 5) + k;
            int32_t  i = (int32_t)(x & Imask) - Ihalf;
            const uint32_t jlocal = x >> logI;
            uint32_t j = jlocal;
            if constexpr (SLABBED) j += j_base;
            if (slot < cap) {
                out_x[slot] = x;
                out_a[slot] = (int64_t)i * a0 + (int64_t)j * b0;
                out_b[slot] = (int64_t)i * a1 + (int64_t)j * b1;
            }
            slot++;
        }
    }
}

/* ---- resieve that scatters into per-survivor lists --------------------- *
 *
 * k_resieve_rewalk with the output changed from an unordered (x,p) array to a
 * per-survivor list. PROPER PRIME POWERS ARE SKIPPED: if p^2 hits a position
 * then p hits it too and p is in the same factor base, so the base prime is
 * always recorded and multiplicity is recovered by repeated division. A p^2
 * entry recorded as if it were prime would be divided out as p^2, which is
 * wrong whenever the true multiplicity is odd.
 *
 * UNROLL is the whole performance story of this kernel, so it is worth stating
 * what it is fixing.
 *
 * The walk is neither arithmetic-bound nor bandwidth-bound. It runs 315.5M
 * steps in 6.69 ms, which is 325 lane-cycles for a step of about eight
 * instructions, at full occupancy (40 registers, 48 warps/SM, no spills). What
 * costs is the summary probe: one scattered dependent load per step, and with
 * one load in flight per warp, 48 warps cannot cover L2 latency.
 *
 * A COARSE PRE-FILTER ABOVE THE SUMMARY WAS TRIED AND LOST at every
 * granularity from 256 to 4096 positions per bit -- 6.69 ms became 6.75 to
 * 7.82 -- even though at 256 it rejected 96.5% of steps before they reached
 * the summary. That is the measurement that identifies the bottleneck:
 * removing 96.5% of the probes did not help, so the probes were never the
 * throughput cost. Rejecting a step still costs one dependent load, and the
 * steps that pass now pay two in series.
 *
 * The fix for latency is more loads in flight, not fewer loads. Walking UNROLL
 * positions ahead and issuing all UNROLL summary probes before branching on
 * any of them multiplies memory-level parallelism per warp by UNROLL. The walk
 * itself is pure register arithmetic, so running ahead costs nothing and needs
 * no speculation.
 *
 * AT I16 THE BOUND IS DIFFERENT, and COARSE is for it. Measured 2026-10-16
 * (ncu, NFS@Home lpba-34 job, 2^29-position slabs): L2 throughput 78-89% of
 * peak with issue active 21-26%, ~0.83 L2 sectors per step and an L1 hit rate
 * of 23% -- the 1 MB summary of a slab no longer sits close, so every probe is
 * a 32-byte L2 sector for one bit. COARSE puts a bitmap of the summary's
 * nonzero words (one bit per 2048 positions, 32 KB per such slab) in shared
 * memory and loads the summary only where that bit is set. Unlike the global
 * pre-filter above, a rejecting probe here costs no L2 traffic at all. The
 * set of (survivor, prime) pairs is unchanged: a coarse bit is clear only
 * when every summary bit under it is.
 */
template <int UNROLL, bool SLABBED = false, bool COARSE = false>
__global__ void k_resieve_scatter(const plat_t *__restrict plat,
                                  const uint32_t *__restrict primes,
                                  const uint8_t *__restrict ispow,
                                  uint32_t n, uint32_t xmax, int logI,
                                  const uint32_t *__restrict summary,
                                  const uint32_t *__restrict bits,
                                  const uint32_t *__restrict gbase,
                                  uint32_t *__restrict plist,
                                  uint32_t *__restrict pcnt,
                                  uint32_t K,
                                  unsigned long long *__restrict noverflow,
                                  int log_gran,
                                  const uint64_t *__restrict walk_cur,
                                  const uint32_t *__restrict coarse = nullptr,
                                  uint32_t ncoarse = 0, int cshift = 0)
{
    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t NONE = 0xFFFFFFFFu;      /* local x < 2^31, cannot collide */
    const uint64_t stride = bench_grid_stride_x();
    unsigned long long ovf = 0;
    /* COARSE: one bit per 2^cshift summary words, set when any is nonzero
     * (k_build_summary_g over the summary, wper = 2^cshift), copied to
     * shared memory once per block. See the note above. */
    extern __shared__ uint32_t s_coarse[];
    if constexpr (COARSE) {
        for (uint32_t w = threadIdx.x; w < ncoarse; w += blockDim.x)
            s_coarse[w] = coarse[w];
        __syncthreads();
    }

    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) continue;
        if (ispow && ispow[k]) continue;
        uint32_t p = primes[k];

        /* Resieve never carries walk state forward, so the only slab
         * difference is WHERE the walk starts -- one loop serves both. The
         * walk itself is 64-bit in both, for the reason measured in
         * k_fill_atomic: pl_next's saturating add is a branch per increment
         * and pl_next64's is not. The run-ahead buffer stays uint32 because
         * every in-slab position is below 2^31. */
        uint64_t x;
        if constexpr (SLABBED) x = walk_cur[k];   /* may already be past xmax */
        else                   x = pl_first64(&P, logI);
        while (x < xmax) {
            uint32_t xs[UNROLL], hit[UNROLL];

            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                if (x < xmax) { xs[u] = (uint32_t)x; x = pl_next64(x, &P, Imask); }
                else xs[u] = NONE;
            }
            if constexpr (COARSE) {
                /* shared memory first; only a step whose coarse bit is set
                 * loads its summary word. The loads stay independent, so
                 * they still issue together. */
                uint32_t pass[UNROLL];
                #pragma unroll
                for (int u = 0; u < UNROLL; u++) {
                    const uint32_t c = (xs[u] >> (log_gran + 5)) >> cshift;
                    pass[u] = (xs[u] == NONE) ? 0u
                                              : ((s_coarse[c >> 5] >> (c & 31u)) & 1u);
                }
                #pragma unroll
                for (int u = 0; u < UNROLL; u++) {
                    uint32_t sb = xs[u] >> log_gran;
                    hit[u] = pass[u] ? ((summary[sb >> 5] >> (sb & 31u)) & 1u) : 0u;
                }
            } else {
                /* every one of these is independent, so they issue together */
                #pragma unroll
                for (int u = 0; u < UNROLL; u++) {
                    uint32_t sb = xs[u] >> log_gran;
                    hit[u] = (xs[u] == NONE) ? 0u
                                             : ((summary[sb >> 5] >> (sb & 31u)) & 1u);
                }
            }
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                if (!hit[u]) continue;
                uint32_t xv = xs[u];
                if (!((bits[xv >> 5] >> (xv & 31u)) & 1u)) continue;
                uint32_t idx = td_rank(bits, gbase, xv);
                uint32_t slot = atomicAdd(&pcnt[idx], 1u);
                if (slot < K) plist[(size_t)idx * K + slot] = p;
                else ovf++;
            }
        }
    }
    if (ovf) atomicAdd(noverflow, ovf);
}

/* ---- --resieve-rows: the row tier of k_resieve_scatter ------------------ *
 *
 * k_resieve_scatter has k_fill_rows' problem: one thread per modulus, so the
 * smallest bucketed modulus (~I) probes xmax/I positions on one thread while
 * the average thread probes a few hundred. For q >= I a row holds at most one
 * lattice point, at i + I/2 == (rt*j + I/2) mod q, so the rows split into bands
 * that start independently. Thread t takes entry t % n over band t / n, so a
 * warp's lanes share their rows. UNROLL rows are probed before any is acted
 * on, for the memory-level parallelism the note above k_resieve_scatter
 * measured. Same (survivor, prime) pairs as the FK walk; the order of primes
 * inside a survivor's list may differ, and k_td divides them in any order.
 *
 * Row 0 needs no test: its only lattice point is (0,0), never a survivor.
 * j_base counts only under SLABBED, as in pos_row. */
template <int UNROLL, bool SLABBED>
__global__ void k_resieve_rows(const uint32_t *__restrict primes,
                               const uint32_t *__restrict rts,
                               const uint8_t *__restrict ispow,
                               uint32_t n, uint32_t nbands, uint32_t rows,
                               int logI,
                               const uint32_t *__restrict summary,
                               const uint32_t *__restrict bits,
                               const uint32_t *__restrict gbase,
                               uint32_t *__restrict plist,
                               uint32_t *__restrict pcnt,
                               uint32_t K,
                               unsigned long long *__restrict noverflow,
                               int log_gran, uint32_t j_base)
{
    const uint32_t I = 1u << logI;
    const uint32_t NONE = 0xFFFFFFFFu;      /* local x < 2^31, cannot collide */
    const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (uint64_t)n * nbands) return;
    const uint32_t k = (uint32_t)(t % n);
    const uint32_t band = (uint32_t)(t / n);
    const uint32_t rt = rts[k];
    if (rt == ROWRT_NONE) return;
    if (ispow && ispow[k]) return;
    const uint32_t q = primes[k];
    const uint32_t r0 = (uint32_t)((uint64_t)band * rows / nbands);
    const uint32_t r1 = (uint32_t)((uint64_t)(band + 1) * rows / nbands);
    /* the offset follows the GLOBAL row; positions are slab-local */
    const uint32_t jb = SLABBED ? j_base : 0u;
    uint32_t off = (uint32_t)(((uint64_t)rt * (jb + r0) + (I >> 1)) % q);
    unsigned long long ovf = 0;

    for (uint32_t r = r0; r < r1; ) {
        uint32_t xs[UNROLL], hit[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            xs[u] = (r < r1 && off < I) ? (r << logI) + off : NONE;
            if (r < r1) { off += rt; if (off >= q) off -= q; r++; }
        }
        /* every one of these is independent, so they issue together */
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const uint32_t sb = xs[u] >> log_gran;
            hit[u] = (xs[u] == NONE) ? 0u
                                     : ((summary[sb >> 5] >> (sb & 31u)) & 1u);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            if (!hit[u]) continue;
            const uint32_t xv = xs[u];
            if (!((bits[xv >> 5] >> (xv & 31u)) & 1u)) continue;
            const uint32_t idx = td_rank(bits, gbase, xv);
            const uint32_t slot = atomicAdd(&pcnt[idx], 1u);
            if (slot < K) plist[(size_t)idx * K + slot] = q;
            else ovf++;
        }
    }
    if (ovf) atomicAdd(noverflow, ovf);
}

/* ---- --resieve-scan: side 0's resieve from its own records ------------- *
 *
 * Under the fill gate side 0 writes a bucket record only within 2^shift
 * positions of a side-1 survivor, and every two-sided survivor is one. Gate
 * and --sieve-skip drop no record on a two-sided survivor, and fill and
 * resieve walk the same entries from the same cursors. So side 0's records
 * on two-sided survivors are exactly the pairs its walk would find: equal
 * survivor by survivor on five jobs (finding 118). Yet they are 1-6% of side
 * 0's records, which are themselves a few percent of side 1's. A record
 * carries log p, not the prime (finding 130), so side 0's fill also stores
 * each record's factor-base index at the same (region, slot) of `index`
 * (fill_store_index).
 *
 * k_record_scan runs right after the intersection. Under --q-pipeline the
 * next slab's fill reuses the bucket array before this slab's TD, so the scan
 * copies each record on a two-sided survivor out as (position, prime) into a
 * list it owns. k_pair_scatter then does in TD what k_resieve_scatter does
 * with a pair: rank the position, append the prime. Only the order of primes
 * within a survivor's list changes. Nothing downstream sees it: k_td divides
 * them in any order, and both writers sort each factor list.
 *
 * One block per region, whole warps per pass so the ballot has every lane.
 * The index shares the bucket's layout (finding 132); a test's
 * --resieve-scan-cap limits it to icap slots a region. Slots past that have no
 * index. The fill counted them, and the host walks that slab instead, so the
 * scan stops there. *npair is the true total even past pcap, so the host can
 * grow the list. */
__global__ void k_record_scan(const uint32_t *__restrict bucket,
                              const uint32_t *__restrict cursor, bklay_t lay,
                              const uint32_t *__restrict index, uint32_t icap,
                              int log_region,
                              const uint32_t *__restrict two,
                              const uint32_t *__restrict primes,
                              uint32_t *__restrict px, uint32_t *__restrict pp,
                              uint32_t pcap, uint32_t *__restrict npair)
{
    const uint32_t b = blockIdx.x;
    const uint32_t offmask = (1u << log_region) - 1u;
    const uint32_t lane = threadIdx.x & 31u;
    uint32_t cap;
    const uint32_t at = bklay_base(lay, b, &cap);
    const uint32_t *rec = bucket + at;
    const uint32_t *idx = index + at;
    const uint32_t m = min(cursor[b], min(cap, icap));

    for (uint32_t base = threadIdx.x & ~31u; base < m; base += blockDim.x) {
        const uint32_t k = base + lane;
        uint32_t x = 0;
        bool hit = false;
        if (k < m) {
            x = (b << log_region) | (rec[k] & offmask);
            hit = (two[x >> 5] >> (x & 31u)) & 1u;
        }
        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, hit);
        if (!mask) continue;
        uint32_t slot = 0;
        if (lane == 0) slot = atomicAdd(npair, (uint32_t)__popc(mask));
        slot = __shfl_sync(0xFFFFFFFFu, slot, 0) + __popc(mask & ((1u << lane) - 1u));
        if (hit && slot < pcap) { px[slot] = x; pp[slot] = primes[idx[k]]; }
    }
}

/* k_resieve_scatter's append, for the pairs k_record_scan copied out. */
__global__ void k_pair_scatter(const uint32_t *__restrict px,
                               const uint32_t *__restrict pp, uint32_t n,
                               const uint32_t *__restrict bits,
                               const uint32_t *__restrict gbase,
                               uint32_t *__restrict plist,
                               uint32_t *__restrict pcnt, uint32_t K,
                               unsigned long long *__restrict noverflow)
{
    const uint64_t stride = bench_grid_stride_x();
    unsigned long long ovf = 0;
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        const uint32_t idx = td_rank(bits, gbase, px[t]);
        const uint32_t slot = atomicAdd(&pcnt[idx], 1u);
        if (slot < K) plist[(size_t)idx * K + slot] = pp[t];
        else ovf++;
    }
    if (ovf) atomicAdd(noverflow, ovf);
}

/* ---- the trial-division kernel ----------------------------------------- */

#define TDF_NORM_OVERFLOW  1u
#define TDF_LIST_TRUNCATED 2u
/* The flags pipe_td_perq refuses a slab on (fails the band, or skips the
 * slab). k_scatter_sel zeroes its device count on these, so a flag added to
 * the host's refusals belongs here too; pipe_td_perq checks the device's
 * count against its own decision on every slab and fails on a mismatch. */
#define TDF_REFUSE (TDF_NORM_OVERFLOW | TDF_LIST_TRUNCATED)

/* One thread per survivor. The norm is built in registers, divided down in
 * place, and only the residual cofactor is written out -- a 224-bit norm never
 * touches global memory.
 *
 * The small-prime table is STAGED IN SHARED MEMORY in tiles. Every thread
 * walks the same entry list, so the address is warp-uniform and the read looks
 * free -- it is not. Reading the entries straight from global cost 40 ms
 * against an arithmetic floor near 1 ms: at 3,633 entries x 841K survivors the
 * kernel issues 3e9 loads of 24 B each, and load issue, not arithmetic, is
 * what it spends its time on. Shared memory turns each into a broadcast.
 *
 * The survivor loop is the OUTER one because the norm has to stay in registers
 * across the whole entry list, and it runs a uniform iteration count so every
 * thread reaches each __syncthreads. */
#define TD_TILE 512
#define TD_MAXHIT 16      /* buffered small-prime hits; ~7 per survivor typical */
/* Recorded factors per survivor, multiplicity included. 32 was not enough: a
 * 224-bit algebraic norm carrying a high power of a small prime records one
 * entry per division, and a candidate at q=120000007 exceeded it. Exceeding
 * the cap fails the run rather than emitting a truncated factorisation, so the
 * observed maximum is reported to keep the margin visible. */
#define TD_FMAX  64

/* DIVIDE == 0 runs the identical hit test but skips the big-integer division
 * it triggers, and counts the hits instead. That is the only way to tell
 * whether this kernel is spending its time on 3e9 congruence tests or on the
 * few million 256-bit divisions those tests find -- the two have completely
 * different fixes and the fused number cannot distinguish them. */
/* SELECT == 1 runs over a LIST of survivor indices rather than all of them.
 * Inputs stay indexed by the survivor index s = sel[t]; outputs are written at
 * the compacted index t. That is what lets the recording pass -- the only pass
 * that writes a 64-word factor list per thread -- run over the ~1,900 joint
 * candidates instead of the ~240,000 survivors, which is the difference between
 * a 61 MB device array read back per side and a 600 KB one. The dense form is
 * still what the measured passes use; recording is not on the hot path. */
template <int DIVIDE, int RECORD, int SELECT, bool SLABBED = false>
__global__ void k_td(const int64_t *__restrict A, const int64_t *__restrict B,
                     const uint32_t *__restrict X,
                     const uint32_t *__restrict sel, uint32_t n, int logI,
                     const tdpoly_t *__restrict P,
                     uint32_t sq,
                     const uint32_t *__restrict plist,
                     const uint32_t *__restrict pcnt, uint32_t K,
                     const tdsmall_t *__restrict sm, uint32_t nsm,
                     bn_t *__restrict cof, uint8_t *__restrict cofbits,
                     uint32_t *__restrict flags,
                     unsigned long long *__restrict nhit,
                     uint32_t *__restrict fac, uint32_t *__restrict faccnt,
                     uint32_t fmax, uint32_t j_base)
{
    __shared__ tdsmall_t tile[TD_TILE];

    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint64_t nthread = bench_grid_stride_x();
    const uint64_t iters = ((uint64_t)n + nthread - 1) / nthread;
    const int deg = P->deg;
    unsigned long long hits = 0;

    for (uint64_t it = 0; it < iters; it++) {
        const uint64_t tt = bench_grid_thread_x() + it * nthread;
        const bool active = (tt < n);
        const uint32_t t = (uint32_t)tt;
        const uint32_t s = (SELECT && active) ? sel[t] : t;
        const int64_t a = active ? A[s] : 0;
        const int64_t b = active ? B[s] : 0;
        const uint64_t ua = (uint64_t)(a < 0 ? -a : a);
        const uint64_t ub = (uint64_t)(b < 0 ? -b : b);
        const int sa = (a < 0) ? -1 : 1, sb = (b < 0) ? -1 : 1;
        uint32_t myflags = 0;
        uint32_t nf = 0;
        uint32_t *myfac = (RECORD && active) ? fac + (size_t)t * fmax : NULL;
        bn_t N;

        /* ---- exact norm ---- */
        {
            bns_t acc; bns_zero(&acc);
            for (int k = 0; k <= deg; k++) {
                bn_t term = P->c[k];
                int s = P->sign[k];
                if (bn_is_zero(&term)) continue;
                for (int e = 0; e < k; e++) {
                    if (bn_mul_u64(&term, ua)) myflags |= TDF_NORM_OVERFLOW;
                    s *= sa;
                }
                for (int e = 0; e < deg - k; e++) {
                    if (bn_mul_u64(&term, ub)) myflags |= TDF_NORM_OVERFLOW;
                    s *= sb;
                }
                if (bns_addmag(&acc, &term, s)) myflags |= TDF_NORM_OVERFLOW;
            }
            N = acc.m;
        }

        /* ---- the special-q, on its own side ---- */
        if (active && sq)
            td_divide_out<RECORD>(&N, sq, bn_recip_u32(sq), myfac, &nf, fmax);

        /* ---- large primes recovered by the resieve ---- */
        if (active) {
            uint32_t c = pcnt[s];
            if (c > K) { c = K; myflags |= TDF_LIST_TRUNCATED; }
            for (uint32_t k = 0; k < c; k++) {
                uint32_t p = plist[(size_t)s * K + k];
                td_divide_out<RECORD>(&N, p, bn_recip_u32(p), myfac, &nf, fmax);
            }
        }

        /* ---- small primes, direct test ----
         *
         * Hits are BUFFERED rather than divided where they are found. A hit is
         * a per-lane event but the division is a whole-warp one: with ~7 hits
         * per survivor spread over 3,633 entries, a warp stalls on roughly 44
         * separate divisions, each with most lanes idle. Buffering turns that
         * into one dense division loop per survivor where every lane has work,
         * and the entry loop keeps its uniform control flow. */
        {
            const uint32_t x = active ? X[s] : 0u;
            const int32_t  i = (int32_t)(x & Imask) - Ihalf;
            const uint32_t jlocal = x >> logI;
            uint32_t jglobal = jlocal;
            if constexpr (SLABBED) jglobal += j_base;
            const uint32_t hi = (uint32_t)(Ihalf - i);      /* in (0, 2^logI] */
            uint32_t hitp[TD_MAXHIT];
            uint64_t hitr[TD_MAXHIT];       /* carry the reciprocal with it:
                                             * recomputing it later would put a
                                             * 64-bit division back in the hot
                                             * path, which is what this whole
                                             * arrangement exists to avoid */
            uint32_t nh = 0;

            for (uint32_t base = 0; base < nsm; base += TD_TILE) {
                const uint32_t cnt = min((uint32_t)TD_TILE, nsm - base);
                __syncthreads();
                for (uint32_t k = threadIdx.x; k < cnt; k += blockDim.x)
                    tile[k] = sm[base + k];
                __syncthreads();
                if (!active) continue;
                for (uint32_t e = 0; e < cnt; e++) {
                    const uint32_t m = tile[e].m, g = tile[e].g;
                    /* Ordinary prime entries have g==1 and use slab-local j
                     * with cst pre-phased once per slab. g>1 implies m==1 for
                     * a prime modulus, so only its global-row condition remains. */
                    uint32_t jp = jlocal;
                    if (g > 1) {
                        if (jglobal % g) continue;
                        jp = jglobal / g;
                    }
                    if (tile[e].magic) {
                        uint32_t w = tile[e].rt * jp + hi;
                        if (td_mod_magic(w, m, tile[e].magic, tile[e].sh) != tile[e].cst)
                            continue;
                    }
                    /* magic == 0 means m == 1: the row condition was the whole
                     * test, and it has already passed. */
                    hits++;
                    if (DIVIDE) {
                        hitp[nh] = m * g; hitr[nh] = tile[e].recip; nh++;
                        if (nh == TD_MAXHIT) {          /* drain and carry on */
                            for (uint32_t z = 0; z < TD_MAXHIT; z++)
                                td_divide_out<RECORD>(&N, hitp[z], hitr[z],
                                                      myfac, &nf, fmax);
                            nh = 0;
                        }
                    }
                }
            }
            if (DIVIDE)
                for (uint32_t z = 0; z < nh; z++)
                    td_divide_out<RECORD>(&N, hitp[z], hitr[z],
                                          myfac, &nf, fmax);
        }

        if (active) {
            cof[t] = N;
            cofbits[t] = (uint8_t)bn_bits(&N);
            if (RECORD) faccnt[t] = nf;
            if (myflags) atomicOr(flags, myflags);
        }
    }
    if (nhit && hits) atomicAdd(nhit, hits);
}

/* ---- the recording pass, one WARP per candidate ------------------------ *
 *
 * `k_td` is the right shape for the two dense passes, which run over every
 * survivor and are throughput-bound on `n`. It is the WRONG shape for the
 * recording pass, and slabbing is what made that visible.
 *
 * That pass runs `SELECT=1` over the ~1,900-5,300 joint candidates, and one
 * thread per candidate means one thread marching the entire `nsm` entry list
 * (6,726 on a c194). Measured per launch across a slab sweep it costs a dead
 * flat ~0.65 ms whether the launch carries 5,342 candidates or 577 -- it is
 * latency-bound on `nsm`, not throughput-bound on `nacc`, which is why sizing
 * the grid to `ceil(nacc/threads)` was tried and measured nothing (RESULTS
 * finding 74). Two launches per slab then made the pass grow at ~1.33 ms per
 * slab, and that per-slab tax is the entire reason the slab-size curve turns
 * back up below 2^29 positions.
 *
 * Here a whole warp takes one candidate and the 32 lanes stride the entry
 * list, so the march is ~32x shorter for the same candidate. Three things make
 * that cheap rather than delicate:
 *
 *   - the norm stays in ONE lane. `bn_t` is 256 bits and the divisions are
 *     inherently sequential on N, so shuffling it between lanes would cost
 *     more than it saves. Lane 0 owns N, `nf` and the factor list; the other
 *     lanes only ever evaluate the congruence. Their idling is free: the
 *     serial part was one thread's work before this change too.
 *   - the hit mask comes back from `__ballot_sync` in ASCENDING LANE ORDER,
 *     and lane index is entry index within the 32-entry chunk, so walking it
 *     with __ffs divides factors out in the same entry order `k_td` uses. That
 *     is a CONVENIENCE, not the correctness argument -- an earlier version of
 *     this comment claimed otherwise and was wrong. `td_divide_out` loops
 *     until p no longer divides, so an entry consumes the whole power of its
 *     prime whichever entry reaches it first, and distinct primes commute:
 *     the recorded multiset is order-invariant. Both emitters sort before
 *     writing anyway (`cf_fmt_sorted`, cofac.cuh; `std::sort`,
 *     pipeline.cuh), so relation TEXT cannot see the order either.
 *     Order is load-bearing in exactly one place -- when `nf` exceeds `fmax`
 *     the tail is dropped, so which factors survive depends on it -- and that
 *     path is a hard error the host reports as "raise TD_FMAX". A
 *     lanes-per-item rewrite of the dense pass (STATUS item 18) is therefore
 *     NOT constrained to preserve this order.
 *   - the shared tile is unchanged and still block-wide: every warp in the
 *     block walks the same entry list, so one staged tile serves all of them
 *     and the `__syncthreads` pair stays uniform (`iters` is computed from the
 *     warp count, so every warp makes the same number of passes).
 *
 * No hit buffer is needed. `k_td` buffers into `hitp[TD_MAXHIT]` because a hit
 * is a per-lane event there while the division is a whole-warp one; here the
 * ballot has already done the compaction, and lane 0's division loop skips
 * empty chunks in a single predicate.
 *
 * DIVIDE=0 and the dense SELECT=0 form have no counterpart here on purpose:
 * this kernel exists for the one launch that was paying the tax, and leaving
 * the diagnostics on `k_td` keeps a second, independent implementation of the
 * same predicate in the tree.
 *
 * WHAT COVERS THIS KERNEL, precisely -- `--verify` does NOT. pipe_td_verify
 * launches `k_td<1,1,0>` into its own d_fac/d_faccnt (pipeline.cuh:796) and
 * runs BEFORE the `if (!nacc) return 0;` that guards phase 3, so it never
 * reads C->d_cfac[side], the only array this kernel writes. A --verify run
 * staying green says nothing about the warp path. The actual gates are
 * out-of-process: cofcheck.sh's A/B against `--td-record-scalar` at two slab
 * shapes, and `--check-relations` rebuilding norms from the emitted factor
 * lists. Anyone adding an in-binary gate for this kernel has to compare
 * d_cfac after phase 3, not before it. */
template <bool SLABBED = false>
__global__ void k_td_record_warp(const int64_t *__restrict A,
                                 const int64_t *__restrict B,
                                 const uint32_t *__restrict X,
                                 const uint32_t *__restrict sel,
                                 uint32_t n, int logI,
                                 const tdpoly_t *__restrict P,
                                 uint32_t sq,
                                 const uint32_t *__restrict plist,
                                 const uint32_t *__restrict pcnt, uint32_t K,
                                 const tdsmall_t *__restrict sm, uint32_t nsm,
                                 bn_t *__restrict cof,
                                 uint8_t *__restrict cofbits,
                                 uint32_t *__restrict flags,
                                 uint32_t *__restrict fac,
                                 uint32_t *__restrict faccnt,
                                 uint32_t fmax, uint32_t j_base,
                                 const uint32_t *__restrict dn = nullptr)
{
    __shared__ tdsmall_t tile[TD_TILE];

    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint32_t lane  = threadIdx.x & 31u;
    const uint32_t wpb   = blockDim.x >> 5;            /* warps per block */
    /* A grid sized for a bound (dn) has blocks with no candidate in any
     * iteration: they leave before the tile loop, whole blocks at once, so
     * its __syncthreads never see a partial block. */
    if (dn) n = min(n, *dn);
    if ((uint64_t)blockIdx.x * wpb >= n) return;
    const uint64_t nwarp = (uint64_t)gridDim.x * wpb;
    const uint64_t w0    = (uint64_t)blockIdx.x * wpb + (threadIdx.x >> 5);
    const uint64_t iters = ((uint64_t)n + nwarp - 1) / nwarp;
    const int deg = P->deg;

    for (uint64_t it = 0; it < iters; it++) {
        const uint64_t tt = w0 + it * nwarp;
        /* Warp-uniform: `tt` derives from `w0`, which is built from
         * (threadIdx.x >> 5), so every lane of a warp shares it. The
         * unconditional __ballot_sync below depends on this -- if `active`
         * were ever per-lane, lanes that took the `continue` would still be
         * named in the mask. Do not derive `tt` from a per-lane quantity. */
        const bool active = (tt < n);
        const bool leader = active && lane == 0;
        const uint32_t t = (uint32_t)tt;
        const uint32_t s = active ? sel[t] : 0u;
        uint32_t myflags = 0;
        uint32_t nf = 0;
        uint32_t *myfac = leader ? fac + (size_t)t * fmax : NULL;
        bn_t N;

        /* Everything that touches N runs on lane 0 alone. The other lanes are
         * masked off for exactly the stretch that used to be the whole
         * kernel's serial cost, so this costs nothing they were not already
         * waiting through. */
        if (leader) {
            const int64_t a = A[s], b = B[s];
            const uint64_t ua = (uint64_t)(a < 0 ? -a : a);
            const uint64_t ub = (uint64_t)(b < 0 ? -b : b);
            const int sa = (a < 0) ? -1 : 1, sb = (b < 0) ? -1 : 1;

            /* ---- exact norm ---- */
            bns_t acc; bns_zero(&acc);
            for (int k = 0; k <= deg; k++) {
                bn_t term = P->c[k];
                int sg = P->sign[k];
                if (bn_is_zero(&term)) continue;
                for (int e = 0; e < k; e++) {
                    if (bn_mul_u64(&term, ua)) myflags |= TDF_NORM_OVERFLOW;
                    sg *= sa;
                }
                for (int e = 0; e < deg - k; e++) {
                    if (bn_mul_u64(&term, ub)) myflags |= TDF_NORM_OVERFLOW;
                    sg *= sb;
                }
                if (bns_addmag(&acc, &term, sg)) myflags |= TDF_NORM_OVERFLOW;
            }
            N = acc.m;

            /* ---- the special-q, on its own side ---- */
            if (sq) td_divide_out<1>(&N, sq, bn_recip_u32(sq), myfac, &nf, fmax);

            /* ---- large primes recovered by the resieve ---- */
            uint32_t c = pcnt[s];
            if (c > K) { c = K; myflags |= TDF_LIST_TRUNCATED; }
            for (uint32_t k = 0; k < c; k++) {
                const uint32_t p = plist[(size_t)s * K + k];
                td_divide_out<1>(&N, p, bn_recip_u32(p), myfac, &nf, fmax);
            }
        }

        /* ---- small primes, direct test, 32 entries at a time ---- */
        {
            const uint32_t x = active ? X[s] : 0u;
            const int32_t  i = (int32_t)(x & Imask) - Ihalf;
            const uint32_t jlocal = x >> logI;
            uint32_t jglobal = jlocal;
            if constexpr (SLABBED) jglobal += j_base;
            const uint32_t hi = (uint32_t)(Ihalf - i);      /* in (0, 2^logI] */

            for (uint32_t base = 0; base < nsm; base += TD_TILE) {
                const uint32_t cnt = min((uint32_t)TD_TILE, nsm - base);
                __syncthreads();
                for (uint32_t k = threadIdx.x; k < cnt; k += blockDim.x)
                    tile[k] = sm[base + k];
                __syncthreads();
                if (!active) continue;
                for (uint32_t c = 0; c < cnt; c += 32) {
                    const uint32_t e = c + lane;
                    bool hit = false;
                    if (e < cnt) {
                        const uint32_t m = tile[e].m, g = tile[e].g;
                        uint32_t jp = jlocal;
                        hit = true;
                        if (g > 1) {
                            if (jglobal % g) hit = false;
                            else jp = jglobal / g;
                        }
                        if (hit && tile[e].magic) {
                            const uint32_t w = tile[e].rt * jp + hi;
                            if (td_mod_magic(w, m, tile[e].magic, tile[e].sh)
                                != tile[e].cst) hit = false;
                        }
                    }
                    /* Ascending lane order == ascending entry order == the
                     * order k_td divides in. Convenient for diffing factor
                     * arrays, but NOT required: see the header -- the multiset
                     * is order-invariant and both emitters sort. */
                    uint32_t mask = __ballot_sync(0xffffffffu, hit);
                    if (lane == 0) {
                        while (mask) {
                            const uint32_t z = c + (uint32_t)__ffs(mask) - 1u;
                            mask &= mask - 1u;
                            td_divide_out<1>(&N, tile[z].m * tile[z].g,
                                             tile[z].recip, myfac, &nf, fmax);
                        }
                    }
                }
            }
        }

        if (leader) {
            cof[t] = N;
            cofbits[t] = (uint8_t)bn_bits(&N);
            faccnt[t] = nf;
            if (myflags) atomicOr(flags, myflags);
        }
    }
}

/* ---- cofactor classification ------------------------------------------- */

/* Applies CADO's check_leftover_norm per survivor. This is the stage that
 * turns hundreds of thousands of survivors into the ~1,851 that are worth
 * handing to a cofactorizer, and the 2026-08-04 harness work established that
 * it is real work the siever has to do, not "a small input adapter". */
__global__ void k_classify(const bn_t *__restrict cof,
                           const uint8_t *__restrict cofbits,
                           const int64_t *__restrict B, uint32_t n,
                           uint32_t lpb, uint32_t mfb, double lim,
                           uint8_t *__restrict status)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        bn_t c = cof[t];
        /* b == 0 is the lattice point (i,j) = (0,1): a = q, b = 0. It passes
         * gcd(i,j) == 1 and both norm bounds, and its cofactors classify
         * cleanly -- but (a,0) is not a relation, and it turned up as a
         * spurious candidate on the first full run. las carries it through
         * after_sieve too and drops it later. */
        if (B[t] == 0) { status[t] = COF_DEGENERATE; continue; }
        status[t] = (uint8_t)cof_classify(&c, cofbits[t], lpb, mfb, lim);
    }
}

/* ---- joint acceptance, compacted on device ----------------------------- *
 *
 * A record is only worth recording if BOTH sides classified it as a
 * cofactorisation candidate; one side accepting is not a candidate at all. The
 * join used to happen on the host over every survivor, which meant reading back
 * both sides' whole factor matrices to reach ~1,900 rows. Doing the intersection
 * here reduces the readback to those rows.
 *
 * Ordered, not atomic: a prefix scan over the accept flags gives each candidate
 * a deterministic slot, so the emitted batch is byte-reproducible run to run.
 * That property is what makes diffing this path against the two-process one a
 * real regression test, so it is not worth trading for one atomicAdd. */
__global__ void k_accept_flags(const uint8_t *__restrict st0,
                               const uint8_t *__restrict st1,
                               uint32_t n, uint32_t *__restrict flag)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        uint32_t a = (st0[t] == COF_ACCEPT || st0[t] == COF_SPLIT);
        uint32_t b = (st1[t] == COF_ACCEPT || st1[t] == COF_SPLIT);
        flag[t] = a & b;
    }
}

/* spec, if not NULL: the count the record pass queued ahead of the host's
 * read of the counter block takes (--q-pipeline, finding 135) -- the accepted
 * count, or 0 wherever the host will not record this slab from this scatter:
 * a flag it skips the slab or fails the band on, or more candidates than sel
 * holds (it grows sel and scatters and records again). tdflags are final
 * here: every kernel that raises them ran before this one. */
__global__ void k_scatter_sel(const uint32_t *__restrict flag,
                              const uint32_t *__restrict off,
                              uint32_t n, uint32_t *__restrict sel,
                              uint32_t cap, uint32_t *__restrict nacc,
                              const uint32_t *__restrict tdflags = nullptr,
                              uint32_t *__restrict spec = nullptr)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (flag[t] && off[t] < cap) sel[off[t]] = t;
        if (t == n - 1) {
            const uint32_t na = off[t] + flag[t];
            *nacc = na;
            if (spec)
                *spec = ((*tdflags & TDF_REFUSE) || na > cap) ? 0u : na;
        }
    }
}

/* dn, here and in the record pass's other kernels: a device count (k_scatter_sel's
 * spec) that replaces n when it is smaller; n is then only the bound the grid
 * was sized for. */
__global__ void k_gather_ab(const int64_t *__restrict A,
                            const int64_t *__restrict B,
                            const uint32_t *__restrict sel, uint32_t n,
                            int64_t *__restrict oa, int64_t *__restrict ob,
                            const uint32_t *__restrict dn = nullptr)
{
    const uint64_t stride = bench_grid_stride_x();
    if (dn) n = min(n, *dn);
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        oa[t] = A[sel[t]]; ob[t] = B[sel[t]];
    }
}

#endif  /* __CUDACC__ */
#endif  /* CUDA_SIEVE_TD_CUH */
