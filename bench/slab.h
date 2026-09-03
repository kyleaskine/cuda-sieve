/* slab.h -- geometry helpers for j-slabbing the production sieve.
 *
 * A slab keeps the hot sieve position x = j_local*I + (i + I/2) in uint32_t
 * and below 2^31, while j_base carries the global row origin outside the walk.
 * The ordinary path never calls any of this from a device hot loop: kernels are
 * specialised on SLABBED=false and the compiler removes every slab operation.
 */
#ifndef CUDA_SIEVE_SLAB_H
#define CUDA_SIEVE_SLAB_H

#include <stdint.h>

#ifdef __CUDACC__
#define SLAB_HD __host__ __device__ static inline
#else
#define SLAB_HD static inline
#endif

typedef struct {
    uint32_t jmax;       /* maximum rows in one slab                    */
    uint32_t nslab;      /* ceil(J / jmax)                              */
    int      enabled;    /* nslab > 1                                  */
} slab_plan_t;

/* Trial division ranks the survivor bitmap in groups of 8 x 32-bit words =
 * 256 positions. Every slab, including the final tail, must contain a whole
 * number of those groups. For logI >= 8 every complete j row already does. */
#define SLAB_TD_GROUP_POS 256u

/* Performance policy: once a full sieve reaches TWICE the region target, auto
 * mode caps the slab. Both halves are region-relative -- at the default
 * --region 14 that trigger is 2^30 positions and the cap is 2^29, but at
 * --region 12 they are 2^28 and 2^27. Do not restate either as an absolute.
 *
 * NOTE THE UPWARD DIRECTION, added 2026-09-02. Holding the region COUNT fixed
 * means the slab AREA scales with the region size, so raising --region raises
 * the auto slab and with it peak bucket memory: 2x at region 15 and 4x at
 * region 16 against what the old absolute 2^29 target produced. That is the
 * intended consequence of the policy, not a regression -- but a --region 16
 * run that fit before may now need 4x the bucket allocation.
 *
 * THE TARGET IS A BUCKET-REGION COUNT, NOT AN AREA (finding 79, 2026-08-26).
 * `fill` is minimised at a fixed number of bucket regions, so the optimal slab
 * AREA halves with --region: 2^29 at region 14, 2^28 at 13, 2^27 at 12. The
 * old form of this constant was `2^29` with no reference to log_region, which
 * left the target wrong by the same factor whenever --region moved -- measured
 * +28.4% fill at region 13 and +68.3% at region 12. Expressing it as regions
 * and deriving the rows is behaviour-preserving at the default (32768 << 14 =
 * 2^29) and correct everywhere else.
 *
 * 32768 is the REACHABLE optimum at region 14, not the global one: finding 80
 * puts the minimum of L(nregion) at 16,384, but reaching it from region 14
 * needs twice the slabs and the factor-base re-stream cancels the gain. Region
 * 14 remains the joint optimum because k_apply launches one block per region
 * and costs +52% at region 13.
 *
 * Two independent benchmarks (Ampere RTX 3090 and Blackwell RTX 5070) found
 * this working set near the fill/TD crossover; an L40 instead preferred twice
 * it -- i.e. 65,536 regions -- and finding 81 leaves that card's preference
 * unexplained, since the mechanism is sector-level read-modify-write and not
 * L2 capacity. So this is a generic performance/memory cap and a per-card
 * autotune candidate (item 2), not a universal speed optimum. Explicit
 * --slab-j overrides it; the 2^31 position and direct-TD bounds below remain
 * mandatory. */
/* THE TRIGGER MUST SCALE WITH THE TARGET, and it keeps its factor of two.
 *
 * Splitting begins at TWICE the target, not at it: an area between one and two
 * target-sized slabs is left alone deliberately, because splitting it buys a
 * slab of half the target while paying a second full factor-base re-stream
 * (929 MB, finding 78). That hysteresis is why `{16, J 16383}` -- just under
 * 2^30 -- is one slab in the gate below, and it must stay.
 *
 * What was wrong was expressing it as an ABSOLUTE 2^30 while the target went
 * region-relative. At --region 12 the target is 32768 << 12 = 2^27, but an
 * area of 2^29 was still measured against 2^30 and so never split: 131,072
 * regions in one slab, 4x the target, the exact shape finding 79 measured at
 * +68.3% fill. Deriving the trigger as `target * 2` fixes that and reproduces
 * the old 2^30 exactly at the default --region 14, so no default moves. */
#define SLAB_PERF_REGIONS      32768u

static inline uint32_t slab_row_quantum(int logI)
{
    if (logI < 0 || logI > 30) return 0;
    return logI >= 8 ? 1u : (1u << (8 - logI));
}

static inline int slab_rows_shape_ok(int logI, uint32_t rows)
{
    const uint32_t q = slab_row_quantum(logI);
    return q && rows && (rows % q) == 0;
}

/* Position-space limit. 2^31 itself is a valid exclusive endpoint in a
 * uint32_t; individual positions are in [0, 2^31).
 *
 * plat_t's walk increments are intentionally NOT a slab-plan constraint. They
 * are uint64_t (plattice.cuh) because a reduced increment can exceed 2^32 even
 * when every slab-local position is below 2^31. The planner only has to bound
 * quantities that remain 32-bit: local x and the direct-TD reciprocal input. */
static inline uint32_t slab_area_jmax(int logI)
{
    if (logI < 0 || logI > 30) return 0;
    return (uint32_t)(((uint64_t)1 << 31) >> logI);
}

/* The one definition of a valid bucket-region exponent. Three sites test it
 * -- slab_perf_jmax, slab_make_plan and bench_main.cu's argument validator --
 * and three hand-copied `1..30` comparisons is the shape the finding-79 bug
 * came from. Keep them in step through this. */
static inline int slab_region_ok(int log_region)
{
    return log_region >= 1 && log_region <= 30;
}

/* Return the auto-mode performance cap in rows. UINT32_MAX means the geometry
 * already fits INSIDE THE HYSTERESIS BAND -- an area below *two* target-sized
 * slabs -- and should not be split for performance alone. If a single row
 * exceeds the target -- SLAB_PERF_REGIONS << log_region positions, so 2^29 at
 * the default --region 14 but 2^27 at region 12 -- one row is the smallest
 * representable slab and the correctness caps still apply. */
static inline uint32_t slab_perf_jmax(int logI, int log_region, uint32_t J)
{
    uint64_t I, area, rows;
    if (logI < 0 || logI > 30 || !J) return 0;
    if (!slab_region_ok(log_region)) return 0;
    I = (uint64_t)1 << logI;
    area = I * (uint64_t)J;
    /* Target a region COUNT; the area follows from log_region, and so does
     * the split trigger. The gate is deliberately at TWO targets, not one:
     * see the hysteresis paragraph in the policy block above before
     * "correcting" it to `area < target`. An area of 1.9 targets stays
     * unsplit on purpose. */
    {
        const uint64_t target = ((uint64_t)SLAB_PERF_REGIONS) << log_region;
        if (area < target * 2u) return 0xffffffffu;   /* the hysteresis */
        rows = target / I;
    }
    if (!rows) rows = 1u;
    return rows > 0xffffffffull ? 0xffffffffu : (uint32_t)rows;
}

/* The direct small-prime predicate uses
 *
 *     w = rt * j_local + hi,    1 <= hi <= I,
 *
 * and td_mod's multiply/shift reciprocal is exact for w < 2^31.  A transformed
 * prime has rt < m <= p, so max_prime is a conservative bound on rt+1.
 * Return the largest slab height whose ENTIRE local row range satisfies that
 * invariant. No small-prime entries means the TD bound is irrelevant.
 */
static inline uint32_t slab_td_jmax(int logI, uint32_t max_prime)
{
    const uint64_t I = (logI >= 0 && logI <= 30) ? ((uint64_t)1 << logI) : 0;
    uint64_t max_rt, room, rows;
    if (!I || I > 0x7fffffffull) return 0;
    if (max_prime <= 1) return 0xffffffffu;
    max_rt = (uint64_t)max_prime - 1u;
    room = 0x7fffffffull - I;
    rows = 1u + room / max_rt;
    return rows > 0xffffffffull ? 0xffffffffu : (uint32_t)rows;
}

/* Build the host-side schedule. forced_j == 0 means auto. A nonzero value is
 * deliberately strict: it is a regression/testing knob, not permission to
 * violate either the position or direct-TD arithmetic bound. */
static inline int slab_make_plan(int logI, int log_region, uint32_t J,
                                 uint32_t max_small_prime,
                                 uint32_t forced_j, slab_plan_t *P)
{
    uint32_t amax, tmax, perf_jmax, jmax, quantum;
    uint64_t n;
    if (!P || !J) return -1;
    /* Validate the region on BOTH paths. It is unused when forced_j != 0, but
     * the parameter is part of this function's contract and a caller passing
     * a bogus region deserves a failure rather than a plan that silently
     * ignored one of its arguments. */
    if (!slab_region_ok(log_region)) return -1;
    quantum = slab_row_quantum(logI);
    if (!quantum || !slab_rows_shape_ok(logI, J)) return -1;
    amax = slab_area_jmax(logI);
    tmax = slab_td_jmax(logI, max_small_prime);
    if (!amax || !tmax) return -1;
    jmax = amax < tmax ? amax : tmax;
    if (forced_j) {
        const uint32_t requested = forced_j < J ? forced_j : J;
        if (requested > jmax || !slab_rows_shape_ok(logI, requested)) return -1;
        jmax = requested;
    } else {
        perf_jmax = slab_perf_jmax(logI, log_region, J);
        if (!perf_jmax) return -1;
        if (perf_jmax < jmax) jmax = perf_jmax;
        jmax -= jmax % quantum;
    }
    if (jmax > J) jmax = J;
    if (!jmax || !slab_rows_shape_ok(logI, jmax)) return -1;
    n = ((uint64_t)J + jmax - 1u) / jmax;
    if (n > 0xffffffffull) return -1;
    P->jmax = jmax;
    P->nslab = (uint32_t)n;
    P->enabled = (P->nslab > 1u);
    return 0;
}

SLAB_HD uint32_t slab_rows_at(const slab_plan_t *P, uint32_t J, uint32_t slab)
{
    const uint64_t base = (uint64_t)slab * P->jmax;
    const uint64_t left = base < J ? (uint64_t)J - base : 0;
    return left > P->jmax ? P->jmax : (uint32_t)left;
}

SLAB_HD uint32_t slab_jbase_at(const slab_plan_t *P, uint32_t slab)
{
    return (uint32_t)((uint64_t)slab * P->jmax);
}

/* Advance the small-prime congruence target by delta_j rows. If cst represents
 *
 *     rt*j_local + hi == cst (mod m)
 *
 * at one slab origin, the returned target represents the same global
 * congruence after that origin moves forward by delta_j. This runs once per
 * small-prime ENTRY per slab, not once per survivor x entry test. */
SLAB_HD uint32_t slab_phase_cst(uint32_t cst, uint32_t rt, uint32_t m,
                                uint32_t delta_j)
{
    uint32_t d;
    if (m <= 1u) return cst;
    d = (uint32_t)(((uint64_t)rt * delta_j) % m);
    return cst >= d ? cst - d : cst + (m - d);
}

#undef SLAB_HD
#endif
