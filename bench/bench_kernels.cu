/* GPU kernels for the bucket-fill benchmark.
 *
 * Stages, matching the design doc's cost pillars:
 *   (T) transform + plattice : one modular inverse per distinct prime per
 *                              special-q, then the FK reduction. Producer
 *                              only -- deliberately NOT fused with scatter
 *                              (msieve-s experiment #11: fusing cost +25%).
 *   (a) FILL_ATOMIC          : one global atomicAdd per record, 16K-way.
 *   (c) FILL_TWOLEVEL        : level 1 stages records in shared memory and
 *                              flushes full cache lines with one atomic per
 *                              flush; level 2 splits super-buckets into
 *                              regions the same way.
 */
#include "bench.h"
#include "platform.h"
#include "plattice.cuh"
#include "bigint.cuh"
#include "td.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <errno.h>

/* Host-side wall clock, for the per-q work that runs off the GPU and so is
 * invisible to cudaEvent timing. Goal 1 is about host demand, so this has to
 * be billed rather than assumed small. */
static double host_ms(void)
{
    return bench_monotonic_ms();
}

static void report_slice_build_error(void)
{
    if (errno == EOVERFLOW)
        fprintf(stderr, "fb_build_slices: factor base requires more than"
                " 65,536 slices; bucket records carry only a 16-bit slice ID\n");
    else if (errno == EINVAL)
        fprintf(stderr, "fb_build_slices: empty factor base or missing log table\n");
    else
        perror("fb_build_slices");
}

static int cuda_check_impl(cudaError_t err, const char *expr,
                           const char *file, int line)
{
    if (err == cudaSuccess) return 0;
    fprintf(stderr, "CUDA %s: %s at %s:%d\n",
            expr, cudaGetErrorString(err), file, line);
    return -1;
}

#define CUDA_CHECKED(x) cuda_check_impl((x), #x, __FILE__, __LINE__)
#define CK(x) do { if (CUDA_CHECKED(x)) return -1; } while (0)

/* The maximum opt-in dynamic shared memory is a property of the selected
 * device, not of a CUDA architecture family.  Query it at runtime so cards
 * such as A100/H100 can use larger legal regions while cards with a smaller
 * limit are rejected before cudaFuncSetAttribute()/launch. */
static int cuda_optin_smem_limit(size_t *out)
{
    int dev = 0, lim = 0;
    if (!out) return -1;
    if (CUDA_CHECKED(cudaGetDevice(&dev))) return -1;
    if (CUDA_CHECKED(cudaDeviceGetAttribute(
            &lim, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev))) return -1;
    if (lim <= 0) {
        fprintf(stderr,
                "CUDA device %d reported an invalid opt-in shared-memory limit"
                " of %d B\n", dev, lim);
        return -1;
    }
    *out = (size_t)lim;
    return 0;
}

/* ---- stage T: root transform + plattice reduction --------------------- */

/* The transform runs through pl_transform_enc, not pl_transform, for three
 * reasons that all bite on CADO's factor base and none of which bite on
 * GGNFS's:
 *   - a projective entry (root >= q) must keep its reciprocal, and the affine
 *     formula would reduce it to a bogus affine root instead;
 *   - q = 2^15 sits exactly at the default bkthresh, so an EVEN modulus
 *     reaches this kernel, and binary-Euclid pl_invmod cannot invert mod 2^k;
 *   - raising -maxbits puts odd prime powers here, where a non-invertible
 *     denominator makes binary Euclid spin forever ON THE DEVICE.
 * pl_transform_enc handles all three; the first two are wrong answers and the
 * third is a hang, so none of them would have shown up as a failed gate.
 * pl_transform_fast is the same function with a branch-free path for odd
 * moduli whose denominator it can prove invertible; everything else still
 * reaches pl_transform_enc, and fbtest compares the two exactly.
 *
 * What is still not expressible is g > 1: hits confined to every g-th row,
 * which is not a plat_t walk. Those emit an empty walk, and the kernel
 * accumulates the number of positions thereby dropped so the loss is a printed
 * number rather than a silence. With the default bkthresh = I >= J it is
 * exactly zero: g > 1 needs q | (rows), and every bucketed q exceeds J. */
template <bool SLABBED>
__global__ void k_transform(const uint32_t *__restrict primes,
                            const uint32_t *__restrict roots,
                            plat_t *__restrict out,
                            uint32_t n, int logI, uint32_t J,
                            int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                            uint32_t *__restrict nproj,
                            unsigned long long *__restrict nlost,
                            uint64_t *__restrict walk_cur,
                            uint32_t *__restrict rt_out = nullptr,
                            uint32_t n_rt = 0)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const uint32_t q = primes[k];
        uint32_t rt, g, m = pl_transform_fast(q, roots[k], a0, a1, b0, b1, &rt, &g);
        /* --fill-rows: the row tier walks i == rt*j (mod q) row by row, which
         * is the same lattice pl_make reduces whenever g == 1 (then m == q). */
        if (k < n_rt) rt_out[k] = (g > 1) ? ROWRT_NONE : rt;
        if (g > 1) {                         /* rows only: emit an empty walk */
            plat_t P; P.inc_warp = PL_INVALID; P.inc_step = PL_VERTICAL;
            P.bound_warp = 0; P.bound_step = 0;
            out[k] = P;
            if constexpr (SLABBED) walk_cur[k] = UINT64_MAX;
            atomicAdd(nproj, 1u);
            atomicAdd(nlost, (unsigned long long)(J / g) * ((1u << logI) / m));
        } else {
            const plat_t P = pl_make(m, rt, logI);
            out[k] = P;
            if constexpr (SLABBED) walk_cur[k] = pl_first64(&P, logI);
        }
    }
}

/* ---- positions gcd(i,j) already rules out (--sieve-skip, finding 100) ----
 *
 * x = j*I + (i + I/2) and I/2 is even, so parity(i) == parity(x). Every stage
 * that skips a both-even position -- fill, apply's init, the small sieve and
 * the scan -- asks these two helpers, because they must agree exactly: fill
 * dropping a record for a cell apply still scans in would lose relations
 * silently, and no gate compares the four sites with each other.
 *
 * pos_row is the GLOBAL row. Slabbed kernels add j_base; unslabbed ones do not,
 * so they stay correct whatever j_base a caller passes. */
template <bool SLABBED>
__device__ __forceinline__ uint32_t pos_row(uint32_t x, int logI, uint32_t j_base)
{
    uint32_t j = x >> logI;
    if constexpr (SLABBED) j += j_base;
    return j;
}

__device__ __forceinline__ int pos_both_even(uint32_t x, uint32_t j)
{
    return ((x | j) & 1u) == 0u;
}

/* --fill-gate: the side filled SECOND stores a record only where the side
 * filled first has a survivor within the same 2^gate_shift positions. A
 * position the first side rejected is dropped by k_intersect_compact whatever
 * the second side's sum is, so the records that matter are all still written
 * and the relations are unchanged. gate == NULL is the ungated fill. */
__device__ __forceinline__ int fill_gate_drops(const uint32_t *__restrict gate,
                                               int gate_shift, uint32_t x)
{
    if (!gate) return 0;
    const uint32_t g = x >> gate_shift;
    return !((gate[g >> 5] >> (g & 31u)) & 1u);
}

/* --resieve-scan (finding 118): INDEX fills also store each record's
 * factor-base index, ibase + k, at the same (region, slot) of a second array
 * of icap slots per region, for k_record_scan. A slot at or past icap has no
 * room and is counted in index_ovf, and the host then walks that slab's
 * resieve instead. The record itself is stored exactly as without INDEX. */
__device__ __forceinline__ void fill_store_index(uint32_t *__restrict index,
                                                 uint32_t icap,
                                                 uint32_t *__restrict index_ovf,
                                                 uint32_t b, uint32_t slot,
                                                 uint32_t fbi)
{
    if (slot < icap) index[(size_t)b * icap + slot] = fbi;
    else atomicAdd(index_ovf, 1u);
}

/* ---- stage (a): naive single-level atomic append ---------------------- */

/* One atomicAdd per record into a 2^log_nbuckets-way split. This is the
 * baseline the design doc says to beat. */
template <int RECBYTES, bool SLABBED = false, bool INDEX = false>
__global__ void k_fill_atomic(const plat_t *__restrict plat,
                              const uint16_t *__restrict slice,
                              uint32_t n, uint32_t xmax, int logI,
                              int log_region,
                              uint32_t *__restrict cursor,
                              uint8_t *__restrict out, uint32_t cap,
                              uint32_t *__restrict overflow,
                              const uint64_t *__restrict walk_cur,
                              uint64_t *__restrict walk_next,
                              int sieve_skip, uint32_t j_base,
                              const uint32_t *__restrict gate = nullptr,
                              int gate_shift = 0,
                              uint32_t *__restrict index = nullptr,
                              uint32_t icap = 0, uint32_t ibase = 0,
                              uint32_t *__restrict index_ovf = nullptr)
{
    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t offmask = (1u << log_region) - 1;
    const uint64_t stride = bench_grid_stride_x();
    /* --sieve-skip: 3 | i  <=>  (x & Imask) == I/2 (mod 3), since i = (x & Imask) - I/2. */
    const uint32_t ih3 = (1u << (logI - 1)) % 3u;

    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) {
            if constexpr (SLABBED) walk_next[k] = walk_cur[k];
            continue;
        }
        /* one read per prime, not per record: the slice hint is what the apply
         * kernel turns back into log p (and what resieve would use to recover
         * the prime itself), exactly as CADO's shorthint does */
        const uint32_t sl = slice[k];
        /* MEASURED 2026-08-24: the walk runs in 64 bits in BOTH specialisations,
         * and the legacy 32-bit form is the slow one. pl_next routes every
         * increment through pl_add32_sat -- `if (hi || x > UINT32_MAX - lo)` --
         * which is a BRANCH per increment, up to two per step, in the hottest
         * loop in the program. pl_next64 is a plain 64-bit add: two IADDs, no
         * control flow. Converting the slabbed path to 32-bit cost 6.0% of fill
         * (116.8 -> 123.8 ms/q, n=4 vs n=5, non-overlapping ranges, c147
         * I14/J65536) -- that experiment ran on a CONTENDED card and its
         * absolutes are ~2x the idle ones, so treat the 6.0% as directional
         * only; it was not repeated once the card freed up. Converting the
         * unslabbed path to 64-bit is what actually pays -- but ONLY there.
         * The slabbed path was already 64-bit before this change, so it gains
         * nothing: measured -6.1% on c147 I14/J8192 (unslabbed) against -0.7%
         * on c147 I14/J65536 and +0.3% on c194 I16/J32768 (both slabbed).
         * Every production geometry at I16 and above is slabbed, so this
         * conversion buys real jobs nothing; it is kept because it also removes
         * a duplicated loop and because sub-2^30 geometries do benefit.
         *
         * Correctness of the narrowing, which is the part to read before
         * touching the walk width: local positions stay below 2^31 either way,
         * so (uint32_t)x is exact for the record and the bucket index.
         *
         * What survives from the 32-bit attempt is the shape: one loop here
         * instead of two. An explicit `x >= xmax` early-out does NOT survive --
         * it was load-bearing only while x was 32-bit and could not hold the
         * continuation. With a 64-bit x the loop below simply does not execute
         * and the store after it writes the identical value, so the branch was
         * pure added divergence. (An exhausted walk holds UINT64_MAX and decays
         * by xmax per slab; pre-existing, and stays far above xmax for any
         * reachable slab count.) */
        uint64_t x;
        if constexpr (SLABBED) x = walk_cur[k];   /* may already be past xmax */
        else                   x = pl_first64(&P, logI);
        for (; x < xmax; x = pl_next64(x, &P, Imask)) {
            const uint32_t xl = (uint32_t)x;
            /* --sieve-skip: a position k_intersect_compact's gcd(i,j) test will
             * reject gets no record. Fill is L2-bound with the ALU nearly idle,
             * so the test is free and every skipped record is one write and one
             * cursor atomic fewer. parity(i) == parity(x) because I/2 is even. */
            if (sieve_skip) {
                const uint32_t j = pos_row<SLABBED>(xl, logI, j_base);
                if (pos_both_even(xl, j)) continue;
                if (sieve_skip > 1 && (j % 3u) == 0u && ((xl & Imask) % 3u) == ih3)
                    continue;
            }
            if (fill_gate_drops(gate, gate_shift, xl)) continue;
            const uint32_t b = xl >> log_region;
            const uint32_t slot = atomicAdd(&cursor[b], 1u);
            if (slot >= cap) { atomicAdd(overflow, 1u); continue; }
            const size_t at = ((size_t)b * cap + slot) * RECBYTES;
            if (RECBYTES == 2) {
                *(uint16_t *)(out + at) = (uint16_t)(xl & offmask);
            } else if (RECBYTES == 4) {
                *(uint32_t *)(out + at) = (xl & offmask) | (sl << 16);
            } else {
                *(uint64_t *)(out + at) = (uint64_t)(xl & offmask) | ((uint64_t)sl << 32);
            }
            if constexpr (INDEX) fill_store_index(index, icap, index_ovf, b, slot, ibase + k);
        }
        /* x is already the exact first hit at/after the slab end. */
        if constexpr (SLABBED) walk_next[k] = x - xmax;
    }
}

/* ---- --fill-rows: the smallest bucketed moduli, row by row ------------ *
 *
 * k_fill_atomic gives each modulus ONE thread, and that thread visits all of
 * its xmax/q hits in sequence, each a cursor atomic whose returned slot the
 * store must wait for. The smallest bucketed modulus is ~I, so its thread
 * runs xmax/I = J dependent atomic->store steps while the average thread
 * runs a few hundred: the kernel's tail is a handful of warps on an
 * otherwise idle card. (C143: fill +60% from I14 to I13 for +14% records.)
 *
 * For q >= I a row holds at most one hit, at i + I/2 == (rt*j + I/2) mod q,
 * so the strip can be enumerated by row instead of by FK step, and the rows
 * split into bands that start independently. A warp takes 32 consecutive
 * moduli over one band of rows; its hits in a row land in one or two
 * regions, so one atomic per region per row reserves slots for all of them
 * (CADO's fill_row_kernel does the same). Same records as the FK walk; only
 * their order inside a bucket changes, which k_apply's sum does not see.
 *
 * rts[k] is k_transform's rt_out: the root, or ROWRT_NONE where k_fill_atomic
 * would skip the entry (PL_INVALID). The row tier needs no walk state; when
 * k_resieve_scatter still walks it (--no-resieve-rows), k_rows_walk_next
 * supplies the cursor. j_base counts only under SLABBED, as in pos_row. */
template <bool SLABBED, bool INDEX = false>
__global__ void k_fill_rows(const uint32_t *__restrict primes,
                            const uint32_t *__restrict rts,
                            const uint16_t *__restrict slice,
                            uint32_t n, uint32_t nbands, uint32_t rows,
                            int logI, int log_region,
                            uint32_t *__restrict cursor,
                            uint8_t *__restrict out, uint32_t cap,
                            uint32_t *__restrict overflow,
                            int sieve_skip, uint32_t j_base,
                            const uint32_t *__restrict gate, int gate_shift,
                            uint32_t *__restrict index = nullptr,
                            uint32_t icap = 0,
                            uint32_t *__restrict index_ovf = nullptr)
{
    const uint32_t I = 1u << logI;
    const uint32_t offmask = (1u << log_region) - 1;
    const uint32_t jb = SLABBED ? j_base : 0u;
    const uint32_t ih3 = (I >> 1) % 3u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    /* whole warps only (blockDim % 32 == 0), so this exit is warp-uniform and
     * the full-mask ballot below stays legal */
    if (warp >= (uint64_t)((n + 31u) >> 5) * nbands) return;
    const uint32_t band = (uint32_t)(warp % nbands);
    const uint32_t k = (uint32_t)(warp / nbands) * 32u + lane;
    const uint32_t r0 = (uint32_t)((uint64_t)band * rows / nbands);
    const uint32_t r1 = (uint32_t)((uint64_t)(band + 1) * rows / nbands);

    uint32_t q = 1, rt = 0, sl = 0;
    const bool active = k < n && rts[k] != ROWRT_NONE;
    if (active) { q = primes[k]; rt = rts[k]; sl = slice[k]; }
    uint32_t off = active
        ? (uint32_t)(((uint64_t)rt * (jb + r0) + (I >> 1)) % q) : I;

    for (uint32_t r = r0; r < r1; r++) {
        const uint32_t j = jb + r;
        /* (0,0) is the only lattice point of row 0 and is never sieved */
        bool hit = off < I && j != 0;
        if (hit && sieve_skip) {
            /* row bases are multiples of I, so parity(x) == parity(off) and
             * x & Imask == off: the same tests as k_fill_atomic */
            if (((off | j) & 1u) == 0u) hit = false;
            else if (sieve_skip > 1 && j % 3u == 0u && off % 3u == ih3) hit = false;
        }
        const uint32_t xl = (r << logI) + off;
        if (hit && fill_gate_drops(gate, gate_shift, xl)) hit = false;
        /* every lane runs every row (the band is warp-uniform), so the
         * full-mask ballot is legal */
        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, hit);
        if (hit) {
            const uint32_t b = xl >> log_region;
            uint32_t slot;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
            /* __match_any_sync needs sm_70; before it, one atomic per hit */
            (void)mask; (void)lane;
            slot = atomicAdd(&cursor[b], 1u);
#else
            const uint32_t peers = __match_any_sync(mask, b);
            const uint32_t leader = __ffs(peers) - 1;
            slot = 0;
            if (lane == leader) slot = atomicAdd(&cursor[b], __popc(peers));
            slot = __shfl_sync(peers, slot, leader) +
                   __popc(peers & ((1u << lane) - 1u));
#endif
            if (slot >= cap) atomicAdd(overflow, 1u);
            else {
                *(uint32_t *)(out + ((size_t)b * cap + slot) * 4) =
                    (xl & offmask) | (sl << 16);
                /* the row tier is the factor base's first n entries */
                if constexpr (INDEX) fill_store_index(index, icap, index_ovf, b, slot, k);
            }
        }
        if (active) { off += rt; if (off >= q) off -= q; }
    }
}

/* Slabbed --fill-rows with --no-resieve-rows: k_resieve_scatter replays
 * walk_cur for the row tier as for every other entry, so each needs the first
 * lattice point at or past the slab end, as k_fill_atomic leaves it for the
 * rest. Stepping row by row to it can take ~q - I rows (rt == 1), which is the
 * single-thread tail the tier exists to remove, so pl_rows_to_hit jumps there
 * in O(log q). row_end is the GLOBAL row after this slab. */
__global__ void k_rows_walk_next(const uint32_t *__restrict primes,
                                 const uint32_t *__restrict rts, uint32_t n,
                                 uint32_t row_end, int logI,
                                 const uint64_t *__restrict walk_cur,
                                 uint64_t *__restrict walk_next)
{
    const uint32_t I = 1u << logI;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const uint32_t rt = rts[k];
        if (rt == ROWRT_NONE) { walk_next[k] = walk_cur[k]; continue; }
        const uint32_t q = primes[k];
        const uint32_t off = (uint32_t)(((uint64_t)rt * row_end + (I >> 1)) % q);
        const uint64_t t = pl_rows_to_hit(off, rt, q, I);
        if (t == UINT64_MAX) { walk_next[k] = UINT64_MAX; continue; }
        const uint32_t o = (uint32_t)((off + (uint64_t)rt * (t % q)) % q);
        walk_next[k] = (t << logI) + o;   /* x - xmax, as k_fill_atomic */
    }
}

/* --fill-gate-shift > 0: fold the first side's survivor bitmap to one bit per
 * 2^shift positions, set when any of them survived. A superset of the bitmap,
 * so the gated fill stores somewhat more records than shift 0 would but the
 * gate is 2^shift times smaller to keep in L2. One thread per gate word. */
__global__ void k_gate_fold(const uint32_t *__restrict bits,
                            uint32_t *__restrict gate, int shift,
                            uint32_t ngword, uint32_t nbitword)
{
    const uint32_t span = 1u << shift;          /* bitmap words per gate word */
    const uint32_t per = 32u >> shift;          /* gate bits per bitmap word */
    const uint32_t low = span == 32u ? 0xFFFFFFFFu : (1u << span) - 1u;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t w = bench_grid_thread_x(); w < ngword; w += stride) {
        uint32_t g = 0;
        for (uint32_t k = 0; k < span; k++) {
            /* the last gate word may cover past the bitmap's end: no
             * survivors there, so no gate bits */
            const uint64_t wi = (w << shift) + k;
            const uint32_t word = wi < nbitword ? bits[wi] : 0u;
            for (uint32_t c = 0; c < per; c++)
                if ((word >> (c << shift)) & low) g |= 1u << (k * per + c);
        }
        gate[w] = g;
    }
}

/* ---- small-prime line sieve, fused into apply -------------------------- */

/* Primes below bkthresh are not bucketed: they hit so often that a record per
 * hit costs more than recomputing the entry point per region. On this job they
 * are only 1,969 factor-base entries but **2.92e9 updates per special-q, 9.4x
 * the entire bucket-sieve volume** -- and 84% of that comes from the 52 entries
 * with p < 64. So the load balance across primes, not the update count, is what
 * has to be engineered.
 *
 * The region is chosen to lie inside a single j-row (log_region <= logI), which
 * makes the entry point one multiply and one remainder: within a row, hits are
 * the arithmetic progression i == rt*j (mod p). No walk state has to be carried
 * between regions, so every block is independent.
 *
 * Three tiers, sized so each entry's hit count matches the number of threads
 * assigned to it. The cut is on the entry's effective MODULUS m = q/g, not on
 * the prime: row-divisor and projective entries have a large q and a small m.
 * (Entry counts below are from an early job; c183's algebraic side has 95
 * block-tier entries.)
 *   m <   64  (52 entries, 84% of updates): the whole block
 *   m < 1024  (165 entries):                one warp per entry
 *   m >= 1024 (1752 entries, <=16 hits):    one thread per entry
 * The two lower tiers share each entry's setup across a warp by shuffle
 * (RESULTS findings 101-102); see sieve_small.
 */
#define SS_BLOCK_CUT   64u
#define SS_WARP_CUT  1024u

template <int CELLBITS, int ATOMIC>
__device__ __forceinline__ void ss_add(uint32_t *S, uint32_t c, uint32_t lp)
{
    const uint32_t CPW = 32 / CELLBITS;
    uint32_t v = lp << ((c % CPW) * CELLBITS);
    if (ATOMIC) atomicAdd(&S[c / CPW], v);
    else        S[c / CPW] += v;
}

/* First offset within [0,width) whose i is congruent to rt*j (mod p). TWO
 * selects the two-step reduction, for the thread tier's tail range only (see
 * the magic path below and ss_bound_ok in td.cuh). */
template <bool TWO = false>
__device__ __forceinline__ uint32_t ss_first(uint32_t p, uint32_t rt,
                                             uint32_t j, int32_t ilo,
                                             uint32_t magic, uint32_t kshift)
{
    /* Magic path: no integer division at all, just __umulhi + shift + mad.
     *
     * ONE table lookup, not three. Only `magic` has to be stored -- it needs a
     * 64-bit divide to build, so the host has to. The other two values used to
     * be stored beside it and are now derived here for free, because the extra
     * global loads cost more than the arithmetic they saved (measured: the
     * three-value form left two thirds of the available win on the table,
     * concentrated in the thread tier that has the fewest hits to amortize a
     * load over):
     *
     *   bias  p << SS_KSHIFT(logI), the same macro the host builds its bound
     *         against. Any multiple of p that is >= Ihalf will do -- it only
     *         has to make the numerator non-negative without moving the
     *         residue -- and p >= 2 gives p<<(logI-2) >= 2^(logI-1) == Ihalf.
     *         The stored "least multiple" was never worth a load; neither was
     *         a per-p shift, which is smaller still but costs more per hit
     *         than the coverage is worth (measured; see SS_KSHIFT in td.cuh).
     *   sh    floor(log2 p) == 31 - clz(p), one instruction.
     *
     * Powers of two take an AND instead: td_magic_build has to special-case
     * them (their general-form magic would be 2^32, which does not fit), and
     * the AND is cheaper than the reciprocal anyway. p == 1 never reaches this
     * branch -- ss_magic_build refuses m < 2 outright, so it takes the
     * fallback below (t % 1 == 0). Do not relax that guard on the theory that
     * the AND covers p == 1: td_magic_build's power-of-two case would then
     * evaluate 1u << 32, which is undefined.
     *
     * The uint32 arithmetic is exact because the true value is non-negative
     * and < 2^31; the host refuses the magic (magic == 0) for any entry where
     * it could not prove that, matching tdsmall_t's convention in td.cuh.
     * td_mod_magic is exact only for dividends < 2^31, which is that bound. */
    if (magic) {
        if constexpr (TWO) {
            /* TWO-STEP: the host could not prove rt*j + bias - ilo < 2^31
             * (at I16/J32768, every p above ~43,690), but did prove rt*j and
             * (rt*j mod p) + bias - ilo each are. Same magic, applied twice:
             * a few instructions, against the 64-bit remainder call below
             * that these entries took in every region (findings 110-111). A power
             * of two needs no bound at all -- p divides 2^32, so the AND is
             * exact even if the 32-bit sum wraps. */
            if ((p & (p - 1)) == 0)
                return (rt * j + (p << kshift) - (uint32_t)ilo) & (p - 1);
            const uint32_t sh = 31u - (uint32_t)__clz((int)p);
            const uint32_t r  = td_mod_magic(rt * j, p, magic, sh);
            return td_mod_magic(r + (p << kshift) - (uint32_t)ilo, p, magic, sh);
        }
        const uint32_t w = rt * j + (p << kshift) - (uint32_t)ilo;
        if ((p & (p - 1)) == 0) return w & (p - 1);
        return td_mod_magic(w, p, magic, 31u - (uint32_t)__clz((int)p));
    }
    /* ONE reduction, not two. base == rt*j (mod p), so (base - ilo) mod p is
     * just (rt*j - ilo) mod p -- the intermediate reduction to `base` buys
     * nothing. The bias makes the numerator unsigned so the sign fixup goes
     * too: p<<20 is a multiple of p (leaves the residue alone) and is >= 2^21
     * for any p >= 2, while |ilo| <= 2^(logI-1) <= 2^19 because main() pins
     * logI to [2,20]. No overflow: the uint32 product is < 2^64 and the bias
     * is < 2^52 for any bkthresh a real job sets.
     *
     * Motivation is AMD-specific: there is no integer-divide instruction, so
     * every % here is a software sequence. Removing one is removing real ALU
     * work, not one op. */
    const uint64_t t = (uint64_t)rt * j + ((uint64_t)p << 20) - (uint64_t)(int64_t)ilo;
    return (uint32_t)(t % p);
}

/* ss_magic_build (the host side of the above) and SS_KSHIFT live in td.cuh,
 * next to the td_magic_build they call -- shared by both builds rather than
 * forked, and reachable from pipeline.cuh without depending on include order. */

/* One entry's per-row setup, shared by all three tiers of sieve_small: returns
 * the first hit c0 (< 2m) and sets *mm_out to the step, or to 0 when the entry
 * has no hits in this row -- g does not divide j, or under --sieve-skip an even
 * modulus whose hits in an even row all fall on even i. */
template <bool TWO = false>
__device__ __forceinline__ uint32_t ss_setup(uint32_t e,
                                             const uint32_t *__restrict sp,
                                             const uint32_t *__restrict srt,
                                             const uint32_t *__restrict sg,
                                             const uint32_t *__restrict smag,
                                             uint32_t j, int32_t ilo,
                                             uint32_t kshift, int evenrow,
                                             uint32_t *mm_out)
{
    /* All four loads before the g test, so they go out together: an entry
     * costs one L2 round trip, not two. The tables do not fit in the L1 left
     * beside apply's shared memory, and the thread tier pays this once per
     * entry per region (finding 110). An entry with g not dividing j loads
     * two words it does not use; they share its lines anyway. */
    const uint32_t m = sp[e], g = sg[e], rt = srt[e];
    const uint32_t mg = smag ? smag[e] : 0u;
    *mm_out = 0;
    if (g > 1 && (j % g)) return 0;
    uint32_t c0 = ss_first<TWO>(m, rt, g > 1 ? j / g : j, ilo, mg, kshift);
    uint32_t mm = m;
    if (evenrow) {
        if (m & 1u) { if (!(c0 & 1u)) c0 += m; mm = 2u * m; }
        else if (!(c0 & 1u)) return 0;
    }
    *mm_out = mm;
    return c0;
}

/* The tier split, for BOTH callers (the pipeline and run_bench). hsp must be
 * sorted by modulus; the counts are the leading runs below each cut, which is
 * what sieve_small's packing relies on: every e < nblk has m < SS_BLOCK_CUT
 * and every e < nwrp has m < SS_WARP_CUT. One copy, so the two callers cannot
 * drift apart and hand the block tier a modulus its 8-bit fields cannot hold. */
static inline void ss_tiers(const uint32_t *hsp, uint32_t n,
                            uint32_t *nblk, uint32_t *nwrp)
{
    uint32_t b = 0, w = 0;
    while (b < n && hsp[b] < SS_BLOCK_CUT) b++;
    w = b;
    while (w < n && hsp[w] < SS_WARP_CUT) w++;
    *nblk = b; *nwrp = w;
}

template <int CELLBITS, int ATOMIC, bool SLABBED = false, bool SSTAIL = true>
__device__ void sieve_small(uint32_t *S, uint32_t region, int logI, int log_region,
                            const uint32_t *__restrict sp,
                            const uint32_t *__restrict srt,
                            const uint32_t *__restrict sg,
                            const uint16_t *__restrict slp,
                            const uint32_t *__restrict smag,
                            uint32_t nsmall, uint32_t nblk, uint32_t nwrp,
                            uint32_t nsplit,
                            uint32_t tid, uint32_t nth, uint32_t j_base,
                            int skipeven)
{
    const uint32_t width = 1u << log_region;
    const uint32_t x0    = region << log_region;
    const uint32_t jlocal = x0 >> logI;
    uint32_t j = jlocal;
    if constexpr (SLABBED) j += j_base;
    const int32_t  ilo   = (int32_t)(x0 & ((1u << logI) - 1)) - (int32_t)(1u << (logI - 1));
    /* Loop-invariant, so it is derived once here rather than per hit: the
     * host proves its bound against SS_KSHIFT(logI) too, so both sides read
     * the same definition. */
    const uint32_t kshift = SS_KSHIFT(logI);
    const uint32_t warp = tid >> 5, lane = tid & 31, nwarps = nth >> 5;
    /* --sieve-skip: in an even row only odd i can ever be a relation -- the
     * row half of pos_both_even -- and parity(c) == parity(i) because ilo is
     * even (x0 is even, I/2 is even). An odd modulus alternates parity, so
     * start at the first odd hit and step 2m; an even one never changes it, so
     * the entry either keeps every hit or contributes nothing to this row.
     * 2m cannot wrap: bench_main caps --bkthresh, and so every line-sieved
     * modulus, at 2^30. */
    const int evenrow = skipeven && !(j & 1u);

    /* Callers launch whole warps -- bench_main refuses an --apply-threads that
     * is not a multiple of 32 -- and every tier below depends on it: the two
     * shuffle tiers need full warps, and the warp tier's stride nth >> 5 would
     * be 0 otherwise. */

    /* Every entry is (m, rt, g): hits this row only when g | j, and then at
     * i == rt*(j/g) (mod m). g == 1 is the ordinary case; m == 1 means every
     * position in the row. See pl_transform_gen. ss_setup is that per-row
     * setup, shared by all three tiers so they cannot sieve different position
     * sets. */
    #define SS_SETUP(e, MM) ss_setup((e), sp, srt, sg, smag, j, ilo, kshift, evenrow, &(MM))

    /* WHOLE-BLOCK TIER (m < SS_BLOCK_CUT), setup shared by shuffle. An entry's
     * setup is the same for every thread in the block, and the per-warp form
     * computed it in all nth/32 warps for every entry: ncu on the c183
     * algebraic side (95 entries) put that replicated setup at ~19% of
     * k_apply's samples, against ~15% for the hits (finding 101). Lane k of
     * each warp now sets up entry base+k and the warp walks those 32 entries
     * with ONE __shfl_sync each: ceil(nblk/32) setups per warp, not nblk.
     *
     * One register carries it all: m < SS_BLOCK_CUT <= 128 bounds c0 (< 2m
     * after the even-row shift) and the step mm (<= 2m) to 8 bits each, and the
     * log is 16. mm == 0 means "no hits in this row". The host guarantees
     * m < SS_BLOCK_CUT for every e < nblk (ss_tiers); the static_assert keeps
     * the constant inside the packing. Cell sums are integer adds, so the
     * reordering is output-identical.
     *
     * Leftover hits always land on the lowest tids, so warp 0 runs the most
     * iterations of each entry. Rotating the start thread per entry evens that
     * out and was measured SLOWER (apply +3.6%, finding 102): leave it. */
    static_assert(SS_BLOCK_CUT <= 128u, "block-tier fields are packed in 8 bits");
    for (uint32_t base = 0; base < nblk; base += 32u) {
        const uint32_t e = base + lane;
        uint32_t pk = 0;
        if (e < nblk) {
            uint32_t mm;
            const uint32_t c0 = SS_SETUP(e, mm);
            if (mm) pk = c0 | (mm << 8) | ((uint32_t)slp[e] << 16);
        }
        const uint32_t ne = (nblk - base < 32u) ? nblk - base : 32u;
        for (uint32_t k = 0; k < ne; k++) {
            const uint32_t q  = __shfl_sync(0xFFFFFFFFu, pk, k);
            const uint32_t mm = (q >> 8) & 0xFFu;
            if (!mm) continue;
            const uint32_t lp = q >> 16;
            for (uint32_t c = (q & 0xFFu) + tid * mm; c < width; c += nth * mm)
                ss_add<CELLBITS,ATOMIC>(S, c, lp);
        }
    }

    /* WARP TIER (SS_BLOCK_CUT <= m < SS_WARP_CUT), setup shared by shuffle
     * too. Warp w still owns entries nblk + w + i*nwarps, but instead of
     * running each entry's setup as a whole-warp sequence for its 0.5-8 hits
     * per lane, lane k sets up the warp's k-th next entry and the warp walks
     * them: apply −7.0% on c183 (finding 102). c0 and mm are < 2m < 2^16. */
    {
        static_assert(SS_WARP_CUT <= 32768u, "warp-tier c0/mm are packed in 16 bits");
        for (uint32_t base = 0; ; base += 32u) {
            const uint32_t e0 = nblk + warp + base * nwarps;
            if (e0 >= nwrp) break;
            const uint32_t e = e0 + lane * nwarps;
            uint32_t pk = 0, lp = 0;
            if (e < nwrp) {
                uint32_t mm;
                const uint32_t c0 = SS_SETUP(e, mm);
                if (mm) { pk = c0 | (mm << 16); lp = slp[e]; }
            }
            const uint32_t left = (nwrp - e0 + nwarps - 1) / nwarps;
            const uint32_t ne = left < 32u ? left : 32u;
            for (uint32_t k = 0; k < ne; k++) {
                const uint32_t q  = __shfl_sync(0xFFFFFFFFu, pk, k);
                const uint32_t kl = __shfl_sync(0xFFFFFFFFu, lp, k);
                const uint32_t mm = q >> 16;
                if (!mm) continue;
                for (uint32_t c = (q & 0xFFFFu) + lane * mm; c < width; c += 32u * mm)
                    ss_add<CELLBITS,ATOMIC>(S, c, kl);
            }
        }
    }

    /* THREAD TIER (m >= SS_WARP_CUT): one thread per entry, so the setup is
     * already lane-parallel. The log is loaded with the setup's four words,
     * not after c0 is known: three dependent L2 round trips per entry became
     * one. At I16 this tier has ~6,500 entries per side, each paid in every
     * quarter-row region; the hoist is apply -2.7% there, -2.0% at c183, and
     * no more because apply is issue-bound and co-resident blocks were
     * already hiding most of the wait (finding 110).
     *
     * THE TWO-STEP TAIL. Entries at or past nsplit (ss_two_step_from) are
     * the large moduli whose one-step numerator the host could not bound --
     * at I16/J32768 every m above ~43,690 -- and take the two-step setup
     * instead of the 64-bit fallback. SSTAIL is a template constant: false
     * (no tail, nsplit == nsmall: every geometry up to logI 15 at the default
     * J) folds the select away and runs the one-step loop alone; true tests
     * e >= nsplit per entry, which is warp-uniform except in the one warp
     * that straddles it.
     *
     * One thread-tier loop per kernel instance, ON PURPOSE. Two loops in one
     * kernel -- back to back over [nwrp, nsplit) and [nsplit, nsmall), or as
     * the two arms of a runtime branch -- made ptxas compile one of them
     * without reconvergence (a YIELD and a plain BSSY in its hit loop): lanes
     * drifted apart and the block waited at the scan's barrier, apply +24%
     * at I16 for the same instructions. And testing e < nsplit in the I15
     * loop, where it is always true, cost apply +0.8% at c183 (finding 111).
     * An edit here should recount YIELD/BSSY per k_apply instance
     * (oracle/ss-two-step-20261014/sass-structure.txt).
     *
     * The one-step instance TRAPS on a table with a tail. Its entries past
     * nsplit carry magics proved only for the two-step form, and the
     * one-step reduction would give them silently wrong residues
     * (ss_magic_build); a launch site that skipped ss_needs_tail must fail
     * loudly instead. */
    if constexpr (!SSTAIL) {
        if (nsplit < nsmall) __trap();
    }
    for (uint32_t e = nwrp + tid; e < nsmall; e += nth) {
        const uint32_t lp = slp[e];
        uint32_t mm;
        const uint32_t c0 = (SSTAIL && e >= nsplit)
            ? ss_setup<true>(e, sp, srt, sg, smag, j, ilo, kshift, evenrow, &mm)
            : SS_SETUP(e, mm);
        if (!mm) continue;
        for (uint32_t c = c0; c < width; c += mm)
            ss_add<CELLBITS,ATOMIC>(S, c, lp);
    }
    #undef SS_SETUP
}

/* ---- stage (A): apply -- accumulate logs into a shared-memory region ---- */

/* One position's survivor test. Factored out because the threshold scan below
 * has two shapes (warp-ballot and the small-region fallback) and a predicate
 * that drifts between them would produce results that depend on --region and
 * --apply-threads -- which the --verify gate does not vary, since it compares
 * CELLS rather than survivor counts.
 *
 * las's `not_both_even` filter: x = j*I + (i + I/2) and I/2 is a power of two
 * >= 2, so parity(i) == parity(x) and parity(j) == parity(x >> logI). Both even
 * means (a,b) are both even and the relation is a duplicate of (a/2, b/2); las
 * marks these 255 so they can never survive. Off by default so that every
 * survivor count recorded before 2026-08-03 still reproduces -- except that
 * k_apply ORs in its sieve_skip argument, because the cells --sieve-skip
 * leaves unsieved must never be scanned in. */
/* The per-cell side effects that accompany the predicate: the gate-5 probe and
 * the optional dump byte. They live here for the same reason apply_keep does --
 * the threshold scan has two shapes and anything duplicated between them drifts.
 * Splitting the predicate out but leaving these inline was the first version of
 * this refactor and it left both explanatory comments stranded in the fallback,
 * which is the branch that essentially never runs. */
__device__ __forceinline__ void apply_cell_side_effects(
        uint32_t x, uint32_t v, uint32_t CINIT,
        uint32_t probe_x, uint32_t *__restrict probe_out,
        uint8_t *__restrict dump)
{
    /* Gate 5's GPU half: this value has been through the root transform, the FK
     * walk, the tiering, the bucket fill, the small sieve and this kernel's
     * apply. probe_out[0] is the initialised norm, so the sieved log sum the
     * PIPELINE actually produced is v - (CINIT - probe_out[0]). */
    if (probe_out && x == probe_x) probe_out[1] = v;
    /* las's byte: S = max(T - sum(logp), 0), and our cell holds
     * CINIT - T + sum, so S = CINIT - cell. */
    if (dump) {
        int32_t sv = (int32_t)CINIT - (int32_t)v;
        dump[x] = (uint8_t)(sv < 0 ? 0 : (sv > 255 ? 255 : sv));
    }
}

/* j is x's global row. The packed scan computes it once per 64 positions,
 * which is why the row is a parameter rather than derived here. */
__device__ __forceinline__ int apply_keep_row(uint32_t x, uint32_t j, uint32_t v,
                                              uint32_t THRESH, int not_both_even)
{
    return (v >= THRESH) && !(not_both_even && pos_both_even(x, j));
}

template <bool SLABBED>
__device__ __forceinline__ int apply_keep(uint32_t x, uint32_t v,
                                          uint32_t THRESH, int logI,
                                          uint32_t j_base, int not_both_even)
{
    return apply_keep_row(x, pos_row<SLABBED>(x, logI, j_base), v, THRESH,
                          not_both_even);
}

/* Whether k_apply's threshold scan takes its store path, which writes EVERY
 * survbits word of the regions it covers with a plain store, so a caller
 * whose launch covers the whole bitmap need not clear it first. One
 * definition, read by the kernel to pick its path and by the pipeline to
 * skip the clear, so the two cannot disagree. 16-bit cells take 64 positions
 * per warp step (two survbits words), 8-bit cells 32. The logI test is the
 * packed scan's: it computes one row per 64 positions, which is exact only
 * when 64 aligned positions never straddle a row. */
template <int CELLBITS>
__host__ __device__ __forceinline__ bool apply_scan_stores(int logI, int log_region,
                                                           uint32_t nth)
{
    const int min_region = (CELLBITS == 16) ? 6 : 5;
    return log_region >= min_region && (CELLBITS != 16 || logI >= 6)
        && nth >= 32u && (nth & 31u) == 0u;
}

/* One block owns one bucket region for its entire life: it initialises the
 * cells to the log-norm bound in shared memory, accumulates every log p that
 * landed in the region, scans for survivors, and writes only the survivors
 * back. The region itself never touches global memory in either direction.
 * That is the structural advantage over CPU las, which must stream the region
 * through cache.
 *
 * Sign convention. GPU atomics are add-only, and a 16-bit half-word subtract
 * would borrow into its neighbour. So instead of "start at the norm and
 * subtract logs until small", we start at CINIT - T(x) and add, where
 * T(x) = log2|F(a,b)| - log2(q) - allowance. A position is a survivor exactly
 * when its cell reaches CINIT. Identical test, no borrow, no CAS.
 *
 * CELLBITS 16 is the doc's recommendation and is exactly correct: an
 * accumulated log cannot reach 65536. CELLBITS 8 packs four cells per word,
 * halving the shared memory a region needs -- it is measured here only to
 * price what correctness costs, since a byte cell overflows into its
 * neighbour (accumulated logs do exceed 255) and cannot be used for real.
 */
template <int CELLBITS, int ATOMIC, int NORMMODE, bool SLABBED = false,
          bool SSTAIL = true>
/* __launch_bounds__(512, 3): ptxas otherwise settles on 45-46 registers, which
 * fits only TWO 512-thread blocks per SM and pins theoretical occupancy at
 * 66.67%. ncu on a 5070 (sm_120) reports the limiter explicitly -- Block Limit
 * Registers 2 against Block Limit Shared Mem 3 and Block Limit Warps 3 -- so
 * registers, not the 32 KB of cells, are what cost the third block. Measured
 * paired and idle: apply -12.6%, wall -4.6% on the C194 (finding 75).
 *
 * **The budget is 40 registers, not the 42 the naive division gives.** Three
 * blocks need 65536/(512*3) = 42.67 per thread, but registers are allocated
 * per warp in units of 256 -- 8 per thread -- so a 42-register kernel is
 * charged 48, giving 65536/(48*512) = 2.67 -> two blocks and no gain at all.
 * 40 is the first value that actually yields three, and it is where ptxas
 * lands. Anyone reusing this recipe on another kernel must round DOWN to a
 * multiple of 8 before believing the arithmetic.
 *
 * ptxas lands on 40 with NO spill for all nine instantiations, on sm_120 and
 * sm_80 alike; total spill bytes across the whole TU stay at 0, as at HEAD.
 * This is the mirror image of k_fill_l1, where the same annotation had to be
 * REVERTED because a widened walk pushed it past the cap into 12 spill stores:
 * check `-Xptxas -v` before assuming a cap is free, in both directions.
 *
 * **VALIDITY WINDOW -- the three-block premise is not universal.** Apply's
 * dynamic shared memory is (1 << log_region) * 2 + nslice_pow2 * 2, and three
 * blocks must fit the 100 KB per-SM budget. At the default --region 14 that
 * allows nslice_pow2 <= 512, which is exactly where the C194 sits today
 * (33792 B/block): one more slice tier doubles it to 1024, smem/block goes to
 * 34816, and only two blocks fit -- the whole -12.6% evaporates silently, with
 * no diagnostic. At --region 15 (~66 KB/block) only ONE block ever fits, so
 * the cap constrains ptxas for no occupancy gain there. It costs nothing in
 * either case, since there is no spill at 40, but do not assume the win
 * survives a larger region or a factor base that cuts to more slices --
 * re-measure occupancy with ncu if either changes.
 *
 * The 512 is a hard ceiling, not a hint: a launch with more than 512 threads
 * per block fails outright, which is why --apply-threads is capped at
 * APPLY_THREADS_MAX rather than the 1024 a block could otherwise carry. */
__global__ __launch_bounds__(512, 3)
void k_apply(const uint32_t *__restrict buckets,
                        const uint32_t *__restrict cnt, uint32_t cap,
                        int logI, int log_region,
                        const uint16_t *__restrict slice_logp, uint32_t nslice,
                        norm_t N, uint32_t CINIT, uint32_t THRESH, uint32_t tconst,
                        uint8_t *__restrict dump,
                        uint32_t *__restrict nsurv,
                        uint16_t *__restrict dbg_cells,
                        uint32_t dbg_region,
                        const uint32_t *__restrict sp, const uint32_t *__restrict srt,
                        const uint32_t *__restrict sg, const uint16_t *__restrict slp,
                        const uint32_t *__restrict smag,
                        uint32_t nsmall, uint32_t nblk, uint32_t nwrp,
                        uint32_t nsplit,
                        uint32_t probe_x, uint32_t *__restrict probe_out,
                        uint32_t *__restrict survbits, int not_both_even,
                        int sieve_skip, uint32_t j_base)
{
    extern __shared__ uint32_t sm[];
    const uint32_t ncell  = 1u << log_region;
    const uint32_t nword  = ncell / (32 / CELLBITS);
    const uint32_t CPW    = 32 / CELLBITS;          /* cells per 32-bit word */
    uint32_t *S   = sm;
    uint16_t *lut = (uint16_t *)(sm + nword);

    const uint32_t b = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = 1 << (logI - 1);
    const uint32_t xbase = b << log_region;
    /* --sieve-skip: both-even cells get no norm, no small-prime hits and are
     * never scanned in. The region lies in one row -- bench_main refuses
     * --region > --logI under --pipeline -- so the row is one value per block. */
    const uint32_t jblk = pos_row<SLABBED>(xbase, logI, j_base);
    const int drop_both_even = not_both_even || sieve_skip;

    for (uint32_t i = tid; i < nslice; i += nth) lut[i] = slice_logp[i];

    /* ---- init: log-norm bound, computed in shared memory ---- */
    for (uint32_t w = tid; w < nword; w += nth) {
        uint32_t word = 0;
        #pragma unroll
        for (uint32_t c = 0; c < CPW; c++) {
            uint32_t t;
            const uint32_t x = xbase + w * CPW + c;
            /* Warp-uniform: cell c has the same parity in every word, because
             * xbase and CPW are even. */
            if (sieve_skip && pos_both_even(x, jblk)) {
                t = CINIT;                       /* cell starts at 0 */
            } else if (NORMMODE == NORM_CONST) {
                t = tconst;
            } else {
                const int32_t  ii = (int32_t)(x & Imask) - Ihalf;
                const uint32_t jlocal = x >> logI;
                uint32_t jj = jlocal;
                if constexpr (SLABBED) jj += j_base;
                const float fi = (float)ii;
                const float fj = (float)jj;
                const float u = fmaf(N.ua, fi, N.ub * fj);
                const float v = fmaf(N.va, fi, N.vb * fj);
                float acc = N.d[N.deg], vp = 1.0f;
                /* The same Horner on |.|, which bounds the cancellation. The
                 * normalisation gives fp32 ample dynamic RANGE, but near a root
                 * line of F the terms are O(1) and cancel to something tiny, so
                 * what is left is mostly rounding noise. Measured before this
                 * guard: 144 of 63,497 positions in a band along the three real
                 * root lines rounded to the wrong sieve-log value, -3.31 to
                 * +2.57 units and BOTH SIGNS -- false survivors and lost
                 * relations alike. */
                float aabs = fabsf(N.d[N.deg]), vpa = 1.0f;
                const float au = fabsf(u), av = fabsf(v);
                #pragma unroll
                for (int k = BENCH_MAX_DEGREE - 1; k >= 0; k--)
                    if (k < N.deg) {
                        vp  *= v;  acc  = fmaf(acc,  u,  N.d[k] * vp);
                        vpa *= av; aabs = fmaf(aabs, au, fabsf(N.d[k]) * vpa);
                    }
                float s = fabsf(acc);
                if (s < NORM_CANCEL_TOL * aabs) {
                    /* redo in fp64: (a,b) exactly in int64, sum at 2^-53.
                     * Doubles are 1/64 rate on this part, but this fires on
                     * well under one cell in a thousand. */
                    const double a = (double)((int64_t)ii * N.a0 + (int64_t)jj * N.b0);
                    const double b = (double)((int64_t)ii * N.a1 + (int64_t)jj * N.b1);
                    const double ud = a / N.A, vd = b / N.B;
                    double accd = N.dd[N.deg], vpd = 1.0;
                    #pragma unroll
                    for (int k = BENCH_MAX_DEGREE - 1; k >= 0; k--)
                        if (k < N.deg) { vpd *= vd; accd = accd * ud + N.dd[k] * vpd; }
                    s = (float)fabs(accd);
                }
                /* las: S = fb_log(|F|) = floor(log2|F| * scale + 0.5).
                 * las clamps here at 255 because its cell IS a byte, and that
                 * is the whole reason `scale` exists (1.28 * 196.61 = 251.7).
                 * Our cell is 16 bits, so the ceiling is CINIT, not 255, and
                 * scale becomes a free parameter rather than a constraint --
                 * at scale 1.28 las discards 0.39 bits of resolution per
                 * position that we can keep. Clamping at 255 threw that away
                 * and, worse, would have silently flattened every norm above
                 * 255/scale into one bucket the moment anyone raised scale. */
                /* log2f, NOT __log2f. The fast intrinsic is ~2 ulp and has no
                 * host equivalent, so the CPU replay could not mirror this
                 * expression exactly and "0 cells differ" was measuring two
                 * slightly different functions. Accurate log2f is reproducible
                 * on both sides; norm init was 1.76 ms of a 27 ms apply, so the
                 * difference is affordable and correctness is not. */
#ifdef NORM_FAST_LOG2
#warning "NORM_FAST_LOG2: __log2f breaks CPU parity; --verify and dumpcmp will \
report spurious cell mismatches. Pricing builds only, never for relations."
                /* PRICING SWITCH, not a shipping option: __log2f is one MUFU
                 * instruction against log2f's software sequence, and the point
                 * of building with it is to measure what the accurate version
                 * costs -- see the paragraph above for why the accurate one is
                 * the default.
                 *
                 * This SILENTLY INVALIDATES THE PARITY GATE, which is why the
                 * #warning above exists. poly.c's norm_target_host mirrors the
                 * device expression deliberately -- same float type, same
                 * clamp, same log2f -- so that "0 cells differ" means
                 * something. Building with this breaks the mirror on the
                 * device side only, and the ~2 ulp difference shows up as
                 * scattered mismatches at threshold boundaries that read
                 * exactly like a kernel regression. */
                float lg = N.scale * (N.log2M + __log2f(fmaxf(s, 1e-30f)) - N.bias);
#else
                float lg = N.scale * (N.log2M + log2f(fmaxf(s, 1e-30f)) - N.bias);
#endif
                int ti = (int)floorf(lg + 0.5f);
                const uint32_t TMAX = (CELLBITS == 8) ? 255u : CINIT;
                t = (ti < 0) ? 0u : ((uint32_t)ti > TMAX ? TMAX : (uint32_t)ti);
            }
            if (probe_out && x == probe_x) probe_out[0] = t;
            word |= (CINIT - t) << (c * CELLBITS);
        }
        S[w] = word;
    }
    __syncthreads();

    /* ---- small primes: line-sieved straight into the same shared region ---- */
    if (nsmall)
        sieve_small<CELLBITS, ATOMIC, SLABBED, SSTAIL>(S, b, logI, log_region, sp, srt, sg, slp,
                                               smag,
                                               nsmall, nblk, nwrp, nsplit, tid, nth,
                                               j_base,
                                               sieve_skip);

    /* ---- apply ---- */
    uint32_t n = cnt[b];
    if (n > cap) n = cap;
    for (uint32_t i = tid; i < n; i += nth) {
        const uint32_t r = buckets[(size_t)b * cap + i];
        const uint32_t off = r & (ncell - 1);
        const uint32_t lp  = lut[(r >> 16) & (nslice - 1)];
        const uint32_t w   = off / CPW;
        const uint32_t sh  = (off % CPW) * CELLBITS;
        if (ATOMIC) atomicAdd(&S[w], lp << sh);
        else        S[w] += lp << sh;               /* racy: speed-of-light probe */
    }
    __syncthreads();

    /* ---- threshold scan ----
     *
     * A WARP covers 32 consecutive positions per ballot, which is exactly one
     * survbits word, so the bitmap is written with ONE unconditional 32-bit
     * store per group instead of one atomicOr per survivor. (16-bit cells take
     * two ballots per step, positions k and k + 32 in lane k; see below.)
     * Adjacent lanes read the same shared word (CPW cells share a word), so
     * the read broadcasts rather than conflicting.
     *
     * MEASURED idle, paired, relations byte-identical throughout:
     *
     *   c147 I14/J65536  n=3   apply 83.3  -> 72.5   -12.9%
     *   c147 I14/J8192   n=3   apply 10.54 -> 9.13   -13.4%
     *   c194 I16/J32768  n=7   apply 158.1 -> 149.6   -5.4%
     *
     * **c194 is the representative job and it gets less than half the c147
     * win.** The gap is unexplained: c194 has 2x the area and 7x the two-sided
     * survivors yet a smaller ABSOLUTE saving (8.00 vs 10.78 ms), so the win
     * tracks neither positions nor survivors. Do not quote -13% as the figure.
     * (An earlier contended-card pass reported -15.0%/-9.8%; superseded.)
     *
     * ncu profiling on 2026-08-25 (finding 75) ruled out the obvious causes
     * without replacing them: all three geometries profile nearly identically
     * (DRAM 8.5-8.9%, IPC 2.88-2.91, occupancy 66.3-66.5%) and per-launch
     * instruction counts differ by 3% for the same 2^29 positions. It also
     * showed this scan is only ~3% of the kernel's instructions, so the win
     * was contention on the global atomics, not instruction count -- there is
     * no more to get by shaving the scan itself.
     *
     * The store is only race-free across blocks when a block owns whole
     * survbits words -- log_region >= 5, and >= 6 for the packed 16-bit scan,
     * whose step is two words -- so smaller regions keep the atomicOr form.
     * (The full-warp half of the guard is belt-and-braces: bench_main
     * already rejects an --apply-threads that is not a multiple of 32, but this
     * kernel is also launched directly by the standalone harness.) The whole
     * condition is apply_scan_stores, which the pipeline also reads to skip
     * its clear: the fast path makes the clear redundant, the fallback does
     * not. The standalone harness still clears unconditionally.
     *
     * **CALLER CONTRACT, unenforceable from inside the kernel.** The fast path
     * OVERWRITES survbits words rather than OR-ing into them, which is correct
     * only when (a) blockIdx.x maps 1:1 onto regions, so `xbase = b <<
     * log_region` makes each block's words private, and (b) exactly one launch
     * writes a given bitmap. Both callers satisfy this today (`<<<nregion,
     * athr>>>`, one launch per side per slab). A caller that grid-strides
     * regions -- the pattern k_purge uses -- or that accumulates two slabs into
     * one bitmap would get a bitmap where only the last-written words are
     * correct, and only when log_region >= 5, so it would pass --region 4 and
     * fail the default --region 14. The kernel cannot check this: it never
     * sees nregion.
     *
     * nsurv is accumulated per warp and committed once. The old per-survivor
     * atomicAdd had its RETURN VALUE consumed by surv[at] = x, which forces a
     * serialising atom.global.add on one address; the list it indexed was
     * write-only in both callers and is gone, so what is left is a single
     * non-returning red.global.add per warp.
     */
    const uint32_t CMASK = (CELLBITS == 16) ? 0xFFFFu : 0xFFu;
    if (apply_scan_stores<CELLBITS>(logI, log_region, nth)) {
        const uint32_t lane = tid & 31u;
        uint32_t nlocal = 0;
        if constexpr (CELLBITS == 16) {
            /* TWO cells per lane, 32 positions apart: lane k takes positions
             * k and k + 32 of a 64-position step, so each ballot is one
             * survbits word as it stands and lanes 0 and 1 store the pair.
             * Lane k's cells sit in the same half of their words (k & 1) in
             * every step, so the shift is fixed per lane, and the row is one
             * value per step (apply_scan_stores requires logI >= 6). One cell
             * per lane spent 14-20% of this kernel's instructions at I16
             * (finding 110); this is half the steps and half the per-step
             * overhead. */
            const uint32_t ngrp = ncell >> 6;
            const uint32_t sh   = (lane & 1u) * 16u;
            /* Only the standalone harness passes either. Tested once here, the
             * per-cell pointer and position tests leave the step: they were
             * ~11 of its ~51 warp instructions. */
            const bool side_fx = probe_out || dump;
            for (uint32_t g = tid >> 5; g < ngrp; g += (nth >> 5)) {
                const uint32_t xg = xbase + (g << 6);
                const uint32_t j  = pos_row<SLABBED>(xg, logI, j_base);
                const uint32_t xa = xg + lane, xb = xa + 32u;
                const uint32_t wa = (g << 5) + (lane >> 1);
                const uint32_t va = (S[wa] >> sh) & CMASK;
                const uint32_t vb = (S[wa + 16u] >> sh) & CMASK;
                const uint32_t ma = __ballot_sync(0xFFFFFFFFu,
                    apply_keep_row(xa, j, va, THRESH, drop_both_even));
                const uint32_t mb = __ballot_sync(0xFFFFFFFFu,
                    apply_keep_row(xb, j, vb, THRESH, drop_both_even));
                /* The masks are warp-uniform, so every lane counts both and
                 * lane 0 commits; the store is predicated, not a branch (a
                 * lane-0/1 branch cost a BSSY/BSYNC pair per step). */
                nlocal += __popc(ma) + __popc(mb);
                if (survbits && lane < 2u)
                    survbits[(xg >> 5) + lane] = lane ? mb : ma;
                if (side_fx) {
                    apply_cell_side_effects(xa, va, CINIT, probe_x, probe_out, dump);
                    apply_cell_side_effects(xb, vb, CINIT, probe_x, probe_out, dump);
                }
            }
        } else {
            const uint32_t ngrp = ncell >> 5;
            for (uint32_t g = tid >> 5; g < ngrp; g += (nth >> 5)) {
                const uint32_t off = (g << 5) + lane;
                const uint32_t x   = xbase + off;
                const uint32_t v   =
                    (S[off / CPW] >> ((off % CPW) * CELLBITS)) & CMASK;
                const int keep = apply_keep<SLABBED>(x, v, THRESH, logI,
                                                     j_base, drop_both_even);
                const uint32_t mask = __ballot_sync(0xFFFFFFFFu, keep);
                if (lane == 0) {
                    if (survbits) survbits[x >> 5] = mask;
                    nlocal += __popc(mask);
                }
                apply_cell_side_effects(x, v, CINIT, probe_x, probe_out, dump);
            }
        }
        if (lane == 0 && nlocal) atomicAdd(nsurv, nlocal);
    } else {
        /* Fallback: the store path's launch-shape condition fails
         * (apply_scan_stores). survbits words may then be shared between
         * blocks, so the store has to be an atomicOr into a pre-cleared
         * bitmap. */
        for (uint32_t w = tid; w < nword; w += nth) {
            const uint32_t word = S[w];
            #pragma unroll
            for (uint32_t c = 0; c < CPW; c++) {
                const uint32_t v = (word >> (c * CELLBITS)) & CMASK;
                const uint32_t x = xbase + w * CPW + c;
                if (apply_keep<SLABBED>(x, v, THRESH, logI, j_base,
                                        drop_both_even)) {
                    atomicAdd(nsurv, 1u);
                    if (survbits) atomicOr(&survbits[x >> 5], 1u << (x & 31u));
                }
                apply_cell_side_effects(x, v, CINIT, probe_x, probe_out, dump);
            }
        }
    }

    /* ---- optional: dump one region for the CPU cross-check ---- */
    if (dbg_cells && b == dbg_region && CELLBITS == 16)
        for (uint32_t w = tid; w < nword; w += nth) {
            dbg_cells[2 * w]     = (uint16_t)(S[w] & 0xFFFFu);
            dbg_cells[2 * w + 1] = (uint16_t)(S[w] >> 16);
        }
}

/* ---- stage (c) level 1: shared-memory staged split into super-buckets -- */

/* NBUF open output streams per block. The doc's rule: fan-out per pass is
 * bounded by how many buffers fit in shared memory, and each flush is one
 * atomic and one full-cache-line store, not one atomic per record. */
#define L1_NBUF   128
#define L1_CAP     64        /* uint32 slots = 256 B; 128*64*4 = 32 KB static smem */
#define L1_RUNMAX   8        /* max positions aggregated per reservation   */
#define L1_FLUSH  (L1_CAP - L1_RUNMAX)

/* Run-aggregated variant. A (p,r) walk emits positions in monotone
 * increasing order, so consecutive hits usually land in the SAME
 * super-bucket -- for the smallest bucket-sieved primes, which produce most
 * of the volume, many in a row do. Collect a run and reserve it with ONE
 * shared atomic, which also cuts the number of block barriers by the same
 * factor. The barrier-per-record version this replaces spent 30% of its
 * issue slots stalled on barriers at 7.5% of DRAM peak. */
__global__ __launch_bounds__(512, 3)   /* 3 x 33 KB = 99 KB of 100 KB; 1536 thr = 100% occ */
void k_fill_l1(const plat_t *__restrict plat,
               uint32_t n, uint32_t xmax, int logI, int log_super,
               uint32_t *__restrict cursor,
               uint32_t *__restrict out, uint32_t cap,
               uint32_t *__restrict overflow)
{
    __shared__ uint32_t buf[L1_NBUF * L1_CAP];
    __shared__ uint32_t cnt[L1_NBUF];
    __shared__ uint32_t base[L1_NBUF];

    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;

    for (uint32_t i = tid; i < L1_NBUF; i += nth) cnt[i] = 0;
    __syncthreads();

    const uint64_t stride = bench_grid_product_u64(gridDim.x, nth);
    uint64_t k = bench_grid_product_u64(blockIdx.x, nth) + tid;
    plat_t P; P.inc_warp = PL_INVALID;
    /* This walk stays 32-bit, unlike k_fill_atomic's. Converting it to
     * pl_next64 was tried on 2026-08-24 and REVERTED: __launch_bounds__(512, 3)
     * above caps this kernel at 40 registers, HEAD fits in exactly 40 with no
     * spill, and the two extra 64-bit live values push it over -- ptxas honours
     * the bound by spilling rather than by dropping to 2 blocks/SM.
     *
     *     HEAD:      40 regs, 0 stack,  0 spill st,  0 spill ld
     *     converted: 40 regs, 8 stack, 12 spill st,  8 spill ld
     *
     * The conversion existed only to keep the fill-strategy A/B honest, and
     * buying that with local-memory traffic in one strategy defeats its own
     * purpose. If this kernel is ever revisited, raise or drop the launch bound
     * first and re-measure; do not widen the walk underneath it. */
    uint32_t x = 0;
    int active = 0;

    for (;;) {
        if (!active) {                       /* pick up the next prime */
            while (k < n) {
                P = plat[(uint32_t)k];
                if (P.inc_warp != PL_INVALID) { x = pl_first(&P, logI); active = (x < xmax); }
                k += stride;
                if (active) break;
            }
        }
        if (!__syncthreads_or(active)) break;

        if (active) {
            /* gather a run of consecutive positions in one super-bucket */
            uint32_t run[L1_RUNMAX];
            const uint32_t b = x >> log_super;
            uint32_t nrun = 0, xn = x;
            do {
                run[nrun++] = xn;
                xn = pl_next(xn, &P, Imask);
            } while (nrun < L1_RUNMAX && xn < xmax && (xn >> log_super) == b);

            uint32_t slot = atomicAdd(&cnt[b], nrun);
            if (slot + nrun <= L1_CAP) {
                for (uint32_t t = 0; t < nrun; t++) buf[b * L1_CAP + slot + t] = run[t];
                x = xn;
                active = (x < xmax);
            } else {
                atomicSub(&cnt[b], nrun);    /* keep cnt exact; retry after flush */
            }
        }
        __syncthreads();

        /* flush buffers at/over the high-water mark. One warp per buffer. */
        const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
        for (uint32_t b = warp; b < L1_NBUF; b += nwarp) {
            uint32_t c = cnt[b];
            if (c < L1_FLUSH) continue;
            if (lane == 0) base[b] = atomicAdd(&cursor[b], c);
            __syncwarp();
            uint32_t dst = base[b];
            if (dst + c <= cap) {
                for (uint32_t t = lane; t < c; t += 32)
                    out[(size_t)b * cap + dst + t] = buf[b * L1_CAP + t];
            } else if (lane == 0) atomicAdd(overflow, c);
            __syncwarp();
            if (lane == 0) cnt[b] = 0;
        }
        __syncthreads();
    }

    /* drain partial buffers */
    {
        const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
        for (uint32_t b = warp; b < L1_NBUF; b += nwarp) {
            uint32_t c = cnt[b];
            if (c == 0) continue;
            if (lane == 0) base[b] = atomicAdd(&cursor[b], c);
            __syncwarp();
            uint32_t dst = base[b];
            if (dst + c <= cap) {
                for (uint32_t t = lane; t < c; t += 32)
                    out[(size_t)b * cap + dst + t] = buf[b * L1_CAP + t];
            } else if (lane == 0) atomicAdd(overflow, c);
            __syncwarp();
            if (lane == 0) cnt[b] = 0;
        }
    }
}

/* ---- stage (c) level 2: split one super-bucket into regions ----------- */

#define L2_NBUF 128
#define L2_CAP   64

template <int RECBYTES>
__global__ __launch_bounds__(512, 3)
void k_fill_l2(const uint32_t *__restrict in, const uint32_t *__restrict incnt,
               uint32_t in_cap, int log_region, int log_super,
               uint32_t *__restrict cursor, uint8_t *__restrict out,
               uint32_t cap, uint32_t *__restrict overflow)
{
    __shared__ uint32_t buf[L2_NBUF * L2_CAP];
    __shared__ uint32_t cnt[L2_NBUF];
    __shared__ uint32_t base[L2_NBUF];

    const uint32_t sb = blockIdx.x;              /* one block per super-bucket */
    uint32_t nrec = incnt[sb];
    if (nrec > in_cap) nrec = in_cap;   /* L1 cursor may overrun on overflow */
    const uint32_t offmask = (1u << log_region) - 1;
    const uint32_t regions_per_super = 1u << (log_super - log_region);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;

    for (uint32_t i = tid; i < L2_NBUF; i += nth) cnt[i] = 0;
    __syncthreads();

    uint32_t idx = tid;
    uint32_t pending = 0, prec = 0;
    for (;;) {
        int have = 0;
        if (pending) { have = 1; }
        else if (idx < nrec) { prec = in[(size_t)sb * in_cap + idx]; have = 1; }
        if (!__syncthreads_or(have)) break;

        if (have) {
            uint32_t b = (prec >> log_region) & (regions_per_super - 1);
            uint32_t slot = atomicAdd(&cnt[b], 1u);
            if (slot < L2_CAP) {
                buf[b * L2_CAP + slot] = prec;
                pending = 0; idx += nth;
            } else pending = 1;
        }
        __syncthreads();

        const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
        for (uint32_t b = warp; b < regions_per_super; b += nwarp) {
            uint32_t c = cnt[b];
            if (c < L2_CAP) continue;
            uint32_t gb = sb * regions_per_super + b;
            if (lane == 0) base[b] = atomicAdd(&cursor[gb], L2_CAP);
            __syncwarp();
            uint32_t dst = base[b];
            if (dst + L2_CAP <= cap) {
                for (uint32_t t = lane; t < L2_CAP; t += 32) {
                    uint32_t v = buf[b * L2_CAP + t];
                    size_t at = ((size_t)gb * cap + dst + t) * RECBYTES;
                    if (RECBYTES == 2) *(uint16_t *)(out + at) = (uint16_t)(v & offmask);
                    else               *(uint32_t *)(out + at) = (v & offmask);
                }
            } else if (lane == 0) atomicAdd(overflow, L2_CAP);
            __syncwarp();
            if (lane == 0) cnt[b] = 0;
        }
        __syncthreads();
    }

    const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
    for (uint32_t b = warp; b < regions_per_super; b += nwarp) {
        uint32_t c = cnt[b];
        if (c == 0) continue;
        if (c > L2_CAP) c = L2_CAP;
        uint32_t gb = sb * regions_per_super + b;
        if (lane == 0) base[b] = atomicAdd(&cursor[gb], c);
        __syncwarp();
        uint32_t dst = base[b];
        if (dst + c <= cap) {
            for (uint32_t t = lane; t < c; t += 32) {
                uint32_t v = buf[b * L2_CAP + t];
                size_t at = ((size_t)gb * cap + dst + t) * RECBYTES;
                if (RECBYTES == 2) *(uint16_t *)(out + at) = (uint16_t)(v & offmask);
                else               *(uint32_t *)(out + at) = (v & offmask);
            }
        } else if (lane == 0) atomicAdd(overflow, c);
        __syncwarp();
    }
}

/* ---- host driver ------------------------------------------------------ */

struct dev_bufs {
    uint32_t *primes, *roots, *cursor, *overflow, *nproj, *l1, *l1cnt;
    unsigned long long *nlost;
    plat_t   *plat;
    uint8_t  *out;
    uint16_t *slice, *slice_logp;
    uint32_t *nsurv, *probe;
    uint16_t *dbg;
    uint32_t *sp, *srt, *sg;
    uint32_t *smag;
    uint16_t *slp;
    uint8_t  *dumpbuf;
    uint32_t *survbits;
};

static float time_kernel(cudaEvent_t a, cudaEvent_t b)
{ float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }

/* ---- intersect + primitive filter + compaction ------------------------- *
 *
 * The two per-side survivor bitmaps are ANDed, positions that cannot give a
 * PRIMITIVE (a,b) are dropped, and what remains is compacted into a dense
 * list.
 *
 * PRIMITIVITY TAKES TWO TESTS, NOT ONE. gcd(i,j) != 1 is the obvious one, and
 * for a long time it was the only one, on the reasoning that (a,b) inherits
 * the primitivity of (i,j). It does not: (a,b) = M(i,j) with
 * det M = +-q, and a non-unimodular map can destroy primitivity. Exactly one
 * way, and it is cheap to test -- q | b (equivalently q | a, since a = rho*b
 * mod q on the lattice), which makes (a,b) = q*(a',b').
 *
 * Any OTHER common prime p is already excluded: p | a and p | b with p != q
 * implies (a/p, b/p) is still on the q-lattice, so (i,j) = p*(i',j') and the
 * gcd test catches it. So gcd(a,b) is 1 or q, and `b % q` decides which.
 *
 * These are not rare curiosities on a small-q job. A point with q | a and q | b
 * lies on the plattice line of EVERY root of q, so the sieve subtracts log(q)
 * once per root; when the algebraic polynomial splits completely mod q, that
 * is deg*log(q), which is precisely the q^deg sitting in F(a,b) = q^deg
 * F(a',b'). Both sides then look perfectly smooth and the position sails
 * through to trial division, which confirms the factorisation -- of a relation
 * that is q times a smaller one. msieve rejects them with "error -6"
 * (relation.c: gcd(a,b) != 1). Measured on an SNFS job with
 * F = (x^7-1)/(x-1), alim 3.5M and q from 400009: 154 of 154,810 emitted
 * relations were non-primitive, and every one had q = 1 (mod 7) -- the
 * condition for that F to split completely.
 *
 * The list carries BOTH the sieve index x and the pair (a,b). Emitting only
 * (a,b) would be the natural-looking choice and it is the wrong one: every
 * downstream stage -- resieve under either layout, trial division, the
 * cofactor queue -- is indexed by x, and recovering x from (a,b) means
 * inverting the lattice basis per survivor. x is 4 bytes; keep it.
 *
 * Bit-order note: bit k of word w is position x = 32*w + k, which is the
 * order k_apply writes one word per warp (atomicOr only in its small-region
 * fallback).
 */
__device__ __forceinline__ uint32_t bgcd(uint32_t u, uint32_t v)
{
    /* gcd(u,0) = u, which is what makes j = 0 fall out correctly: only
     * i = +-1 is primitive on that row. */
    if (!u) return v;
    if (!v) return u;
    int s = __ffs(u | v) - 1;
    u >>= __ffs(u) - 1;
    do {
        v >>= __ffs(v) - 1;
        if (u > v) { uint32_t t = u; u = v; v = t; }
        v -= u;
    } while (v);
    return u << s;
}


/* ---- recovery: the A/B ------------------------------------------------- *
 *
 * A: re-walk the factor base and test each hit against a hierarchical
 *    survivor filter. Touches no bucket memory at all.
 * B: stream the retained bucket array and keep the records that land on a
 *    survivor -- CADO's `purge`. Our 4 B record carries a slice index rather
 *    than a within-slice offset, so it cannot name the prime; that is a
 *    correctness gap, NOT a cost one. The memory traffic being measured here
 *    is exactly the traffic layout B would have, because the record is 4 B
 *    either way and every record is read exactly once.
 */
/* One summary bit per `wper` words of the full bitmap. wper == 2 reproduces
 * k_build_summary's 1-bit-per-64-positions table. Whole output words are built
 * by one thread, including a zero-padded partial final word, so unlike
 * k_build_summary this needs neither atomics nor a cudaMemset between slabs. */
__global__ void k_build_summary_g(const uint32_t *__restrict bits,
                                  uint32_t nword, uint32_t wper,
                                  uint32_t *__restrict summary)
{
    const uint32_t span = wper * 32u;
    const uint32_t nsw = (nword + span - 1u) / span;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t ww = bench_grid_thread_x(); ww < nsw; ww += stride) {
        const uint32_t w = (uint32_t)ww;
        uint32_t out = 0;
        for (uint32_t b = 0; b < 32; b++) {
            uint32_t o = 0;
            const uint32_t base = (w * 32u + b) * wper;
            if (base >= nword) break;
            for (uint32_t k = 0; k < wper && base + k < nword; k++)
                o |= bits[base + k];
            if (o) out |= 1u << b;
        }
        summary[w] = out;
    }
}

__global__ void k_build_summary(const uint32_t *__restrict bits,
                                uint32_t nword, uint32_t *__restrict summary)
{
    /* one summary bit per 64 positions == per 2 words of the full bitmap */
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t ss = bench_grid_thread_x(); ss < nword / 2; ss += stride) {
        const uint32_t s = (uint32_t)ss;
        uint32_t occupied = (bits[2 * s] | bits[2 * s + 1]) != 0u;
        if (occupied) atomicOr(&summary[s >> 5], 1u << (s & 31u));
    }
}

__global__ void k_resieve_rewalk(const plat_t *__restrict plat,
                                 const uint32_t *__restrict primes,
                                 uint32_t n, uint32_t xmax, int logI,
                                 const uint32_t *__restrict summary,
                                 const uint32_t *__restrict bits,
                                 uint32_t *__restrict out_x,
                                 uint32_t *__restrict out_p,
                                 uint32_t cap, uint32_t *__restrict nout,
                                 unsigned long long *__restrict nprobe,
                                 unsigned long long *__restrict npass1)
{
    const uint32_t Imask = (1u << logI) - 1;
    const uint64_t stride = bench_grid_stride_x();
    unsigned long long probes = 0, pass1 = 0;
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) continue;
        uint32_t p = primes[k];
        for (uint64_t x = pl_first64(&P, logI); x < xmax; x = pl_next64(x, &P, Imask)) {
            probes++;
            const uint32_t xn = (uint32_t)x;   /* narrow ONCE: every probe
                                                * below indexes with it, and a
                                                * 64-bit index is pure extra
                                                * address math on the dependent
                                                * load this loop waits on. */
            uint32_t sb = xn >> 6;
            if (!((summary[sb >> 5] >> (sb & 31u)) & 1u)) continue;
            pass1++;
            if (!((bits[xn >> 5] >> (xn & 31u)) & 1u)) continue;
            uint32_t slot = atomicAdd(nout, 1u);
            if (slot < cap) { out_x[slot] = xn; out_p[slot] = p; }
        }
    }
    if (probes) atomicAdd(nprobe, probes);
    if (pass1)  atomicAdd(npass1, pass1);
}

__global__ void k_purge(const uint32_t *__restrict recs,
                        const uint32_t *__restrict cursor,
                        uint32_t nregion, uint32_t cap, int log_region,
                        const uint32_t *__restrict bits,
                        uint32_t *__restrict out_x, uint32_t cap_out,
                        uint32_t *__restrict nout,
                        unsigned long long *__restrict nread)
{
    const uint32_t offmask = (1u << log_region) - 1;
    unsigned long long rd = 0;
    for (uint32_t b = blockIdx.x; b < nregion; b += gridDim.x) {
        uint32_t nrec = cursor[b];
        if (nrec > cap) nrec = cap;
        rd += (threadIdx.x == 0) ? nrec : 0;
        for (uint32_t t = threadIdx.x; t < nrec; t += blockDim.x) {
            uint32_t rec = recs[(size_t)b * cap + t];
            uint32_t x = (b << log_region) | (rec & offmask);
            if (!((bits[x >> 5] >> (x & 31u)) & 1u)) continue;
            uint32_t slot = atomicAdd(nout, 1u);
            if (slot < cap_out) out_x[slot] = x;
        }
    }
    if (rd) atomicAdd(nread, rd);
}


/* Slice cut for LAYOUT B. Two differences from fb_build_slices(), and both
 * are forced by the record format rather than chosen:
 *
 *  - the cap is 65536, not 262144, because the record's high 16 bits hold the
 *    offset WITHIN the slice. CADO has the same constraint and asserts on it
 *    (`fb.hpp:356`: size() <= numeric_limits<slice_offset_t>::max()).
 *  - the slice start indices are kept, because the fill is launched one slice
 *    at a time and the purge needs them to turn (slice, offset) back into a
 *    factor-base index.
 *
 * The tighter cap is what makes B's segment count ~3x the 39 that the looser
 * cut produces -- see the launch-cost discussion.
 */
static uint32_t build_slices_b(const fb_t *fb, uint32_t **starts_out)
{
    uint32_t ns = 0, k, capacity = 256;
    uint32_t *starts = (uint32_t *)malloc(capacity * 4);
    int cur = -1;
    uint32_t cut = 0;
    for (k = 0; k < fb->n; k++) {
        int lp = fb->logp[k];
        if (lp != cur || (ns && (k - cut) >= 65536u)) {
            if (ns + 2 > capacity) { capacity *= 2; starts = (uint32_t *)realloc(starts, capacity * 4); }
            cur = lp; cut = k; starts[ns++] = k;
        }
    }
    starts[ns] = fb->n;          /* sentinel: slice s is [starts[s], starts[s+1]) */
    *starts_out = starts;
    return ns;
}

/* Fill one slice. Identical walk to k_fill_atomic<4>, but the record carries
 * the offset of the prime within this slice instead of a slice index, which is
 * what lets the purge name the prime. Slice identity comes from segmentation:
 * the driver snapshots cursor[] after each launch, giving per-(bucket, slice)
 * boundaries for free -- no counting pass. */
__global__ void k_fill_segmented(const plat_t *__restrict plat,
                                 uint32_t kbeg, uint32_t kend,
                                 uint32_t xmax, int logI, int log_region,
                                 uint32_t *__restrict cursor,
                                 uint32_t *__restrict out, uint32_t cap,
                                 uint32_t *__restrict overflow)
{
    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t offmask = (1u << log_region) - 1;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = (uint64_t)kbeg + bench_grid_thread_x();
         kk < kend; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) continue;
        const uint32_t soff = k - kbeg;          /* < 65536 by construction */
        for (uint64_t x64 = pl_first64(&P, logI); x64 < xmax;
             x64 = pl_next64(x64, &P, Imask)) {
            const uint32_t x = (uint32_t)x64;
            uint32_t b = x >> log_region;
            uint32_t slot = atomicAdd(&cursor[b], 1u);
            if (slot >= cap) { atomicAdd(overflow, 1u); continue; }
            out[(size_t)b * cap + slot] = (x & offmask) | (soff << 16);
        }
    }
}

/* Snapshot cursor[] into column `sl` of the bucket-major boundary table.
 * This replaced a cudaMemcpy2D: a 2D copy 4 bytes wide over 32768 rows is
 * pathological (it was costing ~0.7 ms per slice, 88 ms over 126 slices),
 * whereas this is a plain coalesced read and a strided write. */
__global__ void k_snapshot_bounds(const uint32_t *__restrict cursor,
                                  uint32_t *__restrict bounds,
                                  uint32_t nregion, uint32_t nslice, uint32_t sl)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t bb = bench_grid_thread_x(); bb < nregion; bb += stride) {
        const uint32_t b = (uint32_t)bb;
        bounds[(size_t)b * nslice + sl] = cursor[b];
    }
}

/* Purge, layout B: stream the retained records, keep the ones on a survivor,
 * and name the prime. bounds[b*nslice + s] is the cursor value after slice s
 * finished bucket b, so the slice a slot belongs to is found by binary search
 * over that row -- contiguous per bucket, which is why the table is laid out
 * bucket-major. */
__global__ void k_purge_prime(const uint32_t *__restrict recs,
                              const uint32_t *__restrict cursor,
                              const uint32_t *__restrict bounds,
                              const uint32_t *__restrict starts,
                              const uint32_t *__restrict primes,
                              uint32_t nregion, uint32_t cap, uint32_t nslice,
                              int log_region,
                              const uint32_t *__restrict bits,
                              uint32_t *__restrict out_x,
                              uint32_t *__restrict out_p,
                              uint32_t cap_out, uint32_t *__restrict nout)
{
    const uint32_t offmask = (1u << log_region) - 1;
    for (uint32_t b = blockIdx.x; b < nregion; b += gridDim.x) {
        uint32_t nrec = cursor[b];
        if (nrec > cap) nrec = cap;
        const uint32_t *row = bounds + (size_t)b * nslice;
        for (uint32_t t = threadIdx.x; t < nrec; t += blockDim.x) {
            uint32_t rec = recs[(size_t)b * cap + t];
            uint32_t x = (b << log_region) | (rec & offmask);
            if (!((bits[x >> 5] >> (x & 31u)) & 1u)) continue;
            /* which slice does slot t belong to? row[] is non-decreasing. */
            uint32_t lo = 0, hi = nslice - 1;
            while (lo < hi) { uint32_t mid = (lo + hi) >> 1;
                              if (row[mid] <= t) lo = mid + 1; else hi = mid; }
            uint32_t p = primes[starts[lo] + (rec >> 16)];
            uint32_t slot = atomicAdd(nout, 1u);
            if (slot < cap_out) { out_x[slot] = x; out_p[slot] = p; }
        }
    }
}

template <int AGG, bool SLABBED = false>
__global__ void k_intersect_compact(const uint32_t *__restrict A,
                                    const uint32_t *__restrict B,
                                    uint32_t nword, uint32_t logI,
                                    int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                    int64_t q,
                                    uint32_t *__restrict out_x,
                                    int64_t  *__restrict out_a,
                                    int64_t  *__restrict out_b,
                                    uint32_t cap,
                                    uint32_t *__restrict nout,
                                    unsigned long long *__restrict npre,
                                    uint32_t *__restrict twosided,
                                    unsigned long long *__restrict nqb,
                                    uint32_t j_base)
{
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint64_t stride = bench_grid_stride_x();
    const uint32_t lane = threadIdx.x & 31u;
    unsigned long long pre = 0;
    /* Counted separately so the gcd(i,j)-only population stays recoverable.
     * That is the population CADO's after_sieve holds and dumpcmp --and
     * reproduces, so the parity gate must be able to name it; folding the
     * q|b rejections into the survivor count silently compares two different
     * populations, which is RESULTS finding 40's mistake. */
    unsigned long long qbrej = 0;

    /* Round the trip count up so every lane of a warp runs the same number of
     * iterations: the warp-aggregated atomic below needs a converged warp, and
     * lanes past the end simply contribute nothing. */
    const uint64_t tid = bench_grid_thread_x();
    const uint64_t iters = ((uint64_t)nword + stride - 1) / stride;

    for (uint64_t it = 0; it < iters; it++) {
        const uint64_t ww = tid + it * stride;
        const uint32_t w = (uint32_t)ww;
        uint32_t m = (ww < nword) ? (A[w] & B[w]) : 0u;
        uint32_t keep = 0;
        pre += __popc(m);
        while (m) {
            uint32_t k = __ffs(m) - 1;
            m &= m - 1;
            uint32_t x = (w << 5) + k;
            int32_t  i = (int32_t)(x & Imask) - Ihalf;
            uint32_t jlocal = x >> logI;
            uint32_t j = jlocal;
            if constexpr (SLABBED) j += j_base;
            uint32_t ai = (uint32_t)(i < 0 ? -i : i);
            if (bgcd(ai, j) != 1) continue;
            /* the second test: q | b makes (a,b) = q*(a',b'). One 64-bit
             * modulo per two-sided survivor, which is ~1 position in 400. */
            if (q > 1 && ((int64_t)i * a1 + (int64_t)j * b1) % q == 0) {
                qbrej++;
                continue;
            }
            keep |= 1u << k;
        }

        /* Warp-aggregated allocation: one atomicAdd per warp instead of one
         * per survivor. Survivors are ~1 in 400 positions, so a warp covering
         * 32 words holds only a few -- but they all serialise on the same
         * global counter, and that, not the 134 MB of bitmap, is what this
         * kernel spends its time on. */
        /* Each thread owns a whole 32-position word, so the primitive
         * two-sided bitmap can be stored outright -- no atomics needed. This
         * is what the resieve filters against. */
        if (twosided && ww < nword) twosided[w] = keep;

        uint32_t slot;
        if (AGG) {
            uint32_t n = __popc(keep), scan = n;
            for (int off = 1; off < 32; off <<= 1) {
                uint32_t v = __shfl_up_sync(0xffffffffu, scan, off);
                if (lane >= (uint32_t)off) scan += v;
            }
            uint32_t total = __shfl_sync(0xffffffffu, scan, 31);
            uint32_t base = 0;
            if (lane == 31 && total) base = atomicAdd(nout, total);
            base = __shfl_sync(0xffffffffu, base, 31);
            slot = base + (scan - n);   /* exclusive prefix */
        } else {
            /* the naive form, kept for the A/B: one global atomic per
             * non-empty word. Measured 0.567 ms against 0.447 ms aggregated,
             * so the ~10 lines above buy 21%. */
            slot = keep ? atomicAdd(nout, __popc(keep)) : 0;
        }

        while (keep) {
            uint32_t k = __ffs(keep) - 1;
            keep &= keep - 1;
            uint32_t x = (w << 5) + k;
            int32_t  i = (int32_t)(x & Imask) - Ihalf;
            uint32_t jlocal = x >> logI;
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
    /* one atomic per thread, not one per survivor */
    if (pre) atomicAdd(npre, pre);
    if (nqb && qbrej) atomicAdd(nqb, qbrej);
}

/* ======================= trial division, host side ======================= */

/* The exact homogeneous form for one side. The rational side is not a special
 * case: G(a,b) = Y1*a + Y0*b is the degree-1 member of the same family, so the
 * norm kernel is shared. */
static int td_build_poly(tdpoly_t *T, const poly_t *P, int side)
{
    memset(T, 0, sizeof(*T));
    for (int k = 0; k < BENCH_NCOEFF; k++) T->sign[k] = 1;
    if (side == 0) {
        int s0 = 1, s1 = 1;
        T->deg = 1;
        if (bn_from_dec(&T->c[0], P->y0s, &s0)) return -1;
        if (bn_from_dec(&T->c[1], P->y1s, &s1)) return -1;
        T->sign[0] = s0; T->sign[1] = s1;
        return 0;
    }
    T->deg = P->deg;
    for (int k = 0; k <= P->deg; k++) {
        int s = 1;
        if (!P->cs[k][0]) continue;                 /* absent == zero */
        if (bn_from_dec(&T->c[k], P->cs[k], &s)) return -1;
        T->sign[k] = s;
    }
    return 0;
}

/* Direct-test table for p < bkthresh.
 *
 * PROPER PRIME POWERS ARE EXCLUDED. fb_split_small puts every power in this
 * table regardless of size, but trial division recovers multiplicity by
 * repeated division by the base prime -- which is always present here too,
 * since a power p^k below the factor-base bound forces p well below bkthresh.
 * Keeping the powers would divide a norm by p^2 as though p^2 were prime.
 *
 * For a PRIME modulus the transform's row divisor g can only be 1 or p (it is
 * gcd(D, p)), so p == m*g always holds and the prime never has to be carried
 * separately. */
/* Fills a caller-provided table of at least fbs->n entries. Split out from
 * td_build_small so the pipeline can refill ONE pinned buffer per special-q
 * instead of malloc/free-ing 85 KB on every q of a band. Returns the entry
 * count, or 0 with a message if the m*g invariant fails. */
static uint32_t td_fill_small(const fb_t *fbs, const qlat_t *L, int logI,
                              tdsmall_t *t)
{
    const uint32_t Ihalf = 1u << (logI - 1);
    uint32_t n = 0;
    for (uint32_t i = 0; i < fbs->n; i++) {
        uint32_t rt, g, m;
        if (FB_ISPOW(fbs, i)) continue;
        m = pl_transform_fast(fbs->primes[i], fbs->roots[i],
                              L->a0, L->a1, L->b0, L->b1, &rt, &g);
        if (m * g != fbs->primes[i]) {          /* the invariant above */
            fprintf(stderr, "td_build_small: m*g != p at entry %u"
                            " (p=%u m=%u g=%u)\n", i, fbs->primes[i], m, g);
            return 0;
        }
        t[n].m = m; t[n].rt = rt; t[n].g = g;
        t[n].cst = Ihalf % m;
        t[n].recip = bn_recip_u32(fbs->primes[i]);
        td_magic_build(m, &t[n].magic, &t[n].sh);
        n++;
    }
    if (getenv("TD_DUMP_SMALL")) {
        uint32_t c2 = 0;
        fprintf(stderr, "td_fill_small: %u entries from %u fb rows; first 6:", n, fbs->n);
        for (uint32_t i = 0; i < n && i < 6; i++)
            fprintf(stderr, " p=%u(m=%u,g=%u,rt=%u,magic=%u)",
                    t[i].m * t[i].g, t[i].m, t[i].g, t[i].rt, t[i].magic);
        for (uint32_t i = 0; i < n; i++) if (t[i].m * t[i].g == 2) c2++;
        fprintf(stderr, "  | entries with p==2: %u\n", c2);
    }
    return n;
}

static uint32_t td_build_small(const fb_t *fbs, const qlat_t *L, int logI,
                               tdsmall_t **out)
{
    uint32_t n;
    tdsmall_t *t;
    if (!fbs || !fbs->n) { *out = NULL; return 0; }
    t = (tdsmall_t *)malloc((size_t)fbs->n * sizeof(tdsmall_t));
    if (!t) { *out = NULL; return 0; }
    n = td_fill_small(fbs, L, logI, t);
    if (!n) { free(t); *out = NULL; return 0; }
    *out = t;
    return n;
}

/* ---- the gate: our cofactors against CADO's ---------------------------- */

typedef struct { int64_t a, b; uint32_t idx; } td_ab_t;

static bool td_ab_less(const td_ab_t &x, const td_ab_t &y)
{ return x.b != y.b ? x.b < y.b : x.a < y.a; }

/* oracle/c183.q*.cofac_candidates.txt holds `a b cofac0 cofac1` for every
 * position CADO carried into cofactoring, i.e. its residual after its own
 * trial division. Ours must agree exactly, which is a far stronger statement
 * than agreeing on a count. */
static int td_gate_cofactors_part(const char *path, uint32_t n,
                                  const int64_t *ha, const int64_t *hb,
                                  const bn_t *hcof, int side,
                                  uint32_t *found_out)
{
    FILE *f = fopen(path, "r");
    char line[512];
    uint32_t nref = 0, found = 0, match = 0, absent = 0;
    td_ab_t *tab;
    if (!f) { perror(path); return -1; }

    /* our list, keyed on (a,b) normalised to b > 0 -- our b is i, and las
     * reports the mirrored point (-a, -i) whenever i < 0. */
    tab = (td_ab_t *)malloc((size_t)n * sizeof(td_ab_t));
    if (!tab) { fclose(f); return -1; }
    for (uint32_t k = 0; k < n; k++) {
        int64_t a = ha[k], b = hb[k];
        if (b < 0) { a = -a; b = -b; }
        tab[k].a = a; tab[k].b = b; tab[k].idx = k;
    }
    std::sort(tab, tab + n, td_ab_less);

    printf("\n  --- trial-division gate vs %s (side %d) ---\n", path, side);
    while (fgets(line, sizeof line, f)) {
        char c0[128], c1[128];
        long long ra, rb;
        td_ab_t key, *lo;
        if (line[0] == '#') continue;
        if (sscanf(line, "%lld %lld %127s %127s", &ra, &rb, c0, c1) != 4) continue;
        nref++;
        key.a = ra; key.b = rb; key.idx = 0;
        lo = std::lower_bound(tab, tab + n, key, td_ab_less);
        if (lo == tab + n || lo->a != key.a || lo->b != key.b) { absent++; continue; }
        found++;
        {
            char buf[BN_DEC_MAX];
            bn_to_dec(&hcof[lo->idx], buf);
            if (!strcmp(buf, side ? c1 : c0)) match++;
            else if (match + 8 > found)      /* show the first few only */
                printf("    MISMATCH (a,b)=(%lld,%lld)  ours %s  CADO %s\n",
                       ra, rb, buf, side ? c1 : c0);
        }
    }
    fclose(f);
    printf("  %-30s %8u\n", "reference records", nref);
    printf("  %-30s %8u  (%u not in our survivor list)\n",
           "matched to a survivor", found, absent);
    printf("  %-30s %8u of %u   %s\n", "cofactors identical", match, found,
           !found ? "NO OVERLAP"
                  : (match == found ? "PASS" : "FAIL"));
    if (found_out) *found_out += found;
    free(tab);
    /* No overlap in one slab is not an error: a reference file covers the
     * complete q. The slabbed pipeline accumulates found across all slabs and
     * rejects the q if the total remains zero. Any actual mismatch is fatal. */
    return (int)(found - match);
}

/* Whole-area/harness semantics retain the historical requirement that the
 * reference overlap at least one survivor. */
static int td_gate_cofactors(const char *path, uint32_t n,
                             const int64_t *ha, const int64_t *hb,
                             const bn_t *hcof, int side)
{
    uint32_t found = 0;
    const int rc = td_gate_cofactors_part(path, n, ha, hb, hcof, side, &found);
    if (rc) return rc;
    return found ? 0 : 1;
}

/* ---- the stage ---------------------------------------------------------- */

/* The MEASUREMENT harness for the trial-division chain: best-of-three on every
 * kernel, plus the two diagnostic variants of k_td that separate the small
 * prime congruence test from the divisions it triggers, plus the reconstruction
 * gate. The pipeline used to call this per q and per side, which is what made
 * its post-sieve cost 222 ms/q; it now runs pipe_td_perq, and this stays what
 * `bench --td` reports and what every command in RESULTS.md reproduces. */
static int run_td_stage(const fb_t *fb, const fb_t *fbs, const qlat_t *L,
                        const poly_t *POLY, const bench_cfg_t *cfg,
                        const plat_t *d_plat, const uint32_t *d_primes,
                        const uint32_t *d_two, uint32_t nbitword,
                        uint32_t xmax, int blocks, int threads)
{
    const uint32_t K = 16;          /* large primes kept per survivor */
    const uint32_t ngroup = nbitword / TD_GROUP_W;
    const uint32_t nb = (ngroup + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    const uint32_t nsum = nbitword / 2;
    const uint32_t nsumword = (nsum + 31) / 32;

    uint32_t *d_cnt = NULL, *d_gbase = NULL, *d_bsum = NULL, *d_sum = NULL;
    uint32_t *d_x = NULL, *d_plist = NULL, *d_pcnt = NULL, *d_flags = NULL;
    uint8_t  *d_status = NULL;
    uint32_t *d_fac = NULL, *d_faccnt = NULL;
    int64_t  *d_a = NULL, *d_b = NULL;
    bn_t     *d_cof = NULL;
    uint8_t  *d_cofbits = NULL;
    tdpoly_t *d_poly = NULL;
    tdsmall_t *d_sm = NULL, *h_sm = NULL;
    unsigned long long *d_ovf = NULL;
    tdpoly_t h_poly;
    int rc = 0;
    uint32_t n = 0, nsm = 0, hflags = 0;

    unsigned long long hovf = 0;
    cudaEvent_t t0, t1;
    float ms_rank = 0, ms_emit = 0, ms_sum = 0, ms_scatter = 1e30f, ms_td = 1e30f;
    float ms_td_nosm = 1e30f, ms_td_nodiv = 1e30f, ms_class = 0;
    unsigned long long hhits = 0;

    if (nbitword % TD_GROUP_W) {
        fprintf(stderr, "  --td: %u bitmap words is not a multiple of %d\n",
                nbitword, TD_GROUP_W);
        return -1;
    }
    if (L->q >> 32) {
        fprintf(stderr, "  --td: special-q %llu exceeds 32 bits; the divide-out"
                " path assumes it fits\n", (unsigned long long)L->q);
        return -1;
    }
    if (td_build_poly(&h_poly, POLY, cfg->side)) {
        fprintf(stderr, "  --td: could not parse exact polynomial coefficients\n");
        return -1;
    }
    nsm = td_build_small(fbs, L, cfg->logI, &h_sm);

    printf("\n  --- exact norms + trial division (side %d) ---\n", cfg->side);
    printf("  form: degree %d, |c| up to %d bits;"
           " small direct-test table %u entries (%.1f KB)\n",
           h_poly.deg, bn_bits(&h_poly.c[0]), nsm,
           nsm * sizeof(tdsmall_t) / 1024.0);

    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* ---- survivor rank over the two-sided bitmap ---- */
    CK(cudaMalloc(&d_cnt, (size_t)ngroup * 4));
    CK(cudaMalloc(&d_gbase, (size_t)ngroup * 4));
    CK(cudaMalloc(&d_bsum, (size_t)nb * 4));
    /* Best of 3, like every other timed block here: CUDA loads a kernel's code
     * on its FIRST launch (lazy module loading), and these four kernels have
     * never run at this point. Timing the first launch reported 10.7 ms for
     * what steady-state measurement puts an order of magnitude below. */
    ms_rank = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        k_group_counts<<<blocks, threads>>>(d_two, ngroup, d_cnt);
        k_scan_pass1<<<nb, TD_SCAN_BLK>>>(d_cnt, ngroup, d_gbase, d_bsum);
        k_scan_pass2<<<1, 1024>>>(d_bsum, nb);
        k_scan_pass3<<<nb, TD_SCAN_BLK>>>(d_gbase, ngroup, d_bsum);
        cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_rank) ms_rank = t; }
    }
    {
        uint32_t base = 0, cnt = 0;
        CK(cudaMemcpy(&base, d_gbase + ngroup - 1, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&cnt,  d_cnt   + ngroup - 1, 4, cudaMemcpyDeviceToHost));
        n = base + cnt;
    }
    printf("  %-30s %8u\n", "survivors (rank scan)", n);
    if (!n) { fprintf(stderr, "  --td: no survivors to divide\n"); rc = -1; goto done; }

    /* ---- rank-ordered (x, a, b) ---- */
    CK(cudaMalloc(&d_x, (size_t)n * 4));
    CK(cudaMalloc(&d_a, (size_t)n * 8));
    CK(cudaMalloc(&d_b, (size_t)n * 8));
    ms_emit = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        k_emit_ranked<false><<<blocks, threads>>>(d_two, d_gbase, nbitword, cfg->logI,
                                           L->a0, L->a1, L->b0, L->b1,
                                           d_x, d_a, d_b, n, 0u);
        cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_emit) ms_emit = t; }
    }

    /* ---- large primes: re-walk, filtered, scattered per survivor ---- */
    CK(cudaMalloc(&d_sum, (size_t)nsumword * 4));
    CK(cudaMalloc(&d_plist, (size_t)n * K * 4));
    CK(cudaMalloc(&d_pcnt, (size_t)n * 4));
    CK(cudaMalloc(&d_ovf, 8));
    cudaEventRecord(t0);
    CK(cudaMemset(d_sum, 0, (size_t)nsumword * 4));
    k_build_summary<<<blocks, threads>>>(d_two, nbitword, d_sum);
    cudaEventRecord(t1);
    CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
    ms_sum = time_kernel(t0, t1);

    /* Sweep BOTH knobs, because they attack the same bottleneck from opposite
     * directions and only measurement separates them.
     *
     *   unroll     -- more probes in flight per warp, hiding the latency
     *   granularity-- a coarser summary is a smaller table, so the probe may
     *                 land in L1 instead of L2, lowering the latency itself.
     *                 Coarser also means more steps reach the 67 MB bitmap.
     *
     * The recovered prime counts must be identical at every setting. */
    {
        uint32_t *h_ref = NULL;
        /* Production setting first, so a normal run times exactly one thing.
         * The rest is the sweep that chose it and only runs on request:
         * unroll 4 at the original granularity, measured 5.21 ms against 6.92
         * at unroll 1. Granularity turned out not to matter at all -- 5.21 to
         * 5.35 ms across a 16x change in table size -- which is itself the
         * evidence that this kernel is latency-bound rather than
         * cache-resident-bound. */
        struct { int lg, u; } trial[] = {
            {6,4},
            {6,1},{6,2},{6,8},{7,4},{8,4},{9,4},{10,4},{9,8},{10,8}
        };
        const unsigned ntrial = cfg->resieve_sweep
                              ? (unsigned)(sizeof trial / sizeof *trial) : 1u;
        for (unsigned ti = 0; ti < ntrial; ti++) {
            const int log_gran = trial[ti].lg, U = trial[ti].u;
            const uint32_t wper = 1u << (log_gran - 5);
            const uint32_t nsw = nbitword / (wper * 32);
            float best = 1e30f;
            uint32_t mism = 0;
            if (!nsw) continue;
            k_build_summary_g<<<blocks, threads>>>(d_two, nbitword, wper, d_sum);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            for (int rep = 0; rep < 3; rep++) {
                CK(cudaMemset(d_pcnt, 0, (size_t)n * 4));
                CK(cudaMemset(d_ovf, 0, 8));
                cudaEventRecord(t0);
                switch (U) {
                case 1: k_resieve_scatter<1, false><<<blocks, threads>>>(
                            d_plat, d_primes, NULL, fb->n, xmax, cfg->logI,
                            d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, NULL); break;
                case 2: k_resieve_scatter<2, false><<<blocks, threads>>>(
                            d_plat, d_primes, NULL, fb->n, xmax, cfg->logI,
                            d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, NULL); break;
                case 4: k_resieve_scatter<4, false><<<blocks, threads>>>(
                            d_plat, d_primes, NULL, fb->n, xmax, cfg->logI,
                            d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, NULL); break;
                default: k_resieve_scatter<8, false><<<blocks, threads>>>(
                            d_plat, d_primes, NULL, fb->n, xmax, cfg->logI,
                            d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, NULL); break;
                }
                cudaEventRecord(t1);
                CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
                { float t = time_kernel(t0, t1); if (t < best) best = t; }
            }
            /* Compare the recovered PRIMES, not merely how many there are:
             * two configurations could scatter different primes into a
             * survivor's slot and still agree on the count. The scatter order
             * within a slot is atomic-dependent, so sort each slot first. */
            {
                const size_t sz = (size_t)n * (K + 1) * 4;
                uint32_t *h = (uint32_t *)malloc(sz);
                CK(cudaMemcpy(h, d_pcnt, (size_t)n * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h + n, d_plist, (size_t)n * K * 4,
                              cudaMemcpyDeviceToHost));
                for (uint32_t z = 0; z < n; z++) {
                    uint32_t c = h[z] > K ? K : h[z];
                    std::sort(h + n + (size_t)z * K, h + n + (size_t)z * K + c);
                }
                if (!h_ref) h_ref = h;
                else {
                    for (uint32_t z = 0; z < n; z++) {
                        uint32_t c = h[z] > K ? K : h[z];
                        if (h[z] != h_ref[z]) { mism++; continue; }
                        for (uint32_t y = 0; y < c; y++)
                            if (h[n + (size_t)z * K + y] !=
                                h_ref[n + (size_t)z * K + y]) { mism++; break; }
                    }
                    free(h);
                }
            }
            if (cfg->resieve_sweep)
                printf("  %-30s %8.3f ms   (1 bit/%4d positions, %6.1f KB,"
                       " unroll %d)%s\n", "resieve + scatter", best,
                       1 << log_gran, nsw * 4 / 1024.0, U,
                       mism ? "  ** RECOVERY CHANGED" : "");
            else
                printf("  %-30s %8.3f ms   (unroll %d)\n",
                       "resieve + scatter", best, U);
            if (mism) rc = -1;
            /* ti == 0 IS the production configuration (unroll 4, 1 bit/64).
             * Taking the fastest trial instead would report a chain total for
             * a setting the pipeline does not run. */
            if (ti == 0) ms_scatter = best;
        }
        free(h_ref);
        /* restore the reference summary and leave a correct scatter behind */
        k_build_summary_g<<<blocks, threads>>>(d_two, nbitword, 2u, d_sum);
        CK(cudaMemset(d_pcnt, 0, (size_t)n * 4));
        CK(cudaMemset(d_ovf, 0, 8));
        k_resieve_scatter<4, false><<<blocks, threads>>>(
            d_plat, d_primes, NULL, fb->n, xmax, cfg->logI,
            d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, 6, NULL);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(&hovf, d_ovf, 8, cudaMemcpyDeviceToHost));
    }

    /* ---- the trial division itself ---- */
    CK(cudaMalloc(&d_poly, sizeof(tdpoly_t)));
    CK(cudaMemcpy(d_poly, &h_poly, sizeof(tdpoly_t), cudaMemcpyHostToDevice));
    if (nsm) {
        CK(cudaMalloc(&d_sm, (size_t)nsm * sizeof(tdsmall_t)));
        CK(cudaMemcpy(d_sm, h_sm, (size_t)nsm * sizeof(tdsmall_t),
                      cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&d_cof, (size_t)n * sizeof(bn_t)));
    CK(cudaMalloc(&d_cofbits, (size_t)n));
    CK(cudaMalloc(&d_flags, 4));

    /* Same kernel with the small-prime table empty. The direct test is the
     * part with no prior measurement behind it -- 3,500 entries against every
     * survivor -- so it is worth separating from the norm and the recovered
     * large primes rather than reporting one fused number. Run FIRST so the
     * full pass overwrites its output. */
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        k_td<1, 0, 0, false><<<blocks, threads>>>(d_a, d_b, d_x, NULL, n, cfg->logI, d_poly,
                                           cfg->side == 1 ? (uint32_t)L->q : 0u,
                                           d_plist, d_pcnt, K, d_sm, 0u,
                                           d_cof, d_cofbits, d_flags, NULL,
                                           NULL, NULL, 0, 0u);
        cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_td_nosm) ms_td_nosm = t; }
    }

    /* the same walk with the divisions removed: separates the 3e9 congruence
     * tests from the big-integer divisions they trigger */
    CK(cudaMemset(d_ovf, 0, 8));
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        k_td<0, 0, 0, false><<<blocks, threads>>>(d_a, d_b, d_x, NULL, n, cfg->logI, d_poly,
                                           cfg->side == 1 ? (uint32_t)L->q : 0u,
                                           d_plist, d_pcnt, K, d_sm, nsm,
                                           d_cof, d_cofbits, d_flags, d_ovf,
                                           NULL, NULL, 0, 0u);
        cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_td_nodiv) ms_td_nodiv = t; }
    }
    CK(cudaMemcpy(&hhits, d_ovf, 8, cudaMemcpyDeviceToHost));
    hhits /= 3;                      /* three reps accumulated into it */

    for (int rep = 0; rep < 3; rep++) {
        CK(cudaMemset(d_flags, 0, 4));
        cudaEventRecord(t0);
        k_td<1, 0, 0, false><<<blocks, threads>>>(d_a, d_b, d_x, NULL, n, cfg->logI, d_poly,
                                           cfg->side == 1 ? (uint32_t)L->q : 0u,
                                           d_plist, d_pcnt, K, d_sm, nsm,
                                           d_cof, d_cofbits, d_flags, NULL,
                                           NULL, NULL, 0, 0u);
        cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_td) ms_td = t; }
    }
    CK(cudaMemcpy(&hflags, d_flags, 4, cudaMemcpyDeviceToHost));

    printf("  %-30s %8.3f ms\n", "rank scan", ms_rank);
    printf("  %-30s %8.3f ms\n", "emit (x,a,b) in rank order", ms_emit);
    printf("  %-30s %8.3f ms\n", "build survivor filter", ms_sum);
    if (hovf) printf("  ** resieve list overflow\n");
    printf("  %-30s %8.3f ms   (norm + special-q + %u recovered large primes)\n",
           "  ...without small primes", ms_td_nosm, K);
    printf("  %-30s %8.3f ms   (%llu hits, %.2f per survivor)\n",
           "  ...test only, no division", ms_td_nodiv, hhits, (double)hhits / n);
    printf("  %-30s %8.3f ms   (%u-entry test %.3f + division %.3f)\n",
           "norms + trial division", ms_td, nsm,
           ms_td_nodiv - ms_td_nosm, ms_td - ms_td_nodiv);
    printf("  %-30s %8.3f ms\n", "TD chain total",
           ms_rank + ms_emit + ms_sum + ms_scatter + ms_td);
    /* Both of these silently corrupt factorisations rather than crashing, so
     * they must fail the run. A norm that overflowed 256 bits is wrong, and a
     * truncated prime list leaves factors undivided. */
    if (hflags & TDF_NORM_OVERFLOW) {
        fprintf(stderr, "  ** NORM OVERFLOW: a norm exceeded %d bits\n", BN_LIMBS * 32);
        rc = -1;
    }
    if (hflags & TDF_LIST_TRUNCATED) {
        fprintf(stderr, "  ** %llu large-prime records past the %u/survivor cap\n",
                hovf, K);
        rc = -1;
    }

    /* ---- factorisation record, for relation output ----
     * A separate untimed pass with RECORD=1 rather than stores in the measured
     * kernel: the factors are only wanted when a run is emitting, and the hot
     * path should not carry writes it does not need. */
    if (cfg->emit_cof) {
        CK(cudaMalloc(&d_fac, (size_t)n * TD_FMAX * 4));
        CK(cudaMalloc(&d_faccnt, (size_t)n * 4));
        k_td<1, 1, 0, false><<<blocks, threads>>>(d_a, d_b, d_x, NULL, n, cfg->logI, d_poly,
                                           cfg->side == 1 ? (uint32_t)L->q : 0u,
                                           d_plist, d_pcnt, K, d_sm, nsm,
                                           d_cof, d_cofbits, d_flags, NULL,
                                           d_fac, d_faccnt, TD_FMAX, 0u);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    }

    /* ---- classification: CADO's check_leftover_norm ---- */
    CK(cudaMalloc(&d_status, (size_t)n));
    ms_class = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        k_classify<<<blocks, threads>>>(d_cof, d_cofbits, d_b, n,
                                        cfg->lpb, cfg->mfb, (double)cfg->lim,
                                        d_status);
        cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1)); CK(cudaGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_class) ms_class = t; }
    }

    /* ---- readback: cofactor sizes, gate, emission ---- */
    {
        bn_t *hcof = (bn_t *)malloc((size_t)n * sizeof(bn_t));
        uint8_t *hbits = (uint8_t *)malloc((size_t)n);
        uint8_t *hstat = (uint8_t *)malloc((size_t)n);
        int64_t *ha = (int64_t *)malloc((size_t)n * 8);
        int64_t *hb = (int64_t *)malloc((size_t)n * 8);
        uint32_t hist[257]; uint32_t nfully = 0;
        memset(hist, 0, sizeof hist);
        CK(cudaMemcpy(hcof, d_cof, (size_t)n * sizeof(bn_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hbits, d_cofbits, (size_t)n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hstat, d_status, (size_t)n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ha, d_a, (size_t)n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb, d_b, (size_t)n * 8, cudaMemcpyDeviceToHost));
        for (uint32_t k = 0; k < n; k++) {
            hist[hbits[k]]++;
            if (hbits[k] <= 1) nfully++;
        }
        printf("  %-30s %8u  (%.2f%% of survivors)\n",
               "cofactor == 1 (fully split)", nfully, 100.0 * nfully / n);
        printf("  cofactor bits:");
        for (int b = 0; b <= 256; b++)
            if (hist[b] && (b % 16 == 0 || hist[b] > n / 64))
                printf(" %d:%u", b, hist[b]);
        printf("\n");

        {
            uint32_t cs[6] = {0, 0, 0, 0, 0, 0};
            static const char *nm[6] = {"rejected: > mfb bits",
                                        "rejected: too few factors possible",
                                        "rejected: prime above 2^lpb",
                                        "ACCEPTED for cofactorisation",
                                        "already fully split",
                                        "rejected: b == 0, not a relation"};
            for (uint32_t k = 0; k < n; k++) if (hstat[k] < 6) cs[hstat[k]]++;
            printf("  %-30s %8.3f ms   (lpb %u, mfb %u, lim %u)\n",
                   "classify", ms_class, cfg->lpb, cfg->mfb, cfg->lim);
            for (int k = 0; k < 6; k++)
                printf("    %-32s %8u  (%.3f%%)\n", nm[k], cs[k],
                       100.0 * cs[k] / n);
            printf("    %-32s %8u\n", "-> this side's candidates",
                   cs[COF_ACCEPT] + cs[COF_SPLIT]);
        }

        if (cfg->cofgate &&
            td_gate_cofactors(cfg->cofgate, n, ha, hb, hcof, cfg->side) != 0)
            rc = -1;

        if (cfg->emit_cof) {
            FILE *fo = cfg->emit_cof ? fopen(cfg->emit_cof, "wb") : NULL;
            uint32_t *hfac = (uint32_t *)malloc((size_t)n * TD_FMAX * 4);
            uint32_t *hfn = (uint32_t *)malloc((size_t)n * 4);
            uint32_t checked = 0, bad = 0, overflowed = 0, maxfac = 0;
            CK(cudaMemcpy(hfac, d_fac, (size_t)n * TD_FMAX * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hfn, d_faccnt, (size_t)n * 4, cudaMemcpyDeviceToHost));
            /* Canonical order. The large primes arrive via an atomicAdd on the
             * survivor's slot counter, so their order in the list varies run to
             * run; sorting makes the emitted factorisation byte-reproducible,
             * which is what lets two paths be diffed against each other. */
            /* Unconditional: a candidate whose factor list overflowed TD_FMAX
             * has a TRUNCATED list, and every consumer reads faccnt entries.
             * Detecting that only on the verified q left later q emitting an
             * out-of-bounds read of the factor matrix. */
            for (uint32_t k = 0; k < n; k++) {
                uint32_t c = hfn[k] > TD_FMAX ? TD_FMAX : hfn[k];
                if (hstat[k] != COF_ACCEPT && hstat[k] != COF_SPLIT) continue;
                if (hfn[k] > maxfac) maxfac = hfn[k];
                if (hfn[k] > TD_FMAX) overflowed++;
                std::sort(hfac + (size_t)k * TD_FMAX,
                          hfac + (size_t)k * TD_FMAX + c);
            }

            /* Reconstruction gate: the recorded factors times the residual
             * cofactor must rebuild the exact norm. The CADO gate above checks
             * only the residual, so it would pass even if the factor list were
             * wrong; this checks the list. Run over the candidates, which are
             * the records that will actually be emitted. */
            for (uint32_t k = 0; cfg->td_verify && k < n; k++) {
                bns_t acc; bn_t t;
                int64_t a, b;
                uint64_t ua, ub;
                int sa, sb;
                if (hstat[k] != COF_ACCEPT && hstat[k] != COF_SPLIT) continue;
                if (hfn[k] > TD_FMAX) continue;
                checked++;
                a = ha[k]; b = hb[k];
                ua = (uint64_t)(a < 0 ? -a : a); ub = (uint64_t)(b < 0 ? -b : b);
                sa = (a < 0) ? -1 : 1; sb = (b < 0) ? -1 : 1;
                bns_zero(&acc);
                for (int d = 0; d <= h_poly.deg; d++) {
                    int sgn = h_poly.sign[d];
                    t = h_poly.c[d];
                    if (bn_is_zero(&t)) continue;
                    for (int e = 0; e < d; e++) { bn_mul_u64(&t, ua); sgn *= sa; }
                    for (int e = 0; e < h_poly.deg - d; e++) { bn_mul_u64(&t, ub); sgn *= sb; }
                    bns_addmag(&acc, &t, sgn);
                }
                /* rebuild: cofactor * prod(factors) */
                t = hcof[k];
                for (uint32_t z = 0; z < hfn[k]; z++)
                    bn_mul_u64(&t, (uint64_t)hfac[(size_t)k * TD_FMAX + z]);
                if (bn_cmp(&t, &acc.m) != 0) bad++;
            }
            if (cfg->td_verify)
                printf("  %-30s %8u of %u   %s\n",
                       "factors x cofactor == norm", checked - bad, checked,
                       bad ? "FAIL" : "PASS");
            if (bad) rc = -1;
            if (cfg->td_verify)
                printf("  %-30s %8u of %d\n", "most factors on a candidate",
                       maxfac, TD_FMAX);
            if (overflowed) {
                fprintf(stderr, "  ** %u candidates had more than %d factors;"
                        " raise TD_FMAX\n", overflowed, TD_FMAX);
                rc = -1;
            }

            if (cfg->emit_cof && !fo) { perror(cfg->emit_cof); rc = -1; }
            else if (fo) {
                char buf[BN_DEC_MAX];
                for (uint32_t k = 0; k < n; k++) {
                    int64_t a = ha[k], b = hb[k];
                    if (b < 0) { a = -a; b = -b; }
                    fprintf(fo, "%lld %lld %s %u %u %u", (long long)a, (long long)b,
                            bn_to_dec(&hcof[k], buf), hbits[k], hstat[k], hfn[k]);
                    for (uint32_t z = 0; z < hfn[k] && z < TD_FMAX; z++)
                        fprintf(fo, " %u", hfac[(size_t)k * TD_FMAX + z]);
                    fputc('\n', fo);
                }
                /* a short write shows up at fclose, and this file is now an
                 * input to the next process in the pipeline */
                if (ferror(fo)) rc = -1;
                if (fclose(fo)) { perror(cfg->emit_cof); rc = -1; }
                else if (rc == 0)
                    printf("  wrote %u (a, b, cofactor, bits, status, factors)"
                           " to %s\n", n, cfg->emit_cof);
            }
            free(hfac); free(hfn);
        }
        free(hcof); free(hbits); free(hstat); free(ha); free(hb);
    }

done:
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(h_sm);
    cudaFree(d_cnt); cudaFree(d_gbase); cudaFree(d_bsum); cudaFree(d_sum);
    cudaFree(d_x); cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_plist); cudaFree(d_pcnt); cudaFree(d_flags);
    cudaFree(d_cof); cudaFree(d_cofbits); cudaFree(d_poly); cudaFree(d_sm);
    cudaFree(d_ovf); cudaFree(d_status); cudaFree(d_fac); cudaFree(d_faccnt);
    return rc;
}

/* The scaffold both device arithmetic gates share: upload n host-built cases,
 * run the check kernel, and read back the first failing index (UINT32_MAX if
 * none). Returns -1 on a CUDA error, leaving *first_bad untouched. One copy,
 * so an error-path fix reaches both gates. */
template <typename Case>
static int run_device_cases(void (*kern)(const Case *, uint32_t, uint32_t *),
                            const Case *h, uint32_t n, uint32_t blocks,
                            uint32_t *first_bad)
{
    Case *d = NULL;
    uint32_t *d_bad = NULL, bad = UINT32_MAX;
    int rc = -1;
#define DEVCASE_CK(x) do { if (CUDA_CHECKED(x)) goto out; } while (0)
    DEVCASE_CK(cudaMalloc(&d, sizeof(*h) * n));
    DEVCASE_CK(cudaMalloc(&d_bad, sizeof(*d_bad)));
    DEVCASE_CK(cudaMemcpy(d, h, sizeof(*h) * n, cudaMemcpyHostToDevice));
    DEVCASE_CK(cudaMemcpy(d_bad, &bad, sizeof(bad), cudaMemcpyHostToDevice));
    kern<<<blocks, 256>>>(d, n, d_bad);
    DEVCASE_CK(cudaGetLastError());
    DEVCASE_CK(cudaDeviceSynchronize());
    DEVCASE_CK(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
    *first_bad = bad;
    rc = 0;
out:
    cudaFree(d); cudaFree(d_bad);
#undef DEVCASE_CK
    return rc;
}

typedef struct {
    uint32_t w, m, magic, sh, ref;
} td_mod_case_t;

__global__ void k_verify_td_mod_cases(const td_mod_case_t *__restrict v,
                                      uint32_t n, uint32_t *__restrict first_bad)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const td_mod_case_t c = v[k];
        if (td_mod_magic(c.w, c.m, c.magic, c.sh) != c.ref)
            atomicMin(first_bad, k);
    }
}

/* Device half of the reciprocal gate. slabtest calls the same td_mod_magic()
 * source on the CPU; this gate additionally executes the __umulhi branch used
 * by k_td so a CUDA-codegen/device-only regression cannot hide behind the
 * host implementation. It runs only under --verify. */
static int verify_td_mod_device(void)
{
    enum { NCASE = 4096 };
    td_mod_case_t *h = NULL;
    uint32_t bad = UINT32_MAX, seed = 0x7f4a7c15u;

    h = (td_mod_case_t *)malloc(sizeof(*h) * NCASE);
    if (!h) return -1;
    for (uint32_t k = 0; k < NCASE; k++) {
        uint32_t m, magic, sh, w;
        if (k < 20) {
            static const uint32_t edge[] = {
                2,3,4,5,7,8,15,16,31,32,63,64,127,128,255,256,
                1023,32767,131071,1048575
            };
            m = edge[k];
        } else {
            seed = seed * 1664525u + 1013904223u;
            m = 2u + seed % 1048574u;
        }
        td_magic_build(m, &magic, &sh);
        seed = seed * 1664525u + 1013904223u;
        w = (k & 3u) == 0 ? 0x7fffffffu : (seed & 0x7fffffffu);
        h[k].w = w; h[k].m = m; h[k].magic = magic; h[k].sh = sh;
        h[k].ref = w % m;
    }
    if (run_device_cases(k_verify_td_mod_cases, h, NCASE, 16, &bad)) {
        free(h);
        return -1;
    }
    if (bad != UINT32_MAX) {
        const td_mod_case_t c = h[bad];
        fprintf(stderr,
                "[verify] device td_mod mismatch case %u: w=%u m=%u"
                " magic=%u sh=%u ref=%u\n",
                bad, c.w, c.m, c.magic, c.sh, c.ref);
        free(h);
        return -1;
    }
    free(h);
    return 0;
}

typedef struct {
    uint32_t p, rt, j, magic, kshift, two, ref;
    int32_t  ilo;
} ss_first_case_t;

__global__ void k_verify_ss_first_cases(const ss_first_case_t *__restrict v,
                                        uint32_t n, uint32_t *__restrict first_bad)
{
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const ss_first_case_t c = v[k];
        const uint32_t got = c.two
            ? ss_first<true>(c.p, c.rt, c.j, c.ilo, c.magic, c.kshift)
            : ss_first<false>(c.p, c.rt, c.j, c.ilo, c.magic, c.kshift);
        if (got != c.ref) atomicMin(first_bad, k);
    }
}

/* Device gate for ss_first's two magic forms, against the 64-bit fallback's
 * own formula computed on the host. Each modulus is routed the way the
 * device routes it -- two-step exactly when it fails the one-step bound at J
 * (ss_one_step_fits, the predicate ss_two_step_from splits on), powers of two
 * included -- and gets the magic ss_magic_build builds for that form. Every
 * geometry gets the corners: both sides of the split, both sides of the
 * two-step form's own refusal boundary (the largest modulus it accepts),
 * m = I-1, I-2, a power of two; then random moduli. Each case's inputs come
 * at the proof's corners (rt = m-1, j = jmax, ilo = -ihalf) or at random.
 * Relations would catch a wrong residue too, but only as a smaller relation
 * count; this names the case. Runs only under --verify; slabtest runs the
 * same routing and corners on the CPU, inside `make check`. */
static int verify_ss_first_device(void)
{
    enum { NCASE = 8192, NCORNER = 12 };
    static const uint32_t geo[][2] = {     /* {logI, J} */
        {15, 16384}, {16, 16384}, {16, 32768}, {16, 65536}, {17, 32768},
        {17, 65536}, {18, 65536}
    };
    const uint32_t NG = (uint32_t)(sizeof geo / sizeof geo[0]);
    ss_first_case_t *h = NULL;
    uint32_t bad = UINT32_MAX, seed = 0x2545f491u, n = 0;
    uint32_t ntwo = 0, ncorner2 = 0;
    uint32_t splits[sizeof geo / sizeof geo[0]], tmax[sizeof geo / sizeof geo[0]];

    /* Per geometry: the first modulus the one-step bound refuses (where the
     * device's two-step range starts), and the largest non-power-of-two
     * modulus the two-step proof still accepts (0 if none). */
    for (uint32_t g = 0; g < NG; g++) {
        const uint32_t logI = geo[g][0], J = geo[g][1], I = 1u << logI;
        uint32_t s = 2, t = I - 1;
        while (s < I && ss_one_step_fits(s, J, logI)) s++;
        while (t >= s && ((t & (t - 1)) == 0 || !ss_bound_ok(t, J, logI, 1))) t--;
        splits[g] = s;
        tmax[g] = t >= s ? t : 0;
    }
    h = (ss_first_case_t *)malloc(sizeof(*h) * NCASE);
    if (!h) return -1;
    for (uint32_t tries = 0; n < NCASE && tries < 64u * NCASE; tries++) {
        const uint32_t g = tries % NG, logI = geo[g][0], J = geo[g][1];
        const uint32_t I = 1u << logI, ihalf = I >> 1, k = tries / NG;
        const uint32_t split = splits[g];
        uint32_t m, magic, two;
        seed = seed * 1664525u + 1013904223u;
        switch (k < NCORNER ? k : NCORNER) {    /* the corners first */
        case 0:  m = split - 2; break;
        case 1:  m = split - 1; break;
        case 2:  m = split;     break;
        case 3:  m = split + 1; break;
        case 4:  m = split + 2; break;
        case 5:  m = tmax[g] ? tmax[g] - 1 : 0; break;
        case 6:  m = tmax[g];   break;
        case 7:  m = tmax[g] ? tmax[g] + 1 : 0; break;   /* refused: skipped */
        case 8:  m = I - 1;     break;
        case 9:  m = I - 2;     break;
        case 10: m = I >> 1;    break;                   /* a power of two */
        case 11: m = split | 1u; break;
        default: m = 2u + seed % (I - 2u); break;
        }
        if (m < 2 || m >= I) continue;
        two = !ss_one_step_fits(m, J, logI);
        ss_magic_build(m, J, logI, (int)two, &magic);
        if (!magic) continue;               /* the fallback; not this gate's */
        ss_first_case_t c;
        seed = seed * 1664525u + 1013904223u;
        c.rt = (k & 3u) == 0 ? m - 1u : seed % m;
        seed = seed * 1664525u + 1013904223u;
        c.j = (k % 3u) == 0 ? J : seed % (J + 1u);
        seed = seed * 1664525u + 1013904223u;
        c.ilo = (k % 5u) == 0 ? -(int32_t)ihalf
              : (k % 5u) == 1 ? (int32_t)ihalf - 1
              : (int32_t)(seed % I) - (int32_t)ihalf;
        if (k < NCORNER) {                  /* corner moduli: worst inputs */
            c.rt = m - 1u; c.j = J; c.ilo = -(int32_t)ihalf;
            ncorner2 += two;
        }
        c.p = m; c.magic = magic; c.kshift = SS_KSHIFT(logI); c.two = two;
        c.ref = (uint32_t)(((uint64_t)c.rt * c.j + ((uint64_t)m << 20)
                            - (uint64_t)(int64_t)c.ilo) % m);
        h[n++] = c;
        ntwo += two;
    }
    if (n < NCASE || ntwo < NCASE / 8 || ncorner2 < 3) {
        fprintf(stderr, "[verify] ss_first gate built %u cases (%u two-step,"
                " %u two-step corners); the geometry table no longer reaches"
                " the two-step range\n", n, ntwo, ncorner2);
        free(h);
        return -1;
    }
    if (run_device_cases(k_verify_ss_first_cases, h, NCASE, 32, &bad)) {
        free(h);
        return -1;
    }
    if (bad != UINT32_MAX) {
        const ss_first_case_t c = h[bad];
        fprintf(stderr,
                "[verify] device ss_first mismatch case %u: p=%u rt=%u j=%u"
                " ilo=%d kshift=%u two=%u ref=%u\n",
                bad, c.p, c.rt, c.j, c.ilo, c.kshift, c.two, c.ref);
        free(h);
        return -1;
    }
    printf("[verify] OK: %u device ss_first cases, %u of them two-step"
           " (%u at the corners)\n", n, ntwo, ncorner2);
    free(h);
    return 0;
}

extern "C" int run_bench(const fb_t *fb, const fb_t *fbs, const qlat_t *L,
                         const poly_t *POLY, const bench_cfg_t *cfg)
{
    const uint32_t I = 1u << cfg->logI;
    const uint32_t xmax = I * cfg->J;
    const int log_region = cfg->log_region;
    const uint32_t nregion = xmax >> log_region;
    const uint32_t nbitword = xmax >> 5;   /* survivor bitmap, 1 bit/position */
    const int log_super = log_region + 7;             /* 128 regions/super */
    const uint32_t nsuper = xmax >> log_super;
    const uint32_t CINIT = (cfg->cell_bits == 16) ? 4096u : 255u;
    uint32_t BOUND = 0;
    size_t optin_smem_limit = 0;
    norm_t N;

    if (!fb_is_transform_validated(fb) ||
        (fbs && fbs->n && !fb_is_transform_validated(fbs))) {
        fprintf(stderr,
                "run_bench: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or uploading it\n");
        return -1;
    }
    /* Validate the threshold before norm_setup(), allocations, or launches.
     * run_bench is an exported consumer boundary and must remain safe even
     * when called directly instead of through bench_main's CLI parser. */
    if (sieve_bound_checked(cfg->scale, cfg->allowance, CINIT, &BOUND,
                            cfg->side == 1
                                ? "run_bench side 1 survivor parameters"
                                : "run_bench side 0 survivor parameters"))
        return -1;

    /* Exact trial division constructs the full homogeneous norm. Reject a
     * shape that cannot fit before allocating or launching anything on the
     * GPU; the logarithmic sieve itself remains usable at any supported
     * degree. */
    memset(&N, 0, sizeof(N));
    norm_setup(&N, POLY, L, cfg->logI, cfg->J, cfg->scale, cfg->side == 1);
    if (cfg->td && !norm_fits_exact(&N, BN_LIMBS * 32)) {
        const double bits = norm_exact_bound_bits(&N);
        const int need = bn_limbs_for_bits(bits);
        fprintf(stderr, "  exact degree-%d norm may require %.2f bits;"
                " the trial-division type holds %d\n",
                POLY->deg, bits, BN_LIMBS * 32);
        if (need)
            fprintf(stderr, "  rebuild with `make BN_LIMBS=%d`\n", need);
        else
            fprintf(stderr, "  no supported BN_LIMBS is wide enough"
                    " (the maximum is 16, i.e. 512 bits)\n");
        return -1;
    }

    if (cuda_optin_smem_limit(&optin_smem_limit)) return -1;

    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));
    printf("  device memory: %.2f GB free of %.2f GB\n",
           freeB / 1073741824.0, totalB / 1073741824.0);
    if (cfg->verify) {
        printf("[verify] direct-TD reciprocal on device (__umulhi path)...\n");
        if (verify_td_mod_device()) return -1;
        printf("[verify] OK: 4096 device reciprocal cases through w=0x7fffffff\n");
        printf("[verify] small-sieve first offset, one- and two-step forms...\n");
        if (verify_ss_first_device()) return -1;
    }

    int td_failed = 0;      /* a failed gate must reach the exit status */
    dev_bufs D; memset(&D, 0, sizeof(D));
    CK(cudaMalloc(&D.primes, (size_t)fb->n * 4));
    CK(cudaMalloc(&D.roots,  (size_t)fb->n * 4));
    CK(cudaMalloc(&D.plat,   (size_t)fb->n * sizeof(plat_t)));
    CK(cudaMalloc(&D.overflow, 4));
    CK(cudaMalloc(&D.nproj, 4));
    CK(cudaMalloc(&D.nlost, 8));
    CK(cudaMemcpy(D.primes, fb->primes, (size_t)fb->n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.roots,  fb->roots,  (size_t)fb->n * 4, cudaMemcpyHostToDevice));

    /* ---- slices: bucket record hint -> log p ---- */
    uint16_t *hslice = NULL, *hlogp = NULL;
    uint32_t nslice_pow2 = 1;
    int32_t nslice_rc = fb_build_slices(fb, &hslice, &hlogp, &nslice_pow2);
    if (nslice_rc < 0) {
        report_slice_build_error();
        cudaFree(D.primes); cudaFree(D.roots); cudaFree(D.plat);
        cudaFree(D.overflow); cudaFree(D.nproj); cudaFree(D.nlost);
        return -1;
    }
    uint32_t nslice = (uint32_t)nslice_rc;
    printf("  factor base cut into %u slices (padded to %u), log p in [%u,%u] bits\n",
           nslice, nslice_pow2, hlogp[0], hlogp[nslice - 1]);
    CK(cudaMalloc(&D.slice, (size_t)fb->n * 2));
    CK(cudaMalloc(&D.slice_logp, (size_t)nslice_pow2 * 2));
    CK(cudaMemcpy(D.slice, hslice, (size_t)fb->n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.slice_logp, hlogp, (size_t)nslice_pow2 * 2, cudaMemcpyHostToDevice));

    /* ---- small primes: transform on the host (a few thousand entries) and
     * split into the three load-balance tiers. Entries arrive sorted by p, so
     * the tier boundaries are just two indices. ---- */
    uint32_t nsmall = 0, nblk = 0, nwrp = 0, nsplit = 0;
    uint32_t *hsp = NULL, *hsrt = NULL, *hsg = NULL; uint16_t *hslp = NULL;
    uint32_t *hsmag = NULL;
    /* per-q HOST work, billed separately: it is invisible to cudaEvent timing
     * and Goal 1 is a claim about host demand. */
    double h_ms_transform = 0, h_ms_sort = 0, h_ms_xfer = 0;
    if (cfg->small_sieve && fbs && fbs->n) {
        uint32_t i, k = 0, nrow = 0, nprj = 0;
        /* PINNED, not malloc'd. These four are the only per-special-q host->
         * device transfer, and on this box (WSL2) a pageable cudaMemcpy of
         * this size costs ~1.5 ms per call against ~0.1 ms pinned -- measured
         * at 6.0 ms vs 0.3 ms for the four together. That is per side, so
         * ~12 ms per special-q of pure host overhead, which dwarfs everything
         * else in this block and is a Goal-1 cost. */
        CK(cudaHostAlloc((void **)&hsp,  (size_t)fbs->n * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&hsrt, (size_t)fbs->n * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&hslp, (size_t)fbs->n * 2, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&hsg,  (size_t)fbs->n * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&hsmag, (size_t)fbs->n * 4, cudaHostAllocDefault));
        h_ms_transform = host_ms();
        for (i = 0; i < fbs->n; i++) {
            uint32_t q = fbs->primes[i], r = fbs->roots[i], rt, g, m;
            m = pl_transform_fast(q, r, L->a0, L->a1, L->b0, L->b1, &rt, &g);
            if (g > 1) nrow++;
            if (r >= q) nprj++;
            hsp[k] = m; hsrt[k] = rt; hsg[k] = g;
            hslp[k] = fbs->logp[i];
            k++;
        }
        nsmall = k;
        h_ms_transform = host_ms() - h_ms_transform;

        /* Tier by the EFFECTIVE modulus m, not by q: an entry with q = 32768
         * and g = 32768 has m = 1 and hits every position in its rows, so
         * leaving it in the thread-per-entry tier would hand one thread the
         * whole region. Sorting by m puts every entry in the tier sized for
         * the number of hits it actually produces.
         *
         * This was an insertion sort, which is O(n^2) and ran ~3.3M
         * comparisons at nsmall ~3.6K. `qsort` is NOT a drop-in replacement:
         * the four arrays below are parallel and must be permuted together,
         * and no C library sort can do that. Sort a permutation of indices by
         * the key, then scatter -- which also keeps the device side SoA, which
         * is what the small-sieve kernel wants. */
        h_ms_sort = host_ms();
        {
            uint32_t *idx = (uint32_t *)malloc((size_t)nsmall * 4);
            uint32_t *tp  = (uint32_t *)malloc((size_t)nsmall * 4);
            uint32_t *trt = (uint32_t *)malloc((size_t)nsmall * 4);
            uint32_t *tg  = (uint32_t *)malloc((size_t)nsmall * 4);
            uint16_t *tlp = (uint16_t *)malloc((size_t)nsmall * 2);
            for (i = 0; i < nsmall; i++) idx[i] = i;
            /* stable_sort, not sort: the insertion sort this replaces was
             * stable, and ties in m are common (many entries share a modulus).
             * Stability keeps the output bit-identical to the old code, which
             * is what makes the bitmap regression test meaningful. */
            std::stable_sort(idx, idx + nsmall,
                             [hsp](uint32_t a, uint32_t b) { return hsp[a] < hsp[b]; });
            for (i = 0; i < nsmall; i++) {
                uint32_t s = idx[i];
                tp[i] = hsp[s]; trt[i] = hsrt[s]; tg[i] = hsg[s]; tlp[i] = hslp[s];
            }
            memcpy(hsp,  tp,  (size_t)nsmall * 4);
            memcpy(hsrt, trt, (size_t)nsmall * 4);
            memcpy(hsg,  tg,  (size_t)nsmall * 4);
            memcpy(hslp, tlp, (size_t)nsmall * 2);
            free(idx); free(tp); free(trt); free(tg); free(tlp);
        }
        h_ms_sort = host_ms() - h_ms_sort;

        /* Reciprocals AFTER the sort: the key is the modulus itself, so
         * building them earlier would mean permuting them too, for no gain.
         * And after the tiers, because the thread tier's two-step range
         * starts at nsplit and each magic is proved for the form its index
         * gets (ss_bound_ok). */
        ss_tiers(hsp, nsmall, &nblk, &nwrp);
        nsplit = ss_two_step_from(hsp, nsmall, nwrp, cfg->J, cfg->logI);
        for (i = 0; i < nsmall; i++)
            ss_magic_build(hsp[i],
                           hsg[i] > 1 ? cfg->J / hsg[i] : cfg->J,
                           cfg->logI, i >= nsplit, &hsmag[i]);

        CK(cudaMalloc(&D.smag, (size_t)nsmall * 4));
        CK(cudaMalloc(&D.sp,  (size_t)nsmall * 4));
        CK(cudaMalloc(&D.srt, (size_t)nsmall * 4));
        CK(cudaMalloc(&D.sg,  (size_t)nsmall * 4));
        CK(cudaMalloc(&D.slp, (size_t)nsmall * 2));
        /* Drain everything queued earlier (the factor-base uploads above are
         * large and asynchronous) BEFORE starting the clock -- otherwise the
         * trailing sync below bills their tail to this transfer and reports
         * milliseconds for 54 KB. */
        CK(cudaDeviceSynchronize());
        h_ms_xfer = host_ms();
        CK(cudaMemcpy(D.sp,  hsp,  (size_t)nsmall * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D.srt, hsrt, (size_t)nsmall * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D.sg,  hsg,  (size_t)nsmall * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D.slp, hslp, (size_t)nsmall * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D.smag, hsmag, (size_t)nsmall * 4, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        h_ms_xfer = host_ms() - h_ms_xfer;
        {   double upd = 0; uint32_t xm = (1u << cfg->logI) * cfg->J;
            for (i = 0; i < nsmall; i++) upd += (double)xm / hsp[i] / hsg[i];
            printf("  small sieve: %u entries (%u block-tier m<%u, %u warp-tier m<%u,"
                   " %u thread-tier), %u with a row divisor, %u projective,"
                   " %.3e updates\n",
                   nsmall, nblk, SS_BLOCK_CUT, nwrp - nblk, SS_WARP_CUT,
                   nsmall - nwrp, nrow, nprj, upd);
        }
    }

    /* ---- norm initialisation constants ---- */
    /* las's survivor test is S = max(T - sum, 0) <= bound with
     * bound = round(scale * lambda * lpb). Ours holds CINIT - T + sum, so
     * S = CINIT - cell and the test becomes cell >= CINIT - bound. */
    /* las: bound = (unsigned char)(lambda*lpb*scale + LOGNORM_GUARD_BITS),
     * las-norms.cpp:270 -- a TRUNCATING cast plus a guard bit, not a round.
     * With the exact scales (1.275 / 1.925, not the 2-dp values las prints)
     * this reproduces both of las's bounds exactly: 143 and 141. The old
     * round(scale*allowance) matched only because the rounded scales happened
     * to compensate. */
    uint32_t tconst;
    {
        float t = norm_target_host(&N, 0, cfg->J / 2);
        int ti = (int)(t + 0.5f);
        tconst = (ti < 1) ? 1u : ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
        printf("  init T at (i=0, j=J/2) = %u; survivor bound = %u"
               " (scale %.3f x %.2f bits)\n", tconst, BOUND, cfg->scale, cfg->allowance);
    }

    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4);

    int blocks = cfg->blocks ? cfg->blocks : 48 * 6;
    /* Fill's grid is absolute, not per-SM -- see FILL_BLOCKS_DEFAULT. */
    const int fblocks = cfg->fill_blocks ? cfg->fill_blocks : FILL_BLOCKS_DEFAULT;
    const int fthreads = cfg->fill_threads ? cfg->fill_threads : FILL_THREADS_DEFAULT;
    float t_trans = 0, t_fill = 0, t_l1 = 0, t_l2 = 0;

    /* ---- stage T ---- *
     * Untimed warm-up. k_transform is the FIRST kernel of the run, so without
     * this it absorbs the whole one-time CUDA cost -- module load for a
     * four-architecture fatbin, context setup -- and reports it divided by
     * reps. Measured on WSL2 that fixed cost is ~170-220 ms, which put the
     * "transform" line at 71.2 ms at --reps 3 against a true 0.55 ms: a 98x
     * swing across --reps 3..1000 while fill and apply moved under 1%. Two
     * different people compared that number across GPUs before anyone noticed
     * it was measuring startup. The memsets follow the warm-up because nproj
     * and nlost are accumulators divided by reps. */
    k_transform<false><<<blocks, cfg->threads>>>(D.primes, D.roots, D.plat, fb->n,
        cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1, D.nproj, D.nlost, NULL);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(D.nproj, 0, 4));
    CK(cudaMemset(D.nlost, 0, 8));
    cudaEventRecord(e0);
    for (int rep = 0; rep < cfg->reps; rep++)
        k_transform<false><<<blocks, cfg->threads>>>(D.primes, D.roots, D.plat, fb->n,
            cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1, D.nproj, D.nlost, NULL);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    t_trans = time_kernel(e0, e1) / cfg->reps;

    uint32_t hproj = 0;
    unsigned long long hlost = 0;
    CK(cudaMemcpy(&hproj, D.nproj, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hlost, D.nlost, 8, cudaMemcpyDeviceToHost));
    hproj /= cfg->reps;
    hlost /= (unsigned)cfg->reps;

    /* ---- expected record count, for sizing ---- */
    double exp_rec = 0;
    for (uint32_t i = 0; i < fb->n; i++) exp_rec += (double)xmax / fb->primes[i];
    uint64_t est = (uint64_t)(exp_rec * 1.15) + 4096;

    printf("  transformed roots: %u row-confined (g > 1), %llu positions lost%s\n",
           hproj, hlost, hlost ? "  ** move these to the small tier **" : "");
    printf("  analytic records : %.3e   (sized with 1.15x margin)\n", exp_rec);

    uint32_t cap = 0;               /* records per region, set by the fill path */
    if (cfg->fill_mode == FILL_ATOMIC) {
        cap = (uint32_t)(est / nregion) + 256;
        size_t need = (size_t)nregion * cap * cfg->record_bytes;
        printf("  single-level: %u buckets x cap %u x %d B = %.2f GB\n",
               nregion, cap, cfg->record_bytes, need / 1073741824.0);
        if (need + 64u * 1024 * 1024 > freeB) {
            printf("  SKIP: does not fit in free device memory\n"); return 1;
        }
        CK(cudaMalloc(&D.cursor, (size_t)nregion * 4));
        CK(cudaMalloc(&D.out, need));
        CK(cudaMemset(D.overflow, 0, 4));
/* One dispatch for the control fill AND every concurrency arm below. Two copies
 * of this three-way record-width switch would let the experiment silently
 * measure a kernel specialisation the control never runs. */
#define FILL_ONE(GRID, STREAM, PLAT, CUR, OUT, OVF)                          \
    do {                                                                     \
        if (cfg->record_bytes == 2)                                          \
            k_fill_atomic<2, false><<<(GRID), fthreads, 0, (STREAM)>>>(      \
                (PLAT), D.slice, fb->n, xmax, cfg->logI, log_region,         \
                (CUR), (OUT), cap, (OVF), NULL, NULL, 0, 0u);                       \
        else if (cfg->record_bytes == 4)                                     \
            k_fill_atomic<4, false><<<(GRID), fthreads, 0, (STREAM)>>>(      \
                (PLAT), D.slice, fb->n, xmax, cfg->logI, log_region,         \
                (CUR), (OUT), cap, (OVF), NULL, NULL, 0, 0u);                       \
        else                                                                 \
            k_fill_atomic<8, false><<<(GRID), fthreads, 0, (STREAM)>>>(      \
                (PLAT), D.slice, fb->n, xmax, cfg->logI, log_region,         \
                (CUR), (OUT), cap, (OVF), NULL, NULL, 0, 0u);                       \
    } while (0)

        cudaEventRecord(e2);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
            FILL_ONE(fblocks, 0, D.plat, D.cursor, D.out, D.overflow);
        }
        cudaEventRecord(e3);
        CK(cudaEventSynchronize(e3));
        CK(cudaGetLastError());
        t_fill = time_kernel(e2, e3) / cfg->reps;

        /* ---- item 1: is fill's knee per-KERNEL or per-DEVICE? ----
         *
         * Every card swept so far plateaus at the same ABSOLUTE block count,
         * which is the shape of a device limit; but the 4090 is 1.80x SLOWER
         * at fill than a 5070 with 1.5x its bandwidth, which no device limit
         * explains. The `ncu` profile named the candidate: waves per SM = 1.00,
         * so the whole grid is resident at once and a block that draws a heavy
         * chunk has no queued block to backfill its slot -- SMs idle 26.5% of
         * elapsed cycles. A second INDEPENDENT kernel could occupy those.
         *
         *   CONCURRENT  N workspaces, N streams, fblocks each   -- N workspaces
         *   SERIAL      the same N launches, one stream         -- N workspaces
         *   WIDE        one launch at N * fblocks               -- ONE workspace
         *
         * CONCURRENT vs SERIAL is the comparison: identical work, identical
         * launches, only the stream assignment differs. CONCURRENT ~= SERIAL
         * says the device is saturated and the plateau is real; CONCURRENT <
         * SERIAL says one kernel cannot feed the card.
         *
         * WIDE DOES 1/N THE WORK OF THE OTHER TWO and is not comparable to
         * them in raw ms. k_fill_atomic is a grid-stride loop over fb->n, so
         * N*fblocks blocks still process one factor base into one workspace.
         * It answers a different question -- can a single kernel buy the same
         * capacity just by being wider? -- and is only ever quoted per
         * workspace, which is why every row below is normalised.
         *
         * ARM ORDER AND DRIFT. Boost clocks decay under sustained load, and a
         * fixed arm order would bias whichever arm runs first -- here, in
         * exactly the direction of the conclusion. Arms are therefore
         * interleaved and each keeps its MINIMUM over the outer repeats, the
         * same best-of-N every other A/B in this file uses.
         *
         * WHAT THE WORKSPACES DO AND DO NOT MODEL. Each gets its own plat,
         * cursor and bucket arrays, so the concurrent arm streams N x 42 B per
         * entry from DISTINCT addresses -- sharing one plat would hand it an
         * L2 advantage no pair of real special-q enjoys. `slice` stays shared
         * because two real q on one factor base share it too. But the copies
         * hold IDENTICAL plat values, and the values are the walk: two real q
         * would write different per-region distributions, while these march
         * their bucket frontiers in lockstep. Finding 81 makes that exactly
         * the variable fill is bound by (read-modify-write on partly filled
         * lines), so this measures the SATURATION question honestly and does
         * NOT predict how two real q interleave. That needs the pipeline. */
        if (cfg->fill_streams > 1) {
            int NS = cfg->fill_streams;
            /* run_bench is an exported boundary; the argv clamp lives in a
             * different translation unit and cannot be relied on here. */
            if (NS > FILL_STREAMS_MAX) NS = FILL_STREAMS_MAX;
            const size_t wsz = (size_t)nregion * cap * cfg->record_bytes;
            const size_t psz = (size_t)fb->n * sizeof(plat_t);
            size_t extra = (size_t)(NS - 1) * (wsz + psz + (size_t)nregion * 4);
            size_t fnow = 0, tnow = 0;
            int maxgrid = 0;
            CK(cudaMemGetInfo(&fnow, &tnow));
            CK(cudaDeviceGetAttribute(&maxgrid, cudaDevAttrMaxGridDimX, 0));
            printf("\n  --- fill concurrency, %d workspaces (item 1) ---\n", NS);
            printf("  extra workspaces need %.2f GB of %.2f GB free\n",
                   extra / 1073741824.0, fnow / 1073741824.0);
            if (extra + 64u * 1024 * 1024 > fnow) {
                printf("  SKIP: %d workspaces need %.2f GB but only %.2f GB is"
                       " free%s\n", NS, extra / 1073741824.0,
                       fnow / 1073741824.0,
                       NS > 2 ? "; try --fill-streams 2" : "");
            } else if ((double)fblocks * NS > (double)maxgrid) {
                printf("  SKIP: the WIDE arm would need %d x %d = %.0f blocks,"
                       " past this device's grid-x limit %d\n",
                       fblocks, NS, (double)fblocks * NS, maxgrid);
            } else {
                plat_t   *wplat[FILL_STREAMS_MAX];
                uint32_t *wcur [FILL_STREAMS_MAX];
                uint8_t  *wout [FILL_STREAMS_MAX];
                cudaStream_t st[FILL_STREAMS_MAX];
                uint32_t *xovf = NULL;   /* NOT D.overflow -- see below */
                int k;
                wplat[0] = D.plat; wcur[0] = D.cursor; wout[0] = D.out;
                for (k = 1; k < NS; k++) {
                    CK(cudaMalloc(&wplat[k], psz));
                    CK(cudaMalloc(&wcur[k], (size_t)nregion * 4));
                    CK(cudaMalloc(&wout[k], wsz));
                    CK(cudaMemcpy(wplat[k], D.plat, psz, cudaMemcpyDeviceToDevice));
                }
                for (k = 0; k < NS; k++) CK(cudaStreamCreate(&st[k]));
                /* The run's overflow count is REPORTED and gates --verify, and
                 * these arms issue (2*NS+1)*reps more fills into the same
                 * buckets. Accumulating them into D.overflow would inflate the
                 * figure an operator sizes the rerun from, and would fail
                 * --verify on records this harness dropped rather than the
                 * measured configuration. Give the experiment its own counter
                 * and leave D.overflow untouched. */
                CK(cudaMalloc(&xovf, 4));
                CK(cudaMemset(xovf, 0, 4));

                float t_arm[3] = { 1e30f, 1e30f, 1e30f };
                const int outer = 3;
                for (int pass = 0; pass < outer; pass++) {
                    for (int arm = 0; arm < 3; arm++) {
                        const int grid = (arm == 2) ? fblocks * NS : fblocks;
                        const int nws  = (arm == 2) ? 1 : NS;
                        cudaEventRecord(e2);
                        for (int rep = 0; rep < cfg->reps; rep++) {
                            for (k = 0; k < nws; k++) {
                                if (arm == 0)
                                    CK(cudaMemsetAsync(wcur[k], 0,
                                        (size_t)nregion * 4, st[k]));
                                else    /* same synchronous memset the control
                                         * fill uses, so SERIAL and the control
                                         * share one harness */
                                    CK(cudaMemset(wcur[k], 0, (size_t)nregion * 4));
                            }
                            for (k = 0; k < nws; k++)
                                FILL_ONE(grid, arm == 0 ? st[k] : 0,
                                         wplat[k], wcur[k], wout[k], xovf);
                        }
                        /* Do not rely on legacy default-stream semantics to
                         * fence the blocking streams: one --default-stream
                         * per-thread in NVCCFLAGS would silently turn e3 into
                         * a launch-issue timestamp and report a 20x win. */
                        if (arm == 0)
                            for (k = 0; k < NS; k++) CK(cudaStreamSynchronize(st[k]));
                        cudaEventRecord(e3);
                        CK(cudaEventSynchronize(e3));
                        CK(cudaGetLastError());
                        float t = time_kernel(e2, e3) / cfg->reps;
                        if (t < t_arm[arm]) t_arm[arm] = t;
                    }
                }
                {
                    uint32_t xo = 0;
                    CK(cudaMemcpy(&xo, xovf, 4, cudaMemcpyDeviceToHost));
                    printf("  best of %d passes x %d reps, arms interleaved\n",
                           outer, cfg->reps);
                    printf("  CONCURRENT %2d x %5d blocks, %d streams : %8.3f ms"
                           "  = %7.3f ms per workspace\n",
                           NS, fblocks, NS, t_arm[0], t_arm[0] / NS);
                    printf("  SERIAL     %2d x %5d blocks, one stream : %8.3f ms"
                           "  = %7.3f ms per workspace\n",
                           NS, fblocks, t_arm[1], t_arm[1] / NS);
                    printf("  WIDE        1 x %5d blocks, ONE workspace: %8.3f ms"
                           "  = %7.3f ms per workspace\n",
                           fblocks * NS, t_arm[2], t_arm[2]);
                    printf("  concurrent/serial %.4f   (<1 = one kernel cannot"
                           " feed this card; same work, only the streams differ)\n",
                           t_arm[0] / t_arm[1]);
                    printf("  concurrent vs the best single kernel: %.4f"
                           "  (wide %.3f, single %.3f ms per workspace)\n",
                           (t_arm[0] / NS) / (t_arm[2] < t_fill ? t_arm[2] : t_fill),
                           t_arm[2], t_fill);
                    if (xo) printf("  note: %u records overflowed inside the"
                                   " concurrency arms (own counter; the run's"
                                   " own overflow figure is unaffected)\n", xo);
                }
                cudaFree(xovf);
                for (k = 0; k < NS; k++) CK(cudaStreamDestroy(st[k]));
                for (k = 1; k < NS; k++) {
                    cudaFree(wplat[k]); cudaFree(wcur[k]); cudaFree(wout[k]);
                }
            }
        }
#undef FILL_ONE
    } else {
        /* Both levels stage their fan-out in a fixed number of shared buffers,
         * so the split has to fit: L1 needs nsuper <= L1_NBUF and L2 needs
         * regions_per_super <= L2_NBUF. With 128 each, two-level tops out at
         * 128*128 = 16384 regions -- exactly I15e at region 2^15, and no more.
         * Past that it silently indexed cnt[] out of bounds and livelocked in
         * the retry loop. Refuse instead: the operating point that actually
         * won (region 2^14, 32768 regions) is out of reach for a two-level
         * split at these buffer sizes and would need a third level. */
        if (nsuper > L1_NBUF || (1u << (log_super - log_region)) > L2_NBUF) {
            printf("  two-level cannot express this split: %u super-buckets"
                   " (max %u) x %u regions each (max %u).\n"
                   "  Use --mode atomic, which is 2.7x faster anyway"
                   " (RESULTS.md finding 1).\n",
                   nsuper, L1_NBUF, 1u << (log_super - log_region), L2_NBUF);
            return 1;
        }
        uint32_t l1cap = (uint32_t)(est / nsuper) + 4096;
        uint32_t l2cap = (uint32_t)(est / nregion) + 256;
        size_t need1 = (size_t)nsuper * l1cap * 4;
        size_t need2 = (size_t)nregion * l2cap * cfg->record_bytes;
        printf("  two-level: L1 %u super x cap %u x 4 B = %.2f GB;"
               " L2 %u regions x cap %u x %d B = %.2f GB\n",
               nsuper, l1cap, need1 / 1073741824.0,
               nregion, l2cap, cfg->record_bytes, need2 / 1073741824.0);
        if (need1 + need2 + 64u * 1024 * 1024 > freeB) {
            printf("  SKIP: does not fit in free device memory\n"); return 1;
        }
        CK(cudaMalloc(&D.l1, need1));
        CK(cudaMalloc(&D.l1cnt, (size_t)nsuper * 4));
        CK(cudaMalloc(&D.cursor, (size_t)nregion * 4));
        CK(cudaMalloc(&D.out, need2));
        CK(cudaMemset(D.overflow, 0, 4));

        cudaEventRecord(e2);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(cudaMemset(D.l1cnt, 0, (size_t)nsuper * 4));
            /* Its own DEFAULT, but --fill-blocks still reaches it. k_fill_l1 is
             * the twolevel path's level-1 kernel and has never been swept; the
             * 1152 x 32 geometry was measured on k_fill_atomic, a different
             * kernel with a different write pattern, so inheriting it as a
             * default would move an experimental path nobody remeasured.
             *
             * Pinning it outright was worse: it made --fill-blocks a silent
             * no-op under --mode twolevel while the startup line still printed
             * the requested number, which is how a run gets quoted as a sweep
             * that never happened. An explicit value wins here as everywhere
             * else; only the default differs. Threads stay at 512 -- never
             * swept either, and --fill-threads has no measured meaning for this
             * kernel. */
            k_fill_l1<<<cfg->fill_blocks ? cfg->fill_blocks : FILL_L1_BLOCKS,
                        512>>>(D.plat, fb->n, xmax, cfg->logI, log_super,
                D.l1cnt, D.l1, l1cap, D.overflow);
        }
        cudaEventRecord(e3);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
            if (cfg->record_bytes == 2)
                k_fill_l2<2><<<nsuper, 512>>>(D.l1, D.l1cnt, l1cap, log_region,
                    log_super, D.cursor, D.out, l2cap, D.overflow);
            else
                k_fill_l2<4><<<nsuper, 512>>>(D.l1, D.l1cnt, l1cap, log_region,
                    log_super, D.cursor, D.out, l2cap, D.overflow);
        }
        cudaEventRecord(e4);
        CK(cudaEventSynchronize(e4));
        CK(cudaGetLastError());
        t_l1 = time_kernel(e2, e3) / cfg->reps;
        t_l2 = time_kernel(e3, e4) / cfg->reps;
        t_fill = t_l1 + t_l2;
    }

    uint32_t ovf = 0;
    CK(cudaMemcpy(&ovf, D.overflow, 4, cudaMemcpyDeviceToHost));

    /* ---- count what actually landed ---- */
    uint64_t landed = 0;
    {
        uint32_t *h = (uint32_t *)malloc((size_t)nregion * 4);
        CK(cudaMemcpy(h, D.cursor, (size_t)nregion * 4, cudaMemcpyDeviceToHost));
        uint32_t mx = 0;
        for (uint32_t i = 0; i < nregion; i++) { landed += h[i]; if (h[i] > mx) mx = h[i]; }
        printf("  records landed   : %llu   (max bucket %u, mean %.0f -> imbalance %.2fx)\n",
               (unsigned long long)landed, mx, (double)landed / nregion,
               mx / ((double)landed / nregion));
        /* PER-REGION assertion, not just the total. A global count cannot tell
         * "right total, wrong region" from "right" -- and every placement bug
         * this project has hit (transposed basis, projective reciprocal) had
         * exactly the right total. This FAILS the run rather than printing. */
        if (cfg->verify) {
            uint32_t *ref = (uint32_t *)malloc((size_t)nregion * 4);
            uint64_t tot;
            uint32_t bad = 0, first = 0;
            printf("  [verify] per-region reference (single-threaded)...\n");
            tot = verify_count_updates(fb, L, cfg->logI, cfg->J, log_region, ref);
            for (uint32_t i = 0; i < nregion; i++)
                if (ref[i] != h[i]) { if (!bad) first = i; bad++; }
            if (bad || tot != landed) {
                printf("  [verify] FAILED: %u of %u regions differ (first: region %u,"
                       " gpu %u ref %u); totals gpu %llu ref %llu\n",
                       bad, nregion, first, h[first], ref[first],
                       (unsigned long long)landed, (unsigned long long)tot);
                free(ref); free(h); return -1;
            }
            printf("  [verify] OK: all %u regions match the CPU reference exactly\n"
                   "           (counts are of records ATTEMPTED; bucket overflow is\n"
                   "            gated separately, above)\n", nregion);
            free(ref);
        }
        free(h);
    }
    if (ovf) {
        printf("  ** OVERFLOW: %u records dropped -- resize and re-run **\n", ovf);
        /* The per-region gate CANNOT see this. cursor[] is incremented before
         * the cap test, so it counts records ATTEMPTED, and the CPU reference
         * counts the same thing -- they agree exactly while apply silently
         * truncates each bucket at cap and drops the excess. Overflow has to be
         * fatal on its own, or verification certifies a sieve that lost data. */
        if (cfg->verify) {
            fprintf(stderr, "[verify] FAILED: %u records overflowed their bucket."
                    " The per-region count gate cannot detect this (cursors count\n"
                    "         attempts, not stores), so it is failed explicitly.\n", ovf);
            return -1;
        }
    }

    /* ---- stage A: apply ---- */
    float t_apply = 0;
    if (cfg->stage != STAGE_FILL) {
        if (cfg->fill_mode != FILL_ATOMIC || cfg->record_bytes != 4) {
            printf("\n  apply needs single-level 4 B records (--mode atomic"
                   " --record-bytes 4); skipping\n");
        } else {
            const int CB = cfg->cell_bits;
            const uint32_t ncell = 1u << log_region;
            const size_t smem = (size_t)ncell * CB / 8 + (size_t)nslice_pow2 * 2;
            const uint32_t maxsurv = 1u << 22;
            /* gate 5: the one position whose pipeline-produced cell we read back */
            const uint32_t probe_x = (cfg->probe_j != 0xFFFFFFFFu)
                ? (uint32_t)((cfg->probe_i + (1 << (cfg->logI - 1)))
                             + ((uint64_t)cfg->probe_j << cfg->logI))
                : 0xFFFFFFFFu;
            int athr = cfg->apply_threads ? cfg->apply_threads : 512;
            /* region 0 is the j=0 row and is legitimately almost empty --
             * gating on it would check nothing. Use a mid-range region. */
            const uint32_t dbgreg = nregion / 2;

            CK(cudaMalloc(&D.probe, 8));
            CK(cudaMemset(D.probe, 0, 8));
            CK(cudaMalloc(&D.nsurv, 4));
            if (cfg->survbits || cfg->other_bits) {
                CK(cudaMalloc(&D.survbits, (size_t)nbitword * 4));
                if (cfg->survbits)
                    printf("  writing a survivor bitmap (1 bit per position, x"
                           " order) to %s (%.0f MB)\n",
                           cfg->survbits, nbitword * 4 / 1048576.0);
            }
            if (cfg->not_both_even)
                printf("  las `not_both_even` filter ON: positions with i and j"
                       " both even cannot survive\n");
            if (cfg->dump) {
                CK(cudaMalloc(&D.dumpbuf, (size_t)xmax));
                printf("  dumping the region in las byte convention to %s"
                       " (%.0f MB)\n", cfg->dump, xmax / 1048576.0);
            }
            if (cfg->verify) CK(cudaMalloc(&D.dbg, (size_t)ncell * 2));

            printf("\n  apply: %u regions x %u cells x %d bit = %zu B smem/block,"
                   " %d threads, %s, norm=%s\n",
                   nregion, ncell, CB, smem, athr,
                   cfg->apply_atomic ? "smem atomicAdd" : "PLAIN (racy probe)",
                   cfg->norm_mode == NORM_CONST ? "const" : "horner");
            if (smem > optin_smem_limit) {
                printf("  SKIP: %zu B exceeds this device's %zu B opt-in"
                       " shared-memory limit\n", smem, optin_smem_limit);
                goto after_apply;
            }

            /* ST: the SSTAIL instance. The production configuration (16-bit
             * cells, atomic, Horner) picks it with ss_needs_tail exactly as
             * the pipeline does, so --verify checks the instances the
             * pipeline runs, tail and no tail; the other configurations keep
             * the tail instance, which is right with or without one. */
#define LAUNCH_APPLY(CBV, AT, NM, ST)                                          \
            do {                                                               \
                CK(cudaFuncSetAttribute(k_apply<CBV, AT, NM, false, ST>,       \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));  \
                cudaEventRecord(e3);                                           \
                for (int rep = 0; rep < cfg->reps; rep++) {                    \
                    CK(cudaMemset(D.nsurv, 0, 4));                             \
                    if (D.survbits)                                            \
                        CK(cudaMemset(D.survbits, 0, (size_t)nbitword * 4));   \
                    k_apply<CBV, AT, NM, false, ST><<<nregion, athr, smem>>>(  \
                        (const uint32_t *)D.out, D.cursor, cap,                \
                        cfg->logI, log_region, D.slice_logp, nslice_pow2,      \
                        N, CINIT, CINIT - BOUND, tconst, D.dumpbuf,            \
                        D.nsurv,                                               \
                        D.dbg, dbgreg, D.sp, D.srt, D.sg, D.slp,             \
                        D.smag,                                               \
                        nsmall, nblk, nwrp, nsplit, probe_x, D.probe,          \
                        D.survbits, cfg->not_both_even, 0, 0u);                   \
                }                                                              \
                cudaEventRecord(e4);                                           \
                CK(cudaEventSynchronize(e4));                                  \
                CK(cudaGetLastError());                                        \
                t_apply = time_kernel(e3, e4) / cfg->reps;                      \
            } while (0)

            if (CB == 16) {
                if (cfg->apply_atomic) {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(16, 1, NORM_CONST, true);
                    else if (ss_needs_tail(nsplit, nsmall))
                                                      LAUNCH_APPLY(16, 1, NORM_HORNER, true);
                    else                              LAUNCH_APPLY(16, 1, NORM_HORNER, false);
                } else {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(16, 0, NORM_CONST, true);
                    else                              LAUNCH_APPLY(16, 0, NORM_HORNER, true);
                }
            } else {
                if (cfg->apply_atomic) {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(8, 1, NORM_CONST, true);
                    else                              LAUNCH_APPLY(8, 1, NORM_HORNER, true);
                } else {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(8, 0, NORM_CONST, true);
                    else                              LAUNCH_APPLY(8, 0, NORM_HORNER, true);
                }
            }
#undef LAUNCH_APPLY

            if (cfg->survbits) {
                uint32_t *h = (uint32_t *)malloc((size_t)nbitword * 4);
                FILE *fo = fopen(cfg->survbits, "wb");
                CK(cudaMemcpy(h, D.survbits, (size_t)nbitword * 4,
                              cudaMemcpyDeviceToHost));
                if (!fo) { perror(cfg->survbits); }
                else { fwrite(h, 4, (size_t)nbitword, fo); fclose(fo); }
                free(h);
            }

            /* ---- device intersect + primitive filter + compaction -------- */
            if (cfg->other_bits) {
                uint32_t *hother = NULL, *dother = NULL;
                uint32_t *d_x = NULL, *d_n = NULL, hn = 0;
                int64_t *d_a = NULL, *d_b = NULL;
                unsigned long long *d_pre = NULL, hpre = 0, *d_qb = NULL, hqb = 0;
                uint32_t *d_two = NULL;
                /* cap: the two-sided count is ~1 in 400 of the area, but size
                 * it off the one-sided count so a bad pairing cannot overflow
                 * silently. */
                const uint32_t icap = maxsurv;
                FILE *fi = fopen(cfg->other_bits, "rb");
                float t_isect = 0;

                if (!fi) { perror(cfg->other_bits); td_failed = 1; }
                else {
                    CK(cudaHostAlloc((void **)&hother, (size_t)nbitword * 4,
                                     cudaHostAllocDefault));
                    size_t got = fread(hother, 4, (size_t)nbitword, fi);
                    fclose(fi);
                    if (got != (size_t)nbitword) {
                        fprintf(stderr, "  --other-bits: short read (%zu of %u"
                                " words)\n", got, nbitword);
                        td_failed = 1;
                    } else {
                        CK(cudaMalloc(&dother, (size_t)nbitword * 4));
                        CK(cudaMemcpy(dother, hother, (size_t)nbitword * 4,
                                      cudaMemcpyHostToDevice));
                        CK(cudaMalloc(&d_x, (size_t)icap * 4));
                        CK(cudaMalloc(&d_a, (size_t)icap * 8));
                        CK(cudaMalloc(&d_b, (size_t)icap * 8));
                        CK(cudaMalloc(&d_n, 4));   CK(cudaMemset(d_n, 0, 4));
                        CK(cudaMalloc(&d_pre, 8)); CK(cudaMemset(d_pre, 0, 8));
                        CK(cudaMalloc(&d_qb, 8));  CK(cudaMemset(d_qb, 0, 8));
                        CK(cudaMalloc(&d_two, (size_t)nbitword * 4));
                        CK(cudaMemset(d_two, 0, (size_t)nbitword * 4));

                        /* Time it repeatedly: the first launch of any kernel
                         * carries module/context cost that is not part of the
                         * steady-state per-q price. Report the best. */
                        t_isect = 1e30f;
                        for (int rep = 0; rep < 3; rep++) {
                            CK(cudaMemset(d_n, 0, 4));
                            CK(cudaMemset(d_pre, 0, 8));
                            CK(cudaMemset(d_qb, 0, 8));
                            cudaEventRecord(e3);
                            k_intersect_compact<1, false><<<blocks, cfg->threads>>>(
                                D.survbits, dother, nbitword, cfg->logI,
                                L->a0, L->a1, L->b0, L->b1, (int64_t)L->q,
                                d_x, d_a, d_b, icap, d_n, d_pre, d_two, d_qb, 0u);
                            cudaEventRecord(e4);
                            CK(cudaEventSynchronize(e4));
                            CK(cudaGetLastError());
                            float t = time_kernel(e3, e4);

                            if (t < t_isect) t_isect = t;
                        }

                        CK(cudaMemcpy(&hn, d_n, 4, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(&hpre, d_pre, 8, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(&hqb, d_qb, 8, cudaMemcpyDeviceToHost));

                        printf("\n  intersect+gcd+compact vs %s\n", cfg->other_bits);
                        printf("  %-26s %8llu\n", "two-sided, pre-gcd",
                               (unsigned long long)hpre);
                        /* THIS is the number to compare against dumpcmp --and
                         * and CADO's after_sieve: both apply the gcd(i,j) test
                         * and nothing else. */
                        printf("  %-26s %8llu  (%.1f%% of pre-gcd survive)"
                               "  <- CADO-comparable\n",
                               "primitive gcd(i,j)=1",
                               (unsigned long long)hn + hqb,
                               hpre ? 100.0 * (hn + hqb) / (double)hpre : 0.0);
                        printf("  %-26s %8llu  (q | b: (a,b) = q*(a',b'),"
                               " finding 68)\n",
                               "  of which dropped", (unsigned long long)hqb);
                        printf("  %-26s %8u\n", "emitted", hn);
                        if (hn > icap)
                            printf("  ** OVERFLOW: %u > cap %u, list truncated\n", hn, icap);
                        printf("  %-26s %8.3f ms  (best of 3; the first launch of any kernel\n"
       "                                       carries module-load cost -- 4.6 ms here)\n",
       "intersect+compact", t_isect);

                        if (cfg->emit && hn && hn <= icap) {
                            uint32_t *hx = (uint32_t *)malloc((size_t)hn * 4);
                            int64_t *ha = (int64_t *)malloc((size_t)hn * 8);
                            int64_t *hb = (int64_t *)malloc((size_t)hn * 8);
                            FILE *fo = fopen(cfg->emit, "wb");
                            CK(cudaMemcpy(hx, d_x, (size_t)hn * 4, cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(ha, d_a, (size_t)hn * 8, cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(hb, d_b, (size_t)hn * 8, cudaMemcpyDeviceToHost));
                            if (!fo) perror(cfg->emit);
                            else {
                                for (uint32_t z = 0; z < hn; z++)
                                    fprintf(fo, "%u %lld %lld\n", hx[z],
                                            (long long)ha[z], (long long)hb[z]);
                                fclose(fo);
                                printf("  wrote %u (x, a, b) to %s\n", hn, cfg->emit);
                            }
                            free(hx); free(ha); free(hb);
                        }
                    }
                    /* ---- recovery A/B ------------------------------------ */
                    if (d_two && hn && hn <= icap) {
                        const uint32_t nsum = nbitword / 2;          /* bits */
                        const uint32_t nsumword = (nsum + 31) / 32;
                        uint32_t *d_sum = NULL, *d_rx = NULL, *d_rp = NULL, *d_rn = NULL;
                        uint32_t *d_scratch = NULL;   /* so the plain purge cannot
                                                       * clobber A's (x,p) output */
                        unsigned long long *d_probe = NULL, *d_p1 = NULL, *d_rd = NULL;
                        /* recovery output: one (x, p) per survivor-hit. Size it
                         * generously -- the whole point is that it is small. */
                        const uint32_t rcap = 8u * 1024u * 1024u;
                        uint32_t hrn = 0; unsigned long long hprobe = 0, hp1 = 0, hrd = 0;
                        float tA = 1e30f, tB = 1e30f, tsum = 1e30f;
                        uint32_t hrnB = 0;

                        CK(cudaMalloc(&d_sum, (size_t)nsumword * 4));
                        CK(cudaMalloc(&d_rx, (size_t)rcap * 4));
                        CK(cudaMalloc(&d_rp, (size_t)rcap * 4));
                        CK(cudaMalloc(&d_rn, 4));
                        CK(cudaMalloc(&d_scratch, (size_t)rcap * 4));
                        CK(cudaMalloc(&d_probe, 8)); CK(cudaMalloc(&d_p1, 8));
                        CK(cudaMalloc(&d_rd, 8));

                        printf("\n  --- recovery A/B (side %d) ---\n", cfg->side);
                        printf("  survivor filter: %u summary bits (1 per 64 positions),"
                               " %.2f MB\n", nsum, nsumword * 4 / 1048576.0);

                        for (int rep = 0; rep < 3; rep++) {
                            CK(cudaMemset(d_sum, 0, (size_t)nsumword * 4));
                            cudaEventRecord(e3);
                            k_build_summary<<<blocks, cfg->threads>>>(d_two, nbitword, d_sum);
                            cudaEventRecord(e4);
                            CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                            float t = time_kernel(e3, e4); if (t < tsum) tsum = t;
                        }
                        /* occupancy of the summary table, measured not assumed */
                        {
                            uint32_t *hs = (uint32_t *)malloc((size_t)nsumword * 4);
                            CK(cudaMemcpy(hs, d_sum, (size_t)nsumword * 4,
                                          cudaMemcpyDeviceToHost));
                            unsigned long long occ = 0;
                            for (uint32_t z = 0; z < nsumword; z++)
                                occ += bench_popcount32(hs[z]);
                            printf("  %-26s %8llu  (%.2f%% occupied)\n",
                                   "summary bits set", occ, 100.0 * occ / (double)nsum);
                            free(hs);
                        }

                        for (int rep = 0; rep < 3; rep++) {
                            CK(cudaMemset(d_rn, 0, 4)); CK(cudaMemset(d_probe, 0, 8));
                            CK(cudaMemset(d_p1, 0, 8));
                            cudaEventRecord(e3);
                            k_resieve_rewalk<<<blocks, cfg->threads>>>(
                                D.plat, D.primes, fb->n, xmax, cfg->logI,
                                d_sum, d_two, d_rx, d_rp, rcap, d_rn, d_probe, d_p1);
                            cudaEventRecord(e4);
                            CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                            float t = time_kernel(e3, e4); if (t < tA) tA = t;
                        }
                        CK(cudaMemcpy(&hrn, d_rn, 4, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(&hprobe, d_probe, 8, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(&hp1, d_p1, 8, cudaMemcpyDeviceToHost));

                        if (D.out && D.cursor && cap) {
                            for (int rep = 0; rep < 3; rep++) {
                                CK(cudaMemset(d_rn, 0, 4)); CK(cudaMemset(d_rd, 0, 8));
                                cudaEventRecord(e3);
                                k_purge<<<blocks, cfg->threads>>>(
                                    (const uint32_t *)D.out, D.cursor, nregion,
                                    cap, log_region, d_two, d_scratch, rcap,
                                    d_rn, d_rd);
                                cudaEventRecord(e4);
                                CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                                float t = time_kernel(e3, e4); if (t < tB) tB = t;
                            }
                            CK(cudaMemcpy(&hrnB, d_rn, 4, cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(&hrd, d_rd, 8, cudaMemcpyDeviceToHost));
                        }

                        printf("  %-26s %8.3f ms\n", "build survivor filter", tsum);
                        printf("  A: re-walk %llu hits, %llu passed the summary"
                               " (%.2f%%), %u landed on a survivor\n",
                               hprobe, hp1, hprobe ? 100.0 * hp1 / (double)hprobe : 0.0, hrn);
                        printf("  %-26s %8.3f ms\n", "A: re-walk + filter", tA);
                        if (hrnB || hrd) {
                            printf("  B: purged %llu retained records, %u landed"
                                   " on a survivor\n", hrd, hrnB);
                            printf("  %-26s %8.3f ms\n", "B: purge retained buckets", tB);
                        }
                        if (hrn && hrnB)
                            printf("  %-26s %s\n", "A and B agree on count",
                                   hrn == hrnB ? "YES" : "NO  <-- investigate");

                        /* ---- LAYOUT B, built for real: segmented fill +
                         * purge that names the prime ---------------------- */
                        /* Layout B is a SETTLED experiment -- layout A won by
                         * 2.6x -- so it is opt-in. Two reasons beyond the
                         * wasted seconds. It refills D.out and D.cursor with
                         * segmented records in a different format, and the
                         * --verify replay at the end of this function reads
                         * exactly those buffers, so leaving it on by default
                         * made `--verify --other-bits` report thousands of
                         * differing cells and exit nonzero for a reason that
                         * had nothing to do with the fill under test. */
                        if (cfg->ab_resieve && cfg->verify) {
                            printf("\n  --ab-resieve skipped: it overwrites the"
                                   " bucket array that --verify replays\n");
                        } else if (cfg->ab_resieve &&
                                   D.out && D.cursor && cap && cfg->record_bytes == 4) {
                            uint32_t *hstarts = NULL;
                            uint32_t nsl = build_slices_b(fb, &hstarts);
                            uint32_t *d_starts = NULL, *d_bounds = NULL;
                            uint32_t *d_bx = NULL, *d_bp = NULL;
                            uint32_t hbn = 0, hov = 0;
                            float tfillB = 1e30f, tpurgeB = 1e30f;

                            printf("\n  --- layout B, built ---\n");
                            printf("  slices at the 65536 cap: %u  (the 262144 cut"
                                   " gives %u, but a 16-bit within-slice offset\n"
                                   "    cannot address it -- CADO asserts the same"
                                   " bound at fb.hpp:356)\n", nsl, nslice);
                            CK(cudaMalloc(&d_starts, (size_t)(nsl + 1) * 4));
                            CK(cudaMemcpy(d_starts, hstarts, (size_t)(nsl + 1) * 4,
                                          cudaMemcpyHostToDevice));
                            CK(cudaMalloc(&d_bounds, (size_t)nregion * nsl * 4));
                            CK(cudaMalloc(&d_bx, (size_t)rcap * 4));
                            CK(cudaMalloc(&d_bp, (size_t)rcap * 4));
                            printf("  boundary table: %u buckets x %u slices x 4 B"
                                   " = %.1f MB\n", nregion, nsl,
                                   (double)nregion * nsl * 4 / 1048576.0);

                            for (int rep = 0; rep < 3; rep++) {
                                CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
                                CK(cudaMemset(D.overflow, 0, 4));
                                cudaEventRecord(e3);
                                for (uint32_t sl = 0; sl < nsl; sl++) {
                                    k_fill_segmented<<<blocks, cfg->threads>>>(
                                        D.plat, hstarts[sl], hstarts[sl + 1], xmax,
                                        cfg->logI, log_region, D.cursor,
                                        (uint32_t *)D.out, cap, D.overflow);
                                    /* snapshot cursor[] -> this slice's boundary.
                                     * Bucket-major so the purge's binary search
                                     * reads contiguously. */
                                    k_snapshot_bounds<<<blocks, cfg->threads>>>(
                                        D.cursor, d_bounds, nregion, nsl, sl);
                                }
                                cudaEventRecord(e4);
                                CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tfillB) tfillB = t;
                            }
                            CK(cudaMemcpy(&hov, D.overflow, 4, cudaMemcpyDeviceToHost));
                            /* Merge slices into G super-segments and refill.
                             * Total work is identical; only the number of
                             * passes over the bucket array changes. If the cost
                             * tracks G, the price of segmentation is write
                             * amplification on the bucket array -- adjacent
                             * slots in a cache line being written by different
                             * passes -- and not anything about launches. */
                            for (uint32_t G : {1u, 2u, 4u, 8u, 16u, 32u, 63u}) {
                                float tg = 1e30f;
                                for (int rep = 0; rep < 3; rep++) {
                                    CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
                                    cudaEventRecord(e3);
                                    for (uint32_t g = 0; g < G; g++) {
                                        uint32_t s0 = (uint32_t)((uint64_t)g * nsl / G);
                                        uint32_t s1 = (uint32_t)((uint64_t)(g + 1) * nsl / G);
                                        if (s1 <= s0) continue;
                                        k_fill_segmented<<<blocks, cfg->threads>>>(
                                            D.plat, hstarts[s0], hstarts[s1], xmax,
                                            cfg->logI, log_region, D.cursor,
                                            (uint32_t *)D.out, cap, D.overflow);
                                    }
                                    cudaEventRecord(e4);
                                    CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                                    float t = time_kernel(e3, e4); if (t < tg) tg = t;
                                }
                                printf("  %-26s %8.3f ms   (%u passes over the bucket array)\n",
                                       "B: fill in G passes", tg, G);
                            }
                            /* Split the cost: same 126 fill launches, no
                             * boundary snapshots. The difference is what the
                             * segmentation bookkeeping costs; what remains
                             * above the monolithic fill is the launches
                             * themselves plus per-launch occupancy loss. */
                            float tfillNS = 1e30f;
                            for (int rep = 0; rep < 3; rep++) {
                                CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
                                cudaEventRecord(e3);
                                for (uint32_t sl = 0; sl < nsl; sl++)
                                    k_fill_segmented<<<blocks, cfg->threads>>>(
                                        D.plat, hstarts[sl], hstarts[sl + 1], xmax,
                                        cfg->logI, log_region, D.cursor,
                                        (uint32_t *)D.out, cap, D.overflow);
                                cudaEventRecord(e4);
                                CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tfillNS) tfillNS = t;
                            }
                            /* and: one launch per slice but sized to the slice,
                             * to separate launch count from occupancy loss */
                            float tfillSized = 1e30f;
                            for (int rep = 0; rep < 3; rep++) {
                                CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
                                cudaEventRecord(e3);
                                for (uint32_t sl = 0; sl < nsl; sl++) {
                                    uint32_t nent = hstarts[sl + 1] - hstarts[sl];
                                    int bl = (int)((nent + cfg->threads - 1) / cfg->threads);
                                    if (bl < 1) bl = 1;
                                    if (bl > blocks) bl = blocks;
                                    k_fill_segmented<<<bl, cfg->threads>>>(
                                        D.plat, hstarts[sl], hstarts[sl + 1], xmax,
                                        cfg->logI, log_region, D.cursor,
                                        (uint32_t *)D.out, cap, D.overflow);
                                }
                                cudaEventRecord(e4);
                                CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tfillSized) tfillSized = t;
                            }
                            printf("  %-26s %8.3f ms   (no boundary snapshots)\n",
                                   "B: segmented fill", tfillNS);
                            printf("  %-26s %8.3f ms   (grid sized per slice)\n",
                                   "B: segmented fill", tfillSized);

                            for (int rep = 0; rep < 3; rep++) {
                                CK(cudaMemset(d_rn, 0, 4));
                                cudaEventRecord(e3);
                                k_purge_prime<<<blocks, cfg->threads>>>(
                                    (const uint32_t *)D.out, D.cursor, d_bounds,
                                    d_starts, D.primes, nregion, cap, nsl,
                                    log_region, d_two, d_bx, d_bp, rcap, d_rn);
                                cudaEventRecord(e4);
                                CK(cudaEventSynchronize(e4)); CK(cudaGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tpurgeB) tpurgeB = t;
                            }
                            CK(cudaMemcpy(&hbn, d_rn, 4, cudaMemcpyDeviceToHost));

                            printf("  %-26s %8.3f ms   (%u launches%s)\n",
                                   "B: segmented fill", tfillB, nsl,
                                   hov ? ", OVERFLOWED" : "");
                            printf("  %-26s %8.3f ms\n", "B: purge + name the prime",
                                   tpurgeB);
                            printf("  %-26s %8u  vs A's %u   %s\n",
                                   "B: recovered (x,p)", hbn, hrn,
                                   hbn == hrn ? "MATCH" : "MISMATCH <-- investigate");

                            /* set-equality gate: A and B must recover the same
                             * (x, p) multiset, not merely the same count. */
                            if (hbn == hrn && hrn && hrn <= rcap) {
                                uint32_t *ax = (uint32_t *)malloc((size_t)hrn * 4);
                                uint32_t *ap = (uint32_t *)malloc((size_t)hrn * 4);
                                uint32_t *bx = (uint32_t *)malloc((size_t)hrn * 4);
                                uint32_t *bp = (uint32_t *)malloc((size_t)hrn * 4);
                                CK(cudaMemcpy(ax, d_rx, (size_t)hrn * 4, cudaMemcpyDeviceToHost));
                                CK(cudaMemcpy(ap, d_rp, (size_t)hrn * 4, cudaMemcpyDeviceToHost));
                                CK(cudaMemcpy(bx, d_bx, (size_t)hrn * 4, cudaMemcpyDeviceToHost));
                                CK(cudaMemcpy(bp, d_bp, (size_t)hrn * 4, cudaMemcpyDeviceToHost));
                                /* compare as sorted (x,p) pairs */
                                uint64_t *A64 = (uint64_t *)malloc((size_t)hrn * 8);
                                uint64_t *B64 = (uint64_t *)malloc((size_t)hrn * 8);
                                for (uint32_t z = 0; z < hrn; z++) {
                                    A64[z] = ((uint64_t)ax[z] << 32) | ap[z];
                                    B64[z] = ((uint64_t)bx[z] << 32) | bp[z];
                                }
                                std::sort(A64, A64 + hrn); std::sort(B64, B64 + hrn);
                                uint32_t diff = 0;
                                for (uint32_t z = 0; z < hrn; z++) if (A64[z] != B64[z]) diff++;
                                printf("  %-26s %s (%u of %u pairs differ)\n",
                                       "A vs B (x,p) set equality",
                                       diff ? "FAIL" : "IDENTICAL", diff, hrn);
                                free(ax); free(ap); free(bx); free(bp); free(A64); free(B64);
                            }
                            free(hstarts);
                            cudaFree(d_starts); cudaFree(d_bounds);
                            cudaFree(d_bx); cudaFree(d_bp);
                        }

                        cudaFree(d_sum); cudaFree(d_rx); cudaFree(d_rp); cudaFree(d_rn);
                        cudaFree(d_scratch);
                        cudaFree(d_probe); cudaFree(d_p1); cudaFree(d_rd);
                    }

                    /* ---- exact norms + trial division --------------------- */
                    if (cfg->td && d_two &&
                        run_td_stage(fb, fbs, L, POLY, cfg, D.plat, D.primes,
                                     d_two, nbitword, xmax, blocks, cfg->threads))
                        td_failed = 1;

                    if (hother) cudaFreeHost(hother);
                    cudaFree(dother); cudaFree(d_x); cudaFree(d_a);
                    cudaFree(d_b); cudaFree(d_n); cudaFree(d_pre); cudaFree(d_two);
                    cudaFree(d_qb);
                }
            }
            if (cfg->dump) {
                uint8_t *h = (uint8_t *)malloc((size_t)xmax);
                FILE *fo = fopen(cfg->dump, "wb");
                CK(cudaMemcpy(h, D.dumpbuf, (size_t)xmax, cudaMemcpyDeviceToHost));
                if (!fo) { perror(cfg->dump); }
                else { fwrite(h, 1, (size_t)xmax, fo); fclose(fo); }
                free(h);
            }
            uint32_t hs = 0;
            if (probe_x != 0xFFFFFFFFu) {
                uint32_t pr[2] = {0, 0};
                CK(cudaMemcpy(pr, D.probe, 8, cudaMemcpyDeviceToHost));
                printf("\n  [gate 5] probe (i=%d, j=%u)  x=%u  region %u offset %u\n"
                       "           init norm S   = %u\n"
                       "           final cell    = %u\n"
                       "           SIEVED LOG SUM = %d   <- produced by transform +"
                       " walk + fill + small sieve + apply\n"
                       "           las byte S-sum = %d\n",
                       cfg->probe_i, cfg->probe_j, probe_x,
                       probe_x >> log_region, probe_x & ((1u << log_region) - 1),
                       pr[0], pr[1],
                       (int)pr[1] - ((int)CINIT - (int)pr[0]),
                       (int)CINIT - (int)pr[1]);
            }
            CK(cudaMemcpy(&hs, D.nsurv, 4, cudaMemcpyDeviceToHost));
            printf("  survivors: %u of %u positions (1 in %.3e)\n", hs, xmax,
                   hs ? (double)xmax / hs : 0.0);

            if (cfg->verify && CB == 16) {
                uint32_t *hrec = (uint32_t *)malloc((size_t)cap * 4);
                uint32_t *hcnt = (uint32_t *)malloc((size_t)nregion * 4);
                uint16_t *hgpu = (uint16_t *)malloc((size_t)ncell * 2);
                uint16_t *href = (uint16_t *)malloc((size_t)ncell * 2);
                CK(cudaMemcpy(hcnt, D.cursor, (size_t)nregion * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hrec, D.out + (size_t)dbgreg * cap * 4, (size_t)cap * 4,
                              cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hgpu, D.dbg, (size_t)ncell * 2, cudaMemcpyDeviceToHost));
                uint32_t nr = hcnt[dbgreg] > cap ? cap : hcnt[dbgreg];
                uint32_t rs = verify_apply_region(hrec, nr, hlogp, &N, cfg->logI,
                        log_region, dbgreg, cfg->norm_mode, CINIT, tconst,
                        hsp, hsrt, hsg, hslp, nsmall, href);
                uint32_t bad = 0, first = 0xFFFFFFFFu;
                for (uint32_t i = 0; i < ncell; i++)
                    if (hgpu[i] != href[i]) { if (!bad) first = i; bad++; }
                printf("  [verify] region %u: %u records replayed on CPU, %u cells differ",
                       dbgreg, nr, bad);
                if (bad) printf("  (first at cell %u: gpu %u ref %u)",
                                first, hgpu[first], href[first]);
                printf("\n  [verify] region %u survivors: cpu %u\n", dbgreg, rs);
                free(hrec); free(hcnt); free(hgpu); free(href);
                /* A differing cell is a failed run, not a log line. This used
                 * to print and return 0, so `--verify && echo ok` reported
                 * success on a sieve that disagreed with its own reference. */
                if (bad) {
                    fprintf(stderr, "[verify] FAILED: %u cells differ in region %u\n",
                            bad, dbgreg);
                    return -1;
                }
            }
        }
    }
after_apply:

    printf("\n  %-26s %8.3f ms\n", "transform + plattice (T)", t_trans);
    if (cfg->fill_mode == FILL_ATOMIC)
        printf("  %-26s %8.3f ms\n", "fill: atomic single-level", t_fill);
    else {
        printf("  %-26s %8.3f ms\n", "fill L1: -> super-buckets", t_l1);
        printf("  %-26s %8.3f ms\n", "fill L2: -> regions", t_l2);
        printf("  %-26s %8.3f ms\n", "fill total", t_fill);
    }
    if (t_apply > 0)
        printf("  %-26s %8.3f ms\n", "apply (init+add+scan)", t_apply);
    /* The old "vs ~225 ms replaceable / ~71 ms hybrid-retained" suffix is gone.
     * Those were fixed constants from the GGNFS breakdown at N_eff = 10.24 --
     * a different job at a different logI/J -- printed on every run whatever
     * was actually being sieved, which read as a per-run comparison and is not
     * one. The surviving numbers live in RESULTS.md findings 43 and 45, where
     * the config they belong to is stated. */
    printf("  %-26s %8.3f ms\n",
           "SIEVE CHAIN ms/special-q", t_trans + t_fill + t_apply);
    /* Per-q HOST work. Not part of the sieve chain above -- it runs on the CPU,
     * once per special-q per side, and cudaEvent timing cannot see it. Goal 1
     * is about host demand, so it is billed here rather than left implicit. */
    printf("  %-26s %8.3f ms  (transform %.3f + sort %.3f + H2D %.3f)\n",
           "host per-q work", h_ms_transform + h_ms_sort + h_ms_xfer,
           h_ms_transform, h_ms_sort, h_ms_xfer);
    if (landed) {
        printf("  %-26s %8.2f\n", "ns per record (fill)", t_fill * 1e6 / landed);
        if (t_apply > 0)
            printf("  %-26s %8.2f\n", "ns per record (apply)", t_apply * 1e6 / landed);
    }

    free(hslice); free(hlogp);
    if (hsp)  cudaFreeHost(hsp);
    if (hsrt) cudaFreeHost(hsrt);
    if (hsmag) cudaFreeHost(hsmag);
    if (hsg)  cudaFreeHost(hsg);
    if (hslp) cudaFreeHost(hslp);
    cudaFree(D.primes); cudaFree(D.roots); cudaFree(D.plat); cudaFree(D.cursor);
    cudaFree(D.out); cudaFree(D.overflow); cudaFree(D.nproj); cudaFree(D.nlost);
    cudaFree(D.l1); cudaFree(D.l1cnt); cudaFree(D.slice); cudaFree(D.slice_logp);
    cudaFree(D.nsurv); cudaFree(D.dbg); cudaFree(D.probe);
    cudaFree(D.smag);
    cudaFree(D.sp); cudaFree(D.srt); cudaFree(D.sg); cudaFree(D.slp); cudaFree(D.dumpbuf); cudaFree(D.survbits);
    return td_failed ? -1 : 0;
}

#include "cofac.cuh"
#include "pipeline.cuh"
