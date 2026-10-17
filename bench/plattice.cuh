/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Franke-Kleinjung p-lattice reduction and walk.
 *
 * Ported from CADO-NFS sieve/las-plattice.hpp and
 * sieve/las-reduce-plattice-simplistic.hpp so that the benchmark walks the
 * same positions a real lattice siever does. Marked __host__ __device__ so
 * verify_cpu.c and the GPU kernels run bit-identical code.
 *
 * Proposition 1 of [FrKl05]: a basis <(i0,j0),(i1,j1)> of the p-lattice
 * inside the q-lattice with j0,j1 > 0, -I < i0 <= 0 <= i1 < I, i1-i0 >= I.
 * i0 is stored negated as mi0.
 *
 * Positions are encoded x = (i + I/2) + (j << logI). A slab-local x is kept
 * below 2^31, but the reduced lattice increment itself can exceed 2^32 even
 * for I15/I16 with a large factor-base prime. CADO's wide walk arithmetic is
 * therefore required independently of whole-rectangle size.
 */
#ifndef CUDA_SIEVE_PLATTICE_CUH
#define CUDA_SIEVE_PLATTICE_CUH

#include <stdint.h>

#ifdef __CUDACC__
#define PL_FN __host__ __device__ static inline
#else
#define PL_FN static inline
#endif

/* Leading zeros of a nonzero 32-bit word, on the device and on every host
 * compiler the builds use (MSVC has no __builtin_clz). */
#if defined(__CUDA_ARCH__)
#define PL_CLZ32(x) __clz((int)(x))
#elif defined(__GNUC__) || defined(__clang__)
#define PL_CLZ32(x) __builtin_clz(x)
#else
static inline int pl_clz32_host(uint32_t x)
{
    int n = 0;
    while (!(x & 0x80000000u)) { x <<= 1; n++; }
    return n;
}
#define PL_CLZ32(x) pl_clz32_host(x)
#endif

/* Walk parameters. The increments must be 64-bit even while a slab-local
 * position stays below 2^31. Reduction can produce j0/j1 large enough that
 * (j << logI) exceeds 2^32 (for example an I16, p~250M walk can have a warp
 * increment above 8.5e9). Truncating the increment, or later wrapping x while
 * adding it, creates spurious lattice hits. CADO likewise carries wide walk
 * offsets. The bounds remain I-local and therefore 32-bit. */
typedef struct {
    uint64_t inc_warp;   /* (j0 << logI) + i0 , always > 0 as an offset */
    uint64_t inc_step;   /* (j1 << logI) + i1 , or PL_VERTICAL          */
    uint32_t bound_warp; /* add warp when (x & (I-1)) >= bound_warp     */
    uint32_t bound_step; /* add step when (x & (I-1)) <  bound_step     */
} plat_t;

#define PL_VERTICAL UINT64_MAX
#define PL_INVALID  UINT64_MAX

typedef struct { uint32_t mi0, j0, i1, j1; } pl_basis_t;

PL_FN void pl_reduce_with_vertical_vector(pl_basis_t *B, uint32_t I)
{
    const uint32_t a = (I - 1) % B->mi0;
    uint32_t mi2 = (I - 1) - a - B->i1;
    const uint32_t j2 = B->j1;
    if (B->mi0 + mi2 < I) mi2 += B->mi0;
    B->i1  = B->mi0;
    B->j1  = B->j0;
    B->mi0 = mi2;
    B->j0  = j2;
}

/* CADO's reduce_plattice_simplistic, verbatim in structure. */
PL_FN void pl_reduce(pl_basis_t *B, uint32_t I)
{
    /* needs_special_treatment: i1 == 0 || (j1 > 1 && mi0 < I).
     * On the initial basis j1 == 1, so this is exactly r == 0. */
    if (B->i1 == 0 || (B->j1 > 1 && B->mi0 < I)) {
        pl_reduce_with_vertical_vector(B, I);
        return;
    }
    for (;;) {
        if (B->i1 < I) {
            if (B->i1 == 0) {
                B->j0 = B->j1 - B->j0;
                pl_reduce_with_vertical_vector(B, I);
                return;
            }
            {
                uint32_t a = (B->mi0 + B->i1 - I) / B->i1;
                B->mi0 -= a * B->i1;
                B->j0  += a * B->j1;
            }
            return;
        }
        {
            uint32_t k = B->mi0 / B->i1;
            B->mi0 -= k * B->i1;
            B->j0  += k * B->j1;
        }
        if (B->mi0 < I) {
            if (B->mi0 == 0) {
                uint32_t t = B->i1;
                B->mi0 = t;
                B->i1  = B->j0; B->j0 = B->j1; B->j1 = B->i1;
                B->i1  = 0;
                pl_reduce_with_vertical_vector(B, I);
                return;
            }
            {
                uint32_t a = (B->mi0 + B->i1 - I) / B->mi0;
                B->i1 -= a * B->mi0;
                B->j1 += a * B->j0;
            }
            return;
        }
        {
            uint32_t k = B->i1 / B->mi0;
            B->i1 -= k * B->mi0;
            B->j1 += k * B->j0;
        }
    }
}

/* Build the walk parameters for prime p with transformed root r. */
PL_FN plat_t pl_make(uint32_t p, uint32_t r, int logI)
{
    const uint32_t I = 1u << logI;
    pl_basis_t B;
    plat_t P;
    B.mi0 = p; B.j0 = 0; B.i1 = r; B.j1 = 1;   /* initial_basis, affine */
    pl_reduce(&B, I);

    P.bound_step = B.mi0;
    if (B.i1 >> logI) {                         /* vertical line */
        P.bound_warp = 0;
        P.inc_step   = PL_VERTICAL;
    } else {
        P.bound_warp = I - B.i1;
        P.inc_step   = ((uint64_t)B.j1 << logI) + B.i1;
    }
    /* inc_warp = (j0 << logI) + i0 with i0 = -mi0; positive because
     * the post-conditions give mi0 < I <= j0*I whenever j0 >= 1. */
    P.inc_warp = ((uint64_t)B.j0 << logI) - B.mi0;
    return P;
}

/* Exact walk step. Slab continuation uses this because the first hit after a
 * slab can be several billion positions beyond its end. Only the low logI
 * bits choose the step, so translating by a whole number of rows remains
 * exact even when x itself is wider. */
PL_FN uint64_t pl_next64(uint64_t x, const plat_t *P, uint32_t Imask)
{
    const uint32_t i = (uint32_t)x & Imask;
    if (i >= P->bound_warp) x += P->inc_warp;
    if (i < P->bound_step) {
        if (P->inc_step == PL_VERTICAL) return UINT64_MAX;
        x += P->inc_step;
    }
    return x;
}

/* Bounded 32-bit walk used by the legacy <=2^31 local-position kernels. A
 * mathematically valid next hit can lie above UINT32_MAX even though every hit
 * we care about is below 2^31. Saturate instead of wrapping back into the sieve
 * area. This keeps the hot local x itself 32-bit while fixing the latent
 * monolithic overflow found by the slab-equivalence review. */
PL_FN uint32_t pl_add32_sat(uint32_t x, uint64_t inc)
{
    const uint32_t hi = (uint32_t)(inc >> 32);
    const uint32_t lo = (uint32_t)inc;
    if (hi || x > UINT32_MAX - lo) return UINT32_MAX;
    return x + lo;
}

PL_FN uint32_t pl_next(uint32_t x, const plat_t *P, uint32_t Imask)
{
    const uint32_t i = x & Imask;
    if (i >= P->bound_warp) {
        x = pl_add32_sat(x, P->inc_warp);
        if (x == UINT32_MAX) return x;
    }
    if (i < P->bound_step) {
        if (P->inc_step == PL_VERTICAL) return UINT32_MAX;
        x = pl_add32_sat(x, P->inc_step);
    }
    return x;
}

/* The lattice always contains (i,j) = (0,0), i.e. x = I/2, and for j == 0
 * that is its *only* point (i == 0 mod p and |i| < I/2 < p forces i == 0).
 * pl_start returns it; brute-force verification enumerates from here. */
PL_FN uint32_t pl_start(int logI) { return 1u << (logI - 1); }

/* Sieving starts at j == 1: (0,0) is not a relation, and counting it would
 * add one spurious record *per prime* -- 6.8M of them on this factor base,
 * which is 4x the entire true update count at small sieve areas. Every
 * throughput path must use this, not pl_start. */
PL_FN uint32_t pl_first(const plat_t *P, int logI)
{
    return pl_next(1u << (logI - 1), P, (1u << logI) - 1);
}

PL_FN uint64_t pl_first64(const plat_t *P, int logI)
{
    return pl_next64((uint64_t)1 << (logI - 1), P, (1u << logI) - 1);
}

/* Rows from a row whose strip offset is `off` to the first row that holds a
 * strip point: the least t >= 0 with (off + t*rt) mod q < I, for off, rt < q
 * and q >= I (the row tier's lattice i + I/2 == (rt*j + I/2) mod q). The
 * offsets run through I/2 + gcd(rt,q)Z (mod q), which contains I/2 < I, so a
 * row-tier entry always has a solution; UINT64_MAX means none (rt == 0 with
 * off >= I).
 *
 * Stepping row by row costs up to ~q - I iterations (rt == 1, or rt near a
 * fraction of q), so this is Euclid on (rt, q), O(log q). For t > 0 the
 * condition is (rt*t) mod q in [q - off, q - off + I - 1], an interval inside
 * [1, q) when off >= I. The least x >= 0 with (A*x) mod M in [L, R],
 * 0 < L <= R < M, is ceil(L/A) when that multiple of A is <= R. Otherwise no
 * multiple of A lies in [L, R], and the least y >= 1 with A*x - M*y in [L, R]
 * is the same problem one Euclid step down -- (M mod A)*y mod A in
 * [(-R) mod A, (-L) mod A], which again excludes 0 -- after which
 * x = ceil((L + M*y)/A). verify_rows_to_hit checks it against brute force. */
PL_FN uint64_t pl_rows_to_hit(uint32_t off, uint32_t rt, uint32_t q, uint32_t I)
{
    uint32_t sA[48], sM[48], sL[48];   /* Euclid on 32-bit inputs: < 47 levels */
    uint64_t A = rt, M = q, L, R, x;
    int d = 0;
    if (off < I) return 0;
    L = (uint64_t)q - off;
    R = L + I - 1;
    for (;;) {
        if (A == 0) return UINT64_MAX;
        x = (L + A - 1) / A;
        if (A * x <= R) break;
        if (d == 48) return UINT64_MAX;
        sA[d] = (uint32_t)A; sM[d] = (uint32_t)M; sL[d] = (uint32_t)L; d++;
        {
            const uint64_t nA = M % A, nM = A;
            const uint64_t nL = (A - R % A) % A, nR = (A - L % A) % A;
            A = nA; M = nM; L = nL; R = nR;
        }
    }
    while (d-- > 0)
        x = ((uint64_t)sL[d] + (uint64_t)sM[d] * x + sA[d] - 1) / sA[d];
    return x;
}

/* ---- modular arithmetic used by the root transform ------------------- */

/* Binary extended Euclid. p odd, 0 < a < p. Returns a^-1 mod p, or 0 if
 * a == 0. Divergent by nature; the design doc's recommended replacement is
 * a branch-free 2-adic (Hensel/REDC) inverse, which is variant "TH".
 * The halving adds in 64 bits: x + p passes 2^32 when p >= 2^31, and the
 * 32-bit form lost that carry (q = 3,982,922,849 gave rt 1,330,206,460 for
 * 662,933,767; finding 117's review). The subtractions need no widening:
 * x1 + p - x2 wraps back below p. */
PL_FN uint32_t pl_invmod(uint32_t a, uint32_t p)
{
    uint32_t u = a, v = p, x1 = 1, x2 = 0;
    if (a == 0) return 0;
    while (u != 1 && v != 1) {
        while ((u & 1) == 0) {
            u >>= 1;
            x1 = (x1 & 1) ? (uint32_t)(((uint64_t)x1 + p) >> 1) : (x1 >> 1);
        }
        while ((v & 1) == 0) {
            v >>= 1;
            x2 = (x2 & 1) ? (uint32_t)(((uint64_t)x2 + p) >> 1) : (x2 >> 1);
        }
        if (u >= v) { u -= v; x1 = (x1 >= x2) ? x1 - x2 : x1 + p - x2; }
        else        { v -= u; x2 = (x2 >= x1) ? x2 - x1 : x2 + p - x1; }
    }
    return (u == 1) ? x1 : x2;
}

/* Inverse modulo an arbitrary prime power.
 *
 * pl_invmod above is binary extended Euclid and requires an ODD modulus -- its
 * halving step (x+p)>>1 is only exact when p is odd. GGNFS's .afb.0 has no
 * even entries so this never mattered; CADO's factor base carries the whole
 * ladder 2,4,8,...,32768, and feeding one to pl_invmod does not fail, it
 * LOOPS FOREVER. Route powers of two to a 2-adic Newton iteration instead:
 * x <- x*(2 - a*x) doubles the number of correct bits each step, and x = a is
 * already correct mod 8 for odd a, so five steps cover 32 bits. */
PL_FN uint32_t pl_invmod_any(uint32_t a, uint32_t m)
{
    if (m & 1u) return pl_invmod(a, m);
    if (!(a & 1u)) return 0;              /* not invertible mod 2^k */
    {
        uint32_t inv = a;
        int i;
        for (i = 0; i < 5; i++) inv *= 2u - a * inv;
        return inv & (m - 1u);
    }
}

/* Transformed root: the r' with i == r'*j (mod p) over the q-lattice.
 *
 * BEWARE THE FIELD NAMING. CADO's qlattice groups by coordinate --
 * a = a0*i + a1*j, b = b0*i + b1*j -- and `fb_root_in_qlattice` therefore
 * returns (R*b1 - a1)/(a0 - R*b0). qlat_t groups by *vector*: (a0,a1) and
 * (b0,b1) are the two basis vectors, so a = i*a0 + j*b0 and b = i*a1 + j*b1
 * (this is what qlat_build constructs and what norm_setup consumes). Copying
 * CADO's formula across the two conventions silently transposes the basis --
 * a1 and b0 swap -- which yields a perfectly valid lattice of the right
 * density, so record counts and CPU/GPU cross-checks all still agree. It is
 * simply a *different* lattice from the one the norms describe, and nothing
 * short of parity against las would catch it.
 *
 * In this struct's convention, a == r*b (mod p) becomes
 *     i*(a0 - r*a1) == j*(r*b1 - b0)   (mod p)
 * so r' = (r*b1 - b0) / (a0 - r*a1).
 *
 * Returns p (an out-of-range sentinel) when the denominator vanishes: the
 * transformed root is itself projective. */
PL_FN uint32_t pl_transform(uint32_t p, uint32_t r,
                            int64_t a0, int64_t a1, int64_t b0, int64_t b1)
{
    const int64_t P = (int64_t)p;
    int64_t num = ((int64_t)r * b1 - b0) % P;
    int64_t den = (a0 - (int64_t)r * a1) % P;
    if (num < 0) num += P;
    if (den < 0) den += P;
    if (den == 0) return p;
    return (uint32_t)(( (uint64_t)num * pl_invmod_any((uint32_t)den, p) ) % (uint64_t)p);
}

#define PL_ROWS 0xFFFFFFFFu

/* General transformed root, valid for prime POWERS as well as primes.
 *
 * With a prime modulus the denominator is either invertible or zero. With
 * q = p^k there is a third case: p | D but q does not, and then D has no
 * inverse mod q at all -- binary Euclid does not report failure, it spins
 * forever on gcd > 1. That is exactly what CADO's factor base triggers, since
 * makefb emits 49, 121, 169, ... alongside the plain primes.
 *
 * ROOT ENCODING. Both CADO and GGNFS store a root of q in [0, 2q). A value
 * r < q is AFFINE and means a == r*b (mod q). A value r >= q is PROJECTIVE
 * with reciprocal rr = r - q, and means
 *
 *     a * rr == b   (mod q).
 *
 * rr == 0 is the classical "b == 0 (mod q)" case, and it is the ONLY case that
 * arises for a prime q -- which is why treating every projective root as
 * rr == 0 survives a prime-only factor base. It is wrong for prime powers.
 * A projective ideal above p lifts to p^k as a point of P^1(Z/p^k) whose
 * reciprocal is divisible by p but not by p^k, so the ladder above a
 * projective prime carries NONZERO reciprocals: `c183.fb1` opens with
 * `4:4,3: 6` (q = 4, rr = 2) and has 35 such entries, worth 4.7e8 updates and
 * 1.54 scaled log units per cell. Forcing rr to 0 keeps the density (both are
 * index-q sublattices) and moves every one of those updates to the wrong
 * congruence -- invisible to a density gate like sum(logp/q), and invisible to
 * a CPU replay that shares this function. Pass the reciprocal.
 *
 * Solve it in general. The condition becomes i*D == j*N (mod q) with
 *     affine (a == r*b):      D = a0 - r*a1,     N = r*b1 - b0
 *     projective (a*rr == b): D = rr*a0 - a1,    N = b1 - rr*b0
 * (at rr == 0 the projective pair is (-a1, b1), the negation of the old
 * (a1, -b1) -- same ratio, so this generalises the previous case rather than
 * replacing it). D and N cannot both be divisible by p, or p would divide the
 * determinant, which is the special-q: in the projective case
 * rr*det == D*b1 + a1*N, and p | rr would force p | a1 and p | b1. So with
 * g = gcd(D, q):
 *
 * PRECONDITION: q is a prime power. That is all a factor base ever contains,
 * and the parameterisation below genuinely needs it -- the step "solutions
 * exist only when g | j" uses gcd(N, g) == 1, which follows from "p divides at
 * most one of D and N" and holds only for a single prime p. For a composite
 * modulus with two prime factors (q = 200, D even, N divisible by 5) the
 * solution set is a CRT combination and this form is wrong. `fbtest` gates
 * both halves of that: the transform against the definition over prime-power
 * moduli, and every loaded factor base for being prime powers only.
 *
 *     solutions exist only when g | j, and then, writing j = g*j',
 *     i == rt * j'  (mod m),   m = q/g,   rt = N * (D/g)^-1  (mod m).
 *
 * g == 1 is the ordinary affine case. g == q gives m == 1, meaning "every i,
 * on every q-th row" -- which is precisely the old PL_ROWS special case, now
 * subsumed rather than special. */
PL_FN uint32_t pl_gcd32(uint32_t a, uint32_t b)
{
    while (b) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

/* Returns the modulus m; writes the root and the row divisor g. */
PL_FN uint32_t pl_transform_gen(uint32_t q, uint32_t r, int is_proj,
                                int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                uint32_t *rt_out, uint32_t *g_out)
{
    const int64_t Q = (int64_t)q;
    int64_t D = is_proj ? (((int64_t)r * a0 - a1) % Q) : ((a0 - (int64_t)r * a1) % Q);
    int64_t N = is_proj ? ((b1 - (int64_t)r * b0) % Q) : (((int64_t)r * b1 - b0) % Q);
    uint32_t g, m, dg;
    if (D < 0) D += Q;
    if (N < 0) N += Q;
    g = pl_gcd32((uint32_t)D, q);
    m = q / g;
    if (m == 1) { *rt_out = 0; *g_out = g; return 1; }
    dg = (uint32_t)((uint64_t)D / g);
    *rt_out = (uint32_t)(((uint64_t)((uint32_t)N % m)
                          * pl_invmod_any(dg % m, m)) % (uint64_t)m);
    *g_out = g;
    return m;
}

/* Transform a root in the factor base's OWN encoding -- affine below q,
 * projective with reciprocal r - q at or above it. Every consumer of a loaded
 * factor base should call this and nothing else: it is the single place the
 * encoding is decoded, it is power-safe, and it cannot hang. */
PL_FN uint32_t pl_transform_enc(uint32_t q, uint32_t r_enc,
                                int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                uint32_t *rt_out, uint32_t *g_out)
{
    const int isp = (r_enc >= q);
    return pl_transform_gen(q, isp ? r_enc - q : r_enc, isp,
                            a0, a1, b0, b1, rt_out, g_out);
}

/* ---- the transform's fast path: Montgomery residues, Fermat inverse ---- *
 *
 * Measured at I16 (ncu, NFS@Home lpba-34 job): pl_transform_enc spends 63% of
 * k_transform's instructions in pl_invmod's binary Euclid and 14% in
 * pl_gcd32, with 13.8 of 32 lanes active -- both loops run a data-dependent
 * number of iterations with data-dependent branches inside. For an odd
 * modulus whose denominator is invertible (every prime entry but the
 * special-q's own ideal) this computes the same (m, rt, g) = (q, N/D, 1)
 * without either loop:
 *
 *   - REDC(T) = T * 2^-32 (mod q) for T < q * 2^32 replaces the 64-bit `%`.
 *     The four lattice terms reduce to D * 2^-32 and N * 2^-32, and the
 *     factors 2^-32 cancel in N/D.
 *   - the inverse is D^(q-2) by Montgomery square-and-multiply, the multiply
 *     computed on every bit and selected, so lanes stay converged.
 *   - D * D^-1 == 1 is then CHECKED. That check, not primality, is what
 *     admits the result. It passes ONLY when D is invertible mod q -- always,
 *     for a prime q; for a prime power p^k only when D^(q-1) == 1, so an
 *     invertible D can still fall through (q = 9, D = 2) -- and whenever it
 *     passes, g == gcd(D, q) == 1, m == q and rt = N * D^-1 is the unique
 *     residue pl_transform_gen also returns. A failed check, D == 0, an even
 *     modulus or a lattice coefficient of 2^32 or more falls through to
 *     pl_transform_enc, unchanged. fbtest's gates compare the two exactly. */
PL_FN uint32_t pl_redc(uint64_t T, uint32_t q, uint32_t qinv)
{
    /* T < q * 2^32. The low words of T and m*q agree, so (T - m*q) / 2^32 is
     * the difference of the high words: no 65-bit intermediate for q >= 2^31. */
    const uint32_t m = (uint32_t)T * qinv;
    const uint32_t u = (uint32_t)(((uint64_t)m * q) >> 32);
    const uint32_t h = (uint32_t)(T >> 32);
    return (h >= u) ? h - u : h - u + q;
}

PL_FN uint32_t pl_montmul(uint32_t a, uint32_t b, uint32_t q, uint32_t qinv)
{
    return pl_redc((uint64_t)a * b, q, qinv);
}

/* x * |c| * 2^-32 (mod q), negated when c < 0. x < q and |c| < 2^32. */
PL_FN uint32_t pl_redc_term(uint32_t x, int64_t c, uint32_t q, uint32_t qinv)
{
    const uint32_t v = pl_redc((uint64_t)x * (uint32_t)(c < 0 ? -c : c), q, qinv);
    return (c < 0 && v) ? q - v : v;
}

PL_FN uint32_t pl_subq(uint32_t a, uint32_t b, uint32_t q)
{
    return (a >= b) ? a - b : a + (q - b);
}

/* The fast path alone: returns 1 and writes rt when it applies (then
 * m == q and g == 1), 0 when the caller must use pl_transform_enc. */
PL_FN int pl_transform_fast_try(uint32_t q, uint32_t r_enc,
                                int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                uint32_t *rt_out)
{
    const int isp = (r_enc >= q);
    const uint32_t r = isp ? r_enc - q : r_enc;
    const int64_t LIM = (int64_t)1 << 32;
    if ((q & 1u) && q > 1u && r < q &&
        a0 > -LIM && a0 < LIM && a1 > -LIM && a1 < LIM &&
        b0 > -LIM && b0 < LIM && b1 > -LIM && b1 < LIM) {
        uint32_t qinv = q, d, n, acc, t, e;
        int bit;
        /* q^-1 mod 2^32 by Newton: q*q == 1 (mod 8) for odd q, and each
         * step doubles the correct bits, 3 -> 48. */
        qinv *= 2u - q * qinv; qinv *= 2u - q * qinv;
        qinv *= 2u - q * qinv; qinv *= 2u - q * qinv;
        /* D and N times 2^-32, in pl_transform_gen's two encodings */
        if (isp) {
            d = pl_subq(pl_redc_term(r, a0, q, qinv), pl_redc_term(1u, a1, q, qinv), q);
            n = pl_subq(pl_redc_term(1u, b1, q, qinv), pl_redc_term(r, b0, q, qinv), q);
        } else {
            d = pl_subq(pl_redc_term(1u, a0, q, qinv), pl_redc_term(r, a1, q, qinv), q);
            n = pl_subq(pl_redc_term(r, b1, q, qinv), pl_redc_term(1u, b0, q, qinv), q);
        }
        if (d) {
            /* acc = d^(q-2) * 2^(-32(q-3)), each montmul dropping a 2^32:
             * d^-1 * 2^64 for prime q. Left to right from the top bit, so no
             * 2^32 mod q is needed to start. */
            e = q - 2u;
            acc = d;
            for (bit = 30 - (int)PL_CLZ32(e); bit >= 0; bit--) {
                acc = pl_montmul(acc, acc, q, qinv);
                t = pl_montmul(acc, d, q, qinv);
                acc = ((e >> bit) & 1u) ? t : acc;
            }
            /* REDC(REDC(d * acc)) == 1  <=>  acc * 2^-64 is d's inverse */
            if (pl_redc(pl_montmul(d, acc, q, qinv), q, qinv) == 1u) {
                *rt_out = pl_redc(pl_montmul(n, acc, q, qinv), q, qinv);
                return 1;
            }
        }
    }
    return 0;
}

/* Same contract and outputs as pl_transform_enc. */
PL_FN uint32_t pl_transform_fast(uint32_t q, uint32_t r_enc,
                                 int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                 uint32_t *rt_out, uint32_t *g_out)
{
    if (pl_transform_fast_try(q, r_enc, a0, a1, b0, b1, rt_out)) {
        *g_out = 1u;
        return q;
    }
    return pl_transform_enc(q, r_enc, a0, a1, b0, b1, rt_out, g_out);
}

/* Projective factor-base root (GGNFS and CADO both encode it as root >= p):
 * the condition is b == 0 (mod p) rather than a == r*b. Kept for the
 * prime-modulus paths; pl_transform_gen handles it uniformly. */

PL_FN uint32_t pl_transform_proj(uint32_t p,
                                 int64_t a0, int64_t a1, int64_t b0, int64_t b1)
{
    const int64_t P = (int64_t)p;
    int64_t d = a1 % P, num = (-b1) % P;
    (void)a0; (void)b0;
    if (d < 0) d += P;
    if (num < 0) num += P;
    if (d == 0) return PL_ROWS;
    return (uint32_t)(( (uint64_t)num * pl_invmod_any((uint32_t)d, p) ) % (uint64_t)p);
}

#endif
