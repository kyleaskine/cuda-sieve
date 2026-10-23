/* Single-threaded ground truth for the lattice walk.
 *
 * The design doc's first correctness gate is "fill+apply parity against our
 * own reference on the real FB". This is that reference. It shares
 * plattice.cuh with the GPU, so it validates the walk *algorithm* against
 * brute force, not the GPU against itself. */
#include "bench.h"
#include "plattice.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Brute-force enumerate {x : i == r*j mod p} and compare against pl_*(). */
static int check_one(uint32_t p, uint32_t r, int logI, uint32_t J)
{
    const uint32_t I = 1u << logI, Imask = I - 1;
    const uint32_t xmax = I * J;
    uint32_t *ref = (uint32_t *)malloc((size_t)(xmax / p + J + 16) * 4);
    uint32_t nref = 0, x, n = 0;
    plat_t P;
    int64_t j;

    for (j = 0; j < (int64_t)J; j++) {
        /* i in [-I/2, I/2) with i == r*j (mod p) */
        int64_t base = ((int64_t)r * j) % p;          /* in [0,p) */
        int64_t i;
        for (i = base; i < (int64_t)I / 2; i += p)
            if (i >= -(int64_t)I / 2) ref[nref++] = (uint32_t)(i + I / 2 + (j << logI));
        for (i = base - p; i >= -(int64_t)I / 2; i -= p)
            ref[nref++] = (uint32_t)(i + I / 2 + (j << logI));
    }
    /* ref is grouped by j and each j-group is unsorted; sort by value */
    for (uint32_t a = 1; a < nref; a++) {            /* insertion sort, small */
        uint32_t v = ref[a]; int32_t b = (int32_t)a - 1;
        while (b >= 0 && ref[b] > v) { ref[b + 1] = ref[b]; b--; }
        ref[b + 1] = v;
    }

    P = pl_make(p, r, logI);
    for (x = pl_start(logI); x < xmax; x = pl_next(x, &P, Imask)) {
        if (n >= nref || ref[n] != x) {
            fprintf(stderr,
                "verify_walk MISMATCH p=%u r=%u logI=%d J=%u: step %u walk=%u ref=%u\n",
                p, r, logI, J, n, x, n < nref ? ref[n] : 0xFFFFFFFFu);
            free(ref); return -1;
        }
        n++;
    }
    if (n != nref) {
        fprintf(stderr, "verify_walk SHORT p=%u r=%u: walk %u of %u points\n",
                p, r, n, nref);
        free(ref); return -1;
    }
    free(ref);
    return 0;
}

/* Does the factor-base entry (q, r_enc) hit position (i,j)? Straight from the
 * definition, with no lattice algebra: map (i,j) to (a,b) and test the
 * congruence the encoding names. This is the ground truth. */
static int hits_def(uint32_t q, uint32_t r_enc, const qlat_t *L,
                    int32_t i, int32_t j)
{
    const int64_t a = (int64_t)i * L->a0 + (int64_t)j * L->b0;
    const int64_t b = (int64_t)i * L->a1 + (int64_t)j * L->b1;
    const int64_t Q = (int64_t)q;
    /* Reduce BEFORE multiplying. Over the whole sieve area |a| reaches ~2e12,
     * and a bucketed q reaches 1.3e8, so the direct product a*(r-q) overflows
     * int64 by two orders of magnitude -- which would silently corrupt the one
     * function every gate here treats as ground truth. */
    int64_t am = a % Q, bm = b % Q;
    uint64_t lhs;
    if (am < 0) am += Q;
    if (bm < 0) bm += Q;
    if (r_enc >= q) {                      /* projective: a*rr == b */
        lhs = ((uint64_t)am * (uint64_t)(r_enc - q)
               + (uint64_t)(Q - bm)) % (uint64_t)Q;
    } else {                               /* affine: a == r*b */
        const uint64_t rb = ((uint64_t)bm * (uint64_t)r_enc) % (uint64_t)Q;
        lhs = ((uint64_t)am + (uint64_t)Q - rb) % (uint64_t)Q;
    }
    return lhs == 0;
}

/* Same predicate, exported: `fbtest --trace` needs it to list every ideal that
 * hits a position, which is our half of gate 5. */
int hits_def_pub(uint32_t q, uint32_t r_enc, const qlat_t *L,
                 int32_t i, int32_t j)
{
    return hits_def(q, r_enc, L, i, j);
}

/* What pl_transform_enc claims: solutions exist only on rows j == 0 (mod g),
 * and there i == rt*(j/g) (mod m). */
static int hits_pred(uint32_t m, uint32_t rt, uint32_t g, int32_t i, int32_t j)
{
    int64_t d;
    if (g > 1 && (j % (int32_t)g)) return 0;
    if (m == 1) return 1;
    d = (int64_t)i - (int64_t)rt * (g > 1 ? j / (int32_t)g : j);
    return d % (int64_t)m == 0;
}

/* Gate the root transform against its definition, by SET EQUALITY over a small
 * box. Two things make this stronger than the check it replaces:
 *
 *   - the old test verified only that each predicted hit satisfies the
 *     congruence -- one direction. A transform naming a proper sublattice
 *     (right congruence, half the hits) passed it. This compares both
 *     directions, so a missing hit fails too.
 *   - it drives pl_transform_enc through the encoding, so PROJECTIVE roots
 *     with a nonzero reciprocal are covered. That is the case the factor base
 *     actually contains (c183.fb1 has 35 of them) and the case that was wrong.
 *
 * Every PRIME POWER 2..QMAX and every root in [0, 2q) is exercised: primes,
 * powers, even moduli, affine and projective. The transform is pure lattice
 * algebra and needs no (q,r) to be a genuine root of anything -- it only needs
 * q coprime to the determinant, which is the special-q.
 *
 * Composite moduli with two distinct prime factors are deliberately NOT
 * checked: pl_transform_gen does not claim them (see its precondition), and no
 * factor base contains them. `fb_check_prime_powers` is the gate that keeps
 * that assumption true. */
int verify_transform(const qlat_t *L, int ncheck)
{
    const int32_t IL = 32, JH = 32;
    uint32_t q;
    int bad = 0;
    const uint32_t QMAX = (ncheck > 2 ? (uint32_t)ncheck : 200u);

    for (q = 2; q <= QMAX; q++) {
        uint32_t r_enc;
        if (!fb_is_prime_power(q)) continue;  /* outside the contract */
        if (L->q % q == 0) continue;          /* q | det: no such ideal */
        for (r_enc = 0; r_enc < 2 * q; r_enc++) {
            uint32_t rt, g;
            const uint32_t m = pl_transform_enc(q, r_enc, L->a0, L->a1,
                                                L->b0, L->b1, &rt, &g);
            int32_t i, j;
            uint64_t nd = 0, np = 0;
            for (j = 0; j < JH; j++)
                for (i = -IL; i < IL; i++) {
                    const int d = hits_def(q, r_enc, L, i, j);
                    const int p = hits_pred(m, rt, g, i, j);
                    nd += (unsigned)d; np += (unsigned)p;
                    if (d != p) {
                        fprintf(stderr,
                            "verify_transform MISMATCH q=%u r=%u%s -> (m=%u,rt=%u,g=%u)"
                            " at (i=%d,j=%d): definition=%d predicted=%d\n",
                            q, r_enc, r_enc >= q ? " [projective]" : "",
                            m, rt, g, i, j, d, p);
                        bad++;
                        goto next_root;
                    }
                }
            if (nd == 0 || np == 0) {          /* an empty lattice is a red flag */
                fprintf(stderr, "verify_transform EMPTY q=%u r=%u\n", q, r_enc);
                bad++;
            }
        next_root: ;
        }
    }
    return bad ? -1 : 0;
}

/* The same gate, driven by a real factor base rather than synthetic moduli:
 * every entry's transform is checked by set equality over a small box. This is
 * what catches an encoding that the loader got wrong but the algebra handles.
 * Returns the number of entries checked, or -1. */
int verify_fb_transform(const fb_t *fb, const qlat_t *L, uint32_t maxq)
{
    const int32_t IL = 24, JH = 24;
    uint32_t k, checked = 0;
    for (k = 0; k < fb->n; k++) {
        const uint32_t q = fb->primes[k], r_enc = fb->roots[k];
        uint32_t rt, g, m;
        int32_t i, j;
        if (q > maxq) continue;
        if (L->q % q == 0) continue;
        m = pl_transform_enc(q, r_enc, L->a0, L->a1, L->b0, L->b1, &rt, &g);
        for (j = 0; j < JH; j++)
            for (i = -IL; i < IL; i++)
                if (hits_def(q, r_enc, L, i, j) != hits_pred(m, rt, g, i, j)) {
                    fprintf(stderr,
                        "verify_fb_transform MISMATCH entry %u: q=%u r=%u%s"
                        " -> (m=%u,rt=%u,g=%u) at (i=%d,j=%d)\n",
                        k, q, r_enc, r_enc >= q ? " [projective]" : "",
                        m, rt, g, i, j);
                    return -1;
                }
        checked++;
    }
    return (int)checked;
}

/* pl_transform_fast against pl_transform_enc, EXACT on (m, rt, g): the fast
 * path claims the same output, not merely the same lattice, so the walks it
 * produces -- and every relation downstream -- are identical. Three sources:
 *
 *   1. every entry of a real factor base under the gate's own lattice;
 *   2. small special-q (3, 5, 7, 11, 13), every special-q root, against every
 *      prime power up to 2000 and every root in [0, 2q). This is where a prime power p^k with p dividing the
 *      special-q lives: D and N can then share the factor p, and a fast path
 *      that checked rt*D == N instead of invertibility would return g = 1
 *      where pl_transform_gen returns g = p;
 *   3. deterministic random large moduli (primes, odd prime powers, powers of
 *      two) with random roots, under reduced lattices of random special-q and
 *      under raw coefficients up to and past the 2^32 guard. Every sample is
 *      kept inside pl_transform_gen's own int64 domain (q*|c| < 4e18), since
 *      it is the reference.
 *
 * With fb, only source 1; without, sources 2 and 3. Wherever the fast path
 * answers, its root is also checked against the definition in exact 64-bit
 * arithmetic, and every case is compared with pl_transform_enc. Returns the
 * number of cases compared, or -1 on the first failure; *nfast counts those
 * the fast path answered itself. Every random draw is its own statement:
 * C leaves the order of two calls in one expression unspecified, and the
 * Windows build's compiler must see the same sample stream. */
static uint32_t vtf_modq(int64_t v, uint32_t q)
{
    const int64_t r = v % (int64_t)q;
    return (uint32_t)(r < 0 ? r + (int64_t)q : r);
}

static uint32_t vtf_gcd(uint32_t a, uint32_t b)
{
    while (b) { const uint32_t t = a % b; a = b; b = t; }
    return a;
}

static int fast_cmp(uint32_t q, uint32_t r_enc, int64_t a0, int64_t a1,
                    int64_t b0, int64_t b1, uint64_t *nfast)
{
    uint32_t rt0, g0, rt1, g1, m0, m1, rtt;
    const int took = pl_transform_fast_try(q, r_enc, a0, a1, b0, b1, &rtt);
    if (took) {
        /* the definition, independently and exactly in 64 bits: D is
         * invertible mod q and rt*D == N, so rt is THE root (g = 1, m = q) */
        const int isp = r_enc >= q;
        const uint32_t r = isp ? r_enc - q : r_enc;
        const uint64_t A0 = vtf_modq(a0, q), A1 = vtf_modq(a1, q);
        const uint64_t B0 = vtf_modq(b0, q), B1 = vtf_modq(b1, q);
        const uint64_t D = isp ? ((uint64_t)r * A0 % q + q - A1) % q
                               : (A0 + q - (uint64_t)r * A1 % q) % q;
        const uint64_t N = isp ? (B1 + q - (uint64_t)r * B0 % q) % q
                               : ((uint64_t)r * B1 % q + q - B0) % q;
        if (vtf_gcd((uint32_t)D, q) != 1 || (uint64_t)rtt * D % q != N) {
            fprintf(stderr,
                "verify_transform_fast DEFINITION q=%u r=%u basis (%lld,%lld,%lld,%lld):"
                " fast rt=%u, D=%llu N=%llu\n", q, r_enc, (long long)a0,
                (long long)a1, (long long)b0, (long long)b1, rtt,
                (unsigned long long)D, (unsigned long long)N);
            return 1;
        }
        (*nfast)++;
    }
    m0 = pl_transform_enc(q, r_enc, a0, a1, b0, b1, &rt0, &g0);
    m1 = pl_transform_fast(q, r_enc, a0, a1, b0, b1, &rt1, &g1);
    if (m0 != m1 || rt0 != rt1 || g0 != g1) {
        fprintf(stderr,
            "verify_transform_fast MISMATCH q=%u r=%u basis (%lld,%lld,%lld,%lld):"
            " enc (m=%u,rt=%u,g=%u) fast (m=%u,rt=%u,g=%u)\n",
            q, r_enc, (long long)a0, (long long)a1, (long long)b0, (long long)b1,
            m0, rt0, g0, m1, rt1, g1);
        return 1;
    }
    return 0;
}

static uint64_t vtf_rng(uint64_t *s)
{
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s >> 11;
}

static uint32_t vtf_mulmod(uint32_t a, uint32_t b, uint32_t m)
{
    return (uint32_t)((uint64_t)a * b % m);
}

/* Deterministic Miller-Rabin for n < 2^32: bases 2, 7, 61. */
static int vtf_isprime(uint32_t n)
{
    static const uint32_t B[3] = {2, 7, 61};
    uint32_t d = n - 1, x, k;
    int s = 0, i, r;
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (!(n & 1u)) return 0;
    while (!(d & 1u)) { d >>= 1; s++; }
    for (i = 0; i < 3; i++) {
        uint32_t a = B[i] % n, e = d;
        if (!a) continue;
        x = 1;
        k = a;
        while (e) { if (e & 1u) x = vtf_mulmod(x, k, n); k = vtf_mulmod(k, k, n); e >>= 1; }
        if (x == 1 || x == n - 1) continue;
        for (r = 1; r < s; r++) {
            x = vtf_mulmod(x, x, n);
            if (x == n - 1) break;
        }
        if (r == s) return 0;
    }
    return 1;
}

int verify_transform_fast(const fb_t *fb, const qlat_t *L, uint64_t *nfast)
{
    static const uint64_t SQ[] = {3, 5, 7, 11, 13};
    static const double SKEW[] = {1.0, 3.0e5};
    uint64_t s = 0x9e3779b97f4a7c15ull;
    uint32_t pool[4096];
    int npool = 0, n = 0, i;
    uint32_t k;

    *nfast = 0;
    /* 1. the real factor base, and nothing else when one is given */
    if (fb) {
        for (k = 0; k < fb->n; k++, n++)
            if (fast_cmp(fb->primes[k], fb->roots[k], L->a0, L->a1, L->b0, L->b1, nfast))
                return -1;
        return n;
    }

    /* 2. small special-q, every prime power to 2000, every root */
    for (i = 0; i < (int)(sizeof(SQ) / sizeof(SQ[0])); i++) {
        int z;
        for (z = 0; z < 2; z++) {
            uint64_t rho;
            for (rho = 0; rho < SQ[i]; rho++) {
                qlat_t M;
                uint32_t q, r;
                qlat_build(&M, SQ[i], rho, SKEW[z]);
                for (q = 2; q <= 2000; q++) {
                    if (!fb_is_prime_power(q)) continue;
                    for (r = 0; r < 2 * q; r++, n++)
                        if (fast_cmp(q, r, M.a0, M.a1, M.b0, M.b1, nfast)) return -1;
                }
            }
        }
    }

    /* 3. random large moduli */
    while (npool < 3072) {                       /* primes, 2^16 .. 2^32 */
        const int bits = 17 + (int)(vtf_rng(&s) % 16);
        uint32_t c = (uint32_t)(vtf_rng(&s) & ((bits == 32) ? 0xFFFFFFFFull
                                                            : ((1ull << bits) - 1)));
        c |= (1u << (bits - 1)) | 1u;
        while (!vtf_isprime(c) && c < 0xFFFFFFF0u) c += 2;
        if (vtf_isprime(c)) pool[npool++] = c;
    }
    {                                            /* odd prime powers, powers of 2 */
        static const uint32_t P[] = {3, 5, 7, 11, 13, 101, 257, 65521};
        for (i = 0; i < (int)(sizeof(P) / sizeof(P[0])) && npool < 4096; i++) {
            uint64_t v = (uint64_t)P[i] * P[i];
            for (; v < (1ull << 32) && npool < 4096; v *= P[i])
                pool[npool++] = (uint32_t)v;
        }
        for (k = 1; k < 32 && npool < 4096; k++) pool[npool++] = 1u << k;
    }
    for (i = 0; i < 400000; i++) {
        const uint32_t q = pool[vtf_rng(&s) % (uint64_t)npool];
        const uint32_t r = (uint32_t)(vtf_rng(&s) % (2ull * q));
        int64_t c[4];
        int t;
        if (i & 1) {
            /* a reduced lattice of a random special-q up to 2^40 */
            qlat_t M;
            const uint64_t sqbits = 5 + vtf_rng(&s) % 36;
            const uint64_t sq = 3 + vtf_rng(&s) % (1ull << sqbits);
            const uint64_t rho = vtf_rng(&s) % sq;
            const double skew = exp((double)(vtf_rng(&s) % 40));
            qlat_build(&M, sq, rho, skew);
            c[0] = M.a0; c[1] = M.a1; c[2] = M.b0; c[3] = M.b1;
            if (sq % q == 0) continue;
            for (t = 0; t < 4; t++)          /* the reference's own int64 domain */
                if ((double)q * fabs((double)c[t]) >= 4.0e18) break;
            if (t < 4) continue;
        } else {
            /* raw coefficients, magnitude classes up to and past 2^32; keep
             * the reference's r*c inside int64 */
            for (t = 0; t < 4; t++) {
                const uint64_t u = vtf_rng(&s);
                int64_t v;
                switch (u % 6) {
                case 0:  v = (int64_t)(u >> 8) % 1000; break;
                case 1:  v = (int64_t)((u >> 8) & 0x7FFFFFFFull); break;
                case 2:  v = 0xFFFFFFFFll; break;
                case 3:  v = 0x100000000ll; break;
                case 4:  v = (int64_t)((u >> 8) & 0xFFFFFFFFull); break;
                default: v = (int64_t)((u >> 8) & 0x3FFFFFFFFFull); break;
                }
                if ((u >> 3) & 1u) v = -v;
                while ((double)q * (double)(v < 0 ? -v : v) >= 4.0e18) v /= 2;
                c[t] = v;
            }
        }
        n++;
        if (fast_cmp(q, r, c[0], c[1], c[2], c[3], nfast)) return -1;
    }
    return n;
}

int verify_walk(int logI, uint32_t J, int nprimes)
{
    /* Primes just above I are the interesting regime: that is where the
     * FK reduction actually has work to do. */
    const uint32_t I = 1u << logI;
    uint32_t p, checked = 0;
    for (p = I + 1; checked < (uint32_t)nprimes && p < 40u * I; p += 2) {
        uint32_t d, isprime = 1;
        for (d = 3; (uint64_t)d * d <= p; d += 2) if (p % d == 0) { isprime = 0; break; }
        if (!isprime) continue;
        /* a spread of roots including the r == 0 special case */
        for (uint32_t k = 0; k < 5; k++) {
            uint32_t r = (uint32_t)(((uint64_t)p * k) / 5);
            if (check_one(p, r, logI, J) != 0) return -1;
        }
        checked++;
    }
    return (int)checked;
}

/* Independent wide-coordinate oracle for p > I. There is then at most one
 * lattice hit in each j row, so enumerate the defining congruence directly and
 * compare it with pl_next64(). This does not share either the reduced-increment
 * construction or the slab-continuation recurrence with the walk under test. */
static int check_one64_sparse(uint32_t p, uint32_t r, int logI, uint32_t J)
{
    const uint32_t I = 1u << logI, Imask = I - 1u;
    const int64_t half = (int64_t)I / 2;
    const uint64_t xmax = (uint64_t)I * J;
    const plat_t P = pl_make(p, r, logI);
    uint64_t walk = (uint64_t)1 << (logI - 1);   /* (0,0) */

    if (p <= I) return -1;
    for (uint32_t j = 0; j < J; j++) {
        const int64_t base = (int64_t)(((uint64_t)r * j) % p);
        int64_t i = INT64_MAX;
        if (base < half) i = base;
        else if (base - (int64_t)p >= -half) i = base - (int64_t)p;
        if (i != INT64_MAX) {
            const uint64_t want = (uint64_t)(i + half) + ((uint64_t)j << logI);
            if (walk != want) {
                fprintf(stderr,
                        "verify_walk64 MISMATCH p=%u r=%u logI=%d J=%u j=%u:"
                        " walk=%llu ref=%llu\n",
                        p, r, logI, J, j,
                        (unsigned long long)walk,
                        (unsigned long long)want);
                return -1;
            }
            walk = pl_next64(walk, &P, Imask);
        }
    }
    if (walk < xmax) {
        fprintf(stderr,
                "verify_walk64 SHORT p=%u r=%u logI=%d J=%u:"
                " next=%llu xmax=%llu\n",
                p, r, logI, J,
                (unsigned long long)walk, (unsigned long long)xmax);
        return -1;
    }
    return 0;
}

/* Forced-slab equivalence gate. The oracle is an exact 64-bit uninterrupted
 * walk, not the legacy uint32 walk. Large factor-base primes are deliberate:
 * realistic reduced increments can exceed 2^32 even at I15/I16, which was the
 * hole in the first version of this gate. */
int verify_walk_slabs(int logI, uint32_t J, uint32_t slab_J, int nprimes)
{
    const uint32_t I = 1u << logI, Imask = I - 1u;
    const uint64_t xmax = (uint64_t)I * J;
    uint32_t p = 250000013u, checked = 0;
    int saw_wide_increment = 0, saw_slab32 = 0;
    if (!slab_J || slab_J > J || nprimes <= 0) return -1;

    for (; checked < (uint32_t)nprimes && p < 250100000u; p += 2) {
        uint32_t d, isprime = 1;
        for (d = 3; (uint64_t)d * d <= p; d += 2)
            if (p % d == 0) { isprime = 0; break; }
        if (!isprime) continue;
        for (uint32_t k = 0; k < 5; k++) {
            const uint32_t r = (uint32_t)(((uint64_t)p * k) / 5);
            const plat_t P = pl_make(p, r, logI);
            if (checked == 0 && check_one64_sparse(p, r, logI, J)) return -1;
            pl_basis_t B = { p, 0u, r, 1u };
            uint64_t want_warp, want_step;
            uint64_t mono, cur, base_x = 0;

            /* Independently spell the widening boundary after reduction. This
             * catches a future regression that stores (j << logI) in 32 bits
             * even though both continuation paths consume the same plat_t. */
            pl_reduce(&B, I);
            want_warp = ((uint64_t)B.j0 << logI) - B.mi0;
            want_step = (B.i1 >> logI) ? PL_VERTICAL
                                       : ((uint64_t)B.j1 << logI) + B.i1;
            if (want_warp > UINT32_MAX ||
                (want_step != PL_VERTICAL && want_step > UINT32_MAX))
                saw_wide_increment = 1;
            if (P.inc_warp != want_warp || P.inc_step != want_step) {
                fprintf(stderr,
                        "verify_walk_slabs PLAT WIDTH p=%u r=%u logI=%d:"
                        " got %llu/%llu want %llu/%llu\n",
                        p, r, logI,
                        (unsigned long long)P.inc_warp,
                        (unsigned long long)P.inc_step,
                        (unsigned long long)want_warp,
                        (unsigned long long)want_step);
                return -1;
            }

            mono = pl_first64(&P, logI);
            cur = mono;
            for (uint32_t jb = 0; jb < J; jb += slab_J) {
                const uint32_t rows = J - jb < slab_J ? J - jb : slab_J;
                const uint64_t sx = (uint64_t)I * rows;
                uint64_t x = cur;
                /* pl_next's saturation contract, exercised from a MID-WALK
                 * start rather than only from pl_first. The U32 block at the
                 * end of this function covers a walk that begins at the origin
                 * of a rectangle; a slab continuation begins anywhere, which is
                 * the case any future 32-bit-local kernel would rely on.
                 * k_fill_atomic and k_resieve_scatter walk in 64 bits, but
                 * k_fill_l1 still walks in 32 -- deliberately: its
                 * __launch_bounds__(512, 3) caps it at 40 registers and a
                 * 64-bit walk makes ptxas spill. So pl_next is a live device
                 * path, not just a CPU oracle, and this gate protects it. */
                const int has32 = (cur < sx) && (sx <= 0x80000000ull);
                if (has32) saw_slab32 = 1;
                uint32_t x32 = has32 ? (uint32_t)cur : 0u, last32 = x32;
                while (x < sx) {
                    const uint64_t gx = base_x + x;
                    if (mono >= xmax || gx != mono) {
                        fprintf(stderr,
                                "verify_walk_slabs MISMATCH p=%u r=%u logI=%d"
                                " J=%u slabJ=%u: slab=%llu mono=%llu\n",
                                p, r, logI, J, slab_J,
                                (unsigned long long)gx,
                                (unsigned long long)mono);
                        return -1;
                    }
                    if (has32 && (uint64_t)x32 != x) {
                        fprintf(stderr,
                                "verify_walk_slabs SLAB32 p=%u r=%u logI=%d"
                                " J=%u slabJ=%u: x32=%u x64=%llu\n",
                                p, r, logI, J, slab_J, x32,
                                (unsigned long long)x);
                        return -1;
                    }
                    last32 = x32;
                    x32 = pl_next(x32, &P, Imask);
                    mono = pl_next64(mono, &P, Imask);
                    x = pl_next64(x, &P, Imask);
                }
                if (has32) {
                    if ((uint64_t)x32 < sx) {
                        fprintf(stderr,
                                "verify_walk_slabs SLAB32 did not leave the"
                                " slab p=%u r=%u logI=%d slabJ=%u x32=%u"
                                " sx=%llu\n", p, r, logI, slab_J, x32,
                                (unsigned long long)sx);
                        return -1;
                    }
                    /* The saturation contract AT the boundary. x is the exact
                     * wide first hit past the slab; pl_next from the same
                     * predecessor must reproduce it when it fits in 32 bits and
                     * must saturate, not wrap, when it does not.
                     *
                     * The obvious spelling of this check -- comparing
                     * pl_next64(last32) against x -- is a TAUTOLOGY: the loop
                     * above assigns x = pl_next64(x) from the very value
                     * last32 holds, so it compares pl_next64(v) with itself and
                     * cannot fail. Compare the 32-bit result instead. */
                    if (x > 0xFFFFFFFFull) {
                        if (x32 != 0xFFFFFFFFu) {
                            fprintf(stderr,
                                    "verify_walk_slabs SLAB32 CONT p=%u r=%u"
                                    " logI=%d slabJ=%u: wide next %llu did not"
                                    " saturate, got %u\n", p, r, logI, slab_J,
                                    (unsigned long long)x, x32);
                            return -1;
                        }
                    } else if ((uint64_t)x32 != x) {
                        fprintf(stderr,
                                "verify_walk_slabs SLAB32 CONT p=%u r=%u"
                                " logI=%d slabJ=%u: from last=%u got %u"
                                " want %llu\n", p, r, logI, slab_J, last32,
                                x32, (unsigned long long)x);
                        return -1;
                    }
                }
                if (base_x + x != mono) {
                    fprintf(stderr,
                            "verify_walk_slabs STATE p=%u r=%u logI=%d J=%u"
                            " slabJ=%u: base+x=%llu mono=%llu\n",
                            p, r, logI, J, slab_J,
                            (unsigned long long)(base_x + x),
                            (unsigned long long)mono);
                    return -1;
                }
                cur = x - sx;
                base_x += sx;
            }
            if (mono < xmax || base_x != xmax || base_x + cur != mono) {
                fprintf(stderr,
                        "verify_walk_slabs SHORT p=%u r=%u logI=%d J=%u"
                        " slabJ=%u next=%llu xmax=%llu\n",
                        p, r, logI, J, slab_J,
                        (unsigned long long)mono,
                        (unsigned long long)xmax);
                return -1;
            }

            /* Where the whole rectangle fits the legacy local coordinate,
             * also prove the bounded uint32 walk has exactly the same in-range
             * hits and merely saturates when the true next hit is wider. */
            if (xmax <= 0x80000000ull) {
                uint32_t x32 = pl_first(&P, logI);
                uint64_t x64 = pl_first64(&P, logI);
                while (x64 < xmax) {
                    if ((uint64_t)x32 != x64) {
                        fprintf(stderr,
                                "verify_walk_slabs U32 p=%u r=%u logI=%d:"
                                " x32=%u x64=%llu\n", p, r, logI, x32,
                                (unsigned long long)x64);
                        return -1;
                    }
                    x32 = pl_next(x32, &P, Imask);
                    x64 = pl_next64(x64, &P, Imask);
                }
                if ((uint64_t)x32 < xmax) {
                    fprintf(stderr,
                            "verify_walk_slabs U32 did not terminate p=%u r=%u"
                            " logI=%d x32=%u xmax=%llu\n", p, r, logI, x32,
                            (unsigned long long)xmax);
                    return -1;
                }
            }
        }
        checked++;
    }
    if (!saw_wide_increment) {
        fprintf(stderr,
                "verify_walk_slabs did not exercise a >32-bit lattice increment"
                " at logI=%d J=%u slabJ=%u\n", logI, J, slab_J);
        return -1;
    }
    /* Same fail-closed rule for the SLAB32 block: it self-disables when a slab
     * exceeds the 32-bit local coordinate (sx > 2^31), which is silent. A case
     * added later with I*slabJ = 2^32 would leave k_fill_l1's walk ungated
     * while slabtest still reported success. */
    if (!saw_slab32) {
        fprintf(stderr,
                "verify_walk_slabs SLAB32 block never ran at logI=%d J=%u"
                " slabJ=%u (slab area exceeds 2^31?): the 32-bit walk is"
                " UNGATED for this case\n", logI, J, slab_J);
        return -1;
    }
    return (int)checked;
}

/* Shared forced-slab cases for both --verify-only and slabtest. Keeping the
 * table here prevents the two gates from silently drifting apart. The final
 * three cases exercise native I17+, including hundreds of slab transitions at
 * I20, while the large-prime oracle above keeps their hit count modest. */
int verify_walk_slab_cases(void)
{
    /* Exact reproducer from the slab review: the old uint32 plat_t stored
     * inc_warp=111280357 instead of 4406247653 and enumerated 2431 hits where
     * the defining congruence has 235. Keep it as a named regression. */
    {
        const plat_t P = pl_make(10000019u, 6000067u, 17);
        if (P.inc_warp != 4406247653ull ||
            check_one64_sparse(10000019u, 6000067u, 17, 65536u)) {
            fprintf(stderr,
                    "verify_walk_slab_cases FAILED 32-bit-overflow reproducer:"
                    " inc_warp=%llu\n",
                    (unsigned long long)P.inc_warp);
            return -1;
        }
    }
    static const struct { int logI; uint32_t J, slabJ; } cases[] = {
        { 8,       512u,   127u },
        { 10,     1024u,   257u },
        { 15,    16384u,  4095u },
        { 16,    32768u,  8191u },
        { 17,    65536u, 16384u },
        { 18,   131072u,  8192u },
        { 20,   524288u,  2048u }
    };
    const unsigned ncase = sizeof(cases) / sizeof(cases[0]);
    for (unsigned k = 0; k < ncase; k++) {
        const int checked = verify_walk_slabs(cases[k].logI, cases[k].J,
                                              cases[k].slabJ, 8);
        if (checked != 8) {
            fprintf(stderr,
                    "verify_walk_slab_cases FAILED case %u logI=%d J=%u"
                    " slabJ=%u: checked %d/8 primes\n",
                    k, cases[k].logI, cases[k].J, cases[k].slabJ, checked);
            return -1;
        }
    }
    return (int)ncase;
}

/* Ground truth for the apply kernel: replay one region's bucket records into a
 * 16-bit cell array exactly as k_apply should, and apply the same threshold.
 * Reproduces the GPU's *sign convention* too -- init to CINIT - T, add, test
 * >= CINIT -- so a mismatch means a real kernel bug and not a modelling
 * difference. */
uint32_t verify_apply_region(const uint32_t *records, uint32_t nrec,
                             const norm_t *N, int logI, int log_region,
                             uint32_t region, int norm_mode,
                             uint32_t Cinit, uint32_t tconst,
                             const uint32_t *sp, const uint32_t *srt,
                             const uint32_t *sg, const uint16_t *slp,
                             uint32_t nsmall, uint16_t *cells)
{
    const uint32_t ncell = 1u << log_region;
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = 1 << (logI - 1);
    const uint32_t xbase = region << log_region;
    uint32_t c, surv = 0;

    for (c = 0; c < ncell; c++) {
        uint32_t t;
        if (norm_mode == NORM_CONST) t = tconst;
        else {
            uint32_t x = xbase + c;
            float lg = norm_target_host(N, (int32_t)(x & Imask) - Ihalf, x >> logI);
            int ti = (int)floorf(lg + 0.5f);
            /* ceiling is the cell's, not the byte's -- see k_apply */
            const uint32_t TMAX = (Cinit > 255u) ? Cinit : 255u;
            t = (ti < 0) ? 0u : ((uint32_t)ti > TMAX ? TMAX : (uint32_t)ti);
        }
        cells[c] = (uint16_t)(Cinit - t);
    }
    /* small primes, replayed independently of the GPU's tiering: a flat loop
     * over every entry, so a bug in the block/warp/thread split shows up as a
     * cell difference rather than hiding behind matching code */
    {
        const uint32_t j = xbase >> logI;
        const int32_t ilo = (int32_t)(xbase & Imask) - Ihalf;
        uint32_t e;
        for (e = 0; e < nsmall; e++) {
            uint32_t m = sp[e], rt = srt[e], g = sg[e], lp = slp[e], off, jj;
            if (g > 1 && (j % g)) continue;
            jj = (g > 1) ? j / g : j;
            {
                int32_t base = (int32_t)(((uint64_t)rt * jj) % m);
                int32_t s = (base - ilo) % (int32_t)m;
                if (s < 0) s += (int32_t)m;
                for (off = (uint32_t)s; off < ncell; off += m) cells[off] += (uint16_t)lp;
            }
        }
    }
    for (c = 0; c < nrec; c++) {
        uint32_t r = records[c];
        cells[r & (ncell - 1)] += r >> 16;      /* the record carries log p (finding 130) */
    }
    for (c = 0; c < ncell; c++) if (cells[c] >= Cinit) surv++;
    return surv;
}

/* Reference record count, and -- when `per_region` is non-NULL -- the reference
 * count for every region.
 *
 * A global total is a weak gate: it cannot tell "right total, wrong region"
 * from "right". The transposed-basis bug this project already wrote up was
 * exactly that shape, and so was the projective-reciprocal bug (same density,
 * wrong congruence). Comparing all 32K region counts against the GPU's cursors
 * gates PLACEMENT, at the same host cost.
 *
 * This runs the device's own transform, pl_transform_fast, so it validates the
 * walk and the fill, not the transform. verify_transform gates
 * pl_transform_enc against the definition, and verify_transform_fast gates
 * pl_transform_fast against both. */
uint64_t verify_count_updates(const fb_t *fb, const qlat_t *L,
                              int logI, uint32_t J,
                              int log_region, uint32_t *per_region)
{
    const uint32_t I = 1u << logI, Imask = I - 1;
    const uint32_t xmax = I * J;
    uint64_t total = 0, nrow = 0;
    uint32_t k;

    if (per_region)
        memset(per_region, 0, (size_t)(xmax >> log_region) * sizeof *per_region);

    for (k = 0; k < fb->n; k++) {
        const uint32_t q = fb->primes[k];
        uint32_t rt, g, m;
        plat_t P;
        uint32_t x;
        m = pl_transform_fast(q, fb->roots[k], L->a0, L->a1, L->b0, L->b1, &rt, &g);
        if (g > 1) { nrow++; continue; }      /* the kernel drops these too */
        P = pl_make(m, rt, logI);
        for (x = pl_first(&P, logI); x < xmax; x = pl_next(x, &P, Imask)) {
            total++;
            if (per_region) per_region[x >> log_region]++;
        }
    }
    if (nrow)
        fprintf(stderr, "  (cpu ref: %llu row-confined transformed roots skipped)\n",
                (unsigned long long)nrow);
    return total;
}

/* --fill-rows' slab handoff (k_rows_walk_next) jumps to the next strip row
 * with pl_rows_to_hit instead of stepping row by row. Check it against
 * stepping, over random lattices plus the roots whose gaps run to ~q - I rows
 * (rt = 1, q - 1, just past q/2 and q/3) and rt = 0, which has no answer once
 * off >= I. Returns the number of cases checked, -1 on a mismatch. */
int verify_rows_to_hit(void)
{
    static const int logIs[] = { 8, 11, 13 };
    uint64_t s = 0x9e3779b97f4a7c15ull;
    int ncase = 0;
#define XS() (s ^= s << 13, s ^= s >> 7, s ^= s << 17, s)
    for (unsigned li = 0; li < sizeof logIs / sizeof logIs[0]; li++) {
        const uint32_t I = 1u << logIs[li];
        for (int c = 0; c < 400; c++) {
            uint32_t q = I + (uint32_t)(XS() % (63u * I));   /* [I, 64I) */
            uint32_t rt, off, o;
            uint64_t t = 0, want, got;
            if (c % 50 == 0) q = 4u * I;                     /* a power of two */
            if (c % 50 == 25) q = I;                         /* the tier floor */
            rt = (uint32_t)(XS() % q);
            switch (c % 10) {
            case 1: rt = 1; break;
            case 2: rt = q - 1; break;
            case 3: rt = q / 2 + 1; break;
            case 4: rt = q / 3 + 1; break;
            case 5: rt = 0; break;
            default: break;
            }
            off = (uint32_t)(XS() % q);
            for (o = off; o >= I && t <= q; t++) {
                o += rt;
                if (o >= q) o -= q;
            }
            want = (o < I) ? t : UINT64_MAX;
            got = pl_rows_to_hit(off, rt, q, I);
            if (got != want) {
                fprintf(stderr, "verify_rows_to_hit MISMATCH I=%u q=%u rt=%u"
                        " off=%u: got %llu, stepping gives %llu\n", I, q, rt,
                        off, (unsigned long long)got, (unsigned long long)want);
                return -1;
            }
            ncase++;
        }
    }
#undef XS
    return ncase;
}
