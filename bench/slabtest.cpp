/* CPU-only gates for j-slabbing. No CUDA runtime or GPU required. */
#include "bench.h"
#include "plattice.cuh"
#include "td.cuh"
#include <stdio.h>
#include <stdint.h>

static int check_plan(void)
{
    slab_plan_t P = {0,0,0};
    const struct {
        int logI, lreg; uint32_t J, pmax, forceJ, want_j, want_n;
    } v[] = {
        /* Performance slabbing must not touch areas below 2^30. */
        {16,14,  4096u, 65535u, 0u,      4096u, 1u}, /* 2^28 */
        {16,14,  8192u, 65535u, 0u,      8192u, 1u}, /* 2^29 */
        {16,14, 16383u, 65535u, 0u,     16383u, 1u}, /* just below 2^30 */
        /* At 2^30 and above, auto mode targets 32768 bucket regions, which at
         * the default --region 14 is 2^29 positions -- the value this table
         * pinned before the target became region-relative (finding 79). Every
         * row below is unchanged, which is the point: the fix is
         * behaviour-preserving at the default. */
        {16,14, 16384u, 65535u, 0u,      8192u, 2u}, /* 2^30 */
        {16,14, 32768u, 65535u, 0u,      8192u, 4u}, /* 2^31 */
        {16,14, 65536u, 65535u, 0u,      8192u, 8u}, /* 2^32 */
        {17,14, 65536u,131071u, 0u,      4096u,16u},
        {18,14,131072u,262143u, 0u,      2048u,64u},
        {19,14,262144u,524287u, 0u,      1024u,256u},
        {20,14,524288u,1048575u,0u,       512u,1024u},
        {15,14, 16384u, 32767u, 0u,     16384u, 1u}, /* 2^29 */
        {15,14, 16384u, 32767u, 123u,      123u,134u},
        /* Explicit --slab-j may override the performance target upward. */
        {16,14, 32768u, 65535u,16384u,  16384u, 2u},
        {17,14, 65536u,262143u, 0u,      4096u,16u},
        /* THE BUG THIS SIGNATURE EXISTS TO CLOSE (finding 79). The target is a
         * region COUNT, so halving --region must halve the slab AREA. Before
         * the fix these three rows all returned the region-14 answer, leaving
         * fill +28.4% at region 13 and +68.3% at region 12. */
        {16,13, 16384u, 65535u, 0u,      4096u, 4u}, /* 2^28/slab */
        {16,12, 16384u, 65535u, 0u,      2048u, 8u}, /* 2^27/slab */
        /* Region 15 doubles the slab AREA against the old absolute 2^29
         * target -- the region COUNT is what is held at 32768. Peak bucket
         * memory follows the area, so this row pins a 2x growth that is
         * intended policy, not a regression. Region 16 would be 4x. */
        {16,15, 32768u, 65535u, 0u,     16384u, 2u}, /* 2^30/slab */
    };
    for (unsigned k = 0; k < sizeof(v)/sizeof(v[0]); k++) {
        if (slab_make_plan(v[k].logI, v[k].lreg, v[k].J, v[k].pmax,
                           v[k].forceJ, &P) ||
            P.jmax != v[k].want_j || P.nslab != v[k].want_n ||
            P.enabled != (v[k].want_n > 1u)) {
            fprintf(stderr, "slabtest: plan %u got jmax=%u n=%u enabled=%d;"
                    " want %u/%u/%d\n", k, P.jmax, P.nslab, P.enabled,
                    v[k].want_j, v[k].want_n, v[k].want_n > 1u);
            return -1;
        }
        {
            uint64_t covered = 0;
            for (uint32_t slab = 0; slab < P.nslab; slab++) {
                const uint32_t base = slab_jbase_at(&P, slab);
                const uint32_t rows = slab_rows_at(&P, v[k].J, slab);
                if ((uint64_t)base != covered || !rows || rows > P.jmax) {
                    fprintf(stderr,
                            "slabtest: plan %u coverage error at slab %u:"
                            " base=%u covered=%llu rows=%u jmax=%u\n",
                            k, slab, base, (unsigned long long)covered, rows, P.jmax);
                    return -1;
                }
                covered += rows;
            }
            if (covered != v[k].J) {
                fprintf(stderr, "slabtest: plan %u covered %llu rows, want %u\n",
                        k, (unsigned long long)covered, v[k].J);
                return -1;
            }
        }
    }
    /* A forced slab is allowed to be smaller, never larger than a safety cap. */
    /* The new log_region parameter must be USED, not merely accepted: an
     * out-of-range region has to fail the plan, and a region that moves the
     * target has to move the answer (the table above covers 12/13/15). */
    if (!slab_make_plan(16, 0,  16384u, 65535u, 0u, &P) ||
        !slab_make_plan(16, 31, 16384u, 65535u, 0u, &P)) {
        fprintf(stderr, "slabtest: log_region out of range must fail the plan\n");
        return -1;
    }
    /* The same on the FORCED path, where log_region is unused. It was accepted
     * there until 2026-09-02 because the only range check lived in
     * slab_perf_jmax, which the forced branch never reaches -- so this row
     * covers a branch the two above cannot. */
    if (!slab_make_plan(16, 0,  16384u, 65535u, 8192u, &P) ||
        !slab_make_plan(16, 31, 16384u, 65535u, 8192u, &P)) {
        fprintf(stderr, "slabtest: log_region out of range must fail a FORCED"
                        " plan too\n");
        return -1;
    }
    /* The trigger scales with the target and keeps its factor of two. At
     * region 12 the target is 2^27, so splitting starts at 2^28: J 4095 (area
     * just under 2^28) is one slab, J 4096 (2^28) splits into two. Before the
     * trigger was made region-relative, BOTH ran unsplit -- J 8192 at region 12
     * gave one slab of 131,072 regions, 4x the target. */
    if (slab_make_plan(16, 12, 4095u, 65535u, 0u, &P) || P.nslab != 1u) {
        fprintf(stderr, "slabtest: region-12 below trigger got nslab=%u\n", P.nslab);
        return -1;
    }
    if (slab_make_plan(16, 12, 4096u, 65535u, 0u, &P) || P.nslab != 2u ||
        P.jmax != 2048u) {
        fprintf(stderr, "slabtest: region-12 at trigger got jmax=%u nslab=%u\n",
                P.jmax, P.nslab);
        return -1;
    }
    if (slab_make_plan(16, 12, 8192u, 65535u, 0u, &P) || P.nslab != 4u) {
        fprintf(stderr, "slabtest: region-12 2^29 got nslab=%u (was 1 unsplit)\n",
                P.nslab);
        return -1;
    }
    if (!slab_make_plan(17, 14, 65536u, 131071u, 20000u, &P)) {
        fprintf(stderr, "slabtest: unsafe forced slab height was accepted\n");
        return -1;
    }
    /* At logI=6 a TD rank group spans four j rows. Reject a forced height
     * that would leave partial groups, and accept an aligned one. */
    if (!slab_make_plan(6, 14, 64u, 63u, 3u, &P) ||
        slab_make_plan(6, 14, 64u, 63u, 4u, &P)) {
        fprintf(stderr, "slabtest: slab row-quantum validation failed\n");
        return -1;
    }
    return 0;
}

static int check_bkthresh_integration(void)
{
    /* Exercise the same path as --pipeline after --bkthresh is resolved:
     * validate FB -> split the line/direct tier -> find its largest prime ->
     * build the slab plan. 131072 is also included as a proper power to make
     * sure powers retained by fb_split_small() do not inflate the direct-TD
     * prime bound. */
    uint32_t primes[] = { 2u, 3u, 5u, 131071u, 131072u, 131101u, 262139u, 1048573u };
    uint32_t roots[]  = { 1u, 1u, 1u,      1u,      1u,      1u,      1u,       1u };
    uint8_t  ispow[]  = { 0u, 0u, 0u,      0u,      1u,      0u,      0u,       0u };
    fb_t fb = {}, small = {};
    slab_plan_t P = {0,0,0};
    static const struct {
        uint32_t bkthresh, want_pmax, want_jmax, want_nslab;
    } v[] = {
        {  131072u,  131071u, 4096u, 16u },
        {  262144u,  262139u, 4096u, 16u },
        /* Here the direct-TD safety cap is tighter than the 2^29 target. */
        { 1048576u, 1048573u, 2048u, 32u },
    };

    fb.n = (uint32_t)(sizeof(primes) / sizeof(primes[0]));
    fb.primes = primes;
    fb.roots = roots;
    fb.ispow = ispow;
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIME_POWERS, NULL)) {
        fprintf(stderr, "slabtest: synthetic FB validation failed\n");
        return -1;
    }

    for (unsigned k = 0; k < sizeof(v)/sizeof(v[0]); k++) {
        uint32_t pmax;
        if (fb_split_small(&fb, v[k].bkthresh, &small)) {
            fprintf(stderr, "slabtest: fb_split_small failed for bkthresh=%u\n",
                    v[k].bkthresh);
            return -1;
        }
        pmax = fb_max_td_prime(&small);
        if (pmax != v[k].want_pmax ||
            slab_make_plan(17, 14, 65536u, pmax, 0u, &P) ||
            P.jmax != v[k].want_jmax || P.nslab != v[k].want_nslab) {
            fprintf(stderr,
                    "slabtest: bkthresh=%u integration got pmax=%u jmax=%u"
                    " n=%u; want %u/%u/%u\n",
                    v[k].bkthresh, pmax, P.jmax, P.nslab,
                    v[k].want_pmax, v[k].want_jmax, v[k].want_nslab);
            fb_free(&small);
            return -1;
        }
        fb_free(&small);
    }
    /* The same raised threshold must reject a forced slab that would have
     * been safe at the default threshold. */
    if (!slab_make_plan(17, 14, 65536u, 262139u, 16384u, &P)) {
        fprintf(stderr,
                "slabtest: raised-bkthresh unsafe forced slab was accepted\n");
        return -1;
    }
    return 0;
}

static int check_phase(void)
{
    uint32_t seed = 0x6d2b79f5u;
    for (unsigned t = 0; t < 20000; t++) {
        const int logI = 10 + (int)(seed % 11u);
        const uint32_t I = 1u << logI;
        uint32_t m, rt, cst, base, local, hi, shifted;
        seed = seed * 1664525u + 1013904223u;
        m = 2u + seed % (I - 1u);
        seed = seed * 1664525u + 1013904223u;
        rt = seed % m;
        seed = seed * 1664525u + 1013904223u;
        base = seed & ((1u << 20) - 1u);
        seed = seed * 1664525u + 1013904223u;
        local = seed & 2047u;
        seed = seed * 1664525u + 1013904223u;
        hi = 1u + seed % I;
        cst = (I >> 1) % m;
        shifted = slab_phase_cst(cst, rt, m, base);
        {
            const uint32_t a = (uint32_t)(((uint64_t)rt * (base + local) + hi) % m);
            const uint32_t b = (uint32_t)(((uint64_t)rt * local + hi) % m);
            if ((a == cst) != (b == shifted)) {
                fprintf(stderr,
                        "slabtest: phase mismatch logI=%d m=%u rt=%u base=%u"
                        " local=%u hi=%u cst=%u shifted=%u a=%u b=%u\n",
                        logI, m, rt, base, local, hi, cst, shifted, a, b);
                return -1;
            }
        }
    }
    return 0;
}

static int check_walk_continuation(void)
{
    const int ncase = verify_walk_slab_cases();
    if (ncase < 0) {
        fprintf(stderr, "slabtest: shared walk continuation cases failed\n");
        return -1;
    }
    return 0;
}

static int check_magic(void)
{
    uint32_t seed = 0x31415926u;
    /* Edge divisors plus randomized values through the largest default I. */
    for (unsigned t = 0; t < 20000; t++) {
        uint32_t m, magic, sh;
        if (t < 20) {
            static const uint32_t edge[] = {2,3,4,5,7,8,15,16,31,32,63,64,
                                             127,128,255,256,1023,32767,131071,1048575};
            m = edge[t];
        } else {
            seed = seed * 1664525u + 1013904223u;
            m = 2u + seed % 1048574u;
        }
        td_magic_build(m, &magic, &sh);
        for (unsigned k = 0; k < 12; k++) {
            uint32_t w;
            if (k == 0) w = 0;
            else if (k == 1) w = 1;
            else if (k == 2) w = m - 1u;
            else if (k == 3) w = m;
            else if (k == 4) w = 0x7fffffffu;
            else {
                seed = seed * 1664525u + 1013904223u;
                w = seed & 0x7fffffffu;
            }
            if (td_mod_magic(w, m, magic, sh) != w % m) {
                fprintf(stderr,
                        "slabtest: magic mismatch m=%u w=%u magic=%u sh=%u"
                        " got=%u ref=%u\n", m, w, magic, sh,
                        td_mod_magic(w, m, magic, sh), w % m);
                return -1;
            }
        }
    }
    return 0;
}

/* ss_first's magic path, one- and two-step, written against the same
 * td_mod_magic and SS_KSHIFT the device uses (bench_kernels.cu). A model,
 * not the device code: the device half of this gate (--verify) runs the real
 * ss_first. This half is what puts the host proofs into `make check`. */
static uint32_t ss_first_model(uint32_t p, uint32_t rt, uint32_t j, int32_t ilo,
                               uint32_t magic, uint32_t kshift, int two)
{
    uint32_t sh = 0;
    while ((1u << (sh + 1)) <= p) sh++;
    if ((p & (p - 1)) == 0)
        return (rt * j + (p << kshift) - (uint32_t)ilo) & (p - 1);
    if (two) {
        const uint32_t r = td_mod_magic(rt * j, p, magic, sh);
        return td_mod_magic(r + (p << kshift) - (uint32_t)ilo, p, magic, sh);
    }
    return td_mod_magic(rt * j + (p << kshift) - (uint32_t)ilo, p, magic, sh);
}

/* The thread tier's two-step tail (finding 111): the split search against a
 * linear scan, the routing and both proofs, and the arithmetic at the
 * proof's corners against the 64-bit fallback's formula, at the geometries
 * where a tail exists and one where it does not. */
static int check_ss_tail(void)
{
    static const uint32_t geo[][2] = {     /* {logI, J} */
        {15, 16384}, {16, 16384}, {16, 32768}, {16, 65536}, {17, 32768},
        {17, 65536}, {18, 65536}
    };
    uint32_t seed = 0x5eed2b0bu, ntwo = 0, nall = 0;
    for (unsigned g = 0; g < sizeof geo / sizeof geo[0]; g++) {
        const uint32_t logI = geo[g][0], J = geo[g][1], I = 1u << logI;
        const uint32_t ihalf = I >> 1, kshift = SS_KSHIFT(logI);
        static uint32_t hsp[1u << 18];
        uint32_t n = 0, split, lin = 0;
        /* A sorted table of every modulus from 1024 up: the split search
         * must equal the first entry failing the one-step bound. */
        for (uint32_t m = 1024; m < I; m++) hsp[n++] = m;
        split = ss_two_step_from(hsp, n, 0, J, logI);
        while (lin < n && ss_one_step_fits(hsp[lin], J, logI)) lin++;
        if (split != lin) {
            fprintf(stderr, "slabtest: ss_two_step_from %u != scan %u at logI %u"
                    " J %u\n", split, lin, logI, J);
            return -1;
        }
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t m = hsp[i];
            const int two = i >= split;
            uint32_t magic;
            /* Every entry before the split is proved one-step, and a
             * one-step proof implies the two-step one (jmax >= 1). */
            if (!two && (!ss_bound_ok(m, J, logI, 0) || !ss_bound_ok(m, J, logI, 1))) {
                fprintf(stderr, "slabtest: m=%u before the split is not proved"
                        " both ways at logI %u J %u\n", m, logI, J);
                return -1;
            }
            ss_magic_build(m, J, logI, two, &magic);
            if (!magic) continue;              /* the fallback */
            /* The documented worst-case numerators, restated here rather
             * than taken from ss_bound_ok: every modulus the host accepts
             * must keep each dividend its form reduces below 2^31. A wrong
             * residue is not guaranteed past that bound, so the arithmetic
             * below can pass an over-generous proof by luck; this cannot.
             * Powers of two take the AND and need no bound. */
            if (m & (m - 1)) {
                const uint64_t lim = 1ull << 31, bias = (uint64_t)m << kshift;
                const int ok = two
                    ? (uint64_t)(m - 1) * J < lim && (m - 1) + bias + ihalf < lim
                    : (uint64_t)(m - 1) * J + bias + ihalf < lim;
                if (!ok) {
                    fprintf(stderr, "slabtest: %s-step magic accepted for m=%u at"
                            " logI %u J %u, past the 2^31 bound\n",
                            two ? "two" : "one", m, logI, J);
                    return -1;
                }
            }
            /* Corners every 97th modulus, plus the ends and both sides of
             * the split; the arithmetic is the same for the moduli between. */
            if (i % 97 && i + 3 < n && (i + 3 < split || i > split + 3)) continue;
            for (unsigned k = 0; k < 8; k++) {
                uint32_t rt, j;
                int32_t ilo;
                seed = seed * 1664525u + 1013904223u;
                rt  = k < 4 ? m - 1u : seed % m;
                j   = (k & 1) ? J : (k < 4 ? 1u : seed % (J + 1u));
                ilo = (k & 2) ? -(int32_t)ihalf : (int32_t)ihalf - 1;
                const uint32_t ref = (uint32_t)(((uint64_t)rt * j +
                    ((uint64_t)m << 20) - (uint64_t)(int64_t)ilo) % m);
                const uint32_t got = ss_first_model(m, rt, j, ilo, magic, kshift, two);
                if (got != ref) {
                    fprintf(stderr, "slabtest: ss_first %s-step m=%u rt=%u j=%u"
                            " ilo=%d logI %u J %u: got %u ref %u\n",
                            two ? "two" : "one", m, rt, j, ilo, logI, J, got, ref);
                    return -1;
                }
                nall++;
                ntwo += two;
            }
        }
    }
    if (ntwo < 1000) {
        fprintf(stderr, "slabtest: only %u two-step cases; the geometry table no"
                " longer reaches the tail\n", ntwo);
        return -1;
    }
    return 0;
}

/* --row-cap's layout (bklay_t, finding 132) against a direct model: each row
 * class's share of a row's positions counted with fill's own --sieve-skip
 * predicate, and every region's slots laid end to end in order. A layout
 * that overlapped two regions would mix their records silently; one that
 * left a gap would only waste memory, so both directions are checked. */
static uint32_t bklay_kept(uint32_t j, int logI, int skip)
{
    const uint32_t I = 1u << logI, ih3 = (I >> 1) % 3u;
    uint32_t n = 0;
    for (uint32_t x = 0; x < I; x++) {
        if (skip && ((x | j) & 1u) == 0u) continue;
        if (skip > 1 && j % 3u == 0u && x % 3u == ih3) continue;
        n++;
    }
    return n;
}

static int check_bklay(void)
{
    static const uint32_t caps[] = {257, 300, 3001, 10658, 11315, 11575};
    static const int geom[][2] = {{16, 14}, {15, 14}, {14, 14}, {13, 13},
                                  {16, 12}, {12, 12}};
    unsigned ncase = 0;
    for (int skip = 1; skip <= 2; skip++) {
        /* the table is the predicate's share, to within a position */
        for (uint32_t j = 0; j < 12; j++) {
            const bklay_t L = bklay_make(11575, 1, skip, 16, 14);
            const uint64_t kept6 = (uint64_t)bklay_kept(j, 16, skip) * 6u;
            const uint64_t want6 = (uint64_t)bklay_f(L, j) << 16;
            if (kept6 + 12 < want6 || kept6 > want6 + 12) {
                fprintf(stderr, "slabtest: bklay skip %d row %u keeps %llu/6 of"
                        " 65536 positions, the table says %u sixths\n", skip, j,
                        (unsigned long long)kept6, bklay_f(L, j));
                return -1;
            }
        }
        for (const uint32_t cap : caps)
            for (const auto &g : geom) {
                const int logI = g[0], lr = g[1];
                const uint32_t per_row = 1u << (logI - lr);
                const bklay_t L = bklay_make(cap, 1, skip, logI, lr);
                if (!L.s || L.m != BKLAY_MARGIN || 6u * L.s + L.m < cap ||
                    6u * L.s + L.m >= cap + 6u) {
                    fprintf(stderr, "slabtest: bklay cap %u skip %d: s %u m %u"
                            " does not give the f = 6 class the capacity\n",
                            cap, skip, L.s, L.m);
                    return -1;
                }
                static const uint32_t j0s[] = {0, 1, 2, 3, 4, 5, 7, 4096, 8191, 32767};
                for (const uint32_t j0 : j0s) {
                    const uint32_t nrow = 13;
                    const bklay_t A = bklay_at(L, j0);
                    uint64_t at = 0;
                    for (uint32_t b = 0; b < nrow * per_row; b++) {
                        const uint32_t j = j0 + b / per_row;
                        const uint32_t f = (j & 1u) ? ((skip > 1 && j % 3u == 0u) ? 4u : 6u)
                                                    : ((skip > 1 && j % 3u == 0u) ? 2u : 3u);
                        uint32_t c = 0;
                        const uint64_t base = bklay_base(A, b, &c);
                        if (base != at || c != L.s * f + L.m) {
                            fprintf(stderr, "slabtest: bklay cap %u skip %d logI %d"
                                    " region %d j0 %u: region %u at %llu cap %u,"
                                    " model %llu cap %u\n", cap, skip, logI, lr, j0,
                                    b, (unsigned long long)base, c,
                                    (unsigned long long)at, L.s * f + L.m);
                            return -1;
                        }
                        at += c;
                        if (bklay_slots(L, b + 1, j0) != at ||
                            bklay_slots_max(L, b + 1) < at) {
                            fprintf(stderr, "slabtest: bklay slots for %u regions"
                                    " from row %u\n", b + 1, j0);
                            return -1;
                        }
                        ncase++;
                    }
                }
            }
    }
    /* uniform wherever the class layout does not apply, and b*m there */
    const bklay_t U[] = {bklay_make(11575, 0, 2, 16, 14),
                         bklay_make(11575, 1, 0, 16, 14),
                         bklay_make(11575, 1, 3, 16, 14),
                         bklay_make(256, 1, 2, 16, 14),
                         bklay_make(11575, 1, 2, 13, 14),
                         bklay_uniform(11575)};
    for (const bklay_t &u : U) {
        const bklay_t A = bklay_at(u, 12345);
        for (uint32_t b = 0; b < 4096; b++) {
            uint32_t c = 0;
            if (u.s || bklay_base(A, b, &c) != (uint64_t)b * u.m || c != u.m) {
                fprintf(stderr, "slabtest: bklay uniform layout region %u\n", b);
                return -1;
            }
        }
    }
    /* The 32-bit sums against a 64-bit model at the largest array bklay_fits
     * admits, every region, in both layouts (the uniform one to 2^32 - 1
     * slots); one region more is refused, and so is the 2^19-region case that
     * wrapped at ~371,000 regions when bklay_make fell back to uniform. */
    const bklay_t Z[] = {bklay_make(11575, 1, 2, 16, 14), bklay_uniform(65535)};
    for (const bklay_t &L : Z) {
        const uint64_t per = 6u * L.s + L.m;
        const uint32_t nmax = (uint32_t)((((uint64_t)1 << 32) - 1) / per);
        if (!bklay_fits(L, nmax) || bklay_fits(L, nmax + 1) ||
            bklay_fits(L, 1u << 19) || bklay_fits(bklay_uniform(11575), 1u << 19)) {
            fprintf(stderr, "slabtest: bklay_fits admits the wrong arrays"
                    " (s %u m %u, %u regions)\n", L.s, L.m, nmax);
            return -1;
        }
        static const uint32_t j0s[] = {0, 1, 5};
        for (const uint32_t j0 : j0s) {
            const bklay_t A = bklay_at(L, j0);
            uint64_t at = 0;
            for (uint32_t b = 0; b < nmax; b++) {
                const uint32_t j = j0 + (b >> L.rshift);
                const uint32_t f = !L.s ? 0u
                                 : (j & 1u) ? (j % 3u == 0u ? 4u : 6u)
                                            : (j % 3u == 0u ? 2u : 3u);
                uint32_t c = 0;
                if (bklay_base(A, b, &c) != at || c != L.s * f + L.m) {
                    fprintf(stderr, "slabtest: bklay at %u of %u regions from row"
                            " %u: %u, model %llu\n", b, nmax, j0,
                            bklay_base(A, b, &c), (unsigned long long)at);
                    return -1;
                }
                at += c;
                ncase++;
            }
            if (at >= ((uint64_t)1 << 32) || bklay_slots(L, nmax, j0) != at) {
                fprintf(stderr, "slabtest: bklay slots for %u regions from row"
                        " %u\n", nmax, j0);
                return -1;
            }
        }
    }
    if (ncase < 10000) {
        fprintf(stderr, "slabtest: bklay checked only %u regions\n", ncase);
        return -1;
    }
    return 0;
}

int main(void)
{
    if (check_plan() || check_bkthresh_integration() || check_phase() ||
        check_magic() || check_ss_tail() || check_walk_continuation() ||
        check_bklay()) return 1;
    printf("slabtest: plan, bkthresh wiring, phase, reciprocal, small-sieve tail,"
           " continued-walk and row-class bucket layout gates OK\n");
    return 0;
}
