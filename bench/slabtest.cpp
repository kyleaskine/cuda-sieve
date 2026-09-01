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

int main(void)
{
    if (check_plan() || check_bkthresh_integration() || check_phase() ||
        check_magic() || check_walk_continuation()) return 1;
    printf("slabtest: plan, bkthresh wiring, phase, reciprocal, and continued-walk gates OK\n");
    return 0;
}
