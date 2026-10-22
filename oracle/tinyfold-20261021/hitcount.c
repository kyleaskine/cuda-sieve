/* Small-sieve hit census by EFFECTIVE modulus, host only (no GPU).
 *
 * For a sample of real special-q, builds each side's line-sieved table the way
 * pipe_side_prepare_q does (fb_split_small, pl_transform_fast -> (m, rt, g)),
 * then counts what sieve_small would do over the whole q area under ss_setup's
 * rules: an entry hits row j only when g | j, first hit c0 = (rt*(j/g) - ilo)
 * mod m in each region, and under --sieve-skip an even row keeps odd i only
 * (odd m: step 2m from the first odd hit; even m: all hits or none).
 *
 * Reported per side, per position of the q area (J * I, both-even included):
 * hits by tier, and for the block tier (m < 64) the share of hits, of
 * warp-iterations (the hit loop's trip count per warp, summed over warps, at
 * --apply-threads 512) and of entry visits that fall in a candidate table set:
 *   T420: effective m in {1, 2, 3, 4, 5, 7}   (period 420 cells = 210 words)
 *   T840: T420 plus m == 8                    (period 840 cells = 420 words)
 * Also each side's nslice_pow2 for apply's dynamic shared memory, computed by
 * fb_build_slices on the bucketed factor base at the job's scale.
 *
 * Usage: hitcount --poly JOB [--fb1 FILE] --logI N [--J N] [--region N]
 *        [--maxbits N] [--sq-side S] --qmin Q [--nq N] --scale1 X --scale0 Y
 *        [--threads N] [--no-skip]
 */
#include "bench.h"
#include "plattice.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCLS 3                      /* block, warp, thread tier */
#define MHIST 64

typedef struct {
    double hits[NCLS];              /* summed over q */
    double ent[NCLS];               /* table entries, summed over q */
    double blk_hits_m[MHIST];       /* block-tier hits by effective m */
    double blk_iters, blk_iters_420, blk_iters_840;
    double blk_visits, blk_visits_420, blk_visits_840;  /* (entry, region) with hits */
    double ent_420, ent_840;
} tally_t;

static int in420(uint32_t m) { return m == 1 || m == 2 || m == 3 || m == 4 || m == 5 || m == 7; }

static void count_side(const fb_t *fbs, const qlat_t *L, int logI, uint32_t J,
                       int log_region, uint32_t nth, int skip, tally_t *T)
{
    const uint32_t I = 1u << logI, width = 1u << log_region;
    const uint32_t nreg = I >> log_region, nwarp = nth >> 5;
    for (uint32_t e = 0; e < fbs->n; e++) {
        uint32_t rt, g;
        const uint32_t m = pl_transform_fast(fbs->primes[e], fbs->roots[e],
                                             L->a0, L->a1, L->b0, L->b1, &rt, &g);
        const int cls = m < 64 ? 0 : (m < 1024 ? 1 : 2);
        const int s420 = cls == 0 && in420(m), s840 = cls == 0 && (s420 || m == 8);
        T->ent[cls] += 1;
        if (s420) T->ent_420 += 1;
        if (s840) T->ent_840 += 1;
        double h = 0, it = 0, vis = 0;
        for (uint32_t j = 0; j < J; j++) {
            if (g > 1 && (j % g)) continue;
            const uint64_t jj = g > 1 ? j / g : j;
            const int evenrow = skip && !(j & 1u);
            for (uint32_t k = 0; k < nreg; k++) {
                const int64_t ilo = (int64_t)(k * width) - (int64_t)(I >> 1);
                int64_t c = ((int64_t)((uint64_t)rt * jj % m) - ilo) % (int64_t)m;
                if (c < 0) c += m;
                uint64_t c0 = (uint64_t)c, mm = m;
                if (evenrow) {
                    if (m & 1u) { if (!(c0 & 1u)) c0 += m; mm = 2ull * m; }
                    else if (!(c0 & 1u)) continue;
                }
                if (c0 >= width) continue;
                const double nh = (double)((width - 1 - c0) / mm + 1);
                h += nh;
                if (cls == 0) {
                    vis += 1;
                    for (uint32_t w = 0; w < nwarp; w++) {
                        const uint64_t s = c0 + (uint64_t)(32u * w) * mm;
                        if (s >= width) break;
                        it += (double)((width - 1 - s) / ((uint64_t)nth * mm) + 1);
                    }
                }
            }
        }
        T->hits[cls] += h;
        if (cls == 0) {
            T->blk_hits_m[m] += h;
            T->blk_iters += it; T->blk_visits += vis;
            if (s420) { T->blk_iters_420 += it; T->blk_visits_420 += vis; }
            if (s840) { T->blk_iters_840 += it; T->blk_visits_840 += vis; }
        }
    }
}

static void report(const char *name, const tally_t *T, double npos, int nq)
{
    const double all = T->hits[0] + T->hits[1] + T->hits[2];
    double h420 = 0, h840 = 0;
    for (int m = 1; m < MHIST; m++) {
        if (in420((uint32_t)m)) h420 += T->blk_hits_m[m];
        if (in420((uint32_t)m) || m == 8) h840 += T->blk_hits_m[m];
    }
    printf("  %s: line-sieved hits/position %.3f  (block %.3f, warp %.3f, thread %.3f)\n",
           name, all / npos, T->hits[0] / npos, T->hits[1] / npos, T->hits[2] / npos);
    printf("  %s: entries/q  block %.1f, warp %.1f, thread %.1f;  in T420 %.1f, T840 %.1f\n",
           name, T->ent[0] / nq, T->ent[1] / nq, T->ent[2] / nq, T->ent_420 / nq, T->ent_840 / nq);
    printf("  %s: T420 share  of all hits %.1f%%, of block hits %.1f%%, of block warp-iterations %.1f%%, of block entry visits %.1f%%\n",
           name, 100 * h420 / all, 100 * h420 / T->hits[0],
           100 * T->blk_iters_420 / T->blk_iters, 100 * T->blk_visits_420 / T->blk_visits);
    printf("  %s: T840 share  of all hits %.1f%%, of block hits %.1f%%, of block warp-iterations %.1f%%, of block entry visits %.1f%%\n",
           name, 100 * h840 / all, 100 * h840 / T->hits[0],
           100 * T->blk_iters_840 / T->blk_iters, 100 * T->blk_visits_840 / T->blk_visits);
    printf("  %s: block warp-iterations per position %.4f, block entry visits per 2^14 positions %.1f\n",
           name, T->blk_iters / npos, T->blk_visits / (npos / 16384.0));
    printf("  %s: block hits/position by effective m:", name);
    for (int m = 1; m < MHIST; m++)
        if (T->blk_hits_m[m] > 0) printf(" %d:%.3f", m, T->blk_hits_m[m] / npos);
    printf("\n");
}

static int32_t last_slices;          /* the count before padding, for --slices-only */
static uint32_t slices_pow2(fb_t *fb)
{
    uint16_t *s = NULL, *t = NULL; uint32_t p2 = 0;
    if ((last_slices = fb_build_slices(fb, &s, &t, &p2)) < 0) { perror("fb_build_slices"); exit(1); }
    free(s); free(t);
    return p2;
}

int main(int argc, char **argv)
{
    const char *polypath = NULL, *fb1path = NULL;
    int logI = 0, log_region = 14, maxbits = 0, sq_side = 1, nq = 10, skip = 1;
    uint32_t J = 0, nth = 512;
    uint64_t qmin = 0;
    double scale1 = 0, scale0 = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--poly") && i + 1 < argc) polypath = argv[++i];
        else if (!strcmp(argv[i], "--fb1") && i + 1 < argc) fb1path = argv[++i];
        else if (!strcmp(argv[i], "--logI") && i + 1 < argc) logI = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--J") && i + 1 < argc) J = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--region") && i + 1 < argc) log_region = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) maxbits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sq-side") && i + 1 < argc) sq_side = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--qmin") && i + 1 < argc) qmin = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--nq") && i + 1 < argc) nq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--scale1") && i + 1 < argc) scale1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--scale0") && i + 1 < argc) scale0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nth = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--no-skip")) skip = 0;
        else if (!strcmp(argv[i], "--slices-only")) nq = 0;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!polypath || !fb1path || !logI || !qmin || scale1 <= 0 || scale0 <= 0) {
        fprintf(stderr, "need --poly, --fb1, --logI, --qmin, --scale1, --scale0\n");
        return 2;
    }
    if (!J) J = 1u << (logI - 1);
    if (!maxbits) maxbits = logI;            /* bench_main's default */
    if (log_region > logI) log_region = logI;
    const uint32_t bkthresh = 1u << logI;    /* bench_main's default */

    poly_t P; fb_t fb1, fb0, fbs1, fbs0;
    if (poly_load(polypath, &P) != 0) return 1;
    if (fb_load_cado(fb1path, scale1, &fb1) != 0) return 1;
    if (fb_split_small(&fb1, bkthresh, &fbs1) != 0) return 1;
    if (fb_restrict(&fb1, bkthresh, P.alim) != 0) return 1;
    if (rfb_build(&P, P.rlim, maxbits, scale0, &fb0) != 0) return 1;
    if (fb_fill_logp(&fb0, scale0) != 0) return 1;
    if (fb_split_small(&fb0, bkthresh, &fbs0) != 0) return 1;
    if (fb_restrict(&fb0, bkthresh, P.rlim) != 0) return 1;
    const uint32_t ns1 = slices_pow2(&fb1); const int32_t c1 = last_slices;
    const uint32_t ns0 = slices_pow2(&fb0); const int32_t c0 = last_slices;
    printf("  bucketed entries: side 1 %u, side 0 %u; slices counted: side 1 %d, side 0 %d\n",
           fb1.n, fb0.n, c1, c0);
    const uint32_t nsl = ns1 > ns0 ? ns1 : ns0;
    printf("job %s  logI %d J %u region 2^%d threads %u sieve-skip %s maxbits %d\n",
           polypath, logI, J, log_region, nth, skip ? "on" : "off", maxbits);
    printf("  line-sieved entries: side 1 %u, side 0 %u (bkthresh %u)\n", fbs1.n, fbs0.n, bkthresh);
    printf("  nslice_pow2: side 1 %u, side 0 %u -> apply smem %u B per block\n",
           ns1, ns0, (unsigned)((1u << log_region) * 2u + nsl * 2u));

    if (!nq) return 0;
    sqgen_t *G = sqgen_create(&P, sq_side, qmin, 0, (uint32_t)nq);
    if (!G) return 1;
    tally_t T1, T0; memset(&T1, 0, sizeof T1); memset(&T0, 0, sizeof T0);
    qsel_t s; int got = 0;
    while (got < nq && sqgen_next(G, &s) == 1) {
        qlat_t L;
        qlat_build(&L, s.q, s.rho, P.skew);
        count_side(&fbs1, &L, logI, J, log_region, nth, skip, &T1);
        count_side(&fbs0, &L, logI, J, log_region, nth, skip, &T0);
        got++;
    }
    sqgen_free(G);
    const double npos = (double)got * J * (double)(1u << logI);
    printf("  %d special-q from q >= %llu\n", got, (unsigned long long)qmin);
    report("side 1", &T1, npos, got);
    report("side 0", &T0, npos, got);
    return 0;
}
