/* normscan -- how wide must bn_t be for this job, this geometry, this band?
 *
 * WHY THIS EXISTS, AND WHY IT IS NOT A STARTUP CHECK IN bench. The exact norm
 * that trial division builds must fit BN_LIMBS*32 bits. pipe_side_prepare_q
 * checks that per q and SKIPS the ones that do not fit, warning each time and
 * ending the run at PIPE_SKIP_MAX of them. That keeps a band alive across the
 * rare bad lattice, but it is damage control, not an answer: every skip is a
 * lost special-q, and a job that needs more width than the build has loses them
 * steadily until the run stops.
 *
 * The client cannot fix it either way, because the fix is a wider REBUILD, and
 * its work unit is a few hundred q out of tens of millions -- far too few to
 * contain a ~1e-5 tail, so it cannot even see the problem coming. The width has
 * to be decided once, centrally, before work units are distributed. That is
 * planning time, i.e. testsieve.
 *
 * The 2,1139+ job named throughout is a DEGREE 8 SNFS form, and an earlier
 * version of these comments called it a septic in three places. It is an octic
 * by construction: 1139 = 17 * 67, so the substitution x = 2^67 + 2^-67 leaves
 * the minimal polynomial of a 17th root of unity plus its inverse, of degree
 * (17-1)/2 = 8. In the job file that is c8..c0 = 1 1 -7 -6 15 10 -10 -4 1 with
 * Y1 = 2^67 and Y0 = -(2^134 + 1); F(Y0, Y1) == 0 mod n, checked 2026-09-01.
 *
 * WHY THE SAMPLE MAXIMUM IS NOT THE ANSWER. Measured on the 2,1139+ octic over
 * 60M-460M at logI 15: 2,500 (q,rho) sampled at scattered q gave a maximum of
 * 242 bits and the confident, wrong conclusion that 256 was enough. 160,018
 * sampled across the band found q=367699421 at 273.08 bits -- and its
 * neighbours in the exceedance list sat at 258.9 and 258.4, so even the
 * exceedances are widely spread. A sample of n out of N sees the 1/n quantile,
 * not the 1/N one, and the gap between them is the whole question.
 *
 * So this reports a PROJECTED BAND MAXIMUM from an exponential fit to the upper
 * tail, not the sample maximum, and warns on proximity to the limit rather than
 * only on crossing it. See normscan_project().
 *
 * CPU only, and deliberately built from the same objects the siever uses --
 * sqgen_next for (q,rho), qlat_build for the reduction, norm_setup and
 * norm_exact_bound_bits for the bound. A reimplementation of the reduction in
 * awk or python would answer a subtly different question every time one of them
 * changed. */
#include "bench.h"
/* for BN_LIMBS: the whole point is to judge against THIS build's width.
 * bigint.cuh is plain C outside nvcc -- BN_FN degrades to static inline. */
#include "bigint.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Exit codes. 2 and 3 are VERDICTS about the band and 1 means the survey could
 * not run, so a usage error must not reuse any of them: a caller that maps 2 to
 * "will overflow" would otherwise report a mistyped flag as an exact-norm
 * overflow, with no explanation attached. */
#define EX_USAGE_ 64

static int cmp_dbl(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Primes below x, Riemann-free: x/(ln x - 1) is within ~0.5% over the range a
 * special-q band ever covers, and the band size only ever scales a rate into an
 * expected count, so a percent does not change a decision. */
static double prime_count(double x)
{
    return (x < 3.0) ? 0.0 : x / (log(x) - 1.0);
}

/* Projected maximum over N draws, from n sampled ones.
 *
 * Exponential (Gumbel-domain) fit to the upper tail: above a high threshold u,
 * P(X > u+y) ~ (m/n) exp(-y/beta) with beta the mean excess over u. Setting
 * that to 1/N and solving gives the level exceeded about once in N draws, which
 * is what the band's worst special-q will look like.
 *
 * Returns 0 and leaves *out alone if the tail is too thin to fit (m < 8): with
 * fewer exceedances than that beta is dominated by one point, and a confident
 * projection from it would be worse than admitting ignorance. */
static int normscan_project(const double *sorted, int n, double N, double *out,
                            double *u_out, int *m_out, double *beta_out)
{
    int i, m;
    double u, beta = 0.0;
    if (n < 200) return 0;
    i = (int)(n * 0.99);                 /* threshold at the 99th percentile */
    u = sorted[i];
    m = n - i;
    if (m < 8) return 0;
    for (int k = i; k < n; k++) beta += sorted[k] - u;
    beta /= m;
    if (!(beta > 0.0)) return 0;
    *out = u + beta * log(N * (double)m / (double)n);
    *u_out = u; *m_out = m; *beta_out = beta;
    return 1;
}

static void usage(void)
{
    fprintf(stderr,
        "normscan -- can this build's exact-norm width sieve this band?\n"
        "\n"
        "Trial division builds the exact homogeneous norm and it must fit\n"
        "BN_LIMBS*32 bits. Whether it does depends on the polynomial, the\n"
        "geometry AND the q-lattice, so the answer moves with logI and J and\n"
        "has a tail: the special-q that overflow are a ~1e-5 minority spread\n"
        "across the whole band. Sieving a few hundred q proves nothing, and a\n"
        "client in a group sieve never sees enough of the band to tell. Run\n"
        "this once, per geometry, before distributing work -- the fix is a\n"
        "rebuild, and every client has to carry it.\n"
        "\n"
        "Reports a PROJECTED BAND MAXIMUM from a fit to the upper tail, not the\n"
        "sample maximum, and warns on thin margin as well as on crossing.\n"
        "Exit: 0 ok, 2 will overflow, 3 too little margin to trust,\n"
        "      1 the survey could not run, 64 bad usage. A caller must treat\n"
        "      anything but 0 as 'not surveyed' -- an error is not a pass.\n"
        "\n"
        "usage: normscan --poly JOB --qmin Q --qmax Q [options]\n"
        "  --sq-side S     1 = algebraic (default), 0 = rational\n"
        "  --logI N        sieve width exponent           [15]\n"
        "  --J N           sieve height                   [2^(logI-1)]\n"
        "  --samples N     (q,rho) pairs to draw          [100000]\n"
        "  --windows W     evenly spaced points in the band [40]\n"
        "  --limit-bits B  width to judge against         [the build's]\n"
        "  --dump FILE     write every sampled bound, one per line\n");
}

int main(int argc, char **argv)
{
    const char *poly = NULL, *dump = NULL;
    int side = 1, logI = 15, windows = 40, limit = 0;
    long J = 0, want = 100000;
    uint64_t qmin = 0, qmax = 0;
    poly_t P, P0;
    double *v = NULL;
    long n = 0;
    double nprime_tried = 0, nroot = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *nx = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--poly") && nx) { poly = nx; i++; }
        else if (!strcmp(a, "--dump") && nx) { dump = nx; i++; }
        else if (!strcmp(a, "--sq-side") && nx) { side = atoi(nx); i++; }
        else if (!strcmp(a, "--logI") && nx) { logI = atoi(nx); i++; }
        else if (!strcmp(a, "--J") && nx) { J = atol(nx); i++; }
        else if (!strcmp(a, "--samples") && nx) { want = atol(nx); i++; }
        else if (!strcmp(a, "--windows") && nx) { windows = atoi(nx); i++; }
        else if (!strcmp(a, "--limit-bits") && nx) { limit = atoi(nx); i++; }
        else if (!strcmp(a, "--qmin") && nx) { qmin = strtoull(nx, NULL, 10); i++; }
        else if (!strcmp(a, "--qmax") && nx) { qmax = strtoull(nx, NULL, 10); i++; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a); usage(); return EX_USAGE_; }
    }
    if (!poly || !qmin || !qmax || qmax <= qmin) { usage(); return EX_USAGE_; }
    if (logI < 2 || logI > 20) { fprintf(stderr, "logI out of range\n"); return EX_USAGE_; }
    if (side != 0 && side != 1) { fprintf(stderr, "sq-side must be 0 or 1\n"); return EX_USAGE_; }
    if (windows < 1) windows = 1;
    if (want < 1) { fprintf(stderr, "--samples must be positive\n"); return EX_USAGE_; }
    if (!J) J = 1L << (logI - 1);
    if (!limit) limit = BN_LIMBS * 32;
    if (poly_load(poly, &P)) return 1;
    /* norm_setup's per-q diagnostic is for a human watching one special-q.
     * This calls it a million times. */
    norm_verbose = 0;

    /* Side 0's norm is the degree-1 form G = Y1*x + Y0, exactly as
     * pipe_side_prepare_q builds it. Both sides are measured: the width has to
     * hold whichever is larger, and which one that is depends on the job. */
    P0 = P;
    P0.deg = 1; P0.c[0] = P0.y0; P0.c[1] = P0.y1;
    for (int z = 2; z < BENCH_NCOEFF; z++) P0.c[z] = 0.0;

    v = (double *)malloc((size_t)want * sizeof(*v));
    if (!v) { fprintf(stderr, "out of memory for %ld samples\n", want); return 1; }

    for (int w = 0; w < windows && n < want; w++) {
        const long target = want * (w + 1) / windows;
        const uint64_t q0 = qmin + (uint64_t)((qmax - qmin) *
                                              ((double)w / (double)windows));
        sqgen_t *G = sqgen_create(&P, side, q0, qmax, 0);
        /* lastq, not s.q: a window whose loop never runs (target already met)
         * would otherwise read an uninitialised s, and the roots-per-prime
         * ratio it feeds decides the band size. */
        uint64_t lastq = q0;
        qsel_t s;
        if (!G) { fprintf(stderr, "sqgen_create failed at q=%llu\n",
                          (unsigned long long)q0); free(v); return 1; }
        while (n < target) {
            const int r = sqgen_next(G, &s);
            if (r <= 0) break;                  /* window ran to qmax */
            lastq = s.q;
            {
                qlat_t L; norm_t N1, N0; double b1, b0;
                qlat_build(&L, s.q, s.rho, P.skew);
                norm_setup(&N1, &P,  &L, logI, (uint32_t)J, 1.0, side == 1);
                norm_setup(&N0, &P0, &L, logI, (uint32_t)J, 1.0, side == 0);
                b1 = norm_exact_bound_bits(&N1);
                b0 = norm_exact_bound_bits(&N0);
                v[n++] = (b1 > b0) ? b1 : b0;
                nroot++;
            }
        }
        nprime_tried += prime_count((double)lastq) - prime_count((double)q0);
        sqgen_free(G);
    }
    if (n < 2) { fprintf(stderr, "no special-q generated in [%llu,%llu]\n",
                         (unsigned long long)qmin, (unsigned long long)qmax);
                 free(v); return 1; }

    if (dump) {
        FILE *f = fopen(dump, "w");
        if (!f) { perror(dump); free(v); return 1; }
        for (long i = 0; i < n; i++) fprintf(f, "%.4f\n", v[i]);
        fclose(f);
    }
    qsort(v, (size_t)n, sizeof(*v), cmp_dbl);

    {
        /* Band size in (q,rho), not in q: a work item is a pair, and a degree-d
         * sq side supplies on average one root per prime but up to d at some. */
        const double rpp = nprime_tried > 0 ? nroot / nprime_tried : 1.0;
        const double N = (prime_count((double)qmax) -
                          prime_count((double)qmin)) * rpp;
        double proj = 0, u = 0, beta = 0; int m = 0;
        long over = 0;
        for (long i = n - 1; i >= 0 && v[i] > (double)limit; i--) over++;

        printf("  band %llu..%llu, sq side %d, logI %d, J %ld\n",
               (unsigned long long)qmin, (unsigned long long)qmax, side, logI, (long)J);
        printf("  sampled %ld (q,rho) of ~%.3g in the band (%.2f roots/prime)\n",
               n, N, rpp);
        printf("  exact norm bits: median %.1f  99%% %.1f  99.9%% %.1f  sample max %.2f\n",
               v[n / 2], v[(long)(n * 0.99)], v[(long)(n * 0.999)], v[n - 1]);
        if (over)
            printf("  %ld of %ld sampled exceed %d bits  ->  ~%.0f of the band\n",
                   over, n, limit, (double)over / (double)n * N);
        else
            printf("  0 of %ld sampled exceed %d bits (an unseen rate up to %.1e,"
                   " i.e. ~%.0f of the band, is still consistent with that)\n",
                   n, limit, 3.0 / (double)n, 3.0 / (double)n * N);
        if (normscan_project(v, (int)n, N, &proj, &u, &m, &beta))
            printf("  projected band maximum %.1f bits"
                   " (exponential tail, scale %.2f bits, %d points above %.1f)\n",
                   proj, beta, m, u);
        else
            printf("  projected band maximum: tail too thin to fit from %ld samples\n", n);

        /* VERDICT. Judged on the PROJECTION, not the sample max, and with a
         * margin that SCALES WITH THE FITTED TAIL rather than being a constant.
         *
         * beta is the natural unit of surprise here: on an exponential tail each
         * further beta of headroom is one e-fold rarer, so the same number of
         * bits means completely different things on different jobs. Measured:
         * AS276's quintic has beta = 0.15 bits (its top 0.1% spans 0.08 bits),
         * where 7 bits of margin is ~e^45 safe; the 2,1139+ octic has beta = 5.9,
         * where 7 bits is barely one e-fold. A fixed 16-bit margin -- the first
         * thing this code did -- warned on AS276, which is wrong, and would have
         * been thin comfort on the octic.
         *
         * Four scales plus a floor. The floor covers a degenerate fit where beta
         * collapses toward zero on a near-constant tail; four is judgement, not
         * measurement, chosen so that octic against 384 passes comfortably
         * while AS276 against 256 (7.3 clear, beta 0.15) also passes and the
         * same octic against 256 refuses outright on the projection alone.
         *
         * The octic's own calibration numbers are NOT reproduced and are
         * deliberately not repeated here. An earlier draft called this job a
         * septic in three places, and the pair "(98 bits clear, beta 5.0)"
         * belongs to that confusion -- one polynomial cannot have the beta 5.9
         * quoted above and 5.0 here. Re-measured 2026-09-01 on the real
         * coefficients over 60M-460M: 49.5-57.5 bits clear at beta 5.96-5.99
         * across three geometries. The VERDICTS are unaffected (pass at 384,
         * refuse at 256 and 320) and the 4x scale still holds against the
         * measured beta; the parenthetical did not. See STATUS.md, Known
         * defects. Re-derive from a fresh survey before tightening this rule. */
        {
            const double margin = (beta > 0.0) ? 4.0 + 4.0 * beta : 16.0;
            const double worst = (proj > v[n - 1]) ? proj : v[n - 1];
            /* bn_limbs_for_bits, NOT a second width loop. This file exists so
             * the survey cannot drift from what pipe_side_prepare_q enforces,
             * and a hand-rolled `32*L > worst + margin` was already a different
             * rule -- it ignored the millibit norm_fits_exact applies. 0 means
             * no legal width suffices and must not be printed as one: the old
             * `need ? need : 16` advised `make BN_LIMBS=16` for a norm needing
             * more than 512 bits, i.e. a rebuild onto the same refusal. */
            const int need = bn_limbs_for_bits(worst + margin);
            printf("\n");
            if (worst >= (double)limit) {
                printf("  ** REFUSE: this band will overflow %d bits.\n", limit);
                if (need) printf("     Rebuild with `make BN_LIMBS=%d`.\n", need);
                else      printf("     No supported BN_LIMBS is wide enough"
                                 " (the maximum is 16, i.e. 512 bits).\n");
                free(v); return 2;
            }
            if (worst + margin >= (double)limit) {
                printf("  ** WARNING: %.1f bits of margin under the %d-bit limit,"
                       " against a\n"
                       "     tail scale of %.2f bits -- under %.1f e-folds of headroom, so a\n"
                       "     single unsampled q can plausibly sit above the projection.\n"
                       "     This band needs a wider build.\n",
                       (double)limit - worst, limit, beta,
                       beta > 0 ? ((double)limit - worst) / beta : 0.0);
                if (need) printf("     Rebuild with `make BN_LIMBS=%d` before"
                                 " committing the band.\n", need);
                else      printf("     No supported BN_LIMBS is wide enough"
                                 " (the maximum is 16, i.e. 512 bits).\n");
                free(v); return 3;
            }
            printf("  OK: %.1f bits of margin under the %d-bit limit.\n",
                   (double)limit - worst, limit);
        }
    }
    free(v);
    return 0;
}
