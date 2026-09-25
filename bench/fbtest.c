#define _POSIX_C_SOURCE 200809L

/* CPU-only correctness gates for the factor-base and transform layer.
 *
 * Deliberately a separate binary from `bench`: everything here runs without a
 * GPU, which means these gates can be run on a busy box, in a loop, and by
 * anyone who does not have an sm_120 card. `bench --verify` covers the
 * GPU-vs-CPU cross-check; this covers the layer underneath it, where the two
 * bugs found on 2026-08-02 lived -- both of which were invisible to every gate
 * that compared us against ourselves.
 *
 * Gates, in increasing order of what they can catch:
 *   0. slice-table capacity: all 65,536 record IDs work and the next one is
 *      rejected before allocation or output writes.
 *   1. production validation: malformed mixed composites never leave either
 *      file loader, and structural/root/power-flag errors fail closed.
 *   2. transform vs definition, set equality, synthetic moduli 2..QMAX and
 *      every root in [0,2q). Primes, prime powers, even moduli, affine and
 *      projective, zero and nonzero reciprocal.
 *   3. the same, driven by the real factor bases, plus sum(logp/q), the
 *      oracle-free density check. Placement errors fail the transform check;
 *      parse, power, log-delta and scale errors fail the density check.
 *   4. exact-norm width admission around the 256-bit boundary.
 *
 * usage: fbtest [--poly P] [--fb1 F] [--rlim N] [--maxbits N]
 *               [--q N] [--rho N] [--scale S] [--qmax N]
 */
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>

static int fail = 0;

static void ok(const char *what, int good, const char *fmt, ...)
{
    va_list ap;
    printf("%-6s %-34s ", good ? "PASS" : "FAIL", what);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    if (!good) fail = 1;
}

/* sum over the factor base of logp/q: the expected sieved log per position. */
static double log_density(const fb_t *fb)
{
    double s = 0;
    uint32_t i;
    for (i = 0; i < fb->n; i++) s += (double)fb->logp[i] / (double)fb->primes[i];
    return s;
}

static uint32_t count_proj(const fb_t *fb, uint32_t *nonzero)
{
    uint32_t i, n = 0;
    *nonzero = 0;
    for (i = 0; i < fb->n; i++)
        if (fb->roots[i] >= fb->primes[i]) {
            n++;
            if (fb->roots[i] > fb->primes[i]) (*nonzero)++;
        }
    return n;
}

/* The bucket record reserves 16 bits for a slice ID. Exercise both sides of
 * that exact boundary: all 65,536 IDs are usable, while a 65,537th slice must
 * be rejected before either output allocation is exposed to the caller. */
static int verify_slice_bounds(void)
{
    const uint32_t max_slices = (uint32_t)UINT16_MAX + 1u;
    fb_t fb;
    uint8_t *logp = NULL;
    uint16_t *slice = NULL, *tab = NULL;
    uint32_t p2 = 0, i;
    int32_t ns;
    int good = 0;

    memset(&fb, 0, sizeof(fb));
    logp = (uint8_t *)malloc((size_t)max_slices + 1u);
    if (!logp) goto done;
    for (i = 0; i <= max_slices; i++) logp[i] = (uint8_t)(i & 1u);

    fb.n = max_slices;
    fb.logp = logp;
    ns = fb_build_slices(&fb, &slice, &tab, &p2);
    if (ns != (int32_t)max_slices || p2 != max_slices ||
        !slice || !tab || slice[max_slices - 1] != UINT16_MAX)
        goto done;

    free(slice); free(tab);
    slice = NULL; tab = NULL;

    fb.n = max_slices + 1u;
    p2 = 123u;
    errno = 0;
    ns = fb_build_slices(&fb, &slice, &tab, &p2);
    good = ns == -1 && errno == EOVERFLOW &&
           slice == NULL && tab == NULL && p2 == 0;

done:
    free(slice); free(tab); free(logp);
    return good;
}


/* CLI numeric values are security-sensitive inputs, not best-effort hints.
 * Verify that the shared parser consumes one complete finite token and never
 * overwrites the caller's value on failure. */
static int verify_finite_double_parser(void)
{
    static const char *bad[] = {
        "", " ", "1.25junk", "nan", "-nan", "inf", "-inf",
        "1e9999", "1e-9999"
    };
    double v = 0.0;
    size_t i;

    if (bench_parse_finite_double("1.275", &v) != 0 ||
        fabs(v - 1.275) > 1e-15)
        return 0;
    if (bench_parse_finite_double("-2.5", &v) != 0 || v != -2.5)
        return 0;                       /* sign policy belongs to each option */
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        v = 123.0;
        errno = 0;
        if (bench_parse_finite_double(bad[i], &v) == 0 || v != 123.0)
            return 0;
    }
    return 1;
}

static int verify_u64_decimal_parser(void)
{
    static const char *bad[] = {
        "", " ", " -1", "-1", "+1", "1 ", "1x", "0x10",
        "18446744073709551616"
    };
    uint64_t v = 0;
    size_t i;

    if (bench_parse_u64_decimal("0", &v) != 0 || v != 0) return 0;
    if (bench_parse_u64_decimal("18446744073709551615", &v) != 0 ||
        v != UINT64_MAX)
        return 0;
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        v = UINT64_C(0x1122334455667788);
        errno = 0;
        if (bench_parse_u64_decimal(bad[i], &v) == 0 ||
            v != UINT64_C(0x1122334455667788))
            return 0;
    }
    return 1;
}

/* The kernel subtracts BOUND from CINIT in unsigned arithmetic. Test both the
 * intended truncating semantics and every class of value that used to reach
 * an unchecked floating-to-integer conversion or wrap that subtraction. */
static int verify_survivor_bound(void)
{
    uint32_t b = 0;

    if (sieve_bound_checked(1.275, 112.0, 4096u, &b, NULL) != 0 || b != 143u)
        return 0;
    if (sieve_bound_checked(1.0, 0.0, 4096u, &b, NULL) != 0 || b != 1u)
        return 0;
    /* The source expression used a truncating cast: 4096.999... is valid and
     * becomes 4096, but 4097.0 is not. */
    if (sieve_bound_checked(1.0, 4095.999, 4096u, &b, NULL) != 0 || b != 4096u)
        return 0;
    if (sieve_bound_checked(1.0, 254.999, 255u, &b, NULL) != 0 || b != 255u)
        return 0;

#define MUST_REJECT_BOUND(sc, al, ci)                                      \
    do {                                                                    \
        b = UINT32_C(0xdeadbeef);                                           \
        if (sieve_bound_checked((sc), (al), (ci), &b, NULL) == 0 || b != 0) \
            return 0;                                                       \
    } while (0)
    MUST_REJECT_BOUND(1.0, 4096.0, 4096u);
    MUST_REJECT_BOUND(0.0, 1.0, 4096u);
    MUST_REJECT_BOUND(-1.0, 1.0, 4096u);
    MUST_REJECT_BOUND(DBL_MIN, 1.0, 4096u);  /* underflows norm_t.scale */
    MUST_REJECT_BOUND(DBL_MAX, 1.0, 4096u);
    MUST_REJECT_BOUND(1.0, -1.0, 4096u);
    MUST_REJECT_BOUND(NAN, 1.0, 4096u);
    MUST_REJECT_BOUND(1.0, NAN, 4096u);
    MUST_REJECT_BOUND(INFINITY, 1.0, 4096u);
    MUST_REJECT_BOUND(1.0, INFINITY, 4096u);
    MUST_REJECT_BOUND(1.0, DBL_MAX, 4096u);
    MUST_REJECT_BOUND(1.0, 1.0, 0u);
#undef MUST_REJECT_BOUND
    return 1;
}

/* Factor-base logs are byte records. A failed conversion must not publish a
 * partially initialized array, and the common valid values must retain their
 * byte-for-byte rounding. */
static int verify_factor_base_log_bounds(void)
{
    uint32_t badq[] = { 2u, UINT32_MAX };
    uint32_t goodq[] = { 2u, 3u, 251u };
    fb_t fb;
    uint8_t lg = 77u;

    if (fb_log_delta_checked(2u, 1, 0, 1.0, &lg) != 0 || lg != 1u)
        return 0;
    lg = 77u;
    if (fb_log_delta_checked(UINT32_MAX, 1, 0, 8.0, &lg) == 0 || lg != 77u)
        return 0;
    lg = 77u;
    if (fb_log_delta_checked(3u, 1, 0, NAN, &lg) == 0 || lg != 77u)
        return 0;

    memset(&fb, 0, sizeof(fb));
    fb.n = 2u; fb.primes = badq;
    if (fb_fill_logp(&fb, 8.0) == 0 || fb.logp != NULL)
        return 0;

    memset(&fb, 0, sizeof(fb));
    fb.n = 3u; fb.primes = goodq;
    if (fb_fill_logp(&fb, 1.0) != 0 || !fb.logp)
        return 0;
    if (fb.logp[0] != 1u || fb.logp[1] != 2u || fb.logp[2] != 8u) {
        free(fb.logp);
        return 0;
    }
    free(fb.logp);
    return 1;
}

static int parse_positive_test_double(const char *option, const char *text,
                                      double *out)
{
    double v;
    if (bench_parse_finite_double(text, &v) != 0 || v <= 0.0) {
        fprintf(stderr, "%s wants one finite positive number, got '%s'\n",
                option, text ? text : "");
        return -1;
    }
    *out = v;
    return 0;
}

/* Exercise the validator independently of either parser. The most important
 * negative is an even mixed composite: before this gate, q = 6 reached
 * pl_invmod_any(), whose even branch computes a 2-adic inverse and is only
 * valid when q is 2^k. */
static int verify_validation_core(void)
{
    uint32_t q[] = { 2, 4, 8, 9, 17 };
    uint32_t r[] = { 0, 5, 15, 17, 33 };
    uint8_t ispow[] = { 0, 1, 1, 1, 0 };
    uint32_t qp[] = { 2, 3, 5, 7 };
    uint32_t rp[] = { 0, 1, 9, 7 };
    uint8_t ppow[] = { 0, 0, 0, 0 };
    fb_t fb, small;

    memset(&fb, 0, sizeof(fb));
    fb.n = (uint32_t)(sizeof(q) / sizeof(q[0]));
    fb.primes = q; fb.roots = r; fb.ispow = ispow;
    memset(&small, 0, sizeof(small));
    if (fb_split_small(&fb, 10, &small) == 0 ||
        fb_is_transform_validated(&small)) {
        fb_free(&small);
        return 0;
    }
    if (fb_restrict(&fb, 10, UINT32_MAX) == 0)
        return 0;
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIME_POWERS, NULL) != 0 ||
        !fb_is_transform_validated(&fb))
        return 0;
    if (fb_validate(&fb, FB_VALIDATE_GENERATED_PRIME_POWERS, NULL) != 0 ||
        !fb_is_transform_validated(&fb))
        return 0;
    if (fb_split_small(&fb, 10, &small) != 0 ||
        !fb_is_transform_validated(&small)) {
        fb_free(&small);
        return 0;
    }
    fb_free(&small);

    q[2] = 6;
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIME_POWERS, NULL) == 0 ||
        fb_is_transform_validated(&fb))
        return 0;
    q[2] = 8;

    ispow[2] = 0;
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIME_POWERS, NULL) == 0)
        return 0;
    ispow[2] = 1;

    r[3] = 18;                         /* exactly 2*q: outside the encoding */
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIME_POWERS, NULL) == 0)
        return 0;
    r[3] = 17;

    q[3] = 7;                          /* no longer non-decreasing */
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIME_POWERS, NULL) == 0)
        return 0;
    q[3] = 9;

    memset(&fb, 0, sizeof(fb));
    fb.n = (uint32_t)(sizeof(qp) / sizeof(qp[0]));
    fb.primes = qp; fb.roots = rp; fb.ispow = ppow;
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIMES, NULL) != 0 ||
        !fb_is_transform_validated(&fb))
        return 0;
    qp[2] = 4;                          /* valid power, invalid .afb.0 prime */
    rp[2] = 7;
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIMES, NULL) == 0 ||
        fb_is_transform_validated(&fb))
        return 0;

    return 1;
}

/* Force the validator's dense-factor-base fast path. This is separate from
 * the tiny malformed cases above, which intentionally use Miller-Rabin. */
static int verify_validation_sieve_path(void)
{
    const uint32_t want = 1u << 16;
    size_t nprime = 0;
    fb_t fb;
    int good = 0;

    memset(&fb, 0, sizeof(fb));
    fb.primes = prime_list_build(900000u, &nprime);
    if (!fb.primes || nprime < want) goto done;
    fb.roots = (uint32_t *)calloc(want, sizeof(*fb.roots));
    fb.ispow = (uint8_t *)calloc(want, sizeof(*fb.ispow));
    if (!fb.roots || !fb.ispow) goto done;
    fb.n = want;

    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIMES, NULL) != 0 ||
        !fb_is_transform_validated(&fb))
        goto done;

    /* Consecutive odd primes have an even composite between them. Keep the
     * stream ordered while replacing one prime with that composite. */
    fb.primes[1000] = fb.primes[999] + 1u;
    if (fb_validate(&fb, FB_VALIDATE_EXTERNAL_PRIMES, NULL) == 0 ||
        fb_is_transform_validated(&fb))
        goto done;

    good = 1;
done:
    fb_free(&fb);
    return good;
}

/* Verify the production integration, not only the helper: both file loaders
 * must reject malformed moduli before returning a factor base to their caller,
 * and failure must leave no validated or allocated object behind. */
static int verify_loader_validation(void)
{
    char afb_path[] = "/tmp/cuda-sieve-afb-XXXXXX";
    char cado_path[] = "/tmp/cuda-sieve-cado-XXXXXX";
    const uint32_t n = 3;
    const uint32_t q[3] = { 2, 4, 7 };  /* .afb.0 is base-prime only */
    const uint32_t r[3] = { 0, 1, 2 };
    const char cado_bad[] = "6: 1\n";   /* neither prime nor prime power */
    FILE *f = NULL;
    fb_t fb;
    int fd = -1, good = 1;

    fd = mkstemp(afb_path);
    if (fd < 0) return 0;
    f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(afb_path); return 0; }
    {
        int io_bad = 0;
        if (fwrite(&n, sizeof(n), 1, f) != 1 ||
            fwrite(q, sizeof(q[0]), n, f) != n ||
            fwrite(r, sizeof(r[0]), n, f) != n)
            io_bad = 1;
        if (fclose(f) != 0) io_bad = 1;
        f = NULL;
        if (io_bad) {
            unlink(afb_path);
            return 0;
        }
    }
    memset(&fb, 0, sizeof(fb));
    if (fb_load(afb_path, &fb) == 0) {
        good = 0;
        fb_free(&fb);
    } else if (fb.primes || fb.roots || fb.logp || fb.ispow || fb.n ||
               fb_is_transform_validated(&fb)) {
        good = 0;
        fb_free(&fb);
    }
    unlink(afb_path);

    fd = mkstemp(cado_path);
    if (fd < 0) return 0;
    f = fdopen(fd, "w");
    if (!f) { close(fd); unlink(cado_path); return 0; }
    {
        int io_bad = 0;
        if (fwrite(cado_bad, 1, sizeof(cado_bad) - 1, f) !=
            sizeof(cado_bad) - 1)
            io_bad = 1;
        if (fclose(f) != 0) io_bad = 1;
        f = NULL;
        if (io_bad) {
            unlink(cado_path);
            return 0;
        }
    }
    memset(&fb, 0, sizeof(fb));
    if (fb_load_cado(cado_path, 1.0, &fb) == 0) {
        good = 0;
        fb_free(&fb);
    } else if (fb.primes || fb.roots || fb.logp || fb.ispow || fb.n ||
               fb_is_transform_validated(&fb)) {
        good = 0;
        fb_free(&fb);
    }
    unlink(cado_path);
    return good;
}

static int write_text_file(char *path, const char *text)
{
    int fd = mkstemp(path);
    FILE *f;
    int bad = 0;
    if (fd < 0) return -1;
    f = fdopen(fd, "w");
    if (!f) {
        close(fd);
        unlink(path);
        return -1;
    }
    if (fwrite(text, 1, strlen(text), f) != strlen(text)) bad = 1;
    if (fclose(f)) bad = 1;
    if (bad) {
        unlink(path);
        return -1;
    }
    return 0;
}

static int cado_parse_case(const char *text, int want_ok)
{
    char path[] = "/tmp/cuda-sieve-cado-strict-XXXXXX";
    fb_t fb;
    int got_ok, clean;

    memset(&fb, 0, sizeof(fb));
    if (write_text_file(path, text)) return 0;
    got_ok = fb_load_cado(path, 1.0, &fb) == 0;
    clean = got_ok || (!fb.n && !fb.primes && !fb.roots && !fb.logp &&
                       !fb.ispow && !fb_is_transform_validated(&fb));
    fb_free(&fb);
    unlink(path);
    return got_ok == want_ok && clean;
}

/* A rejection that prints the wrong reason sends the user looking in the
 * wrong place, so check the message as well as the failure. poly_load's
 * stderr goes to a temporary file for the call. */
static int poly_diag_case(const char *text, const char *want)
{
    char path[] = "/tmp/cuda-sieve-poly-diag-XXXXXX";
    char msg[512];
    FILE *cap = tmpfile();
    poly_t P;
    size_t n = 0;
    int saved = -1, got_ok = 1;

    if (!cap || write_text_file(path, text)) goto done;
    fflush(stderr);
    saved = dup(STDERR_FILENO);
    if (saved < 0 || dup2(fileno(cap), STDERR_FILENO) < 0) goto done;
    got_ok = poly_load(path, &P) == 0;
    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    rewind(cap);
    n = fread(msg, 1, sizeof(msg) - 1u, cap);
done:
    msg[n] = '\0';
    if (saved >= 0) close(saved);
    if (cap) fclose(cap);
    unlink(path);
    if (got_ok || !strstr(msg, want)) {
        fprintf(stderr, "poly diagnostic: want \"%s\", got \"%s\"\n", want, msg);
        return 0;
    }
    return 1;
}

/* The text factor-base parser is a trust boundary. Check strict numeric
 * conversion, complete-line consumption, exponent metadata, roots, ordering,
 * and fail-closed cleanup independently of the central validator. */
static int verify_cado_parser_strict(void)
{
    static const char *bad[] = {
        "4294967296: 1\n",       /* narrowing overflow */
        "6: 1\n",                /* mixed composite */
        "4: 1\n",                /* powers need long-form metadata */
        "4:1,0: 1\n",            /* nexp smaller than v_p(q) */
        "4:2,2: 1\n",            /* no positive exponent increment */
        "5:\n",                  /* no roots */
        "5: 10\n",               /* root == 2*q */
        "5: 1,\n",               /* missing root after comma */
        "5: 1 junk\n",           /* trailing junk */
        "5: 1\n3: 1\n",         /* base stream not sorted */
        "# maxbits = 99\n5: 1\n",/* malformed recognized header */
        "# maxbits = 2\n9:2,1: 1\n", /* power exceeds declared bound */
        "# maxbits = 4\n# maxbits = 5\n5: 1\n",/* conflicting metadata */
        "5: 1\rjunk\n"          /* junk after a bare carriage return */
    };
    const char good[] =
        "# DEGREE: 2\n"
        "# maxbits is 15\n"
        "# maxbits = 2\r\n"
        "2: 0\n"
        "3: 1,4\n"
        "5: 1\n"             /* base primes are limited by lim, not maxbits */
        "4:2,1: 1,5\n";
    const char good32[] =
        "# maxbits = 32\n"
        "4293001441:2,1: 1\n"; /* 65521^2: full uint32 bit width is legal */
    size_t i;

    if (!cado_parse_case(good, 1) || !cado_parse_case(good32, 1)) return 0;
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
        if (!cado_parse_case(bad[i], 0)) return 0;
    return 1;
}

static int poly_parse_case(const char *text, int want_ok, poly_t *out)
{
    char path[] = "/tmp/cuda-sieve-poly-strict-XXXXXX";
    poly_t P;
    int got_ok, clean;

    memset(&P, 0x5a, sizeof(P));
    if (write_text_file(path, text)) return 0;
    got_ok = poly_load(path, &P) == 0;
    clean = got_ok || (P.deg == -1 && P.skew == 0.0 && P.y0s[0] == '\0' &&
                       P.y1s[0] == '\0');
    unlink(path);
    if (got_ok && out) *out = P;
    return got_ok == want_ok && clean;
}

static int verify_poly_parser_strict(void)
{
    static const char good[] =
        "n: 12345678901234567890\n"
        "cost: 2.7e18\n"
        "skew: 1.5\n"
        "c0: -1\n"
        "c1: 2\n"
        "Y0: 0\n"
        "Y1: 1\n"
        "rlim: 100\n"
        "rlambda: 2.5 # comment\n";
    static const char *bad[] = {
        "c0:-1\nc1:2\nY0:0\nY1:1\n",                    /* no skew */
        "skew:nan\nc0:-1\nc1:2\nY0:0\nY1:1\n",          /* nonfinite */
        "skew:1\nc0:-1\nc1:2\nY0:0\n",                  /* no Y1 */
        "skew:1\nc0:-1\nc1:2\nY0:0\nY1:0\n",            /* zero rational form */
        "skew:1\nc0:-1\nc1:2junk\nY0:0\nY1:1\n",        /* truncated token */
        "skew:1\nc0:-1\nc1x:2\nc1:2\nY0:0\nY1:1\n",      /* malformed claimed coefficient key */
        "skew:1\nc0:-1\nc9:2\nY0:0\nY1:1\n",            /* unsupported degree */
        "skew:1\nc0:-1\nc1:2\nc1:3\nY0:0\nY1:1\n",      /* duplicate */
        "skew:1\nc0:-1\nc1:2\nY0:0\nY1:1\nrlim:4294967296\n",
        "skew:1x\nc0:-1\nc1:2\nY0:0\nY1:1\n",
        "skew:1\rjunk\nc0:-1\nc1:2\nY0:0\nY1:1\n"
    };
    /* Unicode pasted from a web page must be named, not blamed on the
     * value's syntax. The field-name cases loaded before, silently dropping
     * c2 and giving a degree-1 polynomial. The rest guard the ASCII
     * diagnostics against being shadowed. */
    static const struct { const char *text, *want; } diag[] = {
        { "skew:1\nc0:-1\nc1:2\nY0:0\xE2\x80\x8B\nY1:1\n",
          ":4: Y0 contains U+200B (zero-width space) at column 5; it is invisible" },
        { "skew:1\nc0:-1\nc1:2\nY0:\xC2\xA0" "0\nY1:1\n",
          ":4: Y0 contains U+00A0 (no-break space) at column 4;" },
        { "skew:1\nc0:\xE2\x88\x92" "1\nc1:2\nY0:0\nY1:1\n",
          ":2: c0 contains U+2212 (minus sign) at column 4; use ASCII '-'" },
        { "skew:1\nc0:-1\nc1:2\n\xE2\x80\x8B" "c2:1\nY0:0\nY1:1\n",
          ":4: field name contains U+200B (zero-width space) at column 1;" },
        { "skew:1\nc0:-1\nc1:2\n\xEF\xBD\x83" "2:1\nY0:0\nY1:1\n",
          ":4: field name contains U+FF43 at column 1; job files must be plain ASCII" },
        { "skew:1\nc0:-1\nc1:2\nc2:\xE0\x80\x80\nY0:0\nY1:1\n",
          ":4: c2 contains byte 0xE0 (not UTF-8) at column 4;" },
        { "skew:1\nfoo # caf\xC3\xA9\nc0:-1\nc1:2\nY0:0\nY1:1\n",
          ":2: non-comment line has no ':' separator" },
        { "skew:1\nc0:-1\nc123456789012345678:2\nY0:0\nY1:1\n",
          ":3: coefficient index overflow" },
    };
    /* A Notepad byte-order mark, a zero-padded coefficient name longer than
     * any field name, and non-ASCII inside a comment all still load. */
    static const char *good_deg1[] = {
        "\xEF\xBB\xBFn: 12345\nskew:1\nc0:-1\nc1:2\nY0:0\nY1:1\n",
        "\xEF\xBB\xBFskew:1\nc0:-1\nc1:2\nY0:0\nY1:1\n",
        "skew:1\nc0:-1\nc0000000000000001:2\nY0:0\nY1:1\n",
        "skew:1 # caf\xC3\xA9\nc0:-1\nc1:2\nY0:0\nY1:1\n",
    };
    char long_coeff[512];
    poly_t P;
    size_t i, off;

    if (!poly_parse_case(good, 1, &P) || P.deg != 1 || P.skew != 1.5 ||
        P.rlim != 100u || P.rlambda != 2.5 || strcmp(P.cs[1], "2"))
        return 0;
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
        if (!poly_parse_case(bad[i], 0, NULL)) return 0;
    for (i = 0; i < sizeof(diag) / sizeof(diag[0]); i++)
        if (!poly_diag_case(diag[i].text, diag[i].want)) return 0;
    for (i = 0; i < sizeof(good_deg1) / sizeof(good_deg1[0]); i++)
        if (!poly_parse_case(good_deg1[i], 1, &P) || P.deg != 1 ||
            strcmp(P.cs[1], "2"))
            return 0;

    off = (size_t)snprintf(long_coeff, sizeof(long_coeff),
                           "skew:1\nc0:-1\nc1:");
    memset(long_coeff + off, '7', 100);
    off += 100;
    snprintf(long_coeff + off, sizeof(long_coeff) - off, "\nY0:0\nY1:1\n");
    if (!poly_diag_case(long_coeff,
                        ":3: c1 must be a decimal integer of at most 79 characters"))
        return 0;
    return 1;
}

static int verify_qrange_parser(void)
{
    static const char *bad[] = {
        "", ":", "100", "100:abc", "100:200junk", "-1:2", "1:-2",
        "18446744073709551616:2", "200:100", "100::200",
        " 2:4", "2: 4", "100:0"
    };
    uint64_t lo = 0, hi = 0;
    size_t i;

    if (bench_parse_qrange("100:200", &lo, &hi) || lo != 100 || hi != 200)
        return 0;
    if (bench_parse_qrange("100:", &lo, &hi) || lo != 100 || hi != 0)
        return 0;
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        lo = UINT64_C(0x1111111111111111);
        hi = UINT64_C(0x2222222222222222);
        if (bench_parse_qrange(bad[i], &lo, &hi) == 0 ||
            lo != UINT64_C(0x1111111111111111) ||
            hi != UINT64_C(0x2222222222222222))
            return 0;
    }
    return 1;
}

/* Trace one position: every factor-base ideal that hits (i,j), and the log
 * sum they contribute. This is the half of gate 5 that lives on our side --
 * `las_tracek -traceab a,b` prints the same thing from las's pipeline, and
 * `-traceab` is basis-independent, so the two are directly comparable without
 * worrying that our basis is las's with the first vector negated. */
static void trace_side(const char *name, const fb_t *fb, const qlat_t *L,
                       const poly_t *P, int32_t i, int32_t j,
                       uint32_t bkthresh, int logI, uint32_t J,
                       double scale, int is_sq, uint64_t sq, int maxlist)
{
    uint64_t sum = 0;
    uint32_t k, nhit = 0;
    norm_t N;
    long T;
    memset(&N, 0, sizeof N);
    norm_setup(&N, P, L, logI, J, scale, is_sq);
    T = (long)floorf(norm_target_host(&N, i, j) + 0.5f);
    if (T < 0) T = 0;

    printf("  %s (scale %.2f): ideals hitting (i=%d, j=%d)\n", name, scale, i, j);
    for (k = 0; k < fb->n; k++) {
        const uint32_t q = fb->primes[k], r = fb->roots[k];
        if ((uint64_t)q == sq) continue;      /* the special-q is divided out */
        if (!hits_def_pub(q, r, L, i, j)) continue;
        sum += fb->logp[k];
        if ((int)nhit < maxlist)
            printf("      q=%-10u %-11s logp=%-4u %s\n", q,
                   r >= q ? "projective" : "affine", fb->logp[k],
                   q < bkthresh || fb_is_proper_power(q) ? "line-sieved" : "bucketed");
        nhit++;
    }
    if ((int)nhit > maxlist) printf("      ... %u more\n", nhit - maxlist);
    printf("  %s: %u ideals, init norm S = %ld, log sum = %llu,"
           "  S - sum = %ld\n\n",
           name, nhit, T, (unsigned long long)sum, T - (long)sum);
}

int main(int argc, char **argv)
{
    const char *polypath = "../oracle/c183.poly";
    const char *cadofb = NULL;
    uint64_t q = 120000053ull, rho = 112625526ull;
    uint32_t rlim = 67100000, bkthresh = 1u << 15;
    int maxbits = 15, qmax = 200;
    int32_t ti = 0, tj = 0; int do_trace = 0;
    /* las's EXACT scales. It prints them rounded to 2 dp ("scale=1.28"), but also
 * prints logbase = 2^(1/scale) to 7 figures, from which scale = 1/log2(logbase)
 * comes out at exactly 1.275 and 1.925. The 2-dp values are wrong by enough to
 * move fb_log by one unit for some primes -- p=25811171 gives 32 at 1.28 and
 * 31, which is what las applies, at 1.275. */
    double scale1 = 1.275, scale0 = 1.925;
    double scale = 1.0;
    poly_t P; qlat_t L; fb_t fb, fbs;
    int i, n;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--poly") && i + 1 < argc) polypath = argv[++i];
        else if ((!strcmp(argv[i], "--fb1") || !strcmp(argv[i], "--cadofb")) && i + 1 < argc)
            cadofb = argv[++i];
        else if (!strcmp(argv[i], "--rlim") && i + 1 < argc) rlim = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) maxbits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) q = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--rho") && i + 1 < argc) rho = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) {
            if (parse_positive_test_double("--scale", argv[++i], &scale)) return 2;
        }
        else if (!strcmp(argv[i], "--scale1") && i + 1 < argc) {
            if (parse_positive_test_double("--scale1", argv[++i], &scale1)) return 2;
        }
        else if (!strcmp(argv[i], "--scale0") && i + 1 < argc) {
            if (parse_positive_test_double("--scale0", argv[++i], &scale0)) return 2;
        }
        else if (!strcmp(argv[i], "--qmax") && i + 1 < argc) qmax = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bkthresh") && i + 1 < argc) bkthresh = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d,%d", &ti, &tj) != 2) {
                fprintf(stderr, "--trace wants i,j\n"); return 2;
            }
            do_trace = 1;
        }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }

    if (poly_load(polypath, &P) != 0) return 1;
    qlat_build(&L, q, rho, P.skew);
    printf("q=%llu rho=%llu  basis (a0,a1,b0,b1)=(%lld,%lld,%lld,%lld)\n",
           (unsigned long long)q, (unsigned long long)rho,
           (long long)L.a0, (long long)L.a1, (long long)L.b0, (long long)L.b1);
    /* b = i*a1 + j*b1, and for a special-q of this size the reduced basis has
     * b-components of order sqrt(q/skew) ~ 1. Print it: it is the fact that
     * decides whether a projective ideal hits one row or every row. */
    printf("  b(i,j) = %lld*i + %lld*j   ->  a projective ideal above p hits"
           " %s\n\n", (long long)L.a1, (long long)L.b1,
           L.b1 == 0 ? "i == 0 (mod p) on EVERY row" : "a full lattice");

    printf("[0] bucket slice-table bounds\n");
    ok("16-bit slice ID boundary", verify_slice_bounds(),
       "65,536 slices accepted; 65,537 rejected fail-closed");
    ok("64-bit grid-stride product",
       bench_grid_product_u64(1u << 24, 256u) == (UINT64_C(1) << 32),
       "16,777,216 x 256 remains 2^32 rather than wrapping to zero");
    ok("strict finite decimal parser", verify_finite_double_parser(),
       "junk, range errors, NaN and infinities rejected without assignment");
    ok("strict unsigned decimal parser", verify_u64_decimal_parser(),
       "signs, whitespace, junk and overflow rejected without assignment");
    ok("checked survivor threshold", verify_survivor_bound(),
       "truncation preserved; invalid values and BOUND > CINIT rejected");
    ok("8-bit factor-base logs", verify_factor_base_log_bounds(),
       "out-of-range logs fail without publishing a partial array");
    ok("strict --qrange parser", verify_qrange_parser(),
       "open/closed ranges accepted; junk, overflow and reversed bounds rejected");

    printf("\n[1] production factor-base validation\n");
    ok("validator fail-closed cases", verify_validation_core(),
       "mixed composites, flags, roots, ordering, .afb.0 powers");
    ok("validator dense fast path", verify_validation_sieve_path(),
       "65,536 distinct moduli checked by sieve; injected composite rejected");
    ok("file loaders invoke validator", verify_loader_validation(),
       ".afb.0 and text inputs reject malformed moduli and clean up");
    ok("strict CADO text parser", verify_cado_parser_strict(),
       "numeric bounds, metadata, roots, ordering and trailing input checked");
    ok("strict polynomial parser", verify_poly_parser_strict(),
       "fields, exact integers, finite values, duplicates, pasted Unicode");

    printf("\n[2] transform vs definition, set equality, q <= %d, all roots"
           " in [0,2q)\n", qmax);
    ok("synthetic moduli", verify_transform(&L, qmax) == 0,
       "primes, prime powers, even moduli, affine + projective");

    if (do_trace) {
        const int64_t a = (int64_t)ti * L.a0 + (int64_t)tj * L.b0;
        const int64_t b = (int64_t)ti * L.a1 + (int64_t)tj * L.b1;
        printf("[trace] (i,j) = (%d,%d)  ->  (a,b) = (%lld,%lld)\n",
               ti, tj, (long long)a, (long long)b);
        printf("        compare with:  las_tracek ... -traceab %lld,%lld\n"
               "        (-traceab is basis-independent, so the i-mirroring\n"
               "         between our basis and las's does not matter here)\n\n",
               (long long)a, (long long)b);
        if (cadofb) {
            if (fb_load_cado(cadofb, scale1, &fb) != 0) return 1;
            trace_side("side 1", &fb, &L, &P, ti, tj, bkthresh, 15, 16384,
                       scale1, 1, q, 40);
            fb_free(&fb);
        }
        {
            poly_t R = P;
            if (rfb_build(&R, rlim, maxbits, scale0, &fb) != 0) return 1;
            R.deg = 1; R.c[0] = R.y0; R.c[1] = R.y1;
            for (int z = 2; z < BENCH_NCOEFF; z++) R.c[z] = 0.0;
            trace_side("side 0", &fb, &L, &R, ti, tj, bkthresh, 15, 16384,
                       scale0, 0, 0, 40);
            fb_free(&fb);
        }
        return 0;
    }

    printf("\n[3] transform vs definition, driven by the real factor bases\n");

    if (cadofb) {
        uint32_t np, nzp;
        if (fb_load_cado(cadofb, scale, &fb) != 0) return 1;
        np = count_proj(&fb, &nzp);
        if (fb_split_small(&fb, bkthresh, &fbs) != 0) return 1;
        if (fb_fill_logp(&fbs, scale) != 0) {
            fb_free(&fbs); fb_free(&fb); return 1;
        }
        n = verify_fb_transform(&fbs, &L, bkthresh);
        ok("side 1 small part", n > 0, "%d entries checked", n);
        ok("side 1 moduli are prime powers", fb_check_prime_powers(&fb) < 0,
           "%u ideals", fb.n);
        ok("side 1 projective entries seen", np > 0,
           "%u projective, %u with a NONZERO reciprocal", np, nzp);
        {   /* The oracle-free density gate, as a PASS/FAIL rather than a
             * print, so a log-parser or scale regression fails `make check`.
             *
             * 42.2913 was computed INDEPENDENTLY from c183.fb1 -- a separate
             * implementation that re-derives each entry's base prime and log
             * delta from the file -- not read off our own loader. It is the
             * post-2026-08-02 value: before the is_power fix it was 42.744, and
             * wiring this gate is what caught that the constant was stale.
             * fb_log's rounding is not linear in scale, so gate only at 1.0. */
            const double got = log_density(&fb);
            if (scale == 1.0)
                ok("side 1 sum(logp/q)", fabs(got - 42.2913) < 0.002,
                   "%.4f (independent expectation 42.2913)", got);
            else
                printf("       side 1 sum(logp/q) = %.4f at scale %.3f"
                       " (gate runs at scale 1.0 only)\n", got, scale);
        }
        fb_free(&fbs); fb_free(&fb);
    } else {
        printf("       (skipped: pass --fb1 ../oracle/c183.fb1)\n");
    }

    {
        uint32_t np, nzp;
        poly_t R = P;
        if (rfb_build(&R, rlim, maxbits, scale, &fb) != 0) return 1;
        np = count_proj(&fb, &nzp);
        if (fb_split_small(&fb, bkthresh, &fbs) != 0) return 1;
        if (fb_fill_logp(&fbs, scale) != 0) {
            fb_free(&fbs); fb_free(&fb); return 1;
        }
        n = verify_fb_transform(&fbs, &L, bkthresh);
        ok("side 0 small part", n > 0, "%d entries checked", n);
        ok("side 0 moduli are prime powers", fb_check_prime_powers(&fb) < 0,
           "%u ideals", fb.n);
        /* Y1 = 59*101*127*281*1259*38321*5746453, so seven projective primes,
         * and the ladders 59^2, 101^2, 127^2 stay under 2^15. */
        ok("side 0 projective ladder", nzp >= 3,
           "%u projective, %u with a NONZERO reciprocal (expect >= 3: 59^2,"
           " 101^2, 127^2)", np, nzp);
        {   /* likewise independent: a separate sieve to rlim plus the power
             * ladder, which also reproduces the ideal count 3,957,374 exactly */
            const double got = log_density(&fb);
            if (scale == 1.0)
                ok("side 0 sum(logp/q)", fabs(got - 25.2037) < 0.002,
                   "%.4f (independent expectation 25.2037)", got);
            else
                printf("       side 0 sum(logp/q) = %.4f at scale %.3f"
                       " (gate runs at scale 1.0 only)\n", got, scale);
        }
        fb_free(&fbs); fb_free(&fb);
    }

    printf("\n[4] exact-norm width admission\n");
    {
        norm_t W;
        memset(&W, 0, sizeof(W));
        W.deg = 8;
        W.log2M = 252.0f;       /* + log2(9) = 255.17: fits 256 bits */
        ok("octic below 256-bit bound", norm_fits_exact(&W, 256),
           "upper bound %.2f bits", norm_exact_bound_bits(&W));
        W.log2M = 253.0f;       /* + log2(9) = 256.17: must be refused */
        ok("octic above 256-bit bound", !norm_fits_exact(&W, 256),
           "upper bound %.2f bits refused", norm_exact_bound_bits(&W));
    }

    printf("\n%s\n", fail ? "*** FAILED ***" : "all gates passed");
    return fail;
}
