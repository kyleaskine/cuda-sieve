/* Loader for the native fbgen/CADO-compatible text factor-base format.
 *
 * Using this instead of GGNFS's .afb.0 buys two things that matter for parity:
 * prime powers (which .afb.0 has none of), and the exact per-entry log
 * increment las will use.
 *
 * Format, from fb_entry_general::parse_line:
 *   q: r1,r2,...              short form; q is prime, nexp=1, oldexp=0
 *   q:nexp,oldexp: r1,r2,...  long form;  q may be a prime power p^k
 * A root value >= q is projective, encoding the actual root as r - q
 * (fb_root_p1::to_old_format), the same convention .afb.0 uses.
 *
 * The log increment is NOT log(q). It is
 *     fb_log_delta = fb_log(p^nexp) - fb_log(p^oldexp),
 *     fb_log(n) = floor(log2(n) * scale + 0.5)
 * i.e. the marginal cost of going from p^oldexp to p^nexp, which is what makes
 * powers compose correctly with their base prime.
 *
 * The file is grouped by base prime with the powers of each listed inside the
 * group, so it is NOT globally sorted by q -- and everything downstream
 * (fb_split_small, fb_restrict, the fill's row tier) needs ascending q. The k==1
 * entries do arrive in ascending order and the powers are few (with the usual
 * maxbits 15, every power is <= 32768), so a merge is enough and a
 * full sort of 7.6M entries is not.
 */
#define _GNU_SOURCE
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <limits.h>

/* If q = p^k with k > 1, return the BASE PRIME p and set *kk = k; else 0.
 *
 * It must be the base prime, i.e. k maximal. The previous version searched k
 * upward from 2 and returned the first that worked, which is k MINIMAL and so
 * the largest base: it read 81 as 9^2 and 729 as 27^2. The exponents in the
 * file are relative to the base prime (`81:5,4: 114` means 3^5 over 3^4), so a
 * wrong base silently inflates the log increment -- 4 instead of 2 for 81, 6
 * instead of 2 for 729, 3 instead of 1 for 16. Placement was unaffected, which
 * is why only a byte-level trace against las found it (gate 5, 2026-08-02).
 *
 * Trial division for the smallest prime factor is exact and, with 97 long-form
 * lines in the file, free. */
static uint32_t is_power(uint32_t q, int *kk)
{
    uint32_t d;
    for (d = 2; (uint64_t)d * d <= (uint64_t)q; d++) {
        uint32_t t;
        int k = 0;
        if (q % d) continue;
        for (t = q; t % d == 0; t /= d) k++;
        if (t == 1 && k > 1) { *kk = k; return d; }
        return 0;                     /* not a pure power of its least factor */
    }
    return 0;                         /* prime */
}

typedef struct {
    uint32_t *q, *r;
    uint8_t *logp;
    uint32_t n, cap;
} fbvec_t;

static void fbvec_free(fbvec_t *v)
{
    if (!v) return;
    free(v->q);
    free(v->r);
    free(v->logp);
    memset(v, 0, sizeof(*v));
}

static int fbvec_reserve(fbvec_t *v, uint32_t need)
{
    uint32_t cap;
    uint32_t *q, *r;
    uint8_t *logp;

    if (need <= v->cap) return 0;
    cap = v->cap ? v->cap : 4096u;
    while (cap < need) {
        if (cap > UINT32_MAX / 2u) { cap = need; break; }
        cap *= 2u;
    }
    if (cap < need) {
        errno = EOVERFLOW;
        return -1;
    }
#if SIZE_MAX <= UINT32_MAX
    if ((size_t)cap > SIZE_MAX / sizeof(*v->q) ||
        (size_t)cap > SIZE_MAX / sizeof(*v->r) ||
        (size_t)cap > SIZE_MAX / sizeof(*v->logp)) {
        errno = EOVERFLOW;
        return -1;
    }
#endif

    /* Grow each array in place when possible. If a later realloc fails, the
     * arrays already grown remain valid and v->cap deliberately stays at the
     * old common capacity. A later retry can finish the growth safely, while
     * peak RSS never includes a complete old and complete new three-array
     * vector at the same time. */
    q = (uint32_t *)realloc(v->q, (size_t)cap * sizeof(*v->q));
    if (!q) return -1;
    v->q = q;
    r = (uint32_t *)realloc(v->r, (size_t)cap * sizeof(*v->r));
    if (!r) return -1;
    v->r = r;
    logp = (uint8_t *)realloc(v->logp, (size_t)cap * sizeof(*v->logp));
    if (!logp) return -1;
    v->logp = logp;
    v->cap = cap;
    return 0;
}

static int fbvec_push(fbvec_t *v, uint32_t q, uint32_t r, uint8_t logp)
{
    if (v->n == UINT32_MAX) { errno = EOVERFLOW; return -1; }
    if (fbvec_reserve(v, v->n + 1u)) return -1;
    v->q[v->n] = q;
    v->r[v->n] = r;
    v->logp[v->n] = logp;
    v->n++;
    return 0;
}

typedef struct {
    uint32_t q, r, order;
    uint8_t logp;
} fbsort_entry_t;

static int fbsort_cmp(const void *aa, const void *bb)
{
    const fbsort_entry_t *a = (const fbsort_entry_t *)aa;
    const fbsort_entry_t *b = (const fbsort_entry_t *)bb;
    if (a->q != b->q) return a->q < b->q ? -1 : 1;
    if (a->order != b->order) return a->order < b->order ? -1 : 1;
    return 0;
}

/* A hostile file can contain far more power entries than a normal makefb
 * output. The old insertion sort was quadratic in that count, turning strict
 * parsing into an easy CPU denial of service. Sort a temporary AoS view with
 * qsort and retain input order for equal moduli. */
static int fbvec_sort_by_q(fbvec_t *v)
{
    fbsort_entry_t *a;
    uint32_t i;

    if (v->n < 2u) return 0;
#if SIZE_MAX <= UINT32_MAX
    if ((size_t)v->n > SIZE_MAX / sizeof(*a)) {
        errno = EOVERFLOW;
        return -1;
    }
#endif
    a = (fbsort_entry_t *)malloc((size_t)v->n * sizeof(*a));
    if (!a) return -1;
    for (i = 0; i < v->n; i++) {
        a[i].q = v->q[i];
        a[i].r = v->r[i];
        a[i].logp = v->logp[i];
        a[i].order = i;
    }
    qsort(a, v->n, sizeof(*a), fbsort_cmp);
    for (i = 0; i < v->n; i++) {
        v->q[i] = a[i].q;
        v->r[i] = a[i].r;
        v->logp[i] = a[i].logp;
    }
    free(a);
    return 0;
}

static char *cado_skip_space(char *s)
{
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

static int cado_at_end(char *s)
{
    s = cado_skip_space(s);
    if (*s == '\0' || *s == '#') return 1;
    if (*s == '\n') return s[1] == '\0';
    if (*s == '\r') return s[1] == '\0' ||
                            (s[1] == '\n' && s[2] == '\0');
    return 0;
}

static int cado_u32(char **sp, uint32_t *out)
{
    char *s = cado_skip_space(*sp), *end;
    unsigned long long v;

    if (*s < '0' || *s > '9') return -1;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno == ERANGE || end == s || v > UINT32_MAX) return -1;
    *sp = end;
    *out = (uint32_t)v;
    return 0;
}

static int cado_maxbits_comment(char *line, int *maxbits)
{
    char *s = cado_skip_space(line);
    uint32_t v;

    if (*s++ != '#') return 0;
    s = cado_skip_space(s);
    if (strncmp(s, "maxbits", 7) != 0 ||
        (s[7] && s[7] != ' ' && s[7] != '\t' && s[7] != '='))
        return 0;
    s = cado_skip_space(s + 7);
    /* Only the assignment form is metadata. Other prose comments beginning
     * with "maxbits" are ordinary comments and must not abort the load. */
    if (*s != '=') return 0;
    s++;
    if (cado_u32(&s, &v) || !cado_at_end(s) || v < 1u || v > 32u)
        return -1;
    *maxbits = (int)v;
    return 1;
}

static int cado_line_error(const char *path, unsigned long linenr,
                           const char *msg)
{
    fprintf(stderr, "fb_load_cado: %s:%lu: %s\n", path, linenr, msg);
    return -1;
}

int fb_load_cado(const char *path, double scale, fb_t *fb)
{
    enum { LINE_CAP = 65536 };
    FILE *f = NULL;
    char *line = NULL;
    fbvec_t base = {0}, power = {0};
    uint32_t i, j, k;
    unsigned long linenr = 0;
    uint32_t max_power_q = 0;
    int file_maxbits = 0, maxbits_seen = 0;
    int rc = -1;

    if (!fb || !path) { errno = EINVAL; return -1; }
    memset(fb, 0, sizeof(*fb));
    if (!isfinite(scale) || scale <= 0.0) {
        fprintf(stderr, "fb_load_cado: scale %.17g must be finite and positive\n",
                scale);
        return -1;
    }
    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "fb_load_cado: cannot open %s\n", path);
        return -1;
    }
    line = (char *)malloc(LINE_CAP);
    if (!line) goto done;

    while (fgets(line, LINE_CAP, f)) {
        char *s, *after_first, *scan;
        uint32_t q, p, nexp_u = 1u, oldexp_u = 0u;
        int kk = 1, long_form = 0, comment_kind;
        uint8_t lg;
        fbvec_t *dst;
        uint32_t before;

        linenr++;
        if (!strchr(line, '\n') && !feof(f)) {
            int ch;
            while ((ch = fgetc(f)) != '\n' && ch != EOF) {}
            cado_line_error(path, linenr, "line exceeds 65535 bytes");
            goto done;
        }
        s = cado_skip_space(line);
        if (!*s || ((*s == '\n' || *s == '\r') && cado_at_end(s)))
            continue;
        if (*s == '#') {
            int parsed_maxbits = file_maxbits;
            comment_kind = cado_maxbits_comment(s, &parsed_maxbits);
            if (comment_kind < 0) {
                cado_line_error(path, linenr,
                                "malformed # maxbits comment (want 1..32)");
                goto done;
            }
            if (comment_kind > 0) {
                if (maxbits_seen && parsed_maxbits != file_maxbits) {
                    cado_line_error(path, linenr,
                                    "conflicting # maxbits declarations");
                    goto done;
                }
                file_maxbits = parsed_maxbits;
                maxbits_seen = 1;
            }
            continue;
        }

        if (cado_u32(&s, &q) || q < 2u) {
            cado_line_error(path, linenr,
                            "modulus must be a complete unsigned 32-bit integer >= 2");
            goto done;
        }
        s = cado_skip_space(s);
        if (*s++ != ':') {
            cado_line_error(path, linenr, "missing ':' after modulus");
            goto done;
        }
        after_first = s;
        for (scan = s; *scan && *scan != '\n' && *scan != '\r' && *scan != '#'; scan++)
            if (*scan == ':') { long_form = 1; break; }

        /* Classifying every modulus here costs one deterministic Miller-Rabin
         * per line -- 4.8M of them on a c183 base -- to answer a question
         * fb_validate() answers again below from a single Eratosthenes list
         * over the whole file. The parser therefore does not classify at all;
         * the format is what makes that safe, because a prime-power modulus
         * must use the long form. A short-form line asserts primality, is
         * taken at its word here, and is checked there.
         *
         * Nothing is lost from the diagnostics: a composite smuggled onto a
         * short-form line comes back as "modulus %u is neither prime nor a
         * power of one prime", and a prime power written in the short form as
         * an ispow disagreement -- both naming this file as the source. */
        p = q;

        s = after_first;
        if (long_form) {
            if (cado_u32(&s, &nexp_u)) {
                cado_line_error(path, linenr, "invalid nexp");
                goto done;
            }
            s = cado_skip_space(s);
            if (*s++ != ',') {
                cado_line_error(path, linenr, "missing ',' between exponents");
                goto done;
            }
            if (cado_u32(&s, &oldexp_u)) {
                cado_line_error(path, linenr, "invalid oldexp");
                goto done;
            }
            s = cado_skip_space(s);
            if (*s++ != ':') {
                cado_line_error(path, linenr, "missing ':' after exponents");
                goto done;
            }
            /* Classify only now, and only when the line's own declared
             * exponent says a proper power is possible. is_power() is trial
             * division to sqrt(q), which returns after the first divisor for a
             * composite but runs the full loop for a prime -- ~11,800
             * iterations at lim 1.4e8 and ~65,500 near 2^32. The format permits
             * an ordinary prime in the general long form (`p:1,0: r`), and
             * classifying those would put that loop on a per-line path. */
            if (nexp_u >= 2) {
                int pk = 1;
                const uint32_t base_p = is_power(q, &pk);
                if (base_p) { p = base_p; kk = pk; }
            }
            if (nexp_u > INT_MAX || oldexp_u > INT_MAX ||
                oldexp_u >= nexp_u || nexp_u < (uint32_t)kk) {
                cado_line_error(path, linenr,
                                "exponents must satisfy 0 <= oldexp < nexp and nexp >= v_p(q)");
                goto done;
            }
        }
        if (kk > 1 && q > max_power_q) max_power_q = q;
        if (fb_log_delta_checked(p, (int)nexp_u, (int)oldexp_u, scale, &lg)) {
            cado_line_error(path, linenr,
                            "factor-base log increment is not representable in 8 bits");
            goto done;
        }

        dst = kk > 1 ? &power : &base;
        before = dst->n;
        for (;;) {
            uint32_t r;
            s = cado_skip_space(s);
            if (cado_u32(&s, &r)) {
                cado_line_error(path, linenr,
                                dst->n == before ? "entry has no roots" : "missing root after ','");
                goto done;
            }
            if ((uint64_t)r >= (uint64_t)q * 2u) {
                cado_line_error(path, linenr,
                                "root must be smaller than 2*q");
                goto done;
            }
            if (fbvec_push(dst, q, r, lg)) {
                cado_line_error(path, linenr,
                                errno == EOVERFLOW ? "too many factor-base entries" : "out of memory");
                goto done;
            }
            s = cado_skip_space(s);
            if (*s == ',') { s++; continue; }
            if (!cado_at_end(s)) {
                cado_line_error(path, linenr, "trailing characters after root list");
                goto done;
            }
            break;
        }
    }
    if (ferror(f)) {
        fprintf(stderr, "fb_load_cado: read error on %s\n", path);
        goto done;
    }
    if (!base.n && !power.n) {
        fprintf(stderr, "fb_load_cado: %s contains no factor-base entries\n", path);
        goto done;
    }

    for (i = 1; i < base.n; i++)
        if (base.q[i] < base.q[i - 1]) {
            fprintf(stderr, "fb_load_cado: %s: base primes not ascending at entry %u\n",
                    path, i);
            goto done;
        }
    if (maxbits_seen &&
        (uint64_t)max_power_q > (UINT64_C(1) << file_maxbits)) {
        fprintf(stderr,
                "fb_load_cado: %s: prime-power modulus %u exceeds"
                " # maxbits = %d\n",
                path, max_power_q, file_maxbits);
        goto done;
    }
    if (fbvec_sort_by_q(&power)) {
        fprintf(stderr,
                "fb_load_cado: %s: could not sort prime-power entries: %s\n",
                path, errno == EOVERFLOW ? "size overflow" : "out of memory");
        goto done;
    }

    if ((uint64_t)base.n + power.n > UINT32_MAX) {
        errno = EOVERFLOW;
        fprintf(stderr, "fb_load_cado: %s has too many entries\n", path);
        goto done;
    }
    fb->n = base.n + power.n;
    fb->maxbits = file_maxbits;
#if SIZE_MAX <= UINT32_MAX
    if ((size_t)fb->n > SIZE_MAX / sizeof(*fb->primes)) {
        errno = EOVERFLOW;
        goto done;
    }
#endif
    fb->primes = (uint32_t *)malloc((size_t)fb->n * sizeof(*fb->primes));
    fb->roots = (uint32_t *)malloc((size_t)fb->n * sizeof(*fb->roots));
    fb->logp = (uint8_t *)malloc((size_t)fb->n * sizeof(*fb->logp));
    fb->ispow = (uint8_t *)malloc((size_t)fb->n * sizeof(*fb->ispow));
    if (!fb->primes || !fb->roots || !fb->logp || !fb->ispow) goto done;

    for (i = j = k = 0; k < fb->n; k++) {
        int take_base = i < base.n &&
                        (j >= power.n || base.q[i] <= power.q[j]);
        if (take_base) {
            fb->primes[k] = base.q[i];
            fb->roots[k] = base.r[i];
            fb->logp[k] = base.logp[i];
            fb->ispow[k] = 0;
            i++;
        } else {
            fb->primes[k] = power.q[j];
            fb->roots[k] = power.r[j];
            fb->logp[k] = power.logp[j];
            fb->ispow[k] = 1;
            j++;
        }
    }
    /* The trust boundary for a text file the parser above has only read, never
     * verified. The sieve fast path in fb_validate() is the single authority on
     * what is prime here. */
    if (fb_validate(fb, FB_VALIDATE_EXTERNAL_PRIME_POWERS, path) != 0)
        goto done;

    printf("text factor base %s: %u ideals (%u prime, %u prime-power)"
           " at scale %.3f\n", path, fb->n, base.n, power.n, scale);
    rc = 0;

done:
    if (f && fclose(f) && rc == 0) rc = -1;
    free(line);
    fbvec_free(&base);
    fbvec_free(&power);
    if (rc) fb_free(fb);
    return rc;
}
