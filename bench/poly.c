/* Polynomial load + norm-init setup for the apply kernel.
 *
 * The apply kernel must start each sieve cell at the log-norm bound, not at
 * zero. Evaluating |F(a,b)| directly is impossible in fp32: a,b reach ~4e8 and
 * c0 is ~2e44, so F is ~1e48 and the coefficients alone span 1e39 -- both
 * outside float range. The fix is to homogenise the whole thing once on the
 * host:
 *
 *     a = i*a0 + j*b0,  b = i*a1 + j*b1      (q-lattice)
 *     u = a/A, v = b/B                        with |u|,|v| <= 1 by construction
 *     F(a,b) = M * sum_k d_k u^k v^(deg-k)    d_k = c_k A^k B^(deg-k) / M
 *
 * so log2|F| = log2(M) + log2|sum d_k u^k v^(deg-k)|, and everything the GPU
 * touches is O(1) in fp32. This works only because a skewed polynomial has
 * balanced terms by construction -- which is exactly what skew is for. The
 * balance is checked and printed; if it ever fails the d_k spread will show it.
 */
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <errno.h>
#include <stdarg.h>

int bench_parse_finite_double(const char *text, double *out)
{
    char *end = NULL;
    double v;

    if (!text || !*text || !out) {
        errno = EINVAL;
        return -1;
    }
    errno = 0;
    v = strtod(text, &end);
    if (errno == ERANGE || end == text || *end || !isfinite(v)) {
        errno = (errno == ERANGE) ? ERANGE : EINVAL;
        return -1;
    }
    *out = v;
    return 0;
}

int bench_parse_u64_decimal(const char *text, uint64_t *out)
{
    const unsigned char *p;
    char *end = NULL;
    unsigned long long v;

    if (!text || !*text || !out) {
        errno = EINVAL;
        return -1;
    }
    for (p = (const unsigned char *)text; *p; p++) {
        if (*p < (unsigned char)'0' || *p > (unsigned char)'9') {
            errno = EINVAL;
            return -1;
        }
    }
    errno = 0;
    v = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end) {
        errno = (errno == ERANGE) ? ERANGE : EINVAL;
        return -1;
    }
    *out = (uint64_t)v;
    return 0;
}

static int sieve_bound_error(const char *source, int err, const char *fmt, ...)
{
    va_list ap;
    errno = err;
    if (!source) return -1;
    fprintf(stderr, "%s: ", source);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return -1;
}

int sieve_bound_checked(double scale, double allowance, uint32_t CINIT,
                        uint32_t *bound_out, const char *source)
{
    double raw;
    uint64_t bound;

    if (!bound_out)
        return sieve_bound_error(source, EINVAL,
                                 "null survivor-bound output");
    *bound_out = 0;
    if (!CINIT)
        return sieve_bound_error(source, EINVAL,
                                 "CINIT must be positive");
    /* norm_t stores the scale as float. A positive double that underflows to
     * zero there is not a positive scale in the code that actually runs. */
    if (!isfinite(scale) || scale < (double)FLT_TRUE_MIN ||
        scale > (double)FLT_MAX)
        return sieve_bound_error(source, EINVAL,
            "scale %.17g must be finite, positive, and representable as float",
            scale);
    if (!isfinite(allowance) || allowance < 0.0)
        return sieve_bound_error(source, EINVAL,
            "allowance %.17g must be finite and nonnegative", allowance);

    raw = scale * allowance + 1.0;
    /* For nonnegative raw values, the C cast truncates exactly as floor().
     * raw < CINIT + 1 is the pre-cast proof that the integer result fits both
     * uint32_t and the subtraction CINIT - bound. */
    if (!isfinite(raw) || raw < 0.0 || raw >= (double)CINIT + 1.0)
        return sieve_bound_error(source, ERANGE,
            "scale * allowance + 1 = %.17g yields a survivor bound above CINIT %u",
            raw, CINIT);
    bound = (uint64_t)raw;
    if (bound > CINIT)
        return sieve_bound_error(source, ERANGE,
            "survivor bound %llu exceeds CINIT %u",
            (unsigned long long)bound, CINIT);
    *bound_out = (uint32_t)bound;
    return 0;
}

static const char *skip_hspace(const char *s)
{
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

static int value_ended(const char *s)
{
    s = skip_hspace(s);
    if (*s == '\0' || *s == '#') return 1;
    if (*s == '\n') return s[1] == '\0';
    if (*s == '\r') return s[1] == '\0' ||
                            (s[1] == '\n' && s[2] == '\0');
    return 0;
}

int bench_parse_qrange(const char *text, uint64_t *qmin, uint64_t *qmax)
{
    char *end = NULL;
    unsigned long long lo, hi = 0;

    if (!text || !qmin || !qmax || *text < '0' || *text > '9') {
        errno = EINVAL;
        return -1;
    }
    errno = 0;
    lo = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != ':') {
        errno = (errno == ERANGE) ? ERANGE : EINVAL;
        return -1;
    }
    text = end + 1;
    if (*text) {
        if (*text < '0' || *text > '9') {
            errno = EINVAL;
            return -1;
        }
        errno = 0;
        hi = strtoull(text, &end, 10);
        if (errno == ERANGE || end == text || *end || hi == 0 || lo > hi) {
            errno = (errno == ERANGE) ? ERANGE : EINVAL;
            return -1;
        }
    }
    *qmin = (uint64_t)lo;
    *qmax = (uint64_t)hi;
    return 0;
}

static int poly_diag(const char *path, unsigned long linenr,
                     const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "poly_load: %s", path ? path : "(null)");
    if (linenr) fprintf(stderr, ":%lu", linenr);
    fputs(": ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return -1;
}

static int token_span(const char *value, const char **begin, size_t *len)
{
    const char *s = skip_hspace(value), *e;
    if (!*s || *s == '\n' || *s == '\r' || *s == '#') return -1;
    e = s;
    while (*e && *e != ' ' && *e != '\t' && *e != '\n' &&
           *e != '\r' && *e != '#')
        e++;
    if (!value_ended(e)) return -1;
    *begin = s;
    *len = (size_t)(e - s);
    return 0;
}

static int parse_finite_value(const char *value, double *out)
{
    const char *s;
    size_t n;
    char buf[128];
    double v;

    if (token_span(value, &s, &n) || n >= sizeof(buf)) return -1;
    memcpy(buf, s, n);
    buf[n] = '\0';
    if (bench_parse_finite_double(buf, &v)) return -1;
    *out = v;
    return 0;
}

static int parse_u32_value(const char *value, uint32_t *out)
{
    const char *s;
    size_t n;
    uint64_t v = 0;
    size_t i;

    if (token_span(value, &s, &n) || !n || s[0] == '+' || s[0] == '-')
        return -1;
    for (i = 0; i < n; i++) {
        unsigned d;
        if (s[i] < '0' || s[i] > '9') return -1;
        d = (unsigned)(s[i] - '0');
        if (v > (UINT32_MAX - d) / 10u) return -1;
        v = v * 10u + d;
    }
    *out = (uint32_t)v;
    return 0;
}

static int exact_decimal_is_zero(const char *s)
{
    if (*s == '+' || *s == '-') s++;
    while (*s == '0') s++;
    return *s == '\0';
}

/* Text copied through a web page can carry Unicode that looks like ASCII:
 * vBulletin code blocks insert U+200B, browsers turn spaces into U+00A0, and
 * some sites typeset '-' as U+2212. The line still looks right, so a
 * diagnostic that blames the value's syntax or length sends the reader looking
 * in the wrong place. Name the character and its column instead. Job files
 * are plain ASCII, so any other non-ASCII character is reported the same way.
 *
 * Decodes one UTF-8 sequence at u. Returns the code point, or -1 when u is not
 * the start of a well-formed sequence (overlong forms and surrogates
 * included). Each continuation test fails on the terminating NUL, so a
 * sequence cut short by the end of the line never reads past it. */
static long decode_utf8(const unsigned char *u)
{
#define CONT(x) (((x) & 0xC0u) == 0x80u)
    if (u[0] >= 0xC2u && u[0] <= 0xDFu && CONT(u[1]))
        return (long)((u[0] & 0x1Fu) << 6 | (u[1] & 0x3Fu));
    if (u[0] >= 0xE0u && u[0] <= 0xEFu && CONT(u[1]) && CONT(u[2]) &&
        !(u[0] == 0xE0u && u[1] < 0xA0u) &&     /* overlong */
        !(u[0] == 0xEDu && u[1] > 0x9Fu))       /* surrogate */
        return (long)((u[0] & 0x0Fu) << 12 | (u[1] & 0x3Fu) << 6 |
                      (u[2] & 0x3Fu));
    if (u[0] >= 0xF0u && u[0] <= 0xF4u && CONT(u[1]) && CONT(u[2]) &&
        CONT(u[3]) &&
        !(u[0] == 0xF0u && u[1] < 0x90u) &&     /* overlong */
        !(u[0] == 0xF4u && u[1] > 0x8Fu))       /* above U+10FFFF */
        return (long)((u[0] & 0x07u) << 18 | (u[1] & 0x3Fu) << 12 |
                      (u[2] & 0x3Fu) << 6 | (u[3] & 0x3Fu));
    return -1;
#undef CONT
}

/* Report the first non-ASCII byte in [from, to), if any. Returns 1 after
 * printing, 0 when the range is pure ASCII and the caller's own diagnostic
 * applies. */
static int diag_non_ascii(const char *path, unsigned long linenr,
                          const char *line, const char *from, const char *to,
                          const char *what, int what_len)
{
    static const struct {
        long cp;
        const char *name, *advice;
    } known[] = {
        { 0x00A0, "no-break space", NULL },
        { 0x200B, "zero-width space", NULL },
        { 0x200C, "zero-width non-joiner", NULL },
        { 0x200D, "zero-width joiner", NULL },
        { 0x2060, "word joiner", NULL },
        { 0xFEFF, "zero-width no-break space", NULL },
        { 0x2212, "minus sign", "use ASCII '-'" },
    };
    /* NULL advice marks a character that is invisible or looks like a space. */
    static const char invisible[] =
        "it is invisible, and usually pasted from a web page; retype the line";
    const unsigned char *u = (const unsigned char *)from;
    const char *name = NULL, *advice = "job files must be plain ASCII";
    char desc[48];
    size_t i;
    long cp;

    while (u < (const unsigned char *)to && *u < 0x80u) u++;
    if (u >= (const unsigned char *)to) return 0;
    cp = decode_utf8(u);
    for (i = 0; cp >= 0 && i < sizeof(known) / sizeof(known[0]); i++)
        if (known[i].cp == cp) {
            name = known[i].name;
            advice = known[i].advice ? known[i].advice : invisible;
        }
    if (cp < 0)
        snprintf(desc, sizeof(desc), "byte 0x%02X (not UTF-8)", u[0]);
    else if (name)
        snprintf(desc, sizeof(desc), "U+%04lX (%s)", cp, name);
    else
        snprintf(desc, sizeof(desc), "U+%04lX", cp);
    poly_diag(path, linenr, "%.*s contains %s at column %zu; %s",
              what_len, what, desc,
              (size_t)((const char *)u - line) + 1u, advice);
    return 1;
}

static int parse_exact_integer(const char *value, char *exact, size_t exact_cap,
                               double *approx)
{
    const char *s;
    size_t n, i = 0;
    char *end = NULL;
    double v;

    if (token_span(value, &s, &n) || !n || n >= exact_cap) return -1;
    if (s[i] == '+' || s[i] == '-') i++;
    if (i == n) return -1;
    for (; i < n; i++)
        if (s[i] < '0' || s[i] > '9') return -1;

    memcpy(exact, s, n);
    exact[n] = '\0';
    errno = 0;
    v = strtod(exact, &end);
    if (errno == ERANGE || end == exact || *end || !isfinite(v)) return -1;
    *approx = v;
    return 0;
}

int poly_load(const char *path, poly_t *P)
{
    FILE *f = NULL;
    char line[4096];
    unsigned long linenr = 0;
    unsigned seen_c = 0;
    unsigned seen_skew = 0, seen_y0 = 0, seen_y1 = 0;
    unsigned seen_rlim = 0, seen_alim = 0, seen_lpbr = 0, seen_lpba = 0;
    unsigned seen_mfbr = 0, seen_mfba = 0, seen_rlambda = 0, seen_alambda = 0;
    int rc = -1;

    if (!P || !path) {
        errno = EINVAL;
        return poly_diag(path, 0, "null input");
    }
    memset(P, 0, sizeof(*P));
    P->deg = -1;
    f = fopen(path, "r");
    if (!f) return poly_diag(path, 0, "cannot open file");

    while (fgets(line, sizeof(line), f)) {
        char *s, *colon, *key_end;
        const char *value;
        size_t key_len;
        linenr++;

        if (!strchr(line, '\n') && !feof(f)) {
            int ch;
            while ((ch = fgetc(f)) != '\n' && ch != EOF) {}
            poly_diag(path, linenr, "line exceeds %zu bytes", sizeof(line) - 1u);
            goto done;
        }
        /* Notepad's "UTF-8" encoding starts the file with a byte-order mark.
         * It is an encoding marker, not content. */
        if (linenr == 1u && !strncmp(line, "\xEF\xBB\xBF", 3))
            memmove(line, line + 3, strlen(line + 3) + 1u);
        s = (char *)skip_hspace(line);
        if (!*s || *s == '#' ||
            ((*s == '\n' || *s == '\r') && value_ended(s)))
            continue;
        colon = strchr(s, ':');
        if (!colon) {
            if (!diag_non_ascii(path, linenr, line, s, s + strcspn(s, "#\n"),
                                "line", 4))
                poly_diag(path, linenr, "non-comment line has no ':' separator");
            goto done;
        }
        key_end = colon;
        while (key_end > s && (key_end[-1] == ' ' || key_end[-1] == '\t'))
            key_end--;
        key_len = (size_t)(key_end - s);
        value = colon + 1;
        if (!key_len) {
            poly_diag(path, linenr, "empty field name");
            goto done;
        }
        /* Unknown fields are ignored, so an invisible character pasted into
         * a known name would make the line vanish: "<U+200B>c8:" would drop c8
         * and load a lower-degree polynomial. Every CADO and GGNFS field name
         * is ASCII, so reject any name that is not. */
        if (diag_non_ascii(path, linenr, line, s, key_end, "field name", 10))
            goto done;

#define KEY_IS(k) (key_len == sizeof(k) - 1u && !memcmp(s, (k), sizeof(k) - 1u))
#define DUP_GUARD(flag, name)                                                \
        do {                                                                 \
            if (flag) {                                                      \
                poly_diag(path, linenr, "duplicate %s field", (name));       \
                goto done;                                                   \
            }                                                                \
            (flag) = 1;                                                      \
        } while (0)
/* A value that fails to parse is reported by its invisible character if it
 * has one, and by the caller's message otherwise. A comment after the value
 * may contain anything. */
#define VALUE_FAIL(...)                                                      \
        do {                                                                 \
            if (!diag_non_ascii(path, linenr, line, value,                   \
                                value + strcspn(value, "#\n"),               \
                                s, (int)key_len))                            \
                poly_diag(path, linenr, __VA_ARGS__);                        \
            goto done;                                                       \
        } while (0)

        /* Claim only the coefficient namespace, c<digits>:. Unknown metadata
         * such as CADO's cost: field must fall through like qintsize: and
         * type:, rather than becoming a fatal malformed-coefficient error. */
        if (key_len >= 2u && s[0] == 'c' && s[1] >= '0' && s[1] <= '9') {
            unsigned k = 0;
            size_t z;
            for (z = 1; z < key_len; z++) {
                unsigned d;
                if (s[z] < '0' || s[z] > '9') {
                    poly_diag(path, linenr, "malformed coefficient name");
                    goto done;
                }
                d = (unsigned)(s[z] - '0');
                if (k > (UINT_MAX - d) / 10u) {
                    poly_diag(path, linenr, "coefficient index overflow");
                    goto done;
                }
                k = k * 10u + d;
            }
            if (k > BENCH_MAX_DEGREE) {
                poly_diag(path, linenr,
                          "degree %u exceeds supported maximum %d",
                          k, BENCH_MAX_DEGREE);
                goto done;
            }
            if (seen_c & (1u << k)) {
                poly_diag(path, linenr, "duplicate c%u coefficient", k);
                goto done;
            }
            if (parse_exact_integer(value, P->cs[k], sizeof(P->cs[k]),
                                    &P->c[k]))
                VALUE_FAIL("c%u must be a decimal integer of at most %zu characters",
                           k, sizeof(P->cs[k]) - 1u);
            seen_c |= 1u << k;
        } else if (KEY_IS("skew")) {
            DUP_GUARD(seen_skew, "skew");
            if (parse_finite_value(value, &P->skew) || P->skew <= 0.0)
                VALUE_FAIL("skew must be finite and positive");
        } else if (KEY_IS("Y0")) {
            DUP_GUARD(seen_y0, "Y0");
            if (parse_exact_integer(value, P->y0s, sizeof(P->y0s), &P->y0))
                VALUE_FAIL("Y0 must be a decimal integer of at most %zu characters",
                           sizeof(P->y0s) - 1u);
        } else if (KEY_IS("Y1")) {
            DUP_GUARD(seen_y1, "Y1");
            if (parse_exact_integer(value, P->y1s, sizeof(P->y1s), &P->y1))
                VALUE_FAIL("Y1 must be a decimal integer of at most %zu characters",
                           sizeof(P->y1s) - 1u);
        } else if (KEY_IS("rlim")) {
            DUP_GUARD(seen_rlim, "rlim");
            if (parse_u32_value(value, &P->rlim)) goto bad_u32;
        } else if (KEY_IS("alim")) {
            DUP_GUARD(seen_alim, "alim");
            if (parse_u32_value(value, &P->alim)) goto bad_u32;
        } else if (KEY_IS("lpbr")) {
            DUP_GUARD(seen_lpbr, "lpbr");
            if (parse_u32_value(value, &P->lpbr)) goto bad_u32;
        } else if (KEY_IS("lpba")) {
            DUP_GUARD(seen_lpba, "lpba");
            if (parse_u32_value(value, &P->lpba)) goto bad_u32;
        } else if (KEY_IS("mfbr")) {
            DUP_GUARD(seen_mfbr, "mfbr");
            if (parse_u32_value(value, &P->mfbr)) goto bad_u32;
        } else if (KEY_IS("mfba")) {
            DUP_GUARD(seen_mfba, "mfba");
            if (parse_u32_value(value, &P->mfba)) goto bad_u32;
        } else if (KEY_IS("rlambda")) {
            DUP_GUARD(seen_rlambda, "rlambda");
            if (parse_finite_value(value, &P->rlambda) || P->rlambda < 0.0)
                VALUE_FAIL("rlambda must be finite and nonnegative");
        } else if (KEY_IS("alambda")) {
            DUP_GUARD(seen_alambda, "alambda");
            if (parse_finite_value(value, &P->alambda) || P->alambda < 0.0)
                VALUE_FAIL("alambda must be finite and nonnegative");
        } else {
            /* CADO/GGNFS files carry metadata such as n:. Unknown fields are
             * deliberately ignored, but every recognized field above is
             * parsed strictly and cannot fall through after a partial match. */
        }
        continue;

bad_u32:
        VALUE_FAIL("%.*s must be a complete unsigned 32-bit integer",
                   (int)key_len, s);
#undef VALUE_FAIL
#undef DUP_GUARD
#undef KEY_IS
    }
    if (ferror(f)) {
        poly_diag(path, linenr, "read error");
        goto done;
    }

    if (!seen_skew) {
        poly_diag(path, 0, "missing skew field");
        goto done;
    }
    if (!seen_y0 || !seen_y1) {
        poly_diag(path, 0, "both Y0 and Y1 are required");
        goto done;
    }
    if (exact_decimal_is_zero(P->y0s) && exact_decimal_is_zero(P->y1s)) {
        poly_diag(path, 0, "Y0 and Y1 cannot both be zero");
        goto done;
    }
    for (int k = BENCH_MAX_DEGREE; k >= 0; k--)
        if ((seen_c & (1u << k)) && !exact_decimal_is_zero(P->cs[k])) {
            P->deg = k;
            break;
        }
    if (P->deg < 1) {
        poly_diag(path, 0, "algebraic polynomial must have degree at least 1");
        goto done;
    }
    if (fclose(f)) {
        f = NULL;
        poly_diag(path, 0, "close error");
        goto done;
    }
    f = NULL;
    rc = 0;

done:
    if (f) fclose(f);
    if (rc) {
        memset(P, 0, sizeof(*P));
        P->deg = -1;
    }
    return rc;
}

/* Does f have a root mod 2 -- affine (x = 0 or 1) or projective (2 | c_deg)?
 *
 * If it does not, 2 NEVER divides the algebraic norm, and a factor base with
 * no p = 2 entry is correct rather than truncated. That is not a corner case:
 * the SNFS polynomial x^5 + x^4 - 4x^3 - 3x^2 + 3x + 1 has f(0) = 1, f(1) = -1
 * and an odd leading coefficient, so fbgen emits no p = 2 and the norms are
 * always odd.
 *
 * Parity is read off the EXACT decimal strings, not the doubles: c0 runs to
 * 147 bits on some jobs and the double is good to 53. The last digit is all
 * that is needed, and a leading '-' does not change it. */
int poly_has_root_mod2(const poly_t *P)
{
    /* Initialised, though the loop always assigns both: k runs 0..deg, so the
     * k == 0 and k == P->deg arms each fire once. The compiler cannot see it. */
    int sum = 0, c0_even = 0, lead_even = 0;
    for (int k = 0; k <= P->deg; k++) {
        const char *s = P->cs[k];
        size_t n = strlen(s);
        int odd = n && (s[n - 1] - '0') % 2;   /* empty string = absent = 0 */
        sum ^= odd;
        if (k == 0)       c0_even  = !odd;
        if (k == P->deg)  lead_even = !odd;
    }
    if (c0_even)   return 1;        /* x = 0 is a root      */
    if (!sum)      return 1;        /* x = 1 is a root      */
    if (lead_even) return 1;        /* projective root at 2 */
    return 0;
}

/* Exact decimal coefficient modulo q. The polynomial loader retains decimal
 * strings precisely because the doubles do not carry enough bits for roots or
 * trial division. Keep this small evaluator independent of the fixed-width
 * norm integer: modular reduction never needs to materialise the coefficient. */
static int qsel_dec_mod(const char *s, uint32_t q, uint32_t *out)
{
    uint64_t r = 0;
    int neg = 0, ndigit = 0;

    if (!s || !out || !q) return -1;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-' || *s == '+') { neg = (*s == '-'); s++; }
    while (*s >= '0' && *s <= '9') {
        r = (r * 10u + (uint32_t)(*s - '0')) % q;
        s++;
        ndigit = 1;
    }
    while (*s == ' ' || *s == '\t') s++;
    if (!ndigit || *s) return -1;
    if (neg && r) r = q - r;
    *out = (uint32_t)r;
    return 0;
}

qsel_validate_result_t qsel_validate(qsel_t *sel, const poly_t *P, int side)
{
    uint32_t q;
    uint32_t rho;
    uint64_t acc = 0;
    int deg;

    if (!sel || !P) return QSEL_ERR_POLY;
    if (sel->q < 2 || (sel->q >> 32)) return QSEL_ERR_Q_RANGE;
    q = (uint32_t)sel->q;
    if (!bench_is_prime32(q)) return QSEL_ERR_Q_COMPOSITE;
    if (side != 0 && side != 1) return QSEL_ERR_SIDE;

    /* q is now known nonzero. Work with a canonical local value and publish it
     * only after the complete gate succeeds, so a rejected selection is not
     * partially modified. qlat_build() stores rho in int64_t, which is why the
     * reduction must happen before a valid selection reaches it. */
    rho = (uint32_t)(sel->rho % q);
    deg = side ? P->deg : 1;
    if (deg < 1 || deg > BENCH_MAX_DEGREE) return QSEL_ERR_POLY;

    for (int k = deg; k >= 0; k--) {
        const char *s;
        uint32_t cm = 0;
        if (side) {
            s = P->cs[k];
            if (s[0] && qsel_dec_mod(s, q, &cm)) return QSEL_ERR_POLY;
            /* An absent algebraic coefficient is exactly zero. */
        } else {
            s = k ? P->y1s : P->y0s;
            if (!s[0] || qsel_dec_mod(s, q, &cm)) return QSEL_ERR_POLY;
        }
        acc = (acc * rho + cm) % q;
    }
    if (acc != 0) return QSEL_ERR_NOT_ROOT;
    sel->rho = rho;
    return QSEL_VALID;
}

double job_allowance_bits(double lambda, uint32_t lim)
{
    if (!isfinite(lambda) || lambda <= 0.0 || lim < 2) return 0.0;
    return lambda * (log((double)lim) / log(2.0));
}

/* allowance_bits: how many bits of unfactored cofactor we tolerate before a
 * position stops being a survivor. las derives this from lambda*lpb. */
/* norm_setup's two diagnostic lines are per side PER SPECIAL-Q. That is fine
 * for a single-q run and ruinous for a band: a 378,000-q job printed over a
 * million lines and shredded the \r progress display. The band loop prints the
 * first q's setup and then clears this. */
int norm_verbose = 1;

double las_scale(double maxlog2)
{
    double s;
    if (!isfinite(maxlog2) || maxlog2 <= 0.0) return 0.0;
    s = (255.0 - 1.0) / maxlog2;          /* UCHAR_MAX - LOGNORM_GUARD_BITS */
    if (!isfinite(s) || s <= 0.0 || s > (double)INT32_MAX / 40.0) return 0.0;
    return (double)(int)(s * 40.0) * 0.025;
}

double las_allowance(double maxlog2, double scale, double lambda,
                     unsigned lpb, unsigned mfb)
{
    double r;
    if (!isfinite(maxlog2) || !isfinite(scale) || scale <= 0.0 ||
        !isfinite(lambda) || !lpb)
        return 0.0;
    /* CADO's automatic lambda when none is given. */
    if (lambda <= 0.0) lambda = 0.3 + (double)mfb / (double)lpb;
    r = maxlog2 - 1.0 / scale;            /* LOGNORM_GUARD_BITS / scale */
    if (r > lambda * (double)lpb) r = lambda * (double)lpb;
    return r;
}

/* The survivor allowance we actually want, in bits, derived from THIS tool's
 * behaviour rather than from an imported lambda.
 *
 * A survivor is a position whose approximated remaining log looks small enough
 * to be worth trial dividing. The hard gate downstream is mfb: a cofactor
 * above it is rejected outright, so admitting positions far above mfb buys
 * nothing but work. The only reason to exceed mfb at all is that the survivor
 * test is approximate -- a byte-quantised log against an fp32 norm estimate --
 * so a position whose true cofactor is under mfb can read over it.
 *
 * One byte unit is 1/scale bits, which is the natural granularity of that
 * error, and the floor of 1.5 bits keeps a fine scale from producing a slack
 * so tight that ordinary rounding starts costing relations.
 *
 * Measured, SNFS job side 0 (mfb 88, scale 1.60), 200 special-q:
 *     mfb + 3.8 (91.80, the .job file's rlambda 3.4): 120,317 surv  10,313 rel
 *     mfb + 1.5 (89.50, what this returns)          : 104,615 surv  10,312 rel
 *     mfb + 1.0 (89.00)                             :  99,808 surv  10,312 rel
 *     mfb + 0.0 (88.00)                             :  90,782 surv  10,306 rel
 *     mfb - 3.0 (85.00)                             :  74,849 surv   9,668 rel
 * so 13% of the trial-division input goes away for one relation in ten
 * thousand. Below mfb the cliff is real and fast -- hence a floor, not a cap
 * at mfb exactly.
 *
 * WHY NOT THE .job FILE'S LAMBDA. It is calibrated to a different survivor
 * gate. On the same job and q range, gnfs-lasieve4I14e loses 17.3% of its
 * yield going from 91.8 to 87.5 bits where we lose 0.07% going to 88.0 -- so
 * the number does not transfer, and importing it just inherits another tool's
 * calibration. CADO's automatic (0.3 + mfb/lpb) x lpb is no better a source:
 * on this job it gives 97.3 bits, looser still.
 *
 * Capped by the norm itself: a cofactor cannot exceed the norm it came from,
 * less one guard unit. */
double sieve_allowance(double maxlog2, double scale, unsigned mfb)
{
    double slack, r;
    if (!isfinite(maxlog2) || !isfinite(scale) || scale <= 0.0) return 0.0;
    slack = 2.0 / scale;
    if (slack < 1.5) slack = 1.5;
    r = (double)mfb + slack;
    if (r > maxlog2 - 1.0 / scale) r = maxlog2 - 1.0 / scale;
    return r;
}

void norm_setup(norm_t *N, const poly_t *P, const qlat_t *L,
                int logI, uint32_t J, double scale, int is_sqside)
{
    const double I = (double)(1u << logI);
    /* |a| and |b| are maximised at the corners of the sieve rectangle. Using
     * the max (not the RMS) guarantees |u|,|v| <= 1 everywhere, so no term can
     * overflow no matter where in the region we land. */
    const double A = 0.5 * I * fabs((double)L->a0) + (double)J * fabs((double)L->b0);
    const double B = 0.5 * I * fabs((double)L->a1) + (double)J * fabs((double)L->b1);
    double d[BENCH_NCOEFF], M = 0.0;
    int k, nclamp = 0;

    for (k = 0; k <= P->deg; k++) {
        d[k] = P->c[k] * pow(A, (double)k) * pow(B, (double)(P->deg - k));
        if (fabs(d[k]) > M) M = fabs(d[k]);
    }
    N->deg = P->deg;
    for (k = 0; k <= P->deg; k++) {
        double r = d[k] / M;
        /* below fp32 denormal range the term cannot affect the leading bits.
         * If this fires for a leading coefficient the homogeneous terms are
         * not balanced, which means the q-lattice was reduced without the
         * skew -- see qlat_build.
         *
         * A coefficient that is ZERO in the polynomial is not evidence of
         * that. Sparse SNFS forms are all zeros: x^5 - 6032531 has c1..c4 = 0
         * and flushed four terms on every q. Count only terms that were
         * nonzero and lost, which is the case the check was written for. */
        if (fabs(r) < 1e-37) { N->d[k] = 0.0f; nclamp += (d[k] != 0.0); }
        else N->d[k] = (float)r;
    }
    for (; k < BENCH_NCOEFF; k++) N->d[k] = 0.0f;

    for (k = 0; k <= P->deg; k++) N->dd[k] = d[k] / M;
    for (; k < BENCH_NCOEFF; k++) N->dd[k] = 0.0;
    N->A = A; N->B = B;
    N->a0 = L->a0; N->a1 = L->a1; N->b0 = L->b0; N->b1 = L->b1;

    N->ua  = (float)((double)L->a0 / A);   /* u = ua*i + ub*j */
    N->ub  = (float)((double)L->b0 / A);
    N->va  = (float)((double)L->a1 / B);   /* v = va*i + vb*j */
    N->vb  = (float)((double)L->b1 / B);
    N->log2M = (float)(log(M) / log(2.0));
    N->scale = (float)scale;
    /* ONLY the special-q side's norm carries a factor of q to divide out. The
     * other side's norm is untouched by the special-q. Subtracting log2(q) on
     * both sides makes the non-sq side's threshold ~27 bits too generous and
     * floods the survivor list.
     * Cross-check: las prints log2(maxnorm)=131.86 for side 0 and 196.61 for
     * side 1 here. We compute 131.86 for side 0 (undivided) and 223.6 for
     * side 1, and 223.6 - log2(q) = 196.8. Both match, which is what pins the
     * factor of q to the sq side. */
    N->bias = (float)(is_sqside ? log((double)L->q) / log(2.0) : 0.0);

    /* The unbalanced-terms warning means the q-lattice was reduced without the
     * skew, which is a real defect.
     *
     * It does NOT apply to the degree-1 rational form, which has no balance to
     * lose: G = Y1*a + Y0*b has one huge coefficient and one small one, in
     * whichever orientation the poly generator chose -- input.job has |Y0| = 1
     * against a 136-bit Y1, work/snfs236 has it the other way round -- and the
     * ratio of the two terms is |Y0|B/(|Y1|A), which for this job is ~1e-42. No
     * choice of skew moves that: it is the size of the polynomial, not the
     * shape of the lattice, and skew is fitted to the algebraic side anyway.
     *
     * Dropping the small term is safe, but NOT because of the near-cancellation
     * fallback -- that fallback cannot fire here. Both this file (below) and the
     * device build `aabs` from the already-flushed d[], so with one of two
     * deg-1 terms zeroed, |acc| == aabs identically and `s < TOL*aabs` is never
     * true; norm_acc_fp64 is unreachable on the rational side. What makes it
     * safe is that the estimate is only wrong where the SURVIVING term is
     * smaller than the flushed one, and at a 42-order gap that needs a == 0
     * exactly. A reduced q-lattice has gcd(a0,b0) = 1, so i*a0 + j*b0 == 0 only
     * at i = j = 0, which is outside the region (j >= 1). Zero cells, not few.
     *
     * Named by DEGREE, not by is_sqside. This printed is_sqside in a field
     * labelled "side", which was only ever right while is_sqside == (side==1);
     * --sq-side 0 broke that and the warning then pointed at the wrong
     * polynomial. norm_setup does not know the side index, but it knows the
     * degree, and that identifies the form unambiguously.
     *
     * Deliberately NOT gated on norm_verbose, which the band loop clears after
     * the first q: this is a defect report, not a setup dump, and a lattice that
     * reduces without the skew at q=10^8 has to be able to say so. The zero-term
     * and deg-1 filters above are what make it quiet; the gate would have made
     * it silent. */
    /* RATE-LIMITED, because the caller may not be a band loop. This fires per
     * norm_setup call and is deliberately not gated on norm_verbose (above), so
     * on a job where it fires at all it fires on EVERY q: a long band prints it
     * millions of times, and normscan -- which calls norm_setup twice per
     * sample, 800,000 times at the default --samples -- would bury its own
     * verdict under it, into a shell variable testsieve captures whole. A
     * handful of instances proves the defect; the rest are noise. The counter
     * is per-process and never reset, which is what makes the bound hold. */
    if (nclamp && P->deg > 1) {
        static int nclamp_reported = 0;
        enum { NCLAMP_REPORT_MAX = 5 };
        if (nclamp_reported < NCLAMP_REPORT_MAX) {
            printf("  ** q=%llu algebraic (deg %d): %d normalised term(s)"
                   " flushed to zero -- terms are unbalanced **\n",
                   (unsigned long long)L->q, P->deg, nclamp);
            if (++nclamp_reported == NCLAMP_REPORT_MAX)
                printf("  ** (further unbalanced-term warnings suppressed) **\n");
        }
    }
    if (!norm_verbose) return;
    printf("  norm setup: deg %d, A=%.4e B=%.4e, log2(maxnorm)=%.2f, scale=%.3f%s\n",
           P->deg, A, B, N->log2M - N->bias, N->scale,
           is_sqside ? "  (sq side: log2(q) divided out)" : "");
    printf("  normalised coeffs d[0..%d] =", P->deg);
    for (k = 0; k <= P->deg; k++) printf(" %.3g", N->d[k]);
    printf("\n");
}

/* Host mirror of the device norm evaluation, used by the correctness gate. */
/* Exact-in-fp64 evaluation of the same normalised form. (a,b) are formed in
 * int64, so they are exact; only the polynomial sum rounds, at 2^-53 instead of
 * fp32's 2^-24. */
double norm_acc_fp64(const norm_t *N, int32_t i, uint32_t j)
{
    const double a = (double)((int64_t)i * N->a0 + (int64_t)j * N->b0);
    const double b = (double)((int64_t)i * N->a1 + (int64_t)j * N->b1);
    const double u = a / N->A, v = b / N->B;
    double acc = N->dd[N->deg], vp = 1.0;
    int k;
    for (k = N->deg - 1; k >= 0; k--) { vp *= v; acc = acc * u + N->dd[k] * vp; }
    return acc;
}

double norm_exact_bound_bits(const norm_t *N)
{
    return (double)N->log2M + log2((double)N->deg + 1.0);
}

int norm_fits_exact(const norm_t *N, unsigned bits)
{
    /* log2M is stored as float; a millibit keeps a downward rounding at the
     * exact boundary from admitting a norm that needs one more limb. */
    return norm_exact_bound_bits(N) + 1e-3 < (double)bits;
}

float norm_target_host(const norm_t *N, int32_t i, uint32_t j)
{
    float u = N->ua * (float)i + N->ub * (float)j;
    float v = N->va * (float)i + N->vb * (float)j;
    float acc = N->d[N->deg], vp = 1.0f;
    /* the same Horner on |.|, which bounds the cancellation: the fp32 error is
     * ~deg*2^-24*aabs, so |acc| << aabs means the result is mostly noise */
    float aabs = fabsf(N->d[N->deg]), vpa = 1.0f, av = fabsf(v), au = fabsf(u);
    float s;
    int k;
    for (k = N->deg - 1; k >= 0; k--) {
        vp  *= v;   acc   = acc * u + N->d[k] * vp;
        vpa *= av;  aabs  = aabs * au + fabsf(N->d[k]) * vpa;
    }
    /* Mirror the DEVICE expression exactly -- same float type, same clamp, same
     * log2f -- because this function is the reference the GPU is gated against.
     * Being more accurate here than the kernel would make "0 cells differ"
     * meaningless: it would be comparing two different functions. */
    s = fabsf(acc);
    if (s < NORM_CANCEL_TOL * aabs) s = (float)fabs(norm_acc_fp64(N, i, j));
    if (s < 1e-30f) s = 1e-30f;
    /* las: fb_log(n) = floor(log2(n)*scale + 0.5) */
    return N->scale * (N->log2M + log2f(s) - N->bias);
}

/* Gauss-reduce the lattice generated by (q,0) and (rho,1) to a basis with
 * entries of size ~sqrt(q). This is what las does per special-q. */
void qlat_build(qlat_t *L, uint64_t q, uint64_t rho, double skew)
{
    /* Lagrange/Gauss reduction of <(q,0),(rho,1)>, under the SKEWED norm
     *     |(a,b)|^2 = (a/sqrt(s))^2 + (b*sqrt(s))^2 .
     * The skew is not cosmetic. An unskewed reduction gives |a| ~ |b| ~
     * sqrt(q); the polynomial's coefficients then span 10^39 across the
     * homogeneous terms and the norm is dominated by c0*b^5 alone. With the
     * skew applied, |a| ~ sqrt(q*s) and |b| ~ sqrt(q/s), the terms c_k a^k
     * b^(d-k) come out within an order of magnitude of each other, and
     * log2|F| lands where las expects it. That balance is the entire purpose
     * of the skew parameter.
     *
     * The invariant is that (a0,a1) is the SHORTER vector and (b0,b1) gets
     * reduced against it; getting the swap backwards leaves the basis
     * unreduced, which silently produces a q-lattice of the wrong shape. */
    int64_t a0 = (int64_t)q, a1 = 0, b0 = (int64_t)rho, b1 = 1;
    const double wa = 1.0 / skew, wb = skew;     /* weights on a^2 and b^2 */
    for (int guard = 0; guard < 200; guard++) {
        double na = wa * (double)a0 * a0 + wb * (double)a1 * a1;
        double nb = wa * (double)b0 * b0 + wb * (double)b1 * b1;
        if (nb < na) {                       /* keep a as the shorter vector */
            int64_t t;
            t = a0; a0 = b0; b0 = t; t = a1; a1 = b1; b1 = t;
            na = nb;
        }
        {
            double dot = wa * (double)a0 * b0 + wb * (double)a1 * b1;
            double mu = dot / na;
            int64_t m = (int64_t)(mu < 0 ? mu - 0.5 : mu + 0.5);
            if (m == 0) break;
            b0 -= m * a0; b1 -= m * a1;
        }
    }
    L->a0 = a0; L->a1 = a1; L->b0 = b0; L->b1 = b1; L->q = q; L->rho = rho;
}
