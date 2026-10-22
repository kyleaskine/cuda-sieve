/* GGNFS .afb.0 loader. Format decoded in oracle/ggnfs_afb_format.txt. */
#include "bench.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <stdarg.h>

#define FB_VALIDATION_COOKIE 0x46425631u  /* "FBV1" */

typedef enum {
    FB_MODULUS_INVALID = 0,
    FB_MODULUS_PRIME = 1,
    FB_MODULUS_PROPER_POWER = 2
} fb_modulus_kind_t;

static int fb_validation_error(fb_t *fb, const char *source, uint32_t index,
                               const char *fmt, ...)
{
    va_list ap;
    if (fb) fb->validation_cookie = 0;
    errno = EINVAL;
    if (!source) return -1;
    fprintf(stderr, "fb_validate(%s): ", source);
    if (index != UINT32_MAX) fprintf(stderr, "entry %u: ", index);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return -1;
}

/* q is already known to be composite. Return whether it is p^k for one prime
 * p and k >= 2. The first divisor found by ascending trial division is prime,
 * so removing it completely is sufficient. This path is rare in a valid
 * factor base: only the explicitly marked power entries pay for it. */
static int composite_is_prime_power(uint32_t q)
{
    uint32_t d;
    if (q < 4) return 0;
    if ((q & 1u) == 0) {
        do q >>= 1; while ((q & 1u) == 0);
        return q == 1;
    }
    for (d = 3; (uint64_t)d * d <= (uint64_t)q; d += 2) {
        if (q % d) continue;
        do q /= d; while (q % d == 0);
        return q == 1;
    }
    return 0;
}

int fb_load(const char *path, fb_t *fb)
{
    FILE *f = fopen(path, "rb");
    int64_t sz;
    uint32_t n;
    size_t got;

    memset(fb, 0, sizeof(*fb));
    if (!f) { fprintf(stderr, "fb_load: cannot open %s\n", path); return -1; }
    /* 64-bit: `long` is 32 bits on Windows, where a wrapped size would pass
     * the header-consistency test below against a bogus bound. */
    if (bench_seek_end(f) || (sz = bench_tell(f)) < 0 || bench_seek(f, 0)) {
        fprintf(stderr, "fb_load: cannot size %s\n", path);
        fclose(f); return -1;
    }

    if (fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }
    if (n == 0 || (int64_t)(4 * (1 + 2ull * n)) > sz) {
        fprintf(stderr, "fb_load: header count %u inconsistent with size %lld\n",
                n, (long long)sz);
        fclose(f); return -1;
    }

    fb->n      = n;
    fb->maxbits = 1;                 /* .afb.0 contains base primes only */
    fb->primes = (uint32_t *)malloc((size_t)n * 4);
    fb->roots  = (uint32_t *)malloc((size_t)n * 4);
    /* .afb.0 has no prime powers at all -- that is one of the two reasons the
     * CADO loader exists. Say so explicitly with a zeroed flag array rather
     * than leaving it NULL: NULL makes FB_ISPOW fall back to a primality test
     * per entry, which cost 7.8 s on this file and applied to the DEFAULT
     * invocation, the one path not exercised while fixing finding 29. */
    fb->ispow  = (uint8_t *)calloc((size_t)n, 1);
    if (!fb->primes || !fb->roots || !fb->ispow) {
        fclose(f);
        fb_free(fb);
        return -1;
    }

    got  = fread(fb->primes, 4, n, f);
    got += fread(fb->roots,  4, n, f);
    fclose(f);
    if (got != 2ull * n) {
        fprintf(stderr, "fb_load: short read\n");
        fb_free(fb);
        return -1;
    }
    /* .afb.0 has no way to mark prime powers. Requiring actual primes here is
     * therefore stricter than the transform's prime-power rule on purpose: a
     * power accepted as a base prime would retain ispow == 0 and be routed to
     * the bucket walk even when its transformed solutions are row-confined. */
    if (fb_validate(fb, FB_VALIDATE_EXTERNAL_PRIMES, path) != 0) {
        fb_free(fb);
        return -1;
    }
    return 0;
}

void fb_free(fb_t *fb)
{
    if (!fb) return;
    free(fb->primes); free(fb->roots); free(fb->logp); free(fb->ispow);
    memset(fb, 0, sizeof(*fb));
}

/* Is q a prime power (including a prime)?
 *
 * Primality first, by Miller-Rabin: a factor base is almost entirely primes,
 * and trial division to sqrt(q) on 11.5M of them costs minutes. Only the
 * composites -- a couple of hundred -- pay for a factor search, and their base
 * prime is at most sqrt(q), so the search is short. */
int fb_is_prime_power(uint32_t q)
{
    if (q < 2) return 0;
    if (bench_is_prime32(q)) return 1;
    return composite_is_prime_power(q);
}

/* p^k with k >= 2. These are the entries whose transform can come out
 * row-confined (g > 1), which the bucket walk cannot express. */
int fb_is_proper_power(uint32_t q)
{
    return q >= 4 && !bench_is_prime32(q) && composite_is_prime_power(q);
}

/* Every modulus in a factor base must be a prime power: the root transform's
 * "solutions exist only when g | j" step depends on it (see plattice.cuh), and
 * a composite modulus with two prime factors would be silently mis-placed
 * rather than rejected. makefb and rfb_build both satisfy this; this is the
 * gate that notices if a hand-edited or third-party file does not. Returns the
 * first offending index, or -1 if the base is clean. */
int32_t fb_check_prime_powers(const fb_t *fb)
{
    uint32_t i;
    for (i = 0; i < fb->n; i++)
        if (!fb_is_prime_power(fb->primes[i])) return (int32_t)i;
    return -1;
}

int fb_is_transform_validated(const fb_t *fb)
{
    return fb && fb->validation_cookie == FB_VALIDATION_COOKIE;
}

/* Validate the host representation before any entry can reach
 * pl_transform_enc(). External factor bases are checked independently of the
 * loader's interpretation: this is what catches a malformed even composite
 * such as 6 or 12 before pl_invmod_any() mistakes it for a power of two.
 *
 * A deterministic Miller-Rabin test per ideal is needlessly expensive because
 * a polynomial may have several roots for one modulus. We classify each
 * DISTINCT q once. For normal, dense factor bases below 2^28, building one
 * Eratosthenes prime list is substantially faster than millions of modular
 * exponentiations; sparse or unusually high-q inputs use Miller-Rabin instead
 * so a tiny malicious file cannot force a sieve over almost 2^32 integers. */
int fb_validate(fb_t *fb, fb_validate_policy_t policy, const char *source)
{
    const uint32_t sieve_q_limit = 1u << 28;
    const size_t sieve_unique_min = 1u << 16;
    const uint32_t sieve_max_span_per_modulus = 256u;
    uint32_t i, prev = 0, maxq = 0;
    size_t nunique = 0;
    uint32_t *prime_list = NULL;
    size_t nprime = 0, prime_pos = 0;
    uint32_t classified_q = 0;
    fb_modulus_kind_t classified_kind = FB_MODULUS_INVALID;
    int have_classified = 0;

    if (!fb) {
        errno = EINVAL;
        if (source) fprintf(stderr, "fb_validate(%s): null factor base\n", source);
        return -1;
    }
    fb->validation_cookie = 0;

    if (policy != FB_VALIDATE_EXTERNAL_PRIMES &&
        policy != FB_VALIDATE_EXTERNAL_PRIME_POWERS &&
        policy != FB_VALIDATE_GENERATED_PRIME_POWERS)
        return fb_validation_error(fb, source, UINT32_MAX,
                                   "unknown validation policy %d", (int)policy);
    if (fb->n == 0)
        return fb_validation_error(fb, source, UINT32_MAX,
                                   "empty factor base");
    if (!fb->primes || !fb->roots)
        return fb_validation_error(fb, source, UINT32_MAX,
                                   "missing modulus or root array");
    if (policy == FB_VALIDATE_GENERATED_PRIME_POWERS && !fb->ispow)
        return fb_validation_error(fb, source, UINT32_MAX,
                                   "generated factor base has no ispow array");

    /* Structural pass. It is deliberately separate from classification so we
     * know whether the fast sieve is appropriate before spending any primality
     * work, and so malformed ordering/root encodings fail immediately. */
    for (i = 0; i < fb->n; i++) {
        const uint32_t q = fb->primes[i];
        const uint32_t r = fb->roots[i];
        if (q < 2)
            return fb_validation_error(fb, source, i,
                                       "modulus %u is below 2", q);
        if (i && q < prev)
            return fb_validation_error(fb, source, i,
                                       "modulus %u is below previous modulus %u",
                                       q, prev);
        if ((uint64_t)r >= 2ull * (uint64_t)q)
            return fb_validation_error(fb, source, i,
                                       "root encoding %u is outside [0, 2*q) for q=%u",
                                       r, q);
        if (fb->ispow && fb->ispow[i] > 1)
            return fb_validation_error(fb, source, i,
                                       "ispow value %u is not boolean",
                                       (unsigned)fb->ispow[i]);
        if (!i || q != prev) {
            nunique++;
            maxq = q;
        }
        prev = q;
    }

    if (policy != FB_VALIDATE_GENERATED_PRIME_POWERS &&
        nunique >= sieve_unique_min && maxq <= sieve_q_limit &&
        (uint64_t)maxq <= (uint64_t)nunique * sieve_max_span_per_modulus) {
        prime_list = prime_list_build(maxq, &nprime);
        /* Allocation failure is not a validation failure: deterministic
         * Miller-Rabin remains a bounded-memory fallback. */
    }

    for (i = 0; i < fb->n; i++) {
        const uint32_t q = fb->primes[i];
        const int flagged_power = fb->ispow ? !!fb->ispow[i] : 0;
        int expected_power;

        if (!have_classified || q != classified_q) {
            fb_modulus_kind_t kind;

            if (policy == FB_VALIDATE_GENERATED_PRIME_POWERS) {
                /* The zero-flag entries must already have an independent
                 * primality proof inside the in-process generator.  CPU
                 * builders get that from prime_list_build(); the GPU builder
                 * independently Miller-Rabin-checks every root-bearing q=p
                 * before it can be emitted.  Reclassifying the full stream
                 * here would re-sieve a range this process just proved.
                 * Proper powers are few and remain independently checked so a
                 * bad flag cannot route a mixed composite to the lattice
                 * transform. File input never uses this policy. */
                kind = flagged_power
                    ? (composite_is_prime_power(q)
                       ? FB_MODULUS_PROPER_POWER : FB_MODULUS_INVALID)
                    : FB_MODULUS_PRIME;
            } else {
                int isprime;
                if (prime_list) {
                    while (prime_pos < nprime && prime_list[prime_pos] < q)
                        prime_pos++;
                    isprime = prime_pos < nprime && prime_list[prime_pos] == q;
                } else {
                    isprime = bench_is_prime32(q);
                }
                if (isprime) kind = FB_MODULUS_PRIME;
                else if (composite_is_prime_power(q))
                    kind = FB_MODULUS_PROPER_POWER;
                else
                    kind = FB_MODULUS_INVALID;
            }

            if (kind == FB_MODULUS_INVALID) {
                free(prime_list);
                return fb_validation_error(fb, source, i,
                    "modulus %u is neither prime nor a power of one prime", q);
            }
            if (policy == FB_VALIDATE_EXTERNAL_PRIMES &&
                kind != FB_MODULUS_PRIME) {
                free(prime_list);
                return fb_validation_error(fb, source, i,
                    ".afb.0 modulus %u is a proper prime power, not a base prime", q);
            }
            classified_q = q;
            classified_kind = kind;
            have_classified = 1;
        }

        expected_power = classified_kind == FB_MODULUS_PROPER_POWER;
        if (fb->ispow && flagged_power != expected_power) {
            free(prime_list);
            return fb_validation_error(fb, source, i,
                "ispow=%d disagrees with modulus %u (%s)",
                flagged_power, q, expected_power ? "proper prime power" : "prime");
        }
    }

    free(prime_list);
    fb->validation_cookie = FB_VALIDATION_COOKIE;
    return 0;
}

int fb_fill_logp(fb_t *fb, double scale)
{
    uint32_t i;
    uint8_t *logp;

    if (!fb || (fb->n && !fb->primes) ||
        !isfinite(scale) || scale <= 0.0) {
        errno = EINVAL;
        return -1;
    }
    if (fb->logp) return 0;
    logp = (uint8_t *)malloc(fb->n ? fb->n : 1);
    if (!logp) return -1;
    for (i = 0; i < fb->n; i++) {
        if (fb_log_delta_checked(fb->primes[i], 1, 0, scale, &logp[i])) {
            fprintf(stderr,
                    "fb_fill_logp: modulus %u at entry %u has no finite"
                    " 8-bit log at scale %.17g\n",
                    fb->primes[i], i, scale);
            free(logp);
            errno = ERANGE;
            return -1;
        }
    }
    fb->logp = logp;
    return 0;
}

/* Cut a factor base into constant-log slices. Bucket records encode the slice
 * ID in 16 bits, so 65,536 slices is the hard format limit. Count first: this
 * both permits an exactly-sized allocation and guarantees that malformed or
 * unsorted input cannot write past either the slice-ID range or the log table.
 *
 * A slice starts only where the log changes, so a factor base sorted by
 * modulus cuts into one slice per distinct log: 15-26 per side on every job
 * measured, c135 to 2,1147+, whatever its size. It also used to cut every
 * 262,144 entries, "the original layout used by the CUDA path", which nothing
 * reads: the slice ID is only ever turned back into a log, by k_apply's table
 * and by the CPU replay behind --verify (verify_cpu.c). Bucket records carry
 * it in their high bits; the side-0 resieve and the harness's layout B index
 * the factor base their own way. That cut made the count grow with the
 * factor base, ~entries/262,144 + 14, and past 128 slices the padded log
 * table cost k_apply its third block per SM: 2,1147+ (alim = rlim = 600M,
 * 138 and 137 slices) ran apply 16-18% slower on a 5070 (finding 128).
 * fbtest's slice-cut case keeps it that way. The second pass is deliberately
 * identical to the counting pass; output handles stay fail-closed until every
 * size check and allocation has succeeded. */
int32_t fb_build_slices(const fb_t *fb, uint16_t **slice_out,
                        uint16_t **logp_tab_out, uint32_t *nslice_pow2)
{
    const uint32_t max_slices = (uint32_t)UINT16_MAX + 1u;
    uint16_t *slice = NULL, *tab = NULL;
    uint32_t k, ns = 0, counted, p2 = 1;
    int cur = -1;

    if (!slice_out || !logp_tab_out || !nslice_pow2) {
        errno = EINVAL;
        return -1;
    }
    *slice_out = NULL;
    *logp_tab_out = NULL;
    *nslice_pow2 = 0;

    if (!fb || !fb->logp || fb->n == 0) {
        errno = EINVAL;
        return -1;
    }

    for (k = 0; k < fb->n; k++) {
        const int lp = fb->logp[k];
        if (lp != cur) {
            if (ns == max_slices) {
                errno = EOVERFLOW;
                return -1;
            }
            cur = lp;
            ns++;
        }
    }
    counted = ns;

    while (p2 < counted) p2 <<= 1;
    {
        const size_t slice_bytes = (size_t)fb->n * sizeof(*slice);
        const size_t tab_bytes = (size_t)p2 * sizeof(*tab);
        if (slice_bytes / sizeof(*slice) != fb->n ||
            tab_bytes / sizeof(*tab) != p2) {
            errno = EOVERFLOW;
            return -1;
        }
    }

    slice = (uint16_t *)malloc((size_t)fb->n * sizeof(*slice));
    tab = (uint16_t *)calloc((size_t)p2, sizeof(*tab));
    if (!slice || !tab) {
        free(slice);
        free(tab);
        errno = ENOMEM;
        return -1;
    }

    ns = 0;
    cur = -1;
    for (k = 0; k < fb->n; k++) {
        const int lp = fb->logp[k];
        if (lp != cur) {
            cur = lp;
            tab[ns++] = (uint16_t)lp;
        }
        slice[k] = (uint16_t)(ns - 1);
    }

    *slice_out = slice;
    *logp_tab_out = tab;
    *nslice_pow2 = p2;
    return (int32_t)counted;
}

/* Keep the bucketed window [bkthresh, fb_bound).
 *
 * Projective entries (root >= p) are KEPT. Dropping them used to look safe on
 * the argument that a bucketed p exceeds J, so a projective ideal can only hit
 * row j = 0 -- but that argument confuses the two coordinates. The condition is
 * on b, and b = i*a1 + j*b1; for a special-q of this size the reduced basis has
 * b-components of order sqrt(q/skew) ~ 1, and on the q = 120000053 lattice it
 * is exactly b = i. So a projective ideal hits i == 0 (mod p) on EVERY row:
 * 16,384 positions per entry here, not one row's worth. Side 0 has two of them
 * (38321 and 5746453, both dividing Y1). Let the transform decide -- it now
 * handles the encoding, and reports what it genuinely cannot walk.
 *
 * Proper prime powers are EXCLUDED and line-sieved instead, whatever their
 * size. Their transform can be row-confined (g > 1), which is not a plat_t
 * walk, and the bucket path would have to drop them: measured, q = 2^15 sits
 * exactly at the default bkthresh and lost its 16,384 positions that way. The
 * small sieve carries the row divisor natively, so routing every power there
 * removes the whole class rather than the one instance -- which matters the
 * moment anyone raises --maxbits and creates powers well above bkthresh. It
 * costs ~200 extra entries in the small tier, each with a large modulus and so
 * at most one hit per region. */
int fb_restrict(fb_t *fb, uint32_t bkthresh, uint32_t fb_bound)
{
    uint32_t i, k = 0;
    if (!fb_is_transform_validated(fb)) {
        fprintf(stderr, "fb_restrict: refusing an unvalidated factor base\n");
        errno = EINVAL;
        return -1;
    }
    for (i = 0; i < fb->n; i++) {
        uint32_t p = fb->primes[i], r = fb->roots[i];
        if (p < bkthresh || p >= fb_bound) continue;
        if (FB_ISPOW(fb, i)) continue;            /* line-sieved instead */
        fb->primes[k] = p; fb->roots[k] = r;
        if (fb->logp)  fb->logp[k]  = fb->logp[i];
        if (fb->ispow) fb->ispow[k] = fb->ispow[i];
        k++;
    }
    fb->n = k;
    return 0;
}

/* Copy out the entries the bucket path does not take: everything below
 * bkthresh, plus every proper prime power at any size (see fb_restrict).
 * These are line-sieved per region.
 *
 * Projective entries (r >= p) are KEPT here: the four with p < 12 account for
 * ~0.77*A updates between them, more than the entire bucket-sieve volume, and
 * dropping them is not an option. The entries are already ascending in p, and
 * appending the powers in the same pass preserves that. */
int fb_split_small(const fb_t *fb, uint32_t bkthresh, fb_t *small)
{
    uint32_t i, k = 0, cap = 0;
    if (!small || small == fb) {
        errno = EINVAL;
        return -1;
    }
    memset(small, 0, sizeof(*small));
    if (!fb_is_transform_validated(fb)) {
        fprintf(stderr, "fb_split_small: refusing an unvalidated factor base\n");
        errno = EINVAL;
        return -1;
    }
    small->maxbits = fb->maxbits;
    if (getenv("TD_DUMP_SMALL")) {
        uint32_t n2 = 0, np = 0;
        for (i = 0; i < fb->n; i++) {
            if (fb->primes[i] == 2) n2++;
            if (FB_ISPOW(fb, i)) np++;
        }
        fprintf(stderr, "fb_split_small: input fb n=%u, p==2 rows=%u, ispow rows=%u,"
                        " first p=%u (root %u), bkthresh=%u\n",
                fb->n, n2, np, fb->n ? fb->primes[0] : 0,
                fb->n ? fb->roots[0] : 0, bkthresh);
    }
    for (i = 0; i < fb->n; i++)
        if (fb->primes[i] < bkthresh || FB_ISPOW(fb, i)) cap++;
    small->primes = (uint32_t *)malloc((size_t)(cap ? cap : 1) * 4);
    small->roots  = (uint32_t *)malloc((size_t)(cap ? cap : 1) * 4);
    if (!small->primes || !small->roots) goto nomem;
    if (fb->logp) {
        small->logp = (uint8_t *)malloc((size_t)(cap ? cap : 1));
        if (!small->logp) goto nomem;
    }
    if (fb->ispow) {
        small->ispow = (uint8_t *)malloc((size_t)(cap ? cap : 1));
        if (!small->ispow) goto nomem;
    }
    for (i = 0; i < fb->n; i++) {
        if (fb->primes[i] >= bkthresh && !FB_ISPOW(fb, i)) continue;
        small->primes[k] = fb->primes[i];
        small->roots[k]  = fb->roots[i];
        if (fb->logp)  small->logp[k]  = fb->logp[i];
        if (fb->ispow) small->ispow[k] = fb->ispow[i];
        k++;
    }
    small->n = k;
    /* fb_split_small() rejects an unvalidated source above. Taking an ordered
     * subsequence cannot invalidate its modulus, root, ordering, or power
     * classification, so the subset inherits the source's validation proof. */
    small->validation_cookie = fb->validation_cookie;
    return 0;

nomem:
    fb_free(small);
    errno = ENOMEM;
    return -1;
}

/* Largest PRIME that actually reaches the direct-test table. Proper prime
 * powers live in the small FB too, but td_fill_small deliberately skips them;
 * including one here would make slab planning needlessly pessimistic. Keeping
 * this next to fb_split_small makes the bkthresh -> small-FB -> pmax path one
 * shared implementation for production and the CPU integration gate. */
uint32_t fb_max_td_prime(const fb_t *fb)
{
    uint32_t pmax = 0, k;
    if (!fb) return 0;
    for (k = 0; k < fb->n; k++)
        if (!FB_ISPOW(fb, k) && fb->primes[k] > pmax) pmax = fb->primes[k];
    return pmax;
}
