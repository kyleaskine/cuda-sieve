/* tuned.c -- the saved-tuning file: parse, look up, judge. See tuned.h.
 *
 * Plain C with no CUDA and no bench.h, so tunedtest can gate every rule on a
 * card-less box. The ranges in tuned.h mirror bench.h's; bench_main.cu
 * static_asserts that they agree. */
#include "tuned.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void tuned_card_defaults(int cc_major, int cc_minor, int logI,
                         tuned_card_defaults_t *d)
{
    memset(d, 0, sizeof *d);
    if (cc_major == 8 && cc_minor == 9)
        d->ss_warp_cut = TUNED_CARD_SS_WARP_CUT_SM89;
    if (cc_major == 8 && (cc_minor == 6 || cc_minor == 9) &&
        logI == TUNED_CARD_FILL_LOGI)
        d->fill_blocks = TUNED_CARD_FILL_BLOCKS;
    if (cc_major == 12 && cc_minor == 0) {
        d->slab_fill_ahead = 1;
        d->sparse_apply = 1;
    }
}

/* Flags that only name the job, the band, the outputs and the device: a run
 * carrying nothing else is the plain workload an entry is tuned on. An
 * ALLOWLIST, so a flag added to bench later counts as an override until
 * somebody decides otherwise; where a flag sits in --help plays no part. */
static const char *const TUNED_BASE_OK[] = {
    "--pipeline", "--cofactor", "--poly", "--fb1", "--cadofb",
    "--logI", "--J", "--region", "--sq-side",
    "--rlim", "--alim", "--maxbits", "--lpb", "--lpb0", "--mfb", "--mfb0",
    "--lambda0", "--lambda1",
    "--qrange", "--qlist", "--q", "--rho", "--nq", "--target-rels",
    "--relations", "--log", "--log-every", "--restart", "--stop-file",
    "--watchdog", "--watchdog-kill", "--watchdog-log", "--verbose-q",
    "--device", "--tuned", "--no-tuned", "--auto-params",
    NULL
};

/* The families' flags (tuned.h), and the knobs the file may carry. */
static const struct { const char *flag; unsigned fam; int tunable; } FAMILY[] = {
    { "--fill-blocks",         TUNED_F_FILL_BLOCKS, 1 },
    { "--ss-block-cut",        TUNED_F_SS,          1 },
    { "--ss-warp-cut",         TUNED_F_SS,          1 },
    { "--no-smallsieve",       TUNED_F_SS,          0 },
    { "--fill-gate-shift",     TUNED_F_GATE_SHIFT,  1 },
    { "--fill-gate",           TUNED_F_GATE_SHIFT,  0 },
    { "--no-fill-gate",        TUNED_F_GATE_SHIFT,  0 },
    { "--fill-concurrent",     TUNED_F_GATE_SHIFT,  0 },
    { "--no-fill-merge",       TUNED_F_MERGE,       1 },
    { "--fill-merge",          TUNED_F_MERGE,       0 },
    { "--fill-merge-cut",      TUNED_F_MERGE,       0 },
    { "--slab-fill-ahead",     TUNED_F_SFA,         1 },
    { "--no-slab-fill-ahead",  TUNED_F_SFA,         1 },
    { "--slab-fill-blocks",    TUNED_F_SFA,         1 },
    { NULL, 0, 0 }
};

static void put(char *buf, size_t n, const char *fmt, ...)
{
    va_list ap;
    if (!n) return;
    va_start(ap, fmt);
    vsnprintf(buf, n, fmt, ap);
    va_end(ap);
}

/* Append to a "; "-separated list. */
static void note(char *buf, size_t n, const char *fmt, ...)
{
    size_t k = strlen(buf);
    va_list ap;
    if (k + 3 >= n) return;
    if (k) { memcpy(buf + k, "; ", 3); k += 2; }
    va_start(ap, fmt);
    vsnprintf(buf + k, n - k, fmt, ap);
    va_end(ap);
}

static void trim(char *s)
{
    size_t a = 0, b = strlen(s);
    while (s[a] == ' ' || s[a] == '\t') a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' ||
                     s[b - 1] == '\r' || s[b - 1] == '\n')) b--;
    memmove(s, s + a, b - a);
    s[b - a] = 0;
}

/* A whole decimal in [lo, hi], nothing else. */
static int num(const char *s, long long lo, long long hi, long long *out)
{
    char *end = NULL;
    long long v;
    if (!*s || !(isdigit((unsigned char)*s) || *s == '-')) return -1;
    errno = 0;
    v = strtoll(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return -1;
    *out = v;
    return 0;
}

int tuned_default_path(char *buf, size_t n)
{
#ifdef _WIN32
    const char *a = getenv("APPDATA");
    if (!a || !*a) return -1;
    put(buf, n, "%s\\cuda-sieve\\tuned.txt", a);
#else
    const char *x = getenv("XDG_CONFIG_HOME"), *h = getenv("HOME");
    /* the XDG spec: a relative value is invalid and ignored */
    if (x && x[0] == '/') put(buf, n, "%s/cuda-sieve/tuned.txt", x);
    else if (h && *h)     put(buf, n, "%s/.config/cuda-sieve/tuned.txt", h);
    else return -1;
#endif
    return 0;
}

static void knobs_clear(tuned_knobs_t *k)
{
    memset(k, 0, sizeof *k);
    k->fill_gate_shift = -1;
    k->slab_fill_ahead = -1;
}

/* Append " word" (no leading space on the first) to a space-separated list;
 * 0, or -1 if it does not fit. */
static int cat(char *buf, size_t n, const char *fmt, ...)
{
    size_t k = strlen(buf);
    va_list ap;
    int w;
    if (k + 2 >= n) return -1;
    if (k) buf[k++] = ' ';
    va_start(ap, fmt);
    w = vsnprintf(buf + k, n - k, fmt, ap);
    va_end(ap);
    if (w < 0 || (size_t)w >= n - k) { buf[k ? k - 1 : 0] = 0; return -1; }
    return 0;
}

void tuned_format_flags(const tuned_knobs_t *k, char *buf, size_t n)
{
    buf[0] = 0;
    if (k->fill_blocks)          cat(buf, n, "--fill-blocks %d", k->fill_blocks);
    if (k->ss_block_cut)         cat(buf, n, "--ss-block-cut %d", k->ss_block_cut);
    if (k->ss_warp_cut)          cat(buf, n, "--ss-warp-cut %d", k->ss_warp_cut);
    if (k->fill_gate_shift >= 0) cat(buf, n, "--fill-gate-shift %d", k->fill_gate_shift);
    if (k->no_fill_merge)        cat(buf, n, "--no-fill-merge");
    if (k->slab_fill_ahead == 0) cat(buf, n, "--no-slab-fill-ahead");
    if (k->slab_fill_ahead == 1) cat(buf, n, "--slab-fill-ahead");
    if (k->slab_fill_blocks)     cat(buf, n, "--slab-fill-blocks %d", k->slab_fill_blocks);
}

int tuned_parse_flags(const char *flags, tuned_knobs_t *k, char *err, size_t n)
{
    char tmp[512];
    char *p, *tok[64];
    int nt = 0;
    knobs_clear(k);
    if (strlen(flags) >= sizeof tmp) { put(err, n, "flags too long"); return -1; }
    strcpy(tmp, flags);
    for (p = tmp; *p; ) {
        while (*p == ' ' || *p == '\t') *p++ = 0;
        if (!*p) break;
        if (nt == 64) { put(err, n, "too many flags"); return -1; }
        tok[nt++] = p;
        while (*p && *p != ' ' && *p != '\t') p++;
    }
    for (int i = 0; i < nt; i++) {
        const char *f = tok[i];
        long long v;
        int *dst = NULL;
        long long lo = 0, hi = 0;
        if (!strcmp(f, "--no-fill-merge")) {
            if (k->no_fill_merge) goto dup;
            k->no_fill_merge = 1; continue;
        }
        if (!strcmp(f, "--slab-fill-ahead") || !strcmp(f, "--no-slab-fill-ahead")) {
            if (k->slab_fill_ahead >= 0) {
                put(err, n, "--slab-fill-ahead given twice or both ways");
                return -1;
            }
            k->slab_fill_ahead = f[2] == 's'; continue;
        }
        if (!strcmp(f, "--fill-blocks"))      { dst = &k->fill_blocks;      lo = 1; hi = TUNED_BLOCKS_MAX; }
        else if (!strcmp(f, "--ss-block-cut")) { dst = &k->ss_block_cut;    lo = TUNED_SS_BLOCK_MIN; hi = TUNED_SS_BLOCK_MAX; }
        else if (!strcmp(f, "--ss-warp-cut"))  { dst = &k->ss_warp_cut;     lo = TUNED_SS_BLOCK_MIN + 1; hi = TUNED_SS_WARP_MAX; }
        else if (!strcmp(f, "--fill-gate-shift")) { dst = &k->fill_gate_shift; lo = 0; hi = 5; }
        else if (!strcmp(f, "--slab-fill-blocks")) { dst = &k->slab_fill_blocks; lo = 1; hi = TUNED_BLOCKS_MAX; }
        else { put(err, n, "%s is not a knob this file may set", f); return -1; }
        if (i + 1 >= nt) { put(err, n, "%s has no value", f); return -1; }
        if (num(tok[++i], lo, hi, &v)) {
            put(err, n, "%s %s: want a whole number in [%lld, %lld]", f, tok[i], lo, hi);
            return -1;
        }
        if (dst == &k->fill_gate_shift ? *dst >= 0 : *dst != 0) goto dup;
        *dst = (int)v;
        continue;
dup:
        put(err, n, "%s given twice", f);
        return -1;
    }
    if (k->slab_fill_blocks && k->slab_fill_ahead != 1) {
        put(err, n, "--slab-fill-blocks needs --slab-fill-ahead");
        return -1;
    }
    if (k->ss_block_cut && k->ss_warp_cut && k->ss_warp_cut <= k->ss_block_cut) {
        put(err, n, "--ss-warp-cut %d is not above --ss-block-cut %d",
            k->ss_warp_cut, k->ss_block_cut);
        return -1;
    }
    return 0;
}

int tuned_parse_line(const char *line, tuned_entry_t *e, char *err, size_t n)
{
    static const char *const NAME[12] = {
        "card", "cc", "SMs", "logI", "J", "region", "nslab", "side",
        "alim", "rlim", "policy", "flags"
    };
    char tmp[1024], *f[13], *p;
    long long v[11];
    int k = 0;
    memset(e, 0, sizeof *e);
    if (strlen(line) >= sizeof tmp) { put(err, n, "line too long"); return -1; }
    strcpy(tmp, line);
    f[k++] = tmp;
    for (p = tmp; *p && k < 13; p++)
        if (*p == '|') { *p = 0; f[k++] = p + 1; }
    if (k < 13) {
        put(err, n, "%d fields, want 13 (card|cc|SMs|logI|J|region|nslab|side"
            "|alim|rlim|policy|flags|note)", k);
        return -1;
    }
    for (int i = 0; i < 13; i++) trim(f[i]);
    if (!f[0][0]) { put(err, n, "empty card name"); return -1; }
    if (strlen(f[0]) >= sizeof e->card) { put(err, n, "card name too long"); return -1; }
    strcpy(e->card, f[0]);
    {
        char *dot = strchr(f[1], '.');
        long long a, b;
        if (!dot) { put(err, n, "cc %s: want MAJOR.MINOR", f[1]); return -1; }
        *dot = 0;
        if (num(f[1], 1, 99, &a) || num(dot + 1, 0, 99, &b)) {
            put(err, n, "cc: want MAJOR.MINOR"); return -1;
        }
        e->cc_major = (int)a; e->cc_minor = (int)b;
    }
    {
        static const long long LO[9] = { 1, 2, 1, 1, 1, 0, 1, 1, 0 };
        static const long long HI[9] = { 100000, 20, 0x7fffffffLL, 30, 0x7fffffffLL,
                                    1, 0xffffffffLL, 0xffffffffLL, 1000000 };
        for (int i = 0; i < 9; i++)
            if (num(f[2 + i], LO[i], HI[i], &v[i])) {
                put(err, n, "%s %s: want a whole number in [%lld, %lld]",
                    NAME[2 + i], f[2 + i], LO[i], HI[i]);
                return -1;
            }
    }
    e->sms = (int)v[0]; e->logI = (int)v[1]; e->J = (uint32_t)v[2];
    e->region = (int)v[3]; e->nslab = (int)v[4]; e->side = (int)v[5];
    e->alim = (uint32_t)v[6]; e->rlim = (uint32_t)v[7]; e->policy = (int)v[8];
    if (tuned_parse_flags(f[11], &e->k, err, n)) return -1;
    tuned_format_flags(&e->k, e->flags, sizeof e->flags);
    put(e->note, sizeof e->note, "%s", f[12]);
    return 0;
}

static int family_of(const char *flag, unsigned *fam, int *tunable)
{
    for (int i = 0; FAMILY[i].flag; i++)
        if (!strcmp(flag, FAMILY[i].flag)) {
            *fam = FAMILY[i].fam; *tunable = FAMILY[i].tunable;
            return 1;
        }
    return 0;
}

unsigned tuned_explicit_mask(int argc, char *const argv[])
{
    unsigned m = 0, fam;
    int tunable;
    for (int i = 1; i < argc; i++)
        if (family_of(argv[i], &fam, &tunable)) m |= fam;
    return m;
}

/* Is word w a whole space-separated token of buf? */
static int has_token(const char *buf, const char *w)
{
    const size_t l = strlen(w);
    for (const char *p = buf; (p = strstr(p, w)) != NULL; p += l)
        if ((p == buf || p[-1] == ' ') && (p[l] == ' ' || !p[l])) return 1;
    return 0;
}

void tuned_overrides(int argc, char *const argv[], char *buf, size_t n)
{
    unsigned fam;
    int tunable;
    buf[0] = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int ok = 0;
        if (strncmp(a, "--", 2)) continue;                         /* a value */
        if (family_of(a, &fam, &tunable) && tunable) continue;     /* the knobs */
        for (int k = 0; TUNED_BASE_OK[k]; k++)
            if (!strcmp(a, TUNED_BASE_OK[k])) { ok = 1; break; }
        if (!ok && !has_token(buf, a)) cat(buf, n, "%s", a);
    }
}

static int ratio_far(uint32_t a, uint32_t b)
{
    if (!a || !b) return 0;
    return (uint64_t)a > 2u * (uint64_t)b || (uint64_t)b > 2u * (uint64_t)a;
}

void tuned_judge(const tuned_entry_t *e, const tuned_run_t *run,
                 tuned_verdict_t *v)
{
    const tuned_knobs_t *k = &e->k;
    tuned_knobs_t *a = &v->apply;
    char *nb = v->notes;
    const size_t nn = sizeof v->notes;
    v->e = *e;
    knobs_clear(a);
    v->applied[0] = 0; v->notes[0] = 0; v->why[0] = 0;

    /* Declined as a whole: the workload's shape or the rules have changed. */
    if (e->policy != TUNED_POLICY) {
        v->status = TUNED_DECLINED;
        put(v->why, sizeof v->why, "it was written under tuning policy %d and"
            " this binary's is %d; re-tune", e->policy, TUNED_POLICY);
        return;
    }
    if (e->J != run->J || e->region != run->region) {
        v->status = TUNED_DECLINED;
        put(v->why, sizeof v->why, "it was tuned at J %u region %d and this run"
            " is J %u region %d; re-tune for this shape", e->J, e->region,
            run->J, run->region);
        return;
    }
    /* Declined as a whole: the command line sets a knob of its own. The
     * entry's knobs were confirmed together, and they interact (finding 161:
     * a saved --slab-fill-ahead beside a hand-given --fill-blocks 589824 was
     * slower than either alone), so none of it applies beside one given by
     * hand (Kyle, 2026-11-17). An empty entry has nothing to set aside. */
    if (e->flags[0] && run->explicit_mask) {
        static const struct { unsigned fam; const char *what; } FAM[] = {
            { TUNED_F_FILL_BLOCKS, "--fill-blocks" },
            { TUNED_F_SS,          "the small-sieve cuts" },
            { TUNED_F_GATE_SHIFT,  "the fill gate" },
            { TUNED_F_MERGE,       "the fill merge" },
            { TUNED_F_SFA,         "the slab fill ahead" },
        };
        char what[160] = "";
        for (size_t i = 0; i < sizeof FAM / sizeof FAM[0]; i++)
            if (run->explicit_mask & FAM[i].fam)
                put(what + strlen(what), sizeof what - strlen(what), "%s%s",
                    what[0] ? ", " : "", FAM[i].what);
        v->status = TUNED_DECLINED;
        put(v->why, sizeof v->why, "the command line sets %s, and a saved entry"
            " applies whole or not at all", what);
        return;
    }
    v->status = TUNED_APPLY;

    if (k->fill_blocks) {
        if (run->max_grid > 0 && k->fill_blocks > run->max_grid)
            note(nb, nn, "not --fill-blocks %d: above this card's grid limit %d",
                 k->fill_blocks, run->max_grid);
        else a->fill_blocks = k->fill_blocks;
    }
    if (k->ss_block_cut || k->ss_warp_cut) {
        const int b = k->ss_block_cut ? k->ss_block_cut : run->ss_block_cut;
        const int w = k->ss_warp_cut  ? k->ss_warp_cut  : run->ss_warp_cut;
        if (!run->small_sieve)
            note(nb, nn, "not the small-sieve cuts: no small sieve in this run");
        else if (w <= b)
            note(nb, nn, "not the small-sieve cuts: warp %d would not be above block %d", w, b);
        else { a->ss_block_cut = k->ss_block_cut; a->ss_warp_cut = k->ss_warp_cut; }
    }
    if (k->fill_gate_shift >= 0) {
        if (!run->pipeline || !run->fill_gate)
            note(nb, nn, "not --fill-gate-shift %d: the fill gate is off", k->fill_gate_shift);
        else a->fill_gate_shift = k->fill_gate_shift;
    }
    if (k->no_fill_merge) {
        if (!run->pipeline || !run->fill_merge)
            note(nb, nn, "not --no-fill-merge: no merge in this run");
        else a->no_fill_merge = 1;
    }
    if (k->slab_fill_ahead >= 0) {
        if (!run->pipeline || !run->q_pipeline)
            note(nb, nn, "not the slab fill ahead: it needs --q-pipeline");
        else { a->slab_fill_ahead = k->slab_fill_ahead; a->slab_fill_blocks = k->slab_fill_blocks; }
    }
    tuned_format_flags(a, v->applied, sizeof v->applied);

    /* Applied anyway, with a note: speed only, and the knobs are fill and
     * small-sieve grids whose optimum follows logI and the factor base. */
    if (ratio_far(e->alim, run->alim) || ratio_far(e->rlim, run->rlim))
        note(nb, nn, "tuned on alim %u rlim %u, this job is alim %u rlim %u;"
             " a re-tune may find a different grid", e->alim, e->rlim,
             run->alim, run->rlim);
    if (e->side != run->side)
        note(nb, nn, "tuned with the special-q on side %d, this run is side %d",
             e->side, run->side);
}

int tuned_lookup(const char *path, const tuned_run_t *run, FILE *warn,
                 tuned_verdict_t *v)
{
    FILE *f;
    char line[1100];
    unsigned ln = 0;
    int header = 0, found = 0;
    tuned_entry_t e, best;
    memset(v, 0, sizeof *v);
    f = fopen(path, "r");
    if (!f) {
        v->status = errno == ENOENT ? TUNED_NOFILE : TUNED_BADFILE;
        if (v->status == TUNED_BADFILE)
            put(v->why, sizeof v->why, "cannot read it (%s)", strerror(errno));
        return v->status;
    }
    while (fgets(line, sizeof line, f)) {
        char err[200];
        ln++;
        if (!strchr(line, '\n') && !feof(f)) {
            int c;
            while ((c = fgetc(f)) != '\n' && c != EOF) {}
            if (warn) fprintf(warn, "%s:%u: line too long; skipped\n", path, ln);
            continue;
        }
        trim(line);
        if (!line[0]) continue;
        if (!header) {
            if (strcmp(line, TUNED_FORMAT_LINE)) {
                fclose(f);
                v->status = TUNED_BADFILE;
                put(v->why, sizeof v->why, "its first line is not \"%s\"",
                    TUNED_FORMAT_LINE);
                return v->status;
            }
            header = 1;
            continue;
        }
        if (line[0] == '#') continue;
        if (tuned_parse_line(line, &e, err, sizeof err)) {
            if (warn) fprintf(warn, "%s:%u: %s; skipped\n", path, ln, err);
            continue;
        }
        e.line = ln;
        if (strcmp(e.card, run->card) || e.cc_major != run->cc_major ||
            e.cc_minor != run->cc_minor || e.sms != run->sms ||
            e.logI != run->logI)
            continue;
        if (found && warn)
            fprintf(warn, "%s:%u: a second entry for this card at logI %d"
                    " (the first is line %u); using this one\n", path, ln,
                    run->logI, best.line);
        best = e;
        found = 1;
    }
    if (ferror(f)) {
        fclose(f);
        v->status = TUNED_BADFILE;
        put(v->why, sizeof v->why, "read error");
        return v->status;
    }
    fclose(f);
    if (!header) {
        v->status = TUNED_BADFILE;
        put(v->why, sizeof v->why, "it is empty");
        return v->status;
    }
    if (!found) { v->status = TUNED_NONE; return v->status; }
    tuned_judge(&best, run, v);
    return v->status;
}
