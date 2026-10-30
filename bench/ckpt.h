/* ckpt.h -- crash-safe checkpoint and resume for a --relations band.
 *
 * WHY A SIDECAR AND NOT THE RELATION FILE ITSELF. The obvious way to resume is
 * to read the tail of the relation file and work out which special-q produced
 * the last line, from the sq-side primes it carries. That is ambiguous under
 * the convention the pipeline actually runs: the FULL factor base is in force,
 * so a relation is re-found at every band q dividing its sq-side norm --
 * measured P(re-found) is 70.2%/72.3%, mean 1.82 sq-side primes per relation.
 * A line typically carries two in-band primes and nothing says which one found
 * it. Guessing high skips special-q and loses relations with no symptom;
 * guessing low only re-sieves an overlap. See STATUS.md item 12a.
 *
 * WHAT MAKES THE SIDECAR EXACT. Without slabbing, pipeline.cuh flushes the
 * cofactor queue before enqueuing a q that would not fit, so a flush is a
 * whole-q boundary. With slabbing, the queue can fill between slabs of one q;
 * that flush may write a partial q and is deliberately NOT checkpointed. A
 * fresh cofactor run first records the empty/prefix boundary before q0, and a
 * slab-0 flush (which still contains only earlier q) may advance it. Thus every
 * sidecar names a whole-q prefix even though the .part may temporarily contain
 * a later uncheckpointed tail. Resume truncates that tail and replays from the
 * stored q. Recording a safe point costs an fsync and a rename.
 *
 * The stored byte offset is what makes a torn final line a non-problem: resume
 * truncates to it, so a partial write from a kill -9 is discarded rather than
 * parsed.
 */
#ifndef BENCH_CKPT_H
#define BENCH_CKPT_H

#include "bench.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CKPT_VERSION 1

/* ONE limit governs every derived name. Sizing the buffers differently at
 * different call sites is worse than not checking at all: a path that fits the
 * 2048-byte staging buffers but not a 1200-byte sidecar buffer passes the
 * startup checks, takes the lock, sieves -- and then fails at every checkpoint,
 * so ckpt_written never becomes 1 and a clean stop discards the .part holding
 * the whole session. */
/* Defined from the platform layer's limit rather than repeating the number:
 * bench_sync_parent and friends size their buffers from BENCH_PATH_MAX, and
 * two independent 2048s would silently disagree the first time either moved. */
#define CKPT_PATH_MAX BENCH_PATH_MAX
/* Longest suffix appended below: ".part.recover.tmp" is 17 bytes. */
#define CKPT_SUFFIX_MAX 20

typedef struct {
    char     fp[17];              /* job fingerprint, 16 hex digits + NUL      */
    uint64_t next_q, next_rho;    /* the first special-q NOT yet accounted for */
    unsigned long long rel_bytes; /* valid prefix of NAME.part                 */
    unsigned long long cand_bytes;/* valid prefix of CANDIDATES.part           */
    /* Recovery may have to remove candidate staging output even when a BOINC
     * relaunch omitted --candidates.  The pathname alone is not sufficient:
     * a damaged checkpoint must never turn into an arbitrary unlink target.
     * Record the open file's identity too and require both to match. */
    char     cand_part[CKPT_PATH_MAX];
    unsigned long long cand_dev, cand_ino;
    unsigned long long nrel;      /* relations in that prefix                  */
    unsigned long long nqdone;    /* special-q completed, all sessions         */
    /* Derived once from the FIRST q of the original band and held band-wide
     * (bench_main.cu, "params from q="). Re-deriving them from the resume q
     * would move the survivor gate partway through a run, so they are carried
     * rather than recomputed. */
    double   scale, scale0, allowance, allowance0;
} ckpt_t;

/* ---- paths ------------------------------------------------------------- */

/* Check ONCE, where the path enters the program, that every name derived from
 * it will fit. Callers that pass this may then build .part, .part.ckpt,
 * .part.ckpt.tmp, .part.recover[.tmp] and .lock without any of them being able
 * to truncate. The names are derived, not typed: under BOINC `relations` is a
 * resolved absolute slot/project path, so the margin is not the one the
 * command line suggests. */
static inline int ckpt_path_usable(const char *relations, const char *what)
{
    const size_t n = relations ? strlen(relations) : 0;
    if (n + CKPT_SUFFIX_MAX >= (size_t)CKPT_PATH_MAX) {
        fprintf(stderr,
                "%s: path is %zu bytes; the .part, checkpoint, recovery and"
                " lock names derived from it must fit in %d\n",
                what, n, CKPT_PATH_MAX);
        return -1;
    }
    return 0;
}

/* Silent: a caller that skipped ckpt_path_usable() gets a return code, not a
 * message. ckpt_write() runs this at every checkpoint of a multi-day band, so a
 * diagnostic here would repeat until it buried the log. Argument order matches
 * the wrappers below so a misremembered call cannot silently swap the two
 * const char * parameters. */
static inline int ckpt_path_fmt(const char *relations, char *out, size_t n,
                                const char *suffix)
{
    const int k = snprintf(out, n, "%s%s", relations, suffix);
    if (k < 0 || (size_t)k >= n) {
        if (n) out[0] = '\0';
        return -1;
    }
    return 0;
}

static inline int ckpt_part_path(const char *relations, char *out, size_t n)
{
    return ckpt_path_fmt(relations, out, n, ".part");
}

static inline int ckpt_ckpt_path(const char *relations, char *out, size_t n)
{
    return ckpt_path_fmt(relations, out, n, ".part.ckpt");
}

static inline int ckpt_lock_path(const char *relations, char *out, size_t n)
{
    return ckpt_path_fmt(relations, out, n, ".lock");
}

static inline int ckpt_recovery_path(const char *relations, char *out, size_t n)
{
    return ckpt_path_fmt(relations, out, n, ".part.recover");
}

static inline int ckpt_recovery_tmp_path(const char *relations, char *out,
                                         size_t n)
{
    return ckpt_path_fmt(relations, out, n, ".part.recover.tmp");
}

/* ---- fingerprint ------------------------------------------------------- */

/* FNV-1a over a canonical rendering of everything that decides WHICH relations
 * a run emits. Not a security hash: it exists to catch resuming one job's file
 * with another job's parameters, which produces lines that individually verify
 * and collectively mean nothing.
 *
 * The derived scale/allowance are deliberately NOT here. They depend on the
 * first q of the band, so they differ between the original run and a resumed
 * one by construction; they are carried in the checkpoint instead. */
static inline void ckpt_fp_feed(uint64_t *h, const char *s)
{
    for (; *s; s++) { *h ^= (unsigned char)*s; *h *= 1099511628211ULL; }
}

/* Appends and never advances past the buffer. The previous version accumulated
 * `k` from snprintf's would-have-written return value and kept forming
 * `out + k`, which is undefined once k exceeds n -- and, worse, silently
 * dropped the TAIL of the string on overflow. The tail is the sieve-parameter
 * block, so two jobs differing only in logI or lpb would have hashed
 * identically: the exact collision the fingerprint exists to prevent. */
static inline void ckpt_fp_cat(char *out, size_t n, size_t *k,
                               const char *fmt, ...)
{
    va_list ap;
    int w;
    if (*k >= n) return;
    va_start(ap, fmt);
    w = vsnprintf(out + *k, n - *k, fmt, ap);
    va_end(ap);
    *k = (w < 0 || (size_t)w >= n - *k) ? n : *k + (size_t)w;
}

static inline void ckpt_job_text(const poly_t *P, const bench_cfg_t *cfg,
                                 char *out, size_t n)
{
    size_t k = 0;
    int z;
    if (!n) return;
    out[0] = 0;
    ckpt_fp_cat(out, n, &k, "deg=%d skew=%.17g", P->deg, P->skew);
    for (z = 0; z <= P->deg && z < BENCH_NCOEFF; z++)
        ckpt_fp_cat(out, n, &k, " c%d=%s", z, P->cs[z]);
    ckpt_fp_cat(out, n, &k, " Y0=%s Y1=%s", P->y0s, P->y1s);
    ckpt_fp_cat(out, n, &k,
                " rlim=%u alim=%u lpbr=%u lpba=%u mfbr=%u mfba=%u"
                " rlambda=%.17g alambda=%.17g",
                P->rlim, P->alim, P->lpbr, P->lpba, P->mfbr, P->mfba,
                P->rlambda, P->alambda);
    ckpt_fp_cat(out, n, &k,
                " logI=%d J=%u sqside=%d nbe=%d"
                " lim0=%u lpb0=%u mfb0=%u lim1=%u lpb1=%u mfb1=%u",
                cfg->logI, cfg->J, cfg->sq_side, cfg->not_both_even,
                cfg->lim0, cfg->lpb0, cfg->mfb0,
                cfg->lim, cfg->lpb, cfg->mfb);
    /* The factor-base convention and the cofactor configuration belong here
     * too: STATUS.md 12a names the FB convention explicitly, and dropping
     * --cofactor on a resume changes the relation file from queue-emitted to
     * trial-division-only AND swaps the checkpoint discipline from
     * flush-anchored to a timer. The cofactor method and its bounds decide
     * which cofactors split at all, so they change the yield too. */
    ckpt_fp_cat(out, n, &k,
                " maxbits=%d cofac=%d ecm=%d b1=%u/%u b2=%u/%u curves=%u"
                " abort=%d/%d rounds=%d budget=%u",
                cfg->fb_maxbits, cfg->cofactor, cfg->cof_ecm,
                cfg->ecm_b1[0], cfg->ecm_b1[1], cfg->ecm_b2[0], cfg->ecm_b2[1],
                cfg->ecm_curves, cfg->cof_abort_side[0], cfg->cof_abort_side[1],
                cfg->cof_rounds, cfg->cof_budget);
}

static inline void ckpt_fingerprint(const poly_t *P, const bench_cfg_t *cfg,
                                    char out[17])
{
    char text[2048];
    uint64_t h = 1469598103934665603ULL;
    ckpt_job_text(P, cfg, text, sizeof text);
    ckpt_fp_feed(&h, text);
    snprintf(out, 17, "%016llx", (unsigned long long)h);
}

/* Paths may contain whitespace, so store the candidate staging name as hex
 * rather than relying on the checkpoint's whitespace-delimited parser. */
static inline int ckpt_path_hex_encode(const char *path, char *out, size_t n)
{
    static const char hex[] = "0123456789abcdef";
    const size_t len = strlen(path);
    if (!n || len > (n - 1) / 2) return -1;
    for (size_t i = 0; i < len; i++) {
        const unsigned char c = (unsigned char)path[i];
        out[2 * i] = hex[c >> 4];
        out[2 * i + 1] = hex[c & 15];
    }
    out[2 * len] = '\0';
    return 0;
}

static inline int ckpt_hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static inline int ckpt_path_hex_decode(const char *hex, char *out, size_t n)
{
    const size_t len = strlen(hex);
    if (!len || (len & 1) || len / 2 >= n) return -1;
    for (size_t i = 0; i < len / 2; i++) {
        const int hi = ckpt_hex_nibble(hex[2 * i]);
        const int lo = ckpt_hex_nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0 || (hi == 0 && lo == 0)) return -1;
        out[i] = (char)((hi << 4) | lo);
    }
    out[len / 2] = '\0';
    return 0;
}

/* ---- write ------------------------------------------------------------- */

/* Atomic: a full write to a temporary, fsynced, then renamed over the live
 * checkpoint. A crash mid-write leaves the PREVIOUS checkpoint intact, which is
 * always a valid (merely older) resume point. The caller must have fsynced the
 * relation file itself BEFORE calling this -- the checkpoint asserts that those
 * bytes are durable. */
static inline int ckpt_write(const char *relations, const poly_t *P,
                             const bench_cfg_t *cfg, const ckpt_t *ck)
{
    char path[CKPT_PATH_MAX], tmp[CKPT_PATH_MAX], text[2048];
    char cand_hex[2 * CKPT_PATH_MAX + 1];
    FILE *f;
    /* Both go through the same guard, so ".tmp" cannot be the one suffix that
     * overflows; CKPT_SUFFIX_MAX already reserves room for it. */
    if (ckpt_ckpt_path(relations, path, sizeof path) ||
        ckpt_path_fmt(path, tmp, sizeof tmp, ".tmp") ||
        (ck->cand_part[0] &&
         ckpt_path_hex_encode(ck->cand_part, cand_hex, sizeof cand_hex)))
        return -1;
    /* Binary, so a sidecar is byte-identical whichever platform wrote it.
     * The reader below already tolerates a stray \r on every field, so this
     * is not a fix; it keeps a checkpoint moved between a Windows and a Linux
     * host the same file rather than a differently-encoded equivalent. */
    if (!(f = fopen(tmp, "wb"))) { perror(tmp); return -1; }
    ckpt_job_text(P, cfg, text, sizeof text);
    fprintf(f, "# cuda-sieve resume checkpoint. Delete this file to force a"
               " fresh run.\n");
    fprintf(f, "# job: %s\n", text);
    fprintf(f, "version = %d\n", CKPT_VERSION);
    fprintf(f, "fingerprint = %s\n", ck->fp);
    fprintf(f, "next_q = %llu\n", (unsigned long long)ck->next_q);
    fprintf(f, "next_rho = %llu\n", (unsigned long long)ck->next_rho);
    fprintf(f, "rel_bytes = %llu\n", ck->rel_bytes);
    fprintf(f, "cand_bytes = %llu\n", ck->cand_bytes);
    if (ck->cand_part[0]) {
        fprintf(f, "candidate_part_hex = %s\n", cand_hex);
        fprintf(f, "candidate_dev = %llu\n", ck->cand_dev);
        fprintf(f, "candidate_ino = %llu\n", ck->cand_ino);
    }
    fprintf(f, "relations = %llu\n", ck->nrel);
    fprintf(f, "nq_done = %llu\n", ck->nqdone);
    fprintf(f, "scale = %.17g\n", ck->scale);
    fprintf(f, "scale0 = %.17g\n", ck->scale0);
    fprintf(f, "allowance = %.17g\n", ck->allowance);
    fprintf(f, "allowance0 = %.17g\n", ck->allowance0);
    if (fflush(f) || bench_sync_stream(f) || fclose(f)) {
        perror(tmp); remove(tmp); return -1;
    }
    if (bench_atomic_replace(tmp, path)) { perror(path); remove(tmp); return -1; }
    return 0;
}

/* ---- read -------------------------------------------------------------- */

/* 0 = loaded, -1 = absent, -2 = present but malformed, -3 = unreadable. A
 * malformed checkpoint is NOT treated as absent: silently starting over would
 * discard a file the operator believes is being resumed. */
static inline int ckpt_read(const char *relations, ckpt_t *ck)
{
    char path[CKPT_PATH_MAX], line[2 * CKPT_PATH_MAX + 64];
    FILE *f;
    int version = 0, got = 0, cand_meta = 0;
    /* -2, not -1: -1 is "no sidecar", which makes the caller report that the
     * file does not exist and advise --restart. A name we could not build says
     * nothing about what is on disk, and telling an operator to discard a
     * resumable .part on that basis is the one answer that loses work. */
    if (ckpt_ckpt_path(relations, path, sizeof path)) return -2;
    if (!(f = fopen(path, "rb"))) {
        if (errno == ENOENT) return -1;
        perror(path);
        return -3;
    }
    memset(ck, 0, sizeof *ck);
    while (fgets(line, sizeof line, f)) {
        char key[64]; char val[256];
        static const char cand_path_key[] = "candidate_part_hex = ";
        if (line[0] == '#' || line[0] == '\n') continue;
        if (!strncmp(line, cand_path_key, sizeof cand_path_key - 1)) {
            char *encoded = line + sizeof cand_path_key - 1;
            encoded[strcspn(encoded, "\r\n")] = '\0';
            if (ckpt_path_hex_decode(encoded, ck->cand_part,
                                     sizeof ck->cand_part)) {
                fprintf(stderr, "resume: %s has a malformed candidate path\n",
                        path);
                fclose(f); return -2;
            }
            cand_meta |= 1;
            continue;
        }
        if (sscanf(line, "%63s = %255s", key, val) != 2) continue;
        if      (!strcmp(key, "version"))     { version = atoi(val); got |= 1; }
        /* Length-checked rather than truncated: a short or overlong
         * fingerprint is a corrupt sidecar, and silently clipping one to 16
         * characters could make it compare equal to a real job's. */
        else if (!strcmp(key, "fingerprint")) {
            if (strlen(val) != 16) {
                fprintf(stderr, "resume: %s has a malformed fingerprint"
                        " (%zu characters, expected 16)\n", path, strlen(val));
                fclose(f); return -2;
            }
            memcpy(ck->fp, val, 16); ck->fp[16] = 0; got |= 2;
        }
        else if (!strcmp(key, "next_q"))      { ck->next_q = strtoull(val, 0, 10); got |= 4; }
        else if (!strcmp(key, "next_rho"))    { ck->next_rho = strtoull(val, 0, 10); got |= 8; }
        else if (!strcmp(key, "rel_bytes"))   { ck->rel_bytes = strtoull(val, 0, 10); got |= 16; }
        else if (!strcmp(key, "cand_bytes"))  ck->cand_bytes = strtoull(val, 0, 10);
        else if (!strcmp(key, "candidate_dev")) {
            ck->cand_dev = strtoull(val, 0, 10); cand_meta |= 2;
        }
        else if (!strcmp(key, "candidate_ino")) {
            ck->cand_ino = strtoull(val, 0, 10); cand_meta |= 4;
        }
        else if (!strcmp(key, "relations"))   ck->nrel = strtoull(val, 0, 10);
        else if (!strcmp(key, "nq_done"))     ck->nqdone = strtoull(val, 0, 10);
        else if (!strcmp(key, "scale"))       { ck->scale = strtod(val, 0); got |= 32; }
        else if (!strcmp(key, "scale0"))      { ck->scale0 = strtod(val, 0); got |= 64; }
        else if (!strcmp(key, "allowance"))   { ck->allowance = strtod(val, 0); got |= 128; }
        else if (!strcmp(key, "allowance0"))  { ck->allowance0 = strtod(val, 0); got |= 256; }
    }
    {
        const int read_bad = ferror(f);
        const int saved_errno = errno;
        const int close_bad = fclose(f);
        if (read_bad || close_bad) {
            if (read_bad) errno = saved_errno;
            perror(path);
            return -3;
        }
    }
    if (got != 511) {
        fprintf(stderr, "resume: %s is missing required fields\n", path);
        return -2;
    }
    if (cand_meta && cand_meta != 7) {
        fprintf(stderr, "resume: %s has incomplete candidate file identity\n",
                path);
        return -2;
    }
    if (version != CKPT_VERSION) {
        fprintf(stderr, "resume: %s is version %d, this build writes %d\n",
                path, version, CKPT_VERSION);
        return -2;
    }
    if (ck->scale <= 0.0 || ck->scale0 <= 0.0) {
        fprintf(stderr, "resume: %s carries a degenerate scale\n", path);
        return -2;
    }
    return 0;
}

/* ---- lock -------------------------------------------------------------- */

/* Two sievers appending to one .part is silent corruption, and a job queue
 * makes that a realistic accident rather than a theoretical one. O_EXCL plus
 * the pid.
 *
 * A stale lock CLEARS ITSELF. The crash cases resume exists for -- SIGKILL,
 * OOM, power loss, and the second-signal _exit -- all skip the unlink, and a
 * lock that then needs a human to remove defeats unattended operation more
 * thoroughly than the race it prevents. So an existing lock naming a pid that
 * no longer exists is taken over. The residual hazard is pid reuse: a lock
 * whose pid has been recycled onto an unrelated process reads as live and is
 * refused, which is the safe direction. */
static inline int ckpt_lock(const char *relations, char *path, size_t n)
{
    int fd, retried = 0;
    char buf[64];
    if (ckpt_lock_path(relations, path, n)) return -1;
again:
    fd = bench_lock_create(path);
    if (fd < 0) {
        if (errno == EEXIST) {
            /* Written with a raw fd below, so it is LF-only on every
             * platform; read it the same way. */
            FILE *f = fopen(path, "rb");
            long other = 0;
            if (f) { if (fscanf(f, "%ld", &other) != 1) other = 0; fclose(f); }
            if (!retried && other > 0 && !bench_process_exists(other)) {
                fprintf(stderr, "  %s is stale (pid %ld is gone); taking it"
                        " over\n", path, other);
                remove(path);
                retried = 1;
                goto again;
            }
            fprintf(stderr,
                    "  %s exists: another run (pid %ld) is writing these"
                    " files.\n"
                    "  If it is not running, remove the lock and retry.\n",
                    path, other);
        } else perror(path);
        return -1;
    }
    snprintf(buf, sizeof buf, "%ld\n", bench_getpid());
    if (bench_fd_write(fd, buf, strlen(buf)) < 0) { /* advisory only */ }
    bench_fd_close(fd);
    return 0;
}

/* Held for the process lifetime. Registered with atexit, so every early
 * `return 1` between acquisition and the band drops it -- but atexit is NOT
 * enough on its own, and a BOINC application is the case that proves it.
 *
 * boinc_finish() ends in _exit() on Unix and TerminateProcess() on Windows,
 * deliberately, so that a wedged static destructor cannot hang a task on its
 * way out. atexit handlers are skipped with them, so under a BOINC client the
 * lock file of every SUCCESSFULLY COMPLETED workunit survived. A volunteer
 * reported hundreds of them. Nothing malfunctioned -- the staleness check
 * below means a later run just takes the lock over, which is why this went
 * unnoticed -- but the project directory fills up with litter.
 *
 * So the owner releases it EXPLICITLY before handing control to anything that
 * may not return; see bench_main.cu. Idempotent, so the atexit registration
 * costs nothing when both run. */
static char ckpt_lock_held[CKPT_PATH_MAX];

static inline void ckpt_unlock_held(void)
{
    if (ckpt_lock_held[0]) { remove(ckpt_lock_held); ckpt_lock_held[0] = 0; }
}

static inline void ckpt_unlock(const char *path)
{
    if (path && *path) remove(path);
}

#ifdef __cplusplus
}
#endif
#endif
