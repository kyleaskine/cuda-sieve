/* runlog.c -- implementation of the run log described in runlog.h. */

/* The Makefile builds the C sources as -std=c11, which is strict ISO C and
 * hides clock_gettime and localtime_r behind the feature-test macro. Declared
 * here rather than by loosening the standard for every C object. The loader
 * and time conversions themselves now live in platform.c, so this file needs
 * neither <dlfcn.h> nor <windows.h>. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "runlog.h"
#include "platform.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

/* Build stamp. The Makefile defines this from `git describe --dirty`; the
 * fallback keeps a hand-built or exported tree compiling and says plainly that
 * the field is unknown rather than inventing a commit. */
#ifndef BENCH_GIT_DESC
#define BENCH_GIT_DESC "unknown"
#endif
/* Pricing -D's (NORM_FAST_LOG2, NORM_CANCEL_TOL=...). Empty in every shipping
 * build. Carried into the log header because otherwise a binary built with a
 * deliberately wrong norm stamps the same git description as a real one, and
 * its relations would be indistinguishable from production output. */
#ifndef BENCH_DEFS
#define BENCH_DEFS ""
#endif

/* nvmlDevice_t is an opaque pointer, so void * is the honest local spelling,
 * and NVML_SUCCESS is 0. Only the five entry points the heartbeat needs are
 * resolved, and none of them is required. */
typedef struct { unsigned int gpu, memory; } nvml_util_t;

static struct {
    FILE  *f;
    double period_ms;
    double t_next;
    double t_open;              /* monotonic, for the elapsed column       */
    int    failed;              /* a write failed; warned once, log closed  */
    void  *lib;                 /* libnvidia-ml handle, or NULL             */
    void  *dev;                 /* nvmlDevice_t for this process's card     */
    int  (*nvml_shutdown)(void);
    int  (*nvml_util)(void *, nvml_util_t *);
    int  (*nvml_power)(void *, unsigned int *);
} L;

int runlog_active(void) { return L.f != NULL; }

const char *runlog_build_desc(void) { return BENCH_GIT_DESC; }

const char *runlog_build_defs(void) { return BENCH_DEFS; }

/* ---- open / close ------------------------------------------------------ */

/* The monotonic clock, not the wall clock the timestamps use: NTP steps and DST
 * both move the wall clock, and a backwards step would stall the heartbeat for
 * the length of the step on a run that is otherwise healthy. */
static double runlog_now_ms(void)
{
    return bench_monotonic_ms();
}

int runlog_open(const char *path, double period_s)
{
    if (!path) return 0;
    /* "ab": the run log is machine-parsed (relgeom.py, the oracle sweep
     * scripts), so it stays LF-only whichever platform produced it. Every
     * record is explicitly flushed below, so nothing here depends on the
     * line buffering that MSVC would silently turn into full buffering. */
    L.f = fopen(path, "ab");
    if (!L.f) {
        fprintf(stderr, "  warning: cannot open run log %s (%s); continuing"
                " without it\n", path, strerror(errno));
        L.failed = 1;
        return -1;
    }
    /* Line buffered: a kill -9 loses at most the record being written, never a
     * block of complete ones sitting in a 4 KB buffer.
     *
     * The size is explicit because it has to be. glibc reads a 0 here as "pick
     * a default", but MSVC requires 2 <= size <= INT_MAX whenever the buffer
     * pointer is NULL, and a 0 reaches the CRT's invalid-parameter handler --
     * which does not return. It terminates the process through __fastfail,
     * reported as 0xC0000409, the SAME status Windows uses for a stack-buffer
     * overrun. So the failure presented as memory corruption in code that has
     * none: bench.exe died here, right after opening the log and before
     * writing a single record, on every Windows run that passed --log.
     *
     * MSVC also treats _IOLBF as full buffering, so the guarantee in the
     * paragraph above rests on the explicit fflush after every record in
     * runlog_vwrite and runlog_note rather than on the mode. */
    setvbuf(L.f, NULL, _IOLBF, 4096);
    L.period_ms = period_s > 0.0 ? period_s * 1000.0 : 300000.0;
    L.t_open = runlog_now_ms();
    L.t_next = L.t_open + L.period_ms;
    return 0;
}

void runlog_close(void)
{
    if (L.f) { fclose(L.f); L.f = NULL; }
    if (L.lib) {
        if (L.nvml_shutdown) L.nvml_shutdown();
        bench_dlclose(L.lib);
        L.lib = NULL;
    }
    L.dev = NULL;
    L.nvml_shutdown = NULL;
    L.nvml_util = NULL;
    L.nvml_power = NULL;
}

int runlog_due(void)
{
    const double now = runlog_now_ms();
    if (!L.f || now < L.t_next) return 0;
    /* From now, not from the previous deadline: a q that overruns the period
     * (a slow band, a suspended process) must not leave the log owing a burst
     * of back-to-back records to catch up. */
    L.t_next = now + L.period_ms;
    return 1;
}

/* ---- writing ----------------------------------------------------------- */

static void runlog_stamp(char *out, size_t n)
{
    const time_t t = time(NULL);
    struct tm tmv;
    /* POSIX strftime supplies %z; MSVC does not. Use UTC on Windows so the
     * timestamp remains unambiguous without having to synthesize an offset. */
#ifdef _WIN32
    if (bench_gmtime(&t, &tmv) ||
        !strftime(out, n, "%Y-%m-%dT%H:%M:%SZ", &tmv))
        snprintf(out, n, "%lld", (long long)t);
#else
    if (bench_localtime(&t, &tmv) ||
        !strftime(out, n, "%Y-%m-%dT%H:%M:%S%z", &tmv))
        snprintf(out, n, "%lld", (long long)t);
#endif
}

/* A failed write is reported once and then the log is closed: the usual cause
 * is a full or unwritable filesystem, which will not fix itself, and one
 * warning per record for the remaining hours of a band would bury the stderr
 * BOINC uploads. */
static void runlog_fail(const char *what)
{
    if (!L.failed) {
        L.failed = 1;
        fprintf(stderr, "  warning: run log %s failed (%s); continuing without"
                " it\n", what, strerror(errno));
    }
    if (L.f) { fclose(L.f); L.f = NULL; }
}

static void runlog_vwrite(const char *prefix, const char *fmt, va_list ap)
{
    if (!L.f) return;
    if (fprintf(L.f, "%s", prefix) < 0 ||
        vfprintf(L.f, fmt, ap) < 0 ||
        fputc('\n', L.f) == EOF) {
        runlog_fail("write");
        return;
    }
    /* Line buffering makes this normally a no-op; it is here so `tail -f` on
     * the log of a wedged run shows the last record rather than a stale
     * buffer. No fsync: the .part and its sidecar are the durable artifacts
     * (ckpt.h) and this file is a record of them, not a substitute. */
    if (fflush(L.f)) runlog_fail("flush");
}

void runlog_note(const char *key, const char *fmt, ...)
{
    va_list ap;
    char pre[40], val[4600];
    size_t i;
    if (!L.f) return;
    snprintf(pre, sizeof pre, "# %-11s ", key);
    /* Formatted first, then flattened. A note's value is caller data -- argv
     * and file paths, which may legally contain a newline -- and only the
     * FIRST physical line of a multi-line write would carry the '#'. The rest
     * would be indistinguishable from records, breaking the one convention
     * runlog.h states about this file (`grep -v '^#'` leaves the records
     * alone). Control characters become spaces; nothing is dropped. */
    va_start(ap, fmt);
    vsnprintf(val, sizeof val, fmt, ap);
    va_end(ap);
    for (i = 0; val[i]; i++)
        if ((unsigned char)val[i] < 0x20 || val[i] == 0x7f) val[i] = ' ';
    if (fprintf(L.f, "%s%s\n", pre, val) < 0 || fflush(L.f))
        runlog_fail("write");
}

/* Timestamp, then seconds since this log was opened -- from the MONOTONIC
 * clock, so it survives what the timestamp cannot. Observed on this box
 * 2026-08-16: WSL2 resyncs its clock to the Windows host, and one band's
 * records came out with a two-second backwards step in the middle, which makes
 * a wall-clock delta between two records meaningless and can make one appear
 * to precede the record above it. The elapsed column is the one to compute
 * rates from; the timestamp is for lining a run up against a wall meter,
 * another machine, or a note in RESULTS.md. */
static void runlog_prefix(char *pre, size_t n)
{
    char ts[40];
    runlog_stamp(ts, sizeof ts);
    snprintf(pre, n, "%s +%.0fs  ", ts,
             (runlog_now_ms() - L.t_open) / 1000.0);
}

void runlog_record(const char *fmt, ...)
{
    va_list ap;
    char pre[64];
    if (!L.f) return;
    runlog_prefix(pre, sizeof pre);
    va_start(ap, fmt);
    runlog_vwrite(pre, fmt, ap);
    va_end(ap);
}

void runlog_warn(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    if (L.f) {
        char pre[80], stamp[64];
        /* Leading newlines are terminal spacing -- they exist to break away
         * from the \r progress line -- and in a log they would split the
         * record from its own timestamp. They belong to stderr only. */
        while (*fmt == '\n' || *fmt == ' ') fmt++;
        runlog_prefix(stamp, sizeof stamp);
        snprintf(pre, sizeof pre, "%sWARNING  ", stamp);
        va_start(ap, fmt);
        runlog_vwrite(pre, fmt, ap);
        va_end(ap);
    }
}

/* ---- NVML, resolved at run time ---------------------------------------- */

/* Candidate NVML library names, tried in order.
 *
 * POSIX: the versioned soname, not "libnvidia-ml.so" -- the unversioned link
 * is part of the toolkit's development package, which a volunteer host or a
 * container carrying only the driver need not have.
 *
 * Windows: nvml.dll is normally in the system directory, but some driver
 * packages install it only under the NVSMI directory, which is not on the
 * default search path. Without the second candidate those hosts silently log
 * gpu=n/a board=n/a. */
static const char *const nvml_names[] = {
#ifdef _WIN32
    "nvml.dll",
    "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
#else
    "libnvidia-ml.so.1",
#endif
    NULL
};

/* ONE implementation for both platforms. dlopen/dlsym/dlclose differ only in
 * spelling, and that spelling now lives in the platform layer -- this was two
 * near-identical 33-line copies that had to be kept in step by hand, and every
 * new NVML entry point had to be added to both. */
int runlog_gpu_bind(const char *pci_bus_id)
{
    int (*nvml_init)(void) = NULL;
    int (*nvml_byid)(const char *, void **) = NULL;
    void *lib = NULL;
    int inited = 0, i;

    if (L.lib || !pci_bus_id) return -1;
    for (i = 0; nvml_names[i] && !lib; i++)
        lib = bench_dlopen(nvml_names[i]);
    if (!lib) return -1;

    bench_dlsym(lib, "nvmlInit_v2", &nvml_init);
    bench_dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2", &nvml_byid);
    bench_dlsym(lib, "nvmlShutdown", &L.nvml_shutdown);
    bench_dlsym(lib, "nvmlDeviceGetUtilizationRates", &L.nvml_util);
    bench_dlsym(lib, "nvmlDeviceGetPowerUsage", &L.nvml_power);

    /* `inited` tracks whether nvmlInit_v2 actually RAN, which is not the same
     * as whether it resolved. The previous condition was "if (nvml_init &&
     * L.nvml_shutdown)", so a library where nvmlDeviceGetHandleByPciBusId_v2
     * failed to resolve short-circuited before nvml_init() was ever called and
     * then called nvmlShutdown against a library that was never initialised. */
    if (!nvml_init || !nvml_byid) goto fail;
    if (nvml_init() != 0) goto fail;
    inited = 1;
    if (nvml_byid(pci_bus_id, &L.dev) != 0 || !L.dev) goto fail;

    L.lib = lib;
    return 0;

fail:
    if (inited && L.nvml_shutdown) L.nvml_shutdown();
    bench_dlclose(lib);
    L.dev = NULL;
    L.nvml_shutdown = NULL;
    L.nvml_util = NULL;
    L.nvml_power = NULL;
    return -1;
}

int runlog_gpu_util(unsigned int *pct)
{
    nvml_util_t u;
    if (!L.dev || !L.nvml_util) return -1;
    if (L.nvml_util(L.dev, &u) != 0) return -1;
    *pct = u.gpu;
    return 0;
}

int runlog_gpu_watts(double *w)
{
    unsigned int mw = 0;
    if (!L.dev || !L.nvml_power) return -1;
    if (L.nvml_power(L.dev, &mw) != 0) return -1;
    *w = mw / 1000.0;
    return 0;
}
