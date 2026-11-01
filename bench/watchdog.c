/* watchdog.c -- see watchdog.h for what this is and why it is not a killer. */

/* Same feature-test block as platform.c, and for the same reason: -std=c11 is
 * strict ISO, under which sigaction, nanosleep and pthread_kill are not
 * declared at all -- they compile as implicit int-returning functions and the
 * struct is incomplete. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "watchdog.h"
#include "bench.h"          /* BENCH_EXIT_STALLED */
#include "platform.h"
#include "runlog.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#include <fcntl.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
/* The stalled thread's own call stack, which is the one thing the phase label
 * cannot supply: whether the host is parked in libcuda (a device-side hang) or
 * spinning in our own code. glibc only; a build without execinfo.h simply
 * reports the phase and the GPU columns, which is already the decisive part.
 *
 * backtrace() is not formally async-signal-safe -- its first call may resolve
 * unwind machinery -- so this is fired at most ONCE per run, and only from the
 * second report onward, by which point the process has been stalled for twice
 * the threshold and the small risk of perturbing it is worth the answer. */
#if defined(__GLIBC__)
#include <execinfo.h>
#define WD_BACKTRACE 1
#endif
#endif

#define WD_POLL_MS 500.0
#define WD_BT_FRAMES 64

static struct {
    /* written by the sieve thread */
    volatile unsigned long long seq;
    const char *volatile phase;
    volatile unsigned long long q, rho, nq;
    volatile unsigned slab, nslab;

    /* written once at wd_start, read by both */
    double stall_ms;
    double kill_ms;
    const char *logpath;
    int armed;
    volatile int kill_armed;

    /* the watchdog thread's own */
    volatile int run;
#ifdef _WIN32
    HANDLE th;
#else
    pthread_t th;
    pthread_t stalled;      /* the thread that called wd_start */
    int have_th;
#endif
} W;

/* ---- the heartbeat setters -------------------------------------------- *
 *
 * Guarded on `armed` so that a disarmed run pays one predictable branch on a
 * hot-ish path rather than four stores to a cache line no one reads. */

void wd_phase(const char *name)
{
    if (!W.armed) return;
    W.phase = name;
    W.seq++;
}

const char *wd_phase_now(void)
{
    return W.armed ? W.phase : NULL;
}

void wd_arm_kill(void)
{
    W.kill_armed = 1;
}

void wd_q(unsigned long long q, unsigned long long rho,
          unsigned long long nqdone)
{
    if (!W.armed) return;
    W.q = q; W.rho = rho; W.nq = nqdone;
}

void wd_slab(unsigned slab, unsigned nslab)
{
    if (!W.armed) return;
    W.slab = slab; W.nslab = nslab;
}

/* ---- reporting --------------------------------------------------------- */

/* RAW DESCRIPTORS, NOT stdio, AND THE REASON IS THE WHOLE FEATURE.
 *
 * fputs(text, stderr) takes stderr's FILE lock. One of the two stalls this
 * exists to catch is "the host side is stuck", and a perfectly ordinary way to
 * be stuck on the host is blocked inside printf writing the progress line to a
 * pipe nobody is draining -- a work client that stopped reading, a `| tee` onto
 * a full disk, a dead ssh session. The sieve thread then HOLDS that lock, the
 * watchdog blocks on it, no report is printed, and the give-up below is never
 * reached: the watchdog would hang on exactly the fault it was built to end.
 *
 * write(2) takes no userspace lock, and open/write/close on the log file is the
 * same trade. Both can still block on a wedged filesystem, which is why the
 * kill is sequenced BEFORE any of this in wd_loop rather than after it. */
static void wd_write(int fd, const char *text, size_t n)
{
#ifdef _WIN32
    (void)_write(fd, text, (unsigned)n);
#else
    (void)!write(fd, text, n);
#endif
}

static void wd_emit(const char *text)
{
    const size_t n = strlen(text);
    wd_write(2, text, n);
    if (W.logpath) {
        static int warned = 0;
        int fd = open(W.logpath, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            wd_write(fd, text, n);
#ifdef _WIN32
            _close(fd);
#else
            close(fd);
#endif
        } else if (!warned++) {
            /* Once, and never silently: the flag exists so that reports reach a
             * file when stderr goes somewhere the operator cannot see, so
             * producing no file at all without saying so defeats it. */
            char m[512];
            snprintf(m, sizeof m, "  !! watchdog: cannot append to %s;"
                     " reports go to stderr only\n", W.logpath);
            wd_write(2, m, strlen(m));
        }
    }
}

static void wd_stamp(char *out, size_t n)
{
    time_t t = time(NULL);
    struct tm tmv;
    if (bench_localtime(&t, &tmv) || !strftime(out, n, "%Y-%m-%dT%H:%M:%S", &tmv))
        snprintf(out, n, "%lld", (long long)t);
}

/* `want_gpu` is true only on the FIRST report of a stall, and the GPU line is
 * emitted SEPARATELY, after the core block has already been written.
 *
 * runlog_gpu_util/watts are ioctls into the very driver a stall may have wedged
 * (STATUS.md item 12d: the diagnosed fault is a driver error that leaves CUDA
 * waiting forever). Measured during that incident, NVML kept answering while
 * the CUDA context was stuck -- nvidia-smi returned util and watts every time
 * -- so querying it is worth the risk. But the risk is not zero, so it is taken
 * at most once per stall, and never before the phase, q and slab have reached
 * the operator. */
static void wd_report(int nth, double stalled_s, int want_gpu)
{
    char buf[768], ts[40];
    const char *phase = W.phase ? W.phase : "(none)";

    wd_stamp(ts, sizeof ts);
    snprintf(buf, sizeof buf,
             "\n  !! watchdog [%s]: no progress for %.0f s (report %d)\n"
             "     phase   %s\n"
             "     q       %llu  rho %llu  (q completed this band: %llu)\n"
             "     slab    %u of %u\n"
             "     seq     %llu\n",
             ts, stalled_s, nth, phase,
             (unsigned long long)W.q, (unsigned long long)W.rho,
             (unsigned long long)W.nq,
             W.slab, W.nslab, (unsigned long long)W.seq);
    wd_emit(buf);

    if (want_gpu) {
        char gpu[32], pw[32];
        unsigned int pct = 0;
        double watts = 0;
        if (runlog_gpu_util(&pct) == 0) snprintf(gpu, sizeof gpu, "%u%%", pct);
        else                            snprintf(gpu, sizeof gpu, "n/a");
        if (runlog_gpu_watts(&watts) == 0) snprintf(pw, sizeof pw, "%.1f W", watts);
        else                               snprintf(pw, sizeof pw, "n/a");
        snprintf(buf, sizeof buf,
                 "     gpu     util %s   board %s\n"
                 "     (gpu util near 100%% means a kernel is not terminating;\n"
                 "      near 0%% means the host side is stuck.)\n", gpu, pw);
        wd_emit(buf);
    }
}

static void wd_resumed(double stalled_s, const char *phase)
{
    char buf[256], ts[40];
    wd_stamp(ts, sizeof ts);
    snprintf(buf, sizeof buf,
             "  !! watchdog [%s]: progress resumed after %.0f s stalled in %s\n",
             ts, stalled_s, phase ? phase : "(none)");
    wd_emit(buf);
}

/* ---- the stalled thread's backtrace ------------------------------------ */

#ifdef WD_BACKTRACE
static void wd_sig_backtrace(int sig)
{
    void *bt[WD_BT_FRAMES];
    int n = backtrace(bt, WD_BT_FRAMES);
    static const char hdr[] = "     backtrace of the stalled thread"
                              " (addr2line -e bench -f -C -i <addr>):\n";
    (void)sig;
    (void)!write(2, hdr, sizeof hdr - 1);
    backtrace_symbols_fd(bt, n, 2);
}

static void wd_request_backtrace(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = wd_sig_backtrace;
    sigemptyset(&sa.sa_mask);
    /* SA_RESTART, deliberately. An earlier version omitted it on the theory
     * that interrupting the blocking call was the point; it is not -- the
     * handler runs and walks the stack either way, and without SA_RESTART an
     * in-flight fsync() under "q.checkpoint" or a driver ioctl can come back
     * EINTR in a run that was only slow. Never break the thing you are
     * measuring. */
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGUSR2, &sa, NULL) == 0)
        pthread_kill(W.stalled, SIGUSR2);
}
#else
static void wd_request_backtrace(void) { }
#endif

/* The last thing this process does. Not abort(): a core dump of a 12 GB
 * address space is not the evidence anyone wants, and the evidence that IS
 * wanted -- phase, q, slab, GPU utilisation, backtrace -- has already been
 * printed by the reports leading up to here. Nothing on this path queries
 * NVML or touches stdio, so nothing between a wedged run and the exit that
 * frees the card can itself block. */
static void wd_give_up(double stalled_s)
{
    char buf[640], ts[40];
    wd_stamp(ts, sizeof ts);
    snprintf(buf, sizeof buf,
             "\n  !! watchdog [%s]: no progress for %.0f s -- giving up.\n"
             "     Exiting %d (stalled). The last checkpoint is intact: a\n"
             "     resume replays from the last whole special-q, so at most\n"
             "     the q in flight is lost. If this repeats on the same host,\n"
             "     suspect the card before the code -- check dmesg for Xid,\n"
             "     and `nvidia-smi -q -d PAGE_RETIREMENT,ECC` for retired\n"
             "     pages or ECC errors.\n",
             ts, stalled_s, BENCH_EXIT_STALLED);
    wd_emit(buf);
    bench_fast_exit(BENCH_EXIT_STALLED);
}

/* ---- the thread -------------------------------------------------------- */

static void wd_sleep_ms(double ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000.0);
    ts.tv_nsec = (long)((ms - (double)ts.tv_sec * 1000.0) * 1e6);
    nanosleep(&ts, NULL);
#endif
}

static void wd_loop(void)
{
    unsigned long long last_seq = W.seq;
    double t_change = bench_monotonic_ms();
    double t_report = 0;
    int nreport = 0, bt_done = 0;
    const char *stalled_phase = NULL;

    while (W.run) {
        double now, stalled;
        unsigned long long seq;
        wd_sleep_ms(WD_POLL_MS);
        now = bench_monotonic_ms();
        seq = W.seq;
        if (seq != last_seq) {
            if (nreport) wd_resumed((now - t_change) / 1000.0, stalled_phase);
            last_seq = seq;
            t_change = now;
            nreport = 0;
            continue;
        }
        stalled = now - t_change;

        /* THE KILL IS TESTED FIRST, before the report and before any NVML
         * query. Two reasons, both learned the hard way:
         *
         *  - Sequenced after the report, the exit inherits every way the
         *    report can block -- a log file on a wedged mount, an
         *    unresponsive driver. The one job the kill has is to end a
         *    process that is already stuck, so nothing that can stick may
         *    come before it.
         *  - Tested below the `stalled < stall_ms` guard, a --watchdog-kill
         *    SMALLER than --watchdog was silently raised to the report
         *    threshold, while the startup banner still promised the value the
         *    operator asked for. Here the two thresholds are independent. */
        if (W.kill_ms > 0 && W.kill_armed && stalled >= W.kill_ms) {
            /* A give-up must never be the first thing said about a stall. With
             * a kill threshold below the report one -- which is now honoured
             * rather than silently clamped -- no report has been printed yet,
             * so print one here. want_gpu is 0: this path stays free of
             * anything that could block between the decision and the exit. */
            if (!nreport) wd_report(1, stalled / 1000.0, 0);
            if (!bt_done) { wd_request_backtrace(); bt_done = 1;
                            wd_sleep_ms(250.0); }
            wd_give_up(stalled / 1000.0);
        }
        if (stalled < W.stall_ms) continue;
        if (nreport && now - t_report < W.stall_ms) continue;
        stalled_phase = W.phase;
        t_report = now;
        ++nreport;                     /* separate statement: reading and
                                        * modifying nreport in one call is
                                        * unsequenced, and -Wsequence-point
                                        * says so */
        wd_report(nreport, stalled / 1000.0, nreport == 1);
        /* One backtrace per run, from the second report, and ONLY once the
         * kill is armed -- i.e. inside the band. During setup a slow
         * factor-base load legitimately produces repeat reports, and
         * backtrace() is not async-signal-safe: its first call resolves unwind
         * machinery through the loader and malloc, so a signal landing while
         * the sieve thread holds either lock would deadlock it. That is the
         * watchdog causing the failure it exists to report, so it is gated by
         * the same flag the kill is. */
        if (nreport == 2 && !bt_done && W.kill_armed) {
            wd_request_backtrace();
            bt_done = 1;
        }
    }
}

#ifdef _WIN32
static DWORD WINAPI wd_thread(LPVOID arg) { (void)arg; wd_loop(); return 0; }
#else
static void *wd_thread(void *arg) { (void)arg; wd_loop(); return NULL; }
#endif

int wd_start(double stall_s, double kill_s, const char *logpath)
{
    if (stall_s <= 0) return 0;
    W.stall_ms = stall_s * 1000.0;
    W.kill_ms = kill_s > 0 ? kill_s * 1000.0 : 0;
    W.logpath = logpath;
    W.phase = "startup";
    W.seq = 1;
    W.run = 1;
    W.armed = 1;
#ifdef _WIN32
    W.th = CreateThread(NULL, 0, wd_thread, NULL, 0, NULL);
    if (!W.th) { W.armed = W.run = 0; }
#else
    W.stalled = pthread_self();
    {
        /* Started with every signal blocked, so asynchronous ones stay with
         * the threads that handle them: BOINC's runtime suspends and quits
         * from its SIGALRM handler, which must run on the worker thread, and
         * a process-directed signal goes to any thread not blocking it. The
         * backtrace request is thread-directed at W.stalled, so it is
         * unaffected. */
        sigset_t all, old;
        sigfillset(&all);
        pthread_sigmask(SIG_BLOCK, &all, &old);
        if (pthread_create(&W.th, NULL, wd_thread, NULL) == 0) W.have_th = 1;
        else { W.armed = W.run = 0; }
        pthread_sigmask(SIG_SETMASK, &old, NULL);
    }
#endif
    if (!W.armed) {
        fprintf(stderr, "bench: could not start the watchdog thread;"
                        " continuing without it\n");
        return -1;
    }
    fprintf(stderr, "bench: watchdog armed -- report after %.0f s of no"
                    " progress", stall_s);
    if (W.kill_ms > 0) fprintf(stderr, ", exit %d after %.0f s",
                               BENCH_EXIT_STALLED, kill_s);
    if (logpath) fprintf(stderr, "; also logged to %s", logpath);
    fputc('\n', stderr);
    return 0;
}

void wd_stop(void)
{
    if (!W.armed) return;
    W.armed = 0;
    W.run = 0;
#ifdef _WIN32
    /* Only close the handle if the thread actually ended. Closing it after a
     * timeout returns as though the thread had joined, and the caller's very
     * next statement is runlog_close(), which unloads NVML -- a surviving
     * thread would then dispatch through a function pointer into an unmapped
     * DLL. Leaking one handle at process exit is the cheaper mistake, and the
     * warning says which one happened. */
    if (WaitForSingleObject(W.th, 5000) == WAIT_OBJECT_0) {
        CloseHandle(W.th);
        W.th = NULL;
    } else {
        fprintf(stderr, "bench: the watchdog thread did not exit;"
                        " leaving it running to process exit\n");
    }
#else
    if (W.have_th) { pthread_join(W.th, NULL); W.have_th = 0; }
#endif
}
