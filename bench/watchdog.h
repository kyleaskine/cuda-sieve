/* watchdog.h -- "where is it stuck?", answered from outside the stall.
 *
 * WHY THIS EXISTS. A siever that stops making progress but keeps its process
 * alive tells you nothing: the \r progress line is the last one it printed,
 * the run log's last record is up to five minutes old, and both were written
 * by the very loop that is no longer running. Every instrument in the build is
 * on the stalled thread, so a stall silences all of them at once. The one
 * thing that can still speak is a second thread that was never in the stall.
 *
 * So this is a heartbeat, not a tracer. The sieve thread stores a pointer to a
 * static phase name and bumps a counter -- two stores, no lock, no allocation,
 * nothing that can itself block -- and a watchdog thread wakes twice a second
 * to see whether the counter moved. If it has not moved for the stall
 * threshold, it prints what the sieve thread was last doing, for which
 * (q, rho) and which slab, together with the GPU's utilisation and board watts
 * from NVML.
 *
 * THOSE LAST TWO NUMBERS ARE THE POINT. A stall in this program is one of two
 * completely different bugs and the phase label alone does not separate them:
 *
 *   util pinned near 100%  -> a KERNEL is not terminating. The host is parked
 *                             in cudaEventSynchronize doing exactly what it
 *                             was told; the loop that never exits is on the
 *                             device, in whichever kernel the phase names.
 *   util at/near 0%        -> the HOST is looping, waiting on something that
 *                             will not arrive, or blocked in the driver.
 *
 * That single bit decides where to look next, and it is unavailable after the
 * fact from any log this program writes.
 *
 * IT REPORTS FIRST AND KILLS ONLY MUCH LATER. Two thresholds, an order of
 * magnitude apart, because they answer different questions. The report
 * threshold is "is something wrong?" and a false positive there costs eight
 * lines of stderr, so it sits just past the slowest legitimate phase. The kill
 * threshold is "is this host ever coming back?" and a false positive there
 * costs the special-q in flight, so it sits far beyond anything this program
 * has ever legitimately taken.
 *
 * The kill exists because a frozen process is the worst outcome available.
 * Hardware that has started to fail -- a marginal VRAM chip, a link that
 * resets -- does not always return an error a CUDA_CHECKED can see; it can
 * simply stop answering, and then the run holds its work-unit lease, its card
 * and its output file for as long as nobody is watching. Exiting
 * BENCH_EXIT_STALLED hands all three back and tells a work client to reissue
 * elsewhere, and the last checkpoint is still on disk, so the cost is one
 * replayed special-q.
 *
 * It never touches CUDA and never writes to the run log. The exit is
 * bench_fast_exit -- a bare _exit -- for the same reason the second-^C escape
 * hatch is: unwinding through a wedged driver is exactly the thing that will
 * not complete, and flushing stdio from a second thread while the first may be
 * mid-write is not safe. Everything worth keeping was fsynced at the last
 * checkpoint.
 *
 * THREAD SAFETY is the weak kind on purpose. The setters are plain volatile
 * stores of naturally aligned scalars; the reader can observe a phase name
 * from one side of an update and a slab index from the other. That is fine for
 * the case it is built for -- during a real stall nothing is being written at
 * all, so the snapshot is exact -- and paying for atomics on a path that runs
 * inside the per-slab loop is not.
 */
#ifndef BENCH_WATCHDOG_H
#define BENCH_WATCHDOG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Arms the watchdog and starts its thread. `stall_s` <= 0 disables everything,
 * including the setters' stores and the kill, and returns 0. `kill_s` <= 0
 * leaves it a pure reporter; otherwise a stall lasting that long exits the
 * process with BENCH_EXIT_STALLED -- but only after wd_arm_kill().
 *
 * `logpath` is optional: reports always go to stderr, and additionally to that
 * file (appended, flushed per report) when it is non-NULL, because a siever
 * run under a work client may have its stderr collected somewhere the operator
 * will not think to look.
 *
 * NVML is not opened here. Reports call runlog_gpu_util()/runlog_gpu_watts(),
 * so the GPU columns read "n/a" unless the caller has already bound telemetry
 * with runlog_gpu_bind() -- which bench does whenever the watchdog is armed,
 * with or without --log.
 *
 * Returns 0 when the thread is running, -1 if it could not be started (already
 * warned; the run continues without a watchdog). */
int  wd_start(double stall_s, double kill_s, const char *logpath);

/* Arms the kill. Reporting starts at wd_start; the kill does NOT, because
 * startup -- factor-base generation, a multi-gigabyte resume scan -- is a
 * legitimately slow stretch under a single coarse phase label, and killing a
 * healthy run there would be the watchdog causing the failure it exists to
 * report. run_pipeline arms it on entering the band loop, where every phase is
 * milliseconds and a stall really is a stall. A startup stall still reports;
 * it just does not end the process. */
void wd_arm_kill(void);

/* Stops the thread and joins it. Safe to call when disarmed or already
 * stopped. Must be called before runlog_close(), which unloads NVML. */
void wd_stop(void);

/* The heartbeat. `name` must be a string literal or otherwise outlive the run:
 * the pointer is stored, not the bytes. Every call also counts as progress,
 * so a loop that revisits the same phase still shows the counter moving. */
void wd_phase(const char *name);

/* The phase last set, or NULL when disarmed: for a wait that labels itself
 * and then puts the caller's phase back (cofq_writer_wait). */
const char *wd_phase_now(void);

/* Context that makes a phase name actionable. Neither counts as progress on
 * its own -- they are labels for the next report, not a heartbeat -- but both
 * are cheap enough to call per q and per slab. */
void wd_q(unsigned long long q, unsigned long long rho,
          unsigned long long nqdone);
void wd_slab(unsigned slab, unsigned nslab);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_WATCHDOG_H */
