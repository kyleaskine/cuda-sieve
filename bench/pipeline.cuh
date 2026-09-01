/* ======================= the production pipeline ========================= *
 *
 * Both sides in one process. `bench`'s run_bench remains the measurement
 * harness -- every gate closed so far lives there and every command in
 * RESULTS.md still reproduces -- and this is the path that becomes the siever.
 *
 * The difference that matters is capability, not speed. The two survivor
 * bitmaps stay device-resident, so the intersection, the rank scan and the
 * (x,a,b) emission happen ONCE instead of once per side, and the two sides'
 * cofactors are joined in memory. Nothing here produces a candidate the
 * two-process path could not; what it can do is produce a *stream* of them
 * across many special-q, which is what the cofactor queue needs and what a
 * file round trip between two single-side processes cannot supply.
 *
 * Included at the end of bench_kernels.cu so it can use the static helpers
 * already defined there (run_td_stage, td_build_*).
 */
#ifndef CUDA_SIEVE_PIPELINE_CUH
#define CUDA_SIEVE_PIPELINE_CUH

#include "ckpt.h"
#include "runlog.h"
#include <signal.h>

/* ---- clean stop -------------------------------------------------------- *
 *
 * A band runs for days and gets interrupted. Without a handler the process
 * dies inside a q with the stdio buffer unwritten and the last line possibly
 * torn, so the first signal only sets a flag: the band loop notices it at the
 * top of the next q -- after the next (q, rho) has been pulled but before any
 * work is done for it -- drains the cofactor queue, fsyncs, and records that
 * pair as the resume point. A planned stop then loses nothing at all.
 *
 * The second signal is the escape hatch for a drain that is itself stuck. It
 * skips the checkpoint, which is safe: the previous one is still a valid, just
 * older, resume point. _exit is used because it is async-signal-safe and
 * flushing stdio from a handler is not. */
static volatile sig_atomic_t g_pipe_stop = 0;

/* Called from signal context on POSIX and from the console-control thread on
 * Windows, so it touches nothing but the flag. The ++ is not atomic against
 * the sieve thread on Windows, which makes the second-request escape hatch
 * best-effort there rather than exact -- a lost increment costs one extra
 * request, never a missed stop. */
static void pipe_request_stop(void)
{
    if (++g_pipe_stop >= 2) bench_fast_exit(130);
}

/* Record a resume point.
 *
 * THE PRECONDITION IS THE WHOLE DESIGN: the caller must guarantee that the
 * relation file holds a whole number of special-q. Under --cofactor that is
 * true only at a whole-q boundary with no unflushed candidates belonging to
 * completed work. The pipeline deliberately records such a point before q0,
 * after a flush performed before any slab of the current q has been emitted,
 * and after an explicit clean-stop drain. A flush after an earlier slab of the
 * current q may leave a partial q in the file and is NOT checkpointable; resume
 * truncates that tail to the previous whole-q checkpoint and replays the q.
 * Without --cofactor the host join loop writes each q as it finishes, so any q
 * boundary will do. */
static int pipe_checkpoint(const poly_t *P, const bench_cfg_t *cfg, ckpt_t *ck,
                           FILE *fr, FILE *fc, const qsel_t *next,
                           unsigned long long nrel, unsigned long long nqdone)
{
    static int warned_no_id = 0;
    int64_t ro, co = 0;
    bench_file_id_t cid;
    if (!fr || !cfg->relations) return 0;
    /* fsync before the sidecar, never after: the checkpoint asserts these
     * bytes are on the platter, and a sidecar naming bytes that are still in
     * the page cache is worse than no sidecar at all. */
    if (fflush(fr) || bench_sync_stream(fr)) { perror("relations"); return -1; }
    if (fc && (fflush(fc) || bench_sync_stream(fc))) {
        perror("candidates"); return -1;
    }
    if ((ro = bench_tell(fr)) < 0) { perror("relations"); return -1; }
    if (fc && (co = bench_tell(fc)) < 0) { perror("candidates"); return -1; }
    if (fc) {
        /* Identity from the open handle, not from a stat buffer: st_ino is a
         * hardcoded 0 under MSVC, so the (dev, ino) this used to record would
         * later match any peer .part on the same drive -- and the recorded
         * identity is what stands between a checkpoint pathname and an
         * unlink. On POSIX these are still st_dev/st_ino, so an existing
         * Linux checkpoint compares equal exactly as before. */
        if (bench_file_id_stream(fc, &cid)) { perror("candidates"); return -1; }
        if (ckpt_part_path(cfg->candidates, ck->cand_part,
                           sizeof ck->cand_part))
            return -1;
        /* A filesystem that will not supply an identity leaves this 0/0, which
         * the reader treats as "unknown" and refuses to delete on. Recording
         * it is still right: the alternative is failing the checkpoint over a
         * field that only ever gates a discard. */
        if (!bench_file_id_valid(&cid) && !warned_no_id) {
            warned_no_id = 1;   /* once: this runs at every checkpoint */
            fprintf(stderr, "  warning: %s is on a filesystem that supplies no"
                    " file identity; automatic recovery will not be able to"
                    " discard it\n", ck->cand_part);
        }
        ck->cand_dev = (unsigned long long)cid.volume;
        ck->cand_ino = (unsigned long long)cid.index;
    } else {
        ck->cand_part[0] = '\0';
        ck->cand_dev = ck->cand_ino = 0;
    }
    ck->next_q   = next->q;
    ck->next_rho = next->rho;
    ck->rel_bytes  = (unsigned long long)ro;
    ck->cand_bytes = (unsigned long long)co;
    ck->nrel   = nrel;
    ck->nqdone = nqdone;
    return ckpt_write(cfg->relations, P, cfg, ck);
}

/* A missed checkpoint costs at most one flush of re-sieving on the next
 * resume. Aborting the band costs everything produced since the last
 * successful one, so a full disk, a read-only remount or an fsync EIO warns
 * and the sieve keeps going. Only the FIRST failure is reported: a persistent
 * one would otherwise print on every flush for days. */
static void pipe_try_checkpoint(const poly_t *P, const bench_cfg_t *cfg,
                                ckpt_t *ck, FILE *fr, FILE *fc,
                                const qsel_t *next, unsigned long long nrel,
                                unsigned long long nqdone,
                                int *written, int *warned)
{
    if (pipe_checkpoint(P, cfg, ck, fr, fc, next, nrel, nqdone) == 0) {
        *written = 1;
        return;
    }
    if (!(*warned)++)
        fprintf(stderr,
                "\n  WARNING: cannot write the resume checkpoint for %s.\n"
                "  Sieving continues, but a crash will resume from an older"
                " point, or leave an\n"
                "  unresumable .part if none has been written yet.\n",
                cfg->relations);
}

typedef struct {
    uint32_t *primes, *roots;
    plat_t   *plat;
    /* Present only in the slabbed specialization. fill advances walk_cur into
     * walk_next; resieve replays walk_cur, then the host swaps the pointers. */
    uint64_t *walk_cur, *walk_next;
    uint32_t *survbits;
    uint16_t *slice, *slice_logp;
    uint32_t *sp, *srt, *sg;
    uint16_t *slp;
    uint32_t  nslice_pow2, nsmall, nblk, nwrp, nfb;
    size_t    apply_smem;
    norm_t    N;
    uint32_t  CINIT, BOUND, tconst, nsurv;
    /* persistent across special-q */
    uint32_t *hsp, *hsrt, *hsg; uint16_t *hslp;
    uint32_t *d_nsurv, *d_nproj;
    unsigned long long *d_nlost;
    cudaEvent_t ev[4];
} pside_t;

static void pside_free(pside_t *S)
{
    cudaFree(S->primes); cudaFree(S->roots); cudaFree(S->plat);
    cudaFree(S->walk_cur); cudaFree(S->walk_next);
    cudaFree(S->survbits); cudaFree(S->slice); cudaFree(S->slice_logp);
    cudaFree(S->sp); cudaFree(S->srt); cudaFree(S->sg); cudaFree(S->slp);
    cudaFree(S->d_nsurv);
    cudaFree(S->d_nproj); cudaFree(S->d_nlost);
    if (S->hsp)  cudaFreeHost(S->hsp);
    if (S->hsrt) cudaFreeHost(S->hsrt);
    if (S->hsg)  cudaFreeHost(S->hsg);
    if (S->hslp) cudaFreeHost(S->hslp);
    for (int k = 0; k < 4; k++) if (S->ev[k]) cudaEventDestroy(S->ev[k]);
    memset(S, 0, sizeof(*S));
}

/* Records the fill will produce, for sizing the bucket array. */
static uint64_t pipe_est_records(const fb_t *fb, uint32_t xmax)
{
    double acc = 0;
    for (uint32_t i = 0; i < fb->n; i++) acc += (double)xmax / fb->primes[i];
    return (uint64_t)(acc * 1.15) + 4096;
}

/* One progress estimator shared by the terminal reporter and BOINC.  Its
 * priority matches the user-visible progress line: a relation target is the
 * goal when present; otherwise use a bounded q count/range.  Open-ended work
 * is required by bench_main to have either --target-rels or --nq, so every
 * production band has a denominator. */
static double pipe_progress_fraction(const bench_cfg_t *cfg, int streaming,
                                     uint32_t nq, uint32_t nqdone,
                                     uint64_t current_q,
                                     unsigned long long relations)
{
    double fraction = 0.0;

    if (cfg->target_rels) {
        fraction = (double)relations / (double)cfg->target_rels;
    } else if (streaming && cfg->nq_max) {
        fraction = (double)nqdone / (double)cfg->nq_max;
    } else if (streaming && cfg->qmax && cfg->qmax >= cfg->qmin &&
               current_q >= cfg->qmin) {
        const double span = (double)(cfg->qmax - cfg->qmin) + 1.0;
        fraction = ((double)(current_q - cfg->qmin) + 1.0) / span;
    } else if (nq) {
        fraction = (double)nqdone / (double)nq;
    }

    if (!isfinite(fraction) || fraction < 0.0) return 0.0;
    return fraction > 1.0 ? 1.0 : fraction;
}

/* Everything that does NOT depend on the special-q: the factor base upload,
 * the slice tables, and the pinned staging. Hoisted out of the per-q path so a
 * band of q measures sieving rather than allocation churn. */
template <bool SLABBED>
static int pipe_side_init(const fb_t *fb, const fb_t *fbs,
                          const bench_cfg_t *cfg, uint32_t bound,
                          uint32_t xmax_alloc, size_t optin_smem_limit,
                          pside_t *S)
{
    const uint32_t nbitword = xmax_alloc >> 5;
    uint16_t *hslice = NULL, *hlogp = NULL;
    int rc = -1;

#define SIDE_INIT_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    memset(S, 0, sizeof(*S));
    S->CINIT = 4096u;
    S->BOUND = bound;
    S->nfb = fb->n;
    /* Do not rely on every present and future loader remembering the lattice
     * transform's prime-power precondition. Loaders/builders earn this cookie
     * through fb_validate(); subsets preserve it. Refuse before the first
     * host transform or device upload if that proof is absent. */
    if (!fb_is_transform_validated(fb) ||
        (fbs && fbs->n && !fb_is_transform_validated(fbs))) {
        fprintf(stderr,
                "pipe_side_init: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or uploading it\n");
        goto done;
    }
    SIDE_INIT_CK(cudaMalloc(&S->primes, (size_t)fb->n * 4));
    SIDE_INIT_CK(cudaMalloc(&S->roots,  (size_t)fb->n * 4));
    SIDE_INIT_CK(cudaMalloc(&S->plat,   (size_t)fb->n * sizeof(plat_t)));
    if constexpr (SLABBED) {
        SIDE_INIT_CK(cudaMalloc(&S->walk_cur,  (size_t)fb->n * sizeof(uint64_t)));
        SIDE_INIT_CK(cudaMalloc(&S->walk_next, (size_t)fb->n * sizeof(uint64_t)));
    }
    SIDE_INIT_CK(cudaMalloc(&S->survbits, (size_t)nbitword * 4));
    SIDE_INIT_CK(cudaMemcpy(S->primes, fb->primes, (size_t)fb->n * 4,
                            cudaMemcpyHostToDevice));
    SIDE_INIT_CK(cudaMemcpy(S->roots, fb->roots, (size_t)fb->n * 4,
                            cudaMemcpyHostToDevice));

    if (fb_build_slices(fb, &hslice, &hlogp, &S->nslice_pow2) < 0) {
        report_slice_build_error();
        goto done;
    }
    {
        const size_t smem = ((size_t)1 << cfg->log_region) * 2 +
                            (size_t)S->nslice_pow2 * sizeof(*hlogp);
        if (smem > optin_smem_limit) {
            fprintf(stderr, "  apply needs %zu B of shared memory for %u"
                    " padded slices; selected device supports at most %zu B"
                    " opt-in per block\n",
                    smem, S->nslice_pow2, optin_smem_limit);
            goto done;
        }
        /* cudaFuncSetAttribute is a property of the kernel specialization, not
         * of this side.  Remember the requirement here; run_pipeline configures
         * k_apply once, after both sides are initialized, to the larger value. */
        S->apply_smem = smem;
    }
    SIDE_INIT_CK(cudaMalloc(&S->slice, (size_t)fb->n * 2));
    SIDE_INIT_CK(cudaMalloc(&S->slice_logp, (size_t)S->nslice_pow2 * 2));
    SIDE_INIT_CK(cudaMemcpy(S->slice, hslice, (size_t)fb->n * 2,
                            cudaMemcpyHostToDevice));
    SIDE_INIT_CK(cudaMemcpy(S->slice_logp, hlogp,
                            (size_t)S->nslice_pow2 * 2,
                            cudaMemcpyHostToDevice));
    free(hslice); hslice = NULL;
    free(hlogp); hlogp = NULL;

    /* PINNED staging for the per-q small-sieve tables, allocated ONCE. The
     * 2026-08-04 review measured pageable staging here at 6.2 ms per side for
     * 54 KB; cudaHostAlloc per special-q would be worse than either. */
    if (cfg->small_sieve && fbs && fbs->n) {
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hsp, (size_t)fbs->n * 4,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hsrt, (size_t)fbs->n * 4,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hslp, (size_t)fbs->n * 2,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hsg, (size_t)fbs->n * 4,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaMalloc(&S->sp,  (size_t)fbs->n * 4));
        SIDE_INIT_CK(cudaMalloc(&S->srt, (size_t)fbs->n * 4));
        SIDE_INIT_CK(cudaMalloc(&S->sg,  (size_t)fbs->n * 4));
        SIDE_INIT_CK(cudaMalloc(&S->slp, (size_t)fbs->n * 2));
    }
    SIDE_INIT_CK(cudaMalloc(&S->d_nsurv, 4));
    SIDE_INIT_CK(cudaMalloc(&S->d_nproj, 4));
    SIDE_INIT_CK(cudaMalloc(&S->d_nlost, 8));
    for (int k = 0; k < 4; k++) SIDE_INIT_CK(cudaEventCreate(&S->ev[k]));
    rc = 0;

done:
    free(hslice);
    free(hlogp);
#undef SIDE_INIT_CK
    return rc;
}

/* pipe_side_prepare_q's third return value: this special-q cannot be sieved at
 * this build's BN_LIMBS, but the band continues. Negative values stay reserved
 * for real failures, so `< 0` remains "the band is over". An enum, not a
 * #define, so it does not leak an unscoped macro out of this header. */
enum { PIPE_Q_SKIP = 1 };

/* Skips that end the run. Past this the build is simply too narrow for the job
 * and the band would emit systematically deficient output; ./normscan says so
 * up front and names the width to rebuild with.
 *
 * An ABSOLUTE count, and deliberately so: a build that normscan has cleared for
 * the band should skip ZERO q, because the projection sits well under the limit.
 * Any skip at all means the width was chosen without the survey or the band
 * moved; a hundred means it was chosen wrongly. Note the two ways this net has
 * holes -- nqskip is not checkpointed, so a resumed band starts counting again,
 * and a work unit of a few hundred q can never reach the cap. Both are
 * acceptable only because normscan is supposed to make the whole situation
 * unreachable; neither should be relied on as the primary defence. */
#ifndef PIPE_SKIP_MAX
#define PIPE_SKIP_MAX 100
#endif

/* Per-special-q work that is independent of the slab: transform the small
 * line-sieve tables, set up the norm over the FULL J range, and transform the
 * bucketed factor base once.  In SLABBED mode k_transform also seeds the first
 * slab's continuation walk. */
template <bool SLABBED>
static int pipe_side_prepare_q(const fb_t *fb, const fb_t *fbs,
                               const qlat_t *L, const poly_t *POLY,
                               const bench_cfg_t *cfg, int side, double scale,
                               int blocks, pside_t *S, float *t_transform,
                               double *t_host)
{
    uint32_t *idx = NULL, *tp = NULL, *trt = NULL, *tg = NULL;
    uint16_t *tlp = NULL;
    double h0 = host_ms();
    int rc = -1;
    /* PIPE_Q_SKIP is a THIRD outcome, distinct from both success and failure:
     * this one special-q cannot be sieved at this build's width, but the band
     * can continue past it. See the norm-width check below. */

#define PERQ_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    if (cfg->small_sieve && fbs && fbs->n) {
        uint32_t *hsp = S->hsp, *hsrt = S->hsrt, *hsg = S->hsg;
        uint16_t *hslp = S->hslp;
        uint32_t k = 0;
        for (uint32_t i = 0; i < fbs->n; i++) {
            uint32_t rt, g, m = pl_transform_enc(fbs->primes[i], fbs->roots[i],
                                                 L->a0, L->a1, L->b0, L->b1,
                                                 &rt, &g);
            hsp[k] = m;
            hsrt[k] = rt;
            hsg[k] = g;
            hslp[k] = fbs->logp[i];
            k++;
        }
        S->nsmall = k;
        if (k) {
            idx = (uint32_t *)malloc((size_t)k * sizeof(*idx));
            tp  = (uint32_t *)malloc((size_t)k * sizeof(*tp));
            trt = (uint32_t *)malloc((size_t)k * sizeof(*trt));
            tg  = (uint32_t *)malloc((size_t)k * sizeof(*tg));
            tlp = (uint16_t *)malloc((size_t)k * sizeof(*tlp));
            if (!idx || !tp || !trt || !tg || !tlp) {
                fprintf(stderr,
                        "pipe_side_prepare_q: out of memory sorting %u small-sieve entries\n",
                        k);
                goto done;
            }
            for (uint32_t i = 0; i < k; i++) idx[i] = i;
            std::stable_sort(idx, idx + k,
                             [hsp](uint32_t x, uint32_t y) {
                                 return hsp[x] < hsp[y];
                             });
            for (uint32_t i = 0; i < k; i++) {
                const uint32_t z = idx[i];
                tp[i] = hsp[z];
                trt[i] = hsrt[z];
                tg[i] = hsg[z];
                tlp[i] = hslp[z];
            }
            memcpy(hsp, tp, (size_t)k * sizeof(*hsp));
            memcpy(hsrt, trt, (size_t)k * sizeof(*hsrt));
            memcpy(hsg, tg, (size_t)k * sizeof(*hsg));
            memcpy(hslp, tlp, (size_t)k * sizeof(*hslp));
            free(idx); idx = NULL;
            free(tp); tp = NULL;
            free(trt); trt = NULL;
            free(tg); tg = NULL;
            free(tlp); tlp = NULL;
        }
        S->nblk = S->nwrp = 0;
        for (uint32_t i = 0; i < k && hsp[i] < SS_BLOCK_CUT; i++) S->nblk = i + 1;
        for (uint32_t i = 0; i < k && hsp[i] < SS_WARP_CUT; i++) S->nwrp = i + 1;
        PERQ_CK(cudaMemcpy(S->sp, hsp, (size_t)k * sizeof(*hsp),
                           cudaMemcpyHostToDevice));
        PERQ_CK(cudaMemcpy(S->srt, hsrt, (size_t)k * sizeof(*hsrt),
                           cudaMemcpyHostToDevice));
        PERQ_CK(cudaMemcpy(S->sg, hsg, (size_t)k * sizeof(*hsg),
                           cudaMemcpyHostToDevice));
        PERQ_CK(cudaMemcpy(S->slp, hslp, (size_t)k * sizeof(*hslp),
                           cudaMemcpyHostToDevice));
    }

    /* Side 0's norms are G(x) = Y1*x + Y0, a degree-1 form. run_bench gets
     * that by mutating POLY in the caller; the pipeline must not, because
     * run_td_stage still needs the degree-5 coefficients for side 1. Make the
     * variant here and leave POLY alone. */
    {
        poly_t P = *POLY;
        if (side == 0) {
            P.deg = 1;
            P.c[0] = P.y0;
            P.c[1] = P.y1;
            for (int z = 2; z < BENCH_NCOEFF; z++) P.c[z] = 0.0;
        }
        norm_setup(&S->N, &P, L, cfg->logI, cfg->J, scale,
                   side == cfg->sq_side);
        {
            const double bits = norm_exact_bound_bits(&S->N);
            if (!norm_fits_exact(&S->N, BN_LIMBS * 32)) {
                /* SKIP THIS q, DO NOT KILL THE BAND.
                 *
                 * Aborting here was strictly the worst option available. It is
                 * not even a clean stop: the checkpoint records the failing
                 * (q,rho), so "rerun to resume" returns to this exact q and the
                 * band can never advance -- and under BOINC the work unit fails,
                 * is reissued, and fails identically on the next client. Against
                 * that, one special-q out of the tens of millions in a band is a
                 * rounding error in yield.
                 *
                 * WHAT THIS COSTS, because it is not free. "A wider build is
                 * byte-identical to a narrower one on any job the narrower one
                 * could run" was true before this change, and it is the property
                 * that qualified BN_LIMBS 12 -- the 8- and 12-limb relation files
                 * were diffed. Skipping weakens it to "byte-identical on any job
                 * normscan clears at the narrower width", which is still
                 * checkable, but only because normscan now exists to certify that
                 * condition up front. Do not weaken it further.
                 *
                 * Loud, counted, and fatal in bulk: every skip prints, the total
                 * lands in the band summary, and PIPE_SKIP_MAX of them ends the
                 * run. A hundred skips is not a tail any more, it is the wrong
                 * build for the job, and quietly emitting a whole band of
                 * deficient output would be worse than stopping. */
                const int need = bn_limbs_for_bits(bits);
                /* NAMES rho, NOT JUST q. A q carries up to deg roots and it is
                 * the (q,rho) LATTICE that overflows, not the q: on the 2,1139+
                 * octic, (600002827, 369315816) needs 260.75 bits while
                 * (600002827, 272545945) needs 238.20 and sieves normally.
                 * Printing q alone leaves no way to reproduce or exclude the
                 * pair -- and the log position cannot supply it either, because
                 * this goes to stderr unbuffered while the per-q sieve lines go
                 * to stdout block-buffered, so in a redirected run this line
                 * surfaces above its own iteration's header. */
                runlog_warn("  ** SKIPPED q=%llu rho=%llu: exact side-%d"
                            " degree-%d norm may require %.2f bits; the"
                            " trial-division type holds %d",
                            (unsigned long long)L->q,
                            (unsigned long long)L->rho, side, P.deg, bits,
                            BN_LIMBS * 32);
                if (need)
                    runlog_warn("     rebuild with `make BN_LIMBS=%d` to sieve it"
                                " (run ./normscan over the band first)", need);
                else
                    runlog_warn("     no supported BN_LIMBS is wide enough"
                                " (the maximum is 16, i.e. 512 bits)");
                rc = PIPE_Q_SKIP;
                goto done;
            }
        }
    }
    /* CINIT and BOUND were checked once, before any allocation, and are
     * invariant across the special-q band. Keeping them in the persistent
     * side context avoids reparsing or floating-point conversion per q. */
    {
        const float t = norm_target_host(&S->N, 0, cfg->J / 2);
        const int ti = (int)(t + 0.5f);
        S->tconst = (ti < 1) ? 1u :
                    ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
    }

    *t_host = host_ms() - h0;
    /* k_transform writes these unconditionally; NULL is an illegal access */
    PERQ_CK(cudaMemset(S->d_nproj, 0, 4));
    PERQ_CK(cudaMemset(S->d_nlost, 0, 8));

    PERQ_CK(cudaEventRecord(S->ev[0]));
    k_transform<SLABBED><<<blocks, cfg->threads>>>(S->primes, S->roots, S->plat, fb->n,
        cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1,
        S->d_nproj, S->d_nlost, S->walk_cur);
    PERQ_CK(cudaEventRecord(S->ev[1]));
    PERQ_CK(cudaEventSynchronize(S->ev[1]));
    PERQ_CK(cudaGetLastError());
    *t_transform = time_kernel(S->ev[0], S->ev[1]);
    rc = 0;

done:
    free(idx); free(tp); free(trt); free(tg); free(tlp);
#undef PERQ_CK
    return rc;
}

/* Area-dependent work for one side and one slab. The bucket array is passed
 * in and REUSED between sides. SLABBED=false compiles to the pre-slab fill and
 * apply kernels; no walk-state read/write or global-j adjustment survives. */
template <bool SLABBED>
static int pipe_side_sieve_slab(const fb_t *fb, const bench_cfg_t *cfg,
                                int side, uint32_t xmax, uint32_t j_base,
                                uint8_t *d_bucket, uint32_t *d_cursor,
                                uint32_t cap, uint32_t *d_overflow,
                                int fblocks, int fthreads, pside_t *S,
                                float *t_fill, float *t_apply)
{
    const int log_region = cfg->log_region;
    const uint32_t nregion = xmax >> log_region;
    const uint32_t nbitword = xmax >> 5;
    const int athr = cfg->apply_threads ? cfg->apply_threads : 512;
    int rc = -1;

#define SLAB_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    /* Preserve the pre-slab timing boundary: fill includes clearing the bucket
     * cursors/overflow counter, just as pipe_side_perq did. */
    SLAB_CK(cudaEventRecord(S->ev[1]));
    SLAB_CK(cudaMemset(d_cursor, 0, (size_t)nregion * 4));
    SLAB_CK(cudaMemset(d_overflow, 0, 4));
    k_fill_atomic<4, SLABBED><<<fblocks, fthreads>>>(
        S->plat, S->slice, fb->n, xmax, cfg->logI, log_region,
        d_cursor, d_bucket, cap, d_overflow, S->walk_cur, S->walk_next);
    SLAB_CK(cudaEventRecord(S->ev[2]));
    SLAB_CK(cudaMemset(S->d_nsurv, 0, 4));
    /* Provably redundant on every production geometry: k_apply's warp-ballot
     * path stores every word of the slab bitmap unconditionally, and this
     * function has both inputs its guard tests (log_region and athr, above), so
     * `if (cfg->log_region < 5 || (athr & 31))` would skip the clear safely.
     * Measured cost of keeping it: ~512 MB of device writes per q at I16/J32768
     * x 4 slabs x 2 sides, about 0.7 ms against ~440, i.e. 0.16%.
     *
     * It stays anyway, and the honest reason is not "avoiding coupling" -- the
     * coupling already exists implicitly. It is that the guard would restate
     * k_apply's launch-shape condition in a second place, where it can rot out
     * of sync with the kernel's, and the payoff is 0.16%. Revisit if the clear
     * ever shows up in a profile. */
    SLAB_CK(cudaMemset(S->survbits, 0, (size_t)nbitword * 4));
    k_apply<16, 1, NORM_HORNER, SLABBED><<<nregion, athr, S->apply_smem>>>(
        (const uint32_t *)d_bucket, d_cursor, cap, cfg->logI, log_region,
        S->slice_logp, S->nslice_pow2, S->N, S->CINIT, S->CINIT - S->BOUND,
        S->tconst, NULL, S->d_nsurv, NULL, 0xFFFFFFFFu,
        S->sp, S->srt, S->sg, S->slp, S->nsmall, S->nblk, S->nwrp,
        0xFFFFFFFFu, NULL, S->survbits, cfg->not_both_even, j_base);
    SLAB_CK(cudaEventRecord(S->ev[3]));
    SLAB_CK(cudaEventSynchronize(S->ev[3]));
    SLAB_CK(cudaGetLastError());
    *t_fill  = time_kernel(S->ev[1], S->ev[2]);
    *t_apply = time_kernel(S->ev[2], S->ev[3]);
    SLAB_CK(cudaMemcpy(&S->nsurv, S->d_nsurv, 4, cudaMemcpyDeviceToHost));
    {
        uint32_t hov = 0;
        SLAB_CK(cudaMemcpy(&hov, d_overflow, 4, cudaMemcpyDeviceToHost));
        if (hov) {
            runlog_warn("  side %d: bucket array OVERFLOWED by %u records",
                        side, hov);
            goto done;
        }
    }
    rc = 0;

done:
#undef SLAB_CK
    return rc;
}

/* ======================= the production TD path ========================== *
 *
 * run_td_stage is the MEASUREMENT harness. Calling it per special-q, per side,
 * is what made the pipeline's post-sieve cost 222 ms/q: it runs the rank scan,
 * the emission, the resieve and the classification three times each for a
 * best-of-three, plus two diagnostic variants of k_td that exist only to
 * separate the congruence test from the divisions it triggers, plus a dense
 * recording pass over every survivor -- and it rebuilds the rank scan and the
 * emission for the SECOND side even though both are functions of the shared
 * two-sided bitmap alone.
 *
 * This is the same arithmetic with the harness taken out. Four things change:
 *
 *   1. Rank scan, emission and the resieve summary run ONCE per q, not once
 *      per side. They read the two-sided bitmap; neither side owns them.
 *   2. Every buffer is allocated once for the whole band, not once per q.
 *      Only a survivor count larger than anything seen so far reallocates.
 *   3. The two sides' acceptances are intersected ON DEVICE, and the recording
 *      pass runs over that compacted list. Recording is the only pass that
 *      writes 64 words per thread; over ~240,000 survivors that is a 61 MB
 *      matrix per side, and over the ~1,900 joint candidates it is 486 KB.
 *   4. The reconstruction gate is an explicitly separate phase on the first q,
 *      excluded from the band timing rather than hidden inside it.
 *
 * What does NOT change is the arithmetic: same kernels, same production
 * settings (resieve unroll 4, 1 summary bit per 64 positions). The output of
 * this path is diffed byte-for-byte against the harness path, which is the
 * only reason a restructure this large is safe to make.
 */

#define PIPE_K 16          /* large primes kept per survivor, as run_td_stage */

typedef struct {
    double rank, emit, summary, resieve, td, classify, compact, record;
    double readback, join, hostq, cofac;
} pipe_tm_t;

typedef struct {
    uint32_t nbitword, ngroup, nb, nsumword;
    uint32_t scap, ccap;          /* survivors, joint candidates */
    /* shared by both sides: functions of the two-sided bitmap alone */
    uint32_t *d_cnt, *d_gbase, *d_bsum, *d_sum;
    uint32_t *d_x;  int64_t *d_a, *d_b;
    uint32_t *d_flags;  unsigned long long *d_ovf;
    /* joint acceptance */
    uint32_t *d_aflag, *d_aoff, *d_absum, *d_sel, *d_nacc;
    /* per side */
    uint32_t  *d_plist[2], *d_pcnt[2];
    bn_t      *d_cof[2];
    uint8_t   *d_cofbits[2], *d_status[2];
    tdpoly_t  *d_poly[2];
    uint32_t  *d_stats;          /* {already-relations, candidates, overflows} */
    tdsmall_t *d_sm[2], *h_sm[2];
    uint32_t   nsm[2], nsmcap[2];
    tdpoly_t   hpoly[2];
    /* compacted candidate records, device then pinned host */
    int64_t  *d_ca, *d_cb, *h_ca, *h_cb;
    bn_t     *d_ccof[2], *h_ccof[2];
    uint8_t  *d_cbits[2], *h_cbits[2];
    uint32_t *d_cfac[2], *h_cfac[2], *d_cfn[2], *h_cfn[2];
    cudaEvent_t ev[16];
} pipe_td_t;

static void pipe_td_free(pipe_td_t *C)
{
    cudaFree(C->d_cnt); cudaFree(C->d_gbase); cudaFree(C->d_bsum);
    cudaFree(C->d_sum); cudaFree(C->d_x); cudaFree(C->d_a); cudaFree(C->d_b);
    cudaFree(C->d_flags); cudaFree(C->d_ovf);
    cudaFree(C->d_aflag); cudaFree(C->d_aoff); cudaFree(C->d_absum);
    cudaFree(C->d_sel); cudaFree(C->d_nacc);
    cudaFree(C->d_ca); cudaFree(C->d_cb);
    if (C->h_ca) cudaFreeHost(C->h_ca);
    if (C->h_cb) cudaFreeHost(C->h_cb);
    cudaFree(C->d_stats);
    for (int s = 0; s < 2; s++) {
        cudaFree(C->d_plist[s]); cudaFree(C->d_pcnt[s]);
        cudaFree(C->d_cof[s]); cudaFree(C->d_cofbits[s]); cudaFree(C->d_status[s]);
        cudaFree(C->d_poly[s]); cudaFree(C->d_sm[s]);
        cudaFree(C->d_ccof[s]); cudaFree(C->d_cbits[s]);
        cudaFree(C->d_cfac[s]); cudaFree(C->d_cfn[s]);
        if (C->h_sm[s])   cudaFreeHost(C->h_sm[s]);
        if (C->h_ccof[s]) cudaFreeHost(C->h_ccof[s]);
        if (C->h_cbits[s]) cudaFreeHost(C->h_cbits[s]);
        if (C->h_cfac[s]) cudaFreeHost(C->h_cfac[s]);
        if (C->h_cfn[s])  cudaFreeHost(C->h_cfn[s]);
    }
    for (int k = 0; k < 16; k++) if (C->ev[k]) cudaEventDestroy(C->ev[k]);
    memset(C, 0, sizeof(*C));
}

/* Survivor-sized buffers. Grown, never shrunk: a band's first q pays for its
 * own count plus half, and in practice nothing after it reallocates. */
static int pipe_td_grow(pipe_td_t *C, uint32_t n)
{
    uint32_t cap, nab;
    if (n <= C->scap) return 0;
    cap = n + n / 2 + 1024;
    nab = (cap + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    cudaFree(C->d_x); cudaFree(C->d_a); cudaFree(C->d_b);
    cudaFree(C->d_aflag); cudaFree(C->d_aoff); cudaFree(C->d_absum);
    C->d_x = NULL; C->d_a = NULL; C->d_b = NULL;
    C->d_aflag = C->d_aoff = C->d_absum = NULL;
    CK(cudaMalloc(&C->d_x, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_a, (size_t)cap * 8));
    CK(cudaMalloc(&C->d_b, (size_t)cap * 8));
    CK(cudaMalloc(&C->d_aflag, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_aoff, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_absum, (size_t)nab * 4));
    for (int s = 0; s < 2; s++) {
        cudaFree(C->d_plist[s]); cudaFree(C->d_pcnt[s]); cudaFree(C->d_cof[s]);
        cudaFree(C->d_cofbits[s]); cudaFree(C->d_status[s]);
        C->d_plist[s] = C->d_pcnt[s] = NULL; C->d_cof[s] = NULL;
        C->d_cofbits[s] = C->d_status[s] = NULL;
        CK(cudaMalloc(&C->d_plist[s], (size_t)cap * PIPE_K * 4));
        CK(cudaMalloc(&C->d_pcnt[s], (size_t)cap * 4));
        CK(cudaMalloc(&C->d_cof[s], (size_t)cap * sizeof(bn_t)));
        CK(cudaMalloc(&C->d_cofbits[s], (size_t)cap));
        CK(cudaMalloc(&C->d_status[s], (size_t)cap));
    }
    C->scap = cap;
    return 0;
}

/* Candidate-sized buffers, device and pinned host. */
static int pipe_td_grow_cand(pipe_td_t *C, uint32_t n)
{
    uint32_t cap;
    if (n <= C->ccap) return 0;
    cap = n + n / 2 + 1024;
    cudaFree(C->d_sel); cudaFree(C->d_ca); cudaFree(C->d_cb);
    C->d_sel = NULL; C->d_ca = C->d_cb = NULL;
    if (C->h_ca) cudaFreeHost(C->h_ca);
    if (C->h_cb) cudaFreeHost(C->h_cb);
    C->h_ca = C->h_cb = NULL;
    CK(cudaMalloc(&C->d_sel, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_ca, (size_t)cap * 8));
    CK(cudaMalloc(&C->d_cb, (size_t)cap * 8));
    CK(cudaHostAlloc((void **)&C->h_ca, (size_t)cap * 8, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&C->h_cb, (size_t)cap * 8, cudaHostAllocDefault));
    for (int s = 0; s < 2; s++) {
        cudaFree(C->d_ccof[s]); cudaFree(C->d_cbits[s]);
        cudaFree(C->d_cfac[s]); cudaFree(C->d_cfn[s]);
        C->d_ccof[s] = NULL; C->d_cbits[s] = NULL;
        C->d_cfac[s] = C->d_cfn[s] = NULL;
        if (C->h_ccof[s])  cudaFreeHost(C->h_ccof[s]);
        if (C->h_cbits[s]) cudaFreeHost(C->h_cbits[s]);
        if (C->h_cfac[s])  cudaFreeHost(C->h_cfac[s]);
        if (C->h_cfn[s])   cudaFreeHost(C->h_cfn[s]);
        C->h_ccof[s] = NULL; C->h_cbits[s] = NULL;
        C->h_cfac[s] = C->h_cfn[s] = NULL;
        CK(cudaMalloc(&C->d_ccof[s], (size_t)cap * sizeof(bn_t)));
        CK(cudaMalloc(&C->d_cbits[s], (size_t)cap));
        CK(cudaMalloc(&C->d_cfac[s], (size_t)cap * TD_FMAX * 4));
        CK(cudaMalloc(&C->d_cfn[s], (size_t)cap * 4));
        CK(cudaHostAlloc((void **)&C->h_ccof[s], (size_t)cap * sizeof(bn_t),
                         cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&C->h_cbits[s], (size_t)cap, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&C->h_cfac[s], (size_t)cap * TD_FMAX * 4,
                         cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&C->h_cfn[s], (size_t)cap * 4, cudaHostAllocDefault));
    }
    C->ccap = cap;
    return 0;
}

/* q-independent setup: the exact polynomials, the rank tables sized by the
 * bitmap, and the pinned staging for the per-q small-prime table. */
static int pipe_td_init(pipe_td_t *C, const fb_t *fbs1, const fb_t *fbs0,
                        const poly_t *POLY, uint32_t nbitword)
{
    const fb_t *fbs[2];
    memset(C, 0, sizeof(*C));
    fbs[0] = fbs0; fbs[1] = fbs1;
    if (nbitword % TD_GROUP_W) {
        fprintf(stderr, "  pipeline: %u bitmap words is not a multiple of %d\n",
                nbitword, TD_GROUP_W);
        return -1;
    }
    C->nbitword = nbitword;
    C->ngroup = nbitword / TD_GROUP_W;
    C->nb = (C->ngroup + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    C->nsumword = ((nbitword / 2) + 31) / 32;
    for (int k = 0; k < 16; k++) CK(cudaEventCreate(&C->ev[k]));
    CK(cudaMalloc(&C->d_cnt, (size_t)C->ngroup * 4));
    CK(cudaMalloc(&C->d_gbase, (size_t)C->ngroup * 4));
    CK(cudaMalloc(&C->d_bsum, (size_t)C->nb * 4));
    CK(cudaMalloc(&C->d_sum, (size_t)C->nsumword * 4));
    CK(cudaMalloc(&C->d_flags, 4));
    CK(cudaMalloc(&C->d_ovf, 8));
    CK(cudaMalloc(&C->d_nacc, 4));
    CK(cudaMalloc(&C->d_stats, 12));
    CK(cudaMemset(C->d_stats, 0, 12));      /* accumulates over the whole band */
    for (int s = 0; s < 2; s++) {
        if (td_build_poly(&C->hpoly[s], POLY, s)) {
            fprintf(stderr, "  pipeline: could not parse side %d coefficients\n", s);
            return -1;
        }
        CK(cudaMalloc(&C->d_poly[s], sizeof(tdpoly_t)));
        CK(cudaMemcpy(C->d_poly[s], &C->hpoly[s], sizeof(tdpoly_t),
                      cudaMemcpyHostToDevice));
        C->nsmcap[s] = (fbs[s] && fbs[s]->n) ? fbs[s]->n : 0;
        if (C->nsmcap[s]) {
            CK(cudaHostAlloc((void **)&C->h_sm[s],
                             (size_t)C->nsmcap[s] * sizeof(tdsmall_t),
                             cudaHostAllocDefault));
            CK(cudaMalloc(&C->d_sm[s], (size_t)C->nsmcap[s] * sizeof(tdsmall_t)));
        }
    }
    return pipe_td_grow_cand(C, 8192);
}

/* Per-q, per-side host work: retransform the small-prime direct-test table
 * into the pinned buffer and upload it. Everything else about the side is
 * already resident. */
static int pipe_td_small(pipe_td_t *C, int side, const fb_t *fbs,
                         const qlat_t *L, int logI)
{
    if (!C->nsmcap[side]) { C->nsm[side] = 0; return 0; }
    C->nsm[side] = td_fill_small(fbs, L, logI, C->h_sm[side]);
    if (!C->nsm[side]) return -1;
    CK(cudaMemcpy(C->d_sm[side], C->h_sm[side],
                  (size_t)C->nsm[side] * sizeof(tdsmall_t), cudaMemcpyHostToDevice));
    return 0;
}

/* Build the two direct-test tables once per special-q.  In slabbed mode their
 * congruence constants are then translated between slabs by the tiny
 * pipe_td_advance_small() kernel, keeping the billions-of-tests inner loop
 * unchanged. */
static int pipe_td_prepare_q(pipe_td_t *C, const fb_t *fbs1,
                             const fb_t *fbs0, const qlat_t *L,
                             const bench_cfg_t *cfg, pipe_tm_t *tm)
{
    const double h0 = host_ms();
    if (pipe_td_small(C, 1, fbs1, L, cfg->logI) ||
        pipe_td_small(C, 0, fbs0, L, cfg->logI)) return -1;
    tm->hostq += host_ms() - h0;
    return 0;
}

static int pipe_td_advance_small(pipe_td_t *C, uint32_t delta_j,
                                 int blocks, int threads)
{
    if (!delta_j) return 0;
    for (int s = 0; s < 2; s++) {
        if (!C->nsm[s]) continue;
        k_tdsmall_advance<<<blocks, threads>>>(C->d_sm[s], C->nsm[s], delta_j);
        CK(cudaGetLastError());
    }
    return 0;
}

/* ---- the first-q validation phase -------------------------------------- *
 *
 * Reconstruction: the recorded factors times the residual cofactor must
 * rebuild the exact norm. This runs the DENSE recording pass and checks every
 * candidate on the side, which is what run_td_stage did -- deliberately, so
 * the check keeps its old strength rather than narrowing to the joint set the
 * production path records. It reads back 61 MB per side and rebuilds a 224-bit
 * norm per candidate on the host, so it is a separate phase whose cost is
 * reported on its own and excluded from the band average. */
template <bool SLABBED>
static int pipe_td_verify(pipe_td_t *C, int side, uint32_t n, int logI,
                          uint32_t sq, const bench_cfg_t *cfg,
                          int blocks, int threads, uint32_t j_base,
                          uint32_t *gate_found)
{
    uint32_t *d_fac = NULL, *d_faccnt = NULL;
    uint32_t *hfac = NULL, *hfn = NULL;
    uint8_t *hstat = NULL;
    bn_t *hcof = NULL;
    int64_t *ha = NULL, *hb = NULL;
    uint32_t checked = 0, bad = 0, overflowed = 0, maxfac = 0;
    const tdpoly_t *P = &C->hpoly[side];
    int rc = 0;

#define VERIFY_CK(x) do { if (CUDA_CHECKED(x)) { rc = -1; goto out; } } while (0)
    VERIFY_CK(cudaMalloc(&d_fac, (size_t)n * TD_FMAX * 4));
    VERIFY_CK(cudaMalloc(&d_faccnt, (size_t)n * 4));
    k_td<1, 1, 0, SLABBED><<<blocks, threads>>>(C->d_a, C->d_b, C->d_x, NULL, n, logI,
                                       C->d_poly[side], sq,
                                       C->d_plist[side], C->d_pcnt[side], PIPE_K,
                                       C->d_sm[side], C->nsm[side],
                                       C->d_cof[side], C->d_cofbits[side],
                                       C->d_flags, NULL, d_fac, d_faccnt, TD_FMAX,
                                       j_base);
    if (CUDA_CHECKED(cudaDeviceSynchronize()) ||
        CUDA_CHECKED(cudaGetLastError())) {
        fprintf(stderr, "  verify: recording pass failed\n");
        rc = -1;
        goto out;
    }
    hfac = (uint32_t *)malloc((size_t)n * TD_FMAX * 4);
    hfn = (uint32_t *)malloc((size_t)n * 4);
    hstat = (uint8_t *)malloc((size_t)n);
    hcof = (bn_t *)malloc((size_t)n * sizeof(bn_t));
    ha = (int64_t *)malloc((size_t)n * 8);
    hb = (int64_t *)malloc((size_t)n * 8);
    if (!hfac || !hfn || !hstat || !hcof || !ha || !hb) { rc = -1; goto out; }
    VERIFY_CK(cudaMemcpy(hfac, d_fac, (size_t)n * TD_FMAX * 4,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hfn, d_faccnt, (size_t)n * 4,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hstat, C->d_status[side], (size_t)n,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hcof, C->d_cof[side], (size_t)n * sizeof(bn_t),
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(ha, C->d_a, (size_t)n * 8,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hb, C->d_b, (size_t)n * 8,
                         cudaMemcpyDeviceToHost));

    for (uint32_t k = 0; k < n; k++) {
        bns_t acc; bn_t t;
        int64_t a, b; uint64_t ua, ub; int sa, sb;
        if (hstat[k] != COF_ACCEPT && hstat[k] != COF_SPLIT) continue;
        if (hfn[k] > maxfac) maxfac = hfn[k];
        if (hfn[k] > TD_FMAX) { overflowed++; continue; }
        checked++;
        a = ha[k]; b = hb[k];
        ua = (uint64_t)(a < 0 ? -a : a); ub = (uint64_t)(b < 0 ? -b : b);
        sa = (a < 0) ? -1 : 1; sb = (b < 0) ? -1 : 1;
        bns_zero(&acc);
        for (int d = 0; d <= P->deg; d++) {
            int sgn = P->sign[d];
            t = P->c[d];
            if (bn_is_zero(&t)) continue;
            for (int e = 0; e < d; e++) { bn_mul_u64(&t, ua); sgn *= sa; }
            for (int e = 0; e < P->deg - d; e++) { bn_mul_u64(&t, ub); sgn *= sb; }
            bns_addmag(&acc, &t, sgn);
        }
        t = hcof[k];
        for (uint32_t z = 0; z < hfn[k]; z++)
            bn_mul_u64(&t, (uint64_t)hfac[(size_t)k * TD_FMAX + z]);
        if (bn_cmp(&t, &acc.m) != 0) bad++;
    }
    printf("    side %d: factors x cofactor == norm  %u of %u  %s"
           "   (most factors %u of %d)\n",
           side, checked - bad, checked, bad ? "FAIL" : "PASS", maxfac, TD_FMAX);
    if (bad) rc = -1;
    if (overflowed) {
        fprintf(stderr, "    side %d: %u candidates had more than %d factors;"
                " raise TD_FMAX\n", side, overflowed, TD_FMAX);
        rc = -1;
    }
    if (cfg->cofgate &&
        td_gate_cofactors_part(cfg->cofgate, n, ha, hb, hcof, side,
                               gate_found) != 0) rc = -1;

out:
    free(hfac); free(hfn); free(hstat); free(hcof); free(ha); free(hb);
    cudaFree(d_fac); cudaFree(d_faccnt);
#undef VERIFY_CK
    return rc;
}

/* ---- one special-q of trial division, both sides ----------------------- */

template <bool SLABBED>
static int pipe_td_perq(pipe_td_t *C, const qlat_t *L, const bench_cfg_t *cfg,
                        const pside_t *S1, const pside_t *S0,
                        const uint32_t *d_two, uint32_t xmax,
                        uint32_t j_base, int blocks, int threads, int verify,
                        uint32_t *n_out, uint32_t *nacc_out, pipe_tm_t *tm,
                        double *t_verify, int want_host, int accumulate_stats,
                        uint32_t gate_found[2])
{
    const pside_t *S[2];
    const uint32_t lpb[2] = {cfg->lpb0, cfg->lpb};
    const uint32_t mfb[2] = {cfg->mfb0, cfg->mfb};
    const uint32_t lim[2] = {cfg->lim0, cfg->lim};
    cudaEvent_t *E = C->ev;
    uint32_t n = 0, nacc = 0, nab, hflags = 0;
    unsigned long long hovf = 0;
    const uint32_t nbitword = xmax >> 5;
    const uint32_t ngroup = nbitword / TD_GROUP_W;
    const uint32_t nb = (ngroup + TD_SCAN_BLK - 1) / TD_SCAN_BLK;

    S[0] = S0; S[1] = S1;

    if (!ngroup || nbitword % TD_GROUP_W) {
        fprintf(stderr,
                "  pipeline: slab area %u has %u bitmap words; trial division"
                " requires a nonzero multiple of %u\n",
                xmax, nbitword, (unsigned)TD_GROUP_W);
        return -1;
    }

    /* ---- phase 1: rank over the two-sided bitmap (shared) ---- */
    CK(cudaEventRecord(E[0]));
    k_group_counts<<<blocks, threads>>>(d_two, ngroup, C->d_cnt);
    k_scan_pass1<<<nb, TD_SCAN_BLK>>>(C->d_cnt, ngroup, C->d_gbase, C->d_bsum);
    k_scan_pass2<<<1, 1024>>>(C->d_bsum, nb);
    k_scan_pass3<<<nb, TD_SCAN_BLK>>>(C->d_gbase, ngroup, C->d_bsum);
    CK(cudaEventRecord(E[1]));
    CK(cudaEventSynchronize(E[1])); CK(cudaGetLastError());
    {
        uint32_t base = 0, cnt = 0;
        CK(cudaMemcpy(&base, C->d_gbase + ngroup - 1, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&cnt, C->d_cnt + ngroup - 1, 4, cudaMemcpyDeviceToHost));
        n = base + cnt;
    }
    *n_out = n; *nacc_out = 0;
    /* A slab can legitimately contain no survivors, especially a short tail.
     * Preserve the original q-level invariant after all slabs have run rather
     * than turning an empty individual slab into a failed q. */
    if (!n) return 0;
    if (pipe_td_grow(C, n)) return -1;

    /* ---- phase 2: emit, summary, then both sides ---- */
    CK(cudaMemset(C->d_flags, 0, 4));
    CK(cudaMemset(C->d_ovf, 0, 8));
    CK(cudaEventRecord(E[2]));
    k_emit_ranked<SLABBED><<<blocks, threads>>>(d_two, C->d_gbase, nbitword, cfg->logI,
                                       L->a0, L->a1, L->b0, L->b1,
                                       C->d_x, C->d_a, C->d_b, n, j_base);
    CK(cudaEventRecord(E[3]));
    /* 1 summary bit per 2 bitmap words == per 64 positions, the setting the
     * resieve sweep selected. */
    k_build_summary_g<<<blocks, threads>>>(d_two, nbitword, 2u, C->d_sum);
    CK(cudaEventRecord(E[4]));

    for (int si = 0; si < 2; si++) {
        const int side = si ? 0 : 1;          /* side 1 first, as before */
        const int e = si ? 7 : 4;
        CK(cudaMemsetAsync(C->d_pcnt[side], 0, (size_t)n * 4));
        k_resieve_scatter<4, SLABBED><<<blocks, threads>>>(
            S[side]->plat, S[side]->primes, NULL, S[side]->nfb, xmax, cfg->logI,
            C->d_sum, d_two, C->d_gbase, C->d_plist[side], C->d_pcnt[side],
            PIPE_K, C->d_ovf, 6, S[side]->walk_cur);
        CK(cudaEventRecord(E[e + 1]));
        k_td<1, 0, 0, SLABBED><<<blocks, threads>>>(
            C->d_a, C->d_b, C->d_x, NULL, n, cfg->logI, C->d_poly[side],
            side == cfg->sq_side ? (uint32_t)L->q : 0u,
            C->d_plist[side], C->d_pcnt[side], PIPE_K,
            C->d_sm[side], C->nsm[side],
            C->d_cof[side], C->d_cofbits[side], C->d_flags, NULL, NULL, NULL, 0,
            j_base);
        CK(cudaEventRecord(E[e + 2]));
        k_classify<<<blocks, threads>>>(C->d_cof[side], C->d_cofbits[side],
                                        C->d_b, n, lpb[side], mfb[side],
                                        (double)lim[side], C->d_status[side]);
        CK(cudaEventRecord(E[e + 3]));
    }

    /* ---- joint acceptance, compacted on device ---- */
    nab = (n + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    k_accept_flags<<<blocks, threads>>>(C->d_status[0], C->d_status[1], n, C->d_aflag);
    k_scan_pass1<<<nab, TD_SCAN_BLK>>>(C->d_aflag, n, C->d_aoff, C->d_absum);
    k_scan_pass2<<<1, 1024>>>(C->d_absum, nab);
    k_scan_pass3<<<nab, TD_SCAN_BLK>>>(C->d_aoff, n, C->d_absum);
    k_scatter_sel<<<blocks, threads>>>(C->d_aflag, C->d_aoff, n, C->d_sel,
                                       C->ccap, C->d_nacc);
    CK(cudaEventRecord(E[11]));
    CK(cudaEventSynchronize(E[11])); CK(cudaGetLastError());
    tm->emit     += time_kernel(E[2], E[3]);
    tm->summary  += time_kernel(E[3], E[4]);
    tm->resieve  += time_kernel(E[4], E[5]) + time_kernel(E[7], E[8]);
    tm->td       += time_kernel(E[5], E[6]) + time_kernel(E[8], E[9]);
    tm->classify += time_kernel(E[6], E[7]) + time_kernel(E[9], E[10]);
    tm->compact  += time_kernel(E[10], E[11]);
    tm->rank     += time_kernel(E[0], E[1]);

    CK(cudaMemcpy(&hflags, C->d_flags, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hovf, C->d_ovf, 8, cudaMemcpyDeviceToHost));
    if (hflags & TDF_NORM_OVERFLOW) {
        runlog_warn("  ** NORM OVERFLOW: a norm exceeded %d bits", BN_LIMBS * 32);
        return -1;
    }
    if (hflags & TDF_LIST_TRUNCATED) {
        runlog_warn("  ** %llu large-prime records past the %u/survivor cap",
                    hovf, PIPE_K);
        return -1;
    }
    CK(cudaMemcpy(&nacc, C->d_nacc, 4, cudaMemcpyDeviceToHost));
    if (nacc > C->ccap) {           /* the scatter clamped; redo it once, larger */
        if (pipe_td_grow_cand(C, nacc)) return -1;
        k_scatter_sel<<<blocks, threads>>>(C->d_aflag, C->d_aoff, n, C->d_sel,
                                           C->ccap, C->d_nacc);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    }
    *nacc_out = nacc;

    /* ---- the first q's reconstruction gate, on its own clock ---- */
    if (verify) {
        double v0 = host_ms();
        printf("\n  --- first-q validation (excluded from the band timing) ---\n");
        /* The special-q goes to whichever side carries it, matching the k_td
         * calls that produced d_cof[]. Hardcoding side 1 here survived
         * --sq-side 0 in practice -- td_divide_out is a no-op when q does not
         * divide, and the gate still passed 15,174 of 15,174 -- but the verify
         * pass and the pass it is checking must be given the same q on the
         * same side, or the agreement is luck rather than evidence. */
        if (pipe_td_verify<SLABBED>(C, 1, n, cfg->logI,
                           cfg->sq_side == 1 ? (uint32_t)L->q : 0u,
                           cfg, blocks, threads, j_base,
                           gate_found ? &gate_found[1] : NULL) ||
            pipe_td_verify<SLABBED>(C, 0, n, cfg->logI,
                           cfg->sq_side == 0 ? (uint32_t)L->q : 0u,
                           cfg, blocks, threads, j_base,
                           gate_found ? &gate_found[0] : NULL)) return -1;
        {
            const double vms = host_ms() - v0;
            *t_verify += vms;
            printf("  --- validation took %.1f ms ---\n\n", vms);
        }
    }

    /* ---- phase 3: record the joint candidates only ---- */
    if (!nacc) return 0;
    CK(cudaEventRecord(E[12]));
    k_gather_ab<<<blocks, threads>>>(C->d_a, C->d_b, C->d_sel, nacc,
                                     C->d_ca, C->d_cb);
    /* One warp per candidate, and a grid sized to the candidate count -- which
     * is what makes the grid sizing worth doing here and not in `k_td`. There
     * the per-thread cost is the whole `nsm` march and more threads cannot
     * shorten it; here more warps do, right up to one warp per candidate.
     * Beyond that point extra blocks only re-stage the same shared tile out of
     * L2, so the grid is capped at the candidates present. */
    static_assert(TD_RECORD_THREADS >= 32 && TD_RECORD_THREADS <= 1024 &&
                  (TD_RECORD_THREADS % 32) == 0,
                  "TD_RECORD_THREADS must be a whole number of warps: "
                  "k_td_record_warp maps one warp to one candidate and its "
                  "__ballot_sync assumes a full, converged warp.");
    const int rthreads = TD_RECORD_THREADS;
    const int rblocks  = (int)((nacc + (uint32_t)(rthreads / 32) - 1u) /
                               (uint32_t)(rthreads / 32));
    for (int si = 0; si < 2; si++) {
        const int side = si ? 0 : 1;
        if (cfg->td_record_scalar)
            k_td<1, 1, 1, SLABBED><<<blocks, threads>>>(
                C->d_a, C->d_b, C->d_x, C->d_sel, nacc, cfg->logI, C->d_poly[side],
                side == cfg->sq_side ? (uint32_t)L->q : 0u,
                C->d_plist[side], C->d_pcnt[side], PIPE_K,
                C->d_sm[side], C->nsm[side],
                C->d_ccof[side], C->d_cbits[side], C->d_flags, NULL,
                C->d_cfac[side], C->d_cfn[side], TD_FMAX, j_base);
        else
            k_td_record_warp<SLABBED><<<rblocks, rthreads>>>(
                C->d_a, C->d_b, C->d_x, C->d_sel, nacc, cfg->logI, C->d_poly[side],
                side == cfg->sq_side ? (uint32_t)L->q : 0u,
                C->d_plist[side], C->d_pcnt[side], PIPE_K,
                C->d_sm[side], C->nsm[side],
                C->d_ccof[side], C->d_cbits[side], C->d_flags,
                C->d_cfac[side], C->d_cfn[side], TD_FMAX, j_base);
    }
    CK(cudaEventRecord(E[13]));
    CK(cudaEventSynchronize(E[13])); CK(cudaGetLastError());
    tm->record += time_kernel(E[12], E[13]);

    /* Counts the host loop used to produce, computed where the data already is.
     * They ACCUMULATE across the band and are read at flush points, because a
     * blocking 12-byte readback per q costs about as much as the ~618 bytes per
     * candidate this change removes -- a device round trip per q is the thing
     * the rest of this pipeline is built to avoid. */
    if (accumulate_stats) {
        k_cand_stats<<<blocks, threads>>>(nacc, C->d_cbits[0], C->d_cbits[1],
                                          C->d_cfn[0], C->d_cfn[1],
                                          cfg->lpb0, cfg->lpb, TD_FMAX,
                                          C->d_stats);
        CK(cudaGetLastError());
    }

    const double rb0 = host_ms();
    /* `want_host` is false under inline cofactorisation with no candidate file:
     * the queue has already taken everything it needs straight from device
     * memory, so the ~618 bytes per candidate this used to move existed only to
     * be counted. k_cand_stats counts them in place. */
    if (!want_host) { tm->readback += host_ms() - rb0; return 0; }
    CK(cudaMemcpy(C->h_ca, C->d_ca, (size_t)nacc * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(C->h_cb, C->d_cb, (size_t)nacc * 8, cudaMemcpyDeviceToHost));
    for (int s = 0; s < 2; s++) {
        CK(cudaMemcpy(C->h_ccof[s], C->d_ccof[s], (size_t)nacc * sizeof(bn_t),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(C->h_cbits[s], C->d_cbits[s], (size_t)nacc,
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(C->h_cfn[s], C->d_cfn[s], (size_t)nacc * 4,
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(C->h_cfac[s], C->d_cfac[s], (size_t)nacc * TD_FMAX * 4,
                      cudaMemcpyDeviceToHost));
    }
    tm->readback += host_ms() - rb0;
    return 0;
}

/* Memory reports are telemetry, not part of the scientific result. A failed
 * query must never discard a completed band or bypass the final cofactor queue
 * flush. Admission-control queries remain fatal at their call site. */
static int pipe_mem_info_optional(const char *where,
                                  size_t *free_bytes, size_t *total_bytes)
{
    const cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err == cudaSuccess) return 0;
    fprintf(stderr, "warning: CUDA memory diagnostic '%s' unavailable: %s\n",
            where, cudaGetErrorString(err));
    return -1;
}

/* Close the two staged outputs exactly once. `commit` is true only after the
 * whole band and final cofactor flush succeed. Any close or rename failure
 * converts the operation to discard mode and removes whatever temporary files
 * remain. */
/* commit:       claim the final names. A clean stop is NOT a completed band,
 *               so the caller passes 0 and the .part keeps its staging name.
 * keep_partial: a checkpoint sidecar was actually written, so what is on disk
 *               is resumable work rather than debris. Only ever consulted on
 *               the discard path. */
static int pipe_finalize_outputs(FILE **frp, FILE **fcp,
                                 const bench_cfg_t *cfg,
                                 const char *rtmp, const char *ctmp,
                                 int commit, int keep_partial)
{
    int bad = 0;

    if (frp && *frp) {
        const int io_bad = ferror(*frp);
        if (fclose(*frp) || io_bad) bad = 1;
        *frp = NULL;
    }
    if (fcp && *fcp) {
        const int io_bad = ferror(*fcp);
        if (fclose(*fcp) || io_bad) bad = 1;
        *fcp = NULL;
    }

    if (commit && !bad) {
        int relations_claimed = 0;
        if (cfg->relations) {
            if (bench_atomic_replace(rtmp, cfg->relations)) { perror("rename relations"); bad = 1; }
            else relations_claimed = 1;
        }
        if (!bad && cfg->candidates && bench_atomic_replace(ctmp, cfg->candidates)) {
            perror("rename candidates");
            bad = 1;
            /* Put the relations file back under its staging name. Claiming one
             * output and then reporting the band as failed leaves a finished
             * result on disk that the exit status says was discarded, and the
             * discard below cannot reach it -- rtmp no longer exists. Under
             * BOINC that is a compute error raised for a workunit whose result
             * file is present and complete. */
            if (relations_claimed && bench_atomic_replace(cfg->relations, rtmp))
                perror("rename relations back to staging");
        }
    }
    if (!commit || bad) {
        /* A .part is resumable work only if a sidecar was actually written --
         * NOT merely because a relation file was opened. A fresh --cofactor
         * run seeds an initial whole-q checkpoint before q0, specifically so a
         * later mid-q slab flush may leave an uncheckpointed tail without
         * making the .part disposable. Resume truncates that tail to the safe
         * sidecar prefix. If even the initial checkpoint could not be written,
         * keeping the .part would strand every rerun on the startup refusal. */
        if (!keep_partial) {
            if (rtmp && rtmp[0]) remove(rtmp);
            if (ctmp && ctmp[0]) remove(ctmp);
        }
    }
    return bad ? -1 : 0;
}

/* ---- the whole per-q pipeline ------------------------------------------ */

template <bool SLABBED>
static int run_pipeline_impl(const fb_t *fb1, const fb_t *fbs1,
                             const fb_t *fb0, const fb_t *fbs0,
                             const qsel_t *qlist, uint32_t nq, sqgen_t *qgen,
                             const poly_t *POLY, const bench_cfg_t *cfg,
                             const slab_plan_t *slab_plan)
{
    const uint32_t I = 1u << cfg->logI;
    const uint32_t xmax_alloc = I * slab_plan->jmax;
    const uint32_t nregion_alloc = xmax_alloc >> cfg->log_region;
    const uint32_t nbitword_alloc = xmax_alloc >> 5;
    const int blocks = cfg->blocks ? cfg->blocks : 48 * 6;
    /* Fill's grid is absolute, not per-SM -- see FILL_BLOCKS_DEFAULT. */
    const int fblocks = cfg->fill_blocks ? cfg->fill_blocks : FILL_BLOCKS_DEFAULT;
    const int fthreads = cfg->fill_threads ? cfg->fill_threads : FILL_THREADS_DEFAULT;

    pside_t S1, S0;
    pipe_td_t C;
    pipe_tm_t tm;
    cofq_t Q; cofq_out_t QO;
    uint8_t *d_bucket = NULL;
    double vram_prev = 0;
    size_t need = 0;
    uint32_t *d_cursor = NULL, *d_overflow = NULL, *d_two = NULL;
    uint32_t *d_n = NULL;
    unsigned long long *d_pre = NULL;
    uint64_t est1, est0, est;
    uint32_t cap, nqdone = 0, bound1 = 0, bound0 = 0;
    unsigned long long nqskip = 0;   /* special-q passed over on norm width */
    size_t optin_smem_limit = 0;
    double acc_isect = 0, acc_host = 0, acc_wall = 0;
    /* The three sieve stages, broken out because the band total alone cannot be
     * compared against the standalone bench (no --pipeline), which reports the
     * same three. When the two disagreed on the A100 only the split showed that
     * the whole discrepancy was transform. "sieve, both sides" is their sum --
     * derived, not accumulated separately, so the printed total and its three
     * children cannot drift apart. */
    double acc_tr = 0, acc_fi = 0, acc_ap = 0;
    double acc_td = 0, t_verify = 0, cofac_tail = 0;
    unsigned long long acc_surv = 0, acc_cand = 0, acc_rel = 0;
    FILE *fr = NULL, *fc = NULL;
    char rtmp[CKPT_PATH_MAX] = "", ctmp[CKPT_PATH_MAX] = "";
    int rc = 0, source_exhausted = 0, count_limit_reached = 0;
    int outputs_finalized = 0, vram_reporting = 1;
    int target_reached = 0;
    qsel_t last_q = {0, 0};
    /* Initialised: the done: block destroys whatever is non-NULL, and a CUDA
     * failure can jump there from any stage before these are created. */
    cudaEvent_t ea = NULL, eb = NULL;
    /* Resume state. base_rel/base_nq are what earlier sessions already put on
     * disk: they are added to the goal tests and the progress line, and kept
     * OUT of the band summary, whose per-q figures describe this session. */
    ckpt_t ck;
    const unsigned long long base_rel = cfg->resume ? cfg->resume_nrel : 0;
    const unsigned long long base_nq  = cfg->resume ? cfg->resume_nq  : 0;
    /* ckpt_armed: a relation file is open, so checkpointing and the signal
     * handlers are active. ckpt_written: a sidecar has actually been written,
     * which is the ONLY thing that makes the .part resumable. Conflating the
     * two keeps a .part that no rerun can consume and that the startup check
     * then refuses, wedging an unattended queue. */
    int stopped = 0, ckpt_armed = 0, ckpt_written = 0, ckpt_warned = 0;
    int stop_hooked = 0;

    /* This must precede pipe_est_records(): that helper divides by every
     * modulus, so a zero modulus in an unvalidated object is already too late
     * even though the first device upload happens in pipe_side_init(). */
    if (!fb_is_transform_validated(fb1) ||
        !fb_is_transform_validated(fb0) ||
        (fbs1 && fbs1->n && !fb_is_transform_validated(fbs1)) ||
        (fbs0 && fbs0->n && !fb_is_transform_validated(fbs0))) {
        fprintf(stderr,
                "run_pipeline: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or estimating it\n");
        return -1;
    }
    /* Check both thresholds before pipe_est_records(), CUDA event creation,
     * or the multi-gigabyte bucket allocation. This is also the consumer-side
     * gate for callers that bypass bench_main's CLI validation. */
    if (sieve_bound_checked(cfg->scale, cfg->allowance, 4096u, &bound1,
                            "run_pipeline side 1 survivor parameters") ||
        sieve_bound_checked(cfg->scale0, cfg->allowance0, 4096u, &bound0,
                            "run_pipeline side 0 survivor parameters"))
        return -1;

    memset(&S1, 0, sizeof S1); memset(&S0, 0, sizeof S0);
    memset(&C, 0, sizeof C); memset(&tm, 0, sizeof tm);
    memset(&Q, 0, sizeof Q); memset(&QO, 0, sizeof QO);
#define PIPE_CK(x) do { if (CUDA_CHECKED(x)) { rc = -1; goto done; } } while (0)
    PIPE_CK(cudaEventCreate(&ea));
    PIPE_CK(cudaEventCreate(&eb));

    if (qgen)
        printf("\n=== pipeline: both sides in one process, streaming special-q ===\n");
    else
        printf("\n=== pipeline: both sides in one process, %u special-q ===\n", nq);

    est1 = pipe_est_records(fb1, xmax_alloc);
    est0 = pipe_est_records(fb0, xmax_alloc);
    est = est1 > est0 ? est1 : est0;
    cap = (uint32_t)(est / nregion_alloc) + 256;
    {
        size_t freeB = 0, totalB = 0;
        need = (size_t)nregion_alloc * cap * 4;
        PIPE_CK(cudaMemGetInfo(&freeB, &totalB));
        printf("  bucket array %u x %u x 4 B = %.2f GB, shared by both sides"
               " (%.2f GB free)\n", nregion_alloc, cap, need / 1073741824.0,
               freeB / 1073741824.0);
        if (need + 512u * 1024 * 1024 > freeB) {
            fprintf(stderr, "  pipeline: bucket array does not fit\n");
            rc = -1;
            goto done;
        }
        PIPE_CK(cudaMalloc(&d_bucket, need));
    }
    PIPE_CK(cudaMalloc(&d_cursor, (size_t)nregion_alloc * 4));
    PIPE_CK(cudaMalloc(&d_overflow, 4));

    /* ---- one-time setup, hoisted out of the q loop ---- */
    /* Device memory, by stage. The bucket array dominates on a big-area job
     * like c183 and is a minority of the total on a small-area one like the
     * c151, so "the bucket array is the footprint" is only true at one end.
     * Print the actual split rather than inviting anyone to model it. */
#define VRAM_MARK(label)                                                        \
    do {                                                                        \
        if (vram_reporting) {                                                   \
            size_t fb_ = 0, tb_ = 0;                                            \
            if (pipe_mem_info_optional((label), &fb_, &tb_) == 0) {            \
                printf("    %-28s %6.2f GB   (%.2f GB free)\n", label,          \
                       (vram_prev - (double)fb_) / 1073741824.0,                \
                       fb_ / 1073741824.0);                                     \
                vram_prev = (double)fb_;                                        \
            } else {                                                            \
                vram_reporting = 0;                                             \
            }                                                                   \
        }                                                                       \
    } while (0)
    {
        size_t fb_ = 0, tb_ = 0;
        if (pipe_mem_info_optional("stage-report baseline", &fb_, &tb_) == 0) {
            vram_prev = (double)fb_;
            printf("  device memory by stage:\n");
            vram_prev += (double)need; /* bucket is already allocated */
            VRAM_MARK("bucket array");
        } else {
            vram_reporting = 0;
        }
    }
    if (cuda_optin_smem_limit(&optin_smem_limit))
        { rc = -1; goto done; }
    if (pipe_side_init<SLABBED>(fb1, fbs1, cfg, bound1, xmax_alloc,
                                optin_smem_limit, &S1) ||
        pipe_side_init<SLABBED>(fb0, fbs0, cfg, bound0, xmax_alloc,
                                optin_smem_limit, &S0))
        { rc = -1; goto done; }
    {
        const size_t apply_smem =
            S1.apply_smem > S0.apply_smem ? S1.apply_smem : S0.apply_smem;
        /* MaxDynamicSharedMemorySize belongs to this k_apply specialization.
         * Setting it independently while initializing the two sides is wrong:
         * the second side can lower the limit required by the first. */
        PIPE_CK(cudaFuncSetAttribute(k_apply<16, 1, NORM_HORNER, SLABBED>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)apply_smem));
    }
    VRAM_MARK("factor bases + bitmaps");
    PIPE_CK(cudaMalloc(&d_two, (size_t)nbitword_alloc * 4));
    PIPE_CK(cudaMalloc(&d_n, 4));
    PIPE_CK(cudaMalloc(&d_pre, 8));
    /* Untimed warm-up, n = 0 so it touches nothing. Everything above this line
     * is cudaMalloc/cudaMemcpy, so without it k_transform is the first kernel
     * launched and the one-time CUDA cost -- module load, and on a card with
     * no native cubin the PTX JIT -- lands inside the FIRST q's transform
     * window and is then divided by the band length. At 1340 q that is ~0.15 ms
     * on a 0.954 ms figure; on a 50-q band it is ~4 ms on the same figure.
     * run_bench got this fix (finding 48) and this path, which the RUNBOOK
     * points at as the honest source for transform, did not.
     *
     * Lazy module loading is per-kernel, so this warms k_transform only; fill
     * and apply still pay their own first-launch cost inside q0. They are the
     * reps-stable stages and the cost is one-time either way, but a band short
     * enough to care should be treated as a warm-up run, not a measurement. */
    k_transform<SLABBED><<<blocks, cfg->threads>>>(S1.primes, S1.roots, S1.plat, 0u,
        cfg->logI, cfg->J, 1, 0, 0, 1, S1.d_nproj, S1.d_nlost, S1.walk_cur);
    PIPE_CK(cudaDeviceSynchronize());
    PIPE_CK(cudaGetLastError());
    if (pipe_td_init(&C, fbs1, fbs0, POLY, nbitword_alloc)) { rc = -1; goto done; }
    VRAM_MARK("trial division context");
    /* The cross-q cofactor queue. Per-q the candidate count is ~1,956, which
     * would run the rho kernel at 3% occupancy; the queue accumulates across
     * special-q and flushes ~67 q worth at a time. With it resident, only the
     * ~2% of records that become relations are read back, and the 295 MB
     * candidate file and its ~3 ms/q of host emission are gone. */
    if (cfg->cofactor && cofq_init(&Q, &QO, CQ_FLUSH,
                                   cfg->cof_meth0, cfg->cof_meth1,
                                   cfg->ecm_b1, cfg->ecm_b2, cfg->ecm_curves,
                                   cfg->cof_limbs0, cfg->cof_limbs))
        { rc = -1; goto done; }
    if (cfg->cofactor) VRAM_MARK("cofactor queue");
#undef VRAM_MARK

    if (cfg->relations && cfg->candidates &&
        !strcmp(cfg->relations, cfg->candidates)) {
        fprintf(stderr, "  --relations and --candidates must differ\n");
        rc = -1; goto done;
    }
    /* Written through temporaries and renamed only on success: a band that
     * fails at q 20 must not leave files that look like a complete run.
     *
     * The .part is nonetheless the DURABLE artifact, not scratch. The rename is
     * only the "band completed" marker; an interrupted run leaves the .part and
     * its sidecar in place to be resumed, and this used to delete both. */
    /* Build BOTH staging names before opening either file. Failing on the
     * candidates name after fr is open would, on a resume, already have
     * ftruncated the relation .part to the checkpointed prefix -- and the
     * jump to done: discards with keep_partial = ckpt_written = 0, deleting
     * a .part that holds every earlier session's work. bench_main gates both
     * paths on ckpt_path_usable() before this, so these cannot fire; they
     * keep run_pipeline safe as the public entry point it is. */
    if ((cfg->relations && ckpt_part_path(cfg->relations, rtmp, sizeof rtmp)) ||
        (cfg->candidates && ckpt_part_path(cfg->candidates, ctmp, sizeof ctmp))) {
        fprintf(stderr, "run_pipeline: output path too long for its .part name\n");
        rc = -1; goto done;
    }
    /* Binary mode, on every staging open, on both platforms. Three things
     * depend on it and all three fail quietly in text mode on Windows:
     * relations are a wire format shipped to msieve/CADO, and a CRLF copy is
     * not the same file the Linux build produces; every reader of these two
     * paths opens "rb", so a line stored as "\r\n" reaches the parser with a
     * stray \r and is counted a bad relation by the resume gate; and the
     * checkpoint records bench_tell() on this stream to compare against a
     * physical stat size, which a translating stream makes a different
     * number, so bench_truncate() would cut mid-line on resume. */
    if (cfg->relations) {
        if (cfg->resume) {
            /* Truncate to the checkpointed prefix before appending. This is
             * what makes a torn final line from a kill -9 a non-problem: the
             * partial write is discarded rather than parsed. */
            if (!(fr = fopen(rtmp, "r+b"))) { perror(rtmp); rc = -1; goto done; }
            if (bench_truncate(fr, cfg->resume_rel_bytes) ||
                bench_seek(fr, cfg->resume_rel_bytes)) {
                perror(rtmp); rc = -1; goto done;
            }
        } else if (!(fr = fopen(rtmp, "wb"))) {
            perror(rtmp); rc = -1; goto done;
        }
    }
    if (cfg->candidates) {
        if (cfg->resume) {
            if (!(fc = fopen(ctmp, "r+b"))) { perror(ctmp); rc = -1; goto done; }
            if (bench_truncate(fc, cfg->resume_cand_bytes) ||
                bench_seek(fc, cfg->resume_cand_bytes)) {
                perror(ctmp); rc = -1; goto done;
            }
        } else if (!(fc = fopen(ctmp, "wb"))) {
            perror(ctmp); rc = -1; goto done;
        }
    }
    /* Only meaningful once there is a relation file to point at. */
    if (fr) {
        memset(&ck, 0, sizeof ck);
        ckpt_fingerprint(POLY, cfg, ck.fp);
        /* --resume starts from an existing valid sidecar; preserve the .part
         * even if this session fails before advancing the checkpoint. */
        if (cfg->resume) {
            ckpt_written = 1;
            ck.next_q = cfg->resume_q; ck.next_rho = cfg->resume_rho;
            ck.rel_bytes = cfg->resume_rel_bytes;
            ck.cand_bytes = cfg->resume_cand_bytes;
            ck.nrel = cfg->resume_nrel; ck.nqdone = cfg->resume_nq;
        }
        ck.scale = cfg->scale; ck.scale0 = cfg->scale0;
        ck.allowance = cfg->allowance; ck.allowance0 = cfg->allowance0;
        ckpt_armed = 1;
        /* See bench_stop_hook_install: on Windows this is a console control
         * handler, because signal(SIGTERM) is never delivered there. A task
         * stopped with TerminateProcess still runs nothing at all, so
         * --stop-file remains the only reliable managed stop on Windows. */
        stop_hooked = bench_stop_hook_install(pipe_request_stop) == 0;
        g_pipe_stop = 0;
    }

    /* ---- the band ---- */
    {
    const double t_band = host_ms();
    double t_report = t_band, t_ckpt = t_band;
    /* The \r line is for a human watching a terminal. Redirected to a file it
     * is a single unreadable line that grows a carriage return every 30 s, so
     * a redirect gets whole lines every five minutes instead -- ~864 of them
     * over a three-day band rather than 8,640.
     *
     * FIVE MINUTES IS THIS STREAM'S OWN CONSTANT, not --log-every. Reading the
     * log's period here coupled two unrelated outputs: `--log-every 86400`
     * with no --log at all would have left a redirected run printing one
     * progress line per day, and a plain redirect with no logging flags would
     * have silently moved from 30 s to 300 s because log_every_s defaults to
     * 300 whether or not a log exists. The run log (--log) is a separate,
     * richer stream on its own clock. */
    const int reporting_to_tty = bench_stdout_is_tty();
    const double report_ms = reporting_to_tty ? 30000.0 : 300000.0;
#ifdef HAVE_BOINC
    double t_boinc_report = t_band - 1000.0; /* report after the first q */
#endif
    /* The side-0 special-q polynomial that used to be built here is gone:
     * qsel_validate() derives G = Y1*x + Y0 itself when sq_side == 0, so the
     * per-q gate below covers what pipe_check_root() did and more. */
    for (uint32_t qi = 0;; qi++) {
        qsel_t generated, checked;
        const qsel_t *cur;
        qlat_t Lq;
        float ts1[3] = {0,0,0}, ts0[3] = {0,0,0}, tis = 0;
        double th1 = 0, th0 = 0, qwall = host_ms(), tv = 0;
        uint32_t hn = 0, nacc = 0, ncand = 0, nrel = 0;
        uint64_t side_surv1 = 0, side_surv0 = 0;
        uint32_t cofgate_found[2] = {0, 0};
        int td_verified = 0;
        /* The host loop exists to write files. With inline cofactorisation and
         * no candidate file, nothing reads the host mirrors, so neither the
         * copies nor the loop have to happen.
         *
         * cfg->emit_cof is NOT a third case: --emit-cof is in harness_only and
         * is rejected outright under --pipeline, so it is always NULL here.
         * Including it made this look like it had three outcomes when it has
         * two, and the end-of-band stats fold-in depends on this and
         * accumulate_stats being exact complements. */
        const int want_host = !cfg->cofactor || fc;

        if (qi < nq) {
            cur = &qlist[qi];
        } else if (qgen) {
            int qr;
            /* Enforced here as well as in the generator, because a resumed run
             * hands sqgen no limit at all -- see the sqgen_create comment in
             * bench_main.cu. Without this the budget would not exist. */
            if (cfg->nq_max && nqdone >= cfg->nq_max) {
                count_limit_reached = 1;
                break;
            }
            qr = sqgen_next(qgen, &generated);
            if (qr < 0) {
                fprintf(stderr, "  special-q generator failed after %u q\n",
                        nqdone);
                rc = -1; break;
            }
            if (qr == 0) {
                if (cfg->nq_max && nqdone >= cfg->nq_max)
                    count_limit_reached = 1;
                else
                    source_exhausted = 1;
                break;
            }
            cur = &generated;
        } else {
            if (cfg->nq_max && nqdone >= cfg->nq_max)
                count_limit_reached = 1;
            else
                source_exhausted = 1;
            break;
        }
        /* Validate again at the consumer boundary. bench_main validates every
         * input before scale derivation, but run_pipeline is a public entry
         * point and the streaming generator supplies later q directly here. */
        checked = *cur;
        {
            const qsel_validate_result_t qv =
                qsel_validate(&checked, POLY, cfg->sq_side);
            if (qv != QSEL_VALID) {
                const char *why = qv == QSEL_ERR_Q_RANGE ? "q outside [2,2^32)"
                                : qv == QSEL_ERR_Q_COMPOSITE ? "composite q"
                                : qv == QSEL_ERR_NOT_ROOT ? "rho is not a root"
                                : qv == QSEL_ERR_SIDE ? "invalid special-q side"
                                : "invalid exact polynomial";
                fprintf(stderr, "  q=%llu rho=%llu: special-q rejected: %s\n",
                        (unsigned long long)checked.q,
                        (unsigned long long)checked.rho, why);
                rc = -1; break;
            }
        }
        cur = &checked;

        /* A cofactor queue can first fill in slab 1, 2, ... of q0. Such a
         * flush contains a partial q and is not a resume point. Seed a safe
         * empty/prefix checkpoint before any work so a crash can always keep
         * the .part and replay this q from a whole-q boundary. */
        if (fr && cfg->cofactor && !ckpt_written && !nqdone && Q.n == 0)
            pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                base_rel + Q.nrel, base_nq + nqdone,
                                &ckpt_written, &ckpt_warned);

        /* ---- clean stop ----
         *
         * Checked here, with the next (q, rho) in hand but no work done for it,
         * because that pair is exactly the resume point: drain the queue so
         * every q already sieved reaches the file, checkpoint, and leave this
         * one for the next session. A planned stop loses nothing.
         *
         * This runs AFTER the validation gate above so that the pair written to
         * the sidecar is always one a rerun will accept -- and with the rho
         * qsel_validate canonicalised, not whatever width the source supplied.
         *
         * The stop file is for unattended runs, where there is no terminal to
         * press ^C in and a job queue needs a way to ask for the card back. */
        if (fr && !g_pipe_stop && cfg->stopfile && (qi & 63) == 0 &&
            bench_path_exists(cfg->stopfile)) {
            printf("\n  stop file %s exists\n", cfg->stopfile);
            g_pipe_stop = 1;
        }
        if (fr && g_pipe_stop) {
            printf("\n  stopping cleanly at q=%llu; draining the cofactor"
                   " queue\n", (unsigned long long)cur->q);
            if (cfg->cofactor && Q.n &&
                cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                           cfg->cof_rounds, cfg->cof_budget, blocks,
                           cfg->threads, fr)) { rc = -1; break; }
            pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                base_rel + (cfg->cofactor ? Q.nrel
                                                          : (unsigned long long)acc_rel),
                                base_nq + nqdone, &ckpt_written, &ckpt_warned);
            stopped = 1;
            break;
        }
        /* Without --cofactor the host join loop writes each q's relations as it
         * finishes, so every q boundary is a valid resume point and the only
         * question is how often to pay the fsync. Under --cofactor this is
         * handled at the flush instead, and doing it here as well would record
         * a point the queue's contents are not in the file for. */
        if (fr && !cfg->cofactor && qi && host_ms() - t_ckpt > 30000.0) {
            t_ckpt = host_ms();
            pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                base_rel + (unsigned long long)acc_rel,
                                base_nq + nqdone, &ckpt_written, &ckpt_warned);
        }

        qlat_build(&Lq, cur->q, cur->rho, POLY->skew);
        if (cfg->verbose_q)
            printf("\n  q = %llu, rho = %llu\n",
                   (unsigned long long)cur->q,
                   (unsigned long long)cur->rho);

        {
            /* SHORT-CIRCUITS ON FAILURE, like the `||` chain this replaced.
             * Running side 0 after side 1 has already failed launches more CUDA
             * work against a sticky error, and CUDA_CHECKED then reports side 0
             * for side 1's fault. A SKIP short-circuits too -- one side being
             * unfittable is enough to pass over the q, and the warning naming
             * the side has already been printed by the side that raised it. */
            const int p1 = pipe_side_prepare_q<SLABBED>(fb1, fbs1, &Lq, POLY, cfg, 1,
                                                        cfg->scale, blocks, &S1,
                                                        &ts1[0], &th1);
            const int p0 = (p1 != 0) ? p1
                         : pipe_side_prepare_q<SLABBED>(fb0, fbs0, &Lq, POLY, cfg, 0,
                                                        cfg->scale0, blocks, &S0,
                                                        &ts0[0], &th0);
            if (p1 == PIPE_Q_SKIP || p0 == PIPE_Q_SKIP) {
                if (++nqskip >= (unsigned long long)PIPE_SKIP_MAX) {
                    /* A POLICY STOP, NOT A CRASH. Falling out with rc = -1
                     * skipped the post-loop cofactor flush (guarded on rc == 0)
                     * and threw away every relation queued since the last one,
                     * then reported the band as FAILED with no checkpoint. This
                     * is a deliberate stop on a known condition, so it drains
                     * and checkpoints exactly like the stop-file path. */
                    runlog_warn("  ** %llu special-q skipped for norm width;"
                                " this build is too narrow for the job."
                                " Run ./normscan and rebuild.", nqskip);
                    if (cfg->cofactor && Q.n &&
                        cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim,
                                   cfg->lpb, cfg->cof_rounds, cfg->cof_budget,
                                   blocks, cfg->threads, fr)) { rc = -1; break; }
                    pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                        base_rel + (cfg->cofactor ? Q.nrel
                                            : (unsigned long long)acc_rel),
                                        base_nq + nqdone, &ckpt_written,
                                        &ckpt_warned);
                    stopped = 1;
                    break;
                }
                continue;              /* next q; nqdone is not incremented */
            }
            if (p1 < 0 || p0 < 0 || pipe_td_prepare_q(&C, fbs1, fbs0, &Lq, cfg, &tm)) {
                rc = -1; break;
            }
            /* AFTER the skip check, not before: last_q names the band's final
             * progress point and feeds the BOINC fraction, and a skipped q was
             * never sieved. */
            last_q = *cur;
        }

        /* Only the area-dependent part lives inside this loop. For
         * SLABBED=false slab_plan has exactly one entry and if constexpr
         * removes every continuation-state operation from device code. */
        for (uint32_t slab = 0; slab < slab_plan->nslab; slab++) {
            const uint32_t j_base = slab_jbase_at(slab_plan, slab);
            const uint32_t J_here = slab_rows_at(slab_plan, cfg->J, slab);
            const uint32_t xmax = I * J_here;
            const uint32_t nbitword = xmax >> 5;
            float sf1 = 0, sa1 = 0, sf0 = 0, sa0 = 0;
            uint32_t hn_s = 0, nacc_s = 0;
            double td_start, tv_before, cf_start, join_start;

            if (pipe_side_sieve_slab<SLABBED>(fb1, cfg, 1, xmax, j_base,
                                               d_bucket, d_cursor, cap,
                                               d_overflow, fblocks, fthreads,
                                               &S1, &sf1, &sa1) ||
                pipe_side_sieve_slab<SLABBED>(fb0, cfg, 0, xmax, j_base,
                                               d_bucket, d_cursor, cap,
                                               d_overflow, fblocks, fthreads,
                                               &S0, &sf0, &sa0)) {
                rc = -1; break;
            }
            ts1[1] += sf1; ts1[2] += sa1;
            ts0[1] += sf0; ts0[2] += sa0;
            side_surv1 += S1.nsurv; side_surv0 += S0.nsurv;

            PIPE_CK(cudaMemset(d_n, 0, 4));
            PIPE_CK(cudaMemset(d_pre, 0, 8));
            PIPE_CK(cudaMemset(d_two, 0, (size_t)nbitword * 4));
            PIPE_CK(cudaEventRecord(ea));
            k_intersect_compact<1, SLABBED><<<blocks, cfg->threads>>>(
                S1.survbits, S0.survbits, nbitword, cfg->logI,
                Lq.a0, Lq.a1, Lq.b0, Lq.b1, (int64_t)Lq.q,
                NULL, NULL, NULL, 0u, d_n, d_pre, d_two, NULL, j_base);
            PIPE_CK(cudaEventRecord(eb));
            PIPE_CK(cudaEventSynchronize(eb));
            PIPE_CK(cudaGetLastError());
            tis += time_kernel(ea, eb);
            PIPE_CK(cudaMemcpy(&hn_s, d_n, 4, cudaMemcpyDeviceToHost));

            td_start = host_ms();
            tv_before = tv;
            {
                uint32_t n = 0;
                /* nqdone, NOT qi: these gates run on the first q actually
                 * SIEVED. Keyed on the loop index they were silently disabled
                 * for the whole band by a single skipped q at qi == 0 --
                 * --cofgate would then report success having never found any
                 * reference overlap, and --td-verify would reconstruct nothing
                 * while still reporting a pass. */
                const int do_verify = nqdone == 0 && cfg->td_verify && hn_s != 0 &&
                    (!td_verified || cfg->cofgate);
                if (pipe_td_perq<SLABBED>(&C, &Lq, cfg, &S1, &S0, d_two, xmax,
                                          j_base, blocks, cfg->threads, do_verify,
                                          &n, &nacc_s, &tm, &tv, want_host,
                                          !want_host, cofgate_found)) {
                    rc = -1; break;
                }
                if (nqdone == 0 && cfg->td_verify && !td_verified && hn_s != 0)
                    td_verified = 1;
                if (n != hn_s) {
                    fprintf(stderr,
                            "  q=%llu slab %u: intersect counted %u survivors,"
                            " rank scan %u\n",
                            (unsigned long long)cur->q, slab, hn_s, n);
                    rc = -1; break;
                }
                acc_td += host_ms() - td_start - (tv - tv_before);
            }
            hn += hn_s;
            nacc += nacc_s;

            cf_start = host_ms();
            /* A queue flush is safe inside a q, but it is NOT a new resume
             * point once an earlier slab of this q has been emitted. Resume
             * truncates to the previous whole-q checkpoint, so a crash still
             * replays the q atomically. */
            if (cfg->cofactor && nacc_s) {
                if (Q.n + nacc_s > Q.cap) {
                    if (cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0,
                                   cfg->lim, cfg->lpb, cfg->cof_rounds,
                                   cfg->cof_budget, blocks, cfg->threads, fr)) {
                        rc = -1; break;
                    }
                    if (fr && slab == 0)
                        pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                            base_rel + Q.nrel,
                                            base_nq + nqdone,
                                            &ckpt_written, &ckpt_warned);
                }
                if (nacc_s > Q.cap) {
                    runlog_warn("  q=%llu slab %u: %u candidates exceeds the"
                                " %u-slot cofactor queue",
                                (unsigned long long)cur->q, slab, nacc_s, Q.cap);
                    rc = -1; break;
                }
                if (cof_enqueue(blocks, cfg->threads,
                                C.d_ccof[0], C.d_cbits[0],
                                C.d_ccof[1], C.d_cbits[1], C.d_ca, C.d_cb,
                                C.d_cfac[0], C.d_cfn[0],
                                C.d_cfac[1], C.d_cfn[1],
                                nacc_s, Q.n, cfg->lpb0, cfg->lpb, &Q)) {
                    rc = -1; break;
                }
                if (CUDA_CHECKED(cudaDeviceSynchronize()) ||
                    CUDA_CHECKED(cudaGetLastError())) { rc = -1; break; }
                Q.n += nacc_s;
            }
            tm.cofac += host_ms() - cf_start;

            join_start = host_ms();
            for (uint32_t k = 0; want_host && k < nacc_s; k++) {
                int64_t a = C.h_ca[k], b = C.h_cb[k];
                const uint32_t c0 = C.h_cfn[0][k], c1 = C.h_cfn[1][k];
                const uint32_t *f0 = C.h_cfac[0] + (size_t)k * TD_FMAX;
                const uint32_t *f1 = C.h_cfac[1] + (size_t)k * TD_FMAX;
                const int b0 = C.h_cbits[0][k], b1 = C.h_cbits[1][k];
                char buf[BN_DEC_MAX];
                if (c0 > TD_FMAX || c1 > TD_FMAX) {
                    fprintf(stderr, "  q=%llu: a candidate has %u/%u factors,"
                            " more than the %d recorded; raise TD_FMAX\n",
                            (unsigned long long)cur->q, c0, c1, TD_FMAX);
                    rc = -1; break;
                }
                std::sort((uint32_t *)f0, (uint32_t *)f0 + c0);
                std::sort((uint32_t *)f1, (uint32_t *)f1 + c1);
                if (b < 0) { a = -a; b = -b; }
                if (b0 <= (int)cfg->lpb0 && b1 <= (int)cfg->lpb) {
                    nrel++;
                    if (fr && !cfg->cofactor) {
                        fprintf(fr, "%lld,%lld:", (long long)a, (long long)b);
                        for (uint32_t z = 0; z < c0; z++)
                            fprintf(fr, "%s%x", z ? "," : "", f0[z]);
                        if (b0 > 1)
                            fprintf(fr, "%s%llx", c0 ? "," : "",
                                    (unsigned long long)strtoull(
                                        bn_to_dec(&C.h_ccof[0][k], buf), NULL, 10));
                        fputc(':', fr);
                        for (uint32_t z = 0; z < c1; z++)
                            fprintf(fr, "%s%x", z ? "," : "", f1[z]);
                        if (b1 > 1)
                            fprintf(fr, "%s%llx", c1 ? "," : "",
                                    (unsigned long long)strtoull(
                                        bn_to_dec(&C.h_ccof[1][k], buf), NULL, 10));
                        fputc('\n', fr);
                    }
                    continue;
                }
                ncand++;
                if (cfg->cofactor) continue;
                if (fc) {
                    fprintf(fc, "%s%s,", b0 > (int)cfg->lpb0 ? "-" : "",
                            bn_to_dec(&C.h_ccof[0][k], buf));
                    fprintf(fc, "%s%s:%lld,%lld:",
                            b1 > (int)cfg->lpb ? "-" : "",
                            bn_to_dec(&C.h_ccof[1][k], buf),
                            (long long)a, (long long)b);
                    for (uint32_t z = 0; z < c0; z++)
                        fprintf(fc, "%s%x", z ? "," : "", f0[z]);
                    fputc(':', fc);
                    for (uint32_t z = 0; z < c1; z++)
                        fprintf(fc, "%s%x", z ? "," : "", f1[z]);
                    fputc('\n', fc);
                }
            }
            tm.join += host_ms() - join_start;
            if (rc) break;

            if constexpr (SLABBED) {
                if (slab + 1 < slab_plan->nslab) {
                    if (pipe_td_advance_small(&C, J_here, blocks,
                                              cfg->threads)) {
                        rc = -1; break;
                    }
                    std::swap(S1.walk_cur, S1.walk_next);
                    std::swap(S0.walk_cur, S0.walk_next);
                }
            }
        }
        if (rc) break;
        /* This is the original pre-slabbing invariant: an entire special-q
         * with no two-sided survivors is suspicious and remains fatal. Empty
         * individual slabs are allowed; only their sum is tested here. */
        if (!hn) {
            fprintf(stderr, "  pipeline: no survivors at this q\n");
            rc = -1; break;
        }
        if (nqdone == 0 && cfg->cofgate &&
            (!cofgate_found[0] || !cofgate_found[1])) {
            fprintf(stderr,
                    "  trial-division cofactor gate found no reference overlap"
                    " on side%s%s across the complete q\n",
                    cofgate_found[0] ? "" : " 0",
                    cofgate_found[1] ? "" : " 1");
            rc = -1; break;
        }

        if (cfg->verbose_q) {
            printf("  side 1: transform %.3f + fill %.3f + apply %.3f = %7.3f ms,"
                   " %8llu survivors (bound %u)%s\n",
                   ts1[0], ts1[1], ts1[2], ts1[0] + ts1[1] + ts1[2],
                   (unsigned long long)side_surv1, S1.BOUND,
                   SLABBED ? " [slabbed]" : "");
            printf("  side 0: transform %.3f + fill %.3f + apply %.3f = %7.3f ms,"
                   " %8llu survivors (bound %u)%s\n",
                   ts0[0], ts0[1], ts0[2], ts0[0] + ts0[1] + ts0[2],
                   (unsigned long long)side_surv0, S0.BOUND,
                   SLABBED ? " [slabbed]" : "");
        }

        t_verify += tv;
        acc_tr += ts1[0] + ts0[0];
        acc_fi += ts1[1] + ts0[1];
        acc_ap += ts1[2] + ts0[2];
        acc_isect += tis; acc_host += th1 + th0;
        acc_wall += host_ms() - qwall - tv;
        acc_surv += hn; acc_cand += ncand; acc_rel += nrel;   /* host path only */
        nqdone++;
        norm_verbose = 0;      /* the first q's setup is printed; the rest are not */
#ifdef HAVE_BOINC
        {
            const unsigned long long progress_rels = cfg->cofactor
                ? Q.nrel : (unsigned long long)acc_rel;
            const double now = host_ms();
            const int terminal_known =
                (!qgen && nqdone == nq) ||
                (cfg->target_rels && progress_rels >= cfg->target_rels);
            if (now - t_boinc_report >= 1000.0 || terminal_known) {
                double fraction = pipe_progress_fraction(
                    cfg, qgen != NULL, nq, nqdone, cur->q, progress_rels);
                /* Leave the final one percent for the after-band cofactor flush,
                 * output close/rename, cleanup, and bench_boinc_finish(). A
                 * successful workunit reports exactly 1.0 only after main has
                 * completed successfully. */
                if (fraction > 0.99) fraction = 0.99;
                bench_boinc_fraction_done(fraction);
                t_boinc_report = now;
            }
        }
#endif
        /* --target-rels: stop once enough relations exist. Under inline
         * cofactorisation the only running total the host has without a per-q
         * readback is Q.nrel, which advances at FLUSH boundaries -- so this
         * overshoots by at most one flush (~256 q here, well under 1% of any
         * useful target). Sieving upward until satisfied is what you actually
         * want when the question is "enough for the matrix", since the yield
         * per q falls as q grows and guessing the range wastes either time or
         * relations. */
        if (cfg->target_rels) {
            /* base_rel is what earlier sessions already wrote. Counting only
             * this session's would make a resumed run sieve the whole target
             * again from scratch. */
            unsigned long long have = base_rel + (cfg->cofactor
                                                  ? Q.nrel
                                                  : (unsigned long long)acc_rel);
            if (have >= cfg->target_rels) {
                printf("\n  --target-rels %llu reached after %llu q (%llu relations)\n",
                       (unsigned long long)cfg->target_rels,
                       base_nq + nqdone, have);
                target_reached = 1;
                break;
            }
        }
        /* ncand and nrel are produced by the host join loop, which does not run
         * under inline cofactorisation -- so printing them there reported a
         * flat "0 joint candidates, 0 relations" for every q of the band. The
         * progress line below was fixed for this; this branch was not, and it
         * is the one somebody passes --verbose-q to read. Report what is
         * actually known per q rather than a variable that stayed at its
         * initialiser; the device counters are folded in after the band. */
        if (cfg->verbose_q) {
            if (want_host)
                printf("  q=%llu: %u survivors, %u joint candidates,"
                       " %u relations\n",
                       (unsigned long long)cur->q, hn, ncand, nrel);
            else
                printf("  q=%llu: %u survivors, %u joint candidates"
                       " (relations counted on the device; see band summary)\n",
                       (unsigned long long)cur->q, hn, nacc);
        }
        /* Under inline cofactorisation the host has no per-q relation count --
         * the counters accumulate on the device and Q.nrel only advances at a
         * flush. Reporting acc_rel there showed a flat 0 for the whole run,
         * which on a multi-hour band looks exactly like failure.
         *
         * Q.nrel is not enough on its own either. It advances at a FLUSH, and a
         * flush is 131,072 candidates -- ~67 q on the c183 this was tuned
         * against, but 686 q on the SNFS job, which enqueues 191 records per q
         * instead of 1,956. That is 41 s of "0 rel" before the counter can
         * move at all, and `--relations` stages to NAME.part until the band
         * ends, so `ls` agrees with it. A run that was working normally, with
         * 63,000 candidates queued that went on to yield 39,710 relations, read
         * as a dead run. So carry the queue occupancy alongside -- it advances
         * every q, on every job -- and label it as candidates, which is what it
         * counts.
         *
         * Progress is reported against whichever goal is actually in force: a
         * relation target if one was given, otherwise an explicit --nq count,
         * generated numeric q interval, or list length. Rate-limited to one
         * update per report_ms -- 30 s on a terminal, where it is a single \r
         * line and a band runs for hours, and the log's own period when stdout
         * is redirected and each update costs a line.
         */
        /* Two consumers on two clocks: the console line, and the run log,
         * which is due on its own period and carries the four numbers that say
         * whether the progress ones can be compared to anything (runlog.h,
         * finding 53). Both predicates are evaluated every q -- deliberately,
         * rather than short-circuiting one against the other, which would skip
         * the log's clock on exactly the q where both came due and post the
         * record a special-q late. The shared arithmetic below is done once
         * for whichever is asking. */
        const int console_due = !cfg->verbose_q &&
            (host_ms() - t_report > report_ms || (!qgen && qi + 1 == nq));
        const int log_due = runlog_due();
        if (console_due || log_due) {
            /* Two different numbers, and mixing them is how a resumed run
             * reports a nonsense rate: the goal is measured against everything
             * on disk, the RATE only against what this session produced in the
             * time this session has been running. */
            const unsigned long long mine = cfg->cofactor
                ? Q.nrel : (unsigned long long)acc_rel;
            const unsigned long long rels = base_rel + mine;
            const double el = (host_ms() - t_band) / 1000.0;
            const double rps = el > 0 ? mine / el : 0.0;
            double frac, eta;
            if (console_due) t_report = host_ms();
            frac = pipe_progress_fraction(cfg, qgen != NULL, nq, nqdone,
                                             cur->q, rels);
            if (cfg->target_rels)
                eta = rps > 0 ? ((double)cfg->target_rels - rels) / rps : 0.0;
            else
                eta = frac > 0 ? el * (1.0 / frac - 1.0) : 0.0;
            if (eta < 0.0) eta = 0.0;
            if (console_due) {
                /* Q.n counts queued CANDIDATE records, not relations -- only
                 * ~2/3 of them survive splitting on this job, and the band
                 * summary keeps nseen and nrel apart for that reason. Labelled
                 * "cand" so the line cannot be read as pending relations.
                 *
                 * Shown only while the queue holds work the relation count
                 * cannot see yet, so a steady-state line stays as it was --
                 * which means the line CHANGES WIDTH when cofq_flush resets
                 * Q.n to 0. A \r without an erase leaves the tail of the longer
                 * line on screen ("...ETA 0h 12m  cand"), so clear to
                 * end-of-line rather than trusting a trailing pad to cover the
                 * widest case. */
                char queued[32] = "";
                /* The erase-to-end-of-line and the carriage return are a
                 * terminal's, not a file's: redirected, they make one line
                 * carrying an ANSI escape per update. A redirect gets plain
                 * newline-terminated lines instead. */
                const char *end = reporting_to_tty ? "\033[K\r" : "\n";
                if (cfg->cofactor && Q.n)
                    snprintf(queued, sizeof queued, " +%u cand", Q.n);
                /* rps == 0 means "no relation has flushed yet", not "done".
                 * eta is 0 there as a sentinel, and printing it as 0h 00m says
                 * the band is finishing at the exact moment nothing has been
                 * counted -- which is the reading this line exists to prevent. */
                if (rps > 0.0 && isfinite(eta)) {
                    /* Do not cast the complete ETA to an int: immediately
                     * after the first cofactor flush it can legitimately be
                     * millions of hours. Keep the unbounded hour count as a
                     * double and cast only the modulo-60 minute component. */
                    const double eta_hours = floor(eta / 3600.0);
                    const int eta_minutes =
                        (int)fmod(floor(eta / 60.0), 60.0);
                    printf("    q=%llu  %u q  %llu rel%s  %.0f rel/s  %.1f%%"
                           "  ETA %.0fh %02dm%s",
                           (unsigned long long)cur->q, nqdone, rels, queued,
                           rps, 100.0 * frac, eta_hours, eta_minutes, end);
                } else if (rps > 0.0) {
                    printf("    q=%llu  %u q  %llu rel%s  %.0f rel/s  %.1f%%"
                           "  ETA inf%s",
                           (unsigned long long)cur->q, nqdone, rels, queued,
                           rps, 100.0 * frac, end);
                } else {
                    printf("    q=%llu  %u q  %llu rel%s  -- rel/s  %.1f%%"
                           "  ETA --h --m%s",
                           (unsigned long long)cur->q, nqdone, rels, queued,
                           100.0 * frac, end);
                }
            }
            if (log_due) {
                /* The same progress numbers, plus what makes them auditable.
                 *
                 * GPU-accounted/wall is the running form of the band summary's
                 * line, and identical to it in construction: the cofactor
                 * queue is out of BOTH terms because its flush is host-timed
                 * and mixes device and host work, and leaving it in the
                 * denominator alone would make the ratio move with survivor
                 * density -- a dense band would read as a contended host on an
                 * idle box. Compare it against your own idle baseline on the
                 * same card, job and band length; an absolute reading is not
                 * meaningful (finding 53).
                 *
                 * Utilisation, board watts and load average are sampled here
                 * rather than averaged: this is a spot check that says whether
                 * the box was busy, not a power measurement. The metric of
                 * record is whole-box watts from a meter (STATUS.md item 6),
                 * and a board sensor cannot be promoted to it. */
                const double devsum = tm.rank + tm.emit + tm.summary
                    + tm.resieve + tm.td + tm.classify + tm.compact + tm.record;
                const double denom = acc_wall - tm.cofac;
                const double accwall = denom > 0.0
                    ? (acc_tr + acc_fi + acc_ap + acc_isect + devsum) / denom
                    : 0.0;
                char gpu[48] = "gpu=n/a", pwr[32] = "board=n/a";
                char eta_s[32] = "eta=--";
                double load[3] = {0.0, 0.0, 0.0};
                unsigned int util = 0;
                double watts = 0.0;
                if (runlog_gpu_util(&util) == 0)
                    snprintf(gpu, sizeof gpu, "gpu=%u%%", util);
                if (runlog_gpu_watts(&watts) == 0)
                    snprintf(pwr, sizeof pwr, "board=%.1fW", watts);
#ifdef _WIN32
                /* Unix load average has no direct Win32/MSVC equivalent.
                 * Keep the status field but mark it unavailable on Windows. */
                load[0] = -1.0;
#else
                if (getloadavg(load, 3) < 1) load[0] = -1.0;
#endif
                if (rps > 0.0 && isfinite(eta))
                    snprintf(eta_s, sizeof eta_s, "eta=%.2fh", eta / 3600.0);
                runlog_record(
                    "q=%llu nq=%llu rel=%llu cand=%u rel/s=%.1f pct=%.2f %s"
                    " ms/q=%.2f acc/wall=%.3f %s %s load=%.2f",
                    (unsigned long long)cur->q,
                    base_nq + (unsigned long long)nqdone, rels,
                    cfg->cofactor ? Q.n : 0u, rps, 100.0 * frac, eta_s,
                    nqdone ? acc_wall / nqdone : 0.0, accwall, gpu, pwr,
                    load[0]);
            }
        }
        fflush(stdout);
    }

    /* A streamed source discovers its end only when the NEXT pull returns 0,
     * so the in-loop 30-second reporter cannot know that the preceding q was
     * terminal. Give it an explicit terminal line instead of leaving the last
     * visible progress sample up to 30 seconds stale. */
    if (!cfg->verbose_q && qgen && !rc && nqdone &&
        (source_exhausted || count_limit_reached)) {
        const unsigned long long rels = base_rel + (cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel);
        char queued[32] = "";
        if (cfg->cofactor && Q.n)
            snprintf(queued, sizeof queued, " +%u cand", Q.n);
        /* The erase clears the tail of the \r line this one replaces, and the
         * leading newline breaks away from it. Both are a terminal's, not a
         * file's: redirected, the progress line above already ended in a
         * newline, so the lead would only add a blank line. */
        const char *erase = reporting_to_tty ? "\033[K" : "";
        const char *lead = reporting_to_tty ? "\n" : "";
        if (count_limit_reached)
            /* base_nq + nq_max reconstructs what the operator actually typed:
             * a resumed run holds only the REMAINING budget in cfg->nq_max, so
             * echoing it raw prints a number nobody passed, next to a q count
             * that then disagrees with it. */
            printf("%s    q=%llu  %llu q  %llu rel%s  [--nq %llu reached]%s\n",
                   lead, (unsigned long long)last_q.q, base_nq + nqdone, rels,
                   queued, base_nq + cfg->nq_max, erase);
        else
            printf("%s    q=%llu  %llu q  %llu rel%s  [q range exhausted]%s\n",
                   lead, (unsigned long long)last_q.q, base_nq + nqdone, rels,
                   queued, erase);
    }

#ifdef HAVE_BOINC
    if (!rc && nqdone) {
        const unsigned long long rels = cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel;
        double fraction = pipe_progress_fraction(
            cfg, qgen != NULL, nq, nqdone, last_q.q, rels);
        if (fraction > 0.99) fraction = 0.99;
        bench_boinc_fraction_done(fraction);
    }
#endif

    }   /* t_band scope */

    /* Fold in the device-side candidate counters. Under inline cofactorisation
     * the host loop never ran, so acc_cand and acc_rel are still zero and these
     * are the only counts there are. The overflow check lands here rather than
     * per q: the alternative was a blocking readback every q, which cost more
     * than the host loop it replaced. A truncated factor list still fails the
     * run -- just at the end of the band rather than inside it. */
    if (!rc && cfg->cofactor && !fc) {      /* exactly !want_host */
        uint32_t cs[3] = {0, 0, 0};
        if (CUDA_CHECKED(cudaMemcpy(cs, C.d_stats, 12,
                                    cudaMemcpyDeviceToHost)))
            rc = -1;
        else {
            acc_rel += cs[0]; acc_cand += cs[1];
            if (cs[2]) {
                runlog_warn("  %u candidates in this band have more than the"
                            " %d recorded factors; raise TD_FMAX", cs[2],
                            TD_FMAX);
                rc = -1;
            }
        }
    }

    /* The last partial flush happens AFTER the band loop, so its cost is not
     * in acc_wall. On a band shorter than one flush that is the ENTIRE
     * cofactorisation, which is why it is carried separately and added back
     * rather than folded into the per-q average. */
    if (rc == 0 && cfg->cofactor && Q.n) {
        const double cf0 = host_ms();
        if (cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                       cfg->cof_rounds, cfg->cof_budget, blocks, cfg->threads, fr))
            rc = -1;
        cofac_tail = host_ms() - cf0;
    }
    if (rc == 0 && !stopped && cfg->target_rels && !target_reached) {
        const unsigned long long have = base_rel + (cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel);
        if (have >= cfg->target_rels) {
            printf("\n  --target-rels %llu reached by the final cofactor flush"
                   " after %llu q (%llu relations)\n",
                   (unsigned long long)cfg->target_rels, base_nq + nqdone, have);
            target_reached = 1;
        } else if (count_limit_reached) {
            runlog_warn(
                    "\n  note: --nq %llu reached after %llu q with %llu relations;"
                    " --target-rels %llu was not reached",
                    base_nq + cfg->nq_max, base_nq + nqdone, have,
                    (unsigned long long)cfg->target_rels);
        } else if (source_exhausted) {
            runlog_warn(
                    "\n  WARNING: special-q source exhausted after %llu q with"
                    " %llu relations; --target-rels %llu was NOT reached",
                    base_nq + nqdone, have,
                    (unsigned long long)cfg->target_rels);
        }
    }
    {
        /* A clean stop is not a completed band: the .part and its sidecar stay,
         * and the final name is not claimed. Renaming here would present a
         * partial band as a finished one and strand the rest of the q range. */
        const int commit = (rc == 0 && !stopped);
        if (pipe_finalize_outputs(&fr, &fc, cfg, rtmp, ctmp, commit,
                                  ckpt_written))
            rc = -1;
        outputs_finalized = 1;
        if (commit && rc == 0 && ckpt_armed) {
            char cp[CKPT_PATH_MAX], rp[CKPT_PATH_MAX], rtp[CKPT_PATH_MAX];
            /* Cannot fail here: the same name was built for every checkpoint
             * this band wrote, into a buffer of the same size. Checked anyway
             * because the band is already committed -- a stale sidecar is a
             * better outcome than an ignored return value. */
            if (ckpt_ckpt_path(cfg->relations, cp, sizeof cp) == 0)
                remove(cp);      /* nothing left to resume */
            /* A BOINC recovery budget spans the whole workunit, not merely one
             * checkpoint interval. Clear it only after the final output has
             * committed; otherwise a deterministic resume defect could write
             * one good checkpoint per attempt and restart forever. */
            if (ckpt_recovery_path(cfg->relations, rp, sizeof rp) == 0)
                remove(rp);
            if (ckpt_recovery_tmp_path(cfg->relations, rtp, sizeof rtp) == 0)
                remove(rtp);
        }
    }
#ifdef HAVE_BOINC
    if (rc == 0) bench_boinc_fraction_done(0.99);
#endif
    if (rc == 0) {
        /* Query only after the scientific output is committed. Per-q buffers
         * grow on demand, so this remains useful telemetry, but its failure is
         * never allowed to change a completed workunit into an error. */
        size_t fb_ = 0, tb_ = 0;
        if (pipe_mem_info_optional("steady-state report", &fb_, &tb_) == 0) {
            printf("\n  device memory, steady state: %.2f GB in use of %.2f GB"
                   " (%.2f GB free)\n",
                   (tb_ - fb_) / 1073741824.0, tb_ / 1073741824.0,
                   fb_ / 1073741824.0);
        }
    }
    if (rc) {
        const char *fate = ckpt_written
            ? "the .part is kept; rerun the same command to resume"
            : "nothing was checkpointed, so no output is kept";
        /* Into the log as well as stderr. A band that dies on its FIRST q
         * never reaches the summary below -- that whole block is guarded on
         * nqdone -- so without this the log ends with a header, possibly a few
         * heartbeats, and no indication that anything went wrong. Read a week
         * later, or under BOINC where stderr is a separate upload, that is
         * indistinguishable from a run still in progress. */
        if (qgen)
            runlog_warn("  band FAILED after %u generated q; %s",
                        nqdone, fate);
        else
            runlog_warn("  band FAILED after %u of %u q; %s",
                        nqdone, nq, fate);
    }
    if (stopped && ckpt_written)
        printf("\n  stopped after %u q this session (%llu total, %llu"
               " relations). Rerun the same command to resume at q=%llu.\n",
               nqdone, base_nq + nqdone, ck.nrel,
               (unsigned long long)ck.next_q);
    else if (stopped)
        fprintf(stderr,
                "\n  stopped after %u q, but NO checkpoint could be written."
                " %s cannot be\n  resumed automatically; move it aside or pass"
                " --restart.\n", nqdone, rtmp);

    /* OUTSIDE the `if (nqdone)` below, because the case that most needs saying
     * is a band where skips left nqdone == 0: guarded, the one run that
     * produced nothing but skips would have reported nothing at all. A band
     * runs for days and scrolls, so this has to survive in the summary and not
     * only in the per-q warnings. Silent is the one thing this must not be. */
    if (nqskip)
        printf("\n  ** %llu special-q SKIPPED: exact norm wider than %d bits."
               " Run ./normscan over the band and rebuild wider. **\n",
               nqskip, BN_LIMBS * 32);
    if (nqdone) {
        const double N = nqdone;
        const double dev = (tm.rank + tm.emit + tm.summary + tm.resieve + tm.td
                            + tm.classify + tm.compact + tm.record) / N;
        const double acc_sieve = acc_tr + acc_fi + acc_ap;
        printf("\n\n  --- band of %u special-q ---\n", nqdone);
        if (t_verify > 0)
            printf("  (first-q reconstruction gate: %.1f ms, excluded below)\n",
                   t_verify);
        printf("  %-34s %8.2f ms\n", "wall clock per q", acc_wall / N);
        printf("  %-34s %8.2f ms\n", "  sieve, both sides", acc_sieve / N);
        printf("  %-34s %8.3f ms\n", "    transform + plattice", acc_tr / N);
        printf("  %-34s %8.3f ms\n", "    fill", acc_fi / N);
        printf("  %-34s %8.3f ms\n", "    apply", acc_ap / N);
        printf("  %-34s %8.3f ms\n", "  intersect + gcd", acc_isect / N);
        printf("  %-34s %8.3f ms\n", "  host per-q (sieve tables, staging)",
               acc_host / N);
        printf("  %-34s %8.2f ms   <- of which, on the device:\n",
               "  TD + classify, wall", acc_td / N);
        printf("      %-30s %8.3f ms\n", "rank scan (shared)", tm.rank / N);
        printf("      %-30s %8.3f ms\n", "emit (x,a,b), rank order (shared)",
               tm.emit / N);
        printf("      %-30s %8.3f ms\n", "survivor filter (shared)", tm.summary / N);
        printf("      %-30s %8.3f ms\n", "resieve + scatter, both sides",
               tm.resieve / N);
        printf("      %-30s %8.3f ms\n", "norms + trial division, both sides",
               tm.td / N);
        printf("      %-30s %8.3f ms\n", "classify, both sides", tm.classify / N);
        printf("      %-30s %8.3f ms\n", "joint accept + compact", tm.compact / N);
        printf("      %-30s %8.3f ms\n", "record candidate factorisations",
               tm.record / N);
        printf("      %-30s %8.3f ms\n", "= device total", dev);
        printf("      %-30s %8.3f ms\n", "host: small-prime tables", tm.hostq / N);
        printf("      %-30s %8.3f ms\n", "host: readback of candidates",
               tm.readback / N);
        printf("      %-30s %8.3f ms\n", "host: unaccounted",
               acc_td / N - dev - tm.hostq / N - tm.readback / N);
        printf("  %-34s %8.2f ms\n", "  join and emit", tm.join / N);
        if (cfg->cofactor)
            printf("  %-34s %8.2f ms\n", "  cofactorisation, in-loop flushes",
                   tm.cofac / N);
        printf("  %-34s %8.2f ms\n", "  unaccounted",
               (acc_wall - acc_sieve - acc_isect - acc_host - acc_td - tm.join
                - tm.cofac) / N);
        /* The one number that makes host starvation visible. Every stage above
         * is either a cudaEvent (blind to what the CPU is doing) or a host
         * clock, so a contended box prints PERFECT kernel times next to a bad
         * wall clock and nothing says why. Finding 53: saturating this box's 16
         * cores left fill and apply flat within 1% while wall went 24.30 ->
         * 31.27 ms/q. No idle/loaded pair is quoted here on purpose: the first
         * one measured (0.824 / 0.661) used an earlier expression that kept the
         * cofactor queue in the denominator only, and no pair has yet been
         * taken with THIS one on a box confirmed idle. Finding 53 is canonical
         * and says so; a number duplicated into a comment is a number that
         * drifts, which is how the 0.824 and a hand-derived 0.85 ended up in
         * the docs and this file describing the same measurement.
         *
         * THE COFACTOR QUEUE IS OUT OF BOTH SIDES. dev covers only the
         * event-timed TD/classify kernels; the queue's flush is measured with a
         * host clock (tm.cofac) and mixes device and host, so it can go in
         * neither term. Leaving it in the denominator alone made the ratio move
         * with SURVIVOR DENSITY -- a band with more candidates flushes more and
         * reads as a contended host on a perfectly idle box, which is the exact
         * misdiagnosis this line exists to prevent.
         *
         * Even so the numerator is a LOWER BOUND on device time: k_cof_enqueue,
         * k_cand_stats and the flush's own kernels are real GPU work that no
         * event times, so the ratio understates utilisation by a small fixed
         * amount. That is tolerable for the comparison it is for and fatal for
         * the one it is not -- hence:
         *
         * Deliberately NO threshold and no warning text. The healthy value is a
         * function of the card and the job -- a faster GPU spends relatively
         * more of its wall on the same host work, so it sits LOWER when
         * perfectly healthy -- and a hardcoded "good" constant here would be
         * the same mistake as the 144-block default: one box's number promoted
         * to a universal one. Compare it against your own idle baseline on the
         * same card, job AND band length; that comparison is valid, an absolute
         * reading is not.
         *
         * Band length matters because acc_wall excludes the final cofactor
         * flush (see the comment on cofac_tail above). On a band shorter than
         * one flush that tail IS the whole cofactorisation, so a short smoke
         * run and a production band are not comparable to each other. */
        printf("  %-34s %8.3f\n", "GPU-accounted / wall (excl cofac)",
               (acc_sieve + acc_isect + dev * N) / (acc_wall - tm.cofac));
        printf("  %-34s %8.1f\n", "two-sided primitive survivors/q",
               (double)acc_surv / N);
        printf("  %-34s %8.2f\n", "cofactorisation candidates/q",
               (double)acc_cand / N);
        printf("  %-34s %8.3f\n", "COMPLETE RELATIONS/q", (double)acc_rel / N);
        printf("  %-34s %8llu\n", "total relations", (unsigned long long)acc_rel);
        printf("  %-34s %8llu\n", "total candidates", (unsigned long long)acc_cand);
        /* The band summary in one record, so the log ends with the numbers a
         * reader would otherwise have to reconstruct from the last heartbeat
         * and the console output the session no longer has. Timing terms are
         * the summary's own, so the two cannot disagree.
         *
         * `nq` and `rel` MEAN HERE WHAT THEY MEAN IN A HEARTBEAT: totals
         * across sessions, and `rel` counted the way the heartbeat counts it
         * -- Q.nrel under --cofactor, which is what reached the file, not
         * acc_rel, which is only the subset trial division completed without
         * splitting (111 against 374 on a 3-q c147 smoke run). The
         * session-only aggregates get their own names rather than reusing
         * those two keys for a second quantity: a log where one key means two
         * things is worse than one field short, because nothing in the file
         * says which reading applies to which line. */
        runlog_record("band end  nq=%llu  rel=%llu  nq_session=%u"
                      "  rel_session=%llu  cand_session=%llu  wall=%.2fms/q"
                      "  sieve=%.2fms/q  td=%.2fms/q  acc/wall=%.3f"
                      "  rel/q=%.3f%s",
                      base_nq + nqdone,
                      base_rel + (cfg->cofactor ? Q.nrel
                                                : (unsigned long long)acc_rel),
                      nqdone,
                      cfg->cofactor ? Q.nrel : (unsigned long long)acc_rel,
                      (unsigned long long)acc_cand,
                      acc_wall / N, acc_sieve / N, acc_td / N,
                      (acc_sieve + acc_isect + dev * N) / (acc_wall - tm.cofac),
                      /* rel_session per q, not the summary's COMPLETE
                       * RELATIONS/q, which counts only what trial division
                       * finished -- next to rel_session on the same line the
                       * two would read as a contradiction. */
                      (double)(cfg->cofactor ? Q.nrel
                                             : (unsigned long long)acc_rel) / N,
                      stopped ? "  [stopped cleanly]" : (rc ? "  [FAILED]" : ""));
        if (cfg->cofactor) {
            printf("\n  --- cofactorisation, cross-q queue ---\n");
            printf("  %-34s %8.2f ms\n", "rational queue", Q.ms_rat / N);
            printf("  %-34s %8.2f ms\n", "algebraic queue", Q.ms_alg / N);
            printf("  %-34s %8.3f ms\n", "readback + emit of relations", Q.ms_host / N);
            printf("  %-34s %8.2f ms\n", "  = device time per q",
                   (Q.ms_rat + Q.ms_alg + Q.ms_host) / N);
            printf("  %-34s %8.2f ms\n", "wall: in-loop flushes", tm.cofac / N);
            printf("  %-34s %8.2f ms\n", "wall: final flush after the band",
                   cofac_tail / N);
            /* Everything joint-accepted is enqueued, INCLUDING the records
             * trial division already completed -- the queue is the single
             * emitter, so they have to travel with the rest. They cost nothing
             * to split (their status is already CF_OK, and compaction drops
             * them before the first round), but they are counted here, so this
             * is a record count and not a splitting-work count. Labelling it
             * "cofactorised" made it look like a 7-record discrepancy against
             * the candidate total at the parity q. */
            printf("  %-34s %8llu  (of which %llu needed no splitting)\n",
                   "records enqueued", (unsigned long long)Q.nseen,
                   (unsigned long long)acc_rel);
            /* Per-side outcome, and in particular the STUCK count.
             *
             * This exists because its absence hid two real yield losses: 13.7%
             * on AS276 and 0.33% on the c183 (RESULTS findings 69 and 70). A
             * record whose splitter ran out of budget is left CF_INCOMPLETE,
             * quietly fails to become a relation, and is indistinguishable in
             * every other printed number from a job that simply had no more
             * relations to give. Only a corpus comparison found it. Now the
             * run says so itself. */
            {
                const unsigned long long st0 = Q.nst[0][CF_INCOMPLETE];
                const unsigned long long st1 = Q.nst[1][CF_INCOMPLETE];
                const unsigned long long stuck = st0 + st1;
                for (int sd = 0; sd < 2; sd++) {
                    char lbl[40];
                    snprintf(lbl, sizeof lbl, "  side %d: split / dead / stuck", sd);
                    printf("  %-34s %8llu / %llu / %llu\n", lbl,
                           Q.nst[sd][CF_OK], Q.nst[sd][CF_DEAD],
                           Q.nst[sd][CF_INCOMPLETE]);
                    /* Two producers, so the label names neither: mz_split
                     * sets CF_OVERFLOW when a split yields more than
                     * CF_MAXFAC parts, and k_cof_enqueue sets it when a
                     * residual is wider than the side's limbs. Saying
                     * "CF_MAXFAC" would point an operator at the splitter's
                     * stack guard when the actual cause was mfb vs CF_LMAX. */
                    if (Q.nst[sd][CF_OVERFLOW])
                        printf("  %-34s %8llu\n",
                               "    OVERFLOW (parts, or cofactor width)",
                               Q.nst[sd][CF_OVERFLOW]);
                }
                /* Warned, not merely printed, because the band otherwise
                 * exits 0 looking healthy.
                 *
                 * "Unresolved", NOT "lost". A stuck record's status is
                 * UNKNOWN, and the two populations behind it are very
                 * different: it may be a cofactor that would have split into a
                 * relation, or a composite whose factors ALL exceed lpb --
                 * which can never be a relation, but which the splitter can
                 * only prove dead by factoring it anyway. At a tight mfb/lpb
                 * ratio the second population dominates, so a large stuck
                 * count is not by itself a large yield loss. Measuring which
                 * is which costs a run at a bigger budget, which is exactly
                 * what this line is telling the operator to do. */
                /* A RATE, not any nonzero count. A correctly tuned 3LP band
                 * still leaves plenty of records unresolved -- AS276 at its
                 * saturated-yield configuration sits at 20.9% -- because most
                 * of them are composites whose factors all exceed lpb, which
                 * can only be proven dead by factoring them. Warning on
                 * `stuck > 0` would therefore fire on nearly every healthy
                 * production band, and under BOINC stderr is uploaded per work
                 * unit. The counts above are always printed; this fires only
                 * when the rate is high enough to be worth a look.
                 *
                 * A third is a heuristic with two data points behind it, not a
                 * derived bound: AS276 sits at 20.9% when correctly tuned and
                 * 55% when its budget was five times too small. Nothing inside
                 * a single run distinguishes the two, which is why the message
                 * names the actual test rather than asserting a fault. */
                if (Q.nseen && stuck * 3ull > Q.nseen)
                    runlog_warn("  %llu of %llu records (%.1f%%) left unresolved by"
                                " the splitting budget. That is high; a healthy 3LP"
                                " band runs ~20%%. These are cofactors of UNKNOWN"
                                " status, so some may have been relations -- re-run"
                                " a short band at a higher --cof-budget /"
                                " --cof-rounds and see whether the yield moves.",
                                stuck, Q.nseen,
                                100.0 * (double)stuck / (double)Q.nseen);
            }
            /* Q.nrel is EVERY relation: the queue is the single emitter, so
             * it already holds the ones trial division completed. */
            printf("  %-34s %8.2f\n", "ALL RELATIONS/q", (double)Q.nrel / N);
            printf("  %-34s %8.2f\n", "  of which, from cofactorisation",
                   (double)(Q.nrel - acc_rel) / N);
            printf("  %-34s %8llu\n", "total relations",
                   (unsigned long long)Q.nrel);
            /* acc_wall ALREADY contains the queue: the flush happens inside
             * the q loop. Adding the device times to it would double count,
             * which the first version of this line did -- it reported 132.31
             * ms/q against a 112 ms/q process wall clock. */
            printf("  %-34s %8.2f ms\n", "wall clock per q, COMPLETE",
                   (acc_wall + cofac_tail) / N);
        }
    }

#undef PIPE_CK
done:
    if (stop_hooked) bench_stop_hook_remove();
    /* CUDA failures can jump here from any stage. Normal completion already
     * finalized exactly once above; an early jump closes and discards here --
     * but still honours a written checkpoint, since a .part that a sidecar
     * describes is resumable work whatever killed this session. */
    if (!outputs_finalized) {
        if (pipe_finalize_outputs(&fr, &fc, cfg, rtmp, ctmp, 0, ckpt_written))
            rc = -1;
    }
    pipe_td_free(&C);
    cofq_free(&Q, &QO);
    pside_free(&S1); pside_free(&S0);
    cudaFree(d_bucket); cudaFree(d_cursor); cudaFree(d_overflow);
    cudaFree(d_two); cudaFree(d_n); cudaFree(d_pre);
    if (ea) cudaEventDestroy(ea);
    if (eb) cudaEventDestroy(eb);
    return rc;
}

extern "C" int run_pipeline(const fb_t *fb1, const fb_t *fbs1,
                            const fb_t *fb0, const fb_t *fbs0,
                            const qsel_t *qlist, uint32_t nq, sqgen_t *qgen,
                            const poly_t *POLY, const bench_cfg_t *cfg)
{
    slab_plan_t plan;
    uint32_t pmax1 = fb_max_td_prime(fbs1);
    uint32_t pmax0 = fb_max_td_prime(fbs0);
    uint32_t pmax = pmax1 > pmax0 ? pmax1 : pmax0;

    if (slab_make_plan(cfg->logI, cfg->log_region, cfg->J, pmax,
                       cfg->slab_j, &plan)) {
        fprintf(stderr,
                "  pipeline: cannot make a safe slab plan for logI=%d J=%u"
                " (largest direct-test prime %u, requested --slab-j %u)\n",
                cfg->logI, cfg->J, pmax, cfg->slab_j);
        return -1;
    }
    /* The A/B record path leaves no other trace: the stage label is the same on
     * both branches and the checkpoint job text does not carry the flag, so a
     * sweep log that mixes the two kernels would have no tell. Finding 78's
     * method note is what this is for -- an entire sweep was discarded there
     * over a harness mixup caught only by row ordering. */
    if (cfg->td_record_scalar)
        printf("  recording pass: SCALAR (one thread per candidate,"
               " --td-record-scalar) -- the pre-finding-77 path, for A/B only\n");

    if (plan.enabled) {
        const uint32_t perf_jmax = slab_perf_jmax(cfg->logI, cfg->log_region,
                                                  cfg->J);
        if (!cfg->slab_j && perf_jmax != 0xffffffffu)
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (auto target %u bucket regions at --region %d;"
                   " safety bounds may reduce it further)\n",
                   plan.nslab, plan.nslab == 1 ? "" : "s", plan.jmax,
                   (unsigned)SLAB_PERF_REGIONS, cfg->log_region);
        else
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (<=2^31 local-position safety bound)\n",
                   plan.nslab, plan.nslab == 1 ? "" : "s", plan.jmax);
        return run_pipeline_impl<true>(fb1, fbs1, fb0, fbs0, qlist, nq, qgen,
                                       POLY, cfg, &plan);
    }
    return run_pipeline_impl<false>(fb1, fbs1, fb0, fbs0, qlist, nq, qgen,
                                    POLY, cfg, &plan);
}

#endif  /* CUDA_SIEVE_PIPELINE_CUH */
