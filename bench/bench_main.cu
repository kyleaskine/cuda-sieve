/* CLI for the standalone bucket-fill benchmark. */
#include "bench.h"
#include "platform.h"
#include "ckpt.h"
#include "runlog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>

static int parse_u64_arg(const char *flag, const char *arg, uint64_t *out)
{
    if (bench_parse_u64_decimal(arg, out)) {
        fprintf(stderr, "%s: not an unsigned 64-bit decimal integer: %s\n",
                flag, arg ? arg : "(null)");
        return -1;
    }
    return 0;
}

static int parse_u32_range_arg(const char *flag, const char *arg,
                               uint32_t lo, uint32_t hi, uint32_t *out)
{
    uint64_t v;

    if (parse_u64_arg(flag, arg, &v)) return -1;
    if (v < lo || v > hi) {
        fprintf(stderr, "%s: must be in [%u,%u], got %llu\n",
                flag, lo, hi, (unsigned long long)v);
        return -1;
    }
    *out = (uint32_t)v;
    return 0;
}

static int parse_int_range_arg(const char *flag, const char *arg,
                               int lo, int hi, int *out)
{
    char *end = NULL;
    long v;

    if (!arg || !*arg) {
        fprintf(stderr, "%s: not an integer: %s\n",
                flag, arg ? arg : "(null)");
        return -1;
    }
    errno = 0;
    v = strtol(arg, &end, 10);
    if (errno == ERANGE || end == arg || *end) {
        fprintf(stderr, "%s: not an integer: %s\n", flag, arg);
        return -1;
    }
    if (v < lo || v > hi) {
        fprintf(stderr, "%s: must be in [%d,%d], got %ld\n",
                flag, lo, hi, v);
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int parse_finite_double_arg(const char *flag, const char *arg,
                                   double *out)
{
    if (bench_parse_finite_double(arg, out)) {
        fprintf(stderr, "%s: not a finite decimal number: %s\n",
                flag, arg ? arg : "(null)");
        return -1;
    }
    return 0;
}

static int parse_positive_double_arg(const char *flag, const char *arg,
                                     double *out)
{
    double v;
    if (parse_finite_double_arg(flag, arg, &v)) return -1;
    if (v <= 0.0) {
        fprintf(stderr, "%s: must be positive, got %.17g\n", flag, v);
        return -1;
    }
    *out = v;
    return 0;
}

static int parse_nonnegative_double_arg(const char *flag, const char *arg,
                                        double *out)
{
    double v;
    if (parse_finite_double_arg(flag, arg, &v)) return -1;
    if (v < 0.0) {
        fprintf(stderr, "%s: must be nonnegative, got %.17g\n", flag, v);
        return -1;
    }
    *out = v;
    return 0;
}

static void usage(void)
{
    printf(
"usage: bench [options]\n"
"\n"
"RUNNING A JOB\n"
"  A typical run supplies the job, sieve geometry/band, and output. Everything\n"
"  else has a right answer derived from the polynomial or .job file and is printed:\n"
"\n"
"    bench --pipeline --cofactor --poly JOB.job \\\n"
"          --logI 14 --qrange 15000000: --target-rels 65000000 \\\n"
"          --relations msieve.dat\n"
"\n"
"  --pipeline       BOTH SIDES in one process: sieve, intersect, TD, classify,\n"
"                   join, and emit relations + cofactorisation candidates.\n"
"                   Runs one fixed configuration; the harness options below\n"
"                   are refused rather than silently ignored\n"
"  --poly PATH      polynomial. A GGNFS .job works directly, and its rlim,\n"
"                   alim, lpbr/lpba, mfbr/mfba and lambdas are USED -- they do\n"
"                   not need repeating below. A CADO .poly carries none of\n"
"                   those, so state them or let them derive\n"
"  --fb1 PATH       native fbgen factor base. Optional: if omitted in pipeline\n"
"                   mode, the complete algebraic FB is generated on the assigned\n"
"                   GPU (including exact CPU Hensel/prime-power branches). To\n"
"                   cache that startup once, use fbgen_gpu --out FILE and pass\n"
"                   the resulting FILE here on later runs\n"
"  --cadofb PATH    legacy alias for --fb1 (CADO files remain compatible)\n"
"  --fb PATH        legacy GGNFS .afb.0 input; sieve/debug only, not relations\n"
"  --logI N         log2 of sieve width I      [15]   (gnfs-lasieve4I14e -> 14)\n"
"  --J N            sieve height J             [2^(logI-1), CADO's convention]\n"
"  --slab-j N       pipeline: force at most N j rows per slab; 0/omitted =\n"
"                   automatic. Auto targets 32768 bucket regions/slab, so BOTH\n"
"                   the cap and the split trigger move with --region: at the\n"
"                   default --region 14 the cap is 2^29 positions and the\n"
"                   trigger 2^30, at --region 12 they are 2^27 and 2^28. An\n"
"                   area below TWICE the target is not split for performance\n"
"                   alone (deliberate hysteresis)\n"
"  --qspan          pipeline: report each q's GPU-timeline span, which\n"
"                   splits `unaccounted` into host-with-no-GPU-work-in-\n"
"                   flight and idle between stages (STATUS item 19)\n"
"  --relations F    write complete relations here (GGNFS/msieve format)\n"
"  --cofactor       split the cofactors INLINE, in a cross-q device queue;\n"
"                   --relations then holds every relation, not just TD's\n"
"\n"
"BAND SELECTION\n"
"  --qrange MIN:MAX generate every prime special-q and every affine root of\n"
"                   the selected side's polynomial in that inclusive range.\n"
"                   MIN: generates upward until --target-rels or --nq stops it\n"
"  --sq-side S      which side carries the special-q: 1 = algebraic (GNFS,\n"
"                   the default), 0 = rational. An SNFS job whose algebraic\n"
"                   coefficients are tiny puts the difficulty on the rational\n"
"                   side, and the q and the 3LP mfb go there with it\n"
"  --qlist FILE     band of special-q: `q rho` per line (# comments ok);\n"
"                   q must be prime, rho is reduced mod q, bad lines are fatal\n"
"  --q N / --rho N  a single special-q          [120000011]  (las -v prints rho)\n"
"  --nq N           stop after N special-q from the list or generated range\n"
"  --target-rels N  stop once N relations have been collected. Checked at\n"
"                   flush boundaries, so it overshoots by under a flush.\n"
"                   Pair with --qrange MIN: to sieve upward until satisfied.\n"
"                   Counts relations already on disk when resuming\n"
"\n"
"STOPPING AND RESUMING  (--relations runs only)\n"
"  Relations stage to NAME.part and are renamed to NAME when the band finishes.\n"
"  After every cofactor flush -- the one instant the file holds a whole number\n"
"  of special-q -- the .part is fsynced and NAME.part.ckpt records the next\n"
"  (q, rho), the byte offset and the derived scale. Rerunning the SAME command\n"
"  resumes there: the .part is truncated to that offset, discarding any torn\n"
"  final line, and sieving continues. An empty .part with no checkpoint is\n"
"  discarded automatically. Under a BOINC client, unusable staging/checkpoint\n"
"  artifacts are discarded and recomputed, up to three times per workunit;\n"
"  standalone runs still refuse nonempty or mismatched/corrupt artifacts.\n"
"  SIGINT or SIGTERM stops cleanly at the next special-q, draining the queue\n"
"  first, so a planned stop loses no work; a second signal exits at once and\n"
"  falls back to the previous checkpoint.\n"
"  --restart        discard an existing .part and its checkpoint, start over\n"
"  --stop-file P    stop cleanly once path P exists (for unattended runs)\n"
"  --log PATH       append a run log: a header naming the commit, argv, job\n"
"                   fingerprint, card, geometry and FB convention, then a\n"
"                   timestamped record carrying progress alongside\n"
"                   GPU-accounted/wall, GPU utilisation, board watts and host\n"
"                   load -- the four numbers that say whether the progress\n"
"                   ones can be compared to anything (finding 53)\n"
"  --log-every S    seconds between run-log records         [300]\n"
"\n"
"PARAMETERS  (precedence: this flag > .job file > derived from the poly)\n"
"  The byte scale and survivor allowance are ALWAYS derived, as las does, from\n"
"  the largest norm over the sieve rectangle. Stating one overrides it.\n"
"  --rlim N / --alim N  factor base bounds, side 0 / side 1\n"
"  --lpb N / --lpb0 N   large-prime bound in bits  [32 side 1, 31 side 0]\n"
"  --mfb N / --mfb0 N   max cofactor bits          [92 side 1, 60 side 0]\n"
"  --cof-ecm / --cof-rho   force one method on BOTH sides. The default is\n"
"                   per-side and automatic: rho for a 2LP side, ECM for a 3LP\n"
"                   one (ceil(mfb/lpb) >= 3). Measured: rho wins ~1.1x at 2LP,\n"
"                   ECM wins 2-4x at 3LP. A typical job is one of each.\n"
"  --ecm-b1 N / --ecm-b2 N / --ecm-curves N   [derived from lpb: B1 200 at\n"
"                   lpb<=33, 300 at 34-35, 500 at 36+; B2 = 30*B1; 12 curves]\n"
"  --cof-limbs N / --cof-limbs0 N   cofactor width in 32-bit limbs\n"
"                   [narrowest that mfb needs: 3 up to 96 bits, 4 above].\n"
"                   Force it UP to time the same job at both widths; forcing\n"
"                   it below what mfb needs is refused.\n"
"  --allowance B / --allowance0 B   survivor cofactor BITS, overriding the\n"
"                   derived default. The default is mfb + the slack our own\n"
"                   byte-quantised survivor test needs (~1.5 bits), NOT the\n"
"                   .job file's lambda -- that is calibrated to GGNFS's gate\n"
"                   and does not transfer. It is reported, not applied\n"
"  --lambda0/1 L    opt back into CADO's rule: lambda in CADO units, i.e.\n"
"                   multiples of lpb [0 = CADO's automatic 0.3 + mfb/lpb].\n"
"                   Looser than the derived default on every job measured\n"
"  --scale S / --scale0 S   las byte scale per side\n"
"  --fbbound N      truncate FB at this p      [alim]  (GGNFS truncates at q)\n"
"  --bkthresh N     bucket-sieve p >= this     [1<<logI]\n"
"  --region N       log2 bucket region size    [14]  (16384 16-bit cells, 32 KB)\n"
"  --maxbits N      prime powers below 2^N      [logI]; used by generated FBs,\n"
"                   and should match the maxbits recorded by a cached --fb1 file\n"
"\n"
"COFACTORISATION\n"
"  --cof-rounds N   rho requeue rounds, budget doubling each time\n"
"                   [6 for --cofac; 2 for --pipeline --cofactor]\n"
"  --cof-budget N   rho iterations in the first round\n"
"                   [4096 for --cofac; 65536 for --pipeline --cofactor]\n"
"  --cof-ecm        ECM instead of Pollard-Brent rho; stage 1 alone loses,\n"
"                   while tuned stage 2 is near rho at matched yield\n"
"  --ecm-b1 N       ECM stage-1 bound                              [1000]\n"
"  --ecm-b2 N       ECM D=30 stage-2 bound; 0 disables stage 2         [0]\n"
"  --ecm-curves N   ECM curves attempted per round                 [16]\n"
"  --candidates F   write the cofactorisation batch here\n"
"  --cofac FILE     cofactorise a batch written by --candidates and emit the\n"
"                   relations to --relations; needs no sieving\n"
"\n"
"CHECKING\n"
"  --verify-only    the Franke-Kleinjung walk gate only, then exit. Needs no\n"
"                   GPU, no factor base and no polynomial, which is what lets\n"
"                   `make check` run it on a card-less or busy box\n"
"  --check-relations F  verify an emitted relation file: every factor divides,\n"
"                   both norms rebuild to 1, every prime within its lpb\n"
"  --cofgate FILE   gate the cofactors against CADO's own (a b cof0 cof1);\n"
"                   under --pipeline this runs in the first-q validation\n"
"  --no-td-verify   skip dense TD reconstruction (first q in pipeline; TD\n"
"                   harness otherwise). Saves peak memory; incompatible with --cofgate\n"
"  --verbose-q      print a line per special-q instead of a band summary\n"
"  --td-record-scalar\n"
"                   pipeline: record candidate factorisations with one\n"
"                   THREAD per candidate instead of one WARP. The\n"
"                   pre-2026-08-26 path, kept for A/B only -- it is ~5x\n"
"                   slower per launch and emits identical relations\n"
"\n"
"RUNTIME\n"
#ifdef HAVE_BOINC
"  --device N       select CUDA device N, used only when the BOINC client did\n"
"                   not assign one; its assignment wins  [CUDA's default]\n"
#else
"  --device N       select CUDA device N  [CUDA's default device]\n"
#endif
"  --threads N      threads per block, multiple of 32  [256]\n"
"  --blocks N       0 = auto (6 per SM)        [0]\n"
"  --fill-blocks N  fill only; 0 = auto (4608, absolute -- NOT per SM) [0]\n"
"  --fill-streams N fill only; N independent workspaces on N streams, timed\n"
"                   against the same N issued serially and against 1 kernel\n"
"                   at N x the blocks. 0/1 = off [0]. Costs a bucket array\n"
"                   per workspace (item 1)\n"
"  --fill-threads N fill only; 0 = auto (32), else a multiple of 32 in\n"
"                   [32,1024]. Independent of --threads: fill wants many\n"
"                   narrow blocks, the other kernels do not.            [0]\n"
"  --blocking-sync  yield the CPU while waiting on the GPU instead of spinning.\n"
"                   CUDA busy-waits by default, so a host thread that is 90%%\n"
"                   idle still pegs a core; this frees it, at the cost of a\n"
"                   wakeup latency per synchronisation\n"
"\n"
"BENCHMARK HARNESS  (single-side measurement; REFUSED under --pipeline)\n"
"  These reproduce the numbers in RESULTS.md. They select configurations that\n"
"  were measured and rejected, so they are not knobs to tune a real run with.\n"
"  --fb PATH        GGNFS .afb.0 factor base   [../oracle/input.job.afb.0]\n"
"  --side S         1 = algebraic (special-q side), 0 = rational  [1]\n"
"  --record-bytes N 2 | 4 | 8                  [4]\n"
"  --mode M         atomic | twolevel          [atomic]  (twolevel lost by 2.7x)\n"
"  --stage S        fill | both | apply        [both]\n"
"  --cells N        16 | 8 bits per sieve cell [16]  (8 is unsafe; cost only)\n"
"  --norm M         horner | const             [horner]\n"
"  --apply-mode M   atomic | plain             [atomic]  (plain is racy)\n"
"  --apply-threads N  threads per apply block  [512] (max 512)\n"
"  --reps N         timing repetitions         [3]\n"
"  --verify         run the CPU cross-check (slow)\n"
"  --no-smallsieve  skip the p < bkthresh line sieve\n"
"  --not-both-even  apply las's filter: i,j both even can never survive\n"
"  --survbits FILE  write a survivor bitmap (1 bit/position, x order)\n"
"  --other-bits F   the other side's bitmap -> device intersect+gcd+compact\n"
"  --emit FILE      write the compacted survivor list (x, a, b) here\n"
"  --emit-cof FILE  write (a, b, cofactor, bits) for every survivor here\n"
"  --td             exact norms + trial division on the survivors\n"
"  --ab-resieve     re-run the settled layout A/B resieve experiment (slow)\n"
"  --resieve-sweep  sweep the resieve unroll depth and summary granularity\n"
"  --dump PATH      write the sieve region in las byte convention\n"
"  --probe i,j      read back that cell after apply (gate 5)\n");
}

static int validate_qsel_or_report(qsel_t *sel, const poly_t *P, int side,
                                   const char *source, unsigned long line)
{
    const qsel_validate_result_t vr = qsel_validate(sel, P, side);
    if (vr == QSEL_VALID) return 0;

    if (line) fprintf(stderr, "%s:%lu: ", source, line);
    else      fprintf(stderr, "%s: ", source);
    switch (vr) {
    case QSEL_ERR_Q_RANGE:
        fprintf(stderr, "q = %llu is not in [2, 2^32)\n",
                (unsigned long long)sel->q);
        break;
    case QSEL_ERR_Q_COMPOSITE:
        fprintf(stderr, "q = %llu is composite\n",
                (unsigned long long)sel->q);
        break;
    case QSEL_ERR_SIDE:
        fprintf(stderr, "special-q side %d is not 0 or 1\n", side);
        break;
    case QSEL_ERR_POLY:
        fprintf(stderr, "cannot evaluate the exact %s polynomial modulo q\n",
                side ? "algebraic" : "rational");
        break;
    case QSEL_ERR_NOT_ROOT:
        fprintf(stderr, "rho = %llu is not a root of %s modulo q = %llu\n",
                (unsigned long long)sel->rho, side ? "f" : "G",
                (unsigned long long)sel->q);
        break;
    default:
        fprintf(stderr, "invalid special-q selection\n");
        break;
    }
    return -1;
}

/* Cofactoriser bounds. The representation is FIXED-WIDTH on purpose -- a
 * resulting prime is one uint64 and a cofactor is an mz<L> for one of the two
 * or three L this build carries -- so a bound outside that design does not
 * degrade, it silently truncates, and a relation that has lost the top of a
 * factor stops reconstructing its own norm. Refuse.
 *
 * The widths moved twice. 2026-08-17: split factors and unsplit prime residuals
 * were a single uint32, which capped lpb at 32. They are 64-bit now, which is
 * why the bound below is 64 and why an NFS@Home C194 (lpba 33, mfba 95) runs.
 * 2026-08-18: the COFACTOR width became a per-side run-time choice between 3
 * and 4 limbs, so mfb is bounded by 32*CF_LMAX rather than by a constant, and
 * this function is where the choice is resolved as well as checked.
 *
 * This MUST run after the .job file has been read. It used to sit inline in
 * main() ahead of poly_load, so it only ever saw command-line values -- and
 * since --lpb/--mfb are already range-checked by the parser, the interesting
 * cases (mfba 120 from a job file, or a 4LP mfb/lpb ratio) were exactly the
 * ones it could not see. The RUNBOOK's own SNFS recipe takes mfbr and lpbr
 * from the file. */
/* NAMED FOR WHAT IT DOES: it RESOLVES as well as checks. It fills in
 * cof_limbs0/cof_limbs, cof_meth0/cof_meth1 and the ECM parameters, and both
 * cofq_init() and run_cofac() read those, so this must run before either. A
 * reordering does not misbehave silently -- an unresolved width is 0 and
 * cofq_init refuses it by name -- but the old `check_` prefix suggested the
 * call was merely diagnostic and droppable, which it is not. */
static int resolve_and_check_cofactor_config(bench_cfg_t *cfg, uint32_t alim,
                                             uint32_t rlim, int cof_rounds,
                                             uint32_t cof_budget)
{
    const uint32_t lpb1 = cfg->lpb ? cfg->lpb : 32;
    const uint32_t mfb1 = cfg->mfb ? cfg->mfb : 92;
    int bad = 0;
    /* 64, not 32: split factors and unsplit prime residuals are stored in
     * uint64 throughout the queue, the emitter and the reconstruction gate, so
     * the representation ends at 64 and not before. Everything between 32 and
     * 64 that would nonetheless be wrong is refused by a check with an actual
     * reason -- the parts test below, and lim^2 > 2^lpb further down, which
     * binds around lpb 55 for an alim of 240M. A C194 asks for lpba 33. */
    if (lpb1 > 64 || cfg->lpb0 > 64) {
        fprintf(stderr, "lpb %u / side-0 %u: a resulting prime is stored in one"
                " 64-bit word, so lpb > 64 truncates it\n", lpb1, cfg->lpb0);
        bad = 1;
    }
    /* THE COFACTOR WIDTH. Each side runs at the narrowest instantiation that
     * holds its mfb -- 3 limbs (96 bits) up to a C194's mfba 95, 4 limbs (128)
     * for a C208 -- unless --cof-limbs/--cof-limbs0 forces it wider. Forcing
     * is only ever upward: a width below what mfb needs is the silent
     * truncation this check exists to refuse, so it is refused whether it
     * comes from a job file or from the command line.
     *
     * Resolved HERE rather than at the queue, because this is the one function
     * that has seen the .job file's mfb as well as the command line, and
     * because the width has to be refusable rather than clamped. */
    {
        const int w1 = cf_limbs_for_mfb(mfb1);
        const int w0 = cf_limbs_for_mfb(cfg->mfb0);
        if (!w1) {
            fprintf(stderr, "side-1 mfb %u: the widest cofactor this build has"
                    " is %d limbs (%d bits), so a residual above that loses its"
                    " high limbs -- rebuild with a larger CF_LMAX\n",
                    mfb1, CF_LMAX, 32 * CF_LMAX);
            bad = 1;
        }
        if (!w0) {
            fprintf(stderr, "side-0 mfb %u: the widest cofactor this build has"
                    " is %d limbs (%d bits), so a residual above that loses its"
                    " high limbs -- rebuild with a larger CF_LMAX\n",
                    cfg->mfb0, CF_LMAX, 32 * CF_LMAX);
            bad = 1;
        }
        if (cfg->cof_limbs && cfg->cof_limbs < w1) {
            fprintf(stderr, "--cof-limbs %d is narrower than side-1 mfb %u"
                    " needs (%d limbs)\n", cfg->cof_limbs, mfb1, w1);
            bad = 1;
        }
        if (cfg->cof_limbs0 && cfg->cof_limbs0 < w0) {
            fprintf(stderr, "--cof-limbs0 %d is narrower than side-0 mfb %u"
                    " needs (%d limbs)\n", cfg->cof_limbs0, cfg->mfb0, w0);
            bad = 1;
        }
        if (!cfg->cof_limbs)  cfg->cof_limbs  = w1;
        if (!cfg->cof_limbs0) cfg->cof_limbs0 = w0;
    }
    /* THE METHOD, per side. Same shape as the width resolution above and for
     * the same reason: this is the one function that has seen both the .job
     * file and the command line. AUTO picks rho for a 2LP side and ECM for a
     * 3LP one; --cof-ecm / --cof-rho force both sides, which is what an A/B
     * measurement needs and what reproduces the pre-2026-08-19 default. */
    {
        const int a1 = cof_auto_method(mfb1, lpb1);
        const int a0 = cof_auto_method(cfg->mfb0, cfg->lpb0);
        cfg->cof_meth1 = (cfg->cof_ecm == COF_METHOD_AUTO) ? a1 : cfg->cof_ecm;
        cfg->cof_meth0 = (cfg->cof_ecm == COF_METHOD_AUTO) ? a0 : cfg->cof_ecm;
        /* ECM parameters follow the lpb of whichever side actually runs ECM;
         * when both do, the wider one, since a B1 too small loses relations
         * and one too large only costs time. */
        if (!cfg->ecm_b1_set) {
            unsigned lp = 0;
            if (cfg->cof_meth1 == COF_METHOD_ECM && lpb1 > lp) lp = lpb1;
            if (cfg->cof_meth0 == COF_METHOD_ECM && cfg->lpb0 > lp) lp = cfg->lpb0;
            cfg->ecm_b1 = cof_auto_b1(lp ? lp : lpb1);
        }
        /* Keyed on _set, not on the value: --ecm-b2 0 means "no stage 2" and
         * --ecm-curves 0 must stay a refusal, so neither may be filled in
         * merely because it is zero. */
        /* 30*B1, CLAMPED INTO THE VALIDATOR'S OWN RANGE. A derived value must
         * never be able to fail the check that follows it: B1 below 30 is
         * outside the D=30 continuation's domain, and 30*B1 above 10^7 exceeds
         * the B2 ceiling -- so an unclamped rule turned `--ecm-b1 2` and every
         * `--ecm-b1` above 333,333 into a hard refusal citing an --ecm-b2 the
         * user never passed. Two thirds of --ecm-b1's own documented range was
         * unusable. Clamp, do not refuse: the user asked for a B1, not a B2. */
        if (!cfg->ecm_b2_set) {
            if (cfg->ecm_b1 < 30u)               cfg->ecm_b2 = 0u;
            else if (cfg->ecm_b1 > 10000000u / 30u) cfg->ecm_b2 = 10000000u;
            else                                 cfg->ecm_b2 = 30u * cfg->ecm_b1;
        }
        if (!cfg->ecm_curves_set) cfg->ecm_curves = 12u;
    }
    /* ceil(mfb/lpb) parts must fit in CF_MAXFAC. 3LP is what an SNFS job with
     * mfbr 88 / lpbr 31 asks for and is supported; 4LP is not, and it would
     * present as CF_OVERFLOW on the records that need it -- a partial yield
     * loss, not a failure -- so it is refused up front. */
    {
        const uint32_t p1 = cof_parts(mfb1, lpb1);
        const uint32_t p0 = cof_parts(cfg->mfb0, cfg->lpb0);
        if (p1 > 3 || p0 > 3) {
            fprintf(stderr, "mfb/lpb asks for %u parts on side 1 and %u on"
                    " side 0; the splitter handles at most 3\n", p1, p0);
            bad = 1;
        }
    }
    /* "prime by size" in mz_split rests on lim^2 > 2^lpb: below that a
     * composite could sit under lim^2 and be emitted as a prime. */
    {
        const uint64_t l1 = cfg->lim ? cfg->lim : alim;
        const uint64_t l0 = cfg->lim0 ? cfg->lim0 : rlim;
        if ((double)l1 * l1 <= ldexp(1.0, (int)lpb1)) {
            fprintf(stderr, "alim^2 (%.3g) <= 2^lpb: the prime-by-size test in"
                    " mz_split is unsound\n", (double)l1 * l1);
            bad = 1;
        }
        if ((double)l0 * l0 <= ldexp(1.0, (int)cfg->lpb0)) {
            fprintf(stderr, "rlim^2 (%.3g) <= 2^lpb0: the prime-by-size test in"
                    " mz_split is unsound\n", (double)l0 * l0);
            bad = 1;
        }
    }
    /* budget << r must not shift past the width, and a non-positive round
     * count would run no splitting at all yet still exit successfully. */
    if (cof_rounds < 1 || cof_rounds > 24) {
        fprintf(stderr, "--cof-rounds %d: must be 1..24 (budget << r overflows"
                " beyond that, and < 1 splits nothing)\n", cof_rounds);
        bad = 1;
    }
    if (cfg->cof_rounds < 1 || cfg->cof_rounds > 24) {
        fprintf(stderr, "pipeline cof-rounds %d: must be 1..24\n", cfg->cof_rounds);
        bad = 1;
    }
    /* Validate each method's knobs whenever ANY side uses it. Keyed on the
     * pair and not on a single flag: the common auto case is rho on one side
     * and ECM on the other, so an either/or here would skip the rho budget
     * checks on exactly the jobs that still run rho. */
    if (cfg->cof_meth0 == COF_METHOD_RHO || cfg->cof_meth1 == COF_METHOD_RHO) {
        if (!cof_budget || (uint64_t)cof_budget << (cof_rounds - 1) > 0xFFFFFFFFull) {
            fprintf(stderr, "--cof-budget %u with %d rounds overflows uint32"
                    " (or is zero)\n", cof_budget, cof_rounds);
            bad = 1;
        }
        if (!cfg->cof_budget ||
            (uint64_t)cfg->cof_budget << (cfg->cof_rounds - 1) > 0xFFFFFFFFull) {
            fprintf(stderr, "pipeline cof-budget %u with %d rounds overflows"
                    " uint32 (or is zero)\n", cfg->cof_budget, cfg->cof_rounds);
            bad = 1;
        }
    }
    if (cfg->cof_meth0 == COF_METHOD_ECM || cfg->cof_meth1 == COF_METHOD_ECM) {
        if (!cfg->ecm_curves) {
            fprintf(stderr, "--ecm-curves 0 attempts no curves\n"); bad = 1;
        }
        if (cfg->ecm_b1 < 2 || cfg->ecm_b1 > 1000000u) {
            fprintf(stderr, "--ecm-b1 %u: must be 2..1000000\n", cfg->ecm_b1);
            bad = 1;
        }
        if (cfg->ecm_b2 && (cfg->ecm_b1 < 30u ||
                            cfg->ecm_b2 <= cfg->ecm_b1 ||
                            cfg->ecm_b2 > 10000000u)) {
            fprintf(stderr, "--ecm-b2 %u: want 0 (disabled), or B1 < B2 <="
                    " 10000000 with B1 >= 30\n", cfg->ecm_b2);
            bad = 1;
        }
    } else if (cfg->ecm_b2_set) {
        /* Keyed on _set ALONE. Gating this on --cof-rho as well meant a
         * 2LP/2LP job that resolved to rho automatically accepted --ecm-b2 and
         * silently ignored it, which is the failure the diagnostic exists to
         * prevent. _set is already 1 only when the user typed the flag, so a
         * derived value can never reach here. */
        fprintf(stderr, "--ecm-b2 has no effect: both sides resolved to rho"
                " (2 large primes each)\n");
        bad = 1;
    }
    return bad;
}

/* The walk gate, hoisted so it can run with NO inputs at all: no GPU, no
 * factor base, no polynomial. `make check` wires it in as a CPU gate
 * (`walkcheck`), and a CPU gate that needs a card and a 60 MB untracked oracle
 * file is not one. --verify still runs it and then falls through to the
 * apply-parity benchmark, which is finding 32's gate and does want both. */
static int verify_walk_cases(void)
{
    static const struct { int logI; uint32_t J; const char *shape; }
    walk_cases[] = {
        { 8,   64, "4:1" }, { 8,  128, "2:1" }, { 8,  256, "1:1" },
        { 8,  512, "1:2" }, { 9,  256, "2:1" }, { 9,  512, "1:1" },
        { 10, 512, "2:1" }, { 10,1024, "1:1" },
    };
    const unsigned nwalk = sizeof walk_cases / sizeof walk_cases[0];
    printf("[verify] Franke-Kleinjung walk vs brute force, %u cases"
           " over 4:1 to 1:2...\n", nwalk);
    for (unsigned w = 0; w < nwalk; w++) {
        /* nc is the number of primes CHECKED. verify_walk's loop can fall
         * short of the quota without erroring (`checked < nprimes && p <
         * 40*I`), and testing only nc < 0 would let a case that checked
         * nothing pass vacuously. */
        const int nc = verify_walk(walk_cases[w].logI, walk_cases[w].J, 24);
        if (nc < 0) {
            printf("[verify] FAILED at logI=%d J=%u (%s): walk verifier"
                   " rejected the case\n",
                   walk_cases[w].logI, walk_cases[w].J, walk_cases[w].shape);
            return 1;
        }
        if (nc != 24) {
            printf("[verify] FAILED at logI=%d J=%u (%s): %d of 24 primes\n",
                   walk_cases[w].logI, walk_cases[w].J, walk_cases[w].shape, nc);
            return 1;
        }
    }
    printf("[verify] OK: %u cases x 24 primes x 5 roots enumerated"
           " identically, 4:1 through 1:2\n", nwalk);

    /* Shared with slabtest: one table, one 64-bit oracle, and large-prime
     * cases that exercise increments beyond 2^32 and native I17+ geometry. */
    {
        const int nslab = verify_walk_slab_cases();
        if (nslab < 0) {
            printf("[verify] SLAB FAILED\n");
            return 1;
        }
        printf("[verify] OK: %d forced/native slab walk cases, including"
               " partial tails and I17+\n", nslab);
    }
    return 0;
}

#define BOINC_RESUME_RETRIES 3u

typedef struct {
    const char *part, *ckpt;
    char ckpt_tmp[CKPT_PATH_MAX];
    char candidates_part[CKPT_PATH_MAX];
    char retry[CKPT_PATH_MAX], retry_tmp[CKPT_PATH_MAX];
    int checkpoint_candidate_required, checkpoint_candidate_known;
} resume_recovery_t;

static int resume_recovery_init(resume_recovery_t *r,
                                const bench_cfg_t *cfg,
                                const char *part, const char *ckpt)
{
    memset(r, 0, sizeof *r);
    r->part = part;
    r->ckpt = ckpt;
    if (ckpt_path_fmt(cfg->relations, r->ckpt_tmp, sizeof r->ckpt_tmp,
                      ".part.ckpt.tmp") ||
        ckpt_recovery_path(cfg->relations, r->retry, sizeof r->retry) ||
        ckpt_recovery_tmp_path(cfg->relations, r->retry_tmp,
                               sizeof r->retry_tmp) ||
        (cfg->candidates &&
         ckpt_part_path(cfg->candidates, r->candidates_part,
                        sizeof r->candidates_part)))
        return -1;
    return 0;
}

/* Read a persistent BOINC recovery counter. The temporary is read too: if the
 * process died between fsync and rename, forgetting that attempt would restore
 * the unbounded restart loop this counter exists to prevent. */
/* 1 = valid, 0 = absent, -1 = I/O error, -2 = malformed. The caller treats a
 * malformed durable counter as fatal, but a malformed temporary as evidence
 * that the process died while recording one additional attempt. */
static int recovery_count_read_one(const char *path, unsigned *count)
{
    /* Binary both ends: the format is "%u\n" and the parse below insists on
     * that exact '\n'. A text-mode write on Windows puts "\r\n" on disk, and
     * the same file read anywhere without translation then scans as three
     * fields with newline == '\r' -- reported malformed, which for the
     * durable counter is fatal. */
    FILE *f = fopen(path, "rb");
    unsigned n;
    char newline, extra;
    int fields, close_bad;

    if (!f) {
        if (errno == ENOENT) return 0;
        perror(path);
        return -1;
    }
    fields = fscanf(f, "%u%c%c", &n, &newline, &extra);
    close_bad = fclose(f);
    if (close_bad) {
        perror(path);
        return -1;
    }
    if (fields != 2 || newline != '\n' || n == 0) {
        return -2;
    }
    *count = n;
    return 1;
}

static int recovery_count_write(const resume_recovery_t *r, unsigned count)
{
    FILE *f = fopen(r->retry_tmp, "wb");
    int bad = 0, saved_errno = 0;
    if (!f) { perror(r->retry_tmp); return -1; }
    if (fprintf(f, "%u\n", count) < 0) { bad = 1; saved_errno = errno; }
    if (!bad && fflush(f)) { bad = 1; saved_errno = errno; }
    if (!bad && bench_sync_stream(f)) { bad = 1; saved_errno = errno; }
    if (fclose(f) && !bad) { bad = 1; saved_errno = errno; }
    if (bad) {
        errno = saved_errno;
        perror(r->retry_tmp);
        remove(r->retry_tmp);
        return -1;
    }
    if (bench_atomic_replace(r->retry_tmp, r->retry)) {
        perror(r->retry);
        remove(r->retry_tmp);
        return -1;
    }
    if (bench_sync_parent(r->retry)) {
        perror("recovery checkpoint directory");
        return -1;
    }
    return 0;
}

static const char *last_path_separator(const char *path)
{
    const char *slash = strrchr(path, '/');
#ifdef _WIN32
    const char *backslash = strrchr(path, '\\');
    if (!slash || (backslash && backslash > slash)) slash = backslash;
#endif
    return slash;
}

/* Compare two directory prefixes.
 *
 * On Windows both separators are legal within one path, and the two operands
 * reach the caller from different sources -- r->part is built from the
 * operator's --relations argument, ck->cand_part is read back out of the
 * checkpoint -- so "C:\\work\\out" and "C:/work/out" are one directory spelled
 * two ways. last_path_separator already accepts either; _strnicmp over the
 * raw bytes did not, which turned a difference in spelling into a refused
 * resume.
 *
 * This stays a textual test: it does not resolve "..", symlinks, junctions or
 * 8.3 short names, so it can still call two spellings of one directory
 * different. That is the safe direction -- it withholds a delete rather than
 * authorising one -- and the identity check in the caller, not this function,
 * is what actually establishes the file. */
static int path_prefix_equal(const char *a, const char *b, size_t n)
{
#ifdef _WIN32
    for (size_t i = 0; i < n; i++) {
        unsigned char ca = (unsigned char)a[i], cb = (unsigned char)b[i];
        if (ca == '/') ca = '\\';
        if (cb == '/') cb = '\\';
        if (tolower(ca) != tolower(cb)) return 0;
    }
    return 1;
#else
    return !memcmp(a, b, n);
#endif
}

static int same_parent_directory(const char *a, const char *b)
{
    const char *as = last_path_separator(a), *bs = last_path_separator(b);
    const size_t an = as ? (size_t)(as - a) : 0;
    const size_t bn = bs ? (size_t)(bs - b) : 0;
    if (!!as != !!bs || an != bn) return 0;
    return an == 0 || path_prefix_equal(a, b, an);
}

/* A checkpoint pathname is data, not authority to unlink.  It is usable only
 * when it remains in the relations directory and still names the exact open
 * candidate staging inode recorded at the checkpoint.  Missing is safe: there
 * is then no candidate artifact left to discard. */
static int resume_recovery_add_checkpoint_candidate(resume_recovery_t *r,
                                                     const ckpt_t *ck)
{
    bench_stat_t st;
    bench_file_id_t id;
    const bench_file_id_t want = { (uint64_t)ck->cand_dev,
                                   (uint64_t)ck->cand_ino };
    const size_t len = strlen(ck->cand_part);
    if (!len) return 0;             /* legacy checkpoint: identity unavailable */
    if (!same_parent_directory(r->part, ck->cand_part) ||
        !strcmp(r->part, ck->cand_part) || len < 5 ||
        strcmp(ck->cand_part + len - 5, ".part")) {
        fprintf(stderr,
                "BOINC: checkpoint candidate staging path is not a safe peer"
                " of %s\n", r->part);
        return -1;
    }
    if (bench_stat_path(ck->cand_part, &st)) {
        if (errno == ENOENT) return 1;
        perror(ck->cand_part);
        return -1;
    }
    /* Identity comes from bench_file_id_path, not from the stat above. The
     * stat still answers "regular file, and at least as long as the bytes the
     * checkpoint claims"; it cannot answer "the same file", because st_ino is
     * always 0 on Windows and every peer .part on the drive would compare
     * equal. bench_file_id_equal is false for an unavailable (0/0) identity on
     * either side, so "cannot tell" refuses rather than deletes. */
    if (bench_file_id_path(ck->cand_part, &id)) {
        if (errno == ENOENT) return 1;
        perror(ck->cand_part);
        return -1;
    }
    if (!bench_is_regular_mode((unsigned short)st.st_mode) ||
        !bench_file_id_equal(&id, &want) ||
        (unsigned long long)st.st_size < ck->cand_bytes) {
        fprintf(stderr,
                "BOINC: checkpoint candidate staging identity no longer"
                " matches %s; refusing to delete it\n", ck->cand_part);
        return -1;
    }
    memcpy(r->candidates_part, ck->cand_part, len + 1);
    return 1;
}

/* BOINC volunteers cannot reasonably repair a task's private output files.
 * If a relaunch finds staging state that cannot be resumed safely, recomputing
 * that workunit is preferable to a permanent compute error -- but only a
 * bounded number of times. A deterministic defect must eventually surface to
 * the project rather than consuming the host forever. The counter survives
 * recovery and is cleared only when the whole band commits successfully.
 *
 * Returns 0 outside BOINC, 1 after discarding, and -1 on an error or when the
 * retry limit has been reached. */
static int boinc_discard_bad_resume(const resume_recovery_t *r,
                                    const char *why)
{
    const char *paths[4] = {
        r->part, r->ckpt, r->ckpt_tmp, r->candidates_part
    };
    unsigned attempts = 0, durable = 0, inflight = 0;
    int durable_status, inflight_status;

    if (!bench_boinc_is_managed()) return 0;
    if (r->checkpoint_candidate_required &&
        !r->checkpoint_candidate_known) {
        fprintf(stderr,
                "BOINC: this checkpoint refers to candidate output but does"
                " not carry a verified staging-file identity; refusing to"
                " discard only part of the saved output\n");
        return -1;
    }
    durable_status = recovery_count_read_one(r->retry, &durable);
    if (durable_status == -2) {
        fprintf(stderr, "BOINC: malformed resume-recovery counter %s\n",
                r->retry);
        return -1;
    }
    if (durable_status < 0) return -1;

    inflight_status = recovery_count_read_one(r->retry_tmp, &inflight);
    if (inflight_status == -1) return -1;
    attempts = durable;
    if (inflight_status == -2) {
        /* retry_tmp is scratch written before its atomic rename. Empty or
         * partial content means that one recovery attempt was in flight when
         * the process died; count it, then overwrite the scratch on the next
         * recovery instead of making that torn file a permanent failure. */
        attempts = durable == UINT_MAX ? durable : durable + 1;
        fprintf(stderr,
                "BOINC: torn resume-recovery temporary %s; counting it as "
                "attempt %u\n", r->retry_tmp, attempts);
    } else if (inflight_status > 0 && inflight > attempts) {
        attempts = inflight;
    }
    if (attempts >= BOINC_RESUME_RETRIES) {
        fprintf(stderr,
                "BOINC: saved progress still cannot be resumed (%s); refusing "
                "after %u automatic recovery attempts\n",
                why, attempts);
        return -1;
    }
    if (recovery_count_write(r, attempts + 1)) return -1;

    fprintf(stderr,
            "BOINC: saved progress cannot be resumed (%s); automatic recovery "
            "%u/%u: discarding staging files and restarting this workunit\n",
            why, attempts + 1, BOINC_RESUME_RETRIES);
    for (unsigned i = 0; i < 4; i++) {
        if (!paths[i][0]) continue;
        if (remove(paths[i]) && errno != ENOENT) {
            perror(paths[i]);
            return -1;
        }
    }
    return 1;
}

static int bench_main_impl(int argc, char **argv)
{
    /* Identity of the card this process actually selected, captured where the
     * device is queried and read much later by the run-log header. Captured
     * rather than re-queried: between those two points lie the BOINC
     * assignment, --device and the ordinal bounds check, and a second
     * cudaGetDevice() would record whatever the current device had become
     * instead of what the run was configured with. */
    char dev_name[256] = "";
    char dev_pci[32] = "";          /* NVML's domain:bus:device.function */
    int  dev_ordinal = -1, dev_count = 0;
    const char *fbpath = "../oracle/input.job.afb.0";
    int fbpath_set = 0;
    const char *polypath = "../oracle/c183.poly";
    bench_cfg_t cfg;
    uint64_t q = 120000011ull;          /* prime, mid-range of [50M,190M] */
    uint32_t bkthresh = 0, fbbound = 0;
    int fbbound_set = 0, scale_set = 0;
    uint64_t rho = 0;   /* rho_set distinguishes an explicit zero from omission */
    uint32_t rlim = 67100000;   /* side-0 factor base bound (input.job rlim) */
    uint32_t alim = 134200000;  /* side-1 factor base bound (input.job alim) */
    fb_t fb, fbs; qlat_t L; poly_t POLY;

    /* Defaults are the CONFIGURATION OF RECORD -- single-level atomic fill,
     * 4 B records, 2^14 regions -- not the design the doc started from. Two-
     * level lost by 2.7x (RESULTS.md finding 1) and 2^15 regions lost to 2^14
     * (finding 8); leaving those as defaults meant the commands in RESULTS
     * reproduced a path nobody would ship. */
    cfg.logI = 15; cfg.J = 16384; cfg.slab_j = 0; cfg.log_region = 14;
    cfg.record_bytes = 4; cfg.fill_mode = FILL_ATOMIC; cfg.fill_streams = 0;
    cfg.qspan = 0;
    cfg.threads = 256; cfg.blocks = 0; cfg.fill_blocks = 0; cfg.fill_threads = 0;

    cfg.reps = 3; cfg.verify = 0;
    cfg.stage = STAGE_BOTH; cfg.cell_bits = 16; cfg.norm_mode = NORM_HORNER;
    cfg.apply_atomic = 1; cfg.apply_threads = 0; cfg.allowance = 3.5 * 32.0;
    cfg.td_record_scalar = 0;
    cfg.small_sieve = 1; cfg.side = 1;
    cfg.scale = 1.0; cfg.dump = NULL; cfg.cadofb = NULL;
    cfg.probe_i = 0; cfg.probe_j = 0xFFFFFFFFu;
    cfg.survbits = NULL; cfg.not_both_even = 0;
    cfg.other_bits = NULL; cfg.emit = NULL;
    cfg.td = 0; cfg.cofgate = NULL; cfg.emit_cof = NULL;
    cfg.lim = 0; cfg.lpb = 0; cfg.mfb = 0;   /* 0 == take the side's default */
    cfg.ab_resieve = 0; cfg.resieve_sweep = 0;
    /* Pipeline defaults are the WORKING configuration established 2026-08-04:
     * survivor bound 128 on side 1 and 132 on side 0 (128 on both loses one of
     * las's 37 relations at the parity q), and the full algebraic factor base
     * to alim, since truncating at the special-q costs relations outright. */
    cfg.pipeline = 0; cfg.sq_side = 1;
    cfg.scale0 = 1.925; cfg.allowance0 = 68.1;
    cfg.lim0 = 0; cfg.lpb0 = 31; cfg.mfb0 = 60;
    cfg.cof_limbs0 = 0; cfg.cof_limbs = 0;   /* 0 = derive from mfb */
    cfg.relations = NULL; cfg.candidates = NULL;
    cfg.qlist = NULL; cfg.nq_max = 0; cfg.verbose_q = 0; cfg.td_verify = 1;
    cfg.qmin = 0; cfg.qmax = 0; cfg.target_rels = 0;
    /* 4 rounds, not 2. ECM is the default method on a 3LP side now and its
     * escalation is CURVES-per-round, so 2 rounds is 24 curves and reaches
     * only 4,050 of AS276's 4,089 relations; 4 rounds is 48 and reaches all.
     *
     * IT IS NOT FREE FOR A RHO SIDE, and an earlier version of this comment
     * wrongly said it was. rho escalates by DOUBLING (`budget << r`), so a
     * lane that stays stuck costs 65536+131072 = 196,608 iterations at 2
     * rounds and 983,040 at 4 -- 5x. Only already-resolved records fall
     * through for free. The net is still positive on every job measured
     * (c183 17.21 -> 14.30 ms/q, C194 15.48 -> 13.95) because the ECM side
     * saves more than the rho side loses, but the two escalations are
     * genuinely different knobs sharing one number, and raising --cof-rounds
     * to tune ECM multiplies a rho side's budget exponentially. If that ever
     * bites, the fix is a separate curve-escalation count, not a bigger
     * shared one. */
    cfg.cofactor = 0; cfg.cof_rounds = 4; cfg.cof_budget = 65536;
    cfg.cof_ecm = COF_METHOD_AUTO;  /* per side, by LP count; --cof-ecm/--cof-rho force */
    cfg.ecm_b1 = 0; cfg.ecm_b2 = 0; cfg.ecm_curves = 0;   /* 0 = derive */
    cfg.cof_meth0 = cfg.cof_meth1 = COF_METHOD_RHO;
    cfg.ecm_b1_set = cfg.ecm_b2_set = cfg.ecm_curves_set = 0;
    cfg.fb_maxbits = 0;
    cfg.resume = 0; cfg.restart = 0; cfg.stopfile = NULL;
    /* 300 s is ~864 records over a three-day band, which is why runlog.h needs
     * no rotation. STATUS.md item 12b. */
    cfg.logpath = NULL; cfg.log_every_s = 300.0;
    cfg.resume_q = 0; cfg.resume_rho = 0;
    cfg.resume_rel_bytes = 0; cfg.resume_cand_bytes = 0;
    cfg.resume_nrel = 0; cfg.resume_nq = 0;
    int maxbits = 0, maxbits_set = 0;
    int allowance_set = 0, allowance0_set = 0, scale0_set = 0;
    const char *cofac_in = NULL;
    const char *check_rel = NULL;
    int blocking_sync = 0;
    int cuda_device = -1;       /* -1 = ask BOINC, else CUDA's default */
    /* Which values the COMMAND LINE supplied. Precedence is
     *      explicit flag  >  job file  >  derived  >  refuse
     * and these are what distinguishes the first level from the rest. A
     * default that merely looks like the job file's value is not the same
     * thing: the whole point is that a run says where each number came from. */
    int rlim_set = 0, alim_set = 0, lpb_set = 0, mfb_set = 0;
    int lpb0_set = 0, mfb0_set = 0, J_set = 0;
    double lambda0 = 0.0, lambda1 = 0.0;   /* 0 = CADO's automatic 0.3+mfb/lpb */
    int lambda0_set = 0, lambda1_set = 0;  /* asked for CADO's rule at all?    */
    int cof_rounds = 6;
    uint32_t cof_budget = 4096;
    int qrange_set = 0, rho_set = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fb") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--fb", argv[++i], &fbpath)) return 1;
            fbpath_set = 1;
        }
        else if (!strcmp(argv[i], "--logI") && i + 1 < argc) cfg.logI = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--J") && i + 1 < argc) { cfg.J = (uint32_t)strtoul(argv[++i], 0, 10); J_set = 1; }
        else if (!strcmp(argv[i], "--slab-j") && i + 1 < argc) {
            if (parse_u32_range_arg("--slab-j", argv[++i], 0u, UINT32_MAX,
                                    &cfg.slab_j)) return 1;
        }
        else if (!strcmp(argv[i], "--region") && i + 1 < argc) cfg.log_region = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bkthresh") && i + 1 < argc) bkthresh = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--fbbound") && i + 1 < argc) { fbbound = (uint32_t)strtoul(argv[++i], 0, 10); fbbound_set = 1; }
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) {
            if (parse_u64_arg("--q", argv[++i], &q)) return 1;
        }
        else if (!strcmp(argv[i], "--rho") && i + 1 < argc) {
            if (parse_u64_arg("--rho", argv[++i], &rho)) return 1;
            rho_set = 1;
        }
        else if (!strcmp(argv[i], "--record-bytes") && i + 1 < argc) cfg.record_bytes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "atomic")) cfg.fill_mode = FILL_ATOMIC;
            else if (!strcmp(m, "twolevel")) cfg.fill_mode = FILL_TWOLEVEL;
            else {
                fprintf(stderr, "--mode: want atomic or twolevel, got %s\n", m);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            if (parse_int_range_arg("--device", argv[++i], 0, INT_MAX,
                                    &cuda_device)) return 1;
        }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cfg.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) {
            if (parse_int_range_arg("--blocks", argv[++i], 0,
                                    BENCH_BLOCKS_MAX, &cfg.blocks)) return 1;
        }
        /* Strict integer parsing for all three grid controls -- not a style
         * preference. They treat 0 as "auto", and atoi maps any malformed
         * argument to 0, so
         * `--fill-threads 64x` would run at the default, print no [--flag] tag,
         * and be indistinguishable from an unswept run. A sweep over a typo'd
         * list then reports N identical timings, which reads as flatness --
         * precisely the shape of the conclusion these flags exist to test. */
        else if (!strcmp(argv[i], "--fill-blocks") && i + 1 < argc) {
            if (parse_int_range_arg("--fill-blocks", argv[++i], 0,
                                    BENCH_BLOCKS_MAX,
                                    &cfg.fill_blocks)) return 1;
        }
        else if (!strcmp(argv[i], "--fill-threads") && i + 1 < argc) {
            if (parse_int_range_arg("--fill-threads", argv[++i], 0,
                                    1024, &cfg.fill_threads)) return 1;
        }
        else if (!strcmp(argv[i], "--qspan")) { cfg.qspan = 1; }
        else if (!strcmp(argv[i], "--fill-streams") && i + 1 < argc) {
            if (parse_int_range_arg("--fill-streams", argv[++i], 0,
                                    FILL_STREAMS_MAX, &cfg.fill_streams)) return 1;
        }
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) cfg.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify")) cfg.verify = 1;
        /* The CPU half of --verify, and then exit. --verify itself falls
         * through to run_bench() on purpose -- that is the apply-parity gate
         * finding 32 added -- but the walk/transform gates are pure host
         * enumeration, and `make check` wires them in as a CPU gate that must
         * still work on a box with no card, a busy card, or no 60 MB oracle
         * factor base to load. */
        else if (!strcmp(argv[i], "--verify-only")) return verify_walk_cases();
        else if (!strcmp(argv[i], "--poly") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--poly", argv[++i], &polypath)) return 1;
        }
        else if (!strcmp(argv[i], "--stage") && i + 1 < argc) {
            const char *s = argv[++i];
            if (!strcmp(s, "fill")) cfg.stage = STAGE_FILL;
            else if (!strcmp(s, "apply")) cfg.stage = STAGE_APPLY;
            else if (!strcmp(s, "both")) cfg.stage = STAGE_BOTH;
            else {
                fprintf(stderr, "--stage: want fill, apply, or both, got %s\n", s);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--cells") && i + 1 < argc) cfg.cell_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--norm") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "const")) cfg.norm_mode = NORM_CONST;
            else if (!strcmp(m, "horner")) cfg.norm_mode = NORM_HORNER;
            else {
                fprintf(stderr, "--norm: want const or horner, got %s\n", m);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--apply-mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "atomic")) cfg.apply_atomic = 1;
            else if (!strcmp(m, "plain")) cfg.apply_atomic = 0;
            else {
                fprintf(stderr, "--apply-mode: want atomic or plain, got %s\n", m);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--apply-threads") && i + 1 < argc) {
            if (parse_int_range_arg("--apply-threads", argv[++i], 0,
                                    APPLY_THREADS_MAX, &cfg.apply_threads))
                return 1;
        }
        else if (!strcmp(argv[i], "--td-record-scalar")) {
            cfg.td_record_scalar = 1;
        }
        else if (!strcmp(argv[i], "--allowance") && i + 1 < argc) {
            if (parse_nonnegative_double_arg("--allowance", argv[++i],
                                             &cfg.allowance)) return 1;
            allowance_set = 1;
        }
        else if (!strcmp(argv[i], "--no-smallsieve")) cfg.small_sieve = 0;
        else if (!strcmp(argv[i], "--side") && i + 1 < argc) {
            if (parse_int_range_arg("--side", argv[++i], 0, 1, &cfg.side))
                return 1;
        }
        else if (!strcmp(argv[i], "--rlim") && i + 1 < argc) { rlim = (uint32_t)strtoul(argv[++i], 0, 10); rlim_set = 1; }
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) {
            if (parse_positive_double_arg("--scale", argv[++i], &cfg.scale))
                return 1;
            scale_set = 1;
        }
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--dump", argv[++i], &cfg.dump)) return 1;
        }
        else if ((!strcmp(argv[i], "--fb1") || !strcmp(argv[i], "--cadofb")) && i + 1 < argc) {
            const char *option = argv[i];
            if (bench_boinc_resolve_path(option, argv[++i], &cfg.cadofb))
                return 1;
        }
        else if (!strcmp(argv[i], "--survbits") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--survbits", argv[++i], &cfg.survbits)) return 1;
        }
        else if (!strcmp(argv[i], "--other-bits") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--other-bits", argv[++i], &cfg.other_bits)) return 1;
        }
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--emit", argv[++i], &cfg.emit)) return 1;
        }
        else if (!strcmp(argv[i], "--td")) cfg.td = 1;
        else if (!strcmp(argv[i], "--ab-resieve")) cfg.ab_resieve = 1;
        else if (!strcmp(argv[i], "--resieve-sweep")) cfg.resieve_sweep = 1;
        else if (!strcmp(argv[i], "--pipeline")) cfg.pipeline = 1;
        else if (!strcmp(argv[i], "--no-td-verify")) cfg.td_verify = 0;
        /* strtol, not atoi: atoi("rational") is 0, which is a LEGAL value here,
         * so a typo would silently configure the wrong side instead of being
         * rejected. Every other numeric flag in this parser avoids atoi for
         * the same reason. */
        else if (!strcmp(argv[i], "--sq-side") && i + 1 < argc) {
            char *end = NULL;
            long v = strtol(argv[++i], &end, 10);
            if (!end || *end || (v != 0 && v != 1)) {
                fprintf(stderr, "--sq-side %s: want 1 (algebraic) or 0"
                        " (rational)\n", argv[i]);
                return 1;
            }
            cfg.sq_side = (int)v;
        }
        else if (!strcmp(argv[i], "--cofac") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--cofac", argv[++i], &cofac_in)) return 1;
        }
        else if (!strcmp(argv[i], "--cofactor")) cfg.cofactor = 1;
        else if (!strcmp(argv[i], "--cof-rounds") && i + 1 < argc) { cof_rounds = atoi(argv[++i]); cfg.cof_rounds = cof_rounds; }
        else if (!strcmp(argv[i], "--cof-budget") && i + 1 < argc) { cof_budget = (uint32_t)strtoul(argv[++i], 0, 10); cfg.cof_budget = cof_budget; }
        else if (!strcmp(argv[i], "--cof-ecm")) cfg.cof_ecm = COF_METHOD_ECM;
        else if (!strcmp(argv[i], "--cof-rho")) cfg.cof_ecm = COF_METHOD_RHO;
        /* Deriving is now unconditional, so this is accepted and ignored
         * rather than rejected: it appears in RUNBOOK.md and in scripts, and
         * breaking those to make a point about a flag that now describes the
         * default helps nobody. The note fires so nobody keeps typing it. */
        else if (!strcmp(argv[i], "--auto-params"))
            fprintf(stderr, "note: --auto-params is the default now and is"
                            " ignored; drop it\n");
        else if (!strcmp(argv[i], "--blocking-sync")) blocking_sync = 1;
        else if (!strcmp(argv[i], "--lpb0") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 64) { fprintf(stderr, "--lpb0 %ld out of range 1..64\n", v); return 1; } cfg.lpb0 = (uint32_t)v; lpb0_set = 1; }
        else if (!strcmp(argv[i], "--mfb0") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 32 * CF_LMAX) { fprintf(stderr, "--mfb0 %ld out of range 1..%d\n", v, 32 * CF_LMAX); return 1; } cfg.mfb0 = (uint32_t)v; mfb0_set = 1; }
        /* Cofactor width in 32-bit limbs, per side. Only useful FORCED UPWARD
         * -- the default already picks the narrowest width mfb needs -- and
         * that is the point: it is what lets the same job be timed at 3 limbs
         * and at 4, which is the only way to price the width separately from
         * the job that motivated it. Narrower than mfb needs is refused in
         * resolve_and_check_cofactor_config, where the .job file's mfb is finally known. */
        else if (!strcmp(argv[i], "--cof-limbs0") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < CF_LMIN || v > CF_LMAX) { fprintf(stderr, "--cof-limbs0 %ld out of range %d..%d for this build\n", v, CF_LMIN, CF_LMAX); return 1; } cfg.cof_limbs0 = (int)v; }
        else if (!strcmp(argv[i], "--cof-limbs") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < CF_LMIN || v > CF_LMAX) { fprintf(stderr, "--cof-limbs %ld out of range %d..%d for this build\n", v, CF_LMIN, CF_LMAX); return 1; } cfg.cof_limbs = (int)v; }
        /* Range-checked like --lpb0/--mfb0 above, and for the same reason. An
         * unchecked --lambda1 0.01 gives allowance 0.32, bound 1, and a band
         * that runs for hours and emits nothing with no diagnostic.
         *
         * Exactly 0 stays legal: it is the documented sentinel for "use CADO's
         * automatic", and scripts pass it to mean the default. The window
         * refused is (0, 0.5), which no real lambda occupies -- CADO's
         * automatic lands near 2-3 -- and anything above 8. This is a guard
         * against a typo, not a tuning opinion, so the ends are loose. */
        /* `_set` is tracked separately from the value because 0 is a MEANINGFUL
         * value here -- the documented sentinel for "use CADO's automatic" --
         * so `var > 0.0` cannot stand in for "the user asked for CADO's rule".
         * Testing the value instead routed --lambda1 0 to the derived default,
         * silently giving mfb+1.5 where the script asked for 0.3+mfb/lpb. */
        #define LAMBDA_ARG(flag, var, seen)                                    \
            else if (!strcmp(argv[i], flag) && i + 1 < argc) {                 \
                if (parse_finite_double_arg(flag, argv[++i], &var)) return 1;   \
                seen = 1;                                                       \
                if (var < 0.0 || (var > 0.0 && var < 0.5) || var > 8.0) {      \
                    fprintf(stderr, "%s %g: want 0 (CADO's automatic) or"      \
                            " 0.5..8\n", flag, var);                           \
                    return 1;                                                  \
                }                                                              \
            }
        LAMBDA_ARG("--lambda0", lambda0, lambda0_set)
        LAMBDA_ARG("--lambda1", lambda1, lambda1_set)
        #undef LAMBDA_ARG
        else if (!strcmp(argv[i], "--check-relations") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--check-relations", argv[++i], &check_rel)) return 1;
        }
        else if (!strcmp(argv[i], "--ecm-b1") && i + 1 < argc) { cfg.ecm_b1 = (uint32_t)strtoul(argv[++i], 0, 10); cfg.ecm_b1_set = 1; }
        else if (!strcmp(argv[i], "--ecm-b2") && i + 1 < argc) { cfg.ecm_b2 = (uint32_t)strtoul(argv[++i], 0, 10); cfg.ecm_b2_set = 1; }
        else if (!strcmp(argv[i], "--ecm-curves") && i + 1 < argc) { cfg.ecm_curves = (uint32_t)strtoul(argv[++i], 0, 10); cfg.ecm_curves_set = 1; }
        else if (!strcmp(argv[i], "--qlist") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--qlist", argv[++i], &cfg.qlist)) return 1;
        }
        else if (!strcmp(argv[i], "--qrange") && i + 1 < argc) {
            const char *a = argv[++i];
            uint64_t lo, hi;
            if (bench_parse_qrange(a, &lo, &hi)) {
                fprintf(stderr, "bench: --qrange wants MIN:MAX, or MIN: to"
                        " generate upward until --target-rels/--nq\n"); return 2;
            }
            cfg.qmin = lo; cfg.qmax = hi;   /* 0 = stream until target/nq */
            qrange_set = 1;
        }
        else if (!strcmp(argv[i], "--nq") && i + 1 < argc) {
            if (parse_u32_range_arg("--nq", argv[++i], 1u, UINT32_MAX,
                                    &cfg.nq_max))
                return 1;
        }
        else if (!strcmp(argv[i], "--target-rels") && i + 1 < argc) {
            if (parse_u64_arg("--target-rels", argv[++i], &cfg.target_rels))
                return 1;
            if (!cfg.target_rels) {
                fprintf(stderr, "--target-rels: must be greater than zero\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--verbose-q")) cfg.verbose_q = 1;
        else if (!strcmp(argv[i], "--alim") && i + 1 < argc) { alim = (uint32_t)strtoul(argv[++i], 0, 10); alim_set = 1; }
        else if (!strcmp(argv[i], "--scale0") && i + 1 < argc) {
            if (parse_positive_double_arg("--scale0", argv[++i], &cfg.scale0))
                return 1;
            scale0_set = 1;
        }
        else if (!strcmp(argv[i], "--allowance0") && i + 1 < argc) {
            if (parse_nonnegative_double_arg("--allowance0", argv[++i],
                                             &cfg.allowance0)) return 1;
            allowance0_set = 1;
        }
        else if (!strcmp(argv[i], "--relations") && i + 1 < argc) {
            /* A relation file carries no build stamp and is byte-for-byte the
             * same shape as production output, so a pricing binary's relations
             * become indistinguishable from real ones the moment they leave
             * this process. The run log and the stderr banner are both
             * transient; refusing at the artifact is the only durable answer.
             * Pricing builds exist to time the norm, not to emit relations --
             * see the #warning on NORM_FAST_LOG2. */
            if (*runlog_build_defs()) {
                fprintf(stderr,
                    "--relations refused: this binary was built with %s, which"
                    " alters the norm.\n"
                    "  Its relations are not production output and nothing in"
                    " the file would say so.\n"
                    "  Rebuild without DEFS to emit relations.\n",
                    runlog_build_defs());
                return 1;
            }
            if (bench_boinc_resolve_path("--relations", argv[++i], &cfg.relations)) return 1;
        }
        else if (!strcmp(argv[i], "--candidates") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--candidates", argv[++i], &cfg.candidates)) return 1;
        }
        else if (!strcmp(argv[i], "--restart")) cfg.restart = 1;
        /* Deliberately NOT run through bench_boinc_resolve_path: that maps the
         * logical names of BOINC's own input/output files, and the stop file is
         * neither. It is an operator-side probe for unattended queues, named in
         * whatever filesystem the queue runs in. Under BOINC it is moot anyway,
         * since direct_process_action leaves suspend and quit to the runtime. */
        else if (!strcmp(argv[i], "--stop-file") && i + 1 < argc) cfg.stopfile = argv[++i];
        /* Resolved like every other named output: under BOINC the log is a
         * workunit output file with a logical name, and writing it to the
         * literal string would put it outside the slot directory. */
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--log", argv[++i], &cfg.logpath)) return 1;
        }
        else if (!strcmp(argv[i], "--log-every") && i + 1 < argc) {
            char *end = NULL;
            const double v = strtod(argv[++i], &end);
            /* A zero or negative period would write a record per special-q on
             * a job whose q take milliseconds -- millions of lines over a
             * multi-day band, which is the one failure mode "no rotation"
             * cannot absorb. */
            if (!end || *end || !(v >= 1.0) || v > 86400.0) {
                fprintf(stderr, "--log-every %s: want seconds in [1, 86400]\n",
                        argv[i]);
                return 1;
            }
            cfg.log_every_s = v;
        }
        else if (!strcmp(argv[i], "--lpb") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 64) { fprintf(stderr, "--lpb %ld out of range 1..64\n", v); return 1; } cfg.lpb = (uint32_t)v; lpb_set = 1; }
        else if (!strcmp(argv[i], "--mfb") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 32 * CF_LMAX) { fprintf(stderr, "--mfb %ld out of range 1..%d\n", v, 32 * CF_LMAX); return 1; } cfg.mfb = (uint32_t)v; mfb_set = 1; }
        else if (!strcmp(argv[i], "--cofgate") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--cofgate", argv[++i], &cfg.cofgate)) return 1;
        }
        else if (!strcmp(argv[i], "--emit-cof") && i + 1 < argc) {
            if (bench_boinc_resolve_path("--emit-cof", argv[++i], &cfg.emit_cof)) return 1;
        }
        else if (!strcmp(argv[i], "--not-both-even")) cfg.not_both_even = 1;
        else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) {
            maxbits = atoi(argv[++i]); maxbits_set = 1;
        }
        else if (!strcmp(argv[i], "--probe") && i + 1 < argc) {
            int pi; unsigned pj;
            if (sscanf(argv[++i], "%d,%u", &pi, &pj) != 2) {
                fprintf(stderr, "--probe wants i,j\n"); return 1;
            }
            cfg.probe_i = pi; cfg.probe_j = pj;
        }
        else { usage(); return 1; }
    }
    if (cfg.cell_bits != 8 && cfg.cell_bits != 16) { usage(); return 1; }
    /* logI bounds FIRST: every default below shifts by it (bkthresh, the area
     * check, the probe range), and an out-of-range shift is undefined. */
    if (cfg.logI < 2 || cfg.logI > 20) { usage(); return 1; }
    /* Resolve J before any area/probe validation.  Keeping the old default
     * assignment after polynomial loading meant --logI 17 was validated with
     * logI 17 but the stale I15 J=16384, then silently changed to 65536 later. */
    if (!J_set) cfg.J = 1u << (cfg.logI - 1);
    if (cfg.cofgate && !cfg.td_verify) {
        fprintf(stderr, "--cofgate requires TD verification; remove"
                        " --no-td-verify\n");
        return 1;
    }
    if (!maxbits_set) maxbits = cfg.logI;
    cfg.fb_maxbits = maxbits;      /* the resume fingerprint reads it from cfg */
    if (maxbits < 1 || maxbits > 31) {
        fprintf(stderr, "--maxbits must be in [1,31]\n");
        return 1;
    }
    /* las's survivor bound is scale*lambda*lpb per side; ours is the same
     * quantity in unscaled bits (16-bit cells need no scale). */
    if (!allowance_set && cfg.side == 0) cfg.allowance = 2.35 * 31.0;
    if (!bkthresh) bkthresh = 1u << cfg.logI;
    if (!fbbound && cfg.side == 1 && !cfg.pipeline)
        fbbound = (q > 0xFFFFFFFFull) ? 0xFFFFFFFFu : (uint32_t)q;

    /* Shift in the destination width, not the source width. These are all
     * provably in range -- logI is bounded to [2,20] and log_region to [1,30]
     * above, so a 32-bit shift could not overflow -- but MSVC's C4334 fires on
     * the pattern rather than on the proof, and a Windows build that always
     * emits three warnings is a Windows build whose warnings get ignored.
     * `(uint64_t)1 << n` says what was meant and costs nothing. */
    /* Bound log_region before ANY 1u << log_region: the shift is undefined for
     * >= 32 and UBSan flags --region 32 on the old ordering. */
    if (!slab_region_ok(cfg.log_region)) {
        fprintf(stderr, "--region must be in [1,30] (got %d)\n", cfg.log_region);
        return 1;
    }
    if (cfg.pipeline) {
        const uint32_t effective_slab_j =
            cfg.slab_j && cfg.slab_j < cfg.J ? cfg.slab_j : cfg.J;
        const uint32_t area_jmax = slab_area_jmax(cfg.logI);
        if (!slab_rows_shape_ok(cfg.logI, cfg.J)) {
            fprintf(stderr,
                    "pipeline geometry logI=%d J=%u is not aligned to the"
                    " %u-position TD rank group\n",
                    cfg.logI, cfg.J, (unsigned)SLAB_TD_GROUP_POS);
            return 1;
        }
        if (cfg.slab_j &&
            (!effective_slab_j || effective_slab_j > area_jmax ||
             !slab_rows_shape_ok(cfg.logI, effective_slab_j))) {
            fprintf(stderr,
                    "--slab-j %u is not a valid local slab shape for logI=%d:"
                    " effective rows=%u, max rows by 2^31 area=%u, TD rank"
                    " groups are %u positions\n",
                    cfg.slab_j, cfg.logI, effective_slab_j, area_jmax,
                    (unsigned)SLAB_TD_GROUP_POS);
            return 1;
        }
    }
    /* An out-of-range probe silently ALIASES another cell -- --probe 16384,0
     * lands on the real (-16384,1) -- so it would certify a coordinate nobody
     * asked about. */
    if (cfg.probe_j != 0xFFFFFFFFu) {
        const int32_t half = 1 << (cfg.logI - 1);
        if (cfg.probe_i < -half || cfg.probe_i >= half || cfg.probe_j >= cfg.J) {
            fprintf(stderr, "--probe i,j out of range: i must be in [%d,%d) and"
                    " j in [0,%u)\n", -half, half, cfg.J);
            return 1;
        }
    }
    if (!cfg.pipeline && ((uint64_t)1 << cfg.logI) * cfg.J > 0x80000000ull) {
        fprintf(stderr, "I*J must fit in 31 bits outside --pipeline; the pipeline"
                " uses j-slabs for larger rectangles\n");
        return 1;
    }
    /* Two limits that used to be silent. A 2 B or 4 B record carries the
     * in-region offset in 16 bits, so a region above 2^16 wraps and every
     * record past the wrap lands on the wrong cell; and the fused small sieve
     * derives j from a single shift, which assumes a region lies inside one
     * j-row. Both produced plausible-looking output rather than an error. */
    if (cfg.record_bytes != 2 && cfg.record_bytes != 4 && cfg.record_bytes != 8) {
        fprintf(stderr, "--record-bytes must be 2, 4 or 8 (got %d): any other"
                " value falls through to the 8-byte kernel while allocating the"
                " requested size, which writes out of bounds\n", cfg.record_bytes);
        return 1;
    }
    if (cfg.fill_mode == FILL_TWOLEVEL && cfg.record_bytes != 4) {
        fprintf(stderr, "--mode twolevel only has a 4-byte level-2 kernel;"
                " --record-bytes %d would launch the wrong specialisation\n",
                cfg.record_bytes);
        return 1;
    }
    if (cfg.J == 0 || cfg.reps < 1 || cfg.threads < 32 || cfg.threads > 1024) {
        fprintf(stderr, "--J must be > 0, --reps >= 1, --threads in [32,1024]\n");
        return 1;
    }
    /* Same rule --apply-threads has enforced all along, and for the same class
     * of reason -- it was simply never applied to the block width every OTHER
     * kernel launches with. k_intersect_compact runs a warp-wide inclusive scan
     * under a hardcoded 0xffffffff mask and broadcasts the atomic base from
     * lane 31 (bench_kernels.cu:899-910). A partial final warp has no lane 31,
     * so the base every thread in it reads is undefined.
     *
     * --threads 33 was accepted and ran to completion, reporting "intersect
     * counted 262538 survivors, rank scan 270888". The independent second count
     * turned that into a band FAILED rather than bad relations, which is the
     * gate working -- but it costs a whole band to say what one modulo says
     * here. */
    if (cfg.threads & 31) {
        fprintf(stderr, "--threads must be a multiple of 32 (got %d): the"
                " intersection kernel scans under a full warp mask and reads"
                " its atomic base from lane 31, which a partial warp does not"
                " have\n", cfg.threads);
        return 1;
    }
    /* Fill has no warp-collective code, so a partial warp here would not be
     * wrong -- just wasteful, since the tail lanes are launched and idle. The
     * range matters more: 0 means "auto" so it can never be passed through, and
     * above 1024 every launch fails at runtime with a message that does not
     * mention this flag. */
    if (cfg.fill_threads != 0 &&
        (cfg.fill_threads < 32 || cfg.fill_threads > 1024
         || (cfg.fill_threads & 31))) {
        fprintf(stderr, "--fill-threads must be 0 (auto) or a multiple of 32 in"
                " [32,1024], got %d\n", cfg.fill_threads);
        return 1;
    }
    /* The grid-width query used to sit HERE. It now runs after the
     * --check-relations return further down, so that gate works on a box with
     * no GPU -- see the comment there. */
    /* The small sieve's warp tier strides by nwarps = threads >> 5. Below 32
     * that is ZERO -- an infinite loop on the device. A non-multiple of 32
     * leaves a partial warp whose lanes re-run warp-tier entries, double-adding
     * their logs. Neither is diagnosable from the output. */
    if (cfg.apply_threads != 0 &&
        (cfg.apply_threads < 32 || cfg.apply_threads > APPLY_THREADS_MAX
         || (cfg.apply_threads & 31))) {
        fprintf(stderr, "--apply-threads must be 0 (auto) or a multiple of 32 in"
                " [32,%d] (got %d): the small sieve strides by threads/32, so"
                " under 32 hangs and a partial warp double-counts, and"
                " k_apply's __launch_bounds__ makes %d a hard launch ceiling\n",
                APPLY_THREADS_MAX, cfg.apply_threads, APPLY_THREADS_MAX);
        return 1;
    }
    if (((uint64_t)1 << cfg.logI) * cfg.J % ((uint64_t)1 << cfg.log_region)) {
        fprintf(stderr, "I*J must divide evenly into 2^%d regions\n", cfg.log_region);
        return 1;
    }
    if (cfg.log_region > 16 && cfg.record_bytes < 8) {
        fprintf(stderr, "--region %d needs --record-bytes 8: a %d B record has"
                " only a 16-bit offset field\n", cfg.log_region, cfg.record_bytes);
        return 1;
    }
    if (cfg.log_region > cfg.logI && (cfg.small_sieve || cfg.pipeline)) {
        fprintf(stderr, "--region %d > --logI %d: %s assumes a region lies"
                " within one j-row.\n  Use a region <= 2^%d%s.\n",
                cfg.log_region, cfg.logI,
                cfg.pipeline ? "the slabbed pipeline" : "the fused small sieve",
                cfg.logI, cfg.pipeline ? "" : ", or --no-smallsieve");
        return 1;
    }
    /* Scale is free with 16-bit cells (see k_apply), but not unbounded: the
     * norm must fit under CINIT and each ideal's log must fit in the uint8
     * that fb_t carries. Refuse rather than saturate. */
    {
        const double cinit = (cfg.cell_bits == 16) ? 4096.0 : 255.0;
        /* c183's norm sizes, and only a sanity bound on a user-supplied
         * --scale in the single-side harness modes. The pipeline rechecks both
         * sides against their REAL derived maxnorm and lim after the scale is
         * derived; this one runs too early to see either. */
        const double maxnorm = (cfg.side == 1) ? 196.61 : 131.86;
        const double maxlogp = 27.0;           /* log2(alim) */
        const double scaled_norm = cfg.scale * maxnorm;
        const double scaled_logp = cfg.scale * maxlogp;
        uint32_t bound_unused;
        if (!isfinite(cfg.scale) || cfg.scale <= 0.0) {
            fprintf(stderr, "--scale %.17g must be finite and positive\n",
                    cfg.scale);
            return 1;
        }
        if (!isfinite(scaled_norm) || scaled_norm > cinit) {
            fprintf(stderr, "--scale %.3f x log2(maxnorm) %.1f exceeds CINIT %.0f\n",
                    cfg.scale, maxnorm, cinit);
            return 1;
        }
        if (!isfinite(scaled_logp) || scaled_logp > 255.0) {
            fprintf(stderr, "--scale %.3f x log2(p) %.0f exceeds the 8-bit"
                    " per-ideal log\n", cfg.scale, maxlogp);
            return 1;
        }
        /* In the one-side harness these are already the final parameters. The
         * pipeline derives both sides later from the first q, so checking its
         * placeholder defaults here could reject a valid final combination. */
        if (!cfg.pipeline &&
            sieve_bound_checked(cfg.scale, cfg.allowance, (uint32_t)cinit,
                                &bound_unused,
                                "single-side survivor parameters"))
            return 1;
    }


    printf("=== cuda-sieve bucket-fill benchmark ===\n");

    if (poly_load(polypath, &POLY) != 0) return 1;
    printf("polynomial %s: algebraic degree %d, skew %.4g\n",
           polypath, POLY.deg, POLY.skew);

    /* ---- sieve parameters: explicit flag > job file > derived > refuse ----
     *
     * A GGNFS .job file already carries rlim, alim, lpbr/lpba, mfbr/mfba and
     * the two lambdas. They used to be retyped onto the command line -- eight
     * flags transcribing a file this process has open -- which is both tedious
     * and a place for the two to silently disagree.
     *
     * Whatever is taken from the file is PRINTED, because a parameter that
     * appears from nowhere is worse than one that has to be typed. */
    {
        int used = 0;
        #define JOB_TAKE(cond, dst, src, name)                                 \
            do { if ((cond) && (src)) {                                        \
                     (dst) = (src);                                            \
                     printf("%s%s %u", used++ ? ", " : "  job file: ",         \
                            name, (unsigned)(src));                            \
                 } } while (0)
        JOB_TAKE(!rlim_set, rlim,     POLY.rlim, "rlim");
        JOB_TAKE(!alim_set, alim,     POLY.alim, "alim");
        JOB_TAKE(!lpb0_set, cfg.lpb0, POLY.lpbr, "lpbr");
        JOB_TAKE(!lpb_set,  cfg.lpb,  POLY.lpba, "lpba");
        JOB_TAKE(!mfb0_set, cfg.mfb0, POLY.mfbr, "mfbr");
        JOB_TAKE(!mfb_set,  cfg.mfb,  POLY.mfba, "mfba");
        #undef JOB_TAKE
        if (used) printf("\n");

        /* The .job file's lambdas are REPORTED, not applied.
         *
         * They are calibrated to GGNFS's survivor gate, and that calibration
         * does not transfer: on the SNFS job, same q range, gnfs-lasieve4I14e
         * loses 17.3% of its yield going from 91.8 to 87.5 bits where we lose
         * 0.07% going to 88.0. Importing the number inherits another tool's
         * tuning and, here, 13% of the trial-division input for one relation
         * in ten thousand. CADO's automatic is no better a source -- on this
         * job it is looser still (97.3 bits).
         *
         * So the default comes from sieve_allowance(), derived from our own
         * quantisation. The file's value is printed anyway, because it is
         * useful to see what the job's author intended and how far it sits
         * from what we chose. --allowance / --allowance0 override; so do
         * --lambda0 / --lambda1 if CADO's rule is wanted. */
        if (POLY.alambda > 0.0 && alim)
            printf("  job file: alambda %.3g -> %.2f bits (side 1), reported"
                   " only; the allowance is derived below\n",
                   POLY.alambda, job_allowance_bits(POLY.alambda, alim));
        if (POLY.rlambda > 0.0 && rlim)
            printf("  job file: rlambda %.3g -> %.2f bits (side 0), reported"
                   " only; the allowance is derived below\n",
                   POLY.rlambda, job_allowance_bits(POLY.rlambda, rlim));
    }

    /* HERE, not before poly_load: lpb/mfb/lim may all have come from the .job
     * file just read, and validating only the command line was validating the
     * half that the argument parser had already range-checked. */
    if (resolve_and_check_cofactor_config(&cfg, alim, rlim, cof_rounds,
                                          cof_budget)) return 1;

    /* --check-relations is pure host code -- it re-derives both norms from the
     * polynomial and divides. It cannot run before this point because lpb0/lpb
     * may have just come from the .job file, and it must run before the device
     * query below, because a fatal "cannot query the CUDA device" on a machine
     * with no GPU is exactly what stops an emitted relation file from being
     * verified on the box that has the file rather than the card. */
    if (check_rel)
        return check_relations(check_rel, &POLY, cfg.lpb0,
                              cfg.lpb ? cfg.lpb : 32) ? 1 : 0;

    /* The production pipeline validates q, primality and the exact root below.
     * The single-side harness intentionally permits a synthetic/composite q for
     * microbenchmarks, but q must still fit the transform representation and an
     * explicit rho must be canonical before qlat_build narrows it to int64_t. */
    if (!cfg.pipeline) {
        if (q < 2 || (q >> 32)) {
            fprintf(stderr, "bench: --q must be in [2, 2^32), got %llu\n",
                    (unsigned long long)q);
            return 1;
        }
        if (rho_set) rho %= q;
    }

    /* Which GPU. A BOINC client assigns one per task and reports it in
     * init_data.xml. Reading the command line alone left every task on a
     * multi-GPU host running on device 0 (reported by Greg Childers,
     * NFS@Home); see bench_boinc_gpu_device() for why "--device N" from the
     * client cannot be relied on.
     *
     * The ASSIGNMENT WINS over --device, which is the opposite of this file's
     * usual "explicit flag beats everything" rule, because here the flag is
     * not per-task: a --device in an app version's <cmdline> or a workunit
     * template is shared by every task on every host, so honouring it would
     * silently reinstate the all-tasks-on-one-card bug for the whole project.
     * The client is the only party that knows what else is running on the
     * host, and a volunteer excludes a card through cc_config <exclude_gpu>,
     * which the client already applies before computing this number. --device
     * therefore selects the card only when there is no assignment: standalone
     * runs, non-BOINC builds, and app versions the client treats as CPU-only.
     *
     * Whether an assignment arrived at all is the first thing to check when a
     * host reports every task on one card, so it goes to stderr, which BOINC
     * uploads with the result -- stdout is discarded with the slot. */
    {
        const int boinc_device = bench_boinc_gpu_device();
        if (boinc_device >= 0) {
            if (cuda_device >= 0 && cuda_device != boinc_device)
                fprintf(stderr,
                        "BOINC: ignoring --device %d; the client assigned this"
                        " task CUDA device %d\n", cuda_device, boinc_device);
            cuda_device = boinc_device;
            fprintf(stderr, "BOINC: client assigned CUDA device %d\n",
                    cuda_device);
        }
#ifdef HAVE_BOINC
        else if (cuda_device < 0) {
            /* Distinguishes "the client assigned device 0" from "the client
             * assigned nothing", which is what tells a project whether its app
             * version's plan class actually declares an NVIDIA coprocessor.
             * Without that declaration the client sets neither this field nor
             * --device, and every task on the host lands on the same card no
             * matter what the application does. */
            fprintf(stderr,
                    "BOINC: no usable GPU assignment in init_data.xml; using"
                    " CUDA's default device\n");
        }
#endif
    }

    /* CUDA 12 and later initialise a device's primary context inside
     * cudaSetDevice(), so a scheduling policy must be attached before that
     * call -- and, with the assignment above, to a device that is not
     * necessarily 0. cudaInitDevice() does exactly that for the selected
     * device without accidentally creating a context on device 0. Older
     * runtimes lack that API; there the legacy
     * cudaSetDeviceFlags() ordering is safe only for CUDA's default device, so
     * select first for a nonzero assignment and verify the effective flags. */
    int ndev = 0;               /* CUDA devices visible to THIS process */
    {
        const int selected_device = cuda_device >= 0 ? cuda_device : 0;
        cudaError_t err;

        /* The client and this process can disagree about how many GPUs exist
         * -- a card outside this build's gencode set, a driver/runtime
         * mismatch, CUDA_VISIBLE_DEVICES in the environment. That surfaces
         * downstream as a bare "invalid device ordinal", which does not say
         * whether the ordinal or the enumeration is the wrong one, on a host
         * whose only report is the uploaded stderr. Name both numbers here,
         * once. This does not create a primary context on any device, so it is
         * safe ahead of the flag/selection ordering below. */
        err = cudaGetDeviceCount(&ndev);
        if (err != cudaSuccess) {
            fprintf(stderr, "bench: cannot enumerate CUDA devices: %s\n",
                    cudaGetErrorString(err));
            return 1;
        }
        if (ndev < 1) {
            fprintf(stderr, "bench: this process sees no CUDA device\n");
            return 1;
        }
        if (selected_device >= ndev) {
            fprintf(stderr,
                    "bench: CUDA device %d requested, but this process sees"
                    " only %d device%s (valid ordinals 0..%d).%s\n",
                    selected_device, ndev, ndev == 1 ? "" : "s", ndev - 1,
                    getenv("CUDA_VISIBLE_DEVICES")
                        ? " CUDA_VISIBLE_DEVICES is set, which renumbers"
                          " devices from 0."
                        : "");
            return 1;
        }

        if (blocking_sync) {
#if CUDART_VERSION >= 12000
            err = cudaInitDevice(selected_device,
                                 cudaDeviceScheduleBlockingSync,
                                 cudaInitDeviceFlagsAreValid);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "bench: cannot initialise CUDA device %d with blocking"
                        " synchronization: %s\n",
                        selected_device, cudaGetErrorString(err));
                return 1;
            }
#else
            if (selected_device == 0) {
                err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
                if (err != cudaSuccess) {
                    fprintf(stderr,
                            "bench: cannot configure CUDA device 0 for blocking"
                            " synchronization: %s\n",
                            cudaGetErrorString(err));
                    return 1;
                }
            }
#endif
        }

        if (cuda_device >= 0 || blocking_sync) {
            err = cudaSetDevice(selected_device);
            if (err != cudaSuccess) {
                fprintf(stderr, "bench: cannot select CUDA device %d: %s\n",
                        selected_device, cudaGetErrorString(err));
                return 1;
            }
        }

#if CUDART_VERSION < 12000
        if (blocking_sync && selected_device != 0) {
            err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "bench: cannot configure CUDA device %d for blocking"
                        " synchronization: %s\n",
                        selected_device, cudaGetErrorString(err));
                return 1;
            }
        }
#endif
        if (blocking_sync) {
            unsigned int flags = 0;
            err = cudaGetDeviceFlags(&flags);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "bench: cannot verify CUDA scheduling flags for device"
                        " %d: %s\n",
                        selected_device, cudaGetErrorString(err));
                return 1;
            }
            if ((flags & cudaDeviceScheduleMask) !=
                cudaDeviceScheduleBlockingSync) {
                fprintf(stderr,
                        "bench: CUDA device %d did not retain blocking"
                        " synchronization flags (effective flags 0x%x)\n",
                        selected_device, flags);
                return 1;
            }
        }
    }

    /* Grid width is 6 resident blocks per SM, so it MUST come from the device.
     * It was hardcoded 48*6 = 288 -- 48 being this box's 5070. On an 82-SM
     * 3090 that is 3.5 blocks/SM, i.e. 58% of the occupancy every tuning
     * decision in RESULTS.md assumed, which reads as the bigger card being
     * slower. Resolve it once here; run_pipeline() and bench_kernels.cu both
     * treat a nonzero cfg.blocks as final, so this reaches every launch.
     *
     * Queried and printed UNCONDITIONALLY, on stdout with every other startup
     * line. Printing only in the default case hid the number from exactly the
     * --blocks A/B that wants it, stderr dropped it from any run redirected
     * with `> log`, and a failed query used to leave cfg.blocks at 0 so the
     * three `48 * 6` fallbacks silently reinstated the 5070's SM count with no
     * diagnostic. There is no useful run on a device we cannot query, so a
     * failure here is fatal rather than a default.
     *
     * The current device is the one selected by --device or by the BOINC
     * assignment, or CUDA's default when there is neither. The line below
     * prints the actual selection. */
    {
        cudaDeviceProp prop;
        int dev = 0;
        int auto_blocks, effective_fill_blocks;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
            fprintf(stderr, "bench: cannot query the CUDA device -- refusing to"
                    " fall back to a hardcoded grid width\n");
            return 1;
        }
        snprintf(dev_name, sizeof dev_name, "%s", prop.name);
        /* NVML's own spelling of a bus ID: eight-digit domain, then bus and
         * device in two hex digits each. Function is always 0 for a GPU. */
        snprintf(dev_pci, sizeof dev_pci, "%08x:%02x:%02x.0",
                 prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        dev_ordinal = dev;
        dev_count = ndev;
#ifdef HAVE_BOINC
        /* The one line that answers "did this task actually run on the card the
         * client gave it?". The grid: line below carries the same ordinal, but
         * on stdout, which the client discards with the slot directory. */
        fprintf(stderr, "BOINC: running on CUDA device %d of %d: %s\n",
                dev, ndev, prop.name);
#endif
        {
            const uint64_t ab = (uint64_t)prop.multiProcessorCount * 6u;
            if (!ab || ab > BENCH_BLOCKS_MAX ||
                ab > (uint64_t)prop.maxGridSize[0]) {
                fprintf(stderr, "bench: automatic grid %llu is outside the"
                        " supported/device range (max %d, policy max %d)\n",
                        (unsigned long long)ab, prop.maxGridSize[0],
                        BENCH_BLOCKS_MAX);
                return 1;
            }
            auto_blocks = (int)ab;
        }
        if (cfg.blocks && cfg.blocks > prop.maxGridSize[0]) {
            fprintf(stderr, "--blocks %d exceeds this device's grid-x limit %d\n",
                    cfg.blocks, prop.maxGridSize[0]);
            return 1;
        }
        effective_fill_blocks = cfg.fill_blocks ? cfg.fill_blocks
                                                : FILL_BLOCKS_DEFAULT;
        if (effective_fill_blocks > prop.maxGridSize[0]) {
            fprintf(stderr, "fill grid %d exceeds this device's grid-x limit %d\n",
                    effective_fill_blocks, prop.maxGridSize[0]);
            return 1;
        }
        /* L2 size rides on the card name, on BOTH branches. The fill geometry
         * is an absolute block count that is the same on every card measured,
         * which is itself the interesting thing -- L2 capacity does not explain
         * it (finding 51 already killed capacity, and the 1152 x 32 this
         * paragraph was written for fit cards with 48, 72 and 96 MB alike).
         * The count is now 4608 (finding 76, retuned on the c194 production
         * shape) and that value is measured on a 5070 ONLY, so the
         * same-on-every-card claim above stands for 1152 and is UNTESTED for
         * 4608 -- which is a further reason to print it. It is printed so a sweep log carries the
         * number instead of someone reconstructing it later from a card name.
         * Printing it only in the default branch would hide it from exactly the
         * --blocks A/B that wants it -- the defect the paragraph above this
         * block records having already fixed once. */
        if (cfg.blocks == 0) {
            cfg.blocks = auto_blocks;
            printf("grid: %d SMs x 6 = %d blocks (dev %d: %s, %d MB L2)\n",
                   prop.multiProcessorCount, cfg.blocks, dev, prop.name,
                   prop.l2CacheSize >> 20);
        } else {
            printf("grid: %d blocks on %d SMs (dev %d: %s, %d MB L2)  [--blocks;"
                   " auto would be %d]\n", cfg.blocks, prop.multiProcessorCount,
                   dev, prop.name, prop.l2CacheSize >> 20,
                   auto_blocks);
        }
        /* Reported, NOT resolved. cfg.fill_blocks/fill_threads stay 0 for
         * "auto" all the way to the launch sites, because k_fill_atomic and
         * k_fill_l1 have different defaults (the 1152 x 32 sweep was run on the
         * former only) and collapsing 0 here would erase the distinction they
         * need. Every ?: at a launch site is therefore live, not a dead
         * fallback -- and there is exactly one place per kernel that knows its
         * own default. */
        printf("grid: %d x %d for fill (absolute, not per SM)%s%s\n",
               cfg.fill_blocks  ? cfg.fill_blocks  : FILL_BLOCKS_DEFAULT,
               cfg.fill_threads ? cfg.fill_threads : FILL_THREADS_DEFAULT,
               cfg.fill_blocks  ? "  [--fill-blocks]"  : "",
               cfg.fill_threads ? "  [--fill-threads]" : "");
        /* A knob the banner does not name is a knob a log cannot prove was
         * live -- the --fill-blocks lesson (bench_kernels.cu, twolevel note).
         * --fill-streams reaches k_fill_atomic only, so say so HERE rather
         * than letting --mode twolevel swallow it silently. */
        if (cfg.fill_streams > 1)
            printf("fill concurrency: %d workspaces on %d streams"
                   "  [--fill-streams]%s\n", cfg.fill_streams, cfg.fill_streams,
                   cfg.fill_mode == FILL_ATOMIC
                       ? "" : "  ** IGNORED: --mode atomic only **");
    }

    /* ---- both sides in one process ---- */
    if (cofac_in) {
        if (!cfg.lpb)  cfg.lpb  = 32;
        if (!cfg.lim)  cfg.lim  = alim;
        if (!cfg.lim0) cfg.lim0 = rlim;
        return run_cofac(cofac_in, cfg.relations, cfg.lim0, cfg.lpb0,
                         cfg.lim, cfg.lpb, cof_rounds, cof_budget,
                         cfg.blocks ? cfg.blocks : 48 * 6, cfg.threads,
                         cfg.cof_meth0, cfg.cof_meth1, cfg.ecm_b1,
                         cfg.ecm_b2, cfg.ecm_curves,
                         cfg.cof_limbs0, cfg.cof_limbs) ? 1 : 0;
    }

    /* Only the pipeline reads these, so outside it they were silent no-ops --
     * the very thing the harness_only block below calls an error in the other
     * direction. A run quoted as relation-targeted or lambda-tuned when it was
     * neither is the same defect either way. */
    if (!cfg.pipeline) {
        static const char *pipeline_only[] = {
            "--target-rels", "--lambda0", "--lambda1", "--sq-side",
            "--restart", "--stop-file", "--log", "--log-every",
            "--slab-j", "--qspan", NULL
        };
        int nbad = 0;
        for (int i = 1; i < argc; i++)
            for (int k = 0; pipeline_only[k]; k++)
                if (!strcmp(argv[i], pipeline_only[k])) {
                    fprintf(stderr, "bench: %s applies to --pipeline only\n",
                            argv[i]);
                    nbad++;
                }
        if (nbad) { fprintf(stderr, "  add --pipeline, or drop them.\n"); return 2; }
    }

    if (cfg.pipeline) {
        fb_t fb1, fbs1, fb0, fbs0;
        qsel_t *ql = NULL;
        sqgen_t *qgen = NULL;
        uint32_t nq = 0, capq = 0;
        int prc;
        int fb1_generated = 0;
        /* Options that belong to the measurement harness only. The pipeline
         * runs ONE configuration -- the one every gate was closed against:
         * single-level atomic fill, 4 B records, 16-bit cells, the fp32/fp64
         * Horner norm, both stages, no repetitions, side 1 then side 0. Taking
         * a flag that changes none of that and running anyway is how a run gets
         * quoted as something it was not, so it is an error rather than a
         * silent no-op. */
        static const char *harness_only[] = {
            "--record-bytes", "--mode", "--cells", "--norm", "--apply-mode",
            "--stage", "--reps", "--verify", "--verify-only", "--side", "--dump", "--probe",
            "--survbits", "--other-bits", "--emit", "--emit-cof", "--td",
            "--ab-resieve", "--resieve-sweep", NULL
        };
        int nbad = 0;
        for (int i = 1; i < argc; i++)
            for (int k = 0; harness_only[k]; k++)
                if (!strcmp(argv[i], harness_only[k])) {
                    fprintf(stderr, "bench --pipeline: %s applies to the"
                            " benchmark harness, not the pipeline\n", argv[i]);
                    nbad++;
                }
        if (nbad) {
            fprintf(stderr, "  drop --pipeline to use them, or drop them.\n");
            return 2;
        }
        /* The pipeline uses the FULL factor base up to each side's lim; the
         * special-q stream is separate and may continue beyond that lim.
         *
         * This comment used to say that truncating at the special-q "costs
         * relations outright -- 30 of 1,851 cofactors at the parity q differ
         * by exactly one prime in (q, alim]". That measurement was right and
         * the conclusion was wrong: those 30 are not lost, they are found
         * again when the special-q reaches that larger prime. Counting what a
         * single q loses, without asking whether a later q recovers it, is the
         * error -- and GGNFS's FB_bound truncation is deliberate duplicate
         * avoidance, not a limitation of its design.
         *
         * Measured on the SNFS job: 1.82 sq-side primes in range per relation
         * and 72% re-found when that prime is later sieved as q, giving 1.34
         * finds per unique relation -- ~25% of raw output is duplicates, and
         * we pay full trial division and cofactorisation on every one.
         * Calibrated against msieve's dedup of the c151: 10,594,292 duplicates
         * in 67,165,877 relations, 15.8%, which back-solves to the same 0.73
         * re-find probability. See task #26. */
        if (cfg.qlist && qrange_set) {
            fprintf(stderr, "bench --pipeline: --qlist and --qrange both give"
                    " the band; pass one\n");
            return 2;
        }
        if (!cfg.qlist && !qrange_set && !rho_set) {
            fprintf(stderr, "bench --pipeline: needs a real root of %s mod q."
                    " Pass --rho, --qlist, or --qrange for a band.\n"
                    "  A synthetic root is fine for a sieve microbenchmark but"
                    " not for a path that emits relations.\n",
                    cfg.sq_side ? "f" : "G");
            return 2;
        }
        if (!fbbound_set) fbbound = alim;
        if (!cfg.lim)  cfg.lim  = fbbound;
        if (!cfg.lim0) cfg.lim0 = rlim;
        if (!cfg.lpb)  cfg.lpb  = 32;
        if (!cfg.mfb)  cfg.mfb  = 92;

        /* A stop file that is still present would be honoured at the very
         * first q, so the run would drain nothing, rewrite the same
         * checkpoint and exit 0 -- telling a job queue the work succeeded
         * while making no progress, forever. Refuse to start instead, rather
         * than deleting a path the operator created. */
        if (cfg.stopfile && bench_path_exists(cfg.stopfile)) {
            fprintf(stderr, "bench: --stop-file %s already exists; remove it"
                    " before starting.\n", cfg.stopfile);
            return 1;
        }
        if (cfg.stopfile && !cfg.relations) {
            fprintf(stderr, "bench: --stop-file needs --relations: there is no"
                    " checkpoint to stop against.\n");
            return 2;
        }

        /* ---- resume, before anything reads qmin (STATUS.md item 12a) ----
         *
         * Everything the fingerprint covers is settled by this point, which is
         * why the check lives here: it has to run before sqgen_create fixes the
         * band start and before the scale derivation below, whose result the
         * checkpoint overrides.
         *
         * THE LOCK COMES FIRST, ahead of --restart's unlink and ahead of the
         * factor-base parse. Taking it later meant --restart could delete a
         * running siever's .part and only then discover the lock -- reporting
         * the conflict after doing the damage, with the victim still writing
         * to an unlinked inode and its final rename doomed to ENOENT. */
        /* Length is checked here for the same reason the lock is taken here:
         * before --restart's unlink and before anything is opened. Discovering
         * an unusable --candidates path after remove(part) has already run
         * would report the problem having done the damage, which is precisely
         * what the lock ordering above exists to prevent. Once both pass, no
         * name derived anywhere in this program can truncate. */
        if ((cfg.relations && ckpt_path_usable(cfg.relations, "--relations")) ||
            (cfg.candidates && ckpt_path_usable(cfg.candidates, "--candidates")))
            return 1;
        if (cfg.relations) {
            if (ckpt_lock(cfg.relations, ckpt_lock_held,
                          sizeof ckpt_lock_held)) {
                ckpt_lock_held[0] = 0;
                return 1;
            }
            atexit(ckpt_unlock_atexit);
        }
        if (cfg.relations) {
            char part[CKPT_PATH_MAX], cpath[CKPT_PATH_MAX];
            char fp[17];
            bench_stat_t st;
            ckpt_t ck;
            if (ckpt_part_path(cfg.relations, part, sizeof part) ||
                ckpt_ckpt_path(cfg.relations, cpath, sizeof cpath))
                return 1;
            ckpt_fingerprint(&POLY, &cfg, fp);
            resume_recovery_t recovery;
            int part_exists;

            if (resume_recovery_init(&recovery, &cfg, part, cpath)) return 1;
            if (bench_stat_path(part, &st) == 0)
                part_exists = 1;
            else if (errno == ENOENT)
                part_exists = 0;
            else {
                perror(part);
                return 1;
            }

#define BOINC_RECOVER_OR_RETURN(why) do {                              \
                const int br_ = boinc_discard_bad_resume(&recovery, why); \
                if (br_ < 0) return 1;                                 \
                if (br_ > 0) goto resume_artifacts_ready;              \
            } while (0)

            /* A hard kill can land after fopen(NAME.part, "wb") creates the
             * staging file but before the first cofactor flush writes a
             * checkpoint. There is no resume ambiguity when that file is
             * empty: it contains no relation and restarting from the band's
             * beginning cannot duplicate or lose work. Heal that state here
             * so an unattended BOINC relaunch does not fail forever.
             *
             * A nonempty candidates .part is different: it may contain useful
             * uncheckpointed output even while the relations file is empty.
             * Preserve the old refusal in that case rather than silently
             * overwriting it. An absent or empty candidates file is safe to
             * restart alongside the empty relations file. */
            if (!cfg.restart && part_exists && st.st_size == 0) {
                bench_stat_t sidecar_st;
                int sidecar_missing = 0;
                int candidates_empty = 1, candidates_present = 0;
                char candidates_part[CKPT_PATH_MAX] = "";

                if (bench_stat_path(cpath, &sidecar_st)) {
                    if (errno == ENOENT)
                        sidecar_missing = 1;
                    else {
                        perror(cpath);
                        return 1;
                    }
                }
                if (sidecar_missing && cfg.candidates) {
                    bench_stat_t candidates_st;
                    if (ckpt_part_path(cfg.candidates, candidates_part,
                                       sizeof candidates_part))
                        return 1;
                    if (bench_stat_path(candidates_part, &candidates_st) == 0) {
                        candidates_present = 1;
                        candidates_empty = candidates_st.st_size == 0;
                    } else if (errno != ENOENT) {
                        perror(candidates_part);
                        return 1;
                    }
                }
                if (sidecar_missing && candidates_empty) {
                    printf("resume: discarding empty uncheckpointed %s; "
                           "starting the band from the beginning\n", part);
                    if (remove(part)) {
                        perror(part);
                        return 1;
                    }
                    if (candidates_present && remove(candidates_part)) {
                        perror(candidates_part);
                        return 1;
                    }
                    part_exists = 0;
                }
            }
            if (cfg.restart) {
                const char *restart_paths[6] = {
                    part, cpath, recovery.ckpt_tmp,
                    recovery.candidates_part, recovery.retry,
                    recovery.retry_tmp
                };
                if (part_exists)
                    printf("resume: --restart, discarding %s (%lld bytes)\n",
                           part, (long long)st.st_size);
                for (unsigned ri = 0; ri < 6; ri++) {
                    if (!restart_paths[ri][0]) continue;
                    if (remove(restart_paths[ri]) && errno != ENOENT) {
                        perror(restart_paths[ri]);
                        return 1;
                    }
                }
                part_exists = 0;
            } else if (part_exists) {
                const int r = ckpt_read(cfg.relations, &ck);
                /* Every branch here refuses rather than guesses. An existing
                 * .part is either resumable work or somebody else's data; the
                 * one thing this must never do is silently truncate it. */
                if (r == -1) {
                    BOINC_RECOVER_OR_RETURN("the .part has no checkpoint");
                    fprintf(stderr,
                        "bench: %s exists but %s does not.\n"
                        "  That file cannot be resumed automatically -- the"
                        " special-q that produced a relation\n"
                        "  cannot be recovered from the relation itself under"
                        " the full factor base (a line\n"
                        "  carries ~1.8 in-band primes and none of them is"
                        " marked). Move it aside, or pass\n"
                        "  --restart to discard it.\n", part, cpath);
                    return 1;
                }
                if (r == -2) {              /* ckpt_read explained why */
                    BOINC_RECOVER_OR_RETURN("the checkpoint is malformed");
                    return 1;
                }
                if (r == -3) return 1;       /* unreadable: ckpt_read explained */
                if (!cfg.candidates && ck.cand_bytes &&
                    bench_boinc_is_managed()) {
                    const int cr =
                        resume_recovery_add_checkpoint_candidate(&recovery,
                                                                 &ck);
                    if (cr < 0) return 1;
                    recovery.checkpoint_candidate_required = 1;
                    recovery.checkpoint_candidate_known = cr > 0;
                }
                if (strcmp(ck.fp, fp)) {
                    BOINC_RECOVER_OR_RETURN(
                        "the checkpoint belongs to a different job");
                    fprintf(stderr,
                        "bench: %s belongs to a different job.\n"
                        "  checkpoint fingerprint %s, this run %s.\n"
                        "  Appending would produce a file whose lines all"
                        " verify and whose yield means\n"
                        "  nothing. Check the polynomial, lim/lpb/mfb, logI/J"
                        " and --sq-side, or --restart.\n",
                        cpath, ck.fp, fp);
                    return 1;
                }
                if (ck.rel_bytes > (unsigned long long)st.st_size) {
                    BOINC_RECOVER_OR_RETURN(
                        "the relation file is shorter than its checkpoint");
                    fprintf(stderr,
                        "bench: %s is %lld bytes but %s claims %llu.\n"
                        "  The relation file has been truncated or replaced"
                        " since the checkpoint.\n",
                        part, (long long)st.st_size, cpath, ck.rel_bytes);
                    return 1;
                }
                /* The candidates .part is truncated to a recorded offset just
                 * like the relations one, so it needs the same guard. Two ways
                 * it goes wrong unguarded: a short file is EXTENDED by
                 * ftruncate and the gap is a run of NULs no parser survives;
                 * and because --candidates is not part of the fingerprint, a
                 * resume that omitted it records cand_bytes = 0, so the next
                 * resume that supplies it again truncates the whole file away. */
                {
                    bench_stat_t cst;
                    int cpart_exists = 0;
                    const char *cpart = recovery.candidates_part;
                    if (cfg.candidates) {
                        if (bench_stat_path(cpart, &cst) == 0)
                            cpart_exists = 1;
                        else if (errno != ENOENT) {
                            perror(cpart);
                            return 1;
                        }
                    }
                    if (cfg.candidates && !ck.cand_bytes &&
                        cpart_exists && cst.st_size > 0) {
                        BOINC_RECOVER_OR_RETURN(
                            "candidate output is not represented by the checkpoint");
                        fprintf(stderr,
                            "bench: %s holds %lld bytes but the checkpoint"
                            " records none.\n"
                            "  It was written by a session run without"
                            " --candidates. Resuming would\n"
                            "  discard it; drop --candidates, or move the file"
                            " aside.\n", cpart, (long long)cst.st_size);
                        return 1;
                    }
                    if (cfg.candidates && ck.cand_bytes &&
                        (!cpart_exists || (unsigned long long)cst.st_size
                                          < ck.cand_bytes)) {
                        BOINC_RECOVER_OR_RETURN(
                            "candidate output is shorter than its checkpoint");
                        fprintf(stderr,
                            "bench: %s is missing or shorter than the %llu"
                            " bytes the checkpoint records.\n", cpart,
                            ck.cand_bytes);
                        return 1;
                    }
                    if (!cfg.candidates && ck.cand_bytes) {
                        BOINC_RECOVER_OR_RETURN(
                            "the checkpoint requires candidate output not supplied by this launch");
                        fprintf(stderr,
                            "bench: the checkpoint records %llu bytes of"
                            " candidates but --candidates was not given.\n"
                            "  Resuming would strand them. Pass the same"
                            " --candidates path, or --restart.\n",
                            ck.cand_bytes);
                        return 1;
                    }
                }
                {   /* Cheap, host-only, and it catches the case the
                     * fingerprint cannot: a .part that was hand-edited, or
                     * concatenated from another job, under a checkpoint that
                     * still matches. Proves the polynomial only. */
                    uint32_t nchecked = 0;
                    const int bad = check_relations_sample(part, &POLY,
                                                           cfg.lpb0, cfg.lpb,
                                                           8, ck.rel_bytes,
                                                           &nchecked);
                    if (bad < 0) return 1;
                    if (bad > 0) {
                        BOINC_RECOVER_OR_RETURN(
                            "sampled relations do not reconstruct");
                        fprintf(stderr,
                            "bench: %d of %u sampled relations in %s do not"
                            " reconstruct.\n"
                            "  Refusing to append to it.\n", bad, nchecked,
                            part);
                        return 1;
                    }
                }
                /* The single-q fallback builds ql[0] from --q/--rho and never
                 * consults the checkpoint, so resuming into it would truncate
                 * the .part, sieve one unrelated q, decide the band finished
                 * normally, rename over the final name and delete the sidecar
                 * -- presenting a multi-day partial band as complete. The band
                 * source is not in the fingerprint, so only this check stops
                 * it. */
                if (!cfg.qlist && !qrange_set) {
                    fprintf(stderr,
                        "bench: %s is a resumable band, but neither --qlist nor"
                        " --qrange was given.\n"
                        "  The single-q path cannot continue it. Supply the"
                        " original band, or --restart.\n", part);
                    return 1;
                }
                cfg.resume = 1;
                cfg.resume_q = ck.next_q;
                cfg.resume_rho = ck.next_rho;
                cfg.resume_rel_bytes = ck.rel_bytes;
                cfg.resume_cand_bytes = ck.cand_bytes;
                cfg.resume_nrel = ck.nrel;
                cfg.resume_nq = ck.nqdone;
                /* The gate must not move mid-run. scale/allowance are derived
                 * once from the band's FIRST q and held band-wide; re-deriving
                 * them here from the resume q would sieve the rest of the job
                 * against a different survivor bound than the part already on
                 * disk. Carried in the checkpoint, restored here, and marked
                 * as explicit so the derivation below leaves them alone. */
                cfg.scale = ck.scale;   scale_set = 1;
                cfg.scale0 = ck.scale0; scale0_set = 1;
                cfg.allowance = ck.allowance;   allowance_set = 1;
                cfg.allowance0 = ck.allowance0; allowance0_set = 1;
                if (qrange_set && cfg.qmin > ck.next_q) {
                    fprintf(stderr,
                        "bench: --qrange starts at %llu but the checkpoint"
                        " resumes at %llu.\n"
                        "  The gap would never be sieved.\n",
                        (unsigned long long)cfg.qmin,
                        (unsigned long long)ck.next_q);
                    return 1;
                }
                if (qrange_set) cfg.qmin = ck.next_q;
                /* --nq counts this session's q, so a resumed run must not be
                 * handed the whole original budget again. */
                if (cfg.nq_max) {
                    cfg.nq_max = ck.nqdone >= cfg.nq_max
                        ? 0 : (uint32_t)(cfg.nq_max - ck.nqdone);
                    if (!cfg.nq_max) {
                        printf("resume: --nq already satisfied by %llu"
                               " completed q; nothing to do\n", ck.nqdone);
                        return 0;
                    }
                }
                printf("resume: %s at q=%llu rho=%llu -- %llu relations,"
                       " %llu q done, %llu bytes kept\n",
                       part, (unsigned long long)ck.next_q,
                       (unsigned long long)ck.next_rho,
                       ck.nrel, ck.nqdone, ck.rel_bytes);
                printf("        scale %.4f/%.4f allowance %.2f/%.2f restored"
                       " from the checkpoint\n",
                       cfg.scale, cfg.scale0, cfg.allowance, cfg.allowance0);
            } else {
                bench_stat_t checkpoint_st;
                if (bench_stat_path(cpath, &checkpoint_st) == 0) {
                    BOINC_RECOVER_OR_RETURN(
                        "a checkpoint exists without its relation file");
                    fprintf(stderr,
                        "bench: %s exists but %s does not. Remove the checkpoint"
                        " to start fresh.\n", cpath, part);
                    return 1;
                }
                if (errno != ENOENT) {
                    perror(cpath);
                    return 1;
                }
            }
        }
resume_artifacts_ready:
#undef BOINC_RECOVER_OR_RETURN

        /* --qlist is read HERE, before the factor base, for two reasons. It
         * needs nothing from the base, so a missing or malformed list should
         * fail before a 29 MB parse rather than after it. And the scale
         * derivation below reads ql[0] to build its lattice: this block used
         * to sit ~90 lines further down, past that point, so --qlist left nq
         * at 0 and the derivation silently fell back to the hardcoded default
         * q -- a c183-sized lattice for whatever job was actually running,
         * with the wrong scale applied to the entire band. --qrange was
         * unaffected only because it happens to populate ql[] earlier. */
        if (cfg.qlist) {
            FILE *f = fopen(cfg.qlist, "r");
            char line[256];
            if (!f) { perror(cfg.qlist); return 1; }
            unsigned long lno = 0;
            while (fgets(line, sizeof line, f)) {
                unsigned long long qq, rr;
                const char *s = line;
                lno++;
                while (*s == ' ' || *s == '\t') s++;
                if (*s == '#' || *s == '\n' || *s == '\r' || !*s) continue;
                /* Was `continue`. A typo'd or wrongly-columned q-list then ran
                 * as a SHORTER band with no diagnostic at all -- the count
                 * printed below was the only hint, and only if you knew what it
                 * should have been. */
                if (sscanf(s, "%llu %llu", &qq, &rr) != 2) {
                    fprintf(stderr, "%s:%lu: expected `q rho`, got: %s",
                            cfg.qlist, lno, line);
                    fclose(f); free(ql); return 1;
                }
                /* One validator serves q-lists, generated ranges and the
                 * single-q path. Besides primality it checks the exact root and
                 * reduces rho before qlat_build can narrow it to int64_t. */
                qsel_t ent;
                ent.q = qq; ent.rho = rr;
                if (validate_qsel_or_report(&ent, &POLY, cfg.sq_side,
                                            cfg.qlist, lno)) {
                    fclose(f); free(ql); return 1;
                }
                if (nq == capq) {
                    capq = capq ? capq * 2 : 256;
                    ql = (qsel_t *)realloc(ql, (size_t)capq * sizeof(qsel_t));
                    if (!ql) { fclose(f); return 1; }
                }
                /* ent, not (qq, rr): validate_qsel_or_report has reduced rho
                 * mod q in place, and that canonical form is what the lattice
                 * build and the checkpoint sidecar must both see. */
                ql[nq++] = ent;
                /* When resuming, --nq is a budget for THIS session and the
                 * resume point may be past this many entries, so the list
                 * cannot be truncated until after the skip below. */
                if (!cfg.resume && cfg.nq_max && nq >= cfg.nq_max) break;
            }
            fclose(f);
            if (!nq) { fprintf(stderr, "%s: no `q rho` pairs\n", cfg.qlist); return 1; }
            printf("band: %u special-q from %s\n", nq, cfg.qlist);
            /* A list is resumed by POSITION, not by q: the checkpoint names the
             * exact (q, rho) pair that was next, and the list may legitimately
             * repeat a q with different roots. An entry that is not in this
             * list means the list changed under the checkpoint, which is the
             * same class of error as a fingerprint mismatch. */
            if (cfg.resume) {
                uint32_t at = 0;
                while (at < nq && !(ql[at].q == cfg.resume_q &&
                                    ql[at].rho == cfg.resume_rho)) at++;
                if (at == nq) {
                    fprintf(stderr,
                        "bench: the checkpoint resumes at q=%llu rho=%llu,"
                        " which is not in %s.\n",
                        (unsigned long long)cfg.resume_q,
                        (unsigned long long)cfg.resume_rho, cfg.qlist);
                    free(ql); return 1;
                }
                printf("resume: skipping the first %u entries of %s\n",
                       at, cfg.qlist);
                memmove(ql, ql + at, (size_t)(nq - at) * sizeof(qsel_t));
                nq -= at;
                if (cfg.nq_max && nq > cfg.nq_max) nq = cfg.nq_max;
                if (!nq) {
                    printf("resume: %s is exhausted; nothing to do\n",
                           cfg.qlist);
                    free(ql); return 0;
                }
            }
        }

        /* --qrange is a stream of prime ideals, independent of the factor
         * base.  lim bounds the small ideals used by sieve/TD; it is not an
         * upper bound on q.  Cache only the first generated pair because norm
         * and byte-scale setup needs its lattice before run_pipeline starts;
         * the rest are pulled on demand so MIN: can run until the relation
         * target without allocating a list through 2^32. */
        if (qrange_set) {
            int qr;
            if (!cfg.qmax && !cfg.target_rels && !cfg.nq_max) {
                fprintf(stderr, "bench: open --qrange MIN: needs --target-rels"
                                " or --nq as a stopping condition\n");
                return 2;
            }
            /* sqgen counts every pair it EMITS against its nqmax, and the
             * resume wind-forward below discards up to deg-1 of them to reach
             * the exact root. Charging those to the budget makes the generator
             * stop early and report "q range exhausted" on a range that is
             * nowhere near it, so when resuming the count limit is enforced by
             * run_pipeline's own nqdone check instead. */
            qgen = sqgen_create(&POLY, cfg.sq_side, cfg.qmin, cfg.qmax,
                                cfg.resume ? 0 : cfg.nq_max);
            if (!qgen) return 1;
            ql = (qsel_t *)malloc(sizeof(*ql));
            if (!ql) { sqgen_free(qgen); return 1; }
            qr = sqgen_next(qgen, ql);
            if (qr < 0) {
                fprintf(stderr, "bench: special-q generator failed before its"
                                " first result\n");
                free(ql); sqgen_free(qgen); return 1;
            }
            if (qr == 0) {
                if (cfg.qmax)
                    fprintf(stderr, "bench: no affine special-q roots in"
                            " [%llu, %llu]\n",
                            (unsigned long long)cfg.qmin,
                            (unsigned long long)cfg.qmax);
                else
                    fprintf(stderr, "bench: no affine special-q roots from"
                            " %llu through the 32-bit q range\n",
                            (unsigned long long)cfg.qmin);
                free(ql); sqgen_free(qgen); return 1;
            }
            if (validate_qsel_or_report(ql, &POLY, cfg.sq_side,
                                        "bench: generated special-q", 0)) {
                free(ql); sqgen_free(qgen); return 1;
            }
            nq = 1;
            /* cfg.qmin is already the checkpoint's q, but a prime carries one
             * (q, rho) entry per affine root and the stop may have landed on
             * the second or third of them. Wind forward to the exact pair --
             * restarting at the first root of that q would re-sieve roots
             * whose relations are already in the file. */
            if (cfg.resume) {
                int guard = 0;
                while ((ql->q != cfg.resume_q || ql->rho != cfg.resume_rho) &&
                       ++guard <= BENCH_NCOEFF) {
                    qr = sqgen_next(qgen, ql);
                    if (qr <= 0) break;
                }
                if (ql->q != cfg.resume_q || ql->rho != cfg.resume_rho) {
                    fprintf(stderr,
                        "bench: the generator does not produce q=%llu rho=%llu"
                        " at the checkpoint.\n"
                        "  --sq-side or the polynomial has changed since it was"
                        " written.\n",
                        (unsigned long long)cfg.resume_q,
                        (unsigned long long)cfg.resume_rho);
                    free(ql); sqgen_free(qgen); return 1;
                }
            }
            if (cfg.qmax)
                printf("band: generated prime special-q roots on side %d in"
                       " [%llu, %llu]\n", cfg.sq_side,
                       (unsigned long long)cfg.qmin,
                       (unsigned long long)cfg.qmax);
            else
                printf("band: generating prime special-q roots on side %d from"
                       " %llu upward\n", cfg.sq_side,
                       (unsigned long long)cfg.qmin);
        }

        /* Construct and validate the single-q selection BEFORE scale/norm
         * derivation. The old path waited until after both factor bases were
         * loaded, so q=0 reached qlat_build and the synthetic-rho expression,
         * a composite q could become a relation factor, and rho>=2^63 was
         * narrowed before it was canonicalised. */
        if (!cfg.qlist && !qrange_set) {
            ql = (qsel_t *)malloc(sizeof(*ql));
            if (!ql) return 1;
            ql[0].q = q;
            ql[0].rho = rho;
            if (validate_qsel_or_report(ql, &POLY, cfg.sq_side,
                                        "bench: --q/--rho", 0)) {
                free(ql);
                return 1;
            }
            nq = 1;
            printf("band: one validated special-q on side %d\n", cfg.sq_side);
        }

        /* Derive the byte scale and survivor allowance from the polynomial,
         * as las does. This is UNCONDITIONAL. It used to sit behind
         * --auto-params, whose "off" state was not a mode but a frozen copy of
         * the c183's derived constants -- correct for exactly one polynomial
         * and quietly wrong for every other. An explicit --scale/--allowance
         * still overrides, which is the override that was actually wanted.
         *
         * The scale depends on the largest norm over the sieve rectangle,
         * which needs a q-lattice. The q list or streaming generator has
         * already supplied its first pair above. Derive the real scale first,
         * then load the factor base once with that scale. */
        {
            qlat_t L0; norm_t N1, N0;
            double m1, m0;
            uint32_t bound1 = 0, bound0 = 0;
            const uint64_t q0 = ql[0].q;
            const uint64_t rh0 = ql[0].rho;
            poly_t P0 = POLY;      /* side 0's norm is G = Y1*x + Y0, degree 1 */
            P0.deg = 1; P0.c[0] = P0.y0; P0.c[1] = P0.y1;
            for (int z = 2; z < BENCH_NCOEFF; z++) P0.c[z] = 0.0;
            qlat_build(&L0, q0, rh0, POLY.skew);
            /* Only the sq side's norm carries a factor of q to divide out.
             * Passing is_sqside=1 for side 1 unconditionally made side 0's
             * maxnorm ~log2(q) too large and side 1's too small whenever the q
             * lives on the rational side -- a ~25-bit error in both scales, in
             * opposite directions. */
            norm_setup(&N1, &POLY, &L0, cfg.logI, cfg.J, 1.0, cfg.sq_side == 1);
            norm_setup(&N0, &P0,   &L0, cfg.logI, cfg.J, 1.0, cfg.sq_side == 0);
            m1 = (double)(N1.log2M - N1.bias);
            m0 = (double)(N0.log2M - N0.bias);
            if (!isfinite(m1) || !isfinite(m0)) {
                fprintf(stderr, "derived scale: nonfinite maxnorm"
                        " (side1 %.17g, side0 %.17g)\n", m1, m0);
                return 1;
            }
            /* An explicit --scale / --allowance is a deliberate override: it
             * is how a swept operating point, or a bound the job file does not
             * express, gets stated. */
            if (!scale_set)  cfg.scale  = las_scale(m1);
            if (!scale0_set) cfg.scale0 = las_scale(m0);
            if (!isfinite(cfg.scale) || cfg.scale <= 0.0 ||
                !isfinite(cfg.scale0) || cfg.scale0 <= 0.0) {
                fprintf(stderr, "derived scale: degenerate maxnorm"
                        " (side1 %.2f, side0 %.2f)\n", m1, m0);
                return 1;
            }
            /* The startup guards on --scale ran long before this point and
             * against the c183's norm sizes, so a DERIVED scale never met
             * them. Factor-base logs are uint8 values and are now rejected if
             * they do not fit; recheck here, on both sides, against the real
             * bounds so failure precedes the expensive factor-base load. */
            {
                const double cinit = (cfg.cell_bits == 16) ? 4096.0 : 255.0;
                const struct { double sc, mx; uint32_t lim; const char *nm; } chk[2] = {
                    { cfg.scale,  m1, fbbound, "side 1" },
                    { cfg.scale0, m0, rlim,    "side 0" }
                };
                for (int z = 0; z < 2; z++) {
                    const double lg = chk[z].lim > 1
                        ? log((double)chk[z].lim) / log(2.0) : 0.0;
                    const double sn = chk[z].sc * chk[z].mx;
                    const double sp = chk[z].sc * lg;
                    if (!isfinite(sn) || sn > cinit) {
                        fprintf(stderr, "%s: scale %.3f x log2(maxnorm) %.1f"
                                " exceeds CINIT %.0f\n",
                                chk[z].nm, chk[z].sc, chk[z].mx, cinit);
                        return 1;
                    }
                    if (!isfinite(sp) || sp > 255.0) {
                        fprintf(stderr, "%s: scale %.3f x log2(lim) %.1f exceeds"
                                " the 8-bit per-ideal log\n",
                                chk[z].nm, chk[z].sc, lg);
                        return 1;
                    }
                }
            }
            /* Default: our own rule, mfb + the slack our approximation needs.
             * --lambda0/--lambda1 opt back into CADO's rule for anyone who
             * wants it; an explicit --allowance overrides both. */
            if (!allowance_set)
                cfg.allowance  = lambda1_set
                    ? las_allowance(m1, cfg.scale, lambda1,
                                    cfg.lpb ? cfg.lpb : 32,
                                    cfg.mfb ? cfg.mfb : 92)
                    : sieve_allowance(m1, cfg.scale, cfg.mfb ? cfg.mfb : 92);
            if (!allowance0_set)
                cfg.allowance0 = lambda0_set
                    ? las_allowance(m0, cfg.scale0, lambda0, cfg.lpb0, cfg.mfb0)
                    : sieve_allowance(m0, cfg.scale0, cfg.mfb0);
            if (sieve_bound_checked(cfg.scale, cfg.allowance, 4096u, &bound1,
                                    "side 1 survivor parameters") ||
                sieve_bound_checked(cfg.scale0, cfg.allowance0, 4096u, &bound0,
                                    "side 0 survivor parameters"))
                return 1;
            /* uint32_t, matching pipe_side_init. The (unsigned char) this used
             * to print through wrapped any bound above 255 -- legal against a
             * 16-bit cell with CINIT 4096 -- so the banner disagreed with the
             * bound the sieve actually ran. */
            printf("params from q=%llu: side 1 log2(maxnorm)=%.2f scale=%.3f"
                   " allowance=%.2f bound=%u\n", (unsigned long long)q0, m1,
                   cfg.scale, cfg.allowance, bound1);
            printf("                    side 0 log2(maxnorm)=%.2f scale=%.3f"
                   " allowance=%.2f bound=%u\n", m0, cfg.scale0, cfg.allowance0,
                   bound0);
            /* An allowance far above mfb admits survivors that classify will
             * then reject: the sieve keeps a position whose cofactor is bigger
             * than the cofactoriser is willing to touch, and the whole resieve
             * and trial division for it is thrown away.
             *
             * SOME slack is right -- the survivor test is a byte-quantised log
             * approximation, so a bound at exactly mfb loses relations to
             * rounding. One byte unit is 1/scale bits, and ~1 bit of slack is
             * what that buys. Beyond that it is pure waste. Measured on the
             * SNFS job (rlambda 3.4 -> 91.80 against mfbr 88):
             *
             *     91.80  120,317 survivors/q   10,313 relations
             *     89.00   99,808 survivors/q   10,312   (-0.01%)
             *     88.00   90,782 survivors/q   10,306   (-0.07%)
             *     85.00   74,849 survivors/q    9,668   (-6.25%)
             *
             * so 17% of the trial-division input was buying one relation in
             * ten thousand. Warned rather than clamped: it is the job file's
             * stated parameter and silently overriding it would make a run
             * something other than what was asked for. */
            {
                const double sl1 = cfg.allowance  - (double)(cfg.mfb ? cfg.mfb : 92);
                const double sl0 = cfg.allowance0 - (double)cfg.mfb0;
                /* Compared against what the derivation WOULD have produced,
                 * not against a fixed number of bits over mfb. A fixed 2.0
                 * threshold fired on the derived default itself whenever
                 * scale < 1 -- which happens for any maxnorm above 254 bits,
                 * since slack is 2/scale -- telling the operator to "drop the
                 * override" and use the very value already in force.
                 *
                 * Per side, too: one overridden side used to print and
                 * "correct" both, naming a side-0 override that was never
                 * passed. */
                const double d1 = sieve_allowance(m1, cfg.scale,
                                                  cfg.mfb ? cfg.mfb : 92);
                const double d0 = sieve_allowance(m0, cfg.scale0, cfg.mfb0);
                if (cfg.allowance > d1 + 0.01)
                    fprintf(stderr,
                        "note: side 1 allowance %.2f is %.2f bits looser than"
                        " the derived %.2f; the surplus\n"
                        "      admits survivors the cofactoriser then rejects"
                        " (mfb %u).\n",
                        cfg.allowance, cfg.allowance - d1, d1,
                        cfg.mfb ? cfg.mfb : 92);
                if (cfg.allowance0 > d0 + 0.01)
                    fprintf(stderr,
                        "note: side 0 allowance %.2f is %.2f bits looser than"
                        " the derived %.2f; the surplus\n"
                        "      admits survivors the cofactoriser then rejects"
                        " (mfb %u).\n",
                        cfg.allowance0, cfg.allowance0 - d0, d0, cfg.mfb0);
                (void)sl1; (void)sl0;
            }
            if (cfg.cadofb) {
                if (fb_load_cado(cfg.cadofb, cfg.scale, &fb1) != 0) return 1;
            } else if (fbpath_set) {
                /* Explicit --fb retains the legacy GGNFS .afb.0 path for
                 * sieve-only/debug use.  It remains unsuitable for relation
                 * production because it carries no prime-power metadata. */
                if (fb_load(fbpath, &fb1) != 0) return 1;
                if (fb_fill_logp(&fb1, cfg.scale) != 0) return 1;
            } else {
                /* Production default: regenerate the complete algebraic FB on
                 * the assigned GPU.  --fbbound is a truncation knob, not an
                 * extension beyond the job's alim, matching the file-backed
                 * path and the reporting below. */
                const uint32_t genlim = fbbound < alim ? fbbound : alim;
                fprintf(stderr,
                        "bench: no --fb1 supplied; generating algebraic factor base on GPU through %u\n",
                        genlim);
                if (afb_build_gpu(&POLY, genlim, maxbits, cfg.scale, -1, 0, 1,
                                  &fb1) != 0)
                    return 1;
                fb1_generated = 1;
            }
        }
        if (cfg.cadofb && fb1.maxbits > 0 && fb1.maxbits != maxbits)
            fprintf(stderr,
                    "note: algebraic factor base says maxbits=%d, while --maxbits=%d;\n"
                    "      the file controls algebraic powers and the flag controls rational powers.\n"
                    "      maxbits=1 is supported but prime-only and normally lower-yielding;\n"
                    "      regenerate the file or pass its bound explicitly with --maxbits.\n",
                    fb1.maxbits, maxbits);
        if (fb_split_small(&fb1, bkthresh, &fbs1) != 0) return 1;
        /* The GGNFS .afb.0 format carries neither p = 2 nor any prime power, so
         * the default factor base silently under-divides every algebraic norm by
         * its full power of 2 and loses ~2/3 of the relations -- with no error
         * anywhere, because the norms still reconstruct exactly and the leftover
         * 2^k merely makes the cofactor even. Montgomery arithmetic then needs an
         * odd modulus, so those records burn the whole rho budget and report
         * "stuck". Cost me a day; check the invariant instead. */
        {
            uint32_t i, n2 = 0;
            for (i = 0; i < fbs1.n; i++) if (fbs1.primes[i] == 2) n2++;
            /* Absent p = 2 is only a defect if f actually HAS a root mod 2.
             * When it does not -- true of the SNFS polynomial x^5+x^4-4x^3-
             * 3x^2+3x+1, whose f(0), f(1) and leading coefficient are all odd
             * -- the algebraic norm is always odd, fbgen is right to emit no
             * entry, and refusing the run rejects a perfectly good job. This
             * guard used to test only for the entry's presence and did exactly
             * that on the first SNFS job it saw. */
            /* Historically this p=2 check exposed incomplete legacy .afb.0
             * input.  Relation-producing runs now either load a complete
             * --fb1 file or generate the complete algebraic FB in-process; an
             * explicitly requested legacy --fb is rejected directly below. */
            if (fbpath_set && !cfg.cadofb &&
                (cfg.relations || cfg.candidates || cfg.cofactor)) {
                fprintf(stderr,
                    "ERROR: relation-producing runs cannot use legacy --fb .afb.0 input.\n"
                    "         It carries neither p = 2 metadata nor prime powers.\n"
                    "         Omit --fb to generate the complete algebraic factor base on GPU,\n"
                    "         or pass --fb1 <fbgen output>.\n");
                return 1;
            }
            if (!n2 && !poly_has_root_mod2(&POLY)) {
                printf("note: f has no root mod 2, so the algebraic norm is"
                       " always odd and the factor base correctly has no"
                       " p = 2 entry\n");
            } else if (!n2) {
                const int producing = cfg.relations || cfg.candidates || cfg.cofactor;
                fprintf(stderr,
                        "%s: the algebraic factor base has no entry for p = 2"
                        " (%u small entries, first p = %u).\n"
                        "         Algebraic norms keep their power of 2, so the"
                        " cofactors are EVEN --\n"
                        "         and mz_n0inv requires an odd modulus, so they"
                        " cannot be split at all.\n"
                        "         Pass --fb1 <fbgen output>.\n",
                        producing ? "ERROR" : "WARNING",
                        fbs1.n, fbs1.n ? fbs1.primes[0] : 0);
                /* A sieve-only run may reasonably continue: it never reaches the
                 * cofactoriser, and the survivor counts are still meaningful. A
                 * relation-producing one may not -- it would exit 0 having
                 * quietly lost two thirds of the yield, which is exactly how
                 * this was mistaken for a regression once already. */
                if (producing) return 1;
            }
        }
        if (fb_restrict(&fb1, bkthresh, fbbound) != 0) return 1;

        if (rfb_build(&POLY, rlim, maxbits, cfg.scale0, &fb0) != 0) return 1;
        if (fb_fill_logp(&fb0, cfg.scale0) != 0) return 1;
        if (fb_split_small(&fb0, bkthresh, &fbs0) != 0) return 1;
        if (fb_restrict(&fb0, bkthresh, rlim) != 0) return 1;

        printf("\nside 1 (algebraic): bucketed %u <= p < %u : %u entries,"
               " line-sieved %u\n", bkthresh, fbbound, fb1.n, fbs1.n);
        printf("side 0 (rational) : bucketed %u <= p < %u : %u entries,"
               " line-sieved %u\n", bkthresh, rlim, fb0.n, fbs0.n);

        /* ---- run log (STATUS.md item 12b) ----
         *
         * Opened HERE, after every refusal above has had its chance: a log
         * whose header describes a run that then declined to start is worse
         * than no log, because the header is the part a reader trusts. By this
         * point the band, the resume state, the derived gate and both factor
         * bases are settled, so every field below is the value the run will
         * actually use rather than the value it was asked for. */
        if (cfg.logpath) {
            char job[2048], cmd[4096];
            size_t k = 0;
            int drv = 0, rt = 0;
            const int drv_ok = cudaDriverGetVersion(&drv) == cudaSuccess;
            const int rt_ok = cudaRuntimeGetVersion(&rt) == cudaSuccess;
            char ver[64];
            runlog_open(cfg.logpath, cfg.log_every_s);
            /* Reported rather than dropped. Two of the four numbers this log
             * exists to carry come from NVML, and without this line three days
             * of `gpu=n/a board=n/a` give no way to tell a missing driver
             * library from a PCI lookup that found the wrong card. */
            if (runlog_gpu_bind(dev_pci) == 0)
                runlog_note("telemetry", "NVML bound to %s", dev_pci);
            else
                runlog_note("telemetry", "NVML unavailable; the gpu= and"
                            " board= columns will read n/a");
            ckpt_job_text(&POLY, &cfg, job, sizeof job);
            runlog_note("commit", "%s", runlog_build_desc());
            if (*runlog_build_defs())
                runlog_note("BUILD-DEFS", "%s  <- pricing build, NOT production"
                            " output", runlog_build_defs());
            /* The full command line. Reconstructing it from the printed
             * parameters is what findings 43-44 needed and did not have; see
             * finding 55. ckpt_fp_cat is the tested bounded append -- it
             * clamps k to the buffer size on overflow and never advances past
             * it -- and it is already in this translation unit. */
            cmd[0] = 0;
            for (int i = 0; i < argc; i++)
                ckpt_fp_cat(cmd, sizeof cmd, &k, "%s%s", i ? " " : "", argv[i]);
            runlog_note("argv", "%s", cmd);
            runlog_note("job", "%s", job);
            {
                char fp[17];
                ckpt_fingerprint(&POLY, &cfg, fp);
                runlog_note("fingerprint", "%s", fp);
            }
            /* A failed version query would otherwise read as "driver 0.0",
             * which is indistinguishable from a real answer in the one field
             * that ties a log to an environment. */
            if (drv_ok && rt_ok)
                snprintf(ver, sizeof ver, "driver %d.%d, runtime %d.%d",
                         drv / 1000, (drv % 1000) / 10,
                         rt / 1000, (rt % 1000) / 10);
            else
                snprintf(ver, sizeof ver, "driver/runtime version unavailable");
            runlog_note("device", "CUDA %d of %d: %s [%s], %s",
                        dev_ordinal, dev_count, dev_name, dev_pci, ver);
            /* Geometry in one field, in the units the CPU comparison uses.
             * logI 15 is gnfs-lasieve4I15e and logI 14 is I14e, and grading a
             * band against the wrong one of those is worth 4x (finding 55), so
             * the number belongs in the run's own record and not only in the
             * source defaults. */
            runlog_note("geometry", "logI=%d J=%u (I%de, area %.4g) region=2^%d"
                        " maxbits=%d blocks=%d threads=%d fill=%dx%d",
                        cfg.logI, cfg.J, cfg.logI,
                        (double)(1u << cfg.logI) * cfg.J, cfg.log_region,
                        maxbits, cfg.blocks, cfg.threads,
                        cfg.fill_blocks ? cfg.fill_blocks : FILL_BLOCKS_DEFAULT,
                        cfg.fill_threads ? cfg.fill_threads
                                         : FILL_THREADS_DEFAULT);
            /* The factor-base convention, spelled out rather than implied. The
             * pipeline runs the FULL base while GGNFS truncates at q, and the
             * ratio between the two is 2.54x at q=50M and 1.03x at 130M on the
             * c183 -- finding 55, and the reason a band's ms/q cannot be read
             * without this line. */
            runlog_note("fb", "side1 %u entries p in [%u, %u)%s; side0 %u"
                        " entries p in [%u, %u)", fb1.n, bkthresh, fbbound,
                        fbbound >= alim ? " = full base to alim"
                                        : " (truncated below alim)",
                        fb0.n, bkthresh, rlim);
            runlog_note("fb-source", "side1 %s",
                        fb1_generated ? "generated in-process on GPU with exact CPU Hensel branches" :
                        (cfg.cadofb ? "fbgen/CADO text file" : "legacy GGNFS .afb.0"));
            runlog_note("gate", "sq-side %d  scale %.4f/%.4f  allowance"
                        " %.2f/%.2f  lpb %u/%u  mfb %u/%u", cfg.sq_side,
                        cfg.scale, cfg.scale0, cfg.allowance, cfg.allowance0,
                        cfg.lpb, cfg.lpb0, cfg.mfb, cfg.mfb0);
            if (cfg.resume)
                runlog_note("resume", "at q=%llu rho=%llu after %llu q,"
                            " %llu relations",
                            (unsigned long long)cfg.resume_q,
                            (unsigned long long)cfg.resume_rho,
                            cfg.resume_nq, cfg.resume_nrel);
            {   /* The band as resolved, not the name of the mechanism that
                 * supplied it: "qrange" alone is the one header field that
                 * would tell a reader nothing about which special-q the run
                 * covered. On a resume cfg.qmin is already the checkpoint's q,
                 * which is the number that matters -- the argv note carries
                 * what was typed. */
                char band[128];
                if (cfg.qlist)
                    snprintf(band, sizeof band, "qlist %s", cfg.qlist);
                else if (cfg.qmax)
                    snprintf(band, sizeof band, "qrange %llu:%llu",
                             (unsigned long long)cfg.qmin,
                             (unsigned long long)cfg.qmax);
                else
                    snprintf(band, sizeof band, "qrange %llu: (open)",
                             (unsigned long long)cfg.qmin);
                runlog_note("band", "%s  target-rels %llu  nq %u  relations %s",
                            band, (unsigned long long)cfg.target_rels,
                            cfg.nq_max,
                            cfg.relations ? cfg.relations : "(none)");
            }
            /* Timestamped, unlike the notes above: it is the record every
             * later one is an elapsed time from, and it is what distinguishes
             * the sessions of a resumed job in an appended file. */
            runlog_record("band start%s", cfg.resume ? " (resumed)" : "");
        }

        {
            /* ql is already built and validated: the single-q path now runs
             * before scale derivation rather than here, so that q=0, a
             * composite q and an unreduced rho are all rejected before
             * qlat_build sees them.
             *
             * The .part lock was taken before the resume check, far above --
             * see the comment there. It is released by atexit. */
            prc = run_pipeline(&fb1, &fbs1, &fb0, &fbs0, ql, nq, qgen,
                               &POLY, &cfg);
            free(ql);
            sqgen_free(qgen);
        }
        runlog_close();
        fb_free(&fb1); fb_free(&fbs1); fb_free(&fb0); fb_free(&fbs0);
        return prc;
    }


    /* The special-q lives on side 1, so only side 1's factor base is truncated
     * at q (the GGNFS convention). Side 0 runs to rlim regardless. */
    if (cfg.side == 1) {
        if (cfg.cadofb) {
            if (fb_load_cado(cfg.cadofb, cfg.scale, &fb) != 0) return 1;
            if (fb.maxbits > 0 && fb.maxbits != maxbits)
                fprintf(stderr,
                        "note: factor base maxbits=%d differs from --maxbits=%d;"
                        " maxbits=1 is valid but normally lower-yielding\n",
                        fb.maxbits, maxbits);
        } else {
            if (fb_load(fbpath, &fb) != 0) return 1;
            printf("\nfactor base %s: %u (p,r) pairs\n", fbpath, fb.n);
        }
    } else {
        if (rfb_build(&POLY, rlim, maxbits, cfg.scale, &fb) != 0) return 1;
        if (!fbbound) fbbound = rlim;
        /* side 0 is G(x) = Y1*x + Y0: degree 1, and its norms are what the
         * apply kernel must initialise cells with */
        POLY.deg = 1; POLY.c[0] = POLY.y0; POLY.c[1] = POLY.y1;
        for (int z = 2; z < BENCH_NCOEFF; z++) POLY.c[z] = 0.0;
        printf("\nside 0 (rational): degree 1, G(x) = Y1*x + Y0\n");
    }
    if (fb_fill_logp(&fb, cfg.scale) != 0) return 1;
    if (fb_split_small(&fb, bkthresh, &fbs) != 0) return 1;
    if (fb_restrict(&fb, bkthresh, fbbound) != 0) return 1;
    printf("  bucketed  %u <= p < %u : %u entries\n", bkthresh, fbbound, fb.n);

    /* Cofactor-classification parameters, from oracle/input.job. `lim` is the
     * factor-base bound actually sieved, which is what CADO's gap test uses --
     * so truncating the factor base tightens the classification too. */
    /* --td runs inside the two-sided intersection, so without the other
     * side's bitmap it would silently do nothing at all. */
    if (cfg.td && !cfg.other_bits) {
        fprintf(stderr, "bench: --td needs --other-bits FILE (the other side's"
                        " survivor bitmap); trial division runs on the\n"
                        "       two-sided survivor list, which does not exist"
                        " without it.\n");
        return 2;
    }
    if (!cfg.lim) cfg.lim = fbbound;
    if (!cfg.lpb) cfg.lpb = (cfg.side == 1) ? 32u : 31u;
    if (!cfg.mfb) cfg.mfb = (cfg.side == 1) ? 92u : 60u;
    printf("  line-sieved       p < %u (plus every p^k, k>=2) : %u entries\n",
           bkthresh, fbs.n);
    if (!fb.n) { fprintf(stderr, "empty factor base after restriction\n"); return 1; }

    qlat_build(&L, q, rho_set ? rho
                              : (uint64_t)(0x9E3779B97F4A7C15ull % q),
               POLY.skew);
    printf("q-lattice q=%llu basis (a0,a1,b0,b1) = (%lld,%lld,%lld,%lld), det=%lld\n",
           (unsigned long long)q, (long long)L.a0, (long long)L.a1,
           (long long)L.b0, (long long)L.b1, (long long)(L.a0 * L.b1 - L.a1 * L.b0));
    if (!rho_set)
        printf("  NOTE: rho is synthetic (no root of f mod q computed here). The basis is\n"
               "  a genuine reduced q-lattice basis of the right shape, and transformed\n"
               "  roots are uniform mod p either way, so fill volume and distribution are\n"
               "  representative. Pass --rho to match a real las special-q.\n");
    else
        printf("  rho = %llu (real root of f mod q); compare against the a0/b0/a1/b1\n"
               "  that las -v prints, remembering las groups by coordinate and qlat_t\n"
               "  by vector: las (a0,b0,a1,b1) == qlat_t (a0,a1,b0,b1).\n",
               (unsigned long long)rho);

    printf("\nsieve area I=2^%d x J=%u = %.3e positions; region 2^%d; %u regions\n",
           cfg.logI, cfg.J, (double)(1u << cfg.logI) * cfg.J, cfg.log_region,
           ((1u << cfg.logI) * cfg.J) >> cfg.log_region);
    printf("mode=%s record=%dB threads=%d reps=%d\n\n",
           cfg.fill_mode == FILL_ATOMIC ? "atomic" : "twolevel",
           cfg.record_bytes, cfg.threads, cfg.reps);

    if (cfg.verify) {
        int nc;
        /* SEVERAL ASPECT RATIOS, not just J = I/2. This gate ran only at
         * (8, 128) -- 2:1 -- for the project's whole life, and every geometry
         * ever deployed is also 2:1, so "the walk enumerates the rectangle"
         * was established for one shape and assumed for the rest. That gap
         * surfaced when 2^15 x 2^15 measured 14% short of GGNFS (finding 58)
         * and the walk was the first suspect; finding 63 is where it was
         * eliminated, and these cases are what establish that rather than
         * leave it inferred. */
        if (verify_walk_cases()) return 1;
        printf("[verify] root transform vs its definition, by set equality...\n");
        if (verify_transform(&L, 200) != 0) { printf("[verify] FAILED\n"); return 1; }
        printf("[verify] OK: q <= 200, every root in [0,2q) -- primes, prime\n"
               "         powers, even moduli, affine and projective\n");
        printf("[verify] the same, over the loaded factor base itself...\n");
        nc = verify_fb_transform(&fbs, &L, bkthresh);
        if (nc < 0) { printf("[verify] FAILED\n"); return 1; }
        printf("[verify] OK: %d small-part entries agree with the definition\n", nc);
    }

    int rc = run_bench(&fb, &fbs, &L, &POLY, &cfg);
    fb_free(&fb);
    return rc;
}

int main(int argc, char **argv)
{
    int rc;
    rc = bench_boinc_init();
    if (rc) return bench_boinc_finish(rc);
    /* A pricing build alters the norm, so its relations are not production
     * output. The run log records that (BUILD-DEFS), but ONLY when --log was
     * passed and only for --pipeline, so a pricing binary run without it left
     * no marker anywhere. Say so unconditionally on stderr.
     *
     * AFTER bench_boinc_init(), not before: boinc_init_options() reopens
     * stderr onto the slot's stderr.txt, which is the file that actually gets
     * uploaded. Printing first sends the banner to the client's inherited
     * stderr instead, so the volunteer's result would carry no marker at all --
     * exactly the case this exists to cover. Empty, and therefore silent, in
     * every shipping build. */
    if (*runlog_build_defs())
        fprintf(stderr, "*** PRICING BUILD: %s -- relations from this binary"
                " are NOT production output ***\n", runlog_build_defs());
    rc = bench_main_impl(argc, argv);
    return bench_boinc_finish(rc);
}
