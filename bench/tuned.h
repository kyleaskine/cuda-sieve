/* tuned.h -- per-card saved tuning: the file tune.sh writes and bench reads.
 *
 * tune.sh times the output-identical speed knobs on one card for one job and
 * saves the winner here; bench applies it on that card at that logI without
 * being told. Every knob in the file changes no output (tune.sh byte-compares
 * the relations of every run it times), so a stale or wrong entry costs speed
 * and never relations. That is what lets the rules below be lenient: a saved
 * value that does not fit a run is skipped with a note, never refused.
 *
 * The file is plain text. Its first non-empty line is TUNED_FORMAT_LINE;
 * lines starting with # are comments; each other line is one entry of 13
 * '|'-separated fields:
 *
 *   card|cc|SMs|logI|J|region|nslab|side|alim|rlim|policy|flags|note
 *
 *   card    cudaDeviceProp.name, exactly            } the key: one entry per
 *   cc      compute capability, "8.6"              } card per logI (Kyle,
 *   SMs     multiprocessor count                   } 2026-11-17)
 *   logI    the sieve width
 *   J, region   the run's; an entry is declined where either differs
 *   nslab   slabs per q of the plan it was tuned on; a different class (one
 *           vs several) gets a re-tune note at the plan (run_pipeline)
 *   side    the special-q side; a difference gets a note
 *   alim, rlim  the tuned job's; more than 2x away gets a note
 *   policy  TUNED_POLICY when written; an entry is declined unless equal
 *   flags   the knobs, as bench flags; TUNED_ALLOWED only
 *   note    provenance (date, commit, job, gain); the rest of the line
 *
 * An empty flags field is an entry too: the card was tuned at that logI and
 * the defaults won. */
#ifndef CUDA_SIEVE_TUNED_H
#define CUDA_SIEVE_TUNED_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TUNED_FORMAT_LINE "# cuda-sieve tuned v1"

/* Bump when a change to a default or a kernel makes entries tuned before it
 * untrustworthy. Entries must match it EXACTLY: a newer binary declines an
 * older policy's entry, and a downgraded binary a newer one's.
 *   1  2026-11-17, saved tuning's start
 *   2  2026-11-24, the per-card defaults below (finding 162): an entry holds
 *      only the flags that differ from the defaults, so on sm_86, sm_89 and
 *      sm_120 a policy-1 entry would sit on top of defaults it was never
 *      timed beside, and a policy-2 entry under an older binary on its old
 *      ones. tunedtest.c's POL follows this. */
#define TUNED_POLICY 2

/* The knobs' ranges, mirrored from bench.h (which this header must not
 * include: tuned.c is plain C) and static_asserted equal in bench_main.cu. */
#define TUNED_BLOCKS_MAX     (1 << 20)   /* BENCH_BLOCKS_MAX */
#define TUNED_SS_BLOCK_MIN   33          /* SS_BLOCK_CUT_MIN */
#define TUNED_SS_BLOCK_MAX   128         /* SS_BLOCK_CUT_MAX */
#define TUNED_SS_WARP_MAX    32768       /* SS_WARP_CUT_MAX */

/* Built-in defaults by card. Not saved tuning, but the same per-card
 * question, kept here (plain C) so tunedtest can check the rule on every
 * compute capability without the cards (STATUS row 44, 2026-11-24). Each is
 * what tune.sh found on every card of that architecture it timed, and every
 * knob is output-identical, so on a card not yet timed the worst case is
 * speed, which tune.sh wins back. bench applies them to --pipeline runs only,
 * like saved tuning (the harness reproduces RESULTS numbers), and a flag or
 * an applied saved entry still wins.
 *   sm_89 (Ada): --ss-warp-cut 1024. Never slower than 256 on six jobs on
 *     two cards (4070 SUPER: C143 -0.95% to C194 -6.7%; 4060 Ti: the
 *     NFS@Home I16 job -4.4%, C194 -5.8%), and the 4080 SUPER -4.2% on the
 *     I16 job, where 2048 was best (-5.3%); the 3060 and four Blackwell
 *     cards keep 256.
 *   sm_86 and sm_89 at logI 16: --fill-blocks 589824. The 3060 -18.5 to
 *     -23.4% on three logI 16 jobs; Ada -4.2 to -9.6% on four runs (three
 *     cards; the 4080 SUPER -7.9%).
 *     The one Ada loss was C183 at logI 15 (+2.08%), hence logI 16 only;
 *     the 5060, 5070 and 5070 Ti lose with it; the 5080 gains 1.6% there,
 *     less than its 2.2% at 36864.
 *   sm_120 (Blackwell), any SM count: --slab-fill-ahead on, at bench.h's
 *     SLAB_FILL_BLOCKS_DEFAULT per SLAB_FILL_DEFAULT_SMS SMs: the 5070 -7.2%
 *     (finding 160), the 5060 -4.44% at 384 on 30 SMs, the 5070 Ti -3.86%
 *     at 512 and 768 on 70 SMs held at 1.58 GHz by its host (1024 -2.61%;
 *     896 untimed), and on the 5080 the rule's 1075 (84 SMs) was best or
 *     level (off +1.41%, 512 +1.90%, 1536 +1.29%); Ada and the 3060 lose
 *     with it beside their best fill grid.
 *   sm_120 (Blackwell): --sparse-apply on (finding 164: RTX 5070 -3.8% to
 *     -12.8% across all 8 benchmark jobs); off elsewhere until timed on
 *     Ada/Ampere/Turing.
 * 0 means bench.h's constant, as everywhere else. */
#define TUNED_CARD_SS_WARP_CUT_SM89  1024
#define TUNED_CARD_FILL_BLOCKS       589824
#define TUNED_CARD_FILL_LOGI         16
typedef struct {
    int ss_warp_cut;       /* 0 = SS_WARP_CUT_DEFAULT */
    int fill_blocks;       /* 0 = FILL_BLOCKS_DEFAULT (k_fill_atomic's grid) */
    int slab_fill_ahead;   /* 1 = on by default; bench_main adds the run's conditions */
    int sparse_apply;      /* 1 = on by default on sm_120; bench_main adds conditions */
} tuned_card_defaults_t;
void tuned_card_defaults(int cc_major, int cc_minor, int logI,
                         tuned_card_defaults_t *d);

/* Families of knobs, with the flags that ask for a setting of one: a
 * --fill-merge-cut asks for the merge as surely as --fill-merge does. A
 * saved entry applies only where the command line gives no flag of any
 * family. Its knobs were confirmed together and interact (finding 161), so
 * beside one given by hand none of it applies (tuned_judge). */
enum {
    TUNED_F_FILL_BLOCKS = 1u << 0,  /* --fill-blocks (0 too: auto, asked for) */
    TUNED_F_SS          = 1u << 1,  /* --ss-block-cut --ss-warp-cut --no-smallsieve */
    TUNED_F_GATE_SHIFT  = 1u << 2,  /* --fill-gate-shift --fill-gate --no-fill-gate --fill-concurrent */
    TUNED_F_MERGE       = 1u << 3,  /* --fill-merge --no-fill-merge --fill-merge-cut */
    TUNED_F_SFA         = 1u << 4   /* --slab-fill-ahead --no-slab-fill-ahead --slab-fill-blocks */
};

typedef struct {
    int fill_blocks;       /* 0 = not set */
    int ss_block_cut;      /* 0 = not set */
    int ss_warp_cut;       /* 0 = not set */
    int fill_gate_shift;   /* -1 = not set */
    int no_fill_merge;     /* 1 = --no-fill-merge */
    int slab_fill_ahead;   /* -1 = not set, 0 = off, 1 = on */
    int slab_fill_blocks;  /* 0 = not set; only with slab_fill_ahead 1 */
} tuned_knobs_t;

typedef struct {
    char     card[128];
    int      cc_major, cc_minor, sms, logI;
    uint32_t J;
    int      region, nslab, side;
    uint32_t alim, rlim;
    int      policy;
    tuned_knobs_t k;
    char     flags[256];   /* canonical, rebuilt from k */
    char     note[256];
    unsigned line;         /* in the file */
} tuned_entry_t;

/* What the saved knobs are judged against: the run's key, its
 * applicability record, and the resolved state each knob depends on. */
typedef struct {
    const char *card;
    int      cc_major, cc_minor, sms, logI;
    uint32_t J;
    int      region, side;
    uint32_t alim, rlim;
    int      pipeline, q_pipeline, small_sieve, fill_gate, fill_merge;
    int      ss_block_cut, ss_warp_cut;   /* effective: explicit or default */
    int      max_grid;                    /* cudaDeviceProp.maxGridSize[0] */
    unsigned explicit_mask;               /* TUNED_F_* given on the command line */
} tuned_run_t;

enum {
    TUNED_NOFILE = 0,  /* no file at the path */
    TUNED_BADFILE,     /* unreadable, or not this format */
    TUNED_NONE,        /* a file, and no entry for this key */
    TUNED_DECLINED,    /* an entry, declined as a whole (why) */
    TUNED_APPLY        /* an entry; apply holds what fits this run */
};

typedef struct {
    int           status;
    tuned_entry_t e;
    tuned_knobs_t apply;
    char          applied[256];  /* canonical flags of apply; "" if none */
    char          notes[768];    /* "; "-separated */
    char          why[256];      /* TUNED_DECLINED / TUNED_BADFILE */
} tuned_verdict_t;

/* The default path: $XDG_CONFIG_HOME/cuda-sieve/tuned.txt, else
 * $HOME/.config/cuda-sieve/tuned.txt; %APPDATA%\cuda-sieve\tuned.txt on
 * Windows. -1 when the environment names none. */
int  tuned_default_path(char *buf, size_t n);

/* Parse a flags field into k. -1 with err on anything outside the allowlist,
 * a value out of range, a contradiction, or --slab-fill-blocks without
 * --slab-fill-ahead. */
int  tuned_parse_flags(const char *flags, tuned_knobs_t *k, char *err, size_t n);
void tuned_format_flags(const tuned_knobs_t *k, char *buf, size_t n);

/* One entry line (no newline). -1 with err if malformed. */
int  tuned_parse_line(const char *line, tuned_entry_t *e, char *err, size_t n);

/* The command line, scanned: which families were given, and which flags are
 * neither a family's nor in TUNED_BASE_OK (an override: the run's workload is
 * not the plain one entries are tuned on). Tokens that do not start with "--"
 * are values and skipped. */
unsigned tuned_explicit_mask(int argc, char *const argv[]);
void     tuned_overrides(int argc, char *const argv[], char *buf, size_t n);

/* Read path and judge the entry for run's key. Malformed lines are warned
 * about on warn (if not NULL) as PATH:LINE: and skipped; a later duplicate of
 * a key replaces an earlier one, with a warning. Returns v->status. */
int  tuned_lookup(const char *path, const tuned_run_t *run, FILE *warn,
                  tuned_verdict_t *v);

/* The rules alone, for an entry already parsed (tuned_lookup calls it). */
void tuned_judge(const tuned_entry_t *e, const tuned_run_t *run,
                 tuned_verdict_t *v);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SIEVE_TUNED_H */
