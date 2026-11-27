/* tunedtest.c -- the saved-tuning file's rules, card-free (make tunedcheck).
 *
 * Every rule bench applies to a saved entry is in tuned.c, so this gates all
 * of them without a GPU: the format and its version line, the allowlist and
 * its ranges, the key, each family's explicit-flag rule, the declines (policy
 * either way, J, region), the notes (factor-base bounds, special-q side) and
 * the override classification. Exits nonzero if any check fails. */
#define _POSIX_C_SOURCE 200809L
#include "tuned.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { fails++;               \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__);      \
        printf("\n"); } } while (0)

static const char *DIR_;

static const char *write_file(const char *name, const char *text)
{
    static char path[4][512];
    static int k;
    char *p = path[k++ & 3];
    FILE *f;
    snprintf(p, 512, "%s/%s", DIR_, name);
    f = fopen(p, "w");
    if (!f) { perror(p); exit(2); }
    fputs(text, f);
    fclose(f);
    return p;
}

static int count_lines(FILE *f)
{
    int c, n = 0;
    rewind(f);
    while ((c = fgetc(f)) != EOF) n += c == '\n';
    return n;
}

#define CARD "NVIDIA GeForce RTX 3060"
#define HDR  TUNED_FORMAT_LINE "\n"
/* The policy these entries are written at (this binary's, as tune.sh writes
 * it) and one either side, for the declines. */
#define POL     2
#define POL_OLD 1
#define POL_NEW 3
_Static_assert(POL == TUNED_POLICY && POL_OLD == POL - 1 && POL_NEW == POL + 1,
               "tunedtest's policy literals follow TUNED_POLICY");
#define STR_(x) #x
#define STR(x)  STR_(x)
/* a valid entry for the run below, with FLAGS and POLICY substituted */
#define ENTRY(J, REGION, POLICY, FLAGS) \
    CARD "|8.6|28|16|" #J "|" #REGION "|4|1|300000000|200000000|" STR(POLICY) "|" FLAGS "|tune.sh 2026-11-16 57e9d88 nfs.job -23.4%\n"

static tuned_run_t base_run(void)
{
    tuned_run_t r;
    memset(&r, 0, sizeof r);
    r.card = CARD; r.cc_major = 8; r.cc_minor = 6; r.sms = 28; r.logI = 16;
    r.J = 32768; r.region = 14; r.side = 1;
    r.alim = 300000000u; r.rlim = 200000000u;
    r.pipeline = 1; r.q_pipeline = 1; r.small_sieve = 1; r.fill_gate = 1;
    r.fill_merge = 0;          /* I16: no row tier, so no merge */
    r.ss_block_cut = 36; r.ss_warp_cut = 256;
    r.max_grid = 2147483647;
    return r;
}

static void flags_tests(void)
{
    tuned_knobs_t k;
    char err[200], buf[256];
    static const char *const BAD[] = {
        "--fill-blocks 0", "--fill-blocks 1048577", "--fill-blocks", "--fill-blocks x",
        "--fill-blocks 9216 --fill-blocks 9216", "--ss-block-cut 32", "--ss-block-cut 129",
        "--ss-warp-cut 33", "--ss-warp-cut 32769", "--ss-block-cut 64 --ss-warp-cut 64",
        "--fill-gate-shift 6", "--fill-gate-shift -1", "--slab-fill-blocks 384",
        "--no-slab-fill-ahead --slab-fill-blocks 384", "--slab-fill-ahead --no-slab-fill-ahead",
        "--no-fill-merge --no-fill-merge", "--fill-threads 64", "--q-prep td",
        "--fill-merge", "--fill-merge-cut 32", "fill-blocks 9216", NULL
    };
    CHECK(!tuned_parse_flags("", &k, err, sizeof err) && !k.fill_blocks &&
          k.fill_gate_shift == -1 && k.slab_fill_ahead == -1, "empty flags");
    CHECK(!tuned_parse_flags("  --fill-blocks 589824  ", &k, err, sizeof err) &&
          k.fill_blocks == 589824, "--fill-blocks 589824: %s", err);
    CHECK(!tuned_parse_flags("--fill-blocks 1048576", &k, err, sizeof err), "cap is legal");
    CHECK(!tuned_parse_flags("--slab-fill-blocks 384 --slab-fill-ahead --fill-gate-shift 0"
                             " --ss-warp-cut 512 --ss-block-cut 48 --no-fill-merge",
                             &k, err, sizeof err) &&
          k.slab_fill_ahead == 1 && k.slab_fill_blocks == 384 && k.fill_gate_shift == 0 &&
          k.ss_warp_cut == 512 && k.ss_block_cut == 48 && k.no_fill_merge == 1,
          "every knob, any order: %s", err);
    tuned_format_flags(&k, buf, sizeof buf);
    CHECK(!strcmp(buf, "--ss-block-cut 48 --ss-warp-cut 512 --fill-gate-shift 0"
                       " --no-fill-merge --slab-fill-ahead --slab-fill-blocks 384"),
          "canonical form: '%s'", buf);
    CHECK(!tuned_parse_flags("--no-slab-fill-ahead", &k, err, sizeof err) &&
          k.slab_fill_ahead == 0, "--no-slab-fill-ahead");
    for (int i = 0; BAD[i]; i++)
        CHECK(tuned_parse_flags(BAD[i], &k, err, sizeof err) == -1, "refused: %s", BAD[i]);
}

static void line_tests(void)
{
    tuned_entry_t e;
    char err[200];
    char line[600];
    snprintf(line, sizeof line, "%s", ENTRY(32768, 14, POL, "--fill-blocks 589824"));
    line[strlen(line) - 1] = 0;
    CHECK(!tuned_parse_line(line, &e, err, sizeof err), "valid line: %s", err);
    CHECK(!strcmp(e.card, CARD) && e.cc_major == 8 && e.cc_minor == 6 && e.sms == 28 &&
          e.logI == 16 && e.J == 32768 && e.region == 14 && e.nslab == 4 && e.side == 1 &&
          e.alim == 300000000u && e.rlim == 200000000u && e.policy == POL &&
          e.k.fill_blocks == 589824 && !strcmp(e.flags, "--fill-blocks 589824") &&
          !strcmp(e.note, "tune.sh 2026-11-16 57e9d88 nfs.job -23.4%"), "fields");
    CHECK(!tuned_parse_line(CARD "|8.6|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "||a|b|c",
                            &e, err, sizeof err) && !strcmp(e.note, "a|b|c") && !e.flags[0],
          "empty flags, a note with bars: %s", err);
    CHECK(tuned_parse_line(CARD "|8.6|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-blocks 9216",
                           &e, err, sizeof err) == -1, "12 fields refused");
    CHECK(tuned_parse_line("|8.6|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "||n", &e, err, sizeof err) == -1,
          "empty card refused");
    CHECK(tuned_parse_line(CARD "|86|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "||n", &e, err, sizeof err) == -1,
          "cc without a dot refused");
    CHECK(tuned_parse_line(CARD "|8.6|28|16|32768|14|4|2|300000000|200000000|" STR(POL) "||n", &e, err, sizeof err) == -1,
          "side 2 refused");
    CHECK(tuned_parse_line(CARD "|8.6|28|16|32768|14|4|1|3e8|200000000|" STR(POL) "||n", &e, err, sizeof err) == -1,
          "alim 3e8 refused");
    CHECK(!tuned_parse_line(CARD "|8.6|28|16|32768|14|4|1|4294967295|3000000000|" STR(POL) "||n", &e, err, sizeof err) &&
          e.alim == 4294967295u && e.rlim == 3000000000u, "alim and rlim up to 2^32-1, as bench takes them: %s", err);
    CHECK(tuned_parse_line(CARD "|8.6|28|16|32768|14|4|1|4294967296|200000000|" STR(POL) "||n", &e, err, sizeof err) == -1,
          "alim 2^32 refused");
    CHECK(tuned_parse_line(CARD "|8.6|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-threads 64|n",
                           &e, err, sizeof err) == -1, "a knob outside the allowlist voids the line");
}

static void argv_tests(void)
{
    char buf[512];
    char *a1[] = { "bench", "--pipeline", "--cofactor", "--poly", "x.job", "--logI", "16",
                   "--maxbits", "16", "--qrange", "50000000:", "--nq", "100",
                   "--relations", "r", "--log", "l", "--tuned", "t", "--no-tuned",
                   "--device", "0", NULL };
    char *a2[] = { "bench", "--fill-blocks", "0", NULL };
    char *a3[] = { "bench", "--fill-merge-cut", "32", "--no-smallsieve", "--fill-concurrent",
                   "--slab-fill-blocks", "384", NULL };
    char *a4[] = { "bench", "--pipeline", "--sieve-skip", "1", "--no-row-cap", "--fill-blocks",
                   "9216", "--ecm-b1-0", "100", "--ecm-b1", "200", "--no-row-cap",
                   "--slab-fill-ahead", "--fill-merge-cut", "8", "--no-tiny-fold",
                   "--td-record-scalar", "--q-prep", "td", NULL };
    CHECK(tuned_explicit_mask(22, a1) == 0, "a plain run sets no family");
    tuned_overrides(22, a1, buf, sizeof buf);
    CHECK(!buf[0], "a plain run has no overrides: '%s'", buf);
    CHECK(tuned_explicit_mask(3, a2) == TUNED_F_FILL_BLOCKS, "--fill-blocks 0 is explicit");
    CHECK(tuned_explicit_mask(7, a3) == (TUNED_F_MERGE | TUNED_F_SS | TUNED_F_GATE_SHIFT | TUNED_F_SFA),
          "family members");
    tuned_overrides(19, a4, buf, sizeof buf);
    CHECK(!strcmp(buf, "--sieve-skip --no-row-cap --ecm-b1-0 --ecm-b1 --fill-merge-cut"
                       " --no-tiny-fold --td-record-scalar --q-prep"),
          "overrides, each once, knobs left out: '%s'", buf);
}

static void judge_tests(void)
{
    tuned_entry_t e;
    tuned_verdict_t v;
    tuned_run_t r;
    char err[200], line[600];
#define LOAD(J, REGION, POLICY, FLAGS) do {                                     \
        snprintf(line, sizeof line, "%s", ENTRY(J, REGION, POLICY, FLAGS));      \
        line[strlen(line) - 1] = 0;                                              \
        if (tuned_parse_line(line, &e, err, sizeof err)) {                       \
            printf("FAIL setup: %s\n", err); fails++; return; } } while (0)

    r = base_run();
    LOAD(32768, 14, POL, "--fill-blocks 589824");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_APPLY && v.apply.fill_blocks == 589824 &&
          !strcmp(v.applied, "--fill-blocks 589824") && !v.notes[0], "applies: '%s'", v.notes);

    LOAD(32768, 14, POL_OLD, "--fill-blocks 589824");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && strstr(v.why, "policy " STR(POL_OLD) " and this binary's is " STR(POL)),
          "an older policy declined");
    LOAD(32768, 14, POL_NEW, "--fill-blocks 589824");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && strstr(v.why, "policy " STR(POL_NEW)), "a newer policy declined");
    LOAD(16384, 14, POL, "--fill-blocks 589824");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && strstr(v.why, "J 16384"), "another J declined");
    LOAD(32768, 13, POL, "--fill-blocks 589824");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && strstr(v.why, "region 13"), "another region declined");

    /* A knob given by hand sets the whole entry aside (finding 161: a saved
     * --slab-fill-ahead beside a hand-given --fill-blocks 589824 lost). */
    LOAD(32768, 14, POL, "--fill-blocks 589824");
    r.explicit_mask = TUNED_F_FILL_BLOCKS;
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && !v.applied[0] && strstr(v.why, "sets --fill-blocks") &&
          strstr(v.why, "whole"), "an explicit --fill-blocks sets the entry aside: '%s'", v.why);
    LOAD(32768, 14, POL, "--slab-fill-ahead --slab-fill-blocks 384");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && v.apply.slab_fill_ahead == -1,
          "an explicit --fill-blocks sets aside a saved slab fill ahead too");
    r.explicit_mask = TUNED_F_FILL_BLOCKS | TUNED_F_SS;
    tuned_judge(&e, &r, &v);
    CHECK(strstr(v.why, "sets --fill-blocks, the small-sieve cuts,"), "every family given is named: '%s'", v.why);
    LOAD(32768, 14, POL, "");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_APPLY && !v.applied[0], "an empty entry is not set aside");
    LOAD(32768, 14, POL, "--fill-blocks 589824");
    r = base_run(); r.max_grid = 65535;
    tuned_judge(&e, &r, &v);
    CHECK(!v.apply.fill_blocks && strstr(v.notes, "grid limit"), "a grid above the card's limit is skipped");

    r = base_run(); r.fill_merge = 1;
    LOAD(32768, 14, POL, "--no-fill-merge");
    tuned_judge(&e, &r, &v);
    CHECK(v.apply.no_fill_merge == 1, "--no-fill-merge applies where the merge runs");
    r.explicit_mask = TUNED_F_MERGE;          /* e.g. --fill-merge-cut 32 */
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && !v.apply.no_fill_merge && strstr(v.why, "the fill merge"),
          "--fill-merge-cut keeps the merge against a saved --no-fill-merge");
    r = base_run();
    tuned_judge(&e, &r, &v);
    CHECK(!v.apply.no_fill_merge && strstr(v.notes, "no merge"), "no merge, nothing to turn off");

    r = base_run();
    LOAD(32768, 14, POL, "--fill-gate-shift 2");
    tuned_judge(&e, &r, &v);
    CHECK(v.apply.fill_gate_shift == 2, "gate shift applies with the gate");
    r.fill_gate = 0;
    tuned_judge(&e, &r, &v);
    CHECK(v.apply.fill_gate_shift == -1 && strstr(v.notes, "gate is off"), "no gate, no shift");

    r = base_run();
    LOAD(32768, 14, POL, "--slab-fill-ahead --slab-fill-blocks 384");
    tuned_judge(&e, &r, &v);
    CHECK(v.apply.slab_fill_ahead == 1 && v.apply.slab_fill_blocks == 384, "slab fill ahead applies");
    r.q_pipeline = 0;
    tuned_judge(&e, &r, &v);
    CHECK(v.apply.slab_fill_ahead == -1 && strstr(v.notes, "--q-pipeline"), "needs the q pipeline");
    r = base_run(); r.explicit_mask = TUNED_F_SFA;
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_DECLINED && v.apply.slab_fill_ahead == -1, "an explicit --slab-fill-blocks wins");

    r = base_run();
    LOAD(32768, 14, POL, "--ss-block-cut 48");
    r.ss_warp_cut = 40;                        /* an effective warp cut of 40 */
    tuned_judge(&e, &r, &v);
    CHECK(!v.apply.ss_block_cut && strstr(v.notes, "warp 40"), "a pair that would fail warp > block is skipped");
    r = base_run(); r.small_sieve = 0;
    tuned_judge(&e, &r, &v);
    CHECK(!v.apply.ss_block_cut && strstr(v.notes, "no small sieve"), "no small sieve");
    r = base_run();
    tuned_judge(&e, &r, &v);
    CHECK(v.apply.ss_block_cut == 48 && !v.apply.ss_warp_cut, "the block cut alone applies");

    r = base_run(); r.alim = 60000000u; r.side = 0;
    LOAD(32768, 14, POL, "--fill-blocks 589824");
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_APPLY && v.apply.fill_blocks == 589824 &&
          strstr(v.notes, "alim 300000000") && strstr(v.notes, "side 1"),
          "FB bounds 5x away and the other side: applied, with notes: '%s'", v.notes);
    r = base_run(); r.rlim = 500000000u;
    tuned_judge(&e, &r, &v);
    CHECK(strstr(v.notes, "rlim 200000000"), "rlim 2.5x above: a note: '%s'", v.notes);
    r = base_run(); r.alim = 120000000u;
    tuned_judge(&e, &r, &v);
    CHECK(strstr(v.notes, "alim 300000000"), "alim 2.5x below: a note: '%s'", v.notes);
    r = base_run(); r.alim = 150000000u; r.rlim = 400000000u;
    tuned_judge(&e, &r, &v);
    CHECK(!v.notes[0], "exactly 2x away: no note: '%s'", v.notes);

    LOAD(32768, 14, POL, "");
    r = base_run();
    tuned_judge(&e, &r, &v);
    CHECK(v.status == TUNED_APPLY && !v.applied[0], "an empty entry keeps the defaults");
#undef LOAD
}

static void lookup_tests(void)
{
    tuned_verdict_t v;
    tuned_run_t r = base_run();
    FILE *w = tmpfile();
    const char *p;
    if (!w) { perror("tmpfile"); exit(2); }

    CHECK(tuned_lookup("/nonexistent/tuned.txt", &r, w, &v) == TUNED_NOFILE, "no file");
    p = write_file("empty.txt", "");
    CHECK(tuned_lookup(p, &r, w, &v) == TUNED_BADFILE && strstr(v.why, "empty"), "empty file");
    p = write_file("v2.txt", "# cuda-sieve tuned v2\n" ENTRY(32768, 14, POL, "--fill-blocks 9216"));
    CHECK(tuned_lookup(p, &r, w, &v) == TUNED_BADFILE && strstr(v.why, "first line"),
          "another format version");
    p = write_file("noheader.txt", ENTRY(32768, 14, POL, "--fill-blocks 9216"));
    CHECK(tuned_lookup(p, &r, w, &v) == TUNED_BADFILE, "no version line");

    p = write_file("other.txt", HDR "# a comment\n\n"
        "NVIDIA GeForce RTX 3060 Ti|8.6|38|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-blocks 1|n\n"
        CARD "|8.9|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-blocks 2|n\n"
        CARD "|8.6|30|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-blocks 3|n\n"
        CARD "|8.6|28|15|16384|14|1|1|300000000|200000000|" STR(POL) "|--fill-blocks 4|n\n");
    CHECK(tuned_lookup(p, &r, w, &v) == TUNED_NONE, "the key is card, cc, SMs and logI");

    rewind(w);
    if (ftruncate(fileno(w), 0)) {}
    p = write_file("mixed.txt", "\n" HDR "# comment\r\n"
        "garbage line\n"
        CARD "|8.6|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-threads 64|bad knob\n"
        CARD "|8.6|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-blocks 294912|first\r\n"
        CARD "|8.6|28|16|32768|14|4|1|300000000|200000000|" STR(POL) "|--fill-blocks 589824|second\n");
    CHECK(tuned_lookup(p, &r, w, &v) == TUNED_APPLY && v.apply.fill_blocks == 589824 &&
          !strcmp(v.e.note, "second") && v.e.line == 7, "the last entry for a key wins, CRLF ok");
    CHECK(count_lines(w) == 3, "two malformed lines and one duplicate warned about (%d)", count_lines(w));
    fclose(w);
}

/* The built-in defaults by card: exactly the architectures timed, and the
 * fill grid at logI 16 only. */
static void card_tests(void)
{
    static const struct { int maj, min, logI, warp, fill, sfa, sp; } t[] = {
        {  8, 9, 16, 1024, 589824, 0, 0 },   /* Ada at the NFS@Home shape */
        {  8, 9, 15, 1024,      0, 0, 0 },   /* C183 at logI 15 lost the grid */
        {  8, 9, 17, 1024,      0, 0, 0 },
        {  8, 9, 14, 1024,      0, 0, 0 },
        {  8, 6, 16,    0, 589824, 0, 0 },   /* the 3060 */
        {  8, 6, 15,    0,      0, 0, 0 },
        { 12, 0, 16,    0,      0, 1, 1 },   /* Blackwell: 5070, 5060 */
        { 12, 0, 13,    0,      0, 1, 1 },   /* bench adds the plan's conditions */
        { 12, 1, 16,    0,      0, 0, 0 },   /* untimed: sm_121 */
        {  8, 0, 16,    0,      0, 0, 0 },   /* untimed: A100 */
        {  8, 7, 16,    0,      0, 0, 0 },   /* untimed: Orin */
        {  9, 0, 16,    0,      0, 0, 0 },   /* untimed: Hopper */
        {  7, 5, 16,    0,      0, 0, 0 },   /* untimed: Turing */
        { 10, 0, 16,    0,      0, 0, 0 },   /* untimed: B200 */
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        tuned_card_defaults_t d;
        memset(&d, 0x5a, sizeof d);   /* every field must be written */
        tuned_card_defaults(t[i].maj, t[i].min, t[i].logI, &d);
        CHECK(d.ss_warp_cut == t[i].warp && d.fill_blocks == t[i].fill &&
              d.slab_fill_ahead == t[i].sfa && d.sparse_apply == t[i].sp,
              "card defaults sm_%d%d logI %d: warp %d fill %d sfa %d sp %d, want %d %d %d %d",
              t[i].maj, t[i].min, t[i].logI, d.ss_warp_cut, d.fill_blocks,
              d.slab_fill_ahead, d.sparse_apply, t[i].warp, t[i].fill, t[i].sfa, t[i].sp);
    }
    CHECK(TUNED_CARD_SS_WARP_CUT_SM89 > TUNED_SS_BLOCK_MAX &&
          TUNED_CARD_SS_WARP_CUT_SM89 <= TUNED_SS_WARP_MAX,
          "the sm_89 warp cut is above every block cut and within bounds");
    CHECK(TUNED_CARD_FILL_BLOCKS <= TUNED_BLOCKS_MAX, "the fill grid is within bounds");
}

static void path_tests(void)
{
#ifndef _WIN32
    char buf[512];
    setenv("HOME", "/home/u", 1);
    setenv("XDG_CONFIG_HOME", "/cfg", 1);
    CHECK(!tuned_default_path(buf, sizeof buf) && !strcmp(buf, "/cfg/cuda-sieve/tuned.txt"),
          "XDG_CONFIG_HOME: %s", buf);
    setenv("XDG_CONFIG_HOME", "relative", 1);
    CHECK(!tuned_default_path(buf, sizeof buf) && !strcmp(buf, "/home/u/.config/cuda-sieve/tuned.txt"),
          "a relative XDG_CONFIG_HOME is ignored: %s", buf);
    unsetenv("XDG_CONFIG_HOME");
    CHECK(!tuned_default_path(buf, sizeof buf) && !strcmp(buf, "/home/u/.config/cuda-sieve/tuned.txt"),
          "HOME: %s", buf);
    unsetenv("HOME");
    CHECK(tuned_default_path(buf, sizeof buf) == -1, "neither");
#endif
}

int main(void)
{
    char dir[] = "/tmp/tunedtest.XXXXXX";
    const char *t = getenv("TMPDIR");
    static char tdir[600];
    if (t && *t) { snprintf(tdir, sizeof tdir, "%s/tunedtest.XXXXXX", t); DIR_ = mkdtemp(tdir); }
    else DIR_ = mkdtemp(dir);
    if (!DIR_) { perror("mkdtemp"); return 2; }
    flags_tests();
    line_tests();
    argv_tests();
    judge_tests();
    lookup_tests();
    path_tests();
    card_tests();
    {
        char cmd[700];
        snprintf(cmd, sizeof cmd, "rm -rf '%s'", DIR_);
        if (system(cmd)) {}
    }
    printf("tunedtest: %d of %d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
