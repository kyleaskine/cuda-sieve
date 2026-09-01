#!/bin/bash
# testsieve.sh -- test-sieve a job at several q, project the whole run from the
# yield curve, and compare geometries.
#
# The GGNFS equivalent (`test_sieve.sh` in a ~/code/test-sieve tree) can only
# sweep q, because its siever is compiled per-I: 14e, 15e and 16e are three
# different binaries and J is fixed at I/2. Ours takes I and J at run time, so
# geometry is a sweepable axis here, and so are the factor-base bounds.
#
# UNITS, STATED ONCE, TWICE OVER.
#
# First, the command line. `--qmin`, `--qmax` and `--target-rels` are given in
# MILLIONS: `--qmin 20` is q = 20000000 and `--target-rels 300` is 3e8
# relations. These are the three options whose natural values run to eight and
# nine digits, and typing them out in full was both tedious and easy to get
# wrong by a factor of ten. Fractions are allowed (`--qmin 2.5`), and a value
# that still looks like an absolute count is rejected rather than silently
# multiplied. Everything else on the command line -- `--width`, `--rlim`,
# `--alim` -- is an ordinary absolute number, and so is `bench`'s own
# `--target-rels`, which is a different flag on a different program.
#
# Second, the measurement. `bench` processes **(q, rho) pairs** -- one special-q and
# one root of it. Short q intervals contain a noisy number of those pairs, so
# raw relation counts (and projections made directly from them) are misleading.
# As in ~/code/test-sieve, every sample is normalized to the expected number of
# special-q objects in its interval: width / ln(q0). The yield projection, time,
# energy, and target crossing all use those normalized samples.

set -u
usage() {
    cat <<'EOF'
usage: ./testsieve.sh --poly JOB.job [--fb1 FB] [options]
       ./testsieve.sh                       # prompt for parameters

  --poly PATH        the job file (required)
  --fb1 PATH         custom caller-managed algebraic factor base
                     [omit for checked/rebuilt fbase.mLOGI cache]
  --fb-backend B     gpu, cpu, or auto: which generator rebuilds the default
                     factor base                                        [auto]
  --fb-device N      CUDA device for the GPU factor-base generator        [0]
  --fb-threads N     CPU-backend threads when rebuilding it, 1..256  [all CPUs]
  --qmin M           band start, IN MILLIONS                       [20 = 2e7]
  --qmax M           band end for the projection, IN MILLIONS     [10 x qmin]
  --points N         sample points across the band                        [5]
  --width N          q-interval sieved at each point (absolute)        [2000]
  --target-rels M    also report where the target is met, IN MILLIONS   [off]
  --side a|r         sieve algebraic or rational side              [a]
  --sq-side 1|0      numeric alias for --side (bench convention)
  --geom "logI,J"    geometry to test; repeat for several   [15,16384]
  --rlim N/--alim N  override the job's factor-base bounds
  --extra "FLAGS"    passed through to bench verbatim
  --keep             keep the relation files and logs
  --normscan-samples N  (q,rho) drawn for the exact-norm width survey
                     that runs per geometry before sieving   [400000]
  -i, --interactive  prompt for parameters (no arguments does this too)

--qmin, --qmax and --target-rels are in MILLIONS; --width, --rlim and --alim
are absolute. `--qmin 20 --qmax 200 --target-rels 300` sieves [2e7, 2e8) and
reports where 3e8 relations are reached.

--fb-backend auto uses ./fbgen_gpu when it is already built and the CPU
./fbgen otherwise; the two produce byte-identical files, so a cache built by
either is valid for the other. Neither setting ever runs `make` -- build the
generator once with `make GPU_ARCH=native fbgen_gpu`. --fb-backend gpu differs
from auto only in refusing to fall back to the CPU generator.

The backend is resolved on the first cache that actually needs rebuilding, so a
run that finds every cache valid neither selects nor mentions a generator.

Each point sieves a WIDTH-wide q interval, so wider costs more and measures
better; the default holds roughly a hundred (q, rho) pairs at q ~ 2e7, which
is a few seconds a point. Startup (factor-base load) is excluded from every
timing: it is a one-off of seconds against a run of days.

Yield normalization follows the reference test sieve:
  expected pairs = WIDTH / ln(q0)
  n-yield        = raw relations * expected pairs / observed pairs
EOF
}

POLY=
FB=fbase
FB_MANAGED=1
FB_THREADS=${FB_THREADS:-}
FB_BACKEND=auto
FB_DEVICE=
QMIN=20
QMAX=
POINTS=5
WIDTH=2000
TARGET=
SQ_SIDE=1
RLIM=
ALIM=
EXTRA=
KEEP=0
GEOMS=()
NORMSCAN_SAMPLES=400000
NORMSCAN_FLAGGED=()
INTERACTIVE=0
# Only prompt when there is a human to prompt. With stdin closed or redirected
# -- a Makefile, cron, CI -- every `read` returns EOF instantly, so this used to
# accept ALL the built-in defaults in silence and launch real sieve runs against
# whatever input.job/fbase happened to be lying around. No arguments and no tty
# is a usage error, which is what it was before the interactive mode existed.
if [ $# -eq 0 ]; then
    if [ -t 0 ]; then INTERACTIVE=1; else usage; exit 2; fi
fi
while [ $# -gt 0 ]; do
    case "$1" in
        --poly) POLY=$2; shift 2 ;;
        --fb1) FB=$2; FB_MANAGED=0; shift 2 ;;
        --fb-threads) FB_THREADS=$2; shift 2 ;;
        --fb-backend) FB_BACKEND=$2; shift 2 ;;
        --fb-device) FB_DEVICE=$2; shift 2 ;;
        --qmin) QMIN=$2; shift 2 ;;
        --qmax) QMAX=$2; shift 2 ;;
        --points) POINTS=$2; shift 2 ;;
        --width) WIDTH=$2; shift 2 ;;
        --target-rels) TARGET=$2; shift 2 ;;
        --side)
            case "$2" in
                a|A|1|algebraic) SQ_SIDE=1 ;;
                r|R|0|rational) SQ_SIDE=0 ;;
                *) echo "side must be algebraic/a or rational/r" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --sq-side) SQ_SIDE=$2; shift 2 ;;
        --geom) GEOMS+=("$2"); shift 2 ;;
        --rlim) RLIM=$2; shift 2 ;;
        --alim) ALIM=$2; shift 2 ;;
        --extra) EXTRA=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --normscan-samples) NORMSCAN_SAMPLES=$2; shift 2 ;;
        -i|--interactive) INTERACTIVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option $1" >&2; usage; exit 2 ;;
    esac
done

prompt_value() {
    local label=$1 default=$2 answer
    read -r -p "$label [$default]: " answer
    printf '%s' "${answer:-$default}"
}

default_fb_threads() {
    local n
    n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
    [[ "$n" =~ ^[0-9]+$ ]] || n=1
    [ "$n" -le 256 ] || n=256
    printf '%s' "$n"
}

canonical_decimal() {
    local n=$1
    while [ "${#n}" -gt 1 ] && [ "${n#0}" != "$n" ]; do n=${n#0}; done
    printf '%s' "$n"
}

# Scale a millions-denominated option to its absolute value: 20 -> 20000000.
#
# Entirely string arithmetic, on purpose. The raw argument is matched against a
# literal decimal BEFORE anything expands it, for the same reason the integer
# checks below run before their $(( )): bash arithmetic expands array
# subscripts, so `--qmin 'a[$(cmd)]'` reaching a $(( )) would execute cmd. It
# also keeps 2.5 exact, which a float round-trip through awk or python would
# not guarantee for every input.
# Divide a decimal string by a million by moving the point, so the hint in the
# out-of-range message below is the operator's own number translated exactly --
# fraction included. `20000000.5` must come back as `20.0000005`, not `20`: the
# whole value of that message is that the suggestion can be pasted verbatim.
shift_point_left_6() {   # $1 = decimal string, e.g. 500000 or 20000000.5
    local v=$1 int frac
    int=${v%%.*}
    frac=""
    [ "$v" = "$int" ] || frac=${v#*.}
    while [ "${#int}" -lt 7 ]; do int="0$int"; done
    frac="${int: -6}$frac"
    int=$(canonical_decimal "${int:0:${#int}-6}")
    while [ -n "$frac" ] && [ "${frac%0}" != "$frac" ]; do frac=${frac%0}; done
    if [ -z "$frac" ]; then printf '%s' "$int"; else printf '%s.%s' "$int" "$frac"; fi
}

scale_millions() {   # $1 = option name, for the error text; $2 = raw argument
    local opt=$1 raw=$2 int frac rest hint scaled
    if ! [[ "$raw" =~ ^([0-9]+)(\.([0-9]*))?$ ]]; then
        echo "$opt takes a number in millions, e.g. 20 for 20000000" >&2
        return 1
    fi
    int=$(canonical_decimal "${BASH_REMATCH[1]}")
    frac=${BASH_REMATCH[3]}
    # Digits past the millionth have nowhere to go. Silently truncating them
    # would make 2.0000004 and 2 the same band without saying so.
    rest=${frac:6}
    if [[ "$rest" == *[1-9]* ]]; then
        echo "$opt $raw is not a whole number of units (millions have six decimal places)" >&2
        return 1
    fi
    frac=${frac:0:6}
    while [ "${#frac}" -lt 6 ]; do frac="${frac}0"; done
    scaled=$(canonical_decimal "${int}${frac}")
    # Muscle memory from before this units change is the failure mode that
    # matters: `--qmin 20000000` would silently become 2e13 and sieve a band
    # nowhere near the intended one, and every number downstream -- yield,
    # days, the target crossing -- would be a confident answer to the wrong
    # question.
    #
    # THE TEST IS ON THE SCALED VALUE, not on how many digits were typed. A
    # digit-count rule has a blind spot exactly one digit wide underneath it:
    # `--qmin 500000` is six digits, passes such a rule, and means q = 5e11 --
    # the same mistake, silently accepted. Nothing this script projects reaches
    # 1e10: q is bounded well under it by lim (a uint32), and a relation target
    # of ten billion is an order of magnitude past the largest job anyone here
    # is costing. Past that ceiling it is a units error, not a value.
    if [ "$scaled" -gt 10000000000 ]; then
        hint=$(shift_point_left_6 "$raw")
        echo "$opt $raw is in MILLIONS, so it means $scaled -- past this script's" \
             "1e10 ceiling. Did you mean $opt $hint?" >&2
        return 1
    fi
    printf '%s' "$scaled"
}

if [ "$INTERACTIVE" = 1 ]; then
    echo "Interactive test-sieve setup (press Enter to accept a default)."
    POLY=$(prompt_value "Job file" "${POLY:-input.job}")
    read -r -p "Algebraic factor base [$FB]: " answer
    if [ -n "$answer" ]; then
        FB=$answer
        # Typing a path is an explicit choice and must not authorize overwrite.
        FB_MANAGED=0
    fi
    if [ "$FB_MANAGED" = 1 ]; then
        FB_BACKEND=$(prompt_value "Factor-base generator (gpu/cpu/auto)" "$FB_BACKEND")
        [ -n "$FB_THREADS" ] || FB_THREADS=$(default_fb_threads)
        FB_THREADS=$(prompt_value "Factor-base build threads (CPU backend only)" \
                                  "$FB_THREADS")
    fi
    QMIN=$(prompt_value "Band start q, in millions" "$QMIN")
    # No arithmetic on the answers here any more. The 10x default for qmax is
    # deferred to the shared validation block below, which computes it from an
    # already-scaled qmin, so an empty answer here is not a value to fill in --
    # it is the default itself, and pressing Enter passes it straight through.
    read -r -p "Band end q, in millions [${QMAX:-10 x qmin}]: " answer
    QMAX=${answer:-$QMAX}
    POINTS=$(prompt_value "Sample points" "$POINTS")
    WIDTH=$(prompt_value "q-interval width per point (absolute)" "$WIDTH")

    read -r -p "Target relations, in millions [${TARGET:-none}]: " answer
    TARGET=${answer:-$TARGET}

    side_default=a
    [ "$SQ_SIDE" = 0 ] && side_default=r
    read -r -p "Sieve algebraic (a) or rational (r) side? [${side_default}]: " answer
    case "${answer:-$side_default}" in
        a|A|1|algebraic) SQ_SIDE=1 ;;
        r|R|0|rational) SQ_SIDE=0 ;;
        *) echo "side must be algebraic/a or rational/r" >&2; exit 2 ;;
    esac

    geom_default=${GEOMS[*]:-15,16384}
    read -r -p "Geometries, space separated [$geom_default]: " answer
    read -r -a GEOMS <<< "${answer:-$geom_default}"

    read -r -p "Rational factor-base bound [job]: " answer
    RLIM=${answer:-$RLIM}
    read -r -p "Algebraic factor-base bound [job]: " answer
    ALIM=${answer:-$ALIM}
    read -r -p "Extra bench flags [none]: " answer
    EXTRA=${answer:-$EXTRA}
    read -r -p "Keep relation files and logs? [y/N]: " answer
    case "$answer" in y|Y|yes|YES) KEEP=1 ;; esac
    echo
fi

if [ -z "$POLY" ] || [ -z "$FB" ]; then usage; exit 2; fi
[ ${#GEOMS[@]} -gt 0 ] || GEOMS=("15,16384")
case "$FB_BACKEND" in
    gpu|cpu|auto) ;;
    *) echo "fb-backend must be gpu, cpu, or auto" >&2; exit 2 ;;
esac
if [ -n "$FB_DEVICE" ] && ! [[ "$FB_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "fb-device must be a nonnegative integer" >&2; exit 2
fi
[ -z "$FB_DEVICE" ] || FB_DEVICE=$(canonical_decimal "$FB_DEVICE")
# --fb1 means the caller manages the factor base and this script never
# generates one, so every generator option is dead. Saying so is the difference
# between "I asked for the GPU and got it" and "I asked for the GPU and nothing
# was generated at all", which the exit status cannot distinguish.
if [ "$FB_MANAGED" = 0 ]; then
    [ "$FB_BACKEND" = auto ] ||
        echo "[testsieve] --fb-backend is ignored with --fb1:" \
             "a caller-managed factor base is never generated" >&2
    [ -z "$FB_DEVICE" ] ||
        echo "[testsieve] --fb-device is ignored with --fb1:" \
             "a caller-managed factor base is never generated" >&2
fi
# Scale BEFORE the $(( )) below. scale_millions validates by regex against the
# raw string, so the qmin it returns is a plain decimal integer by the time the
# 10x default multiplies it; bash arithmetic expands array subscripts, and
# `--qmin 'a[$(cmd)]'` reaching a $(( )) would execute cmd.
QMIN=$(scale_millions --qmin "$QMIN") || exit 2
if [ "$QMIN" -le 1 ]; then
    echo "qmin must be greater than 1; got $QMIN" >&2; exit 2
fi
if [ -n "$QMAX" ]; then
    QMAX=$(scale_millions --qmax "$QMAX") || exit 2
else
    QMAX=$((QMIN * 10))
fi
if [ "$QMAX" -le "$QMIN" ]; then
    echo "qmax must be greater than qmin" >&2; exit 2
fi
if ! [[ "$POINTS" =~ ^[0-9]+$ && "$WIDTH" =~ ^[0-9]+$ ]] ||
   [ "$POINTS" -eq 0 ] || [ "$WIDTH" -eq 0 ]; then
    echo "points and width must be positive integers" >&2; exit 2
fi
POINTS=$(canonical_decimal "$POINTS")
WIDTH=$(canonical_decimal "$WIDTH")
if [ -n "$TARGET" ]; then
    TARGET=$(scale_millions --target-rels "$TARGET") || exit 2
fi
for bound_name in RLIM ALIM; do
    bound=${!bound_name}
    if [ -n "$bound" ]; then
        if ! [[ "$bound" =~ ^[0-9]+$ ]]; then
            echo "${bound_name,,} must be a positive integer" >&2; exit 2
        fi
        bound=$(canonical_decimal "$bound")
        if [ "$bound" -eq 0 ]; then
            echo "${bound_name,,} must be a positive integer" >&2; exit 2
        fi
        printf -v "$bound_name" '%s' "$bound"
    fi
done
if [ "$SQ_SIDE" != 0 ] && [ "$SQ_SIDE" != 1 ]; then
    echo "sq-side must be 1 (algebraic) or 0 (rational)" >&2; exit 2
fi
NORMALIZED_GEOMS=()
for geom in "${GEOMS[@]}"; do
    if ! [[ "$geom" =~ ^[0-9]+,[0-9]+$ ]]; then
        echo "geometry must be logI,J with logI in [2,20] and positive J" >&2
        exit 2
    fi
    logI=$(canonical_decimal "${geom%%,*}")
    J=$(canonical_decimal "${geom##*,}")
    if [ "$logI" -lt 2 ] || [ "$logI" -gt 20 ] || [ "$J" = 0 ]; then
        echo "geometry must be logI,J with logI in [2,20] and positive J" >&2
        exit 2
    fi
    NORMALIZED_GEOMS+=("$logI,$J")
done
GEOMS=("${NORMALIZED_GEOMS[@]}")
[ -x ./bench ] || { echo "run me from the bench directory (no ./bench here)" >&2; exit 2; }

# The default factor base is a managed cache, like input.job.afb.0 in the
# GGNFS test sieve. Native fbgen files identify their exact polynomial, lim,
# and maxbits in the first four lines, so inspect those directly instead of
# trusting a filename. Custom --fb1 paths remain caller-managed and are never
# overwritten.
MANAGE_FB=$FB_MANAGED
declare -A FB_BY_LOGI=() SEEN_LOGI=()
for geom in "${GEOMS[@]}"; do SEEN_LOGI[${geom%%,*}]=1; done

# WHICH GENERATOR BUILDS THE CACHE. `fbgen_gpu` is the production GPU factor
# base generator -- the same engine `bench` runs in-process when --fb1 is
# omitted (bench/FBGEN_GPU.md) -- and its --out file is intended to be
# byte-identical to native CPU `fbgen` output. That byte identity is what makes
# this a backend choice rather than a format choice: the cache-header
# validation below is unchanged, and a cache built by either generator
# validates and is reused by the other, including across machines.
#
# NEITHER SETTING RUNS `make`, and the advice it prints instead has to name the
# arch THIS TREE was built for. An earlier version had --fb-backend gpu build
# the binary: a bare `make fbgen_gpu` parses the Makefile with the default
# GPU_ARCH=all, and the $(shell ...) at Makefile:124 rewrites .arch.stamp at
# parse time -- every CUDA object depends on that stamp, so a tree built
# `make GPU_ARCH=native bench` was silently invalidated and the operator paid
# for a full fat rebuild on their next make.
#
# The printed instruction had the SAME defect pointing the other way: it said
# `make GPU_ARCH=native fbgen_gpu` unconditionally, which on a default (fat)
# tree -- what `make all` produces -- rewrites the stamp just as surely.
# Verified 2026-08-27 on a fat tree: that command moved .arch.stamp from
# 86f70f29 to 882b8e71, under `make -n`, because the $(shell) runs at parse
# time whether or not anything is built.
#
# gpu_arch_arg reconstructs a GPU_ARCH whose expansion is byte-identical to the
# stamp, so the suggested command cannot move it. It returns failure rather
# than guessing when the stamp is missing or unreadable, and the caller then
# says nothing about arch at all.
#
# ONE DECISION, TWO FACTS. FB_BACKEND_USED is which binary runs now and may be
# demoted at runtime; FB_FALLBACK_OK is policy and is fixed at selection. The
# earlier code re-read the raw $FB_BACKEND at fallback time, which meant the
# resolved state and the policy could disagree -- and did: the fallback did not
# demote, so every later geometry retried a generator already known to fail.

# GPU_ARCH that reproduces the arch this tree was ALREADY built for.
# .arch.stamp holds the expanded -gencode flags, not the GPU_ARCH value, so this
# reconstructs a value whose expansion is byte-identical. Prints it (possibly
# empty, meaning "the default already matches"), or returns 1 when it cannot
# tell -- callers must then omit the arch rather than guess, because a wrong
# value rewrites the stamp and invalidates every CUDA object in the tree.
gpu_arch_arg() {
    local stamp=.arch.stamp n cc
    [ -r "$stamp" ] || return 1
    n=$(tr ' ' '\n' < "$stamp" | grep -c -- '^-gencode$')
    case "$n" in
        1)  cc=$(sed -n 's/.*compute_\([0-9][0-9]*\).*/\1/p' "$stamp")
            [ -n "$cc" ] || return 1
            printf 'GPU_ARCH=%s ' "$cc" ;;
        0)  return 1 ;;
        *)  # >1 gencode can only be the default fat list: GPU_ARCH accepts
            # all|native|<cc>, and the latter two expand to exactly one.
            printf '' ;;
    esac
}

# The build command to suggest for ./fbgen_gpu, arch-correct for this tree.
fbgen_gpu_build_hint() {
    local a
    if a=$(gpu_arch_arg); then printf 'make %sfbgen_gpu' "$a"
    else printf 'make fbgen_gpu'; fi
}

FB_BACKEND_USED=cpu
FB_FALLBACK_OK=1
FB_BACKEND_RESOLVED=0
select_fb_backend() {
    FB_BACKEND_RESOLVED=1
    case "$FB_BACKEND" in
        cpu) FB_BACKEND_USED=cpu; FB_FALLBACK_OK=1 ;;
        gpu)
            [ -x ./fbgen_gpu ] || {
                echo "--fb-backend gpu needs ./fbgen_gpu, which is not built." >&2
                echo "Build it with: $(fbgen_gpu_build_hint)   (or plain: make all)" >&2
                exit 1
            }
            FB_BACKEND_USED=gpu
            FB_FALLBACK_OK=0
            ;;
        auto)
            FB_FALLBACK_OK=1
            if [ -x ./fbgen_gpu ]; then
                FB_BACKEND_USED=gpu
            else
                FB_BACKEND_USED=cpu
                echo "[testsieve] ./fbgen_gpu is not built; using the CPU factor-base generator"
                echo "            (it is part of \`make all\`; to build just it: $(fbgen_gpu_build_hint))"
            fi
            ;;
    esac
}

# One entry point so the cache-rebuild site does not have to know which
# generator ran. fbgen_gpu stages FILE.part and only atomically replaces FILE
# on success, so a failure here cannot leave a truncated cache behind for the
# header check to accept next time.
#
# THE BACKEND IS RESOLVED HERE, on the first call, and this function is only
# called for a cache that is actually stale. Resolving it up front instead put
# a two-line "fbgen_gpu is not built" advisory at the top of every repeat run
# whose caches were all valid and whose generators were therefore never going
# to run -- build advice for a build that would not have been used.
#
# The announcement is inside this function for the same reason: printed at the
# call site it had to name a generator before one had been chosen, and could
# never reflect a fallback that happens two lines later.
build_fb() {   # $1 = maxbits (the geometry's logI), $2 = destination file
    local logI=$1 dest=$2
    [ "$FB_BACKEND_RESOLVED" = 1 ] || select_fb_backend
    if [ "$FB_BACKEND_USED" = gpu ]; then
        echo "[testsieve] rebuilding stale or missing factor base on the GPU: $dest"
        if ./fbgen_gpu --poly "$POLY" --lim "$EFFECTIVE_ALIM" \
                --maxbits "$logI" ${FB_DEVICE:+--device "$FB_DEVICE"} \
                --out "$dest"; then
            return 0
        fi
        # A card that is busy, too small, or absent should not cost the
        # operator the run: the CPU generator produces the same bytes, just
        # more slowly. An explicit --fb-backend gpu asked a question that a
        # silent CPU fallback would answer wrongly, so that one stops.
        if [ "$FB_FALLBACK_OK" = 0 ]; then
            echo "fbgen_gpu failed and --fb-backend gpu leaves no fallback" >&2
            return 1
        fi
        # Demote for good. A card that just failed will fail again, and a
        # geometry sweep would otherwise pay for one doomed CUDA context per
        # logI and print a GPU banner over CPU work every time.
        FB_BACKEND_USED=cpu
        echo "[testsieve] fbgen_gpu failed; falling back to the CPU generator" \
             "for this and every later factor base" >&2
    fi
    echo "[testsieve] rebuilding stale or missing factor base on the CPU: $dest"
    ./fbgen --poly "$POLY" --lim "$EFFECTIVE_ALIM" \
        --maxbits "$logI" --threads "$FB_THREADS" --out "$dest"
}

if [ "$MANAGE_FB" = 1 ]; then
    [ -n "$FB_THREADS" ] || FB_THREADS=$(default_fb_threads)
    if ! [[ "$FB_THREADS" =~ ^[0-9]+$ ]]; then
        echo "fb-threads must be an integer in [1,256]" >&2; exit 2
    fi
    FB_THREADS=$(canonical_decimal "$FB_THREADS")
    if [ "$FB_THREADS" -eq 0 ] || [ "$FB_THREADS" -gt 256 ]; then
        echo "fb-threads must be an integer in [1,256]" >&2; exit 2
    fi
    # The CPU generator is built unconditionally even when the GPU one will do
    # the work. It is three C files and a second of compile, it is the fallback
    # path, and the metadata probe below runs it at lim=2 -- a probe worth no
    # CUDA context, on a binary that must exist anyway.
    [ -x ./fbgen ] || make fbgen
    [ -x ./fbgen ] || { echo "could not build ./fbgen" >&2; exit 1; }

    EFFECTIVE_ALIM=$ALIM
    if [ -z "$EFFECTIVE_ALIM" ]; then
        EFFECTIVE_ALIM=$(sed -nE \
            's/\r$//; s/^[[:space:]]*alim[[:space:]]*:[[:space:]]*([0-9]+)[[:space:]]*$/\1/p' \
            "$POLY" | tail -1)
    fi
    if ! [[ "$EFFECTIVE_ALIM" =~ ^[0-9]+$ ]] || [ "$EFFECTIVE_ALIM" -lt 2 ]; then
        echo "the default factor base needs --alim or an alim: value in $POLY" >&2
        exit 2
    fi
    EFFECTIVE_ALIM=$(canonical_decimal "$EFFECTIVE_ALIM")

    fb_header_err=$(mktemp)
    if ! expected_fb_text=$(./fbgen --poly "$POLY" --lim 2 --maxbits 1 \
            --threads 1 2>"$fb_header_err"); then
        cat "$fb_header_err" >&2
        rm -f "$fb_header_err"
        echo "could not derive factor-base metadata from $POLY" >&2
        exit 1
    fi
    rm -f "$fb_header_err"
    mapfile -t EXPECTED_FB_HEADER < <(printf '%s\n' "$expected_fb_text" | sed -n '1,2p')
    if [ ${#EXPECTED_FB_HEADER[@]} -ne 2 ]; then
        echo "could not derive factor-base metadata from $POLY" >&2; exit 1
    fi

    for logI in "${!SEEN_LOGI[@]}"; do
        cache_fb="${FB}.m${logI}"
        FB_BY_LOGI[$logI]=$cache_fb

        # `-n 1,4p` alone still reads the entire ~200 MB text factor base.
        # Quit after the metadata so cache validation stays constant-time.
        mapfile -t actual_header < <(sed -n '1,4p;4q' "$cache_fb" 2>/dev/null)
        cache_valid=1
        [ ${#actual_header[@]} -eq 4 ] || cache_valid=0
        if [ "$cache_valid" = 1 ]; then
            [ "${actual_header[0]}" = "${EXPECTED_FB_HEADER[0]}" ] || cache_valid=0
            [ "${actual_header[1]}" = "${EXPECTED_FB_HEADER[1]}" ] || cache_valid=0
            [ "${actual_header[2]}" = "# lim = $EFFECTIVE_ALIM" ] || cache_valid=0
            [ "${actual_header[3]}" = "# maxbits = $logI" ] || cache_valid=0
        fi

        if [ "$cache_valid" = 1 ]; then
            echo "[testsieve] factor-base cache valid: $cache_fb"
        else
            build_fb "$logI" "$cache_fb" || exit 1
        fi
    done
    echo
fi

TMP=$(mktemp -d)
# Announce the path when keeping: mktemp -d names are random, nothing else in
# the output contains one (the per-point messages print basenames), so a kept
# directory was previously findable only by hunting /tmp by mtime.
# shellcheck disable=SC2317  # invoked indirectly by trap
cleanup() {
    if [ "$KEEP" = 1 ]; then echo "[testsieve] kept relation files and logs in $TMP"
    else rm -rf "$TMP"; fi
}
trap cleanup EXIT

# Echo the ABSOLUTE band, not the millions that were typed. The command line
# is in millions for the typing; the run is in q, and every other number below
# -- q0, exp-rel, the target crossing -- is absolute. Printing the scaled value
# back is also the cheapest confirmation that the units landed as intended.
echo "[testsieve] $(basename "$POLY")  band [$QMIN, $QMAX)  $POINTS points x ${WIDTH}-wide q intervals"
echo "            yields normalized to WIDTH/ln(q0) expected (q, rho) pairs"
echo "            sieve side: $([ "$SQ_SIDE" = 1 ] && echo algebraic || echo rational)"
[ -n "$RLIM$ALIM" ] && echo "            lim override: rlim=${RLIM:-job} alim=${ALIM:-job}"
echo

ROWS="$TMP/rows.txt"; : > "$ROWS"

# Device memory for the geometry just sampled.
#
# WHY IT BELONGS HERE. This script answers "should I commit days to this job?",
# and "will it fit on the card?" is half of that question. Memory is a function
# of geometry and lims, not of q, so it is constant across a geometry's samples
# and belongs in that block's footer rather than as a tenth column.
#
# THE HEADLINE IS THE STEADY-STATE TOTAL, and the breakdown is explicitly NOT a
# sizing figure. RUNBOOK "Do not size a job from an aborted startup" has said so
# since before this footer existed: the per-stage table covers only what is
# allocated during setup, and per-q buffers plus CUDA's lazily-reserved
# per-kernel local memory arrive afterwards. On the c183 at 15e that gap is
# 1.22 of 3.28 GB -- 37% of the job, most of it real job memory rather than
# context. An earlier version of this footer called the setup sum "this job"
# and told the operator to plan with it, which would have under-sized every
# job by roughly that much. The breakdown is here to show WHICH KNOB moves
# memory (bucket array vs factor base), not how much to budget.
#
# Contention shows up as disagreement between samples. Each stage figure is a
# difference of two free-memory probes, so a neighbour allocating or freeing
# between them lands in whichever stage straddled it -- observed at 1.57 GB for
# a cofactor queue that is a fixed 0.15 GB. The samples are all the same
# geometry, so their totals must agree; when they do not, the card was busy and
# neither number should be trusted.
report_memory() {   # $@ = this geometry's bench output files
    awk '
      FNR == 1 { stage_done = (nstage > 0) }
      /device memory by stage/ { inb = 1; next }
      inb && / GB   \(/ {
          if (stage_done) next
          for (i = 1; i <= NF; i++) if ($i == "GB") { v = $(i-1); break }
          lbl = ""; for (j = 1; j < i-1; j++) lbl = lbl (j > 1 ? " " : "") $j
          # A negative delta means a neighbour FREED between the two probes.
          # Report it rather than summing it into a smaller-looking job.
          if (v + 0 < 0) bad = 1
          parts = parts (nstage++ ? ", " : "") lbl " " v
          sum += v
          next
      }
      /steady state/ {
          inb = 0
          u = $5 + 0; card = $10; f = $12; gsub(/\(/, "", f)
          if (nuse == 0 || u < lo) lo = u
          if (nuse == 0 || u > hi) hi = u
          nuse++; last_free = f
      }
      END {
          if (nuse == 0) { print "  memory: not reported (no sample completed)"; exit }
          spread = hi - lo
          if (spread > 0.05)
              printf "  memory: %.2f-%.2f GB in use of %s GB across %d samples --" \
                     " THE CARD WAS BUSY, re-measure idle\n", lo, hi, card, nuse
          else
              printf "  memory: %.2f GB in use of %s GB, %s free   <- size from this\n",
                     hi, card, last_free    
          if (nstage == 0 || bad || sum <= 0 || sum > hi)
              print "          setup breakdown unavailable or inconsistent" \
                    " (a memory probe failed, or the card was busy)"
          else {
              printf "          setup %.2f GB = %s\n", sum, parts
              printf "          (+%.2f GB of per-q buffers and CUDA context after these" \
                     " marks; do not size from the setup figure)\n", hi - sum
          }
      }' "$@"
}

for geom in "${GEOMS[@]}"; do
    logI=${geom%%,*}; J=${geom##*,}
    geom_fb=$FB
    [ "$MANAGE_FB" = 0 ] || geom_fb=${FB_BY_LOGI[$logI]}
    area=$(python3 -c "import math;print('2^%.4g'%math.log2((1<<$logI)*$J))")
    printf '  --- logI %s, J %s  (area %s) ---\n' "$logI" "$J" "$area"
    # Exact-norm width for THIS geometry over the WHOLE projected band, before
    # any sieving. Per geometry because the answer moves with it -- the 2,1139+
    # octic needs 250 bits at 15e and 257 at 16e -- and over the whole band
    # because the overflowing lattices are a ~1e-5 tail: the per-point sieve
    # windows below are far too short to contain one, and a client in a group
    # sieve is shorter still. This is the moment the width can still be changed,
    # since the fix is a rebuild and every client has to carry it.
    if [ -x ./normscan ]; then
        NS_OUT=$(./normscan --poly "$POLY" --sq-side "$SQ_SIDE" \
                     --logI "$logI" --J "$J" --qmin "$QMIN" --qmax "$QMAX" \
                     --samples "$NORMSCAN_SAMPLES" 2>&1)
        NS_RC=$?
        echo "$NS_OUT" | sed 's/^/  /'
        # ONLY 0 IS A PASS. 2 = will overflow, 3 = too little margin, 1 = the
        # survey could not run (bad poly, no special-q in range, out of memory),
        # 64 = bad usage. `-le 1` treated 1 as a pass, so a normscan that failed
        # to load the polynomial certified the width silently -- the exact
        # outcome this block exists to prevent. An error is "not surveyed", not
        # "surveyed and fine".
        #
        # None of them aborts the test sieve: the geometry sweep is still worth
        # having, and refusing to measure would hide the yield numbers that say
        # whether this geometry is even the one to rebuild for. Recorded and
        # repeated at the end.
        if [ "$NS_RC" -ne 0 ]; then
            ns_why=$(echo "$NS_OUT" | grep -oE '(REFUSE|WARNING).*' | head -1)
            [ -n "$ns_why" ] || ns_why="normscan could not survey this geometry (exit $NS_RC)"
            NORMSCAN_FLAGGED+=("logI $logI, J $J: $ns_why")
        fi
        echo
    fi
    printf '  %-12s %-8s %-9s %-10s %-12s %-9s %-8s %-8s %-8s\n' \
           q0 pairs exp-pairs n-yield exp-rel rel/pair ms/pair rel/s board_W
    PREV="$TMP/prev.txt"
    MEM_SRCS=()
    : > "$PREV"
    for i in $(seq 0 $((POINTS - 1))); do
        # Linear spacing. Yield falls smoothly with q, so the trapezoid rule
        # over evenly spaced points is the right shape; log spacing would
        # oversample the cheap end where the curve is flattest.
        #
        # The span is (QMAX - QMIN - WIDTH) so that the LAST window ends at
        # QMAX rather than starting there: spacing over the full span put the
        # final sample -- a fifth of the evidence, and the lowest-yield one --
        # entirely outside the interval being integrated.
        q0=$(python3 -c "print(int($QMIN + max(0,$QMAX - $QMIN - $WIDTH) * $i / max(1,$POINTS-1)))")
        # bench's qrange upper bound is inclusive, so subtract one to sieve
        # exactly WIDTH integers and keep the normalization denominator exact.
        q1=$((q0 + WIDTH - 1))
        tag="g${logI}_${J}_$i"
        # shellcheck disable=SC2086
        if ! ./bench --pipeline --cofactor --poly "$POLY" --fb1 "$geom_fb" \
            --sq-side "$SQ_SIDE" \
            --logI "$logI" --J "$J" --qrange "$q0:$q1" \
            ${RLIM:+--rlim $RLIM} ${ALIM:+--alim $ALIM} $EXTRA \
            --relations "$TMP/$tag.dat" --log "$TMP/$tag.log" --log-every 1 \
            > "$TMP/$tag.out" 2>&1; then
            printf '  %-12s FAILED: %s\n' "$q0" \
              "$(grep -iE 'error|cannot|refus|does not fit' "$TMP/$tag.out" | head -1 | cut -c1-56)"
            continue
        fi
        # EVERY sample, not just the first. They are the same geometry, so
        # they must report the same memory; disagreement is the contention
        # signal, and it is free to collect since the runs already happened.
        grep -q 'steady state' "$TMP/$tag.out" && MEM_SRCS+=("$TMP/$tag.out")
        pairs=$(grep -oP -- '--- band of \K[0-9]+' "$TMP/$tag.out" | head -1)
        rel=$(grep -oP 'total relations\s+\K[0-9]+' "$TMP/$tag.out" | tail -1)
        # COMPLETE, not the plain 'wall clock per q'. The plain line excludes
        # cofac_tail, the final flush after the band -- and a sample this short
        # never reaches an in-loop flush (CQ_FLUSH is 131072 candidates, ~67
        # special-q), so on these windows that tail IS the whole
        # cofactorisation. Reading the exclusive number understated the C194
        # projection by about 30%. Falls back for a run without --cofactor.
        ms=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$TMP/$tag.out" | head -1)
        [ -n "$ms" ] || ms=$(grep -oP 'wall clock per q\s+\K[0-9.]+' "$TMP/$tag.out" | head -1)
        bw=$(grep -oP 'board=\K[0-9.]+' "$TMP/$tag.log" 2>/dev/null |
             awk '{s+=$1;n++} END{if(n)printf "%.1f",s/n}')
        if [ -z "${pairs:-}" ] || [ -z "${rel:-}" ] || [ -z "${ms:-}" ]; then
            # Silence here would drop a point from the integration while the
            # banner still promised N of them, and the flat extrapolation would
            # quietly cover the gap.
            printf '  %-12s UNPARSED: band summary incomplete in %s\n' \
                   "$q0" "$tag.out"
            continue
        fi
        python3 - "$q0" "$WIDTH" "$pairs" "$rel" "$ms" "${bw:-0}" \
            "$logI,$J" "$ROWS" "$PREV" <<'EOF'
import math
import sys
q0,w,pairs,rel,ms,bw,geom,rows,prev_file = sys.argv[1:]
q0,w,pairs,rel,ms,bw = int(q0),int(w),int(pairs),int(rel),float(ms),float(bw)
relpair = rel/pairs if pairs else 0.0
expected = w/math.log(q0)
nrel = relpair*expected
nsecs = expected*ms/1000.0
rels    = relpair/(ms/1000.0) if ms else 0.0
exp_rel = ""
previous = open(prev_file).read().split()
if previous:
    prev_q0, prev_nrel = int(previous[0]), float(previous[1])
    exp_rel = "%.0f" % (((prev_nrel + nrel) / 2.0) * (q0 - prev_q0) / w)
print("  %-12d %-8d %-9.1f %-10.1f %-12s %-9.2f %-8.2f %-8.0f %-8s" %
      (q0, pairs, expected, nrel, exp_rel, relpair, ms, rels,
       ("%.1f"%bw) if bw else "n/a"))
open(rows,"a").write("%s %d %d %d %.9f %.9f %.6f %.3f\n" %
                     (geom,q0,w,pairs,expected,nrel,nsecs,bw))
open(prev_file,"w").write("%d %.9f\n" % (q0,nrel))
EOF
    done
    [ ${#MEM_SRCS[@]} -eq 0 ] || report_memory "${MEM_SRCS[@]}"
    echo
done

projection_status=0
python3 - "$QMIN" "$QMAX" "${TARGET:-0}" "$ROWS" <<'EOF' || projection_status=$?
import sys
qmin, qmax, target, rows = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
data = {}
for ln in open(rows):
    g,q0,w,pairs,expected,nrel,nsecs,bw = ln.split()
    data.setdefault(g, []).append((int(q0), int(w), int(pairs), float(expected),
                                   float(nrel), float(nsecs), float(bw)))
if not data:
    print("  no usable points"); raise SystemExit(1)

print("  === projection over [%d, %d) ===" % (qmin, qmax))
print("  %-12s %-14s %-12s %-12s %-10s %s" %
      ("geometry", "n-relations", "GPU days", "energy kWh", "rel/kJ", "target reached at q"))
best = None
for g, pts in data.items():
    pts.sort()
    # Trapezoid over normalized relations and seconds per unit q. Normalizing
    # both removes accidental prime/root-density swings from short samples.
    rel_dens  = [(q, nrel / w) for q, w, p, expected, nrel, nsecs, bw in pts]
    time_dens = [(q, nsecs / w) for q, w, p, expected, nrel, nsecs, bw in pts]
    pw = [bw for q, w, p, expected, nrel, nsecs, bw in pts if bw > 0]
    watts = sum(pw) / len(pw) if pw else 0.0

    def integrate(dens, lo, hi):
        """Trapezoid, extrapolating flat outside the sampled range."""
        tot = 0.0
        xs = [d[0] for d in dens]
        for i in range(len(dens) - 1):
            (xa, ya), (xb, yb) = dens[i], dens[i + 1]
            a, b = max(lo, xa), min(hi, xb)
            if b <= a: continue
            fa = ya + (yb - ya) * (a - xa) / (xb - xa)
            fb = ya + (yb - ya) * (b - xa) / (xb - xa)
            tot += (fa + fb) / 2 * (b - a)
        if lo < xs[0]:  tot += dens[0][1]  * (min(hi, xs[0]) - lo)
        if hi > xs[-1]: tot += dens[-1][1] * (hi - max(lo, xs[-1]))
        return tot

    rels = integrate(rel_dens, qmin, qmax)
    secs = integrate(time_dens, qmin, qmax)
    kwh  = watts * secs / 3.6e6
    relkj = rels / (watts * secs / 1000.0) if watts and secs else 0.0

    tq = "-"
    if target:
        lo, hi = qmin, qmax
        if rels < target:            # already integrated over [qmin, qmax)
            tq = "not within band"
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                if integrate(rel_dens, qmin, mid) < target: lo = mid
                else: hi = mid
            d = integrate(time_dens, qmin, hi) / 86400.0
            tq = "%.0f  (%.2f d)" % (hi, d)
    print("  %-12s %-14s %-12.2f %-12.1f %-10.1f %s" %
          (g, "%.3g" % rels, secs / 86400.0, kwh, relkj, tq))
    if watts > 0 and (best is None or relkj > best[1]): best = (g, relkj)

if len(data) > 1:
    if best:
        print("\n  best relations per kilojoule: %s" % best[0])
    else:
        # Every relkj is 0 because no board-power sample arrived (no NVML, no
        # sensor). Picking a winner here would be picking whichever geometry
        # was measured first and calling it an energy result.
        print("\n  no board-power samples: the energy columns are not meaningful")
EOF

if [ "${#NORMSCAN_FLAGGED[@]}" -gt 0 ]; then
    echo
    echo "  *** EXACT-NORM WIDTH: these geometries were NOT cleared for this binary:"
    for f in "${NORMSCAN_FLAGGED[@]}"; do echo "      $f"; done
    echo "  *** Such special-q are SKIPPED with a warning, and 100 skips ends the"
    echo "  *** run: a REFUSE here costs relations at best and a stopped band at"
    echo "  *** worst. The fix is a rebuild, so it has to happen before work is"
    echo "  *** distributed -- a client cannot rebuild itself."
    [ "$projection_status" -ne 0 ] || projection_status=4
fi
echo
echo "--- $(basename "$POLY") (source job plus selected sieve side) ---"
cat -- "$POLY" || [ "$projection_status" -ne 0 ] || projection_status=1
if [ "$SQ_SIDE" = 1 ]; then
    # Match the GGNFS test sieve's result.job display without modifying the
    # user's input file or leaving a generated job in the worktree.
    if [ -s "$POLY" ] && [ -n "$(tail -c1 -- "$POLY")" ]; then echo; fi
    echo "lss: 0"
fi
[ -z "$RLIM" ] || echo "# test-sieve override: rlim=$RLIM"
[ -z "$ALIM" ] || echo "# test-sieve override: alim=$ALIM"
[ -z "$EXTRA" ] || echo "# test-sieve extra flags: $EXTRA"
echo
exit "$projection_status"
