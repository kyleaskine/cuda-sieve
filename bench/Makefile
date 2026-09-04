# Standalone bucket-fill benchmark for the GPU NFS sieve feasibility probe.
# No CADO, no msieve, no GMP. Builds or loads factor bases directly.
# Pattern copied from ~/msieve-s/gerbicz_bench/Makefile, which worked well.

# Declared here, at the top, because it must not end up attached to whatever
# rule happens to come first. Without it GNU make takes the default goal from
# the first explicit rule in the file -- `$(OBJS) $(TEST_OBJS): $(DEFS_STAMP)
# ...` further down -- whose first word is bench_main.o. A bare `make` then
# compiled ONE object file, printed nothing alarming and exited 0, while
# `bench` kept whatever binary was already on disk. That is indistinguishable
# from a successful incremental build unless you check the binary's mtime, and
# it silently invalidates any A/B measured right after it: the "new" binary is
# the old one. Found 2026-08-25 (RESULTS finding 75), where it turned a real
# -12.6% apply win into a flat result that was very nearly written off.
.DEFAULT_GOAL := all

NVCC      ?= nvcc
CC        ?= gcc
CXX       ?= g++

# sm_120 native (RTX 5070, Blackwell consumer), sm_86 native (RTX 3090/A10).
# PTX fallback is compute_80, NOT compute_89: the driver only JITs PTX to a
# target >= the virtual arch, so a compute_89 fallback cannot load on any
# Ampere card at all -- the previous default was unrunnable on a 3090.
# sm_80 is native, not PTX: the A100 is a comparison target (RESULTS.md finding
# 50), and JIT-ing the whole module from PTX on first load costs seconds on a
# cold ~/.nv/ComputeCache -- far more than the ~170-220 ms of module load that
# finding 48's --reps 100 floor is calibrated against, and it recurs every
# process start if the cache is disabled or evicted.
#
# sm_80 is also the floor, and it is a tuning floor rather than an instruction
# one. The only warp intrinsics used are __syncwarp, __shfl{,_up}_sync and
# __ballot_sync (k_apply's threshold scan), which all long predate Ampere --
# but k_fill_l1's, k_fill_l2's AND k_apply's __launch_bounds__(512, 3) target
# 1536 threads/SM and 3 x 33 KB of shared memory, which are Ampere-and-later
# tuning numbers. k_apply acquired that annotation on 2026-08-25 (RESULTS
# finding 75); its opt-in per-block shared-memory LIMIT is still queried from
# the selected device at runtime rather than hardcoded, but the three-block
# occupancy target it now carries is not. sm_75 still compiles with ptxas
# warning that .minnctapersm is out of range on every fill kernel -- and now on
# k_apply too, Turing offering 1024 threads/SM against the annotation's 1536 --
# and then oversubscribes Turing's 64 KB of shared memory in the fill path.
# Below that nothing is possible anyway: CUDA 13 dropped Volta, so nvcc 13.2
# rejects sm_70 outright.
GPU_ARCH_all := -gencode arch=compute_120,code=sm_120 \
                -gencode arch=compute_90,code=sm_90 \
                -gencode arch=compute_89,code=sm_89 \
                -gencode arch=compute_86,code=sm_86 \
                -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_80,code=compute_80

# `make GPU_ARCH=native` builds only for the card in this machine,
# `make GPU_ARCH=86` cross-builds for a named one (bare compute capability, no
# "sm_" and no dot), and the default builds the portable fat binary.
#
# Named GPU_ARCH, not ARCH: ARCH is exported by kernel-build and cross-compile
# shells and by some CI images, and make imports the environment, so an inherited
# ARCH=x86_64 would silently retarget the build to compute_x86_64.
#
# Measured on bench_kernels.cu, which is the entire build -- bench_main.cu is
# 2.5 s and the C files are noise. Re-measured 2026-08-20; the numbers below
# REPLACE a table taken 2026-08-11, which predated the 4-limb/ECM cofactor
# dispatch and understated a default build by 3.5x. See STATUS.md item 16.
#
#                              CF_LMAX=3   CF_LMAX=4 (default)
#     sm_120 alone                 277 s         754 s
#     sm_80 alone                   15 s          26 s
#     all six targets, -t 0           --         750 s   <- the default
#
# Two conclusions, and only the second one changed. ptxas on sm_120 is ~29x
# slower than on sm_80 and swamps everything else, so the six-target fat build
# costs no more than its slowest target alone (750 s vs 754 s): -t 0 is doing
# its job, adding sm_90 was free, and dropping targets buys nothing. Keep the
# fat binary here. GPU_ARCH=native is the real win on an Ampere or Ada host,
# where it drops sm_120 and takes the build to well under a minute.
#
# What changed is the CAUSE. The 4th cofactor limb costs sm_120 2.72x, and
# that -- not the target list -- is why a full build now takes ~12.5 min. For
# iteration on a Blackwell host, CF_LMAX=3 is 277 s and still shippable (see
# the CF_LMAX block below). Why ptxas scales this way is unsolved; STATUS.md
# item 16 carries the open experiment.
GPU_ARCH ?= all

ifeq ($(GPU_ARCH),all)
NVCC_ARCH := $(GPU_ARCH_all)
else
ifeq ($(GPU_ARCH),native)
# Detected explicitly rather than via nvcc's -arch=native, which does not fail
# when it cannot see a GPU: it warns and silently builds for the toolkit default,
# yielding a binary that will not load on the card you meant to target.
#
# The digit check is not paranoia. nvidia-smi prints its errors to STDOUT, not
# stderr -- an old driver that does not know the compute_cap query field answers
# `Field "compute_cap" is not a valid field to query.` on fd 1, which sails past
# 2>/dev/null and past any test for emptiness, and would land inside the gencode
# flag. Evaluated here rather than at file scope so that `make clean`, `make
# fbgen` and the default fat build never spawn nvidia-smi at all; probing the
# driver is not free while a job is running (finding 53).
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                  | head -1 | tr -d .)
GPU_CC_VALID := $(shell echo '$(GPU_CC)' | grep -qxE '[0-9]+' && echo yes)
ifneq ($(GPU_CC_VALID),yes)
$(error GPU_ARCH=native: no usable compute capability from nvidia-smi (got "$(GPU_CC)"). Pass GPU_ARCH=<cc>, e.g. GPU_ARCH=86, or GPU_ARCH=all)
endif
NVCC_ARCH := -gencode arch=compute_$(GPU_CC),code=sm_$(GPU_CC)
else
GPU_ARCH_VALID := $(shell echo '$(GPU_ARCH)' | grep -qxE '[0-9]+' && echo yes)
ifneq ($(GPU_ARCH_VALID),yes)
$(error GPU_ARCH must be all, native, or a bare compute capability like 86 or 120 -- got "$(GPU_ARCH)". Not sm_86, not 8.6)
endif
NVCC_ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
endif
endif

# Objects carry no record of which -gencode built them, so switching GPU_ARCH
# leaves them "up to date" and the next link silently mixes or mistargets: build
# once with GPU_ARCH=native on the Ada box, then plain `make` for the portable
# binary, and you get sm_89-only objects in a binary that dies on the A100 with
# "no kernel image is available for execution on the device" -- on the machine
# you cannot debug on, while working fine on the build host. Stamp the arch and
# depend on it. The $(shell) rewrites the file only when the value changed, so
# its mtime moves only then and repeat builds stay incremental.
ARCH_STAMP := .arch.stamp
$(shell [ "$$(cat $(ARCH_STAMP) 2>/dev/null)" = '$(NVCC_ARCH)' ] || printf '%s' '$(NVCC_ARCH)' > $(ARCH_STAMP))

# Extra -D's for pricing experiments, e.g. make DEFS=-DNORM_CANCEL_TOL=0.0f or
# make DEFS=-DNORM_FAST_LOG2. Nothing in the default build sets this; it exists
# so a variant can be timed without editing a header and forgetting to put it
# back.
#
# STAMPED, for the same reason GPU_ARCH and HAVE_BOINC are: objects carry no
# record of the -D's that built them, so `make DEFS=-DNORM_FAST_LOG2` on an
# existing tree would report everything up to date and link the ACCURATE
# binary -- and the measurement would then report 0% for a switch that was
# never compiled in. The stamp also reaches the run log, since a pricing build
# otherwise stamps the same git description as a shipping one and its relations
# would be indistinguishable from real output.
DEFS      ?=

# Widest cofactor the build carries, in 32-bit limbs: 3 = 96-bit residuals,
# 4 = 128-bit. See "cofactor WIDTH" in bench.h.
#
# ITS OWN VARIABLE, NOT A DEFS VALUE, and that distinction is the point. Any
# non-empty DEFS marks a PRICING build: bench refuses to emit relations from
# one, because the -D's it exists for change the norm or the survivor bound and
# a relation file would not say so. CF_LMAX is categorically different -- it
# adds or removes a template instantiation and the cofcheck gate pins the
# output of a 3-limb and a 4-limb run as byte-identical -- so a CF_LMAX=3
# binary is shippable and must not be branded unshippable. Riding on DEFS made
# `make DEFS=-DCF_LMAX=3` build a binary that refused --relations, which is how
# this was found.
#
# Stamped like GPU_ARCH and DEFS, for the same reason: objects carry no record
# of the -D's that built them, so switching the width on an existing tree would
# report everything up to date and link the previous binary.
CF_LMAX ?= 4
ifeq ($(filter 3 4,$(CF_LMAX)),)
$(error CF_LMAX must be 3 (96-bit cofactors) or 4 (128-bit) -- got "$(CF_LMAX)")
endif
CF_LMAX_STAMP := .cflmax.stamp
$(shell [ "$$(cat $(CF_LMAX_STAMP) 2>/dev/null)" = '$(CF_LMAX)' ] || printf '%s' '$(CF_LMAX)' > $(CF_LMAX_STAMP))
CF_LMAX_DEF := -DCF_LMAX=$(CF_LMAX)

# Width of the exact-norm big integer, in 32-bit limbs. 12 = 384 bits. 8 (256)
# was the previous default and covers the quintic/sextic corpus with ~28 bits
# to spare, but a large octic needs 232-292 depending on the q-lattice and dies
# mid-band without them. See "WHY 384 BITS" in bigint.cuh for the measurement
# and for what the extra limbs cost.
#
# ITS OWN VARIABLE, NOT A DEFS VALUE, for the reason spelled out for CF_LMAX
# above: DEFS marks a PRICING build and bench refuses to emit relations from
# one. This changes only the declared width of a magnitude type -- every value
# that fit the narrower build is represented identically in the wider one, and
# trial division loops on bn_top rather than on BN_LIMBS -- so a wider binary
# is shippable and must produce byte-identical relations on any job that ran
# in the narrower one. That equality is the A/B gate for this knob.
#
# Even values only: cf_bn_divmod_u64 steps the limb array in 64-bit pairs.
BN_LIMBS ?= 12
ifeq ($(filter 4 6 8 10 12 14 16,$(BN_LIMBS)),)
$(error BN_LIMBS must be an even limb count in 4..16 -- got "$(BN_LIMBS)")
endif
BN_LIMBS_STAMP := .bnlimbs.stamp
$(shell [ "$$(cat $(BN_LIMBS_STAMP) 2>/dev/null)" = '$(BN_LIMBS)' ] || printf '%s' '$(BN_LIMBS)' > $(BN_LIMBS_STAMP))
CF_LMAX_DEF += -DBN_LIMBS=$(BN_LIMBS)

# Optional BOINC application build.  Keep BOINC dependencies out of the default
# executable; HAVE_BOINC=1 adds the compile-time API gate and the libraries.
# BOINC_CPPFLAGS should point at directories containing boinc_api.h and its
# companion headers (usually .../include/boinc, or BOINC's api and lib trees).
HAVE_BOINC     ?= 0
BOINC_CPPFLAGS ?=
BOINC_LIBS     ?= -lboinc_api -lboinc -lpthread
# Distribution binaries carry their C++ and GCC support runtimes. These flags
# apply only to the BOINC link; set BOINC_HOST_STATIC= to opt out explicitly.
BOINC_HOST_STATIC ?= -static-libstdc++ -static-libgcc
ifeq ($(HAVE_BOINC),1)
BOINC_DEFS := -DHAVE_BOINC
BENCH_LIBS := $(BOINC_LIBS)
BENCH_LINKFLAGS := $(foreach flag,$(BOINC_HOST_STATIC),-Xcompiler=$(flag))
# A BOINC application binary is distributed to CPUs other than the build host.
# Do not bake the build machine's ISA into the host-only C/C++ objects.
HOST_TUNE ?=
else
BOINC_DEFS :=
BENCH_LIBS :=
BENCH_LINKFLAGS :=
HOST_TUNE ?= -march=native
endif

# The CUDA runtime is linked statically in EVERY build, not only BOINC ones. A
# distributed binary cannot rely on a matching libcudart on the volunteer's
# host, and a local build that links it dynamically fails to start wherever the
# toolkit's lib directory is not on the loader path -- a plain `make && ./bench`
# on a stock CUDA install. It costs ~1 MB against a binary the fat gencode
# already dominates. Pass CUDART_LINK=shared to opt out.
CUDART_LINK ?= static

# -MMD -MP: generate header dependencies. Without them the pattern rules below
# listed only bench.h and plattice.cuh, so editing bigint.cuh, prp.cuh or
# td.cuh left a stale bench_kernels.o and the binary silently did not contain
# the change under test.
# -t 0: run the per-target ptxas passes concurrently, one thread per CPU. The
# gencode list above is otherwise walked serially, which is where the bulk of a
# fat build goes; see the timing table. Harmless when GPU_ARCH picks one target.
NVCC_FLAGS = $(NVCC_ARCH) -t 0 -O3 -std=c++17 -lineinfo -MMD -MP \
             -Xcompiler "-O3 -Wall" $(DEFS) $(CF_LMAX_DEF) $(BOINC_DEFS) $(BOINC_CPPFLAGS)
CFLAGS     = -O3 $(HOST_TUNE) -Wall -Wextra -std=c11 -MMD -MP \
             $(DEFS) $(CF_LMAX_DEF) $(BOINC_DEFS)
CXXFLAGS   = -O3 $(HOST_TUNE) -Wall -Wextra -std=c++17 -MMD -MP \
             $(DEFS) $(CF_LMAX_DEF) $(BOINC_DEFS) $(BOINC_CPPFLAGS)

CU_SRCS  = bench_main.cu bench_kernels.cu
C_SRCS   = fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c platform.c
CPP_SRCS = boinc_support.cpp

CU_OBJS  = $(CU_SRCS:.cu=.o)
C_OBJS   = $(C_SRCS:.c=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
# runlog.o is deliberately NOT in C_SRCS: fbtest links C_OBJS, and it is a
# CPU-only gate that must keep building on a machine with no driver library.
OBJS     = $(CU_OBJS) $(C_OBJS) $(CPP_OBJS) runlog.o fbgen_lib.o fbgen_gpu_lib.o
TEST_OBJS = slabtest.o

# Switching HAVE_BOINC on an existing tree must rebuild boinc_support.o; make
# otherwise sees the same source timestamp and quietly reuses the no-op object.
# Include the BOINC include/link settings and host tuning in the signature so a
# changed deployment build cannot reuse objects produced for another one.
BOINC_STAMP := .boinc.stamp
BOINC_SIGNATURE := HAVE_BOINC=$(HAVE_BOINC)|BOINC_CPPFLAGS=$(BOINC_CPPFLAGS)|BOINC_LIBS=$(BOINC_LIBS)|BOINC_HOST_STATIC=$(BOINC_HOST_STATIC)|HOST_TUNE=$(HOST_TUNE)
$(shell [ "$$(cat $(BOINC_STAMP) 2>/dev/null)" = '$(BOINC_SIGNATURE)' ] || printf '%s' '$(BOINC_SIGNATURE)' > $(BOINC_STAMP))

# The commit the binary was built from, stamped into the run log's header so a
# log can be traced back to a source tree (STATUS.md item 12b). --dirty matters
# more than the hash: an uncommitted tree is exactly the case where the hash
# alone would be a lie. Stamped like the two above so that moving HEAD rebuilds
# it -- and ONLY runlog.o depends on it, which is why the string lives in that
# 200-line object and not in bench_main.cu: a commit must not cost a recompile
# of a CUDA translation unit. Unlike the nvidia-smi probe further up, `git
# describe` is local, cheap, and safe to run while a job is sieving.
#
# --dirty alone is not enough: git considers a tree with only UNTRACKED files
# clean, so a build from commit X plus a new uncommitted source stamps a bare
# X and the log's central claim becomes false in exactly the case the flag was
# added for. The porcelain check covers that; it lists untracked files but not
# ignored ones, so build products do not trip it.
#
# Kept as three plain assignments: a `case` inside $(shell ...) ends the make
# function at its first `*)`, and `wc -l` rather than the file list keeps a
# filename containing `$` or a paren out of a macro definition.
#
# The -f test is load-bearing and is NOT redundant with the content compare.
# DEFS is empty in every default build, and `cat` on a missing file also yields
# empty, so a content-only check reports "unchanged" and never creates the
# stamp -- while $(OBJS) still depends on it, leaving every fresh clone and
# every `make clean && make` dead at "No rule to make target '.defs.stamp'".
# The four sibling stamps never hit this because none of their values can be
# empty. Test existence first, then content.
DEFS_STAMP := .defs.stamp
$(shell { [ -f $(DEFS_STAMP) ] && [ "$$(cat $(DEFS_STAMP))" = '$(DEFS)' ]; } || \
        printf '%s' '$(DEFS)' > $(DEFS_STAMP))
# Every object AND the two tools that compile straight from .c to a binary
# without going through $(OBJS). Those two escaped the stamp, so
# `make DEFS=... dumpcmp` reported "up to date" and linked a binary built
# without the define -- the exact hole the stamp exists to close.
$(OBJS) $(TEST_OBJS): $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)

GIT_STAMP := .git.stamp
GIT_RAW  := $(shell git -C . describe --always --abbrev=8 2>/dev/null)
GIT_DIRT := $(shell git -C . status --porcelain 2>/dev/null | wc -l)
GIT_DESC := $(if $(GIT_RAW),$(GIT_RAW)$(if $(filter-out 0,$(GIT_DIRT)),-dirty),unknown)
$(shell [ "$$(cat $(GIT_STAMP) 2>/dev/null)" = '$(GIT_DESC)' ] || printf '%s' '$(GIT_DESC)' > $(GIT_STAMP))

runlog.o: CFLAGS += -DBENCH_GIT_DESC='"$(GIT_DESC)"' -DBENCH_DEFS='"$(DEFS)"'
runlog.o: $(GIT_STAMP)

# CUDART_LINK changes the link line and nothing else, so it gets its own stamp.
# Folding it into BOINC_SIGNATURE would make every object depend on it, and
# `make CUDART_LINK=shared` would recompile the whole fat CUDA build (214-309 s
# by the table above) to change one linker argument.
LINK_STAMP := .link.stamp
LINK_SIGNATURE := CUDART_LINK=$(CUDART_LINK)
$(shell [ "$$(cat $(LINK_STAMP) 2>/dev/null)" = '$(LINK_SIGNATURE)' ] || printf '%s' '$(LINK_SIGNATURE)' > $(LINK_STAMP))

# fbgen_gpu IS in `all`, and normscan with it. Both were left out on the
# reasoning that bench already compiles fbgen_gpu.cu as fbgen_gpu_lib.o, so
# building the standalone tool compiles that translation unit twice -- true, but
# it reasons about SERIAL cost, and the second compile is not on the critical
# path. Measured 2026-08-27, clean tree, -j8:
#
#     make -j8 all              667.1 s
#     make -j8 all fbgen_gpu    648.6 s     <- marginal cost -18.5 s, i.e. noise
#
# fbgen_gpu.o is ~105 s and disappears inside bench_kernels.o's ~750 s ptxas
# window. Leaving it out bought nothing and cost every fresh clone the slower
# factor-base path (10.6 s vs 3.4 s at alim 200M, 16 threads vs the GPU), plus a
# testsieve message telling the operator to run `make GPU_ARCH=native
# fbgen_gpu`, which on a default fat tree rewrites .arch.stamp and invalidates
# every CUDA object -- the exact hazard that message exists to avoid.
all: bench fbtest fbgen fbgen_gpu sqgentest normscan dumpcmp mkcofbatch

# -ldl for runlog.o, which dlopen()s libnvidia-ml at run time rather than
# linking it: the heartbeat's GPU columns must not become a build dependency,
# and a host with a driver but no toolkit still has the versioned soname. On
# glibc 2.34 and later dlopen is in libc and this resolves to an empty stub.
bench: $(OBJS) $(ARCH_STAMP) $(BOINC_STAMP) $(LINK_STAMP)
	$(NVCC) $(NVCC_ARCH) --cudart $(CUDART_LINK) -Xlinker --gc-sections \
		$(BENCH_LINKFLAGS) -o $@ $(OBJS) $(BENCH_LIBS) -lm -ldl

# Only the CUDA objects care about the arch; the C objects are host-only.
$(CU_OBJS): $(ARCH_STAMP)
# HAVE_BOINC changes boinc_support.cpp and may change host-code portability.
$(OBJS): $(BOINC_STAMP)

# CPU-only correctness gates. No CUDA, so these run on a busy box and on
# machines without the card.
fbtest: fbtest.c $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ fbtest.c $(C_OBJS) -ldl -lm

# Native algebraic factor-base generator; deliberately CPU-only and free of
# CADO/GMP dependencies. It shares the job/poly parser and prime enumerator.
fbgen: fbgen.c poly.o primes.o
	$(CC) $(CFLAGS) -o $@ fbgen.c poly.o primes.o -lm -pthread

# Standalone GPU factor-base generator and benchmark.  It can either price the
# ordinary-root kernels or write a reusable native roots file with --out.
# In `all` since 2026-08-27; see the note there for why compiling this
# translation unit twice turned out to be free.
fbgen_gpu: fbgen_gpu.o fbgen_lib.o fb_load.o fb_cado.o poly.o primes.o platform.o $(ARCH_STAMP) $(LINK_STAMP)
	$(NVCC) $(NVCC_ARCH) --cudart $(CUDART_LINK) -o $@ \
		fbgen_gpu.o fbgen_lib.o fb_load.o fb_cado.o poly.o primes.o platform.o -lm

fbgen_gpu.o: $(ARCH_STAMP) $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)

# Same CUDA root generator, linked into production bench without the standalone
# benchmark main().  The generated algebraic FB is now the pipeline default
# when no --fb1/--cadofb file is supplied.
fbgen_gpu_lib.o: fbgen_gpu.cu bench.h $(ARCH_STAMP) $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)
	$(NVCC) $(NVCC_FLAGS) -DFBGEN_GPU_LIBRARY -c $< -o $@

# The same exact root finder drives on-the-fly special-q generation in bench.
# Keep it in one source: a copied degree-5 root finder would eventually drift
# from the factor-base generator whose output is gated against CADO. Function
# sections let the final links discard fbgen's CLI/thread-pool half, so sharing
# the implementation does not give bench or sqgentest a pthread dependency.
fbgen_lib.o: fbgen.c bench.h
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -DFBGEN_LIBRARY -c $< -o $@

sqgentest: sqgentest.c fbgen_lib.o poly.o primes.o
	$(CC) $(CFLAGS) -Wl,--gc-sections -o $@ $^ -ldl -lm

# Planning-time norm-width survey. Same objects the siever uses -- sqgen for
# (q,rho), qlat_build/norm_setup for the bound -- so it cannot drift from what
# pipe_side_prepare_q will actually enforce. CPU only, no CUDA.
normscan: normscan.c fbgen_lib.o poly.o primes.o $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)
	$(CC) $(CFLAGS) -Wl,--gc-sections -o $@ normscan.c fbgen_lib.o poly.o primes.o -ldl -lm

dumpcmp: dumpcmp.c $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)
	$(CC) $(CFLAGS) -o $@ $< -lm

# CPU-only checkpoint-format gate. In particular, BOINC output paths may
# contain whitespace after filename resolution, so candidate identity must not
# depend on the checkpoint parser's ordinary whitespace tokenisation.
ckpttest: ckpttest.c ckpt.h bench.h platform.o $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)
	$(CC) $(CFLAGS) -o $@ ckpttest.c platform.o -ldl -lm

stoptest: stoptest.c platform.h platform.o $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)
	$(CC) $(CFLAGS) -o $@ stoptest.c platform.o -ldl -lm

# joins the two per-side --emit-cof files into cofactorisation batch input
mkcofbatch: mkcofbatch.c $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)
	$(CC) $(CFLAGS) -o $@ $< -lm

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJS:.o=.d) $(TEST_OBJS:.o=.d) fbgen_gpu.d

clean:
	rm -f bench fbtest fbgen fbgen_gpu sqgentest normscan dumpcmp mkcofbatch ckpttest stoptest slabtest *.o *.d $(ARCH_STAMP) $(BOINC_STAMP) $(LINK_STAMP) $(GIT_STAMP) $(DEFS_STAMP) $(CF_LMAX_STAMP) $(BN_LIMBS_STAMP)

.PHONY: all clean check ckptcheck stopcheck cofcheck fbgencheck fbgpucheck sqgencheck walkcheck slabcheck skipcheck

# `check` depended on fbtest alone, so the whole CUDA path -- kernels, the
# cofactoriser, the pipeline -- could fail to compile while the gates reported
# "all gates passed". Depend on `all`.
check: all ckptcheck stopcheck cofcheck fbgencheck sqgencheck walkcheck slabcheck
	./fbtest --fb1 ../oracle/c183.fb1

ckptcheck: ckpttest
	./ckpttest

# The clean-stop hook. A CPU gate, like ckptcheck: it needs no card and no
# oracle input, and it guards a path -- signal disposition around the band --
# whose regression is silent until an operator asks a real run to stop.
stopcheck: stoptest
	./stoptest

# The lattice-walk gate, which nothing used to run. `--verify` is refused under
# --pipeline (harness_only), and cofcheck/testsieve/every band run use
# --pipeline -- so the walk cases were "permanent" in the source and dead in
# practice. A regression in pl_make/pl_next at an untested aspect ratio would
# have shipped.
#
# `--verify-only`, NOT `--verify`. The latter deliberately falls through to
# run_bench() -- that fall-through is finding 32's apply-parity gate -- which
# drags in a CUDA device and the 60 MB untracked ../oracle/input.job.afb.0.
# This target is a CPU gate inside `check`, and a CPU gate that needs a card
# and an untracked oracle file is not one: it would fail on a driverless box,
# on a box whose card is busy, and on any fresh clone. --verify-only runs the
# walk cases and exits, touching no input at all.
walkcheck: bench
	@out=$$(./bench --verify-only 2>&1); st=$$?; \
	 printf '%s\n' "$$out" \
	   | grep -E 'Franke-Kleinjung|primes x 5 roots|FAILED' \
	   | sed 's/^/  /'; \
	 if [ $$st -ne 0 ]; then \
	   printf '%s\n' "$$out" | tail -20 | sed 's/^/  | /'; \
	   echo "  walkcheck FAILED (exit $$st)"; exit 1; \
	 fi


slabtest: slabtest.o $(C_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -ldl -lm

# CPU-only arithmetic/scheduling gate for j-slabbing.  It covers automatic and
# forced plans, partial tails, actual bkthresh -> small-FB -> pmax wiring,
# congruence phase translation, and the exact w<2^31 range used by the existing
# reciprocal in the hot direct-TD loop.
slabcheck: slabtest
	./slabtest

fbgencheck: fbgen
	./fbgencheck.sh

# Explicit GPU gate: not part of `check`, because the ordinary CPU suite must
# remain runnable on build hosts with no NVIDIA device.
fbgpucheck: fbgen fbgen_gpu
	./fbgpucheck.sh

# The PIPE_Q_SKIP gate: a special-q too wide for this build's exact-norm type is
# passed over and the band continues. Needs a CARD, so it is deliberately NOT in
# `check` -- same reason fbgpucheck is not.
#
# Needs `normscan` as well as `bench`: the jobs are generated to overflow THIS
# build's width, and normscan is the oracle that says the construction landed.
#
# RUN IT AT `make BN_LIMBS=4`. Above 128 bits it skips itself with an
# explanation rather than failing: a job that overflows a wider type has norms
# too large to yield survivors at bounds a gate can afford, and a survivor-less
# special-q is fatal by design (pipeline.cuh:2018). The path it covers is
# width-independent, so one width is enough.
skipcheck: bench normscan
	./skipcheck.sh

sqgencheck: sqgentest
	./sqgentest

# A GPU golden test over the cofactoriser: rho and ECM on the parity special-q,
# the even-cofactor invariant, the bounds refusals, and a candidate file with no
# trailing newline. Each case asserts an exact relation count, so a silent
# change of method or a truncated parse fails here rather than in a band run.
cofcheck: bench
	@./cofcheck.sh
