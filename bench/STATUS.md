# STATUS — what exists now

`RESULTS.md` and `../prototype.md` are lab notebooks: they record findings in
the order they were discovered, including the ones later refuted, because the
refutations are the most useful part. That makes them bad at answering "what
does this thing do today". This file answers only that, and holds nothing that
is not current. **Last updated 2026-11-10.**

**Pruned 2026-11-10.** Closed items are cut to their outcome, their finding and
whatever they left open; superseded measurements are gone, and the finding in
`RESULTS.md` has the rest. The text as it stood before the prune is
`git show eaba635:bench/STATUS.md` (and before the 2026-10-29 prune,
`git show cadcf3c:bench/STATUS.md`).

## Architecture

One process, both sides, **two units of work in flight** (a slab, or a whole
special-q on one-slab plans):

```
per special-q (per slab where the plan has several):
  host    reduce the q-lattice, build the three modulus tiers
  device  k_transform    plattice transform of the factor base
          k_fill_rows    bucketed q < 4I at logI <= 15: row-banded, one
                         atomic per region per row for a warp's hits
          k_fill_atomic  the rest: bucket-sieve into regions (no record where
                         gcd(i,j) is visibly > 1: both even, or 3 | both, and
                         5 or 7 | both with several slabs per q; side 0 stores
                         only under side 1's survivors)
          k_fill_merged  one slab per q: the two above as one launch, the
                         row tier to q < 8I
          k_apply        norm init + small-prime line sieve + bucket add
                         + threshold scan  ->  survivor bitmap
                         (both-even cells get no norm and no small-prime hits;
                         side 0 computes the norm only under side 1's
                         survivors, --norm-gate)
  (side 1 first, then side 0)
  device  k_intersect_compact   both-sides bitmap AND, gcd filter, rank scan
          trial division, classification, resieve
                               (resieve of q < 4I row-banded at logI <= 13;
                               side 0 resieves from its gated bucket records,
                               --resieve-scan)
                               (the candidate RECORDING pass is
                               k_td_record_warp, one warp per candidate;
                               the two dense passes are k_td, one thread
                               per survivor)
          k_cof_enqueue  into a cross-q device queue; at each flush, k_cofac
                         (ECM on both sides) splits the queue
  host    emit relations (on a writer thread)
```

**Two units in flight (findings 108-109, 121, 123, 136; `--q-pipeline`).** Once
a slab's intersection is done, the next slab -- or, after a q's last slab, the
next q -- is prepared and both its sides are sieved on a non-blocking stream,
while this slab's trial division and cofactorisation run on the legacy stream.
On one-slab plans, fill ahead (`--fill-ahead`) also runs the q after next's
side-1 fill during this q's TD, `--q-prep auto` prepares the next q during
this q's TD when its sieve has already finished, and `--td-ahead` queues TD's
record pass ahead of its count readback. Only `plat` and `rowrt` are doubled,
plus a third walk-cursor buffer on slabbed plans; the bucket array is not.
When that extra does not fit in free VRAM the run stays single-queued, and the
banner and run log say so (`# q-pipeline` line). `--fill-concurrent` and
`--qspan` exclude it; `--no-q-pipeline` turns it off. Per-stage times then
include sharing the card: only wall clock per q is additive, and `unaccounted`
and acc/wall are reported as n/a; use a `--no-q-pipeline` band to diagnose
host starvation.

**The two sides share one bucket allocation and run in turn**, side 1 first.
Side 0's fill stores records only where side 1's survivor bitmap (folded to
8 positions per bit, `--fill-gate-shift 3`) has a survivor (`--fill-gate`),
and side 0's apply computes the log norm only at side 1's survivors
(`--norm-gate`, on wherever the fill gate is, finding 141).
`--fill-concurrent` is an opt-in two-stream path with a second bucket
allocation (finding 94); it overlaps the sides of one q, so it turns both
gates off, and it excludes `--q-pipeline`.

**Defaults, all output-identical** (each has a `--no-` form or a level that
restores the old path, and the run log names what a run got):

- **Sieve skip** (finding 100, `--sieve-skip 2`): both-even positions are
  never sieved, `3 | i, 3 | j` is skipped in fill, and (finding 150) 5 or 7
  dividing both on plans of several slabs per q (`--fill-skip57`).
- **Fill** (findings 107, 132, 150): the row tier (`--fill-rows 4` at
  logI <= 15, `--resieve-rows` at logI <= 13), the side-0 fill gate,
  bucket regions sized by row class (`--row-cap`), one launch for both
  tiers on one-slab plans (`--fill-merge`, row cut 8, 128-thread blocks),
  and 4608 x 32 fill blocks elsewhere (`--fill-blocks`). `bench/fillcheck.sh`
  is the identity gate.
- **Small-prime sieve** (findings 101-102, 129, 131, 145-146): the block and
  warp tiers share their per-entry setup; powers of two <= 32 are folded
  (`--tiny-fold`) and 3, 5, 7, 9 come from tables (`--tiny-table`); moduli
  below 36 are walked by the whole block and below 256 by one warp each
  (`--ss-block-cut`, `--ss-warp-cut`, runtime flags since finding 146).
- **Bucket records carry log p** (finding 130); slices start only where the
  log changes (finding 128).
- **Cofactorisation**: ECM on both sides with an effort ladder by `lpb`
  (finding 125), a per-side plan and the 3LP abort rule (finding 126); see
  "Cofactorisation" below.
- **Output** (findings 137, 138, 142): relations are formatted by hand and
  written by a writer thread; the in-loop checkpoint waits at least 30 s.

`bench/tune.sh` (finding 146) times the speed-only knobs (the tier cutoffs,
`--fill-blocks`, `--fill-gate-shift`, `--fill-merge`) for one job on one card,
interleaved and rotated on the same q, relations compared, and adopts a value
only if it is slower in no round and faster by more than 1% and by more than
the defaults' spread. The defaults above were chosen on the 5070 (sm_120);
row 44 has what other cards want.

**The algebraic factor base is generated in-process by default, added
2026-08-24.** With `--fb1` omitted the pipeline calls the same GPU root finder
that backs the standalone `fbgen_gpu`, so a run no longer needs a
multi-gigabyte roots file staged first — which is what lets a BOINC work unit
carry only the job. `FBGEN_GPU.md` is the reference and `fbgpucheck.sh` gates it
across degrees; `--fb1`/`--cadofb` still take a file when one is wanted.

`--relations NAME` stages to `NAME.part` and renames to `NAME` only when the
band completes. The `.part` is the **durable artifact**, not scratch: at most
every 30 s, at a flush boundary under `--cofactor`, it is fsynced and
`NAME.part.ckpt` records the next `(q, rho)`, the byte offset, the relation
count, the derived scale/allowance and a job fingerprint. Rerunning the same
command resumes there; `--restart` discards; SIGINT/SIGTERM and `--stop-file`
stop cleanly at the next special-q; `NAME.lock` refuses a second writer. A
stopped and resumed band reproduces the uninterrupted one byte for byte,
`kill -9` and a torn final line included, and the install gates re-check
that on c135 (kill -9 at 20 s and 45 s, and each clean stop). See item 12a.

`--log PATH` appends a run log: a header block naming the commit, argv, job
fingerprint, card, geometry and factor-base convention, then a timestamped
record every `--log-every` seconds (300 by default) carrying progress
alongside `GPU-accounted / wall`, GPU utilisation, board watts and host load.
Item 12b.

### Distributed packaging (BOINC), added 2026-08-15/16

Optional and compiled only under `HAVE_BOINC=1`; the ordinary build has no
BOINC dependency and its wrappers are no-ops. The app initialises and finishes
through the BOINC API, resolves every named input and output through
`boinc_resolve_filename_s()`, requests normal host-thread priority for the
feeder thread, and reports a nondecreasing fraction done at special-q
boundaries. `README.md` is the reference for building and for the workunit
command line.

**Which GPU a task runs on** is read from `init_data.xml`
(`<gpu_device_num>`, via `boinc_get_init_data()`) and **outranks `--device`**,
which is the opposite of this codebase's usual precedence and deliberate: a
`--device` in an app version or workunit template is shared by every task on
every host, so honouring it reinstates the all-tasks-on-card-0 bug the
assignment exists to prevent. `--device` selects only when there is no
assignment. Each task's stderr records which of the two happened, the device
count this process can see, and the card actually used. Startup also refuses an
ordinal past `cudaGetDeviceCount()` with both numbers named, rather than
letting it surface as a bare "invalid device ordinal".

**Reviewed and deployed 2026-08-17**: Greg Childers — who reported the
original all-tasks-on-device-0 failure — signed off on the assignment change,
and a BOINC queue is running with it. That closes item 13 as far as the
application is concerned. There are no checkpoint files in the BOINC sense:
a BOINC client quits a GPU task whenever it suspends it, and the restarted
task resumes from 12a's last checkpoint (the repo's own sidecar, not BOINC's),
so each suspend loses up to ~30 s at c135 (Known defects). Under a BOINC
client, unusable staging or checkpoint artifacts are discarded and recomputed,
up to three times per work unit; standalone runs still refuse them.

**Native Windows is now a supported build target.** `platform.c` carries the
filesystem/process/stop-hook differences, `build_windows.bat` builds a static-CRT
`bench.exe`, and the Windows binary includes the same in-process GPU algebraic
factor-base generator used on Linux. `build_windows.bat fbgen_gpu` additionally
builds the standalone reusable-roots-file generator. Cross-platform relation
bytes are gated separately by `wintest.bat`; Windows process termination still
has the documented limitation that `TerminateProcess` cannot run a checkpoint
handler, so use `--stop-file` for a clean stop.

## Current size limits and j-slabbing

The production pipeline supports rectangles larger than `2^31` positions by
**j-slabbing** them (since 2026-08-24). There is no unsafe override of the
local arithmetic bounds.

| quantity | current limit | immediate reason |
|---|---:|---|
| local sieve slab | `2^31` positions | bucket/bitmap/rank positions remain `uint32_t` |
| full pipeline rectangle | no total-area cap under `--pipeline`; `logI` in [2,20] | host scheduler splits `J` into safe slabs. Outside `--pipeline` the old `I*J <= 2^31` refusal still stands (`bench_main.cu`, the `!cfg.pipeline` check) |
| `lpb` | 64 | a resulting prime is stored in one `uint64_t` (was 32 until 2026-08-17) |
| `mfb` | 127 | the cofactor queue narrows residuals to `mz<4>`, whose even/odd product needs them below 2^127 (128 until 2026-11-09); 96 in a `CF_LMAX=3` build |
| large primes per side | 3 | `ceil(mfb/lpb) <= 3` is checked before the run |
| exact norm | 384 bits (`BN_LIMBS` 12) | build-time knob, even limbs 4..16; a lattice needing more is SKIPPED with a warning, never wrapped (was 256 until 2026-08-27) |

**Slab planning.** Automatic planning targets **32,768 bucket regions per
slab** (`SLAB_PERF_REGIONS`), which at the default `--region 14` is `2^29`
positions, and splits only an area of at least twice the target (deliberate
hysteresis; `slab.h` and `slabtest` pin both). With default `J=2^(logI-1)` and
`bkthresh=I`, I15 uses 1 slab, I16 4, I17 16, I18 64, I19 256, and I20 1024.
`--slab-j N` overrides it; the `2^31` position bound stays mandatory. Raising
`bkthresh` can still make the planner choose smaller slabs. What the target
rests on, all on the 5070 at c194 I16:

- **Fill is minimised at a fixed region count, not a fixed area** (findings
  79-80): `fill = L(nregion) + 1.178 x nslab`, `L` least at 16,384 regions
  and 32,768 the reachable optimum at region 14. Region 14 is the joint
  optimum because `k_apply` launches one block per region and costs +52% at
  region 13, +157% at region 12. **The card-dependent quantity is
  `nregion`, not slab size.**
- **`L(nregion)` is DRAM traffic** (finding 81): past the knee, bucket lines
  are evicted before eight 4-byte appends fill a 32-byte sector, so the
  memory system read-modify-writes them (DRAM reads +135% at flat L2 reads).
  That is associativity/sector-bound, not capacity-bound, which is why a 6 MB
  3090 and a 48 MB 5070 agree.
- **The sweep** (finding 78, total area `2^31`, slab count the only variable):
  complete ms/q 439.6 / 386.3 / **355.7** / 361.1 / 396.3 / 460.3 / 611.1 at
  1 / 2 / **4** / 8 / 16 / 32 / 64 slabs. Each slab streams ~42 B per
  factor-base entry (929 MB at c194), 1.18 ms per slab of fill; below `2^29`
  the dense TD pass grows too (item 18). At area `2^32` the optimum stayed at
  `2^29` per slab (8 slabs), so slab size is what controls it.
- **Open:** an L40 preferred `2^30` slabs (65,536 regions) at area `2^32`
  (finding 72), 4.6% faster; capacity was never the reason, so that needs its
  own explanation, on a matched job.

### Why the lattice walk itself is wide

A local sieve position still fits in 31 bits, but the Franke-Kleinjung reduced
**increment** need not fit in 32. For realistic factor-base primes,
`(j0 << logI) - mi0` or `(j1 << logI) + i1` can exceed `2^32` even on I15/I16.
The earlier `uint32_t plat_t` therefore had a latent wrap: it could create
spurious sieve hits even before whole-area slabbing was needed. `plat_t` now
stores 64-bit increments; the bounded local walk terminates instead of wrapping
when its exact next hit leaves the 32-bit coordinate, and slab continuation is
carried exactly in 64 bits between slab origins.

**The walk POSITION is now 64-bit too, in every production fill and resieve
kernel (`k_fill_l1` excepted, see below), and
that is a speed decision rather than a correctness one (finding 73).** The
32-bit form (`pl_next`) reaches its bound by saturating, which costs a branch
per increment; `pl_next64` is a plain wide add. Measured idle at −6.1% of fill
on an unslabbed geometry; converting the slabbed walk the other way — down to
32-bit — cost +6.0% on a contended card and was not repeated. Positions
themselves still fit in 31 bits, so records, bucket indices and bitmap offsets
remain 32-bit; only the walk variable is wide. **`k_fill_l1` is the exception
and keeps `pl_next`**: `__launch_bounds__(512, 3)` caps it at 40 registers, it
fits in exactly 40 today, and widening its walk made ptxas spill (12 B stores /
8 B loads) rather than lower occupancy — the only spill in the build. So the
32-bit walk is still a shipping device path and `verify_walk_slabs` still gates
a real kernel.

This widening costs 8 additional bytes per uploaded full-FB entry. Slabbed
runs additionally keep two 64-bit continuation values per entry (16 bytes per
entry total); unslabbed runs allocate no continuation arrays.

### Larger rectangles: memory is now the main constraint

The bucket array and survivor bitmaps are allocated for **one slab** and reused,
so a full I17/I18/... rectangle no longer requires a monolithic `2^33`/`2^35`
position workspace. Area-proportional work still scales with the full rectangle,
and every slab streams the whole factor base again -- 929 MB at c194 I16,
measured, finding 78 -- so the number of slabs remains a runtime consideration
even though it no longer multiplies peak bucket memory.

The measured pre-slabbing memory model for the two large allocations remains a
useful per-local-area reference:

| local area | bucket array | three survivor bitmaps | subtotal |
|---:|---:|---:|---:|
| `2^31` | 5.53 GB | 0.81 GB | 6.34 GB |
| `2^32` | 11.06 GB | 1.61 GB | 12.67 GB |

For automatically planned full areas of `2^30` and above, `2^29` is now the
production performance target per slab. Larger local slabs remain available
through explicit `--slab-j` when they satisfy the safety limits. The `2^32` row
is retained as the old monolithic projection, not as an allocation the automatic
slabbed path makes.

### Exact-norm width — WIDENED 256 -> 384 BITS 2026-08-27

`bn_t` is the fixed-width magnitude that trial division builds the exact norm
in. It was 8 limbs, sized on a quintic: at q=120000053 the largest homogeneous
term was 224 bits, a 6-term sum stayed under 227, and 256 left ~28 spare.

**That headroom was a property of that polynomial, and a large octic blows it.**
On the Cunningham 2,1139+ SNFS form — **degree 8**, and provably so: 1139 =
17 x 67, so `x = 2^67 + 2^-67` leaves the minimal polynomial of `zeta_17 +
zeta_17^-1`, of degree `(17-1)/2 = 8`, with `Y1 = 2^67`, `Y0 = -(2^134 + 1)`
and `F(Y0, Y1) == 0 mod n` (checked 2026-09-01) — the exact algebraic norm runs **232-292
bits** depending on the q-lattice, and the tail is driven by the SHAPE of the
reduced basis rather than by the sieve area — **shrinking `logI`/`J` does not
escape it**, and neither does a tighter estimate, since `norm_exact_bound_bits`
sits only 2-3 bits above the true maximum term. Measured: a 2000-q band at 8
limbs died after 116 q needing 260.75 bits, having written **no relations**, and
could not resume past that q because the checkpoint names it. The same band at
12 limbs completed and all 4,015 relations rebuilt both norms exactly.

**Cost, A/B on AS276** (C208, deg 5, logI 15, three runs each): `k_td` 1.138 ->
1.593 ms (+40%) and `k_td_record_warp` 0.234 -> 0.284 ms (+21%) — **+0.45 ms on
a 90 ms special-q, inside the ±0.8 ms run-to-run spread**. Registers 58 -> 78
and 68 -> 80 with **no spill**; `k_apply`, `k_classify`, `k_cof_enqueue` and
`k_rel_pack` untouched; the divide-down does not scale with the width at all
because `td_divide_out` loops on `bn_top` and the added limbs are leading zeros.
Memory is the term the timing hides: `sizeof(bn_t)` 32 -> 48 B, allocated **per
survivor and per candidate**, so it scales with survivors/q — ~2 MB at AS276's
39,042, and re-measure rather than assume on a job with far more.

**The gate is byte-identity, not a benchmark.** Every value that fit the narrow
build is represented identically in the wide one, so a wider binary must
reproduce a narrower one exactly on any job the narrow one could run; 12 limbs
qualified on AS276 with the same relation-file md5 and the same survivor,
candidate and split/dead/stuck counts. `make BN_LIMBS=N` (even, 4..16) sets it,
with a stamp file so changing it forces the rebuild.

At runtime `pipe_side_prepare_q` checks each q against the built width and
**skips** the ones that do not fit, warning each time and ending the band at
`PIPE_SKIP_MAX`. That keeps a band alive across a rare bad lattice; it is damage
control, not an answer, because every skip is a lost special-q. The band still
drains and checkpoints when it ends that way, but since 2026-09-05 it exits
`BENCH_EXIT_UNSUPPORTED` (3) and names the width to rebuild at, rather than
reporting a completed work unit (item 3c).

#### `normscan` — decide the width before distributing work

The client cannot fix an overflow: the fix is a wider **rebuild**, and a work
unit of a few hundred q out of tens of millions cannot even see a ~1e-5 tail
coming. So the width has to be chosen once, centrally, at planning time.
`normscan` (CPU only, built from the siever's own `sqgen_next`, `qlat_build`,
`norm_setup` and `norm_exact_bound_bits`) surveys a whole projected band for a
given poly and geometry.

**The sample maximum is the wrong statistic and the tool does not report it.**
On the 2,1139+ over 60M-460M at logI 15, 2,500 sampled (q,rho) gave a maximum of
242 bits and the confident, wrong conclusion that 256 sufficed; 160,018 samples
found q=367699421 at **273.08 bits**, with its nearest exceedances at 258.9 and
258.4. A sample of n out of N sees the 1/n quantile, not the 1/N one. `normscan`
therefore reports a **projected band maximum from an exponential fit to the
upper tail** and warns on proximity, not only on crossing.

**Exercised on the motivating job 2026-09-01**, with the 2,1139+ octic's real
coefficients, band 60M-460M, special-q on the algebraic side, 160,000 samples:

| geometry | median | 99% | sample max | projected band max | verdict vs 384 |
|---|---:|---:|---:|---:|---|
| `logI 15, J 16384` | 235.1 | 253.1 | 304.70 | 326.5 | OK, 57.5 bits clear |
| `logI 16, J 16384` | 238.4 | 253.1 | 304.70 | 326.1 | OK, 57.9 bits clear |
| `logI 16, J 32768` | 243.1 | 261.1 | 312.70 | 334.5 | OK, 49.5 bits clear |

Exit codes behave as documented (2 against `--limit-bits` 256 and 320, 0 at the
build's 384), the refusal names the right rebuild (`make BN_LIMBS=12`), and the
fitted tail scale **5.96-5.99 bits reproduces the beta = 5.9** the margin rule
was calibrated on. Doubling `J` costs ~8 bits at the 99th percentile, which is
why the survey is per geometry.

Exit codes are verdicts: **0 pass, 2 will overflow, 3 too little margin, 1 the
survey could not run, 64 usage.** `testsieve.sh` runs it **per geometry** —
the answer moves with the geometry (250 bits at 15e, 257 at 16e on that job) —
records a non-zero verdict and repeats it in the summary, but does not abort the
sweep, since the yield numbers are what say whether this geometry is the one to
rebuild for.

### LPB and MFB are separate widths

`lpb` bounds an *individual resulting prime*. `mfb` bounds the *composite
residual sent to cofactorisation*. Raising one does not inherently require
raising the other:

- Raising `lpb` above 32 required a `uint64_t` representation for split
  primes, unsplit prime residuals, sorting, emission, primality checking, and
  relation reconstruction. It did **not** require four-limb rho/ECM at
  `mfb <= 96`. **Done 2026-08-17** — see "64-bit large primes" below.
- Raising `mfb` above 96 requires `mz<4>` arithmetic even when the actual
  maximum is only 101 bits. Rho, ECM, probable-prime testing, GCD, and exact
  division all pay the wider modular-arithmetic cost. **Built 2026-08-18** —
  see "Cofactorisation" below.

The useful dispatch is fixed-width kernels selected on the host, independently
for each side:

| side's `mfb` | cofactor type | status |
|---:|---|---|
| `<= 64` | `mz<2>` | **not built, and deliberately so** — see below |
| `<= 96` | `mz<3>` | built |
| `<= 127` | `mz<4>` | built 2026-08-18; 127 since finding 153 |

The underlying `mz<L>` arithmetic was already templated; what was hardcoded was
the production queue/storage and its launches at `mz<3>`. Runtime
variable-length loops inside a kernel are not the intended design: they would
lose compile-time unrolling and likely increase register pressure. The
criterion is the bit bound, not the label: a 2LP side with `mfb=65` still needs
three limbs.

The `mz<2>` tier is left out on purpose. `CF_LMIN` is 3, so a side with
`mfb <= 64` runs three limbs and pays for a limb it does not use. That is the
same trade the rational queue already made when it went from two limbs to
three: the stage nearly doubled (+83%) and wall clock moved +0.64%, because
that queue is ~1% of a special-q. Reinstating a narrow tier means a third
instantiation to keep in step for a fraction of a percent.

### 64-bit large primes — BUILT AND VERIFIED 2026-08-17

`lpb` up to 64 is supported. The widening was confined to values bounded by
`lpb` rather than by `lim`: the split primes and unsplit prime residuals in the
cofactor queue (`d_sp0/1`, `d_sm0/1`), `mz_split`'s output, the relation
emitter on both the inline and `--cofac` paths, and the reconstruction gate.
The trial-division lists stay 32-bit deliberately — those are factor-base
primes, bounded by `lim`, which is well under `2^32` even for a C208. The
cofactor arithmetic did not change at all: `mfb <= 96` is still `mz<3>`.

The gate needed its own 64-bit arithmetic, which is the part that was not a
retype: `bench_is_prime32` cannot test a 33-bit factor and `bn_divmod_u32_pre`
cannot divide one out, so `cf_is_prime64` (deterministic Miller-Rabin, the
seven bases valid below `2^64`) and `cf_bn_divmod_u64` (128-bit long division,
stepping in 64-bit limbs because a divisor above `2^32` overflows a 32-bit
quotient digit) were added host-side. Without them the gate would have skipped
exactly the factors the change introduces.

Verified three ways: the 1500-q c147 band is **byte-identical** to its
pre-change MD5, so nothing moved at `lpb <= 32`; the c183 golden q at `lpb 33`
emits 58 relations carrying 24 factors above `2^32`, all reconstructing; and
that same file is **refused** when re-read at `lpb 32`, so the bound is
load-bearing rather than decorative. All three are now cases in `cofcheck.sh`.
On a 400-q c147 band at `lpb 33`, 8,810 of 2,075,496 emitted factors exceed
`2^32`, the largest `0x1ffd77b3b`.

What this does **not** cover is `mfb > 96`, which is the four-limb change
under "Cofactorisation" below.

This makes `lpb=33` with `mfb=64/95` a materially smaller extension than
`lpb=35` with `mfb=64/101`: the former needed 64-bit factor outputs but retains
two- and three-limb cofactor kernels; the latter needs a four-limb kernel on
the 101-bit side. Both ratios still ask for at most three large primes, so
`CF_MAXFAC` is not the blocker.

### Cofactorisation

**Width, per side.** Each side's cofactors run in the narrowest instantiation
that holds its `mfb`: `mz<3>` up to 96 bits, `mz<4>` up to 127 (no `mz<2>`
tier, for the reason under "LPB and MFB" above). `lpb >= 33` is necessary but
not sufficient for a fourth limb: `mfb` decides. AS276 (`~/code/ggnfs-distributed/AS276.job`, the C208, `lpbr/mfbr`
33/64, `lpba/mfba` 35/101) resolves to 3 limbs on the rational side and 4 on
the algebraic with no flag.

Forcing the wide shape on a job that does not need it cost a widened queue
x1.72 and **+8.8% of wall on the c183, +8.2% on the C194** (finding 70, before
the PTX arithmetic), so the dynamic per-side selector is what ships; it costs
one host-side switch per flush. `--cof-limbs N` / `--cof-limbs0 N` force a side
**wider** than its `mfb` needs (to time one job at both widths); narrower is
refused. `make CF_LMAX=3` builds the 3-limb splitter alone: same 3-limb
kernels, about half the `.nv_fatbin` and the `ptxas` time, a packaging
option, not a speed one.

Where the width lives:

- `CF_LMAX` / `CF_LMIN` / `cf_limbs_for_mfb` in `bench.h` — the build's width
  range and the mfb → limbs rule. `CF_LMAX` defaults to 4.
- `cofq_t.d_c0/d_c1` are **raw limb arrays plus `L0`/`L1`**, not `mz<3> *`. The
  stride is a run-time choice and `mz<3>`/`mz<4>` are distinct types, so the
  array cannot carry one of them in its type. `cf_run_rounds_dyn` is the only
  place that casts back.
- `k_cof_enqueue` is templated on `<L0, L1>` (four cheap instantiations) and
  the standalone `--cofac` batch parses at `CF_LMAX` and narrows per side.
- `resolve_and_check_cofactor_config` is the one place that has seen both the
  `.job` file and the command line, and refuses an `mfb` above what the
  chosen width holds.

The width invariant is asserted rather than trusted. `cof_classify` rejects a
residual above `mfb` and `resolve_and_check_cofactor_config` refuses an `mfb`
the side's width cannot hold, so nothing can reach the queue too wide for its
side — and if it does, `k_cof_enqueue` marks it `CF_OVERFLOW`, counts it, and
`cofq_flush` stops the run. The failure mode being closed is a *silently
truncated* cofactor, which does not crash: it emits a relation that
reconstructs to the wrong norm. Both previous versions of this array (2 limbs,
then 3) had exactly that bug.

**The ceiling that binds first is not the width.** `CF_MAXFAC` caps a split at
3 large primes, so `mfb <= 3*lpb` regardless. Going past 127 bits means 4LP,
which is `CF_MAXFAC` and `mz_split`'s `sp + 2 > CF_MAXFAC` stack guard, not
another limb.

**Method: ECM on every side, the default since 2026-10-21** (finding 125), at 8
curves x 4 rounds, `B2 = 30 * B1`, and `B1` on a ladder by `lpb`, one step
higher for a 3LP side: 2LP 100 / 140 / 200 / 280 / 400 at lpb 29 / 30-31 /
32-33 / 34-35 / 36-37, 3LP 140 / 200 / 280 / 400 / 560, continuing the ladder
(unmeasured) at 38 and up to 1120, each side its own plan from its own lpb
(`cof_auto_b1`, `COF_AUTO_CURVES`; per side since finding 126). On a 3LP ECM
side, after round 2 only records that have split once are run again
(`--cof-abort`, default 2, finding 126: +0.7-4.9% relations per second on C183
to C208 and SNFS-301 for 0.09-0.24% of relations). Measured within 0.067% of
saturated yield at every lpb 29-36 on c135, c183 and I16. ECM beat rho at 2LP
by 1.2x (lpb 29) to 3.3x (lpb 35) once the PTX carry chains landed.
`--cof-rho` forces rho on both sides; cofcheck pins the automatic method and
the derived plan for a mixed and a 2LP-only shape. A record that exhausts its
budget or its curves returns `CF_INCOMPLETE` and is dropped silently, so an
under-provisioned schedule shows up as a yield hole, not as a cofactoriser
error (finding 69: AS276 lost 13.7% of GGNFS's relations to a rho budget
calibrated on `lpba 32`).

**The arithmetic.** The curves are Suyama's, projective `A24 = (an : ad)`,
sigma = `c0 * 1000 + cv + 6`, with `c0 = r + 1` in round `r` (so round 0 runs
sigmas 1006 and up). The ladder costs 12 products per bit (finding 154: x_dbl's
redundant one removed). Products are PTX carry chains (finding 125), and since
finding 153 `mz_mul<3>`/`<4>` use the even/odd carry-chain layout of Greg
Childers' gmp-ecm branch `gpu-stage1-dynamic`, reimplemented (that code is
GPLv3, this tree CC0): 95 -> 46 SASS instructions per product at 3 limbs, 158
-> 59 at 4, which is what needs the 4-limb modulus below 2^127. `mz_mul`
returns the canonical `a*b/R mod n` in `[0, n)`, so regrouping products cannot
change a result. Not adopted from that branch: its ladder (GMP-ECM's param 3,
a third of Suyama's expected torsion) and its dedicated square (more SASS than
`mz_mul(a, a)` at 3 and 4 limbs). `make mulcheck` (`multest`, needs a card)
checks both widths against a bit-serial reference on structured words; rerun
`work/eomul/mutants.py` against it after changing a chain.

**`k_cofac` carries `__launch_bounds__(256, 2)`** (finding 97): at 256 threads
per block, 128 registers per thread is exactly two blocks per SM, and where
the stage-2 kernel lands without the bound depends on the toolkit and the
target, not on the source alone (130 registers on CUDA 13.4, 150 on 13.2, 168
on 12.8 for sm_120). Every instantiation is at or under 128 registers (finding
154). **The 256 is a hard launch ceiling** (`COFAC_THREADS_MAX`,
bench.h): `cf_run_rounds` launches `k_cofac` at `min(--threads, 256)` and the
chunk floor uses the same width, so a wider `--threads` still applies to every
other kernel instead of failing at the first cofactor flush.

**Gates.** `cofcheck.sh`'s width block: a 3-limb and a 4-limb run of the same
job are byte-identical (rho and ECM iterate in the true domain whatever
`R = 2^(32L)` is, and `gcd(qR^k, n) = gcd(q, n)` for odd `n`, so widening a
side changes the cost and nothing else); `lpb 33 / mfb 99` derives 4/3 with
nobody choosing it and every relation reconstructs; the same shape with
`--cof-limbs 3` is refused. The genuine 4-limb population is AS276's, checked
against the 1.5B-relation GGNFS corpus for that job (finding 69): over 30
special-q and an identical sub-rectangle, recall 3,044 / 3,045 = 99.97%, and
the 64 relations of ours that exist nowhere in their corpus all reconstruct.

### Performance accounting

**bench reads CLOCK_MONOTONIC_RAW (finding 134).** The WSL guest's
CLOCK_MONOTONIC is steered between nominal and 6.6-8.3% fast every few
seconds, so wall figures taken on this host before finding 134 read ~5% long
and carry up to ~8% of clock noise per run; RAW matches the card's timer to 1
part in 10,000. Two time sources are active (`systemd-timesyncd` and
`hv_utils`); which one steers is not identified, and choosing between them is
Kyle's call. **The wall clock (CLOCK_REALTIME) is stepped back instead**: about
2.48 s every ~32 s, measured 2026-11-11 at load ~18 (`work/twin/mtime/`).
`/usr/bin/time`'s elapsed and file mtimes use it. So a process time comes out
~2.5 s short when a step lands inside the run (finding 155's process-time
column), and `make -q` can find an object newer than the link that used it.
Time processes in-process on RAW, or over enough runs to average the steps.
**Cause, 2026-11-11 (`work/twin/mtime/fight.txt`):**
- The Windows clock is 4.07 s fast against NTP (`w32tm /stripchart`). The
  Windows Time service is stopped, with Manual start.
- WSL's host-time sync slews the guest towards Windows' time with the kernel
  tick at 10833 (+8.3%).
- `systemd-timesyncd` steers the guest to `ntp.ubuntu.com` and steps it back
  ~2.5 s at every 32 s poll.

The two never agreed.

**Fixed 2026-11-11 ~07:52 by Kyle.** The Windows Time service now runs with
Automatic start (time.windows.com, 1024 s poll), and `systemd-timesyncd` is
disabled in the guest, so the guest follows Windows alone. Afterwards:
- Windows is within 16 ms of NTP;
- the tick is 10000 in every sample;
- there are 0 wall-clock steps in 300 s;
- CLOCK_MONOTONIC / RAW is 1.00000 (`work/twin/mtime/*.after.txt`).

Timings taken before then carry the caveats above. Bench keeps reading RAW.

**Where the time goes** (finding 147, nsys, installed `0c7dda1f`, 2026-11-01;
findings 150 and 153-154 have changed fill and `k_cofac` since). Share of
summed kernel time:

| kernel | c135 | C148 | I16 |
|---|---:|---:|---:|
| `k_fill_atomic` | **32.8%** | **29.3%** | 34.8% |
| `k_apply` | 25.6% | 22.3% | **37.0%** |
| `k_resieve_scatter` | 9.9% | 16.7% | 8.1% |
| `k_td` | 8.7% | 7.8% | 6.1% |
| `k_transform` | 4.8% | 12.1% | 2.4% |
| `k_fill_rows` | 5.9% | 4.9% | — |
| `k_classify` | 3.3% | 2.6% | 1.2% |
| `k_cofac` | 2.6% | (no flush in the window) | 5.2% |
| card idle | 5.0% | 1.4% | 0.5% |

What binds each, on the 5070:

- **`k_fill_atomic` is bound by L2 request rate, additive per request**
  (finding 148): side 1's time splits walk 16-21%, atomic 41-50%, store
  29-43%; side 0's is 45-58% gate. More warps, cursor spacing and running fill
  beside apply do not help (beside apply at greatest priority the two overlap
  96% and apply slows from 9.5 to 17.0 ms: +28-30 ms/q of wall either way).
  One launch for both tiers (`--fill-merge`) and the 5/7 gcd skip are what
  landed (finding 150). No large fill win is left on this card; on the 3060
  the fill grid is worth up to 19% (row 44).
- **`k_apply` is issue-bound** (~88%, findings 110-111), with the block
  tier's shared-memory atomics and the norm leading; occupancy was the cheap
  win there (finding 75, `__launch_bounds__(512, 3)`, three blocks per SM).
  Its barrier stall after the bucket add comes from the tier cuts, worth
  about 0.6-0.8% of wall (finding 147). The two big kernels want opposite
  tuning: do not carry a conclusion from one to the other.
- **`k_transform`** is balanced (issue 61%, DRAM up to 52% at I16); its
  inverse is branch-free square-and-multiply since finding 116. Row 47 is
  the open lead.
- **`k_resieve_scatter`** at C148 waits on its summary and bitmap loads; the
  coarse filter below 2^29 positions is slower, so `PIPE_RS_COARSE_MIN`
  stands; `UNROLL` 8 is slower (48 registers cost a block per SM), so 4
  stands (finding 147).

**Card idle** is 5.0% at c135 in finding 147's trace (3.6% after findings
133-137, rising as the kernels shrank), 1.4% at C148 and 0.5% at I16. What is left of it is listed in row 33 and item 4.

**The final metric is time to a filterable matrix**, not raw relations/s:

```
new time / old time
  = (new ms/q / old ms/q)
  * (new required usable relations / old required usable relations)
  / (new usable relations/q / old usable relations/q)
```

LPB directly expands the possible large-prime ideal universe and therefore the
column and singleton burden during filtering. As a scale indication only,
`pi(2^(b+d))/pi(2^b) ~= 2^d*b/(b+d)`: 32 -> 33 is about 1.94x and 32 -> 35
about 7.31x. Those are **not** required-relation multipliers—a run does not
sample the entire prime universe—but they show why extra raw yield cannot be
credited without filtering. MFB does not directly enlarge the ideal universe
at fixed LPB, but admitting more and larger 3LP products can still raise the
required surplus. A change that shrinks the sieve area per q (a shorter J, say)
has to be priced the same way: it needs more special-q for the same relations,
at higher q where yield is lower and duplicates more common.

### Work status

| item | status |
|---|---|
| A=32 by j-slabbing, state carried across slabs | **done 2026-08-24**, tuned to 2026-08-26, validated at NFS@Home's own shape 2026-09-01 (finding 82) |
| A=32 as ONE unslabbed rectangle | not wanted: finding 78 makes the slabbed path the faster one |
| 64-bit large-prime outputs and gates | **done 2026-08-17** |
| per-side `mz<3>` / `mz<4>` dispatch | **done 2026-08-18** (finding 70) |
| ECM on every side, effort ladder by `lpb`, per-side plan, 3LP abort | **done 2026-10-21** (findings 125-126) |
| PTX carry chains, then the even/odd products | **done 2026-10-21** (finding 125) and **2026-11-09** (finding 153) |
| cofactor outcome reporting (`split / dead / stuck`) | **done 2026-08-19** |
| C208 validated against a 1.5B-relation GGNFS corpus | **done 2026-08-19**, 99.97% recall (finding 69) |
| 384-bit exact norms, `BN_LIMBS` a build knob | **done 2026-08-27** |
| `normscan` band width survey, wired into `testsieve.sh` | **done 2026-08-27** |
| in-process GPU factor-base generation | **done 2026-08-24** |
| a complete factorisation from our relations | **done 2026-08-05**, a C123 (`work/c123run/msieve.log`) |

## Validated

| | what | where |
|---|---|---|
| **Golden suite** | `./cofcheck.sh` — c183 q=120000053 and the width, method, tier-cut, skip and gate cases, exact relation counts, not ranges | `cofcheck.sh` |
| **Install gates** | `abcheck.sh` (36 runs over four job sets, relations byte-identical against the installed binary), the c135 20,000-q reference (1,855,599 relations, all reconstructing), `kill -9` resumes and clean stops back to that reference, SNFS301, `make check`, `fillcheck.sh`; SASS and resource identity outside the kernels a change touches | `bench/*.sh`, `work/*/gate.sh` |
| **Product gate** | `mz_mul<3>`/`<4>` against a bit-serial reference, with mutants for every carry | `make mulcheck` |
| **Reconstruction gate** | every emitted factor divides, is prime, is within lpb, `gcd(a,b) == 1`, and both norms rebuild to exactly 1 | `--check-relations` |
| **Transform gates** | root transform against its definition, over a real factor base; the fast path against the general path exactly | `fbtest` |
| **Band runs** | c147: 1340 q → 159,837 relations, both sides PASS | 5070, 5090, 4090, A100 |
| **Real job** | snfs236, ~20M relations before deliberate interruption | 5070 |
| **Full NFS factorisation** | C123 sieved entirely by us, filtered and solved by msieve: `p42 * p82` | `work/c123run` |
| **`A = 32`** | AS276 at `2^17 x 2^15`, 8 slabs: 1,322/1,322 norms rebuilt, rectangle confirmed from the relations (finding 82) | 5070 |
| **Completeness at `A = 32`** | AS276 at GGNFS's own `2^17 x 2^15`, 8 slabs, 30 q: 4,563 of GGNFS's 4,564 relations found (99.98%), no loss at any slab boundary (finding 100) | 5070 |
| **Older cards** | GTX 1080 (sm_61) and RTX 2080 Ti (sm_75): `--verify-only`, `fbgen_gpu`, `cofcheck.sh` and a c183 band reproduce the 3090/4090 counts (item 15) | k8s pods |

Cards with measured band data: **RTX 5070** (WSL2, the development card),
**RTX 5070 Ti Laptop** (WSL2, 115 W; finding 139), **RTX 5090**, **RTX 4090**,
**A100 80GB** (native Linux), an **RTX 3090** via an external reporter, and
`tune.sh` reports from a rented **RTX 3060** and **RTX 4070 SUPER** (row 44).

`--check-relations` runs on a machine with no GPU.

## Measured, and what it means

- **Against cado-gpu** (row 27; latest finding 147, 2026-11-01, installed
  `0c7dda1f`, quiet host, raw relations per second at equal yield, its
  `p < 96` skip off): cado-gpu 1.031x ours at c135, within the rounds' noise;
  ours 1.139x cado-gpu at I16. Findings 150 and 153-154 have landed since and
  have not been compared. CADO's `p < 96` skip is worth 1.13x (c135) to 1.27x
  (I16) per relation to it (finding 124). CADO's ~8 CPUs are GPU-feeding
  threads spinning in `cudaStreamSynchronize`; its real host work is ~1.2
  cores, and ours is under one.

- **Against GGNFS on the CPU** (item 0): on this box, c183 I15e, GGNFS at 16
  workers, **~3x the time and ~2.8-2.9x the whole-box energy** per unique
  relation, on two rectangles and three q (findings 71, 83, 87, 2026-08/09).
  The GPU side is several times faster since (findings 100-154); the margin
  has not been re-measured. **Yield matches GGNFS at matched geometry**:
  46.09 against 46.47 relations per (q, rho) pair on the c183 (finding 57),
  0.979-0.981 of GGNFS's on the C194 at all four matched rectangles (finding
  65), and 99.97-99.98% recall on AS276 (findings 69, 100).

- **Geometry: the GPU prefers smaller areas than the CPU, and buys area with
  I.** Doubling J at fixed I costs 2.03x the time for 1.47x the relations
  (finding 57); the CPU's reason for large rectangles is amortising its per-q
  root transform, 12% of its wall and about 1% of ours. At equal area the wide
  rectangle beats the square, +11.7% relations at `2^28` and +16.3% at `2^30`
  for −0.3% device time and less VRAM (findings 64-65), because `i`
  multiplies the shorter vector of the reduced q-lattice. Take the larger J
  only when the q range, not speed, binds.

- **Our relations filter and factor.** The C123 `223187...173681` was sieved
  entirely by this siever (29,933 special-q over `[400000, 800000]`, CADO for
  polynomial selection, msieve for everything downstream) and msieve took it
  to `p42 * p82` on 2026-08-05: 29,339,493 relations, 24.0% duplicates,
  22,387,804 unique, 849,784 cycles against 825,717 needed (2.9% surplus).
  It settles that the relations are filterable and sufficient; it does not
  say how many unique relations GGNFS would have needed.

- **The C143 campaigns** (findings 105, 107, 108; 2026-10-12/13): both
  sievers' completed 78M-raw campaigns hold 54.24M (CADO) and 54.35M (ours)
  unique relations; with CADO's skip off the two I13 relation sets overlap
  99.9%, and our extras are marginal (87% with two algebraic large primes).
  Singleton removal on both is row 15.

- **Choosing lpb and the relation target** (findings 98-99). Per-step growth
  in required relations, controlled on Murphy-E, is 1.62-1.74x (finding 99),
  so the lower `lpb` rung wins across essentially the whole range, by under
  1% at the bottom to 9% at the top; Kyle's 1.75-1.80 rule is mildly
  conservative, and the ideal-count model's 1.94x is an upper bound. The
  harness counts raw relations per q while targets are unique: the two
  sources disagree on the level of that correction (~59% unique on a 6x span
  by Kyle's dedup sheet, ~78% by msieve's duplicate line on the same 40 jobs,
  most likely because NFS@Home dedups server-side first) but agree on the
  slope (GNFS ~1 pt per `lpb`, ~9.8 pt from a 3x to a 6x span), so rankings
  are safe and absolute GPU-day counts are not. AS276 factored at `33/35`,
  A=32, on 1.671B raw relations over q = 80-427M, 0.913x the target this
  project carried. Compare a current c183 build against finding 70's applied
  table (14.30 ms/q stage; 14.60 in finding 98), not its sweep row (19.71,
  `mfb 93` rather than the shipped 92).

- **Power and the undervolt.** The card has run at stock since 2026-09-11
  (the undervolt was removed on Kyle's hypothesis that it caused this box's
  hangs; the hangs are the card's, item 12d). Numbers taken 2026-08-17 to
  2026-09-11 are undervolted: stock is 5.10% faster and draws 29.57% more
  (finding 94). Whole-box rel/J needs the UPS meter (item 6), so the
  cross-card rel/J table from rented machines is board-only, an architecture
  note and not a verdict input: by board power the 5070 and 5090 tie (28.1
  against 29.5) and the 4090 trails (18.1); with this box's host constant the
  5090 would lead by ~1.3x.

- **Fill does not scale with SM count** (findings 52, 76, 81): a 5090 has
  3.5x a 5070's SMs and returns far less on fill, against 3.33x on the
  transform and 2.01x on apply. At 1152 blocks (finding 52), 32 threads per
  fill block were necessary: the 5090 cost 23% at 256 threads.

## Known defects

- **Remaining-norm approximation.** The largest-term approximation differs from
  the true rectangle maximum by ~2 bits and costs ~0.13% of CADO's relations.
  The band-wide fixed scale (derived from the band's first q and carried in the
  checkpoint) is a second, unchecked approximation. This is the main known
  yield loss; item 5.
- **Duplicate share 15.8–25%** from sieving the full special-q-side factor
  base, at ~1.34x the downstream TD/cofactor work for the same unique yield.
  These are whole-FB-band numbers: raw inflation runs 1.0021x on the c147's
  0.4%-wide band, 1.1877x on the c151's full-FB band and 1.2325x on the snfs2
  corpus. Truncating at `q` does not fix it (finding 67): on a real 275M-relation
  band it removes only 1.78% of the duplicate finds, and on a band stopping
  below `lim` it destroys more unique relations than it saves. A corpus can
  also carry duplication that is not ours (the snfs2 corpus holds a
  hand-stitched restart's re-sieved window, 1.4%); the `k = 1` stratum
  detects it.
- **Under BOINC every suspend quits a GPU task without a checkpoint**, so each
  suspend loses up to ~30 s of sieving at c135 and a minute or more at I16.
  Handling BOINC's quit as a clean stop is open with Greg (row 35).
- **Pipelined 2,1147+ sometimes takes another 1.1-1.3 GB after setup** that is
  none of the program's allocations; cause not identified (row 32).
- **`--mode twolevel`'s `k_fill_l1` has a live race** and has never been
  swept; benchmark-only, nothing in `--pipeline` reaches it (item 17).
- **Power is board-only** in every cross-card figure; host draw is measured
  only on this box (item 6). The A100 has no sampled power.
- **Host contention.** On the 9800X3D it costs a few percent at the geometry
  we run: ~3% near one runnable thread per logical CPU, ~6% at 1.7x (finding
  96, c183). On a 4-core i5-2500K the cost is time-slicing: competing work at
  nice 0 on twice the cores costs C143 37%, C147 22%, C183 6.7%, and at nice 19
  1.6%, 0.8%, 0% (finding 152). GPU counters are blind to it: quote
  `acc/wall` (run log and band summary) beside any timing from a shared box,
  and never compare wall clock across boxes without host load on both.
  Finding 53's much larger c147 figures remain unexplained.
- **`normscan.c`'s calibration comment for the 2,1139+ octic does not
  reproduce** ("98 bits clear, beta 5.0" beside a measured beta 5.9; the 2026-09-01
  survey gives 49.5-57.5 bits clear at beta 5.96-5.99). No behavioural defect:
  the verdicts are the intended ones. Flagged in place; re-derive from a fresh
  survey before tightening the rule.
- **Environment.** Restart WSL (`wsl --shutdown`) after any Windows display-driver
  update before trusting anything measured here (an update under a running
  session once left `nvidia-smi` segfaulting while CUDA still ran), and do not
  compare absolute timings across a driver or toolkit change: interleave
  against a fresh same-session baseline. Windows-side GPU users are invisible to
  `nvidia-smi` in WSL: a Firefox tab at 4.0-4.4% of the 3D engine coincided
  with runs 6-8% slower (finding 137). Read the Windows GPU Engine counters.

## Open experiments, in order

**This list is the task list of record.** Session-local task trackers do not
survive; if work is worth returning to, it belongs here. Two lists drifted
apart once already — items 7–9 below existed only in a chat session and were
absent from every document in the repo.

### NOW / NEXT — agreed 2026-09-03, updated 2026-11-10

The numbered list below is the full record and contains many *closed* items;
this block is the short open-only view. Order is by what is actually blocking,
not by size. Closed rows here and closed items below are cut to their outcome,
their finding and anything they left open (rows 1-32 on 2026-10-29, the rest
on 2026-11-10); the finding in `RESULTS.md` has the rest.

| # | do | needs | state |
|---|---|---|---|
| 1 | Add and run a `PIPE_Q_SKIP` gate | local GPU, minutes | **DONE 2026-09-04, finding 93** — `skipcheck.sh` / `make skipcheck`, 4 cases, all pass at `BN_LIMBS=4` |
| 2 | Three doc fixes: RUNBOOK fill grid, README BOINC checkpoint, STATUS rho/ECM | nothing | **DONE 2026-09-03** |
| 3 | Extend `walk_cases[]` to `{15,32768}` and `{16,32768}`; comment the `uint32_t xmax` ceiling | nothing | **DONE 2026-09-05** — four production geometries added, 12 cases pass; the `I*J <= 2^31` ceiling (`check_one()`'s `uint32_t xmax`) is commented in place |
| 3b | Decide whether a capped band should advance faster than ~`PIPE_SKIP_MAX` q per invocation | policy | **DECIDED 2026-09-05: it should not advance at all** — a capped band exits `BENCH_EXIT_UNSUPPORTED` (3), so a client stops reissuing it to the same app version |
| 3c | Exit outcomes: a finished band, a checkpointed stop and a too-narrow build must not all be `boinc_finish(0)` | nothing | **DONE 2026-09-05, untested under a real BOINC client** — `enum bench_outcome` (`bench.h`): stop → `boinc_temporary_exit`, cap → `boinc_finish(3)`, only a completed band reports fraction 1.0, and `--stop-file` defers at startup (`skipcheck.sh` case C). The `HAVE_BOINC` branch is type-checked only against a stub `boinc_api.h`. Open with Greg: `boinc_temporary_exit`'s signature against upstream, and the convention for "build too narrow" so the scheduler reassigns to a wider app version |
| 4 | Three-position `--qspan` delay calibration (before first launch, between, after last) | local GPU, idle box | **optional** — settles the unreconciled `wall - span`; frame it as testing event-endpoint/submission semantics, not as perf work |
| 5 | Next rental: **concurrent fill primary, concurrent resieve as a second arm**, interleaved, fresh baseline | rented card (3090/L40S/4090) | **ANSWERED ON A 5090, 2026-09-10, finding 94** — `--fill-concurrent` (the two sides on two streams): −7.62% of wall on c183 I15e; rel/J withdrawn, the 5090's only power reading (`board=`) being aliased by tens of percent |
| 6 | Leave `pipeline.cuh:1924`'s `cudaDeviceSynchronize` alone | — | **decided, no action** |
| 7 | Stop sieving positions `gcd(i,j)` rules out | local GPU | **DONE 2026-09-23, finding 100, now the default** — `--sieve-skip 2`: wall −12.3% (c183), −13.1% (C194), −10.0% (AS276 I17), byte-identical; `--sieve-skip 0` restores the old behaviour |
| 8 | Profile the fused small-prime sieve | local GPU, `ncu` | **DONE 2026-09-23, finding 101** — the block tier's per-entry setup shared by warp shuffle: apply −24%, wall −9.3% (c183), −7.8% (C194), −7.0% (AS276), byte-identical |
| 8a | Share the warp tier's setup; fix the block tier's load balance | local GPU | **DONE 2026-09-23, finding 102** — the warp tier's setup shared too: wall −1.7 to −2.9%, byte-identical. Rotating the block tier's start thread was slower and is dropped |
| 8b | Pattern-sieve the tiniest moduli | local GPU | **stages A and B DONE, findings 129 (`--tiny-fold`, powers of two <= 32) and 131 (`--tiny-table`, 3, 5, 7, 9), both default** — apply −3.8 to −7.8% and −2.6 to −4.5%. Dropping {1,2,3,4,5,7} outright would be worth at most ~15% of apply. Left: the fold's remaining per-entry cost (row 29) |
| 9 | Reclaim bucket VRAM freed by the skip | local GPU | **DONE 2026-10-27, finding 132 (`--row-cap`, default)** — each bucket region gets its row class's share, the array 67.4% of the uniform one (row 32) |
| 10 | Lazy / bounded norm evaluation | local GPU | **parked by finding 104** — five byte-identical designs; the best takes c183 wall −4.3% but C194 apply +3% and AS276 +16%, and a per-side switch does not help, so none ships. Next: why barrier stall grows with job size (in the lazy designs only: the shipping eager `k_apply`'s falls with job size, 1.73 → 1.42 → 1.23 per issue at C135, C148 and I16. The tier cuts raised it, a net win, and restoring the lost issue rate would save about 0.6-0.8% of wall; finding 147). Patches `bench/attic/lazy_norm_v4.patch`, `lazy_norm_v5.patch` (default off). A different lever, row 36: side 0's norm gated by side 1's survivors |
| 11 | `--bkthresh` below `I` | nothing | **FIXED 2026-09-23** — below `I` it silently lost relations (the Franke-Kleinjung walk needs `p >= I`); now refused at startup and in `run_pipeline_impl`, and capped at 2^30. Raising it is correct but not faster |
| 12 | Complete blocking event waits and measure host CPU use | CPU/API trace; idle and contended C143 + C183/C194 bands | **parked 2026-11-04, Kyle's call: freeing part of a core is not worth wall time, so the blocking events are not built and `--blocking-sync` stays opt-in.** Proposed 2026-10-12: NFS@Home objective is sustained use below one logical CPU. Existing `--blocking-sync` leaves default event waits spinning. Then evaluate fewer count readbacks, CUDA graphs and a bounded queue driven by one host thread. [Item 4](#low-cpu-plan), [finding 106](RESULTS.md#finding-106). **Native Linux, 2026-11-04 ([finding 151](RESULTS.md#finding-151), rented 3060):** a blocking wait sleeps (0.003 of a core over 10 ms waits, ~23 µs more per wake). Plain events spin at a full core under every schedule flag, so the existing `--blocking-sync` leaves them spinning. Even so, it takes C143 from 1.00 to 0.69 of a core, with wall 0 / +0.7% over two rounds. Not built: under `--blocking-sync`, the host-awaited events created with `cudaEventBlockingSync`. **Not checked on WSL2** (CADO's sievers spin at 80-90% of a core with blocking sync on, finding 107; `work/waitbench/run.sh` on an idle 5070) or on native Windows |
| 13 | Fix fill's tail, gate side 0's fill | local GPU | **DONE 2026-10-13, finding 107, now the defaults** — row-banded fill (`--fill-rows`) and resieve (`--resieve-rows`), CADO's side-0 fill gate (`--fill-gate`): C143 first band −25.7% (I13) / −10.5% (I14), c183 −6.4%, C194 −5.5%, byte-identical |
| 14 | Several special-q in flight | local GPU | **DONE 2026-10-13, finding 108, now the default** — the next q's sieve runs under this q's TD: C143 −19.6% (I13) / −12.8% (I14), c183 −8.1%, byte-identical. Multi-slab plans: row 17 |
| 15 | Filter both C143 campaigns | CPU (msieve or CADO purge) | **open** — 54.35M against 54.24M unique; our extras are marginal 2LP relations. Excess after singleton removal prices them |
| 16 | `--fill-concurrent` against the gate on wider cards | rented card | **answered on the 5070 at C143, open elsewhere** — at C143 the flag (gate off) was 1-2% faster than finding 107's default (9.08 vs 9.26 ms/q at I13), but the two-q pipeline beats both (7.46) and the flag excludes it (finding 108). Not measured on slabbed jobs (C194 and larger), which are pipelined too now (finding 109), nor on wider cards |
| 17 | Overlap across slabs for multi-slab plans | local GPU | **DONE 2026-10-13, finding 109, default** — each slab's sieve runs under the previous slab's TD: C194 −2.6%, the NFS@Home lpba-34 I16 job −4.2%, byte-identical |
| 18 | Profile apply at I16 (the NFS@Home geometry) | local GPU, `ncu` | **DONE 2026-10-14, findings 110-111** — apply at I16 costs per position what it costs at c183 (issue-bound at ~88%). The load hoist (apply −2.7%) and the two-step tail (apply −2.4 to −3.1% at I16) are landed; the threshold scan followed in finding 117 (row 22) |
| 19 | Price CADO's overlap compression before building a deeper queue | local GPU, `nsys` | **answered 2026-10-14, findings 112-114** — mostly our card idling, not CADO's overlap compression: at C143, equal yield, 1.38x CADO per raw relation, per kernel 1.12x; two of the three host waits per q removed |
| 20 | The NFS@Home I16 job against cado-gpu before TD in async phases | local GPU, `nsys` | **measured 2026-10-16, finding 115** — at equal yield 1.22x CADO per raw relation at I16; neither card idles, so the gap is kernel work, most of it cofactorisation |
| 21 | Resieve + scatter and transform at I16 (finding 115's kernel leads) | local GPU, `ncu` | **DONE 2026-10-16, finding 116** — the transform's inverse branch-free (7.43 -> 3.13 ms/q at I16, under CADO's 3.37) and a shared-memory filter for the resieve (33.6 -> 20.9 ms/q): pipelined wall I16 −9.7%, C194 −7.6%, c183 −7.9%, byte-identical |
| 22 | Exact wins outside CADO: anything duplicated, read twice or waited for | local GPU, `ncu` | **DONE 2026-10-16, finding 117** — apply's threshold scan takes two cells per lane, and two whole-bitmap clears go: `k_apply` −7.6% at I16 pipelined, byte-identical |
| 23 | Side 0's resieve from its gated bucket records | local GPU | **DONE 2026-10-18, findings 118-119** — `--resieve-scan` (default with the fill gate): side 0's resieve reads its gated bucket records instead of walking its factor base: pipelined wall I16 −4.0%, C194 −3.2%, c183 −2.8%, C143 −6.6%, byte-identical |
| 24 | Card idle after findings 116-119, async TD's memory cost, and two host waits | local GPU, `nsys` | **DONE 2026-10-20, finding 120** — card idle C143 10.1%, I16 2.0%; two thirds of C143's is host work at the q boundary. Async TD needs no buffer doubling. Two small waits removed, wall within noise |
| 25 | The q boundary at small jobs, and a c135 against cado-gpu | local GPU | **DONE 2026-10-20, finding 121** — `--q-prep auto` (default): when a q's sieve has finished by the time the host reaches it, the next q is prepared during that q's TD. c135 −4.5% (host load ~10), C143 −0.8%, byte-identical |
| 26 | Small jobs: why cado-gpu is 2x per special-q at c135, and fill ahead | local GPU, `nsys`, `ncu` | **measured 2026-10-20, finding 122; fill ahead DONE, finding 123 (default under `--q-pipeline`)** — cado-gpu's 2x per q at c135 is concurrency, not kernel cost (3.66 against 3.56 ms per q isolated). Fill ahead prepares the q after next during this q's TD: C143 −7.0%, c135 −6.3%, c183 −1.3% on a quiet host. Idle work since: row 33 |
| 27 | Where we stand against cado-gpu at equal yield (its `p < 96` skip off) | local GPU | **measured 2026-10-20, finding 124** — quiet host, raw relations per second: cado-gpu 1.12x ours at I16, 1.37x at c135, yields equal within 2%; its `p < 96` skip is worth 1.13x (c135) to 1.27x (I16) per relation to it. **Latest, 2026-10-29 (finding 137, `7cb58341`, Windows clock):** cado-gpu 1.19x ours at c135 (1.26x in finding 133), ours 1.066x cado-gpu at I16 (1.015x in finding 127); the emitter change after it is worth −5.1% at C135. **Latest, 2026-11-01 (finding 147, `0c7dda1f`, same method, 6 rounds, with `7cb58341` as an anchor that reproduced finding 137's ratios within 1%):** cado-gpu 1.031x ours at c135, within the rounds' noise (it was faster in 4 of 6); ours 1.139x cado-gpu at I16 |
| 28 | Larger jobs: apply's slice cliff, and 2,1147+ | local GPU | **DONE 2026-10-21, finding 128** — slices start only where the log changes, so apply keeps its third block per SM on factor bases past ~30M entries a side: 2,1147+ wall −6.0%, apply −15.8% |
| 29 | The tiny moduli (item 8b) and an outside review's leads | local GPU | **stage A DONE 2026-10-21, finding 129** — `--tiny-fold` (default): pipelined wall I16 −4.5%, C194 −3.1%, AS276 −3.6%, c183 −2.9%, 2,1147+ −6.8%; stage B is row 31. Left: the fold's remaining per-entry cost. (Side 0's norm only under side 1's survivors was done as row 36, findings 140-141.) |
| 30 | Bucket records carry log p (finding 129's review, item 7) | local GPU | **DONE 2026-10-22, finding 130** — fill writes each prime's log into its bucket record and apply adds `r >> 16`; the slice table and its shared memory are gone, speed unchanged within ~1% |
| 31 | Small-sieve moduli 3, 5, 7, 9 from tables (item 8b stage B) | local GPU | **DONE 2026-10-22, finding 131 (`--tiny-table`, default)** — apply c135 −4.3%, c183 −4.4%, C194 −2.7%, I16 −1.6%; I16 pipelined wall −2.1% |
| 32 | Bucket regions by row class; the side-0 record index sized by them (rows 9, 28) | local GPU | **DONE 2026-10-27, finding 132 (`--row-cap`, default; `--no-row-cap` the uniform layout)** — each bucket region is sized by its row class, and side 0's record index shares the bucket's slots: 2,1147+ resieves every slab from records, wall −4.3% single-queued, −4.5% pipelined; bucket + index −10% (2,1147+) to −18-19% (c135, c183). Open: pipelined 2,1147+ sometimes takes another 1.1-1.3 GB after setup that is none of the program's allocations (cause not identified). Measured 2026-10-30 (finding 140): no class is undersized on the NFS@Home I16 job or SNFS301 (q 60-450M). Side 1's fullest region is 88-90% of its class's capacity on every job, and side 0's at most 85% on these two (SNFS301 at q 60M); 2,1147+'s side 0 reaches 86%. No overflow in 3,030 fills a side. An undersized class would show up only as skipped slabs (the exit status changes at 1,000, `PIPE_SLAB_SKIP_MAX`), so the skip counters remain the gate |
| 33 | Small-job idle at c135: cross-stream waits and late launches | local GPU, `nsys` | **DONE 2026-10-28/29, findings 133-137** — c135 idle 8.4% -> 3.6%; C135 −1.9% then −2.9%, C143 −0.7% then −1.1%, relations identical. `--td-ahead` (TD's pass ahead, default; `bench/qhcheck.sh`, card needed). Open, each under 1%: the fill prologue's copies and memsets on `st_sv` (0.3%) and the memset before the early apply (0.2%), one zeroing kernel built in `work/hold/src/` (`PIPE_ZERO`) and not measured; TD's survivor-count round trip (0.42% at c135, item 4); single CUDA calls stalling 1-13 ms on the host (~1% of one C143 run) |
| 34 | The cofactor flush's relation output | local GPU, `nsys` | **DONE 2026-10-29, finding 137** — relations formatted by hand instead of `fprintf`/`qsort` (0.195 -> 0.053 ms/q at C135): C135 −5.1%, C143 −4.5%, C148 −2.7%, C183 and I16 flat, byte-identical. Follow-ups rows 35, 37, 38 |
| 35 | The `--cofactor` band's in-loop checkpoint | local GPU | **DONE 2026-10-29, finding 138, Kyle's call** — the checkpoint after a mid-band flush waits at least 30 s (`PIPE_CKPT_MS`): c135 −0.7%, C143 −0.4%, C148 −1.3%; a crash loses up to ~30 s more sieving. Its crash gate found and fixed an unresumable crash before a band's first checkpoint (the resume sampler). Open, with Greg: handling BOINC's quit as a clean stop (Known defects) |
| 36 | Side 0's norm gated by side 1's survivors | local GPU, `ncu` | **DONE 2026-10-30/31, findings 140-141** — `--norm-gate`, default wherever the fill gate is: a cell whose bit in side 1's bitmap is clear starts at 0 with no norm. Wall −1.4% (SNFS301) to −3.2% (I16), byte-identical. The gate is a template parameter (`NGATE`): a runtime test cost both sides' apply 2.5-4.0% with the gate off |
| 37 | The cofactor flush's emit on a writer thread | local GPU, `nsys` | **DONE 2026-10-31, finding 142** — `cofq_flush` hands each batch to a writer thread; the next flush, every checkpoint and the band's end wait for it. Wall c135 −1.6%, C143 −1.3%, C148 −0.8%, I16 flat, byte-identical; c135's idle gap after a flush 42.0 -> 1.8 ms. The writer and watchdog threads start with every signal blocked (BOINC's SIGALRM) |
| 38 | The cofactor flush's tail rounds | local GPU, `ncu` | **closed 2026-10-31, Kyle's call: not worth the complexity (finding 143)** — rounds 2+ run on one SM for ~27 ms a flush (~1.2% at c135, nothing at I16); spreading a round over the SMs made `k_cofac` +5.4%. Not pursued: an asynchronous flush with a second queue (~86 MB), at most ~1.2%; curves in parallel (changes results) |
| 39 | The norm's degree as a compile-time constant | local GPU, `ncu` | **DONE 2026-11-01, finding 144** — `k_apply`'s init loop over a compile-time degree, one of eight copies picked per block: wall −1.5% (C135) to −3.0% (I16), byte-identical, 3 blocks/SM kept; sm_120 compile +17%. The tiny fold priced alongside (1.9-4.1% of a side's instructions), parked |
| 40 | The small-sieve tier cutoffs and the fill grid, re-swept | local GPU | **DONE 2026-11-01, finding 145** — `SS_BLOCK_CUT` 64 -> 36 (the floor) and `SS_WARP_CUT` 1024 -> 256: wall −1.1% (C1147) to −5.3% (C143), byte-identical. `--fill-blocks` 4608 stands. Measured on sm_120 only (row 44) |
| 41 | The tier cutoffs as runtime flags, and `bench/tune.sh` | local GPU | **DONE 2026-11-01, finding 146** — `--ss-block-cut` (33-128, default 36) and `--ss-warp-cut` (above it, at most 32768, default 256), host code only; `tune.sh` (Architecture). On the 5070 it keeps the defaults at C135 and C148 |
| 42 | Where findings 137-146 leave us | local GPU, `nsys`, `ncu` | **measured 2026-11-01, finding 147** — see row 27 and Performance accounting. Closed there: wider fill blocks (C135 +0.8 to +2.5%), the resieve coarse filter below 2^29 (+1.8 to +3.5%), `UNROLL` 8 (+0.4 to +2.0%), `k_apply`'s barrier stall (0.6-0.8%, not a lever) |
| 43 | Fill redesign: what binds it, and what a redesign buys | local GPU, `ncu`, `nsys` | **measured 2026-11-02, finding 148** — L2 request rate, additive per request (Performance accounting); fill beside apply +28-30 ms/q whatever the priority. `--fill-gate-shift` is job-dependent (C148 −1.05% at 1, I16 −0.67% at 2), default 3 kept. Not tried: a buffered two-level partition (sorted in shared memory, whole sectors written), estimated poor. What landed: row 45 |
| 44 | `tune.sh` on other generations | rented cards | **open; findings 149-150 and their follow-ups, 2026-11-02 to 11-04** — RTX 3060 (sm_86, 2 MB of L2): the tier defaults hold; the fill grid wins where the open bucket lines (4 MB at C183 and C194) outgrow L2: `--fill-blocks 589824` (every bucketed entry its own thread) C194 −19.02%, C183 −6.30% merged and −9.86% with `--no-fill-merge`; C143 (1 MB) loses. RTX 4070 SUPER (sm_89, 48 MB): `--ss-warp-cut 1024` on the larger jobs, C147 −2.94%, C183 −3.52%, C194 −6.64% (C143 wants 512-768, −1.70 to −1.98%), all in apply, 6 of 6; C194 also gains from 589824 (−4.17%, all in fill, so the footprint rule fails there), and the two add (−10.85%); C183 keeps the fill defaults. The 5070 (sm_120, 48 MB) loses with both (C194 589824 +5.83% on a quiet rerun; finding 145's sweep had the warp cut bottoming out at 256-384). The 3060 loses at warp cut 1024 even with fill out of the way (apply +1.6%); Ada's preference is unexplained. Next: defaults chosen per card (sm_89 warp cut 1024; sm_86 and sm_120 keep 256; the fill grid by footprint against L2), Kyle's call, after a second Ada card agrees; a small-L2 card |
| 45 | One launch for fill, the gcd skip on 5 and 7, and `tune.sh`'s larger grid | local GPU | **DONE 2026-11-02, finding 150** — `--fill-merge` (`k_fill_merged`, one-slab plans) and `--fill-skip57` (several slabs per q), output-identical. Wall against `0c7dda1f`: C135 −0.6, C143 −2.4, C148 −3.6, C183 −1.0, I16 −1.2, SNFS301 −2.0, C194 −1.5, C1147 −2.2%. `tune.sh` gains `--fill-blocks 36864`, `--fill-gate-shift` 1, 2, 4 and `--no-fill-merge`. Measured on sm_120 only |
| 46 | ECM products from Greg's gmp-ecm GPU branch | local GPU | **DONE 2026-11-09/10, findings 153-154; fat `bench/bench` `5512830e` installed 2026-11-10 09:41, Kyle's call** — the even/odd carry-chain products (Cofactorisation): wall AS276 −8.6%, I16 −1.6%, c135 −1.2%, relations byte-identical; mfb capped at 127. Then x_dbl's redundant product (13 -> 12 per bit): flush −3.7 to −4.7%, wall below the noise, byte-identical. Not adopted: Greg's param-3 ladder, the dedicated square (finding 154). Evidence `work/eomul/`, `work/xdbl/`, `work/sqr/` |
| 47 | The transform's inverse in 2-bit windows | local GPU | **DONE 2026-11-10/11, finding 155; fat `bench/bench` `2c590187` installed 2026-11-11 09:54, Kyle's call** — `pl_transform_fast_try`'s exponentiation in fixed 2-bit windows over `{d, d^2, d^3}`, still branch-free (21-24% fewer montmuls per inverse by the parity of q's bit length), and `pl_redc`'s high word by `PL_MULHI32` (`__umulhi` on the device; nvcc's 64-bit compare made a montmul 9 instructions where 6 do). On the device only `k_transform` changes (inverse loop 23 SASS instructions per bit -> 29 per 2 bits; fat: sm_90 `k_transform<1>` REG 30 -> 34, untimed); on the host the per-q small-sieve tables run the same loop. Bit-identical by construction; eqv and devcheck on the same 92M inputs (hash-checked) 0 mismatches, 9 of 9 host mutants, abcheck 36 of 36, 40 timed runs byte-identical. Transform + plattice stage −8.3 to −10.5% (c143 −6.4%); per-q wall −1.0% c135 (every round), −1.3% C148, noise elsewhere; whole-process time by `/usr/bin/time` +0.5 to +4.6%, the size and shape of the wall clock's ~2.5 s steps (this host stepped CLOCK_REALTIME back every ~32 s until Kyle fixed its clock 2026-11-11), so not evidence either way. After Kyle's `/code-review`: `bench/xftest.cu` + `make xfcheck` (device == host, in `make check`), `pl_sel3`, stale comments fixed. On the card before the install: xfcheck 0 of 4 failed (9.7M transform inputs, device == host), and all 7 device-only mutants in `work/twin/xfmutants.py` caught. `work/twin/install_card.sh` then gated it: SASS and resources differ only in `k_transform`; abcheck 36/36; the c135 20000-q reference; kill -9 and clean-stop resumes; SNFS-301; make check. It installed it after those passed; `5512830e` is kept as `work/xdbl/bench.installed.154`. Work in `work/twin/` |

**On (6).** The barrier keeps `k_cof_enqueue`'s device work inside the
host-timed `tm.cofac` bucket. Removing it could overlap only the small enqueue
tail against later host preparation -- it cannot reclaim the 0.583 ms
`k_cof_enqueue -> k_transform` gap, which *starts* after that kernel finishes
and is the q boundary with the GPU already empty. Not worth disturbing the
accounting.

**Provenance.** Items 1-4 and 6 came out of an outside review of this repo on
2026-09-03 and two rounds of correction. The review was right about the three
stale doc lines and about `PIPE_Q_SKIP`; it initially ranked resieve above
fill and proposed removing the enqueue barrier, and withdrew both. Its
`qsort` proposal and the first rebuttal of it were *both* wrong -- see
finding 92.

0. **The verdict band against the CPU — RUN 2026-08-20 to 09-02, findings 71,
   83 and 87.** Both sievers over the same q bands (same first q, width
   `10000 * ln q`, ~10,000 (q, rho) pairs each), c183 `I15e`, GGNFS at 16
   workers on this 9800X3D, neither running while the other was on the box,
   watts from the UPS with the monitors' 45 W subtracted:

   | | wall ms/pair, GPU / CPU | time | J per unique relation, GPU / CPU | energy |
   |---|---:|---:|---:|---:|
   | q = 130M | 103.13 / 298.98 | **2.90x** | 0.5649 / 1.5624 | **2.77x** |
   | q = 190M | 100.95 / 301.47 | **2.99x** | 0.5771 / 1.6958 | **2.94x** |
   | `2^15 x 2^14`, each siever's own best shape | | **3.09x** | | **2.89x** |
   | `2^16 x 2^14` | | **3.03x** | | **2.77x** |

   Yield agrees to 0.07-0.37%. **Quote ~3x time and ~2.8-2.9x energy**; the
   margin is rectangle-invariant and grows with q, because GGNFS's truncated
   factor base grows toward the full one (271.95 -> 298.98 -> 301.47 ms/pair at
   50M / 130M / 190M). These predate findings 100-154, which made the GPU side
   several times faster; the margin has not been re-measured. Rules that
   survive: **state the unit** (per (q, rho) pair or per prime q: the c183
   averages 1.528 roots per prime, and mixing them cost two corrections);
   **grade where the factor-base conventions agree** (above `alim`, or the
   q=50M band's 3.12x measures GGNFS's truncation, not the hardware); state the
   geometry in the log header and grade against the matching CPU row; dedup
   before quoting (raw is inflated 1.19-1.34x at band scale, 1.0002-1.0009 at
   probe width).
1. **Concurrent-q throughput — CLOSED.** Measured 2026-09-01 (finding 84): the
   knee is per kernel, and the gain saturates at two streams. Built as
   `--fill-concurrent` (finding 94, row 5) and two units in flight (findings
   108-109, rows 14 and 17). Left open: whether Ada shares the pre-Blackwell
   fill mechanism (the 3090 takes 13.5% off fill under concurrency where the
   5090 takes 23.7%).
2. **Startup fill autotune — BUILT AND DELETED 2026-09-01 (finding 85).** The
   ladder timed repeated fills of one lattice, warm and steady-state, and picked
   the worst of three grids against a real band. `tune.sh` (finding 146) is the
   offline replacement, and the per-card answers are row 44. If an in-band
   tuner is ever built, finding 85 sets its shape: run the candidates
   interleaved on the **same** first q of the band (15 of 16 interleaved reps
   picked the in-band winner; 10 q rank but do not size the effect), suppress
   or deduplicate the tune passes' relations, check whether L2 warmth from the
   previous candidate's fill of the same lattice biases it, and re-derive both
   guards (a >3% margin refuses the real 2-3% wins; a >10% spread guard never
   fires at two reps). `nregion` cannot be tuned this way: it sizes the bucket
   array, `k_apply`'s shared memory and the slab plan before allocation, and
   trades fill against apply, so it needs finding 81's traffic model. The ladder
   code exists in no ref.
3. **Full vs q-truncated factor base — CLOSED 2026-08-18, do not build it
   (finding 67).** Worth 10-15%, not the ~25% an entry count suggests:
   cutting the base 53% cuts the bucket records only 8%, since they go as
   `Σ1/p` (finding 59). Left: the duplicate tool matches msieve's duplicate
   count on the c151 exactly but not its denominator (67,165,877 against the
   67,043,952 on disk), so do not quote it as reproducing msieve until that is
   reconciled.
4. <a id="low-cpu-plan"></a> **Low CPU use and tolerance of competing work —
   reopened 2026-10-12 for NFS@Home.**

   **Deployment objective:** retain GPU cofactorisation and keep sustained
   aggregate process CPU use below one logical CPU, preferably with headroom.
   Measure CPU seconds and the throughput of other host tasks as well as
   sieve relations/s. A small isolated sieve slowdown can be acceptable if
   total useful work on a volunteer host improves. No fractional-core target
   has yet been demonstrated. [Finding 106](RESULTS.md#finding-106) records
   the code inspection, sources and proposed experiments.

   **WSL2 caveat, 2026-10-13 (finding 107).** CADO enables
   `cudaDeviceScheduleBlockingSync` by default, yet its GPU-feeding threads
   use 80-90% of a core each on this box, mostly as system time, while
   waiting in `cudaStreamSynchronize`. Before the step below, check with a
   small wait microbenchmark, or `--blocking-sync` CPU-seconds, that a
   blocking wait actually sleeps under WSL2, and repeat on native Linux.
   **Native Linux done, 2026-11-04 ([finding 151](RESULTS.md#finding-151)):**
   it sleeps, and plain events spin under every flag. WSL2 and native Windows
   are not run (`work/waitbench/`). **Parked 2026-11-04, Kyle's call:**
   freeing part of a core is not worth wall time.

   **2026-11-09, for wall time under load rather than CPU (finding 152):**
   two count reads per q set the size of trial division's launches, after
   the intersect and after the rank scan. They are the remaining points
   where the GPU waits on the host. Traced so far on the 5070, with a quiet
   host and before finding 150, the card's total idle time is 2.8-5.0% of
   wall on small jobs (C143 and c135, findings 134, 135 and 147), 1.4% at
   C148 and 0.5% at I16. TD's survivor-count round trip is 0.4-0.6% at c135.
   C183 and C194 are not traced. If an `nsys` trace of C143 on HEAD shows
   they cost wall on an idle card, first size those launches on the device, as
   `--td-ahead` does for the record pass. Then try a third q in flight where
   VRAM allows, which means the small jobs. More depth without removing the
   count reads only buys slack. A general scheduler for N q in flight is the
   cleaner design, but it would rewrite `run_pipeline_impl`'s staging
   drains, speculative issue and checkpoint replay; not worth it for this.

   **First: complete and measure the blocking-wait option.** The device
   scheduling flag set by `--blocking-sync` does not cover
   `cudaEventSynchronize` on events made with plain `cudaEventCreate`.
   That is how the side, TD and intersection events are currently created.
   NVIDIA documents that they busy-wait unless the event itself has
   `cudaEventBlockingSync`. Make this an opt-in experiment, preserving
   timing where required. The earlier +2.4% wall-time result for the existing
   flag (finding 56) is not a measurement of fully blocking event waits or
   of the CPU capacity they could release.

   Then, guided by a CPU/API trace:

   1. **Reduce GPU-to-host-to-GPU handoffs.** Keep survivor/accepted counts
      on-device where subsequent kernels can consume them. Use bounded
      allocations and checked overflow/retry, batch status readback at a
      completion boundary, and retain all correctness/error gates. Merely
      making a count copy asynchronous does not help if the next launch
      still needs the count on the host.
   2. **Graph repeated stage sequences.** Cofactor rounds are a candidate
      because selection counts already stay on-device; TD needs the
      readback work above. Explicitly handle changing parameters, capacity
      growth and cofactor chunk adaptation. Graph capture of the existing
      synchronizing path is not a drop-in operation. Keep watchdog-sized
      kernels. This renews investigation of historical item 4.2 for CPU use,
      without reviving its withdrawn 1.56 ms/q speedup estimate.
   3. **Queue ahead with one submitting thread.** Two q in flight with one
      host thread is now built and the default (findings 108-109,
      NOW/NEXT rows 14 and 17): 8-20% faster on one-slab plans, 2.6-4.2% on
      slabbed ones, still one host thread
      at ~1.0 CPU, so the CPU objective still rests on the steps around it.
      Deeper queues: start with two or three work descriptors and pinned
      staging buffers, retaining each buffer
      until its GPU consumer has completed. Prepare successors or format
      earlier output while the GPU runs; block only when space/results are
      needed. Ordered work may reuse the bucket allocation; concurrently
      executing whole q requires independent mutable state and a VRAM
      budget. Bound stop latency, and checkpoint only durably completed work.
      Queue depth and useful stream overlap must come from measurements.
   4. **Make host preparation/output cheaper.** Reuse sort scratch instead
      of per-q allocations, investigate sharing the small-prime transforms
      for sieve and TD, and batch formatting/readback. Keep one host worker
      initially. Work already overlapped with GPU execution still consumes
      CPU, so hiding it does not satisfy the CPU-use objective on its own.

   **What the earlier measurements still establish:**

   - Preparation-only overlap (historical item 4.1) has limited measured
     headroom. Finding 89 already hides side-0 and TD preparation under
     transforms. Finding 91 leaves roughly **0.5 ms/q**, about 0.5% on its
     C183 case, exposed in side-1 preparation. Adding preparation threads
     alone has no demonstrated large benefit.
   - Nice, whole-process affinity and real-time priority did not improve
     the measured throughput in finding 56. They are not the proposed
     mechanism. Avoid CPU oversubscription; a logical CPU left idle is not
     a reserved physical core, and WSL adds the host scheduler.
   - Finding 96 is the current C183 contention measurement: about 3% rate
     loss near one runnable thread per logical CPU and 6% at roughly 1.7x.
     Finding 53's much larger penalty remains unexplained and is not a
     universal bound. Small jobs have more frequent host handoffs.
   - **Finding 152 (2026-11-06, i5-2500K, 4 cores, no SMT):** there the
     cost is time-slicing. Competing work at nice 0, twice the cores, costs
     C143 37%, C147 22% and C183 6.7%; at nice 19 it costs 1.6%, 0.8% and
     0%. `--blocking-sync` does not help. Nice works on that box, unlike
     finding 56's, because nothing time-slices below one thread per logical
     CPU on the 9800X3D.

   **Acceptance:** interleave baseline/variant runs on C143 and C183/C194,
   idle and with realistic competing CPU work. Record process user+system
   CPU seconds/wall across all threads, CPU seconds per million relations,
   relation rate, competing-task throughput, GPU gaps and wakeup delays.
   Check relation identity for scheduling-only changes, buffer limits,
   memory use, stop responsiveness and checkpoint/resume. Validate relevant
   Linux and Windows/WSL paths. Investigate finding 105's conflicting elapsed
   clocks before precise cross-timer attribution. All changes above are
   **proposed and unmeasured**, not shipped.

5. **True rectangle maximum** for the norm approximation — one safe band-wide
   scale, or a few scale/slice-table buckets, gated at low/mid/high q on more
   than one job. **Low value at the geometry we deploy, and blocking nothing.**
   The 1:1 anomaly that drove it was a geometry mismatch (finding 65:
   `gnfs-lasieve4I15e -J 15` covers our `2^16 x 2^14`); at matched rectangles
   our yield is 0.979-0.981 of GGNFS's, flat in aspect ratio, area and j, which
   is the shape of a constant downstream residual (finding 62), not of a norm
   approximation degrading with the rectangle. `sieve_allowance` and
   `sieve_bound_checked` are exonerated by arithmetic. The payoff is job-shaped:
   where the special-q side is degree 1 (snfs236) largest-term and
   true-rectangle are the same number. What came out of it is the rule in
   "Measured": buy area with I, not J.
6. **Whole-box power — the meter exists** (the UPS reports real watts; read W,
   not VA, which is ~4% high; subtract the monitors' ~45 W). The host constant
   is **~105 W** (idle box including the GPU, ±7%, average it), and the GPU
   idles at **30 W** on this box, not the ~16 W P8 figure first assumed. At the
   verdict band (item 0, undervolted card): GPU sieving **240 W**, CPU sieving
   at 16 workers **~236 W** (finding 71). The board sensor is trustworthy: the
   card's swing at the wall sits a few percent above its swing on the board
   sensor, as PSU losses predict. Two traps: elastic background load (ECM
   expanded into the sieve's freed core and drew more), so pin or stop it; and
   power falls as a workload gets more memory-bound, so an ALU-bound stand-in
   overstates a siever's draw (ECM drew 33% more per core than GGNFS).
   Ambient matters too: the same voltage curve drew 133.5 W board in August
   and 147-153 W in September heat, so quote board draw beside any rel/J.
7. **Our survivor gate against GGNFS's — CLOSED 2026-08-17 (finding 62).** At
   each siever's own gate our cofactor volume matches GGNFS's within 0.4%
   (c183 I14e) and 0.03% (C194 15e). Do not "fix" the matched-lambda asymmetry
   by tightening the default: the gate finds every relation GGNFS finds.
8. **A = 32 sieve areas and MFB widening — CLOSED 2026-09-01 (finding 82).**
   `lpb` to 64, `mfb` to 127 per side, and `A = 32` slabbed: AS276 at
   `--logI 17 --J 32768 --maxbits 17` plans 8 slabs of 4096 rows unaided, every
   norm rebuilds, and the recovered rectangle is GGNFS's own `2^17 x 2^15`;
   setup allocation 2.71 GB, so a 12 GB card runs it. Left: a like-for-like
   timing against NFS@Home at that geometry on an idle card.
9. **Dead factor-base parses under `--sq-side 0` -- CLOSED 2026-09-02,
   finding 86: both defects are stale, and startup on the client's job class
   is 1.9 s** (0.2% of a 900 s unit). Left, neither worth scheduling: runs
   without `--fb1` (GPU factor-base generation) were not timed, and c194's
   ~3.45 s is a bound rather than a measurement.
10. **GPU power-limit sweep — MEASURED 2026-08-17 (finding 61); a floor, not
    a knee.** The sieve is not power-limited at stock. At the 175 W floor:
    +5.0% whole-box rel/J (the board sensor overstates it 1.8x). The undervolt
    (~2900 MHz at 950 mV) gave +14.6% whole-box rel/J for +6.7% time and was
    removed 2026-09-11 ("Measured": power and the undervolt). Left: where the
    knee is (900 mV untried).
11. **Apply breakdown — CLOSED (findings 60, 75).** Apply is issue-limited with
    a balanced mix, and the cheap win was occupancy: `__launch_bounds__(512, 3)`
    gave apply −12.6%, wall −4.6%. **Check `Block Limit Registers` against
    `Block Limit Shared Mem` before concluding a kernel is out of cheap
    headroom.** `NORM_FAST_LOG2` and `NORM_CANCEL_TOL` are build switches for
    re-pricing the norm's accurate `log2f` and cancellation guard on another
    job.
12. **Unattended operation — checkpoint, resume, clean stop, logging**
    *(added 2026-08-07, owner-stated requirement; scoped 2026-08-11)*. The GPU
    should not idle waiting on a human any more than it should idle waiting on
    the host thread; item 4 is the within-job half of this and this is the
    between-job half. Split into three pieces, in dependency order.

    **12a. Checkpoint / resume / clean stop — BUILT 2026-08-11, VERIFIED ON A
    GPU 2026-08-16**, in `bench/ckpt.h` plus changes to `pipeline.cuh`,
    `bench_main.cu` and `cofac.cuh`.

    **The result is stronger than "no relations lost": a stopped and resumed
    band reproduces the uninterrupted one byte for byte.** A 1500-q c147 band
    (`--cofactor`, 178,406 relations, 23.2 MB) was run three ways — straight
    through; `SIGINT` at q 776 and resumed; `kill -9` at q 671, a 25-byte torn
    line appended by hand to the `.part`, and resumed — and all three files
    have the same MD5. The torn tail was truncated away, and all 178,406
    relations pass `--check-relations`. The same comparison without
    `--cofactor` (where the `--candidates` batch is non-empty) gives
    byte-identical relation *and* candidate files, with no NUL run from a
    mis-sized `ftruncate`.

    Each constraint in the checklist below was exercised as a case rather than
    read: the clean stop drained 125k queued candidates and exited 0 in 0.2 s;
    `--stop-file` stopped at q 896 with a valid resume point, and a stop file
    that already exists refuses to start; the lock named the live pid and
    refused a second writer, released itself on a clean exit, and was taken
    over when the recorded pid was gone; a resume with `--lpb 29` was refused
    on the fingerprint; the single-q path refused to continue a resumable band;
    `--restart` discarded 3.5 MB; `--nq 1500` became 724 remaining after 776
    completed, so the two sessions summed to exactly the band that was asked
    for; and resuming a checkpoint holding 9.6 MB of candidates without
    `--candidates` was refused rather than stranding them. `make check` passes
    (cofcheck's 30 cases, fbgencheck, sqgencheck, fbtest).

    Unverified still: **anything on a multi-day timescale** — every case above
    is seconds to a minute long — and resume across a *reboot* or a different
    machine, where the page cache is cold and the fsync ordering is what is
    actually being tested.

    **One unexplained observation, recorded so it is findable if it recurs.**
    The first interrupt attempt of the session (2026-08-16 22:20, `timeout -s
    INT 15s`, another GPU job having just exited) died with no clean-stop
    output at all: empty `.part`, no sidecar, and the lock left behind. Six
    later interrupts — three direct `kill -INT`, `timeout -s INT`, `timeout
    --foreground -s INT`, and `--stop-file` — all drained and checkpointed
    correctly, so it did not reproduce and the verification above stands. The
    signature to watch for is a band that exits on a signal with no "stopping
    cleanly" line: that means the handler did not run, and the run loses
    everything since the last flush rather than nothing.

    The mechanism rests on one property of the existing code:
    `pipeline.cuh:1228` flushes the cofactor queue *before* enqueuing the
    current q rather than splitting it, so `cofq_flush` never straddles a
    special-q. At every flush the `.part` file therefore holds exactly the
    complete relation set for every q processed so far and nothing partial — a
    checkpoint the run already produced and simply did not record.

    So: after each flush, `fflush` + `fsync` the `.part`, then write a sidecar
    `NAME.part.ckpt` recording the next `(q, rho)`, the byte offset, the
    relation count, and a job fingerprint. Resume truncates the `.part` to that
    offset (which also removes any torn final line) and restarts at that
    `(q, rho)`. Exact, and independent of the factor-base convention.
    `--restart` discards; `--stop-file PATH` stops cleanly for unattended runs;
    a `NAME.lock` refuses a second writer and clears itself if the recorded pid
    is gone.

    **Rejected: inferring the resume point from the primes in the relation
    lines.** It is the obvious approach and it is ambiguous under the
    convention the pipeline actually runs. The full factor base is in force —
    item 3's per-q truncation is not built, all three kernels still take the
    full `fb->n` — so a relation is re-found at *every* band q dividing its
    sq-side norm: measured P(re-found) 70.2%/72.3%, mean 1.82 sq-side primes
    per relation. A line typically carries ~2 in-band primes and nothing
    distinguishes the one that produced it. The failure is asymmetric —
    guessing high skips q and loses relations silently, guessing low only
    re-sieves an overlap that msieve deduplicates — so a guess is survivable
    but not auditable. Note it becomes exact *under* truncation, where the
    producing q is the largest sq-side FB prime; that it depends on an open
    experiment (item 3) is itself the reason not to build resume on it. Keep
    prime-scanning as a separate human-read diagnostic for files with no
    sidecar, reporting a conservative (minimum) resume point.

    Constraints the implementation had to meet, each of which is a way to lose
    or corrupt relations. They are listed because they are the review checklist
    for anything that touches this code again, not because any is outstanding:

    - **The `.part` used to be destroyed.** `pipeline.cuh:1588` removed it on
      any failure and `:1030` opens it `"w"`, which would truncate the file
      being resumed from. The reframe is that `.part` is the durable artifact
      and the rename to the final name is only the "band completed" marker.
      It is now kept on failure **iff a sidecar was actually written** — not
      merely because a relation file was opened. Under `--cofactor` the file is
      empty until the first flush and that flush is what writes the first
      checkpoint, so an unresumable `.part` holds nothing; keeping one would
      strand every rerun on the startup refusal, which is the one wedge an
      unattended queue cannot recover from.
    - **A failed checkpoint write must not fail the band.** A missed checkpoint
      costs at most one flush of re-sieving; aborting costs everything since
      the last good one. It warns once and continues.
    - **Never checkpoint between flushes under `--cofactor`.** At the top of an
      arbitrary q the queue holds up to a flush of candidates that are not in
      the file; recording that position as the resume point drops them.
    - **A clean stop must drain.** SIGINT/SIGTERM sets a flag, checked at the
      top of the q loop after the next `(q, rho)` is pulled: flush the queue,
      fsync, checkpoint that pair, exit 0. A planned stop then loses nothing,
      against up to one flush of re-sieving after a crash (~67 q on the c183,
      but **686 q on the SNFS job**, which enqueues 191 records/q instead of
      1,956 — `pipeline.cuh:1385`). Second signal exits immediately; the
      previous checkpoint stays valid.
    - **The derived scale and allowance must be carried in the checkpoint.**
      They are derived once from the *first* q of the band
      (`bench_main.cu:1312`, `q0 = ql[0].q`) and held band-wide — that is the
      fixed-scale approximation under "Known defects". Re-deriving them from
      the resume q would silently change the survivor gate partway through a
      run, which is exactly the kind of inconsistency an item-0 verdict cannot
      absorb. Measured on the c147: the carried gate is scale 0.900/1.100,
      bound 56/65, against a fresh derivation at the resume q of 1.250/2.925,
      bound 76/169. Not a rounding difference — a different sieve.
    - **A relation that verifies proves the polynomial, not the parameters.**
      `--check-relations` (`bench_main.cu:732`) is host-only and cheap, so
      spot-checking a few lines of the `.part` is worth doing — but logI, J,
      lambda, mfb, lpb, sq-side and the FB convention can all differ while
      every line still reconstructs. Hence the fingerprint; refuse a mismatch
      rather than appending incoherent yield to a valid-looking file. The
      fingerprint therefore covers the polynomial, lim/lpb/mfb on both sides,
      logI/J, sq-side, **`--maxbits`** and **the whole cofactor configuration**
      — dropping `--cofactor` on a resume switches the file from queue-emitted
      to trial-division-only *and* the checkpoint discipline from
      flush-anchored to a timer, and the ECM/rho bounds decide which cofactors
      split at all.
    - **Sample only the checkpointed prefix.** A torn final line past the
      offset is the normal shape of a `kill -9` and is about to be truncated
      away; scanning to EOF would refuse a sound file.
    - **Take the lock before anything destructive.** `--restart` unlinks the
      `.part`, so acquiring the lock after that (or after the factor-base
      parse) lets it delete a running siever's output and report the conflict
      afterwards, with the victim still writing to an unlinked inode and its
      final rename doomed to ENOENT.
    - **Resume needs the original band.** The single-q fallback builds `ql[0]`
      from `--q`/`--rho` and never consults the checkpoint, so resuming into it
      would truncate the `.part`, sieve one unrelated q, call the band complete,
      rename over the final name and delete the sidecar. The band source is not
      in the fingerprint, so this is checked separately.
    - **`--target-rels` counts what is already on disk**, or a resumed run
      overshoots by the whole prior amount.
    - **`--candidates` has its own `.part`** and must be truncated to a
      consistent point in the same checkpoint — with the same length guard the
      relation file gets. Unguarded, a short file is *extended* by `ftruncate`
      and the gap is a run of NULs no parser survives; and because
      `--candidates` is not in the fingerprint, a resume that omitted it would
      record `cand_bytes = 0` and the next resume that supplied it again would
      truncate the whole file away.
    - **A lockfile**, because two processes appending to one `.part` is silent
      corruption and a job queue makes that a realistic accident. It must clear
      itself when the recorded pid is gone: `SIGKILL`, OOM, power loss and the
      second-signal `_exit` all skip the unlink, and a lock needing a human
      defeats unattended operation more thoroughly than the race it prevents.
    - **`--nq` is a per-session budget** once a run can resume. It is reduced by
      the completed count, so the messages that echo it must add that count back
      or they print a number nobody typed. The wind-forward to the exact `(q,
      rho)` also must not be charged against the generator's own limit, or the
      band stops early and reports a range exhausted that is nowhere near it.

    **12b. Logging — BUILT 2026-08-16**, in `bench/runlog.{c,h}` plus the
    reporter and warning sites in `pipeline.cuh` and the header block in
    `bench_main.cu`. `--log PATH` appends a run log; `--log-every S` sets the
    record period (default 300 s, ~864 lines over three days, so **no
    rotation**). Both are `--pipeline`-only and resolved through
    `boinc_resolve_filename_s()` like every other named output.

    A tee was the obvious version and worth less than it looks: finding 53 is
    that host contention costs up to 29% of wall while every GPU counter reads
    normal, and that any cross-box wall or ETA comparison is invalid without
    knowing host load on both. So each record carries `GPU-accounted / wall`
    (the running form of the band summary's line, built from the same terms),
    GPU utilisation, board watts and the host load average alongside q,
    relations, candidates, ms/q, percent and ETA. The header block carries the
    commit (`git describe --dirty`, stamped into `runlog.o` alone so a commit
    costs no CUDA recompile), the full argv, the job fingerprint, the card with
    its PCI ID and driver, the **geometry labelled `I14e`/`I15e`**, the
    **factor-base convention** with entry counts and whether the base is
    truncated, the derived gate, and the resume point. Finding 55 is what that
    header exists to prevent.

    Verified on a GPU 2026-08-16, c147: header, heartbeat, band-end record,
    the `isatty(1)` split (a redirected console gets whole lines every five
    minutes — its own constant, *not* `--log-every` — instead of a `\r` line
    and a stray `\033[K`), and NVML binding. The `resume` note was exercised by
    12a's verification the same day — a resumed session appends its own header
    block, carrying the resume point and the narrowed band. **Not yet
    exercised: a BOINC-resolved log path.**

    Three implementation notes worth keeping:

    - **`nq` and `rel` mean the same thing in every record**, including the
      band-end one: totals across sessions, with `rel` counted the way the
      heartbeat counts it (what reached the file, not the subset trial
      division finished — 374 against 111 on one smoke run). Session-only
      aggregates carry their own `_session` keys. A log where one key means two
      things is worse than one field short, because nothing in the file says
      which reading applies to which line.

    - **NVML is bound by PCI bus ID, never by CUDA ordinal.** NVML enumerates
      by PCI order while CUDA can be reordered by `CUDA_DEVICE_ORDER` and
      renumbered by `CUDA_VISIBLE_DEVICES`, so an ordinal reads another card's
      watts whenever those disagree — silently, and precisely on the multi-GPU
      hosts of item 13. It is `dlopen`ed, so there is no build dependency and a
      host without the driver library logs `n/a`.
    - **Every record carries monotonic elapsed seconds as well as a
      timestamp.** Observed on this box during the verification run: WSL2
      resyncs its clock to the Windows host, and one band's records came out
      with a two-second backwards step mid-run, which is enough to make a
      record appear to precede the one above it. Compute rates from the
      elapsed column; the timestamp is for lining a run up against a meter or
      another machine.

    **12c. The queue itself.** A job list consumed without intervention, so
    finishing one job starts the next rather than leaving the card idle
    overnight. Depends on 12a; unscoped beyond that. **Item 9 stops being
    cosmetic here**: its ~15–20 s of redundant startup is noise in a multi-day
    run and real overhead in anything that restarts the process per job.

    **12d. Stall detection — BUILT 2026-09-05**, in `bench/watchdog.{c,h}`
    plus heartbeat calls in `pipeline.cuh` and `cofac.cuh`. *(Prompted by
    intermittent freezes on this box over the preceding few days: the process
    stays alive and responsive to signals, and simply stops making progress.
    Seen both under `ggnfs-distributed` and running `bench` directly, with no
    identified trigger.)*

    **The problem it solves is that a stall silences every instrument at
    once.** The `\r` progress line, the run-log record and the per-q timers are
    all written by the loop that stopped, so after the fact there is no way to
    tell the two possible bugs apart. A watchdog thread — the one thing not in
    the stall — watches a heartbeat the sieve thread bumps (a pointer store
    and a counter, no lock, nothing that can itself block) and reports the
    phase, the `(q, rho)`, the slab, and **NVML utilisation and board watts**.
    That last pair is the whole point: **near 100% means a kernel is not
    terminating** and the phase names it; **near 0% means the host side is
    stuck**. From the second report — and only once the kill is armed, i.e.
    inside the band — it also signals the sieve thread once for a backtrace:
    frames in `libcuda` say device-side, ours say ours.

    **OFF BY DEFAULT, decided 2026-09-06.** The give-up threshold is a claim
    about how long a phase can legitimately take, measured on one card, one job
    and one geometry; on a slower host or a much larger per-q workload it would
    kill a run that was merely slow, on a machine whose owner never asked for
    it. Disarmed it is two predictable branches per slab and no thread, so
    nothing in the band pays for its existence. `--watchdog S` turns it on for
    a host suspected of freezing.

    **`--watchdog-kill` (default 600 s, once armed) then makes a freeze fail
    instead of hang.** A frozen process holds the card, the work-unit lease and the
    output file for as long as nobody is watching, which is worse than any
    exit; hardware that has begun to fail does not always return an error
    `CUDA_CHECKED` can see. It exits `BENCH_EXIT_STALLED` (4), distinct from 1
    and 3 so a client reads it as "this host wedged, reissue elsewhere". The
    exit is a bare `_exit` from the watchdog thread — it writes no output
    file, so the last checkpoint is intact and 12a's resume replays from the
    last whole special-q, losing at most the q in flight.

    Two design points that are easy to get wrong:

    - **The kill is armed only on entering the band loop.** Factor-base
      generation and a multi-gigabyte `--check-relations` resume scan are
      legitimately slow under one coarse phase label, and killing a healthy
      run there would be the watchdog causing the failure it exists to report.
      Startup still *reports*. The three resume-scan loops in `cofac.cuh` carry
      heartbeats so a long `.part` gate does not read as a stall.
    - **Report and kill thresholds are an order of magnitude apart** because
      they answer different questions. A false report costs eight lines of
      stderr; a false kill costs a special-q. Verified 2026-09-05: a two-q
      AS276 band at a deliberately absurd `--watchdog 3` reported only the ~5 s
      factor-base load and nothing inside the band, and a harness driving the
      real `watchdog.o` produced report → report-with-backtrace → exit 4 on
      schedule.

    **HARDENED after review, 2026-09-06.** Five fixes worth keeping as design
    constraints rather than diff noise, because each is a way the watchdog
    could have become the failure it exists to report:

    - **Nothing on the report or kill path may touch stdio.** `fputs(stderr)`
      takes the FILE lock, and "blocked in printf on a pipe nobody drains" is
      an ordinary way to be host-stuck — the sieve thread would hold the lock
      the watchdog needs, and the kill would never fire. Raw `write(2)` and
      `open`/`write`/`close` throughout.
    - **The kill is tested before the report and before NVML**, so no query and
      no write can sit between a wedged run and the exit that frees the card.
      It also makes a `--watchdog-kill` below `--watchdog` behave as asked
      instead of being silently clamped up to the report threshold.
    - **NVML at most once per stall, and never before the phase/q/slab have
      been written.** Those calls are ioctls into the very driver a stall may
      have wedged. They kept answering throughout the 2026-09-06 incident,
      which is why they are still here at all.
    - **The backtrace is gated on the kill being armed.** `backtrace()` is not
      async-signal-safe — its first call goes through the loader and malloc —
      so firing it during a merely-slow startup, while the sieve thread holds
      either lock, would deadlock the run. `SA_RESTART` too: an interrupted
      `fsync` or driver ioctl must not be the price of a diagnostic.
    - **`runlog_gpu_bind` is called exactly once, before the thread exists.**
      A second bind on the `--log` path raced the running watchdog over
      `L.dev`/`L.nvml_util`, which the failure path NULLs and `dlclose`s.

    **WHAT IT FOUND, 2026-09-06 — the freezes are the CARD, and this box's
    freezes are not a `cuda-sieve` bug.** A GMP-ECM GPU run (a wholly unrelated
    codebase) hung with the identical signature: last progress at 24.8% with a
    steady ETA, then nothing. Live state showed the GPU at 0-1% util clocking
    *down* (P1 to P5, 38 W to 19 W) while the main thread burned 100% user CPU
    with **system time completely flat** — a userspace spin making no syscalls,
    i.e. the CUDA runtime busy-waiting on a completion that never arrived.
    Linux `dmesg` was clean, because under WSL2 the driver is on the Windows
    side: `nvlddmkm` Event 153, "Error occurred on GPUID: 100", **11 times in
    14 days and accelerating** (2 on 8/24, then near-daily, 5 on 9/6), one of
    them 7 minutes after that last ECM progress line.

    So the fault is real and in the card or its driver; WSL2's contribution is
    only that it turns a driver error into a silent hang rather than a
    `CUDA_ERROR` the caller could see. Two consequences worth keeping: **a
    freeze here is not evidence of a bug in this program**, and the same fault
    **cannot explain a wrong relation** — it wedges rather than corrupts, which
    is consistent with a long history of msieve never rejecting a relation from
    this card.

    **12e. The OTHER watchdog -- the DEVICE's, not ours -- MERGED 2026-09-11,
    MEASURED 2026-09-12 (RESULTS finding 95).** 12d is a host thread watching
    for a stall. This is the opposite direction: the HOST OS killing a kernel
    that runs too long. Windows' display watchdog (TDR) terminates a kernel at
    ~2 s by default, surfacing as `cudaErrorLaunchTimeout`; AMD's reports the
    vaguer "unspecified launch failure". Both were seen in the field on slower
    volunteer hardware. A cofactor round's duration is set by its parameters
    (rho runs `budget << r` iterations, ECM runs `curves` curves, both over
    every selected record), so a slow enough device crosses the limit on a
    launch a fast one completes comfortably.

    `--cof-chunk` (Greg, in `cofac.cuh`) slices a round across several launches
    on the RECORD axis, which is the only axis that cannot change a result --
    each record's `mz_split` is independent and untouched. 0 = AUTO, sized from
    the device: the slice floor is `blocks * threads`, one record per thread.

    Verified on the 5070 2026-09-12: relation files are BYTE-IDENTICAL across
    `--cof-chunk` 131072 / 0 / 4096 (one md5, 769,448 bytes, 5245 relations
    over a 300-q band), and `cofcheck.sh` passes 51 of 51. Auto costs **+1.86%
    of the cofactor stage and +0.72% of wall** on this card, where the floor
    (73,728) sits below the flush (130,031) so auto parks at two launches per
    round for the whole band. **An earlier reading of this projected ~20% by
    carrying across the RTX 3090 numbers in `cofac.cuh`; that was wrong by an
    order of magnitude. The 3090's floor is 125,952, and its table's +21.7%
    datapoint is at chunk 65,536 -- BELOW that floor, the regime the floor
    exists to prevent and which auto cannot select. Its 131,072 datapoint is
    at or above the floor and costs +0.5%, which is the one that actually
    establishes the floor works.** Finding 95 has the full partition.
    Recommendation: leave it on auto.

    **What chunking CANNOT bound, in two forms -- the floor is one record, and
    one record's work is whatever its parameters say.** `cofac.cuh` documents
    the ECM form: a large enough `--ecm-b2` makes a single record's stage 2
    exceed a watchdog no matter how the list is sliced. `cofq_init` warns above
    20,000 giant steps/curve; the derived default is ~194-500, and the
    configuration known to kill gfx1103 is `--ecm-b1 400000` (~320,000 steps).

    The RHO form is not documented there and is worth stating (found in review
    2026-09-12). Slices are cut against the batch `n`, but each round compacts
    the survivors into the low indices, so once the live count falls below the
    slice the whole round lands in `[0, step)` and the remaining slices launch
    empty -- `k_cofac`'s own comment concedes this ("what the later rounds
    mostly do"). Those late rounds carry `budget << r`, the LONGEST per-thread
    work in the flush, and they are the ones that end up unsplit. Slicing
    harder cannot help: below one record per thread the launch duration stops
    falling and only idle threads are added.

    This is a real limit on the protection, not a defect in it. At the
    pipeline default of 4 rounds it still cuts the peak launch roughly in half
    on a small device (round 0 unchunked is ~15 records x `budget`; chunked,
    1 x `budget`; round 3 is 1 x `budget << 3`, which is below the unchunked
    round 0). It would stop helping at 6 rounds -- the `--cofac` default,
    which does not auto-chunk anyway. **So `--cof-chunk` bounds the FIRST
    rounds well and the last rounds not at all**, and a device that still trips
    its watchdog should be given fewer rounds or a smaller budget, not a
    smaller chunk.

13. **Validate the BOINC GPU assignment — CLOSED 2026-08-17.** Greg Childers,
    who reported the original failure (every task on a multi-GPU host landing
    on device 0), reviewed and signed off on the assignment change, and a BOINC
    queue runs with it. To re-check the read path after a change to it (a
    silent regression looks exactly like success on a single-card box): a
    `HAVE_BOINC=1` build run in a directory holding a hand-written
    `init_data.xml` exercises the whole path. Pin (a) an NVIDIA assignment read
    and used; (b) an assignment overriding a conflicting `--device`, with the
    notice printed; (c) a non-NVIDIA `gpu_type` refused; (d) no assignment
    falling through to `--device` and then CUDA's default; and an assignment of
    1 on a one-card box giving the bounded "sees only 1 device" refusal.
14. **Use the survivor cell's real resolution — MEASURED AND CLOSED 2026-08-18
    (finding 66): no win, do not build it.** Dropping the 10-13% of survivors
    admitted on `las_scale`'s 8-bit rounding changed nothing (112.44 against
    112.43 ms/q): the survivor list is nearly free downstream, so **survivor
    counts are a poor proxy for work** in any cross-siever comparison.
15. **Older cards.** Both the pipeline and the standalone apply benchmark read
    `cudaDevAttrMaxSharedMemoryPerBlockOptin` from the selected device at run
    time (2026-08-21); `--region 14` stays the default. **Qualified on sm_61
    and sm_75, 2026-09-09:** a GTX 1080 and an RTX 2080 Ti (k8s pods, CUDA
    12.8.93, `GPU_ARCH=61` / `75`) passed `--verify-only`, reproduced
    `oracle/c183.fb1` with `fbgen_gpu` byte for byte, passed `cofcheck.sh`, and
    gave a c183 `--pipeline --cofactor` band the same counts as the 3090/4090.
    `GPU_ARCH_all` is sm_75/80/86/89/90/120 (`Makefile` and
    `build_windows.bat`); `GPU_ARCH=full` adds sm_50/52/60/61/70 and needs CUDA
    <= 12.8 (CUDA 13 rejects sm_61; both build paths refuse a >= 13 nvcc for it
    with an explicit error). Only sm_61 and sm_75 are hardware-confirmed.

    On Turing, `__launch_bounds__(512, 3)` asks for more threads than an SM
    holds (1024), so ptxas ignores the min-blocks hint with twelve warnings,
    nine of them `k_apply`'s. Making the bound depend on `__CUDA_ARCH__` is
    cosmetic: shared memory binds first (`k_apply` at region 14 needs ~33 KB
    against Turing's 64 KB, so one block per SM, 50% occupancy; region 13 gives
    two). **The only real lever on Turing is `--region`, and it is unmeasured**;
    a rental for it is Turing only (20x0 / 2080 Ti / T4). Deferred from the
    2026-09-12 review: `bench.h`'s `TD_RECORD_THREADS` comment omits sm_75, and
    the version gate checks only that nvcc is not too new, not that it is new
    enough for sm_90/sm_120.
16. **Build wall time — MEASURED 2026-08-20: the cause is `CF_LMAX=4`.** The
    fourth limb took sm_120's ptxas from 277 s to 754 s; dropping targets buys
    nothing, since `-t 0` builds them concurrently. Not re-measured since the
    PTX arithmetic (findings 125, 153) or the degree copies (finding 144, sm_120
    kernel compile +17%). **A zero exit from `make` is not evidence a build
    happened** (finding 75): check the binary's mtime before benchmarking
    against it. For iteration, `make GPU_ARCH=120 CF_LMAX=3` is a shippable
    binary; cofcheck pins 3- and 4-limb output byte-identical.
17. **`--mode twolevel` has a live race in `k_fill_l1` — FOUND 2026-08-24, NOT
    fixed, benchmark-only.** `--mode twolevel --verify` misfiles records in
    300-560 of 8192 regions per run, differently every run, in both directions,
    with the total of 77,382,815 exact (`--mode atomic --verify` is exact). Mechanism, from
    code reading: `k_fill_l1` reserves a whole run with one
    `atomicAdd(&cnt[b], nrun)` and gives it back with `atomicSub` when it does
    not fit, but the shared `buf` is never cleared and the flush emits
    `buf[0 .. cnt[b])`, so a third thread taking a slot above a failed thread's
    range leaves a hole below the final count: stale records re-emitted, live
    ones lost, one for one. `k_fill_l2` reserves one slot at a time and never
    subtracts, and shows the fix: write the part of a run that fits and carry
    the remainder. Low priority because the two-level path lost by 2.7x
    (finding 1) and nothing in `--pipeline` reaches it, but the verify gate is
    not in `make check`, so whoever revisits two-level fan-out must fix this
    **before** trusting any number it produces. `k_fill_l1` also keeps the
    32-bit walk (`pl_next`) and its own untuned 144 x 512 grid.
18. **Lanes-per-item for the dense trial-division pass — DO NOT BUILD (finding
    82).** It exists to make slabs below `2^29` affordable, and nothing asks
    for them: a 12 GB card runs `A = 32` at NFS@Home's own shape in 8 slabs of
    `2^29`, the sweep optimum. Its ceiling was ~3% of wall at that geometry
    (finding 78). The recording pass's per-slab tax was already removed by the
    warp recorder (finding 77, `k_td_record_warp`, 56.6 µs fixed + 55 ns per
    candidate against a flat 653 µs).
19. **The August -> September environmental slowdown — PARKED.** A rebuild of
    the exact August commit reproduced `unaccounted` 0.50 -> 7.86 ms/q at c183
    I15e on 2026-09-02 with no source change (finding 88); the CUDA toolkit
    line and the Windows driver were ruled out, leaving the `apt upgrade` of
    2026-08-27 as a coincidence in time. Then: `unaccounted` is host time with
    the GPU idle, not GPU inefficiency (finding 90, `--qspan`), and its size
    swung 6.67 -> 2.83 ms/q overnight on one commit, so it is day-to-day
    variance to characterise (CPU frequency, WSL scheduling, Windows host
    activity), not a commit to bisect. Finding 89 removed two serialisation
    points it exposed (−1.36%). The idle work since (rows 19, 24-26, 33) took
    c135's card idle to 3.6-5.0%. **GPU contention is not this item**: a sieve
    sharing the card runs slowly by design, and `acc/wall` (flat under GPU
    contention, falling under CPU contention) tells the two apart. Do not
    quote `wall - span`'s magnitude: finding 91 could not reconcile it with the
    region timers.
