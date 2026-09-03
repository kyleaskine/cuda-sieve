# STATUS — what exists now

`RESULTS.md` and `../prototype.md` are lab notebooks: they record findings in
the order they were discovered, including the ones later refuted, because the
refutations are the most useful part. That makes them bad at answering "what
does this thing do today". This file answers only that, and holds nothing that
is not current. **Last updated 2026-09-01.**

## Architecture

One process, both sides, one special-q at a time:

```
per special-q:
  host    reduce the q-lattice, build the three modulus tiers
  device  k_transform    plattice transform of the factor base
          k_fill_atomic  bucket-sieve into regions
          k_apply        norm init + bucket add + threshold scan  ->  survivor bitmap
  (repeat for the other side)
  device  k_intersect_compact   both-sides bitmap AND, gcd filter, rank scan
          trial division, classification, resieve
                               (the candidate RECORDING pass is
                               k_td_record_warp, one warp per candidate;
                               the two dense passes are k_td, one thread
                               per survivor -- see the slab section)
          cofactorisation (Pollard rho and ECM)
  host    emit relations
```

Two sides run **sequentially through one shared bucket allocation**. There is
no stream concurrency and no second workspace. Every timing in `RESULTS.md` is
one-q-at-a-time.

**The algebraic factor base is generated in-process by default, added
2026-08-24.** With `--fb1` omitted the pipeline calls the same GPU root finder
that backs the standalone `fbgen_gpu`, so a run no longer needs a
multi-gigabyte roots file staged first — which is what lets a BOINC work unit
carry only the job. `FBGEN_GPU.md` is the reference and `fbgpucheck.sh` gates it
across degrees; `--fb1`/`--cadofb` still take a file when one is wanted.

`--relations NAME` stages to `NAME.part` and renames to `NAME` only when the
band completes. The `.part` is the **durable artifact**, not scratch: after
every cofactor flush it is fsynced and `NAME.part.ckpt` records the next
`(q, rho)`, the byte offset, the relation count, the derived scale/allowance
and a job fingerprint. Rerunning the same command resumes there; `--restart`
discards; SIGINT/SIGTERM and `--stop-file` stop cleanly at the next special-q;
`NAME.lock` refuses a second writer. Verified on a GPU 2026-08-16: a stopped
and resumed band reproduces the uninterrupted one byte for byte, `kill -9` and
a torn final line included. See item 12a.

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
application is concerned. There are no checkpoint files in the BOINC sense
either: a suspended process resumes,
but a process that exits and restarts begins the current band again (12a's
sidecar is the repo's own mechanism, not BOINC's).

**Native Windows is now a supported build target.** `platform.c` carries the
filesystem/process/stop-hook differences, `build_windows.bat` builds a static-CRT
`bench.exe`, and the Windows binary includes the same in-process GPU algebraic
factor-base generator used on Linux. `build_windows.bat fbgen_gpu` additionally
builds the standalone reusable-roots-file generator. Cross-platform relation
bytes are gated separately by `wintest.bat`; Windows process termination still
has the documented limitation that `TerminateProcess` cannot run a checkpoint
handler, so use `--stop-file` for a clean stop.

## Current size limits and j-slabbing

The cofactor-width work landed on 2026-08-18 and the production pipeline now
also supports rectangles larger than `2^31` positions by **j-slabbing** them.
There is still no unsafe override of the local arithmetic bounds.

| quantity | current limit | immediate reason |
|---|---:|---|
| local sieve slab | `2^31` positions | bucket/bitmap/rank positions remain `uint32_t` |
| full pipeline rectangle | no total-area cap under `--pipeline`; `logI` in [2,20] | host scheduler splits `J` into safe slabs. Outside `--pipeline` the old `I*J <= 2^31` refusal still stands (`bench_main.cu`, the `!cfg.pipeline` check) |
| `lpb` | 64 | a resulting prime is stored in one `uint64_t` (was 32 until 2026-08-17) |
| `mfb` | 128 | the cofactor queue narrows residuals to `mz<4>`; 96 in a `CF_LMAX=3` build |
| large primes per side | 3 | `ceil(mfb/lpb) <= 3` is checked before the run |
| exact norm | 384 bits (`BN_LIMBS` 12) | build-time knob, even limbs 4..16; a lattice needing more is SKIPPED with a warning, never wrapped (was 256 until 2026-08-27) |

Automatic planning uses `2^29` positions as a performance target once the full
sieve reaches `2^30` positions; below that trigger it does not split for
performance alone. With default `J=2^(logI-1)` and `bkthresh=I`, I15 uses 1
slab, I16 4, I17 16, I18 64, I19 256, and I20 1024. The target is empirical
and is a performance/memory default, not a correctness bound or a claim of a
universal speed optimum. RTX 3090 and RTX 5070 both minimized complete time at
`2^29` per slab. On an L40, `2^30` was instead 4.6% faster than `2^29`
(531.16 vs 555.70 ms/q), but `2^29` still beat the former `2^31` behavior by
1.5% (564.13 ms/q) while reducing steady VRAM from 7.76 GB to 3.20 GB.
`--slab-j N` can override the default upward or downward for
regression/memory/performance tuning, while the `2^31` position bound and the
actual largest direct-tested prime remain mandatory safety constraints. Raising
`bkthresh` can therefore still make the planner choose smaller slabs.

> **What `2^29` actually is — MEASURED 2026-08-26, finding 79.** It is **not a
> slab-size constant.** `--region` x `--slab-j` swept jointly shows `fill`
> minimised at a fixed bucket-region COUNT in all three rows tested, so the
> optimal AREA halves with the region: `2^29` at `--region 14`, `2^28` at 13,
> `2^27` at 12. **Refined the same day by finding 80:** `fill = L(nregion) +
> 1.178 x nslab` collapses both sweeps to 2.3-4.2%, `L` is minimised at
> **nregion 16,384**, and the 32,768 that finding 79 reported is the *reachable*
> optimum at region 14 — getting to 16,384 there needs twice the slabs, and the
> re-stream cancels the gain. `2^29` is what 32,768 regions means at the default
> `log_region 14`, and nothing more. Region 14 remains the joint optimum
> because `k_apply` launches one block per region and costs +52% at region 13,
> +157% at region 12 — so the defaults are right, and that is exactly why the
> coupling was never noticed. Consequence: **the card-dependent quantity to
> autotune is `nregion`, not slab size**, and the L40's preference for `2^30`
> is a preference for 65,536 regions.
>
> **Cause SETTLED 2026-08-26 by `ncu` (finding 81): read-modify-write on
> partially-filled bucket lines.** Profiling `k_fill_atomic` with everything but
> `nregion` held constant, **L2 read sectors are flat within 3% while DRAM read
> sectors rise 135%** (18.5M -> 43.5M) — the memory system fetching lines the
> kernel never requested, which in a write-only stream is RMW and nothing else.
> Records are 4 B and a sector is 32 B, so eight consecutive appends fill a
> sector; past the knee the frontier is evicted before they arrive. DRAM writes
> go from **1.15x to 1.96x** the theoretical floor, and total DRAM traffic tracks
> finding 80's `L` curve (1.00/1.17/1.49/1.94 against 1.00/1.07/1.44/2.09) at a
> flat effective bandwidth, so fill is DRAM-traffic-bound and `L(nregion)` is a
> traffic curve. Earlier candidates are both dead: cursor atomic contention (a
> 16x thread sweep does not move the argmin) and L2 *capacity* (the frontier is
> 2 MB against 48 MB). Being associativity/sector-bound rather than
> capacity-bound is why a 6 MB 3090 and a 48 MB 5070 agree — **and it means the
> L40's preference for 65,536 regions now needs its own explanation, since
> capacity was never the reason. That is the only part still open.**
>
> **No action at the operating point:** the default already runs at 1.28x the
> write floor, and the best reachable point is worth ~4 ms of fill against
> +53 ms of apply.

**Open investigation -- slab target on large-L2 GPUs. NARROWED 2026-08-24
(finding 74).** The 5070 now has the complete matched sweep on `c194` I16/J32768,
including the two points that were previously missing:

| local area | slabs | bucket array | complete ms/q | vs best |
|---|---:|---:|---:|---:|
| `2^31` | 1 | 5.20 GB | 521.16 | +18.1% |
| `2^30` | 2 | 2.60 GB | 465.92 | +5.6% |
| **`2^29`** | **4** | **1.30 GB** | **441.30** | **best** |
| `2^28` | 8 | 0.65 GB | 454.44 | +3.0% |

So `2^29` is this job's optimum outright, not a memory compromise -- it is both
the fastest point and a quarter of the `2^31` footprint.

> **The ORDERING above is current; the ABSOLUTE numbers are not.** A paired
> re-sweep on 2026-08-25 (finding 75), after `k_apply` got 12% faster, measures
> the same points at 486.55 / 436.48 / **411.32** / 424.00 ms/q for that same
> reference binary and 467.77 / 414.84 / **392.28** / 404.39 after the change
> -- about 7% below this table, with the gap concentrated in cofactorisation
> rather than in `fill` or `apply`. The two sweeps ran under different machine
> conditions, so quote ms/q from finding 75 and treat this table as the shape.
> `2^29` remains the optimum on both binaries, and the re-sweep adds `2^27`
> (+15.8%), which brackets the minimum from below for the first time.

**Superseded 2026-08-26 by a full-range sweep on the current binary (finding
78).** Same job, total area held at `2^31`, `--slab-j` swept so only the slab
COUNT moves, three repeats, idle card. **Quote these numbers.**

| slabs | local area | fill | apply | dense TD | **complete** |
|---:|---:|---:|---:|---:|---:|
| 1 | `2^31` | 184.17 | 131.19 | 10.91 | 439.63 |
| 2 | `2^30` | 130.02 | 132.87 | 9.96 | 386.25 |
| **4** | **`2^29`** | **99.85** | 131.63 | 10.39 | **355.65** |
| 8 | `2^28` | 100.17 | 131.99 | 13.34 | 361.11 |
| 16 | `2^27` | 115.97 | 132.29 | 22.03 | 396.27 |
| 32 | `2^26` | 128.07 | 135.54 | 41.83 | 460.30 |
| 64 | `2^25` | 166.15 | 137.22 | 82.05 | 611.05 |

`2^29` is now bracketed four doublings below instead of one, and `apply` is flat
across a 64x range in slab count, which is the control. Going one step past the
optimum is nearly free (+1.5% at 8 slabs); the penalty then compounds, and both
halves of it -- `fill` and dense TD -- are quantified in item 18.

**Settled 2026-08-25 by a second sweep at double the area.** An earlier draft
claimed the two cards genuinely disagree; that was retracted as confounded
(finding 72's L40 run is area `2^32`, this one area `2^31`, so "`2^30` per slab"
meant 4 slabs there and 2 here), and by slab COUNT both cards appeared to
optimise at four. The discriminating run — same 5070, same job, `J 65536`,
area `2^32` — was then done:

| local area | slabs | complete ms/q |
|---|---:|---:|
| `2^31` | 2 | 892.78 |
| `2^30` | 4 | 795.78 |
| **`2^29`** | **8** | **766.04** |
| `2^28` | 16 | 801.33 |

**The optimum stayed at `2^29` positions while the count doubled 4 -> 8, so slab
SIZE is the controlling variable and the count hypothesis is dead.** Fill cost
normalised per unit area traces the same curve at both total areas (27.92 vs
26.66 ms per `2^29` at the optimum), which is the locality mechanism behaving as
it should. The `2^29` default is correctly parameterised.

That also restores finding 72's conclusion on sound evidence: at the matched
area `2^32` the L40 prefers `2^30` and this 5070 prefers `2^29`, so the cards do
differ and neither optimum generalises. **Remaining gap:** finding 72 does not
name the L40's job and this sweep is c194, so card-vs-card still needs a matched
JOB, not just a matched area.

What finding 74 adds beyond another data point is the mechanism. The curve is
two terms crossing: **fill** improves 41% from `2^31` to `2^29` and then
saturates (190.65 -> 111.68 -> 112.22 ms; finding 77 extends the sweep and
finds it *reverses* below that, not merely saturates), while the **`record
candidate factorisations` pass grows linearly at ~1.33 ms per slab** (1.39 ms
at 1 slab, 10.64 at 8) and is what makes `2^28` lose. Apply is flat across the
whole range. A card whose fill term keeps improving past `2^29` crosses later,
which is exactly the L40's behaviour. **The recording half of that crossing was
removed on 2026-08-26 (finding 77) and the dense trial-division pass took its
place; see below.**

The per-slab tax was then run down to its cause (finding 74's mechanism note):
`k_td` gives one thread per candidate and each thread marches the whole
small-prime list, so a recording launch costs ~0.65 ms flat -- 5,342 candidates
or 577, it makes no difference. **Sizing the grid to the candidate count does
not help and was measured not to.** The effective fix is to split that march
across a warp per candidate; batching the pass across slabs is the weaker
variant that only removes the surplus launches.

**BUILT 2026-08-26 -- finding 77.** `k_td_record_warp` gives one warp per
candidate and takes the hit mask from `__ballot_sync`, whose ascending lane
order is ascending entry order, so factors are divided out in exactly the order
`k_td` produces and **relations are byte-identical at every slab count on
c194, c147 and AS276**. Batching across slabs was not built and is not needed.

The pass is now `O(nacc)` rather than `O(nsm)`: measured across the sweep, a
launch costs **56.6 us fixed + 55 ns per candidate**, against the old flat
653 us. The **per-slab fixed tax is 1,306 -> 113 us, 11.5x**, and the operating
point gains **wall -1.09%** (349.69 -> 345.87 ms/q, arms non-overlapping) with
`fill` unmoved as the control. Finding 74's ~32x projection applied to the
march alone; what remains per launch is the serial norm-and-division work,
which is per candidate and was always real. `--td-record-scalar` restores the
old path for A/B.

`2^29` remains the optimum, as projected. What the change buys is the shape
below it: `2^28` now costs **+1.7% instead of +3.4%**, and `2^27` +12.1%
instead of +15.9% -- the memory decoupling that makes small slabs affordable
for an A=32 job on a 12 GB card, which was always the reason to do this rather
than throughput.

**The tax then changed hands.** Below `2^29` the two terms that now grow are
the DENSE `norms + trial division` pass (10.34 / 13.18 / 21.83 ms at 4 / 8 / 16
slabs) and `fill`, which does not merely saturate but **reverses** -- 98.90 /
99.59 / 115.20 over the same range. The dense pass has the same disease, and
the arithmetic names its fix: `k_td` runs the `nsm` march inside its grid-stride
loop, so a launch costs `ceil(n/(blocks*threads))` marches. From 1 to 4 slabs
`iters` falls 4 -> 2 -> 1 in lockstep with the split, the total march count
stays at 8, and the cost is flat. At 4 slabs `iters` bottoms out at 1, and every
halving after that doubles the marches -- 16 then 32 -- which is the column
exactly. Its fix is NOT this one applied again: warp-per-item would lose
wherever survivors exceed the thread count, which is the common case at large
slabs. It wants lanes-per-item chosen from `n` against the grid -- and NOT a
bigger grid, which was tested on 2026-08-26 and is worth 0.0-2.5% at every
slabbed geometry (item 18), nor on the unslabbed one either (finding 78b).
`fill`'s reversal was **measured on 2026-08-26 (finding 78)** and is not "walk
re-entry": `k_fill_atomic` streams 42 B per factor-base entry per slab --
`plat` 24 + `slice` 2 + `walk_cur` 8 + `walk_next` 8 -- x 22.1M entries =
**929 MB per slab**, independent of slab size. It is bandwidth, it is close to
irreducible, and the slope is **1.18 ms/slab**, not the 1.98 the three-point
window implied. Both terms are item 18's, but only the TD one is addressable.

Collecting matched sweeps on more large-L2 Ada/Hopper parts remains worthwhile.
A cache-aware automatic target is now less blocked than it was -- the recording
tax it had to balance against is 11.5x smaller -- but the dense-TD term has
replaced it in that role, so the same argument for waiting still applies, with a
different term named.

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
control, not an answer, because every skip is a lost special-q.

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
  see "Four-limb cofactors" below.

The useful dispatch is fixed-width kernels selected on the host, independently
for each side:

| side's `mfb` | cofactor type | status |
|---:|---|---|
| `<= 64` | `mz<2>` | **not built, and deliberately so** — see below |
| `<= 96` | `mz<3>` | built |
| `<= 128` | `mz<4>` | built 2026-08-18 |

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
described next.

This makes `lpb=33` with `mfb=64/95` a materially smaller extension than
`lpb=35` with `mfb=64/101`: the former needed 64-bit factor outputs but retains
two- and three-limb cofactor kernels; the latter needs a four-limb kernel on
the 101-bit side. Both ratios still ask for at most three large primes, so
`CF_MAXFAC` is not the blocker.

The output-width change itself should be cheap. The TD factor lists remain
32-bit because factor-base primes remain 32-bit; only the much smaller residual
and split-prime arrays need widening. The expensive part is any extra limb in
the repeatedly executed modular arithmetic.

There is one direct measurement of the per-side dispatch opportunity. Widening
the formerly two-limb rational queue to three limbs changed that queue from
2.15 to 3.94 ms/q (**+83%**) while changing the then-current whole pipeline
from 172.65 to 173.75 ms/q (**+0.64%**). Narrow arithmetic matters greatly to
the stage that uses it, but the easy side was only about 1% of wall time. That
makes per-side dispatch sensible work while the queue is already being changed,
not a high-value standalone optimisation.

### Four-limb cofactors — BUILT 2026-08-18, NOT YET TIMED

`mfb` up to 128 is supported, and the width is chosen **per side** at run time
as the narrowest instantiation that holds that side's `mfb`. Everything up to
and including a C194's `mfba 95` still gets 3/3 and is unchanged.

**AS276 (`~/code/ggnfs-distributed/AS276.job`, the C208) is the motivating job
and it resolves to 4/3 with no flag.** Its parameters, and what each one does
here:

| | value | consequence |
|---|---|---|
| `lpbr / mfbr` | 33 / 64 | 2 parts, **3 limbs** — 64 bits, but `CF_LMIN` is 3 |
| `lpba / mfba` | 35 / 101 | 3 parts, **4 limbs** — 101 bits does not fit 96 |
| `rlim / alim` | 181.6M / 268.4M | `lim^2` ~ 2^54.7 / 2^56, both well above `2^lpb` |
| derived allowance | 102.63 bits (side 1) | sits just above `mfba 101`, as it should |
| `log2(maxnorm)` | 205.02 at `logI 15, J 16384` | inside `bn_t`'s 256-bit budget, 51 bits spare |

Two things fall out of that table. The **rational** side is the exact case an
`mz<2>` tier would serve — `mfbr 64` is 64 bits carried in 96 — and it is still
not worth building, for the reason given above: that queue is ~1% of a
special-q. And `lpb >= 33` is **necessary but not sufficient** for a fourth
limb; AS276's rational side has `lpbr 33` and still runs 3 limbs, because
`mfbr` is what decides.

**AS276 was sieved end to end on 2026-08-18** at `logI 15, J 16384`, factor
base from `fbgen --maxbits 15` (44 s, 230 MB). Three special-q, `--cofactor`:

- width line reports `side 0 3 limbs (96 bits), side 1 4 limbs (128 bits)` —
  derived, no flag;
- 134 relations, 44.67 rel/q, 39.00 of them from cofactorisation;
- **134 of 134 rebuild both norms exactly** through `--check-relations`;
- 130 emitted factors exceed `2^32`, the largest `2^34.96` — hard against the
  `lpba 35` ceiling.

That is the end-to-end proof the golden test cannot give: a genuine 4-limb
population, split by the 4-limb kernel, emitted through the 64-bit output path,
and verified against the norms. **No timing was taken** — the ECM job had the
GPU, and a 3-q band amortises the final queue flush over nothing.

**Cross-validated 2026-08-19 against the 1.5B-relation GGNFS corpus for this
job** (`~/code/ggnfs-distributed/AS276/`, already filtered): over the identical
30 special-q and an identical sub-rectangle, **recall 3,044 / 3,045 = 99.97%**,
with 64 of our 4,089 relations existing nowhere in their corpus and all 64
reconstructing exactly. Full method and the re-find analysis in RESULTS finding
69. This is the strongest correctness evidence the 4-limb path has: an
independent siever, an independently filtered corpus, and set membership rather
than self-comparison.

**It also surfaced the one thing the width change actually broke, which is not
the width.** At `--cof-rounds 2 --cof-budget 65536` (the default until
2026-08-19) the same run
lost **389 of GGNFS's 2,846 relations (13.7%)**; `6 / 262144` recovered all but
2, for **+19.3% relations**. Rho's iteration count scales as the square root of
the factor sought, so a 35-bit large prime costs 5.7x a 30-bit one, and the
default was calibrated on the c183's `lpba 32`. An exhausted budget returns
`CF_INCOMPLETE` and the relation is silently dropped — it presents as a yield
hole in the siever, not as a cofactoriser problem. **The schedule around the
splitter needs re-deriving per width class; the splitter itself did not.**

Note `A = 29` there, not the `A = 32` NFS@Home's own geometry for this job
would want; the area limit is a separate, still-open blocker and does not stop
the job being sieved at a smaller rectangle.

What moved:

- `CF_LMAX` / `CF_LMIN` / `cf_limbs_for_mfb` in `bench.h` — the build's width
  range and the mfb → limbs rule. `CF_LMAX` defaults to 4;
  `make CF_LMAX=3` compiles the 3-limb splitter alone.
- `cofq_t.d_c0/d_c1` are **raw limb arrays plus `L0`/`L1`**, not `mz<3> *`. The
  stride is a run-time choice and `mz<3>`/`mz<4>` are distinct types, so the
  array cannot carry one of them in its type. `cf_run_rounds_dyn` is the only
  place that casts back.
- `k_cof_enqueue` is templated on `<L0, L1>` (four cheap instantiations) and
  the standalone `--cofac` batch parses at `CF_LMAX` and narrows per side.
- `--cof-limbs N` / `--cof-limbs0 N` force a side **wider** than its `mfb`
  needs. Narrower is refused, in `resolve_and_check_cofactor_config`, which is the one
  place that has seen both the `.job` file and the command line.

The width invariant is asserted rather than trusted. `cof_classify` rejects a
residual above `mfb` and `resolve_and_check_cofactor_config` refuses an `mfb` above
`32*L`, so nothing can reach the queue too wide for its side — and if it does,
`k_cof_enqueue` marks it `CF_OVERFLOW`, counts it, and `cofq_flush` stops the
run. The failure mode being closed is a *silently truncated* cofactor, which
does not crash: it emits a relation that reconstructs to the wrong norm. Both
previous versions of this array (2 limbs, then 3) had exactly that bug.

**The ceiling that binds first is not the width.** `CF_MAXFAC` caps a split at
3 large primes, so `mfb <= 3*lpb` regardless, and at `lpb 32` that is 96 bits —
precisely what 3 limbs already held. A side needs 4 limbs only once its `lpb`
reaches 33. Going past 128 bits means 4LP, which is `CF_MAXFAC` and
`mz_split`'s `sp + 2 > CF_MAXFAC` stack guard, not another limb.

#### What the correctness gate can and cannot prove without a C208

`cofcheck.sh` gained a width block. The load-bearing case is **byte-identical
output between a 3-limb and a 4-limb run of the same job**. That is a real
proof, not a smoke test: rho and ECM are Montgomery-domain algorithms whose
iteration is `y <- y^2 + c` in the *true* domain regardless of `R = 2^(32L)`
(`c = c0*R`, `y0 = 2*R`), and `gcd(qR^k, n) = gcd(q, n)` because `n` is odd. So
widening a side must change the cost and nothing else. The 2 → 3 limb widening
of the rational side produced the same md5, which is the precedent.

Cases added: the build reports its own width range; `mfb` above `32*CF_LMAX` is
refused; rho at 4/4, rho at 4/3, and ECM-with-stage-2 at 4/3 are each
byte-identical to the 3/3 run; `lpb 33 / mfb 99` **derives** 4/3 with nobody
choosing it, emits relations, and every one reconstructs; and that same shape
with `--cof-limbs 3` is refused.

**All 45 cases pass on 2026-08-18**, `sm_120`, including the four new ones. The
`lpb 33 / mfb 99` case derives 4/3 with nobody choosing it and emits 64
relations that all reconstruct, so the 4-limb splitter is not merely compiled —
it has produced verified relations.

What no gate here can reach is a genuine 4-limb *population*. `lpb 33 / mfb 99`
on the c183 admits a few candidates above 96 bits; a C208 is made of them —
which is what AS276 and its GGNFS corpus were used for (finding 69).

#### The three deployment options — RESOLVED 2026-08-19

All three questions this section used to pose are answered; the numbers are in
finding 70 and the decisions are made.

- **Dynamic per-side selector — CHOSEN, and it is what ships.** Forcing the
  wide shape on a job that does not need it costs **+8.8% of wall on the c183
  and +8.2% on the C194** (a widened queue is ×1.72), which is too much to
  give away for the simplicity of always running wide. The selector costs
  nothing at run time — one host-side switch per flush.
- **"Always 4/3" — rejected**, on those same numbers.
- **Separate executables — rejected as a performance measure**, kept as a
  packaging option. The 3-limb kernels are register-identical in both builds
  (78 / 86 / 122, verified against a `CF_LMAX=3` compile), so a wide build does
  not slow the narrow path. What `make CF_LMAX=3` does buy is **binary size**:
  `.nv_fatbin` 8.99 MB → 4.02 MB, and about half the `ptxas` time. That is an
  argument about BOINC distribution, not about sieving.

#### Register cost of the fourth limb — measured 2026-08-18, no GPU needed

The register question was the one genuinely new risk: ECM stage 2 holds
`mpt<L> baby[CF_ECM_NBABY]` plus four more points, which at `L = 4` is 64
registers of live state before any working value. `ptxas -v` answers it without
touching the card. `sm_120`, `-O3`, all six `k_cofac` instantiations:

| `k_cofac<L, method, stage2>` | registers | stack frame | spills | reg-limited warps/SM |
|---|---:|---:|---:|---:|
| `<3, rho, ->` | 78 | 96 B | **0** | 25 |
| `<4, rho, ->` | 94 | 112 B | **0** | 21 |
| `<3, ECM, no s2>` | 86 | 96 B | **0** | 25 |
| `<4, ECM, no s2>` | 112 | 112 B | **0** | 18 |
| `<3, ECM, s2>` | 122 | 368 B | **0** | 16 |
| `<4, ECM, s2>` | 154 | 448 B | **0** | 12 |

(warps/SM from 65,536 registers per SM and 8-register granularity; it is an
upper bound from registers alone, not a measured occupancy.)

**Nothing spills at any width**, which was the failure mode that would have made
the 4-limb ECM path far worse than the CIOS ratio predicts. It does not happen.

What the table does show is that **the cost compounds on the ECM stage-2 path
and only there.** Rho loses 25 → 21 warps (−16%) on top of ~1.8x arithmetic;
ECM with stage 2 loses 16 → 12 (−25%) on top of the same arithmetic.

**That worry did not materialise.** Measured a day later, ECM at 4 limbs is
2.7x *cheaper* than rho on AS276 at matched yield and is now the default on a
3LP side (finding 70) — the occupancy penalty is real but far smaller than
rho's `sqrt(p)` growth over the same width step. `CF_ECM_NBABY` is still the
knob to reach for if 4-limb ECM ever does look disproportionate; it has not
needed touching.

The 3-limb instantiations are **identical in both builds** — 78 / 86 / 122
registers, measured on a `CF_LMAX=3` compile of the same source. They are
separate template instantiations and ptxas allocates registers per kernel, so a
wide build does not make the narrow kernel slower. That closes "separate
executables" as a *performance* option; it survives only as the binary-size
argument above.

### Cofactor width and method — MEASURED 2026-08-19, GPU idle

Both questions the width work opened are now answered. The box's ECM job was
suspended (`kill -STOP`) so these are contention-free.

**1. What the fourth limb costs on work that does not need it.** c183, 200 q,
and C194, 100 q, forced wide with `--cof-limbs`. **All four width combinations
emit byte-identical relation files** (c183 md5 `17d8f1d8…`, 9,363 relations),
so this is a pure width price on unchanged work:

| job | side0/side1 | rational q | algebraic q | stage ms/q | wall ms/q |
|---|---|---:|---:|---:|---:|
| c183 | 3/3 | 2.86 | 13.90 | 16.88 | 112.17 |
| c183 | 4/3 | 4.93 | 14.18 | 19.22 | 116.37 |
| c183 | **3/4** (the C208 shape) | 2.85 | 23.46 | 26.38 | **121.99** |
| c183 | 4/4 | 4.89 | 23.74 | 28.72 | 124.38 |
| C194 | 3/3 | 3.32 | 12.10 | 15.48 | 116.30 |
| C194 | **3/4** | 3.26 | 20.87 | 24.18 | **125.82** |

A side's queue costs **x1.72** at four limbs (rational 2.86 -> 4.93; algebraic
13.90 -> 23.46 and 12.10 -> 20.87), against the 1.8-2x the CIOS ratio
predicted. Wall cost of running the C208 shape on a job that does not need it:
**+8.8% (c183), +8.2% (C194)**.

**Verdict: keep the dynamic selector.** 8-9% of wall is too much to give away
for the simplicity of "always 4/3", and the selector costs nothing at run time
(a host-side switch per flush). Separate executables remain unjustified —
the 3-limb kernels are register-identical in both builds. **This is the
decision the width work was blocking, and it is now made.**

**2. rho vs ECM. SHIPPED as the default 2026-08-19** — chosen per side from
`ceil(mfb/lpb)`, rho at 2LP and ECM at 3LP, with ECM's `B1` derived from `lpb`
and the requeue round default raised 2 -> 4 (ECM escalates in curves per
round). Zero-flag effect: c183 17.21 -> 14.30 ms/q *and* 9,363 -> 9,394
relations; C194 15.48 -> 13.95; AS276 3,443 -> 4,089 relations. `--cof-rho` /
`--cof-ecm` force one method on both sides. Two gate cases pin the automatic
choice and the overrides.

The measurement behind it: **ECM wins 2-4x at 3LP and loses narrowly at 2LP** — the
discriminant is the large-prime count, not `lpb`. Measured cheapest-saturating
config for each, both swept from below: c183 `lpb 32/mfb 92` 30.83 -> 15.38
ms/q (2.00x), AS276 `lpb 35/mfb 101` 332.63 -> 123.22 (2.70x). Full sweep over
`lpb 29-36` in RESULTS finding 70; guidance in RUNBOOK "Method: ECM for 3LP,
rho for 2LP". Making it the default is a shipped-behaviour change and has
**not** been made.

*(An earlier version of finding 70 reported 15-18x. That was wrong: rho had
been priced at an over-large budget rather than swept from below — the same
one-sided-tuning error the finding was written to expose. Corrected in place.)*

### Performance accounting

**`k_fill_atomic` is L2-bound, measured 2026-08-25 (finding 76).** ncu on a
5070: L2 throughput 68.7%, DRAM 24.7%, SM throughput 9.4%, IPC 0.22 of 4.0,
101 warp-cycles per issued instruction. Its theoretical occupancy is 50%,
capped by `Block Limit SM` = 24 because 32-thread blocks are one warp each --
not by registers (limit 64) or shared memory (32). **That ceiling is benign.**
Raising block width to reach 100% occupancy is measurably SLOWER at matched
total threads (101.84 ms at 32 threads against 106.70 at 64), because extra
resident warps contend for the resource that is actually scarce. This is the
opposite character to `k_apply`, which is issue-bound with DRAM at 8.5% and
where occupancy was worth -12.6% (finding 75). The two large sieve kernels
need opposite tuning; do not carry a conclusion from one to the other.

Three-to-four-limb Montgomery multiplication expands the CIOS inner product
from `3*3 = 9` to `4*4 = 16` limb multiply-adds. That suggests roughly a
1.8–2x cofactor-stage cost from width alone, subject to register pressure. A
quoted **5x** is not a four-limb estimate: it combines the width cost with a
pessimistic rho workload in which the factor being sought is three bits larger
and therefore takes about `sqrt(8) = 2.8x` as many expected iterations. ECM
also pays for four-limb operations, but does not have rho's square-root scaling;
the rho/ECM crossover for the new cofactor population is unmeasured.

On the current measured profile, cofactorisation is 14.37 of 98.26 ms/q
(14.6%). If the *entire* stage changed and everything else remained fixed:

| cofactor-stage multiplier | total ms/q | whole-pipeline slowdown |
|---:|---:|---:|
| 1.8x | 109.8 | 11.7% |
| 2x | 112.6 | 14.6% |
| 3x | 127.0 | 29.3% |
| 5x | 155.7 | 58.5% |

**Measured against this table 2026-08-19:** the widened queue came in at
**x1.72**, inside the 1.8-2x row, and whole-pipeline cost moved **+8.8%**
(c183) and **+8.2%** (C194) — close to the 11.7% the 1.8x row predicts, and
lower because only ONE side widens on a real job. The 5x row never applied: it
priced rho's `sqrt` growth into the width, and the fix for that was to stop
using rho on the side that suffers it.

That table is an Amdahl illustration, not a projection for AS276. The larger
job has a different area, cofactor distribution, stage mix, and yield. The 5x
case would normally apply only to its hard four-limb subset, not every queued
record.

The final metric is **time to a filterable matrix**, not raw relations/s:

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
required surplus.

### Work status and rough scope

| item | status | rough engineering scope, including GPU validation |
|---|---|---:|
| A=32 by j-slabbing, state carried across slabs | **done 2026-08-24**, merged from `greg/slab`, tuned through 2026-08-26, **validated at NFS@Home's own shape 2026-09-01** (finding 82) | — |
| A=32 as ONE unslabbed rectangle (`2^32` exclusive endpoint) | not started, **and not wanted** — finding 78 makes the slabbed path the faster one | about 1 week if a reason appears |
| 64-bit large-prime outputs and gates | **done 2026-08-17** | — |
| per-side `mz<3>` / `mz<4>` dispatch | **done 2026-08-18**, gated and timed (+8-9% wall when forced wide) | — |
| per-side rho/ECM dispatch, default | **done 2026-08-19**, gated and timed (finding 70) | — |
| cofactor outcome reporting (`split / dead / stuck`) | **done 2026-08-19** | — |
| C208 validated against a 1.5B-relation GGNFS corpus | **done 2026-08-19**, 99.97% recall (finding 69) | — |
| 384-bit exact norms, `BN_LIMBS` a build knob | **done 2026-08-27**, byte-identical gate on AS276, +0.45 ms of a 90 ms q | — |
| `normscan` band width survey, wired into `testsieve.sh` | **done 2026-08-27** | — |
| in-process GPU factor-base generation | **done 2026-08-24** | — |
| filter test on a real corpus | **captured — a C123 was factored end to end from our relations, 2026-08-05** (`work/c123run/msieve.log`) | — |

**The 3–5 engineering-week slabbing estimate that stood here was spent: the
implementation landed 2026-08-24 and was tuned to 2026-08-26.** What is left of
A=32 is the unslabbed whole-area path, which nothing currently needs. The
remaining estimates are source review, not measured schedules.

## Validated

| | what | where |
|---|---|---|
| **Golden suite** | `./cofcheck.sh` — 30 cases on c183 q=120000053, exact relation counts, not ranges | `cofcheck.sh` |
| **Reconstruction gate** | every emitted factor divides, is prime, is within lpb, and both norms rebuild to exactly 1 | `--check-relations` |
| **Transform gates** | root transform against its definition, over a real factor base | `fbtest` |
| **Band runs** | c147: 1340 q → 159,837 relations, both sides PASS | 5070, 5090, 4090, A100 |
| **Real job** | snfs236, ~20M relations before deliberate interruption | 5070 |
| **Full NFS factorisation** | C123 sieved entirely by us, filtered and solved by msieve: `p42 * p82` | `work/c123run` |
| **`A = 32`** | AS276 at `2^17 x 2^15`, 8 slabs: 1,322/1,322 norms rebuilt, rectangle confirmed from the relations (finding 82) | 5070 |

Cards with measured band data: **RTX 5070** (WSL2), **RTX 5090**, **RTX 4090**,
**A100 80GB** (native Linux), and an **RTX 3090** via an external reporter.

`--check-relations` runs on a machine with no GPU.

## Measured, and what it means

- **Our relations filter and factor — a complete NFS run, not a gate.** The
  C123 `223187...173681` was sieved entirely by this siever (29,933 special-q
  over `[400000, 800000]`, CADO for polynomial selection, msieve for
  everything downstream) and msieve took it to `p42 * p82` on 2026-08-05.
  `work/c123run/msieve.log` is the record:

  | | |
  |---|---:|
  | relations in the file | 29,339,493 (+121,515 free) |
  | duplicates removed | 7,073,204 — **24.0%** |
  | unique relations | 22,387,804 |
  | cycles found / needed | 849,784 / 825,717 — **2.9% surplus** |
  | matrix | 823,658 x 823,845, 114.50 weight/col |
  | outcome | `p42` x `p82`, BLanczos 61 s, sqrt 153 s |

  **What this does and does not settle.** It settles that the relations are
  filterable and sufficient — no structural defect survives to the matrix, and
  the 2.9% cycle surplus says the band was sized about right. It does **not**
  give the comparative number a performance claim needs: how many unique
  relations GGNFS would have needed for this same job. The 24.0% duplicate
  share is our own, at a band covering the full factor-base range, and it sits
  at the top of the 15.8-25% range under Known defects — which is what that
  defect predicts for full-band coverage.

- **Fill does not scale with the GPU.** 5090 has 3.5× the SMs of a 5070 and
  returns far less than that on fill, against 3.33× transform and 2.01× apply.

  **Two halves of this bullet were superseded on 2026-08-25/26; read the
  correction before the profile below.**

  *(a) The block default is 4608, not 1152.* The "absolute knee at 1152 blocks
  × 32 threads, flat above it" was measured on one job; finding 76's two-axis
  sweep on the production shape moved the default **1152 -> 4608** (fill −8.6%,
  wall −5.7% on c194). **Wherever this file still says "the shipped 1152
  blocks", read 4608.** No single constant serves every job — c147 unslabbed
  wants more than 4608, c147 slabbed wants 1152 — which is item 2.

  *(b) The mechanism is settled, and it is not work granularity.* Finding 81
  profiled `k_fill_atomic` with `ncu` at the production geometry: **L2 read
  sectors flat within 3% while DRAM read sectors rise 135%** (18.5M -> 43.5M)
  as `nregion` moves past its knee, with DRAM writes going **1.15x -> 1.96x**
  the theoretical floor. That is read-modify-write on partially-filled bucket
  lines — 4 B records into 32 B sectors, the frontier evicted before eight
  consecutive appends can fill one. Fill is **DRAM-traffic-bound at a flat
  effective bandwidth**, so the tuned quantity is `nregion` (findings 79/80),
  and "work granularity" is retired. This does not contradict the 12.8%
  throughput row below: the cost is excess traffic, not saturation.

  The profile that follows is retained for what it establishes about the
  configuration it was taken on — `work/c147/fill_5070.ncu-rep`, a single
  `k_fill_atomic` launch on a local 5070. `ncu` remains blocked on the rented
  boxes (`ERR_NVGPUCTRPERM`). It did kill the L2-capacity story outright:

  | | |
  |---|---:|
  | DRAM throughput | 12.83% |
  | L2 hit rate | 96.45% |
  | L2 (max bandwidth) | 52.11% |
  | L1/TEX hit rate | 18.91% |
  | bytes used per 32-byte load sector | 8.7 |
  | bytes used per 32-byte store sector | 4.6 |
  | stall cycles on L1TEX scoreboard | 157.7 of 191.0 (82.6%) |
  | SM busy | 3.75% |
  | waves per SM | 1.00 |
  | SM active / elapsed cycles | 8.00M / 10.88M |

  Read together: it is not DRAM-limited (12.8%), not L2-capacity-limited
  (96.5% hit — independent confirmation that story was dead), and **not at an
  L2 bandwidth ceiling either** (52%). It is latency-bound on uncoalesced
  scatter — 82.6% of stall cycles waiting on L1TEX, at 4.6 useful bytes per
  32-byte store sector. That last number matters for item 11: the prior-art
  claim that scatter tuning here "dies at the L2 transaction ceiling" does not
  describe this kernel, and ~7× store-sector waste is headroom, not a wall.

  **Waves per SM = 1.00 is the granularity mechanism made visible.** The whole
  grid is resident at once, so a block that draws a heavy chunk has no queued
  block to backfill its slot — and SMs sit idle for 26.5% of elapsed cycles,
  which is that tail. More, smaller blocks fix it twice over: finer chunks
  shorten the tail, and more resident warps hide the 157-cycle L1TEX stall.

  **Caveat: this profile is at 288 blocks × 256 threads** — the pre-finding-52
  geometry, and 256 threads is precisely the held-fixed axis that made
  finding 51 an artifact. So it characterises a configuration the project has
  abandoned twice over, since the block default has since moved to 4608. The
  288-vs-1152 A/B **at 32 threads** was never run and is **no longer the
  decisive experiment**: finding 81 settled the mechanism from traffic
  counters instead.
- **Finding 51's Ada-vs-Blackwell block response was an artifact.** It held
  `--threads` at 256; at 32 the 4090's degradation reverses to improvement and
  all three cards behave alike. See finding 52.
- **rel/J is flat between the 5070 and 5090** — 28.1 vs 29.5 on sampled board
  power, i.e. a tie, with the 5090 1.68× faster. The Ada 4090 is the outlier at
  18.1. Generation separates these cards; width does not. **This tie is an
  artifact of board-only power and probably does not survive the metric of
  record.** First wall reading on the local box (2026-08-07, UPS, monitors
  excluded): **395–400 W whole-box against 165 W GPU board**, so the card is
  ~42% of the box and host overhead is roughly as large as the GPU's own draw.
  Host overhead is approximately *fixed*, so the faster card amortises it over
  more work: `whole-box rel/J = board rel/J × W_gpu / (W_gpu + W_host)`. With
  the host constant now measured at **~105 W idle / ~115 W with the sieve's own
  core** (item 6), that puts the 5090 at ~23.5 against the 5070's ~17.8 — the
  5090 **ahead by ~1.3×** rather than tied. The tie was an artifact of pricing
  only the card. Caveat: this places both cards in *this* box; the 5090 numbers
  came from a rented machine whose own host draw is unknown and probably
  higher, and per the next bullet that can never be measured.
- **The cross-card rel/J table can never be restated on the metric of record.**
  Whole-box power needs a physical meter on the box, and the 5090/4090/A100
  numbers came from rented machines. Those rows are permanently board-only.
  What *is* answerable with a meter is the thing the project was chartered to
  answer — the local 5070 against the local CPU box, item 0 — because both
  sides of that comparison are on this UPS. Treat cross-card rel/J as an
  architecture note, not as a verdict input.
- **THE BOX IS UNDERVOLTED AS OF 2026-08-17, and every timing taken after that
  date is ~6.7% slower than one taken before it.** The card's V/F curve is
  pinned to ~2900 MHz at 950 mV (stock was 2910 MHz at 1080 mV), which trades
  6.7% of throughput for 28% of board power — **+14.6% whole-box rel/J**,
  finding 61. It is *not* a code change and it is *not* reflected in any
  finding numbered below 61: finding 58's 108.27 ms/pair, finding 60's apply
  breakdown and every c147/c183 timing in this file were measured at stock.
  Comparing a new measurement against an old one without accounting for this
  will read as a 6.7% regression that is not there. Board draw is now ~140 W
  and the card sits at 50–56 °C rather than 69 °C.
- **Superseded:** an earlier rel/J table built on *nameplate* TDP concluded the
  design "does not want wide expensive GPUs". It is retracted — see the
  RETRACTED block in `RESULTS.md`. Do not quote the old numbers.
- **Confirmed on a second job, at every rectangle tested** *(findings 58, 65)*.
  On the NFS@Home C194, sieving above `alim` so GGNFS cannot truncate its base
  either, yield is **0.979–0.981 of GGNFS's at all four matched rectangles**
  (`2^14 × 2^13`, `2^15 × 2^13`, `2^15 × 2^14`, `2^16 × 2^14`) — flat in
  aspect ratio, flat in area, and flat in `j` within a rectangle. The earlier
  "−14.4% at a 1:1 rectangle" was a geometry mismatch: `-J 15` covers our
  `2^16 × 2^14`, so it compared our square against GGNFS's wide rectangle. There is
  no measured cost of the norm approximation (item 5).
  Throughput at that equal-work point is **3.31×** the CPU box and **2.74×** on
  energy, against the c183's 3.11× and 2.53×. The same measurement at q=20M
  reads 4.91×, inflated by the factor-base convention alone.
- **Yield matches GGNFS at matched geometry, and the verdict margin is ~2.5×
  rather than 4.3–5.5×.** A same-session control (finding 57) sieved the c183
  over one q interval with `gnfs-lasieve4I15e` and with this pipeline at the
  identical `2^15 × 2^14` rectangle: **46.09 vs 46.47 relations per (q, rho)
  pair**, a 0.8% agreement, with our side carrying 11% more factor base. The
  correction to the margin is a *unit* error, not a performance one — the CPU
  rows are per **prime q** and ours are per **(q, rho) pair**, and this
  polynomial averages 1.528 roots per prime. **Always state which.**

Measured end-to-end on the 5070 (c147 band, 1340 q, same binary, geometry the
only variable): fill −6.1%, apply +0.06%, wall −2.1%, relations byte-identical.
`--fill-threads` is measured to be necessary, not assumed: at the shipped 1152
blocks the 5090 costs **23%** at 256 threads (3.239 ms) against 32 (2.633), so
raising the block default alone would have left that on the table.

Projected but **not** measured end-to-end: the same pipeline A/B on the 5090,
where the standalone gain is largest (16.7%).

## Known defects

- **DOC 2026-09-01 — `normscan.c`'s calibration numbers for the 2,1139+ octic
  do not reproduce.** The comment at `normscan.c:256` justifies the `4 + 4*beta`
  margin with "(98 bits clear, beta 5.0)" — while the same comment block,
  eleven lines earlier, gives that same polynomial **beta = 5.9**. One job
  cannot have both, and the pair dates from when three comments called it a
  septic. Re-run 2026-09-01 on the real coefficients over 60M-460M:
  **49.5-57.5 bits clear at beta 5.96-5.99** across three geometries, matching
  neither parenthetical. `testsieve.sh`'s "250 bits at 15e and 257 at 16e"
  lands near the measured **99th percentile** (253.1 / 261.1), not near the
  projected maximum the tool judges on. **No behavioural defect** — the
  verdicts are the intended ones (pass at 384, refuse at 256 and 320) and the
  `4 + 4*beta` scale still holds against the measured beta. The unreproduced
  pair is now flagged in place in `normscan.c`; re-derive from a fresh survey
  before tightening the rule.

- **NO GATE EXERCISES `PIPE_Q_SKIP` — noted 2026-09-02.** The norm-width skip
  path is now load-bearing: finding 89's async H2D uploads depend on it
  carrying its own `cudaStreamSynchronize`, because it never reaches the slab
  loop's `ev[3]` sync. Nothing in the repo triggers it — c183's norms are
  ~197 bits and AS276's fit inside 384 — so the fix is reasoned and compiled
  but **not run**. Repro for a gate: `make BN_LIMBS=6` (192 bits, allowed by
  the Makefile's `4 6 8 10 12 14 16` filter) and sieve any c183 band; most q
  will skip. Cheap to add and it covers a path that is otherwise only ever
  taken in production by the jobs we cannot sieve.

- **FIXED 2026-09-01 (shipped in `57480cd`) — the slab target was silently
  coupled to `log_region` (finding 79).** `slab_perf_jmax` computed
  `rows = 2^29 / I` with no reference to `cfg->log_region`, but the quantity
  the target tunes is the **bucket region count** (32,768), not the slab area.
  Auto planning picked 8192 rows at `--region` 14, 13 and 12 alike, leaving the
  target wrong by the same factor — **+28.4% fill at region 13, +68.3% at
  region 12**. Never a live regression: `--region` defaults to 14 and
  production never moves it.

  **Both halves are now region-relative**, which is the part worth noting —
  the cap became `rows = (SLAB_PERF_REGIONS << log_region) / I` *and* the split
  trigger became `area < target * 2`, where it had been left as an absolute
  `2^30`. Fixing only the cap would have left an area of `2^29` at
  `--region 12` never splitting, i.e. 131,072 regions in one slab: the exact
  shape finding 79 measured at +68.3% fill. `slabtest` pins regions 12/13/15,
  the trigger in both directions, and out-of-range rejection on the auto and
  forced paths. Behaviour-preserving at the default; see item 2 and the policy
  block in `slab.h`.

- **FIXED 2026-08-18 — non-primitive relations at small q (finding 68).**
  `k_intersect_compact` filtered on `gcd(i,j) == 1` and assumed that made
  `(a,b)` primitive. The lattice map has determinant q, not 1, so positions
  with `q | b` gave `gcd(a,b) = q`; msieve rejects those with `error -6`. They
  survive the sieve because such a point lies on the plattice line of *every*
  root of q, so the sieve subtracts `deg * log(q)` — exactly the `q^deg` in the
  norm — and the position looks as smooth as the much smaller `(a',b')` it is a
  multiple of. Affected positions per q go as `I*J / q`, so it is invisible at
  production q (~2 per q on a C194 at `2^29`) and conspicuous on a small SNFS
  (~335 per q at `2^27` against q = 400009, giving 0.0995% bad relations).
  Fixed by also rejecting `q | b`; the c147 band is byte-identical.
  **`--check-relations` now tests `gcd(a,b) == 1`** — it previously verified
  every factor and both norms, all of which a non-primitive relation passes,
  which is why this shipped.

- **Remaining-norm approximation.** The largest-term approximation differs from
  the true rectangle maximum by ~2 bits and costs ~0.13% of CADO's relations.
  The band-wide fixed scale is a second, unchecked approximation. This is the
  main known yield loss.
- **Duplicate share 15.8–25%** from the full special-q-side factor base, at
  ~1.34× the downstream TD/cofactor work for the same unique yield.
  **These are whole-FB-band numbers.** Measured raw inflation runs 1.0021× on
  the c147's 0.4%-wide band, 1.1877× on the c151's full-FB band and 1.2325× on
  the snfs2 corpus, so the defect scales with band coverage rather than being a
  fixed property of the design.
  **Truncating at `q` is now measured and does not fix it (finding 67):** on a
  real 275M-relation band it removes only 1.78% of the duplicate finds, and on
  a band stopping below `lim` it destroys more unique relations than it saves
  in duplicate work. On a generous-`mfb` job — the c183's `mfba 92`, the C195's
  95 — it does essentially nothing either way. Item 3 is closed; **this defect
  stands, without that remedy.**
  Note also that a corpus can carry duplication that is not ours at all: the
  snfs2 corpus holds a hand-stitched restart's re-sieved q window (finding 67),
  worth 1.4% of its relations. The `k = 1` stratum detects it.
- **The walk gate tops out at logI 10, and every deployed geometry is 14–16.**
  `--verify`'s eight cases (4:1 through 1:2, `make check` runs them via
  `walkcheck`) establish that the Franke-Kleinjung walk is aspect-ratio-correct,
  which is what finding 63 needed — but at I ≤ 1024. The x packing
  `i + I/2 + (j << logI)` is a uint32 and only gets tight in the untested
  logI 11–16 range, so a `pl_make`/`pl_next` bug specific to a large logI would
  still ship. The blocker is `verify_cpu.c:35`: `check_one` sorts its reference
  with an insertion sort, so at logI 15 the reference is ~16k entries and the
  gate would cost ~2.7e8 comparisons per (p, root). Replacing that sort is what
  would let the gate reach the geometry it is about.
- **`k_fill_l1` (twolevel path) has never been swept** at any geometry. It
  takes an explicit `--fill-blocks` but defaults to its own 144 × 512, because
  the 1152 × 32 result was measured on `k_fill_atomic` — a different kernel
  with a different write pattern. (That reference point is now 4608 × 32 on
  `k_fill_atomic`; `k_fill_l1` still ships its own untuned 144 × 512.)
- **Power is board-only.** The metric of record is whole-box
  relations/sec/watt; host draw is unmeasured. The A100 has no sampled power.
- **Host contention costs up to 29% of wall clock, invisibly.** Saturating the
  CPU leaves every `cudaEvent`-timed kernel flat within 1% while wall goes
  24.30 → 31.27 ms/q; half the cores already costs +22.6%. All GPU counters we
  have are blind to it, so a busy box reports perfect kernel times and a bad
  ETA. Stated as throughput that is an **18.4% relation-rate loss at half load
  and 22.3% at full** — not the 22.6/28.7% wall figures, which are
  percent-of-baseline and overstate the bar for co-scheduling. **Any cross-box
  wall-clock or ETA comparison is invalid without knowing host load on both.**
  The pipeline prints `GPU-accounted / wall (excl cofac)` to expose it; the
  first published pair used a formula since corrected, so current values await
  a confirmed-idle box. Finding 53 is canonical for all of this.

## Open experiments, in order

**This list is the task list of record.** Session-local task trackers do not
survive; if work is worth returning to, it belongs here. Two lists drifted
apart once already — items 7–9 below existed only in a chat session and were
absent from every document in the repo.

0. **The verdict band — RUN 2026-08-20, finding 71. 2.99x time and 2.94x
   whole-box relations per joule, every term measured on this box in one
   session.**

   Both sievers over the **same q bands** (same first q, width `10000 * ln q`,
   ~10,000 (q, rho) pairs each), `I15e`, neither running while the other was on
   the box. GPU probes at 50M / 130M / 190M; matched GGNFS 16-worker controls
   at 50M and 190M.

   **Grade on q=190M.** It is above `alim`, so GGNFS cannot truncate — both
   sides run the identical 7,605,406-entry base, and yield then agrees to
   **0.07%**:

   | q=190M | GPU | CPU, 16 workers | advantage |
   |---|---:|---:|---:|
   | wall ms/pair | 100.95 | 301.47 | **2.99x** |
   | unique rel/pair | 41.98 | 41.95 | 1.001 |
   | whole box, at the wall | 240 W | 236 W | |
   | **J per unique relation** | **0.5771** | **1.6958** | **2.94x** |

   **Do not quote the q=50M band's 3.12x.** There GGNFS trims to 39% of the
   base while we sieve all of it, and our 24% yield surplus is duplicate work
   that a truncating siever re-finds later (finding 67). It measures a
   convention, not the hardware.

   **The margin grows with q** (2.57x at 50M, 2.99x at 190M) because GGNFS's FB
   discount shrinks as q rises: it sieves 3.0M entries at 50M and all 7.6M at
   190M, costing it +10.9% wall, while ours falls 4.8%. A real job spends most
   of its range where the discount is small.

   Power measured both sides, monitors (45 W) subtracted: GPU sieving 285 ->
   **240 W**, CPU sieving 280-282 -> **~236 W**. Implied host constant 106.5 W,
   1.4% from item 6's measured ~105 W. **Item 6's 220 W CPU row is superseded**
   — this box idles its GPU at 30 W, not the ~16 W assumed, which is 14 of the
   16 W difference. Dedup measured at 1.0002-1.0009 on both sievers, so the
   RUNBOOK's band-scale 1.19-1.34x does not apply at probe width. All 1,455,951
   GPU relations pass `--check-relations`.

   **Finding 43's `N_eff` 10.24 is vindicated**: it implied 306 ms/pair whole
   box, measured 301.47. **Retire finding 57's 2.53x** (stock card, derived
   270 W) and **item 10's 3.14x** (a C194 figure on a different job).

   **TWO OF THE THREE HOLES CLOSED 2026-09-01 (finding 83).** A full 2x2 --
   both sievers on both rectangles, one session, idle box, metered watts:

   | | time | energy |
   |---|---:|---:|
   | at `2^15 x 2^14` (each siever's own best shape) | **3.09x** | **2.89x** |
   | at `2^16 x 2^14` | **3.03x** | **2.77x** |

   Yield agrees to 0.07% and 0.08%. **The margin is rectangle-invariant**, and
   the premise this item carried is refuted: the bigger rectangle is a rel/J
   loss for BOTH sievers (0.785x for us, 0.819x for them), so we pay slightly
   *more* to grow it. The phrase "our better rel/J shape" below was sloppy --
   `2^16 x 2^14` is the better of the `2^30` shapes, not better than the `2^29`
   we deploy. **Quote ~3x time and ~2.8-2.9x energy**, now confirmed on two
   rectangles rather than one.

   The control also reproduced 2026-08-20 **byte for byte** (same md5), which
   is the first end-to-end proof that 12-limb norms, the 4608 fill default,
   `k_apply`'s launch bounds and the warp recorder are all performance-only on
   a real job.

   **THE LAST HOLE IS CLOSED, 2026-09-02 (finding 87).** The matched CPU
   control at 130M was run, and the GPU arm re-run the same day so both sides
   are current:

   | q = 130M | GPU | CPU, 16 workers | advantage |
   |---|---:|---:|---:|
   | wall ms/pair | 103.13 | 298.98 | **2.90x** |
   | unique rel/pair | 46.095 | 45.927 | 1.0037 |
   | whole box | 252.5 W | 240 W | |
   | **J per unique relation** | **0.5649** | **1.5624** | **2.77x** |

   Yield agrees to 0.37%. **Item 0 now has three probes and three matched
   controls**, plus both rectangles from finding 83. The margin's q-dependence
   is visible on the CPU side directly: 271.95 -> 298.98 -> 301.47 ms/pair
   across 50M / 130M / 190M as GGNFS's truncated base grows toward the full
   one, with yield falling 46.44 -> 45.93 -> 41.95.

   Two cautions from that session, both recorded in finding 87:

   - **The August-to-September wall differences are ENVIRONMENTAL, not our
     code -- finding 88, and it is now item 19.** Rebuilding `4b581b33`, the
     exact commit August was built from, reproduces `unaccounted`
     0.50 -> 7.86 ms/q with no source change. **The CUDA toolkit line is RULED
     OUT** -- linking August's own cudart 13.2.75 gives 7.83, against 7.87 at
     13.2.86 and 7.96 at 13.3.29 -- as is the Windows driver. It is narrowed to
     the broad `apt upgrade` of 2026-08-27. The environmental penalty is
     ~10.6 ms/q and has been masking **9.44 ms/q of real apply+fill work**;
     the picture closes to 0.01 ms: `102.01 + 10.62 - 9.44 = 103.19`.
   - **Cross-session timing comparisons spanning 2026-08-20 -> 2026-09-01 are
     confounded**, including finding 83's attribution of the 190M speedup to
     our work. **The verdict rows themselves are fine** -- each arm was
     measured within one session on one runtime.

   Ambient still matters: the same 950 mV curve drew 133.5 W board in August
   and 147-152.8 W in September heat, so a rel/J figure is comparable across
   sessions only with its board draw quoted alongside.

   *Original statement of the item follows.* The question the
   project was chartered to answer — unique relations/sec/watt on the **c183**
   against the measured CPU box — has never actually been run; every
   end-to-end number so far is a proxy job (c147, c151, snfs236). The CPU side
   is already frozen: 3.12 core-s/q at q=130M and N_eff 10.24 at 16 workers
   give **0.295 s/q for the whole box**, so green is ≤ 0.54 s/q and beating
   the CPU outright on component-proxy perf/watt is **≤ 0.179 s/q end to end**
   (prototype.md, "The GPU's per-special-q budget").

   **Energy bar from measured wall power** *(2026-08-07)*. Those s/q bars were
   derived without a meter. Item 6 now has one, including the host constant
   (~105 W), so the comparison can be made in joules per special-q rather than
   in seconds against an assumed wattage. **Assuming the 16-worker CPU
   baseline is this same box** — it is a 9800X3D, 8 cores / 16 threads, so a
   16-worker sweep fits it, but confirm it:

   Both sides' wattage is now measured (item 6). The GPU side draws ~270 W
   whole-box; the CPU side draws 220 W and its throughput was measured in the
   same session. But **which CPU throughput** turns out to be the whole
   question:

   | CPU comparator | J/q | GPU at 70–90 ms/q × 270 W | **margin** |
   |---|---:|---:|---:|
   | `I14e`, measured | 37.4 | 18.9–24.3 J/q | **1.5–2.0×** |
   | `I15e`, measured | 104.5 | 18.9–24.3 J/q | **4.3–5.5×** |
   | `I15e`, finding 43 | 64.9 | 18.9–24.3 J/q | 2.7–3.4× |

   > **RETRACTED 2026-08-17 — the two columns are in different units.** The
   > CPU J/q above are **per prime q**; the GPU column is **per (q, rho)
   > pair**, which is what `--nq` and the band summary count. GGNFS averages
   > **1.528 roots per prime** on this polynomial, so every margin in the table
   > is inflated by that factor. Finding 57 measured both sides in one session
   > at matched geometry and matched yield (46.47 rel/pair against GGNFS's
   > 46.09, a 0.8% agreement) and the honest figure is:
   >
   > | | per (q, rho) pair | per prime q |
   > |---|---:|---:|
   > | GPU wall / energy | 98.5 ms / 26.6 J | 150.5 ms / 40.6 J |
   > | CPU whole box / energy | 306 ms / 67.4 J | 468 ms / 103.0 J |
   > | **advantage** | **3.11× time, 2.53× energy** | same |
   >
   > **~2.5× on whole-box relations/joule**, which still clears the "beats the
   > CPU outright" bar. Do not quote 4.3–5.5×.
   >
   > **State the unit in every future number.** "Per q" is ambiguous in this
   > project and has now cost two corrections; item 0's own `3.12 core-s/q` is
   > per *pair* while item 6's `474.8 ms/q` is per *prime*, in the same file.

   **Geometry: checked 2026-08-16, and it matches — finding 55.** The concern
   was that the GPU sieved an `I14e` rectangle (`2^14 × 2^13`) while being
   graded against finding 43's `I15e` CPU baseline (`2^15 × 2^14`, four times
   the area), flattering the verdict by up to 4×. It did not: the standalone
   benchmark defaults to `logI 15, J 16384` (`bench_main.cu:426`, unchanged
   since the initial checkin), the finding 43/44 commands pass neither flag,
   and `5.369e8` positions is what this file's own header records. **So
   64.371 ms/q is an `I15e` number, the `I15e` row is its comparator, and the
   margin for it is 4.3–5.5×.**

   What is still unpinned is the *pipeline*: every pipeline run on record uses
   `--logI 14 --J 8192`, so item 0's 70–90 ms/q projection takes standalone
   `I15e` milliseconds and calls them a pipeline number. **A verdict band
   states its geometry in its log header and is graded against the matching CPU
   row** — `I14e` → 37.4 J/q, `I15e` → 104.5 J/q. Mixing them is still worth
   4×; it is now a run-discipline requirement rather than an open question.

   **The mismatch that survives is the factor-base convention, and it is
   q-dependent** (finding 55). Standalone side 1 truncates the base at q, as
   GGNFS does (`bench_main.cu:773-774`); the pipeline does not (`:1360`,
   `fbbound = alim`), because per-q truncation is item 3 and unbuilt. Counted
   on `oracle/c183.fb1`, the pipeline's full base carries **2.54× the bucketed
   entries GGNFS sieves at q ≈ 50M**, 1.11× at 120M, 1.03× at 130M, and 1.00×
   at 190M (above `alim`, where truncation is a no-op). So the 70–90 ms/q
   projection — anchored at q=120M, truncated, standalone — is roughly right at
   the 130M probe and **optimistic at the 50M probe**, the three probes will
   show a q-dependence in ms/q that is not yield drift, and the item-3 A/B has
   no signal at all at 190M.

   Note also that `I14e` and `I15e` are not interchangeable for the CPU: the
   larger area wins 2.2× the relations per special-q (68.9 vs 31.6) while
   costing 2.9× the time, so `I14e` is the better rel/J (0.846 vs 0.659) and
   `I15e` the better relations-per-q-range. A real c183 job picks `I15e` or
   above. Which of those the GPU should be held to is a question about the
   deployment, not about the hardware.

   **The GPU shows the same trade and a worse one** *(finding 57,
   2026-08-17)*. Doubling J at fixed I — `2^15 × 2^14` → `2^15 × 2^15` — costs
   **2.03× the time for 1.47× the relations**, so rel/J falls 27% (2.57 →
   1.87). The CPU's reason for large rectangles is amortising its per-q root
   transform, 12% of its wall and about 1% of ours, so **the GPU should prefer
   smaller areas than the CPU does**. Take the larger J only when the q range,
   not energy, is the binding constraint.

   **But if you do want more area, buy it with I** *(finding 65)*. The two axes
   are not interchangeable: at equal area the wide rectangle beats the square
   by **+11.7% relations at `2^28` and +16.3% at `2^30`, for −0.3% device
   time** — `2^16 × 2^14` returns 10,671 relations against `2^15 × 2^15`'s
   9,176 on the same window. Doubling I costs 1.86 bits of `log2(maxnorm)`
   against doubling J's 3.86, because `i` multiplies the shorter vector of the
   reduced q-lattice and `j` the longer one. Measured rel per device-second
   across the grid: `2^14 × 2^13` 390.6, `2^15 × 2^13` 387.2, `2^14 × 2^14`
   336.0, `2^15 × 2^14` 323.9, `2^16 × 2^14` 260.6, `2^15 × 2^15` 223.4.

   **And it is the cheaper shape too** *(finding 64, row added 2026-08-18)*:
   `2^16 × 2^14` sizes at **4.98 GB** against the square's 5.32 GB, because
   `bkthresh` defaults to `1 << logI` and so a wider I shortens the bucketed
   prime range — 2.60 GB of bucket array against 2.91 GB. Same mechanism as the
   yield win, so there is no trade-off to weigh: at fixed area, wider is better
   on relations, on device time, and on VRAM simultaneously.

   The c183's own standalone
   sieve measures 38.2 + 26.2 ms for the two sides (RESULTS.md "Reproduce",
   `I15e` geometry, side 1 truncated at q, at q=120M), so with TD and the
   heavier mfba-92 3LP cofactor load the pipeline projects to roughly
   **70–90 ms/q — about 2× inside the harder bar**. That is a projection, not a measurement, and nothing measured
   contradicts it — which is exactly why the run should happen now: it is a
   wall-plug meter and a weekend, not a project. Needs: the meter (item 6);
   bands at q ≈ 50M / 130M / 190M for the yield drift; relations
   **deduplicated before quoting** (raw is inflated 1.19–1.34×, see RUNBOOK);
   an idle host per finding 53. Fold item 3's full-vs-truncated A/B into the
   same bands — the FB convention moves both the yield accounting and the
   downstream load, so it should be settled in the graded configuration.
   **But not in these bands as specified.** Three narrow probes at 50M / 130M
   / 190M are the right shape for yield drift and the wrong shape for the FB
   convention: a band covering a fraction of a percent of `lim` has almost no
   duplicates to remove (measured 1.0021× on the c147), and under truncation
   the relations whose re-finding q lies outside the probe are lost outright
   rather than deduplicated. Grade perf/watt on the three probes; settle the
   FB convention on one contiguous band wide enough to contain both q of a
   duplicate pair, or by replaying the attribution offline as item 3 now does.
1. **Concurrent-q throughput. MEASURED 2026-09-01 on the 5070 (finding 84):
   the knee is per-KERNEL, and two concurrent fills run in 85% of serial time.**
   `--fill-streams N` is built in the standalone benchmark; the production
   pipeline is untouched. Arms interleaved, best of three passes, three
   invocations: **concurrent/serial 0.849 +- 0.002**. The gain **saturates at
   two streams** (four give the same 11.53 ms per workspace), and widening ONE
   kernel's grid recovers 6.6% against concurrency's 15.4%. So a single fill
   kernel cannot saturate even the narrowest card in the set, which is the
   mechanism the `ncu` profile predicted (`waves per SM = 1.00`, SMs idle 26.5%
   of elapsed cycles).

   A first version of the experiment ran the arms in fixed order and read
   0.840; the drift correction is worth about one point and the spread fell
   from +-1.5% to +-0.2%.

   **THE 5090 RAN THE SAME DAY AND CHANGES THE RECOMMENDATION.** Rented card,
   same job and geometry:

   | card | best single | best concurrent | saturates | gain |
   |---|---:|---:|---|---:|
   | 5070, 48 SM | 12.73 ms | **11.53** (N=2) | N=2 | 9.4% |
   | 5090, 170 SM | 8.04 ms | **5.83** (N=4) | N=4 | **27.4%** |

   **Only fill fails to scale with the card**: transform 3.77x and apply 3.59x
   across the two cards against an SM ratio of 3.54x, while fill returns 1.62x
   on one kernel and 1.98x under concurrency. So the plateau is one kernel
   failing to feed a wide card, not a device limit -- and the number of streams
   a card wants is a **per-card quantity** (2 here, 4 there, with N=8 matching
   N=4 to 0.1%), which makes it item 2's kind of problem.

   **Worth ~2% of wall on a 5070 -- do not build it for that. Worth 9.7% of
   wall on a 5090, measured in the pipeline, and that is the case for building
   it.** A 2,000-q band on the rented card gives wall 44.19 ms/q against the
   5070's 97.46, with apply scaling 3.48x, transform 3.35x and **fill only
   1.66x** -- so fill's share of wall *grows* with the card, 26.6% -> 35.3%,
   and a 27.4% cut is 4.28 ms of 44.19.

   It is also a conclusion about hardware, not just code: **wide cards are
   underfed, not poorly suited**, so the flat rel/J between a 5070 and a 5090
   (finding 47) was measured against a handicapped configuration and should be
   re-taken once this ships.

   **The effect grows as the geometry shrinks.** At `c147 I14 J8192` -- 8,192
   regions against c183's 32,768 -- the 5090 gives concurrent/serial **0.5975**
   and 39.7% off fill, against 27.4% at the larger geometry. Small jobs and
   heavily slabbed geometries are the best case for this change, not the worst.

   **A second stage has the same shape.** `resieve + scatter` scales **1.51x**
   across the two cards, worse than fill, and is 13.5% of the 5090's wall. It
   is bucket-structured work and nobody has looked at it under this lens. Not
   part of this item; the next place to look.

   **OPEN TODO -- one more rented card, before any production design.**
   Both data points are Blackwell (48 SM -> 2 streams, 170 SM -> 4), so nothing
   says whether an autotuner can PREDICT the stream count from device
   properties or has to measure it. A third architecture settles it.

   *Pick on price, not model.* A **3090** (GA102, 82 SM) is the best value: a
   third architecture, an SM count between the two we have, an existing 3090
   datapoint in the corpus to cross-check, and `GPU_ARCH=native` builds sm_86
   in ~15 s against sm_120's 277 s. An **L40/L40S** (AD102, 142 SM) is the next
   best -- same silicon family as the 4090 and it re-uses finding 72's L40.
   A **4090** adds the historical anomaly (1.80x SLOWER at fill than a 5070
   despite 1.5x the bandwidth) but is not required: that table was taken at 256
   threads before the 4608 default, finding 52 already showed that axis
   manufactures artifacts, and closing it properly needs a fresh 5070 baseline
   too -- which is free, locally.

   ```sh
   make GPU_ARCH=native CF_LMAX=3 -j$(nproc) bench && make fbgen
   ./fbgen --poly input.job --maxbits 15 --threads $(nproc) --out c183.fb1
   for N in 2 4 8; do ./bench --poly input.job --fb1 c183.fb1        --logI 15 --J 16384 --reps 20 --fill-streams $N; done
   ./bench --pipeline --cofactor --poly input.job --fb1 c183.fb1        --logI 15 --J 16384 --qrange 190000000: --nq 2000        --relations g.rels --log g.log --log-every 60
   ```

   Wanted from it: `concurrent/serial` at each N, the N where per-workspace
   time stops falling, and the pipeline `band of` stage breakdown so fill's
   share of wall is known for that card. `CF_LMAX=3` is valid -- c183's
   `mfba 92` is under the 96-bit ceiling and the cofactor width cannot touch
   `k_fill_atomic`. `ncu` is blocked on Vast.ai (three boxes now), so do not
   plan on a profile.

   If Ada does NOT recover under concurrency the way Blackwell did, it has a
   second mechanism and that changes the design before anyone writes it. **When it is built, the two SIDES of one q are the
   cheaper pairing than two q**: they already share the factor bases and run
   sequentially through one bucket allocation today. Note the production gain
   is not the benchmark gain -- the pipeline number needs a pipeline run, and
   real special-q do not march their bucket frontiers in lockstep the way this
   benchmark's identical workspaces do (finding 84's caveat).

   *Original statement of the item follows.* Two independent fill workspaces,
   two q or two sides in separate streams, sweep 1/2/4. This is the decisive test for
   whether wide cards are a poor fit or are simply being fed too little
   independent work. **Design it at the new knee** — the "144 blocks each" this
   item used to specify is the pre-finding-52 geometry and would reproduce the
   flaw described next: the old framing
   ("two 144-block fills vs one 288-block fill") is now measured entirely below
   saturation, where any configuration scales, so it would have credited
   streams for reaching a thread count one kernel already reaches. Compare
   2 streams × 1152 against 1 kernel × 2304 instead. The old note here said the
   two architectures predict differently — Blackwell's flatness fitting "idle
   capacity" and Ada's degradation not. That distinction is gone: at 32 threads
   all three cards flatten, so one prediction covers them all.
2. **Startup fill autotune. LADDER BUILT AND DELETED 2026-09-01; the DEFAULT
   CONFIRMED IN-BAND and the ladder's WRONG-ANSWER failure pinned to its
   repeat-fill regime 2026-09-02 (finding 85).** Finding 85 addressed two of
   the three recorded failures -- wrong regime and (as a rival explanation)
   sample count. **The third is untouched:** the intermittent ~1.9 s stall
   inside the fill event window remains unexplained, and the redesign below
   measures that same window, so it inherits the outlier. The redesign's
   *measurement regime* is validated; the design as a whole is not, and the
   guard constants below need re-deriving before anyone builds it.
   `nregion` remains the untouched axis.

   **STRONGER CASE 2026-08-25 (finding 76).** The
   two-axis sweep that item warned about has now been run on the production
   shape, and it moved the block default 1152 -> 4608 (**fill -8.6%, wall
   -5.7%** on c194). It also showed why one constant cannot serve: c147
   unslabbed prefers MORE blocks than 4608, c147 slabbed prefers 1152, and
   c194 prefers 4608 -- two of those share a factor base and still disagree on
   direction. The new default is right for the I16 production class and costs
   c147-slabbed ~0.8%; the autotuner is what removes that trade entirely. The
   thread axis needs no sweep: 32 is optimal at every block count measured, and
   raising it to buy occupancy makes fill slower (see below).

   **A THIRD AXIS, added 2026-08-26 by finding 79: `nregion`.** `fill`'s
   optimum is a fixed bucket-region count (32,768 measured here), not a slab
   area — so the card-dependent number to tune is `nregion`, and both
   `SLAB_PERF_TARGET_LOG2` and `log_region` are derived views of it. The L40's
   preference for `2^30` slabs is a preference for 65,536 regions. Fixing the
   slab/region coupling logged under Known defects belongs in this item, not on
   its own, because the constant should become an autotuned quantity rather
   than a better-derived hardcode. Note that `nregion` trades fill against
   apply in opposite directions (finding 79's second table), so it is the one
   axis here that cannot be tuned on `fill` alone. **Finding 81 makes this axis
   cheap to tune:** `L(nregion)` is a DRAM-traffic curve, and traffic is
   predictable from `nregion` alone — so a candidate geometry can be priced from
   the model or from counters, without running trial fills for it.

   **BUILT 2026-09-01: the block axis and the region/slab coupling fix.**

   - `slab_perf_jmax` now takes `log_region` and targets **32,768 bucket
     regions** rather than `2^29` positions, deriving rows as
     `(SLAB_PERF_REGIONS << log_region) / I`. Arithmetically identical at the
     default `--region 14`, which is why every pre-existing `slabtest` row is
     unchanged; three new rows pin region 12/13/15, all of which returned the
     region-14 answer before.
   - **The split TRIGGER moved too, and that is the half that changes
     behaviour.** It had been left as an absolute `2^30` while the target went
     region-relative; it is now `area < target * 2`. Fixing only the cap would
     have left an area of `2^29` at `--region 12` never splitting -- 131,072
     regions in one slab, the shape finding 79 measured at +68.3% fill. Four
     `slabtest` assertions cover it in both directions. The `* 2` is deliberate
     hysteresis, not a bug; `slab.h`'s policy block now says so in three
     places, because a reviewer proposed "correcting" it to `area < target`
     and `slabtest` caught that.
   - **Upward consequence, documented 2026-09-02:** holding the region count
     fixed means the slab AREA scales with region size, so `--region 15` now
     auto-plans a `2^30` slab where the old absolute target gave `2^29`, and
     region 16 gives `2^31`. Peak bucket memory follows. Intended policy, but a
     `--region 16` run that fit before may not now.
   - **Out-of-range `log_region` now fails on the FORCED path too**
     (2026-09-02). The only range check lived in `slab_perf_jmax`, which
     `forced_j != 0` never reaches, so `slab_make_plan` accepted any region
     there and silently ignored it. `bench_main.cu` validated independently, so
     nothing shipped wrong; the header's contract is now enforced. That closes
     the Known-defects entry.
   - **`--fill-blocks` autotune -- built, then DEFAULTED OFF the same day when
     it was measured against a band (see below).** Times the ladder
     `{1152, 2304, 4608, 9216, 18432}` on the first special-q's real data,
     three reps, keeping the minimum. `--fill-autotune` forced it on for
     experiments; **an explicit `--fill-blocks` disabled it** -- a knob the
     operator set must not be silently overridden. The choice went to stdout
     and to the run log, and the log distinguished `chosen-by-ladder` from
     `kept-default-or-refused` so a guard firing was visible.

     **All of that is past tense: NONE of it exists in the tree.** The deletion
     took the flags with the harness, so `--fill-autotune`, `chosen-by-ladder`
     and `kept-default-or-refused` appear in this file and nowhere else. Where
     the disposition below says "keep the guards and the flags", read it as
     *keep the design decisions on record* -- which is what this prose is.
     There is no flag to find and nothing to re-enable.
     Repeating the fill is safe on slab 0 including SLABBED, because
     `k_fill_atomic` reads `walk_cur` and writes `walk_next` and the host swaps
     only after the slab, so every trial reads the same input.

   **Two guards, both learned by accident the same day.** The first run of the
   tune fired while a GGNFS sieve client had the GPU at 96% and **picked 1152
   over the 4608 default** -- a choice that would then have stood for the whole
   multi-hour band, on evidence that was pure scheduling noise.

   - **Unstable-device guard:** if any ladder point's slowest rep exceeds its
     fastest by **>10%**, the device is busy, no timing means anything, and the
     tune abandons and keeps the default, printing the observed spread. The
     threshold is calibrated, not guessed: 1.05 was the first value and it sat
     *inside* natural jitter -- an idle 5070 prints spreads of 2.1-6.6% -- so
     one run in three discarded a real result, while the case it must catch ran
     at ~2x inflation. The observed spread is printed on every outcome so the
     next person recalibrates from data.
   - **Margin guard:** the winner must beat the default by >3% to be adopted.
     4608 is a *measured* value (finding 76), not an arbitrary start, and a
     tune that switches on a 1% difference is picking noise.

   This is the same lesson as finding 53 in a new place: **a number measured on
   a contended box is not a number.** A startup autotune is exactly where that
   bites hardest, because one bad instant sets a parameter for hours.

   **VERIFIED 2026-09-01, and guard 1 was exercised for real.** A 1,000-q c183
   I15e band on the new binary emitted 42,184 relations and **every one of them
   appears in finding 83's arm A file** (`comm -23` of the sorted sets: 0
   lines). That one check clears three things at once -- the slab region-count
   fix is behaviour-preserving, the autotune is output-neutral, and the binary
   is sound despite two overlapping `make` runs having touched the tree.

   The run went in deliberately against a GGNFS sieve client holding the GPU at
   97%, and the guard did its job:

   ```
   fill autotune, on this q:  1152:28.73  2304:27.34  4608:27.09
                              9216:23.90  18432:24.49
     -> keeping 4608: reps disagree by >5%, the device is busy
   ```

   Note what an unguarded tuner would have taken from that: 9216 looks like a
   12% win. It is scheduling noise, and the run would have carried it for the
   whole band. **That transcript is from the 1.05 build** -- the message format
   and threshold both changed afterwards -- so it is evidence that the guard
   concept works, not that the shipped 1.10 constant has fired in anger. It has
   not; the contended case is ~2x inflation and clears 1.10 by a wide margin on
   arithmetic, but nobody has re-run it.

   **THE TUNER AS BUILT DOES NOT WORK -- MEASURED 2026-09-01, DO NOT SHIP IT.**
   On an idle 5070 it reproducibly picks 18432 (12.57-12.67 ms, 4/4 runs) over
   the 4608 default, an apparent 8.5% win. Against a real 200-q band, three
   reps each, spread 0.18%:

   | `--fill-blocks` | band fill | vs default |
   |---:|---|---:|
   | 4608 (default) | 26.035 / 26.048 / 26.081 | -- |
   | **9216** | **25.505 / 25.507 / 25.461** | **-2.1%** |
   | 18432 (the tuner's pick) | 26.116 / 26.100 / 26.080 | **+0.2%** |

   **The tuner picks the worst of the three** -- 2.4% behind the right answer
   and 0.2% behind doing nothing.

   **Cause: the ladder measures the wrong regime.** It runs repeated fills of
   ONE lattice back-to-back, which is steady-state and cache-warm. A band gives
   every q a freshly transformed `plat` and fills it ONCE. The proxy does not
   predict the target, and the single-q numbers are internally consistent
   (4/4 runs agree) while being wrong about production -- which is the most
   dangerous shape a measurement can have.

   **CONFIRMED 2026-09-02 by controlled comparison -- finding 85.** The rival
   explanation was sample count: maybe any short measurement is unreliable.
   It is not. On **the same job, geometry and axis the ladder got wrong**
   (c183/I15e), a 10-q band picks **9216** -- the in-band answer -- while the
   ladder picked 18432. Sample count is exonerated and the repeat-fill
   structure is the cause. **A short band is not a bad measurement; a
   repeat-fill proxy is.**

   **The 4608 default is CONFIRMED IN-BAND, 2026-09-02 (finding 85).** The
   check this paragraph asked for was run: `--fill-blocks` swept at `--nq 200`,
   arms interleaved, on finding 76's own c194/I16 configuration.

   | arm | in-band fill, n=3 | `--nq 10`, n=4 |
   |---:|---:|---:|
   | 2304 | 102.343 | 103.233 |
   | **4608** | **99.872** | **100.535** |
   | 9216 | 102.200 | 102.277 |
   | 18432 | 103.492 | 103.653 |

   4608 wins both, arms non-overlapping (its worst rep beats the runner-up's
   best by 1.9 ms), and the whole *ranking* is identical across regimes. So
   finding 76's `--nq 10` sweep was never in the ladder's regime -- `--nq 10`
   is a real band with ten distinct q, each freshly transformed and filled
   once. **Do not move the constant.**

   What remains true is that the optimum is geometry-dependent: c183/I15e
   prefers 9216 by 2.4%, and 4608 sits at the bottom of the four arms there --
   last outright at `--nq 10`, and in-band tied with 18432 for last (26.438
   against 26.443, a 0.02% gap inside a 1.46-3.35% spread). That is an
   argument for tuning per geometry, which is this item, not for a different
   hardcode.

   **A third failure, 2026-09-01, after the warm-up fix.** With the spread now
   printed on every outcome, two runs in three abort with a **130x spread**
   (`reps disagree by 12936.3%`) on a completely idle card, while every printed
   per-rung minimum is a normal 12-15 ms -- so one REP of some rung took ~1.9
   seconds. Not contention, not jitter: a stall inside the event window that
   the earlier build simply could not see. Unverified hypothesis: an
   asynchronous side-0 transform still in flight when the ladder starts, so a
   trial queues behind it.

   **Disposition: delete the ladder harness, keep the guards and the flags.**
   Three distinct failures in one session -- wrong regime, wrong answer against
   a band, intermittent multi-second outlier -- each surfaced by fixing the
   last. That is a design to rebuild, not to patch, and ~95 lines of
   permanently-disabled code inside the per-q loop is a standing cost to every
   future reader of `run_pipeline_impl`. What is worth keeping is the part that
   was validated: the two guard constants and their calibration, the
   distinction between `chosen-by-ladder` and `kept-default-or-refused` in the
   run log, and the rule that an explicit `--fill-blocks` always wins.

   **The redesign, if this is taken up again:** tune IN-BAND. Run the first N q
   of the band at each candidate grid -- 20 q per candidate is ~100 q against a
   50,000-q work unit, i.e. free -- and compare the band-average fill each
   produced. That measures exactly the quantity being optimised, needs no
   proxy, and the guards built today (unstable-device, margin) carry over
   unchanged.

   **The ladder code exists in NO ref.** It was written and deleted between
   commits, so `git grep` finds it on no branch and there is nothing to rebase
   -- verified 2026-09-02 across every ref (`main`, `greg/main`, `greg/slab`,
   `origin/main`, `origin/fix/windows-review`). An earlier sentence here said
   "the code is on the branch and the flags work"; that was wrong and is
   withdrawn. This item's prose is the only surviving record, which is why it
   is written at the length it is.

   **Finding 85 validates the measurement regime and sharpens it, but it also
   breaks two of the constants this design was going to reuse. Read all five
   points before building.**

   - **N=10 is enough TO RANK.** A 10-q band picked the in-band winner on both
     jobs and reproduced the full four-arm ranking on c194.
   - **N=10 is NOT enough to size the effect**, and the design depends on that
     more than on ranking. c183 reports a 4.37% margin at `--nq 10` against
     2.42% in-band; c194 reports 1.70% against 2.28%. Errors in both
     directions, up to ~1.8x.
   - **INTERLEAVE the candidates, and the rep count collapses.** Ranking each
     rep on its own, **15 of 16 interleaved reps** across two jobs and two
     regimes picked the in-band winner outright; the single miss took 2304
     over 9216 by 0.19%, between the two best arms. This held despite
     within-arm spreads up to 4.09%, because whatever the host and clocks do
     during a rep, all candidates see it. A tuner that runs each candidate to
     completion in turn cannot borrow this and needs many more reps.
   - **THE >3% MARGIN GUARD MUST BE RE-DERIVED -- as written it refuses every
     win this tuner exists to capture.** Both real margins finding 85 measured
     are *below* the threshold: 2.42% in-band on c183 and 2.28% on c194. The
     guard was calibrated against the ladder's fictitious 8.5% and 12%
     readings, not against true in-band effect sizes, which live at 2-3%.
     Lowering it is not simply safe either: finding 85 saw the same margin
     double run-to-run under an identical protocol (0.84% -> 1.70% in August
     vs September on c194 at `--nq 10`), so a 2% threshold against a
     10-q estimate is inside the noise it must reject. **This is the open
     design problem, and it is not a constant to guess -- it needs the
     margin's own reproducibility measured on an idle box.**
   - **THE >10% UNSTABLE-DEVICE GUARD GOES INERT at 2 reps.** Slowest-vs-
     fastest over two samples is a single pairwise difference with no power to
     detect anything. Worse, it would not fire even with more: every spread
     finding 85 measured on a *contended* box tops out at 4.09%, comfortably
     under 10% -- and c183 under contention is exactly the case the caveat
     below calls unreadable without interleaving. The threshold was calibrated
     against the ladder's ~2x inflation and does not transfer to in-band
     spreads. Re-derive it too, or replace it with an idle-box precondition.

   **The shape, stated precisely** -- the earlier one-line version was
   ambiguous in a way that matters:

   - The grid is **five rungs** `{1152, 2304, 4608, 9216, 18432}`, not the four
     finding 85 swept. Do not drop 1152: it is c147-slabbed's measured optimum
     (finding 76), so a four-rung grid cannot find the winner on one of the
     three documented geometries.
   - `for rep in 1..2, for candidate in grid, sieve the SAME 10 q`. Every
     candidate must see identical lattices, because finding 85's 15-of-16
     single-rep result is *paired on q* -- give each candidate its own 10 q and
     the pairing that cancels drift is gone, and that result does not transfer.
   - That is 2 x 5 x 10 = **100 q**, matching this item's original estimate;
     the "~80 q" figure that stood here assumed the four-rung grid.
   - **Two consequences of re-sieving the same q that are NOT yet measured.**
     (a) Relations from tune passes must be suppressed or deduplicated, or the
     band emits each of those q 10 times. (b) Re-sieving one lattice
     repeatedly is structurally closer to the ladder than a normal band is:
     each pass does its own transform, so it is not the ladder, but L2 is warm
     from the previous candidate's fill of the *same* lattice. Whether that
     warmth biases the comparison is untested, and it is the first thing to
     check when building this.

   Keep `--no-fill-autotune`-equivalent behaviour until then: **the default
   must stay off.** The "or ship 9216 instead" alternative that stood here is
   withdrawn -- finding 85 shows 9216 is right for c183/I15e and wrong for
   c194/I16 by 2.3%, so swapping the constant just moves the loss to the
   production class.

   **Contention caveat for whoever builds this** (finding 85). Across both
   regimes on the same busy box, within-arm spreads were **0.18-1.71% on
   c194/I16** and **1.46-4.09% on c183/I15e**, against an idle-box figure of
   0.18% -- so contention inflated the small geometry **8x to 23x** and left
   the large one comparatively alone. On c183 the effect being measured (2.4%)
   was smaller than one arm's spread (3.35%); only interleaving made it
   readable.

   **The mechanism for that difference is NOT established.** The natural story
   -- host work per q is fixed while GPU work is not, and the fill event window
   includes two `cudaMemset`s and the launch (`pipeline.cuh:554-560`) so host
   stalls land inside it -- fails its own arithmetic: that window is entered
   once per side per slab, so slabbed c194 has 8 host-issued windows per q
   against unslabbed c183's 2, and the 4x exposure roughly cancels the 3.8x
   GPU-work ratio. **So do not treat "large geometry is contention-tolerant"
   as a property that transfers.** Tune on an idle box, or measure the spread
   on the geometry in front of you.

   **`nregion` is NOT addressed and cannot be, by this design.** The bucket
   array sizing, `k_apply`'s shared memory and the slab plan all derive from
   `log_region`, so it must be chosen BEFORE allocation -- a predict-then-
   allocate design, not measure-then-use -- and it cannot be judged on fill
   alone, since it trades against apply in opposite directions (+52% apply at
   region 13). Finding 81's traffic model is the way in. Still open.

   The **stream count** (item 1) is a third per-card axis and belongs here once
   item 1 ships: 2 on a 5070, 4 on a 5090, N=8 matching N=4 to 0.1%.

   `--fill-threads` is done; the `nregion` autotuner is not.
   Sweep both axes, since holding one fixed is how the old default was reached.
   Opt-in for the standalone benchmark, where reproducibility is the point;
   default-on is defensible for `--pipeline`, where a per-job knee cannot be
   derived and 15 trial fills cost ~60 ms of a multi-hour run.
3. **Full vs q-truncated factor base — CLOSED 2026-08-18, do not build it
   (finding 67).** Was: compared at equal *deduplicated* yield.
   **Downgraded 2026-08-17 — worth 10–15%, not the ~25% an entry count
   suggests, and it may simply be the wrong trade (finding 59).** Replayed
   offline over the c151 corpus: truncation removes 18.4% of *potential*
   re-finds and **zero unique relations** — but only because that band ends
   exactly at `alim`, which is the favourable case, and because the potential
   (1.52 finds/relation) is well above what is actually emitted (1.19, msieve's
   own dedup). The sieving prize is small for a reason that generalises: cutting
   the base from 33.5M to 15M drops the **entry count 53%** but the **bucket
   records only 8%**, because they go as `Σ1/p ~ ln ln p`. That model is
   confirmed by the C194 lim sweep, where 4× the lims moved the bucket array
   +12% against +12.6% predicted. Entry count is the right axis only for the
   root transform.

   **ANSWERED AND CLOSED 2026-08-18 on a 275M-relation corpus — finding 67.**
   Replayed over the SNFS `17327^61-1` dataset (41.5 GB, 339,445,456 lines,
   band `q in [40M, 175M)` recovered from the corpus, `rlim` 67.1M):

   - **On the band as run, truncation loses zero unique relations and removes
     1.78% of the duplicate finds.** The zero is structural, not lucky: under
     truncation a relation is found at its largest band prime `q`, and any
     unsieved FB prime above `q` would itself be a band prime larger than `q`.
     So **truncation is lossless exactly when `band_top >= lim`** — which is
     also why the c151 showed zero, its band ending precisely at `alim`. Both
     corpora were the favourable case.
   - **In every configuration an operator would actually run, it is lossless.**
     Only two shapes occur: `lim <= band_bottom` (truncation never binds) and
     `band_bottom < lim <= band_top` (this job). A band whose *top* is below
     `lim` would cap the base at `q < lim` for every q in the run, so the `lim`
     that was set is never the one used — an operator lowers it instead. The
     sweep confirms the boundary sits exactly at `band_top = lim`: below it the
     trade goes negative (9.6% of unique relations lost at `0.70 x lim`, 6.0%
     at 0.80, 2.8% at 0.90, 0 at 1.00), but that regime is a **test band**, not
     a deployment.
   - **This reconciles all four replays.** The two corpora that lost relations
     are the two whose band stops short of `lim`, and neither is a production
     run: the c147's band is 0.15M wide (a probe, `band_top` = 0.45x `alim`,
     22.68% lost) and the snfs236 corpus is a partial slice (0.28x `rlim`). The
     two complete bands — c151 at exactly 1.00x `alim`, snfs2 at 2.61x `rlim` —
     both lose **zero**. So the c147's 22.68%, quoted here as the alarming
     number, is a property of a narrow probe rather than of the design.
   - **Generous `mfb` collapses both sides to under 2.5%.** At `mfb 92` — the
     c183's value, and close to the C195's `mfba 95` — the unsieved primes fit
     in the cofactor, and truncation becomes very nearly a no-op. So the
     C195-shaped job this item was worrying about is the case where truncation
     matters *least*, in either direction.

   **So there is no version of this worth building.** The duplicate prize is
   1.8% where it is safe and is outweighed by outright loss where it is not,
   and the job class we care about is the one where it does nothing at all. The
   separate sieving-work prize (fewer factor-base entries per q) remains what
   finding 59 measured — 10–15%, bucket records only 8% — and is unaffected by
   this; it is also unaffected by *this* item, since it needs the same per-q
   truncation machinery for a return that finding 59 already called small.

   Two by-products worth keeping. **P(re-found) is 0.6968 here** against 0.702
   (snfs236) and 0.723 (c151) — three jobs, one number. And the **`k = 1`
   stratum is a free integrity check on any corpus**: relations with only one
   possible q cannot be re-found by our convention, so that row must read
   exactly 1.0000 copies. Here it read 1.0141, and histogramming those
   duplicates by q localised them to `q` in **[55,469,851, 57,129,679]** — 100%
   duplicated inside that ~1.66M-wide window, exactly zero outside it. That is
   a **stop/restart overlap sieved twice** (this run predates 12a, so the halves
   were stitched by hand), not a property of the siever. Removing it takes the
   duplicate inflation from 1.2325x to **1.2096x**. It is also the clearest
   measurement of what 12a's byte-exact resume is worth: ~1.7M q of duplicate
   sieving avoided.

   `nactive = lower_bound(primes, q)` on the already-sorted base, passed to
   transform/fill/resieve — no rebuild, no re-upload. The standalone harness
   already truncates statically (`--fbbound`, `fb_restrict` at
   `bench_main.cu:1252`); what is missing is making that bound *per-q* in the
   pipeline, where all three kernels still take the full `fb->n`
   (`pipeline.cuh:221`, `:227`, `:644`).

   **Measured 2026-08-07 over relations already on disk, no GPU.** Checked
   against ground truth on the c151: the tool finds **10,594,292 duplicates**,
   digit-for-digit the count msieve's own filtering reported (`RUNBOOK.md:461`).
   **The denominators do not match, though** — msieve quotes that count over
   67,165,877 relations, while the file on disk holds 67,043,952, which is also
   what this repo's own run record says (`RUNBOOK.md:399`). The 121,925 gap
   makes the ratios 15.77% and 15.80% rather than one number. An exact match on
   the duplicate count across different totals is what you would see if those
   extra lines were all unique — free relations are the obvious candidate — so
   the duplicate-detection logic looks validated and the *denominator* is what
   remains unreconciled. Do not quote this as "reproduces msieve exactly" until
   it is.

   | corpus | band as run | band / lim | `mfb` | raw inflation | truncation floor |
   |---|---|---:|---:|---:|---:|
   | c147 | [15.00M, 15.15M] | 0.4% | 59 | **1.0021×** | −17.3% |
   | snfs236 (partial) | [30.0M, 36.97M] | 5.2% | 88 | **1.0391×** | −6.1% |
   | c151 (complete) | [15.0M, 33.5M] | 55% | 59 | **1.1877×** | −15.0% |

   ("truncation floor" is over a band reaching `lim`; see point 3.)

   Three things follow, and two of them cut against doing this first.

   1. **The duplicate share is not a constant of the design — it is set by how
      much of the factor base the q band covers.** The 15.8–25% figure quoted
      under "Known defects" is a *whole-FB-band* number. A narrow band has
      almost nothing to deduplicate, so any A/B run on one will measure noise.
   2. **Truncation is yield-neutral only if the band reaches `lim`.** Under
      truncation a relation is found at its *largest* sq-side FB prime; if the
      band stops below that prime, it is never found at all. On the c147's
      as-run band the mfb ceiling alone says truncation would find **nothing**
      for 22.68% of unique relations. That is outright loss, not deduplication.
   3. **How much truncation actually deduplicates is bounded by `mfb`
      headroom, and on the c183's configuration that headroom is large.** An
      unsieved prime in `(q, lim]` does not vanish — it lands in the cofactor.
      On the snfs236 (`mfbr 88`, 3LP) it fits there comfortably, so over a
      hypothetical full band [30M, rlim] truncation is guaranteed to remove
      only **6.1%** of emissions against the full base's 1.7918× — the rest of
      the way down to 1.0× depends entirely on whether the survivor threshold
      refuses those relations. The c183 carries `mfba 92`, i.e. *more*
      headroom still.

   So the honest range for truncation on a c183-shaped job is "removes between
   6% and ~44% of downstream work", and **where it lands inside that range is
   item 7's question, not this one.** Items 3 and 7 are one experiment: our
   gate is the loose one, and a loose gate is exactly what lets a truncated
   base keep re-finding the duplicates truncation was supposed to remove.
   Run them together, on a band wide enough to have duplicates to remove.

   The 1.7918× on the snfs236 independently reproduces the 1.82 mean sq-side
   primes in `RUNBOOK.md:444`, and **P(re-found) now agrees across both jobs
   measured directly rather than back-solved**: 70.2% on the snfs236
   (n = 1,089,564) and 72.3% on the c151 (n = 13,709,863), against RUNBOOK's
   71.8% at n = 1,023 and its 72.8% back-solve. Two cautions for anyone
   redoing this:

   - The raw slot-take rate is **biased low** — a relation is in the corpus
     only because one slot already hit, so subtracting that forced hit from
     both numerator and denominator understates P. At k = 2 the correction is
     exact (`E[copies | copies ≥ 1] = 2/(2−P)`, so `P = 2r/(1+r)`); it is what
     turns the raw 54.1% / 56.6% into 70.2% / 72.3%.
   - **Detect the band from the run, not from the corpus.** Counting hits per
     prime does not separate band q from ordinary primes at the low end: an
     ordinary p divides ~N/p relations, so on a 67M-relation corpus every
     prime near 1M clears any fixed threshold. Doing that put the c151's band
     at [1M, 33.5M] instead of [15M, 33.5M] and produced a P(re-found) of
     18.6% — the false low-end "q" can never re-find anything, so whole
     buckets read exactly 0.0%. Cross-check against the q count in the run log
     (`RUNBOOK.md:110`: 1,088,865 q for the c151, i.e. π(33.5M) − π(15M)).
4. **Host cost — overlap first, graphs second, micro-optimisation last.**
   Identifiable host work is 1.694 ms/q, only **7%** of a 24.30 ms idle wall,
   but it triples under CPU contention (finding 53). It is three different
   problems and they do not have the same fix:

   **MEASURED 2026-08-17, and it moves this item DOWN — finding 56.** The free
   experiment below was run and **renice buys nothing**: ~1% in every paired
   comparison, at 12 competing workers and at full saturation alike. The reason
   is that at 16 runnable threads on 16 logical CPUs nothing waits for a slot —
   the feeder thread *shares a physical core* over SMT, and nice values do not
   decide who you share a core with.

   **Affinity, `SCHED_FIFO` and `--blocking-sync` were then all tried and all
   fail** (finding 56). `taskset` pins the whole process including the CUDA
   runtime's threads, so pinning *alone* costs 5.8% and reserving the physical
   core only returns to parity; `chrt -f 1` costs 1.6%; `--blocking-sync`
   costs 2.4%; and the two combined — the textbook low-latency recipe — are
   the **worst** arm at 4.5%. **Every zero-code lever is exhausted.**

   **What the contention actually is:** it tracks *oversubscription*, not
   busyness. At one runnable thread per logical CPU it costs ~6%, measured
   twice independently (12 real sievers, 15 spinners); at 1.7x subscribed it
   roughly doubles. No scheduler knob reaches it because it is contention for
   core resources — SMT siblings, cache, memory bandwidth. **The operational
   lever is worker count: keep competing work at or below `nproc - 1`.**

   Note also that `taskset` never keeps *others* off a CPU. Real reservation is
   cgroup v2 `cpuset.cpus.partition`, `isolcpus=` or `nohz_full=`; under WSL2
   all of them still sit above the Windows host's own scheduler.

   Two numbers now bound the prize. On a **verified-idle box** `acc/wall` is
   0.885 and GPU utilisation 89.5%, so the structural gap is **11.5%** on the
   c147 — contention adds ~6% at 12 workers and ~14% at saturation on top.
   But the gap **shrinks with area**, because host work per q is fixed while
   GPU work is not: 11.5% at `2^27`, 5.4% at `2^29`, **3.8% at `2^30`**. At the
   geometry a C195 would deploy at, this whole item is worth under 4%, part of
   which is interleaved launch/sync that overlap cannot recover. Treat it as a
   small-job concern; it is not a prerequisite for the LPB work or item 0.

   Two thirds of it is *prep* (host per-q tables/staging, small-prime tables)
   and one third is *launch + sync overhead* (`host: unaccounted`). **Finding
   53 holds the canonical table** — do not copy it here; it has drifted once
   already.

   1. **Overlap the prep with GPU execution.** Both prep terms depend only on
      the q-lattice, not on any GPU result, so q+1's host work can run during
      q's kernels. This is double-buffering, **not threading**: 1.166 ms of
      prep against ~20.7 ms of GPU work per q fits entirely inside the GPU's
      shadow with 18× room to spare, so perfect overlap takes it to *zero* on
      the critical path. Threading the same work would reach maybe 0.4 ms with
      four threads and leave it *on* the critical path — strictly worse, for
      more code and a synchronisation problem we do not currently have.
   2. **CUDA graphs for the per-q kernel sequence.** `unaccounted` is
      wall-minus-device inside TD/classify: the CPU issuing launches and
      waiting on syncs. Overlap cannot help — it is interleaved with GPU
      execution by nature — and it is the term that grew **443%** under load,
      so it is what makes the box fragile. The per-q sequence is fixed, so it
      can be captured once and replayed.
   3. **Then make the prep cheaper** — three-way partition replacing the per-q
      stable sort, fusing the small-ideal transform that is currently done
      twice, and **hoisting the per-q allocations**: `pipeline.cuh:169-183`
      mallocs and frees five temporaries (`idx/tp/trt/tg/tlp`) per side per q,
      i.e. 10 malloc/free pairs per special-q, plus three memcpys back over the
      pinned arrays. Reusable scratch removes that churn, and it is the part a
      better sort algorithm does *not* fix. **This was previously the whole of this item and it is the least
      valuable third of it**: after (1) the work is hidden in the GPU shadow
      and costs nothing regardless of how fast it is.

   **Priority note, 2026-08-07 (owner).** Sharing the box is a **requirement,
   not a robustness nicety**: a sieve that only performs on an idle host is
   much less useful, because the host is a desktop someone wants to use. That
   moves this item up and changes what "done" means — the target is that the
   GPU does not wait on the CPU, not that host work is small on a quiet box.

   **Do the free experiment before any of the code below.** The host thread is
   not obviously starved of CPU *time* — measured 2026-08-07 it holds 92.9% of
   one core, because CUDA's default sync spins. What it competes for is the
   *moment* it needs to issue the next launch, and at equal priority it loses
   that race as often as it wins. Two zero-code levers:

   - **Renice the competing work.** Raising your own processes' nice value
     needs no privileges. With the sieve at nice 0 and everything else at 19,
     CFS weights the sieve ~1024 to 15 and it wins essentially every contest.
   - **Leave headroom, but know what you are leaving.** This box is 8 physical
     cores / 16 threads, so 14 `ecm` workers plus one spinning sieve thread is
     15 of 16 *threads* and **every physical core is loaded** — there is no
     idle core to fall back on, and the sieve thread shares a core with an ECM
     worker whatever you do. Dropping batch work to `nproc - 2` frees two
     threads, i.e. one core's worth of SMT capacity, not a quiet core. That is
     why the renice lever matters more here than on a wide box: on 8 cores you
     cannot isolate the sieve by subtraction, only by priority.

   **The live instrument is GPU utilisation, and it is now automatic.** Every
   `--log` record carries `gpu=` and `acc/wall` (item 12b), so the pair of runs
   this rule needs is a `grep`. The rule was: if renicing moves utilisation
   toward the mid-90s the penalty is a scheduling problem, and otherwise the
   dependency is architectural. **Resolved 2026-08-17: it does not move, so the
   dependency is architectural** — but see the measured prize above before
   concluding that makes this item urgent. A caution that survives: 88–89%
   during the co-scheduled snfs236 run is *not* comparable to the 89.5% idle
   figure on the c147, because utilisation is a function of the job's area as
   much as of the host.

   One caution retained: this interacts with item 1: two concurrent q roughly
   **doubles host work per unit time**, so on a contended host that experiment
   can come back negative for host reasons unrelated to the GPU. Run it on a
   verified-idle box and check `GPU-accounted / wall` first.
5. **True rectangle maximum** for the norm approximation — one safe band-wide
   scale, or a few scale/slice-table buckets, gated at low/mid/high q on more
   than one job. Note the payoff is job-shaped: for the snfs236 polynomial the
   special-q side is degree 1 with `d = [1, -3.5e-24]`, so largest-term and
   true-rectangle are the *same number* and this buys nothing there. The ~2-bit
   gap is a GNFS-side problem.

   **The 1:1 anomaly that drove this item DOES NOT EXIST — finding 65,
   2026-08-17.** `gnfs-lasieve4I15e -J 15` covers what we call `2^16 x 2^14`,
   while our `--J 32768` sieves `2^15 x 2^15`. (It is a `2^15 x 2^15` square in
   GGNFS's own axes; the two sievers order the reduced basis oppositely, so
   their (i,j) is our (j,i) — and because their j is non-negative while our i
   is signed, the swap also halves one axis and doubles the other rather than
   being a plain transpose. Corrected 2026-08-18 — see finding 65.) Findings 58 and 63 compared a square against a wide
   rectangle. Recovering each run's rectangle from its own relations (invert
   the q-lattice) and re-comparing at **matched** rectangles gives ours/theirs
   = 0.9795, 0.9805, 0.9797, 0.9786 at `2^14 x 2^13`, `2^15 x 2^13`,
   `2^15 x 2^14` and `2^16 x 2^14` — **flat, with no aspect-ratio dependence
   and no j-dependence within a rectangle**. Finding 63 is retracted; its four
   eliminations stand.

   `sieve_allowance` and `sieve_bound_checked` are **exonerated by
   arithmetic**, no GPU needed: the allowance is `mfb + max(2/scale, 1.5)` with
   no geometry term, so the gate in *bits* goes 96.63 → 96.67 between those two
   rectangles — very slightly looser. The bound falling 119 → 117 is
   `las_scale` renormalising for a 3.86-bit larger norm, and both values
   reproduce exactly from the source. Do not start there.

   So this item is back to being justified only by its original ~2-bit argument
   and the ~0.13% of CADO's relations under "Known defects": **low value at the
   geometry we deploy, and no longer blocking anything.** The ~2% we sit below
   GGNFS is uniform across every rectangle and every j, which is the shape of a
   constant downstream residual (finding 62), not of a norm approximation that
   degrades as the rectangle grows.

   **What replaced it is an operational rule — buy area with I, not J.** At
   equal area the wide rectangle yields +11.7% at `2^28` and +16.3% at `2^30`
   for −0.3% device time, because doubling I costs 1.86 bits of
   `log2(maxnorm)` where doubling J costs 3.86: `i` multiplies the shorter
   vector of the reduced q-lattice and `j` the longer one. See item 0.
6. **Whole-box power**, to close the metric of record — the meter half of
   item 0. **The meter exists**: the UPS reports real watts. It is not a
   parallel task to item 10, it is a **prerequisite** for it (see there).

   Readings (2026-08-07, UPS, real watts). Monitors measured once at **~45 W**
   and subtracted throughout; the raw readings were taken with them on.

   | state | monitors off | notes |
   |---|---:|---|
   | **everything idle** | **95–115 W** | GPU P8 ~16 W, load avg 1.06 |
   | CPU full (14 `ecm`), GPU idle | 240 W | ≈9.6 W per busy core |
   | CPU full, GPU sieving | 395–400 W | GPU board 165 W |

   | derived | |
   |---|---:|
   | **host constant** (idle box incl. P8 GPU) | **~105 W** |
   | GPU **delta**, at the wall | 155–160 W |
   | GPU **delta**, board sensor (165 − 16) | ~150 W |
   | implied delivery loss | 3–7% |
   | power factor (W ÷ VA) | 0.96 |

   The idle reading is **jumpy, ±7%** — normal for a desktop drifting through
   C-states — so average it rather than taking an instant. Everything derived
   from it inherits that spread.

   **The board sensor is trustworthy.** Compare deltas to deltas: the card
   swings 150 W on the board sensor and 155–160 W at the wall, so the wall
   figure sits a few percent *above* the board figure, which is exactly what
   PSU and delivery losses predict. First independent check this project has
   had on the number every rel/J figure rests on, and it passes.

   **The GPU never reaches zero.** It idles at 15 W in P8 and it drives this
   box's displays (`Disp.A = On`), so there is no configuration of this machine
   where the card is absent. That 15 W belongs in the host constant on *both*
   sides of the item-0 comparison, which is the honest accounting anyway — the
   CPU box needs display output too.

   Cautions. **Read W, not VA** — VA is apparent power, ~4% high here. **Subtract
   the monitors** once and for all. And **none of these is a sieve-only number**:
   the ECM job is in all of them, and per finding 53 it is simultaneously
   costing the sieve throughput, so both sides of rel/J are wrong here. These
   are the *co-scheduled* data, worth having, but not the verdict data.

   **Elastic background load is still a trap**, even though it did not bite
   here. ECM expands to fill whatever cores are free, so stopping the GPU job
   handed it the sieve's spinning core and it drew *more* CPU power — the 240 W
   row therefore contains somewhat more CPU work than the 395–400 W row does,
   which understates the GPU delta by roughly the draw of one core. The
   agreement above survives that because the residual is ~10 W against a 150 W
   signal. It would not survive it on a smaller difference. Pin the background
   worker count across both readings, or stop it.

   **The host constant is now measured, which was the term every projection in
   this file was missing.** At ~105 W it is roughly a third of a sieving box,
   not the 200 W+ the earlier ranges allowed for.

   **The CPU side is measured too**, on the c183 at q=130M, 16 workers, all
   monitors-off and steady-state (2026-08-07):

   The box is a **9800X3D: 8 physical cores, 16 threads, 96 MB L3**, so "16
   workers" is two per physical core and the per-worker column below is per
   *thread*, not per core.

   | load | whole box | per worker |
   |---|---:|---:|
   | idle | ~105 W | — |
   | 14 `ecm` workers | 240 W | 9.6 W |
   | 16 `gnfs-lasieve4I14e` | **225 W** | 7.5 W |
   | 16 `gnfs-lasieve4I15e` | **220 W** | 7.2 W |
   | GPU sieving alone (derived) | ~270 W | — |

   **Power falls as the workload gets more memory-bound**, which is exactly why
   ECM was a bad proxy: it is ALU-bound and drew 33% more per core than the
   siever it was standing in for. The earlier ~260 W estimate scaled from ECM
   was ~18% high. Use 220 W for the CPU side.

   > **SUPERSEDED 2026-08-20 by finding 71, which metered both sides again.**
   > The CPU row is **~236 W**, not 220 W, and the GPU row is **240 W**
   > measured, not ~270 W derived. The CPU difference is not the cores: this
   > box idles its GPU at **30 W**, not the ~16 W P8 figure assumed above, and
   > that is 14 of the 16 W. The GPU row also moved because the card has been
   > undervolted since (finding 61). Both new figures are UPS readings with the
   > 45 W of monitors subtracted, taken in one session with the two workloads
   > run separately. Use **240 W GPU / 236 W CPU**.

   Throughput measured in the same session, so both sides of the CPU figure
   come from one box on one day:

   | siever | ms/q | rel/q | rel/s | J/q | rel/J |
   |---|---:|---:|---:|---:|---:|
   | I14e | 166.2 | 31.6 | 190.4 | **37.4** | 0.846 |
   | I15e | 474.8 | 68.9 | 145.0 | **104.5** | 0.659 |

   **I15e measures 475 ms/q here against finding 43's 295 ms/q** — same siever,
   same 16 workers, same stated q, 1.6× apart. A worker sweep run to chase it
   eliminated every mechanism except one:

   | workers | finding 43 | measured 2026-08-07 |
   |---:|---:|---:|
   | 8 | 2.120 q/s | 1.140 q/s |
   | 16 | 3.392 q/s | 2.106 q/s |
   | SMT gain | 1.60× | **1.85×** |

   Not cache capacity — 16 workers beat 8 by 1.85×, so SMT is working and the
   96 MB L3 is not the binding constraint at I15e (a tempting hypothesis on an
   X3D part, and wrong). Not thermal — cores held 4691 MHz under sustained
   load. Not startup — these are steady-state windows. The slowdown is
   *uniform across worker counts*, and contention or thermal effects would bite
   harder at 16 than at 8, not equally. **That leaves configuration**: finding
   43 does not record which `input.job` it swept, and this run used the c183
   (`alim 134.2M, mfba 92, alambda 3.5, lpba 32`) at 68.9 rel/q. A job yielding
   fewer relations per q would sieve faster per q and prove nothing.

   So the two numbers are probably measuring different work, and **s/q is the
   wrong axis to compare them on — rel/J is the right one**. Resolve by
   recording the job file alongside any future N_eff sweep.
7. **Why our survivor gate is looser than GGNFS's at matched lambda —
   DOWNGRADED 2026-08-17: the surplus does not exist at shipping defaults
   (finding 62).** Measured at each siever's own gate, our cofactor volume
   matches GGNFS's within **0.4%** on item 7's own config (c183 I14e: 784.85
   against 781.7 per pair) and within **0.03%** on the C194 at 15e (1594.12
   against 1594.8). Relations from those submissions differ by +0.9% and −1.7%
   respectively — opposite signs, so no systematic deficit either.

   The observation below stands as written, because it was made *at matched
   lambda*: a given nominal bit value does mean different things to the two
   gates. What is now measured is that this asymmetry **produces no surplus at
   the defaults we ship** — our derived allowance lands on the same cofactor
   volume GGNFS's lambda rule does, on two jobs and two geometries. The
   operational worry below — that the bound "can only be set by sweeping, never
   derived for a new job" — is answered: it *is* derived, and the derivation
   now has two independent external checks. **No sweep is needed for a new
   job.** Reopen only if a job is found where the volumes diverge.

   The original observation, retained:

   Same q
   range, same job, same nominal bits: `gnfs-lasieve4I14e` loses 17.3% of its
   yield going 91.8 → 87.5 bits where we lose 0.07% going to 88.0. We submit
   1,426 cofactors per special-q against GGNFS's 978 (`COF: 60664 tests`, 62 q)
   for comparable *unique* yield. So we admit ~46% more and the surplus is
   almost all unproductive — time, not relations. **This is also the other
   half of item 3**: the dedup a q-truncated base buys is exactly the set of
   relations the gate refuses once their `(q, lim]` primes move into the
   cofactor, so a loose gate spends the truncation prize before it is
   collected. Measured, truncation is guaranteed only 6.1% of the available
   ~44% on an `mfb 88` job; the gate decides the rest. Do not run these two
   apart. Until the cause is known the
   bound can only be set by sweeping, never derived for a new job. Candidates:
   a systematically low per-position norm estimate; the byte scale cancelling
   only if cell init and bound use it consistently; GGNFS's lambda gating more
   than the report threshold. **Do not "fix" it by tightening the default** —
   the current gate finds every relation GGNFS finds (verified, zero misses
   over 2,531), so any trade of yield for speed must be made deliberately.
8. **A = 32 sieve areas and MFB widening**, for jobs like AS276. **Two of the
   three blockers are gone**: `lpb` now goes to 64 (2026-08-17), so a C194's
   `lpba 33 / mfba 95` runs today; and `mfb` now goes to 128 (2026-08-18) on a
   per-side 3-or-4-limb dispatch, so a C208's 4/3 shape runs without a flag.
   AS276 has since been sieved end to end and validated against its own GGNFS
   corpus at 99.97% recall (finding 69), and the width measured at **×1.72 on
   the widened queue, +8-9% of wall** (finding 70) — the 1.8-2× projection was
   close.

   **CORRECTED 2026-09-01 — the area blocker is gone; what is left is one
   untested aspect ratio.** `--pipeline` applies no total-area cap at all:
   `bench_main.cu`'s `I*J <= 2^31` refusal is guarded by `!cfg.pipeline`, the
   non-pipeline
   path, and the planner slabs any geometry through `logI 20`. **`A = 32` has
   been sieved twice** — `I16 J65536` on the 5070 (finding 74's discriminating
   run, 8 slabs of `2^29`) and `I=J=2^16` on the L40 (finding 72).

   **CLOSED 2026-09-01 by finding 82: their shape is sieved, gated and
   geometrically confirmed.** AS276 at `--logI 17 --J 32768 --maxbits 17`, 10 q
   from 80000023, plans **8 slabs of 4096 rows** unaided and emits 1,322
   relations; `--check-relations` rebuilds **1,322 of 1,322** norms exactly, and
   `relgeom.py` recovers `i in [-65477, 65239]`, `j in [1, 32761]` — `2^17 x
   2^15`, the same extent finding 69 recovered from GGNFS's own output for this
   job. Setup allocation 2.71 GB (bucket array 1.18), so a 12 GB card runs it.
   Finding 69's "their shape is `A = 32`, which we still refuse" is dated
   2026-08-19, predates the slab merge, and is no longer true; the `2^17 x
   2^14` runs in findings 69 and 77 are `2^31` sub-rectangles and finding 77's
   "(A=32)" label on one was wrong. **The 14-16 GB figure applies only to the
   monolithic allocation nothing makes now.**

   **What is left of this item is a performance comparison, not a capability.**
   A like-for-like run against NFS@Home at this geometry needs an idle card and
   a matched q band; finding 82's run was on a card at 96-100% foreign load and
   quotes no timing.

   **The per-slab cost side of that footprint improved on 2026-08-26**
   (finding 77): halving the slab now costs +1.7% rather than +3.4%, and on
   AS276 itself the recording pass fell 9.234 -> 1.256 ms/q. That does not lift
   either remaining blocker, but it is what makes trading slab size for VRAM
   cheap enough to be worth doing. Item 18 is the next term in the same
   direction.

   **Neither remaining blocker binds a C195 at NFS@Home's geometry.** They
   sieve `2^15 × 2^14` and would prefer `2^15 × 2^15` — `2^30`, half the
   current area limit — so no A=32 work is required for that class of job at
   all. The current design assessment, performance accounting, alternatives,
   and work status are consolidated in **"Current size limits and j-slabbing"**
   above; that section is canonical rather than duplicating a moving design
   here.
9. **Dead factor-base parses under `--sq-side 0` -- CLOSED 2026-09-02,
   finding 86: BOTH DEFECTS ARE STALE and startup on the client's job class is
   1.9 s.** (The 15-20 s snfs236 figure was not re-measured and is not
   refuted; it is simply not a number the client pays.)

   The item described code that had been restructured underneath it, and was
   promoted on that stale reading. In a `--pipeline` run everything lives in
   `if (cfg.pipeline) {` (`bench_main.cu:1873-2871`) and inside it `fb1` is
   loaded once (`:2640`/`:2645`), `fb_fill_logp`'d once (`:2646`), and
   `rfb_build` is called **once** (`:2726`).

   - **The throwaway `fb1` parse is gone.** No factor base is loaded anywhere
     before line 1873. The item's own "used to supply the q list" was the
     whole story.
   - **`rfb_build` does not run twice.** The second call at `:2889` is in the
     `else` of `if (cfg.side == 1)` (`:2876`) -- a *sibling* of the pipeline
     branch, i.e. the standalone path. Same call, other run mode.

   Measured, `T(n) = S + n*p` from `--nq 1` against `--nq 21`, reps agreeing
   to 4-21 ms, fitted per-q landing on the band's own 97.5 ms/q:

   | config | startup | of a 900 s work unit |
   |---|---:|---:|
   | c183, `--sq-side 1` | **1.91 s** | 0.21% |
   | c183, `--sq-side 0` | **1.88 s** | 0.21% |
   | c194, `--sq-side 1` | ~3.45 s | 0.38% |

   The `--sq-side` pair is the direct test: the dead parse was specific to
   side 0, so that row should have been the expensive one, and it is 0.03 s
   cheaper. Startup scales with `rlim`, not with ms/q, and the client's job
   class has a smaller base than c183 -- so it pays **at most ~1.9 s of ~900 s,
   0.2%**. The 15-20 s figure was snfs236, a bigger base than the client ever
   sieves.

   **Nothing to build.** The 2%-of-every-unit-forever case that motivated the
   promotion does not exist.

   Two things this leaves open, neither worth scheduling on its own:
   `--fb1`-omitted runs now do in-process GPU factor-base *generation*
   (landed 2026-08-24) and were not timed here, though `client.c` passes an
   explicit `--fb1` so the client never takes that path; and c194's `T(21)`
   spans 8.19-11.10 s across reps, so its ~3.45 s is a bound rather than a
   measurement.

10. **GPU power-limit sweep — MEASURED 2026-08-17 (finding 61); a floor, not
    a knee.** The premise was that consumer cards ship past their efficiency
    knee and a 60–80% cap buys 15–30% rel/J. **The sieve is not power-limited
    at stock**: it draws ~195 W against a 250 W limit, so only 70–77% binds at
    all, and 70% is `power.min_limit`.

    At the floor (175 W), three runs per setting with byte-identical relations:
    **−10.0% board power for +1.83% time**, giving **+9.1% board rel/J and
    +5.0% whole-box rel/J**. Nine percent of the power for half a percent of
    the clock (2917 → 2902 MHz) — past the knee, as predicted.

    **The board sensor overstated the gain 1.8×** (9.1% against 5.0%), the
    mechanism this item warned about: the cap lengthens the run and the ~105 W
    host constant is paid for the extra time. Grading on the board sensor would
    pick a cap that is too low. Robust to the host constant — 115 W gives 4.8%.

    **The undervolt is the real lever, and it is ~3× the cap.** The card was
    never power-limited; it sat far past its efficiency knee at 2910 MHz /
    1080 mV. Re-pinned to ~2900 MHz at **950 mV**: −28.0% board power for
    +6.7% time, i.e. **+30.2% board and +14.6% whole-box rel/J**, and 50–56 °C
    instead of 69 °C. Both correctness gates were run first — `cofcheck`'s 30
    exact counts and a byte-identical c147 band — because an unstable undervolt
    computes *wrong answers* rather than crashing, which is the failure mode
    that actually threatens a sieve.

    **Applied to this box permanently**, so see the note under "Measured": every
    timing after 2026-08-17 is ~6.7% slower than the numbers in findings 43–60.

    Still unmeasured: **where the knee is**. 900 mV was not tried, so 950 mV is
    known to be past the knee and stable, not known to be optimal. Reopen if
    the last few percent are wanted, on this card or a different one.

    Item-0 verdict: **3.10× time and 3.14× energy** against the CPU box at the
    undervolt, from finding 58's equal-work 3.31× / 2.74× at stock.
11. **Apply breakdown — ANSWERED 2026-08-17 (finding 60) as "no cheap win";
    that conclusion is PARTLY WRONG, corrected 2026-08-25 (finding 75).** The
    cheap win was occupancy, which this item never examined: one
    `__launch_bounds__` was worth **apply −12.6%, wall −4.6%**. The diagnosis
    below is still right and still worth reading — see the correction at the
    end of the item for what it missed and why.

    Apply is where the milliseconds are — **54% of the sieve
    chain** on the C194 at its deployment geometry, 21.94 ms against fill's
    15.47 — which is why this was ranked ahead of further fill tuning.

    It is **not** at its memory-system limit: DRAM 9.0%, L2 bandwidth 4.7%,
    1.43 sectors per global load request, 341 waves per SM, SM throughput 71%.
    The opposite character to fill, which is a latency-bound scatter with the
    SMs idle — so the prior-art warning about the L2 transaction ceiling
    (`~/msieve-s/POLYSELECT_OPTIMIZATION_NOTES.md`) describes neither kernel.

    **But no single pipe is saturated either** — LSU 32%, ALU 30%, XU 23%, FMA
    19%, issuing 0.70 instructions per cycle, largest stall short-scoreboard on
    the shared-memory cells. Apply is *issue-limited with a balanced mix*.

    Every identifiable lever was priced and all are small: norm init is 29% of
    apply but `__log2f` recovers only 2.8% of it and dropping the fp64
    cancellation guard only 1.8%; smem atomics are 3.4%; halving the cell width
    moves 0.3%. **The accurate-`log2f` and cancellation-guard decisions both
    stand** — each buys correctness for under 3%.

    What remains is algorithmic: fewer instructions per position or per record,
    i.e. a change to *what* apply computes. That is a much larger piece of work
    than this item was scoped as, and nothing in the tuning space is worth
    doing first. A `NORM_FAST_LOG2` build switch was added alongside the
    existing `NORM_CANCEL_TOL` so both prices can be re-measured on another job
    without editing the kernel.

    > **PARTLY WRONG — CORRECTED 2026-08-25, finding 75.** "No cheap win" held
    > only because every lever priced here was an *instruction-count* lever.
    > The diagnosis above is right and still stands — apply is issue-limited
    > with a balanced mix, 0.70 instructions per cycle per scheduler — but
    > **occupancy was never examined**, and it was the cheap win.
    >
    > `k_apply` compiled to 45 registers, which fits two 512-thread blocks per
    > SM and pins theoretical occupancy at 66.67%; shared memory would have
    > allowed three. `__launch_bounds__(512, 3)` brings ptxas to 40 registers
    > with **zero spill**, occupancy to 99.66%, and SM throughput from 72.9% to
    > 84.7%. Measured paired and idle: **apply −12.6%, sieve −7.1%, wall −4.6%
    > on the C194**, relations byte-identical, uniform across all three
    > geometries tested.
    >
    > The lesson generalises past this kernel. An issue-limited kernel with no
    > saturated pipe is *also* the signature of one starved of warps, and this
    > item read that signature as "nothing left but algorithms". **Check
    > `Block Limit Registers` against `Block Limit Shared Mem` before
    > concluding a kernel is out of cheap headroom** — `ncu` prints the
    > limiter and an estimated speedup outright, and here it said 33.33%.
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
13. **Validate the BOINC GPU assignment — CLOSED 2026-08-17.** Greg Childers,
    who reported the original failure (every task on a multi-GPU host landing
    on device 0), reviewed and signed off on the assignment change, and a BOINC
    queue is now running with it. Both things this item said could not be
    checked here — that the client emits `<gpu_device_num>` for this app
    version's plan class, and that concurrent tasks land on distinct cards —
    are answered affirmatively by a live queue on a real multi-GPU host.

    What is written below stayed open for one session and is kept only as the
    recipe for re-checking the read path after a change to it, since a silent
    regression still looks exactly like success on a single-card box:

    Testable here, no second card and no client: `boinc_get_init_data()` reads
    the `init_data.xml` in the working directory, so a `HAVE_BOINC=1` build run
    in a directory holding a hand-written one exercises the whole read path.
    Four cases worth pinning, each identified by its stderr line — (a) an
    NVIDIA assignment is read and used; (b) an assignment overrides a
    conflicting `--device`, with the notice printed; (c) a non-NVIDIA
    `gpu_type` is refused rather than acted on; (d) no assignment falls through
    to `--device` and then to CUDA's default. An assignment of 1 on a one-card
    box should also produce the bounded "sees only 1 device" refusal rather
    than an invalid-ordinal error.

    Not testable here, and both are project-side rather than application-side:
    that the client emits `<gpu_device_num>` at all for this app version's plan
    class, and that concurrent tasks land on distinct cards. Those need a
    multi-GPU host running the real client, i.e. the reporter who saw the
    original failure.
14. **Use the survivor cell's real resolution — MEASURED AND CLOSED 2026-08-18
    (finding 66): no win, do not build it.** `las_scale` derives the byte scale
    from las's 255-value cell (`poly.c:584`) although ours is 16-bit with
    `CINIT` 4096, so one byte unit is 0.82 bits and 10–13% of the survivor list
    is admitted on rounding error alone.

    **Removing them buys nothing.** Holding the gate fixed in bits and raising
    the scale to 4.0, on an idle box, alternated: **112.44 against 112.43 ms/q**
    — nothing, against a ~1.3% run-to-run spread — while survivors fall 10.3%
    (71,580 → 64,241). Yield is neutral too (6,743 → 6,740).

    The reason is that **the survivor count is not a cost driver on this
    pipeline.** Fill and apply are ~70 ms of the 112 and are per *position*;
    the whole trial-division block is 14 ms and moves 0.25 ms. Candidates/q is
    flat to 0.04% (1590.68 → 1590.08), so the noise survivors never reach
    cofactorisation — the rank scan and survivor filter kill them for 0.5 ms
    total.

    So `las_scale`'s 8-bit inheritance is harmless rather than a defect.
    Recorded chiefly so the inference "fewer survivors must be faster" is not
    made again: on this pipeline the survivor list is nearly free downstream,
    which also means **survivor counts are a poor proxy for work** in any
    cross-siever comparison (see item 7's funnel numbers).
15. **Older cards, and the shared-memory ceiling that blocks them — code
    read 2026-08-20, NOT tested on any such card.** **Fixed 2026-08-21:** both
    the production pipeline and standalone apply benchmark now query
    `cudaDevAttrMaxSharedMemoryPerBlockOptin` from the selected device at runtime;
    the default remains `--region 14`. `GPU_ARCH_all` starts at sm_80, but the
    apply path no longer assumes one architecture family's opt-in limit.

    Before the fix, both `pipeline.cuh` and `bench_kernels.cu` hardcoded
    `101376` bytes as "the opt-in limit". CUDA does not define one universal
    value; representative limits are:

    | target | max dynamic smem per block | old 101376-byte bound |
    |---|---:|---|
    | sm_70 Volta | 96 KB | too high |
    | sm_75 Turing | 64 KB | **too high — the opt-in call fails** |
    | sm_80 A100 | 163 KB | **too low** |
    | sm_86 / sm_89 / sm_120 | 99 KB | matched |
    | sm_90 Hopper | 227 KB | **too low** |

    Apply sizes its shared memory at `(1 << log_region) * 2 + nslice_pow2 * 2`,
    so `--region 15` needs roughly 66 KB and `--region 16` roughly 131 KB. The
    old literal therefore prevented region 16 on A100/Hopper even though the
    hardware permits it, while it would have admitted sizes that are illegal on
    some older devices. The runtime query fixes both directions.

    `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
    remains the opt-in mechanism; the requested value is checked against the
    selected device's reported limit before that attribute is set or the apply
    kernel is launched. This is capability detection only: `--region 14` is
    still the default because larger regions have not been shown to be faster.

    **What is NOT established.** No card older than the current sm_80 build
    floor has been qualified by this change. Lowering `GPU_ARCH_all` would still
    require compilation and byte-comparison testing on that hardware.
    Launch-bounds/shared-memory tuning remains a separate constraint, and as of
    2026-08-25 it is **no longer confined to the fill kernels**: `k_apply`
    carries `__launch_bounds__(512, 3)` too (finding 75), so it targets 1536
    threads/SM and three blocks of ~33 KB. On a 1024-thread/SM part such as
    sm_75 that annotation both trips the `.minnctapersm` warning and cannot
    reach its three-block target — on the kernel that is 54% of the sieve
    chain. Qualify apply alongside fill, not after it.
16. **Build wall time — MEASURED 2026-08-20, and the cause is `CF_LMAX=4`.**
    **See first: the default-goal trap, fixed 2026-08-25 (finding 75).** Until
    that date the Makefile had no `.DEFAULT_GOAL`, and its first explicit rule
    is `$(OBJS) $(TEST_OBJS): ...` — so a bare `make` in `bench/` took its goal
    from the first word of `$(OBJS)` and built **`bench_main.o` alone**, then
    exited 0 without relinking `bench`. It looked exactly like a successful
    incremental build and silently invalidated any measurement taken against
    the binary afterwards, because the binary was the old one. `.DEFAULT_GOAL
    := all` is now set above that rule. **A zero exit from `make` is not
    evidence a build happened — check the binary's mtime before benchmarking
    against it.**

    A full `make all` is ~12.5 min on this 16-thread box and about 30 on a
    busy Vast.ai 5090. The Makefile's timing table (`Makefile:43-55`) claims
    214 s and blames sm_120's ptxas. **That table was measured 2026-08-11
    (efd7219), before the 4-limb/ECM cofactor dispatch landed 2026-08-19
    (e4a47f2), and it is now wrong by 3.5x.**

    Measured here, `make -j8`, clean tree per arm, one target at a time:

    | target | `CF_LMAX=3` | `CF_LMAX=4` (default) | cost of the 4th limb |
    |---|---:|---:|---:|
    | sm_120 | 277 s | **754 s** | **2.72x** |
    | sm_80 | 15 s | 26 s | 1.73x |

    Three things follow.

    1. **The 4-limb dispatch is the regression, not the target count.** At
       `CF_LMAX=3` sm_120 returns to 277 s, near the table's 215 s; the
       residual is plausibly the ECM rewrite that shipped in the same commit.
    2. **`-t 0` already works, so dropping targets buys nothing.** The full
       six-target fat build measured **750 s** against sm_120 alone at 754 s.
       The fat binary is free and adding sm_90 cost nothing measurable — the
       one target that cannot be dropped is the expensive one. This confirms
       the stale table's *conclusion* even though its numbers are wrong.
    3. **The ptxas asymmetry widened.** sm_120/sm_80 was ~15x when the table
       was written and is **29x** now (754/26). The 4th limb costs sm_120
       disproportionately, so this is a ptxas scaling problem on Blackwell
       rather than a linear code-size effect.

    **For iteration today:** `make GPU_ARCH=120 CF_LMAX=3` is 277 s and is a
    shippable binary — the cofcheck gate pins 3-limb and 4-limb output as
    byte-identical, and `CF_LMAX` is deliberately not a `DEFS` value so such a
    build still emits relations (`Makefile:112-130`). An Ampere or Ada box
    with `GPU_ARCH=native` is 26 s. Neither helps a release build.

    **Not yet investigated:** why the 4th limb triples ptxas time on sm_120.
    `bench_kernels.cu` is a single 138 KB translation unit that includes the
    125 KB `cofac.cuh`, and that is the only site including it
    (`bench_kernels.cu:2682`), so splitting the cofactor kernels into their
    own `.cu` would both parallelise across TUs and stop a sieve-kernel edit
    from recompiling the cofactor templates. That is the first experiment;
    instantiation count and register pressure are the obvious suspects.
17. **`--mode twolevel` has a live race in `k_fill_l1` — FOUND 2026-08-24, NOT
    fixed, benchmark-only.** `--mode atomic --verify` matches the CPU
    per-region reference exactly (all 8192 regions). `--mode twolevel --verify`
    does not, and **the answer changes every run**:

    | binary | differing regions, three runs |
    |---|---|
    | merge commit `5e96cf0` (pre-slab-tuning) | 543 / 488 / 564 of 8192 |
    | after the finding-73 walk change | 395 / 365 / 327 of 8192 |

    The card was running an unrelated job at ~100% throughout, so these counts
    are **scheduling-dependent and not reproducible figures** — the two columns
    are not a before/after comparison and the gap between them means nothing.
    What is robust is the shape: every run fails, no two runs agree, and the
    total is always exact.

    It is **pre-existing**: the unmodified merge commit fails it. Note that
    `k_fill_l1` was converted by finding 73 and then **reverted** (it spilled
    against its launch bound), so it is byte-identical to HEAD in the shipped
    tree. The two columns below therefore measure *the same* `k_fill_l1` and
    the gap between them is noise, not an effect — the table shows only that
    every run fails and no two agree. `--mode atomic`, which does carry the
    converted walk, is exact. **For bisecting: this race predates all of
    finding 73's work; do not start from that range.** The record *total* is 77,382,815 every time, matching
    the reference exactly; individual regions are wrong in **both** directions
    (`gpu 9485 ref 9484` and `gpu 9529 ref 9530`). So records are conserved and
    misfiled, which is why nothing downstream ever noticed.

    **The mechanism, from code reading.** `k_fill_l1` reserves a whole run with
    one `atomicAdd(&cnt[b], nrun)` and, when the run does not fit, gives the
    space back with `atomicSub` — but the shared `buf` is never cleared and the
    flush emits `buf[0 .. cnt[b])` unconditionally. A third thread can take a
    slot *above* a failed thread's range after that thread subtracts, leaving a
    **hole below the final count**:

        cnt=58, L1_CAP=64
        T_B add(8) -> slot 58, cnt 66   (66 > 64: will subtract)
        T_A add(4) -> slot 66, cnt 70   (will subtract)
        T_B sub(8) -> cnt 62
        T_C add(2) -> slot 62, cnt 64   (fits: writes buf[62..64))
        T_A sub(4) -> cnt 60
        flush emits buf[0..60) -- slots [58,60) hold LAST cycle's records,
        and T_C's real records at [62,64) are dropped after it advanced x.

    Stale-but-valid positions are re-emitted while live ones are lost, one for
    one. That reproduces the exact signature: conserved total, nondeterministic
    per-region error, both directions. It needs three threads and a specific
    interleaving, which is why it is a few hundred records in 77M.

    `k_fill_l2` is immune and shows the shape of the fix: it reserves **one**
    slot at a time and never subtracts, so slots fill densely from 0 and the
    flush always writes a fully populated prefix. The fix for L1 is to write
    the part of a run that fits and carry the remainder pending, rather than
    reserving all-or-nothing — records within a run are independent, so
    splitting one across two flushes is harmless.

    **Priority is low and the reason is not "it is only a benchmark".** It is
    that `k_fill_l1` is the two-level path, which finding 1 measured as losing
    by 2.7x and which nothing in `--pipeline` reaches. But the verify gate is
    not in `make check`, so this sat undetected; whoever revisits two-level
    fan-out must fix this **before** trusting any number it produces.
18. **Lanes-per-item for the DENSE trial-division pass — the per-slab tax that
    replaced the recording one. NOT BUILT, opened 2026-08-26 by finding 77,
    CEILING BOUNDED THE SAME DAY by finding 78 at ~3% of wall at the realistic
    A=32 geometry. Gated on item 8: build it only if that measurement forces
    slabs below 16.**

    **DO NOT BUILD — the gate was measured 2026-09-01 and it does not fire
    (finding 82).** A 12 GB card runs `A = 32` at NFS@Home's own shape in **8
    slabs of `2^29`** — the sweep optimum itself, not a memory compromise —
    with 2.71 GB of setup allocation. This item exists to make slabs *below*
    `2^29` affordable, and nothing is asking for them: at 8 slabs the dense-TD
    excess is +2.95 ms and the whole item is worth −0.8% of wall, against
    rewriting the hot dense path. Reopen only if a card smaller than 12 GB, or
    an area beyond `2^32`, actually forces the slab count past 16.

    `k_td<1,0,0>` ("norms + trial division, both sides") gives one thread per
    survivor and each thread marches the whole `nsm` list — and because that
    march sits INSIDE the grid-stride loop, a launch costs
    `iters = ceil(n/(blocks*threads))` of them. Measured on the warp binary,
    c194 I16 J32768, default grid `288 x 256` = 73,728 threads, 237,256
    two-sided survivors/q:

    | slabs | survivors/launch | `iters` | total marches | TD |
    |---:|---:|---:|---:|---:|
    | 1 | 237,256 | 4 | 8 | 10.71 |
    | 2 | 118,628 | 2 | 8 | 9.76 |
    | 4 | 59,314 | **1** | 8 | 10.34 |
    | 8 | 29,657 | 1 | **16** | 13.18 |
    | 16 | 14,828 | 1 | **32** | 21.83 |

    **Splitting the slab is free only while `iters` can absorb it.** It falls
    4 -> 2 -> 1 across the first two halvings and the cost does not move; at 4
    slabs it is pinned at 1 and every halving after that doubles the march
    count. That is the entire column, and it is the same disease item 77
    removed from the recording pass — ~0.96 ms per slab past the knee, against
    the recording pass's 0.12. **It is now the larger of the two, by 8x.**

    **The cheap version of this was tested first and does NOT work.** If the
    knee is at `blocks*threads`, raising `--blocks` should move it — two runs
    rather than a kernel rewrite. Three grids x five slab counts, idle card
    (finding 77):

    | slabs | TD b288 | b512 | b1024 |
    |---:|---:|---:|---:|
    | 1 | 10.85 | 9.56 | **9.15** |
    | 4 | 10.49 | 10.38 | 10.43 |
    | 8 | 13.39 | 13.31 | 13.38 |
    | 16 | 21.82 | 21.96 | 22.36 |

    **Where `iters` is already 1 — every slabbed geometry, which is this item's
    entire case — a 3.5x grid range is worth 0.0 to 2.5%, i.e. nothing.** Only
    the unslabbed row moves, and only by −15.6% of one stage (−0.4% of wall).
    So `--blocks` is not a substitute: it is the kernel change or nothing.

    **Taken further on 2026-08-26 (finding 78b): that unslabbed cell is not
    worth taking as a default either.** Three reps, two jobs. On c194 forced to
    1 slab the stage win reproduces exactly (−15.0% against −15.6%, arms
    non-overlapping) but is −0.51% of wall against a 1.8–2.9% within-arm rep
    spread, and the sign flips between reps. On c147 I14/J8192, natively
    unslabbed, it **reverses**: b1024 is worst on every column, +5.6% TD and
    +1.45% wall, because 1024x256 threads already exceeds the survivor count.
    `--blocks` stays on auto. Third measurement pointing the same way, and it
    strengthens this item: **threads are not the lever.** More threads help only
    where `iters > 1` and hurt where `n` is already below the grid — which is
    precisely the regime lanes-per-item exists to serve.

    It also falsified half the model. Cutting the march count 4x at one slab
    bought 15.6%, not 75%, so cost is NOT proportional to marches; the march
    count predicts the shape and not the magnitude. What survives, and what
    this item rests on, is the confirmed half: **at equal march count the grid
    does not matter**, so threads are not the lever and lanes-per-item is.

    **Duplication debt to settle first (code review, 2026-08-26).**
    `k_td_record_warp` already copy-pastes ~90 lines of `k_td` -- the exact-norm
    builder, the special-q/large-prime step, and the whole small-prime
    congruence. Building this item as a third copy would make that three. The
    review's suggestion is right and should be taken with the item rather than
    after it: extract `td_small_hit()` and `td_build_norm()` as device helpers
    and call them from both kernels, or go further and parameterise `k_td` by
    lanes-per-item -- `template <int DIVIDE, int RECORD, int SELECT, int LANES,
    bool SLABBED>` with LANES=1 collapsing to today's per-thread loop and
    LANES=32 to the ballot form -- which covers both call sites with one body.
    That is the shape this item wanted anyway.

    **Do not just apply finding 77's fix again.** One warp per survivor would
    lose wherever survivors exceed the thread count, which is the ordinary
    case at large slabs and at 1 slab is every job. What this wants is
    L lanes per item with L chosen on the host from `n` against the grid —
    `L = clamp(nthreads/n, 1, 32)` rounded to a power of two — so the kernel
    degenerates to today's behaviour at L=1 and spreads the march only when
    there are idle threads to spread it over. `__ballot_sync` gives the hit
    mask for a full warp; at L<32 it needs masking to the item's lane group,
    and the divisions go to the group's first lane rather than lane 0.

    **The ordering constraint that an earlier draft of this item inherited from
    finding 77 does not exist** (corrected 2026-08-26 by code review).
    `td_divide_out` consumes the whole power of a prime per call and distinct
    primes commute, so the factor multiset is order-invariant; both emitters
    sort before writing. This rewrite is therefore free to visit entries in
    whatever order the lane grouping makes cheapest. The acceptance test is the
    same as finding 77's — factor multiset and relation count against the
    scalar path, at both slab shapes — but it is a multiset test, not an
    ordering one.

    **Value, stated honestly: nothing at the operating point.** At `2^29` this
    pass is already near its floor (10.34 against 10.71 at one slab), so the
    prize is entirely below `2^29` — about 2.8 ms at 8 slabs and 11.5 at 16.
    Like item 77's, the reason to build it is that it decouples slab size from
    cost for an A=32 job on a 12 GB card (item 8), not throughput. It is a
    bigger and riskier change than the recording pass was, because this kernel
    IS the hot dense path, so it should be measured against the unslabbed case
    as carefully as against the slabbed one.

    **The other term below `2^29` is `fill`, and MEASURING IT LOWERED THIS
    ITEM'S CEILING -- finding 78, 2026-08-26.** The sweep now runs to 64 slabs
    at fixed total area, and `fill` is a bandwidth floor: `k_fill_atomic` is
    launched with `fb->n` every slab and streams 42 B per entry (`plat` 24,
    `slice` 2, `walk_cur` 8, `walk_next` 8) x 22.1M entries = **929 MB per
    slab** before it can know whether a prime hits. Slope **1.18 ms/slab**
    asymptotically, against the 1.98 the old 4/8/16 window implied -- so the
    floor is real but ~1.7x lower than this item was written against. Skipping
    `plat`/`slice` for non-hitting entries was costed and **does not pay**: 1.1
    ms of 16.1 at 16 slabs, because `p > xmax` still hits with probability
    `xmax/p`.

    The consequence is the column this item was missing:

    | slabs | fill excess | dense TD excess | complete vs `2^29` | **this item's best case** |
    |---:|---:|---:|---:|---:|
    | 8 | +0.31 | +2.95 | +1.5% | −0.8% |
    | 16 | +16.11 | +11.64 | +11.4% | **−2.9%** |
    | 32 | +28.21 | +31.44 | +29.4% | −6.8% |
    | 64 | +66.30 | +71.65 | +71.8% | −11.7% |

    The TD projections above are confirmed (2.95 and 11.64 measured against 2.8
    and 11.5 projected). What changed is that **at 16 slabs -- the worst case a
    12 GB card actually reaches for an A=32 job -- fill's irreducible excess is
    larger than this item's entire prize.** Ceiling at the realistic geometry is
    about 3% of wall. That is not zero, but it is bounded, and it should be
    weighed against the risk of rewriting the hot dense path before anyone
    starts. **Build it only if item 8's geometry measurement forces slabs below
    16.**
19. **An ENVIRONMENTAL ~10%-of-wall regression, cause still open -- MEASURED
    2026-09-02 (finding 88). Worth more than every open item except 1.**

    Rebuilding `4b581b33` -- the exact commit August was built from --
    unchanged, today, reproduces `unaccounted` **0.50 -> 7.86 ms/q**:
    **+7.36 ms/q of GPU idle** (band-length insensitive, so exact) and
    **~10.6 ms/q of wall, +10.4%** at c183/I15e. **Our code is exonerated.**

    **Ruled out by direct test:** the CUDA 13.2/13.3 toolkit line (HEAD built
    against each gives unaccounted 7.87 and 7.96 -- worth ~1.4 ms of wall and
    none of the idle), and the Windows driver / WSL passthrough (every real
    binary in `/usr/lib/wsl/lib` is dated 2026-07-22, and the driver available
    since 2026-08-26 is still not installed).

    **Narrowed to the `apt upgrade` of 2026-08-27 17:51**, which installed
    CUDA 13.3, bumped the 13.2 line (cudart 13.2.75 -> 13.2.86) *and* upgraded
    59 non-CUDA packages. The toolkit test above compares two POST-upgrade
    toolkits, which is why it came out flat.

    It has been masking real work: at matched runtime our apply+fill
    improvements are worth **9.44 ms/q**, which is why the two nearly cancelled
    and why 130M and 190M disagreed in sign.

    **What is NOT known: where inside the runtime it goes.** It is GPU idle
    spread across the per-q launch sequence, so the candidates are launch
    overhead and default-stream semantics. **The instrument is Nsight Systems**
    -- gaps BETWEEN kernels -- not `ncu`, which profiles what happens inside
    one and would show nine healthy kernels and no gap.

    Steps, in order:

    **DONE 2026-09-02, both negative on the cause but positive on a fix:**

    - **CUDA fully eliminated.** HEAD linked against the August-era
      `libcudart_static.a` (13.2.75-1, extracted from the .deb via a
      symlink-farm shadow toolkit, nothing installed) gives `unaccounted`
      **7.83** -- against 7.87 at 13.2.86 and 7.96 at 13.3.29. The exact
      runtime August ran reproduces today's number, not August's.
    - **`unaccounted` is NOT simply GPU idle.** An Nsight Systems trace puts
      the GPU at **97.6% kernel-busy** with total inter-kernel gap
      **3.44 ms/q**, well under the 7.83. Part of that quantity is device work
      outside the event brackets. Read it as "wall not attributed to a
      bracketed stage", and do not repeat the idle framing.

    What remains, in order:

    1. **DONE 2026-09-02 (finding 89): both serialisation points fixed,
       -1.40 ms/q (-1.36%), relations unchanged.** The `ev[1]` transform sync
       existed only to read a timer, so the transform end moved to a dedicated
       `ev[4]` read after the slab loop. That exposed a second block behind it
       -- four synchronous H2D `cudaMemcpy` calls that, once the GPU was no
       longer drained, waited on the other side's in-flight transform -- now
       `cudaMemcpyAsync` from the already-pinned staging buffers. Measured
       interleaved, four paired reps, every pair favouring the fix.
    2. **Attribute the remaining `unaccounted`, still 6.67 ms/q (~6.6% of
       wall).** This is the residue of the 08-27 regression; the fixes above
       do not touch it. **Nsight will NOT answer this** -- it ranks gaps
       reliably (that is how step 1 was found and confirmed) but inflates
       device time ~46%, more than the quantity being measured, so its
       "2.06 ms/q idle" does not transfer to an unprofiled run.
       **The instrument that would work:** bracket the WHOLE per-q GPU
       sequence with two events and compare that span against the sum of the
       stage device times; the difference is true idle, with no profiler in
       the loop. That is measurement scaffolding in the production path --
       cheap, but it needs the owner's sign-off rather than being bundled with
       functional changes.
    3. Only then chase the 08-27 trigger, if the attribution points at
       something a package could plausibly have changed.
    4. If the attribution lands on per-launch overhead, the standing candidate
       is **CUDA graphs** (item 4.2): capture the fixed per-q sequence once and
       replay it, attacking launch count and latency directly. The other
       candidate that used to sit here -- the mid-sequence transform sync --
       was removed by finding 89 and is no longer available as a lever.

    **The separate inefficiency found while chasing this is now FIXED**
    (finding 89, above). `556a631` had split the per-q chain so that
    `pipe_side_prepare_q` blocked on the transform purely to read a timer,
    where the pre-slab code queued transform -> fill -> apply asynchronously
    with one sync at `ev[3]`. It was never finding 88's regression -- commits
    predating the split measure just as bad -- but it was ours and it is gone.

