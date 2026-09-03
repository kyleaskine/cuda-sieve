# Path 1 — bucket-fill microbenchmark results

> **This is a lab notebook, in discovery order, including refuted findings.**
> For what the siever does *today* — architecture, validated jobs and cards,
> measured vs projected, open experiments — read **`STATUS.md`** instead.

**Hardware:** RTX 5070 (sm_120, 48 SM, 48 MB L2, 99 KB opt-in smem, 672.0 GB/s)
**Toolchain:** CUDA 13.2.78; sm_120 + sm_89 + sm_86 native, compute_80 PTX (finding 49)
**Also measured:** RTX 3090 (sm_86), A100 80GB PCIe (sm_80) — finding 50
**Date:** 2026-08-01
**Input:** `oracle/input.job.afb.0` — the real C183 algebraic factor base, no synthetic data
**Config unless stated:** I15e (`logI=15`, `J=16384`, A=5.369e8), region 2^15, q=120000011,
FB truncated at q (GGNFS convention) → 6,843,511 entries, **312,265,384 records/special-q**

## Architecture scope

The project's **primary goal is GPU-resident relation collection**, not a
permanent GPU-sieve + CPU-cofactor split. These results first isolate the sieve
because it is the largest regular stage. A completed primary path must also put
survivor intersection/compaction, primitive filtering, resieve/factor recovery,
trial division, and cofactorization on the GPU wherever practical, so sustained
host compute does not scale roughly one-for-one with GPU count.

A hybrid that leaves GGNFS/CADO factor recovery and cofactoring on the CPU is a
useful secondary deployment option, especially for the measured 9800X3D + RTX
5070. Any hybrid throughput or energy number below is labelled as a projection
and does not set the primary roadmap.

## Correctness

The Franke-Kleinjung reduction and walk are ported from CADO
(`las-plattice.hpp`, `las-reduce-plattice-simplistic.hpp`) and shared
`__host__ __device__` between `verify_cpu.c` and the kernels.

| gate | result |
|---|---|
| FK walk vs brute-force enumeration (logI=8, J=128, 24 primes × 5 roots) | **exact** |
| GPU records landed vs single-threaded CPU reference | **1,690,293 = 1,690,293** |
| Bucket imbalance (max/mean) | **1.03×** — the doc predicted "mild"; confirmed |

## Headline

| stage | ms / special-q |
|---|---:|
| transform + plattice (T) | **1.78** |
| fill, single-level 4 B (best) | **12.09** |
| **total, algebraic side** | **13.87** |
| *(both sides ≈ 2×, extrapolated)* | *≈ 28* |

Against the CPU lines from the GGNFS breakdown. **Superseded 2026-08-03**: at
the *measured* `N_eff = 10.24` (finding 43) these become **~225 ms** to tie the
box on sieve work and **~70 ms** for the optional hybrid's retained CPU TD
stage. The 182/56
figures below assumed `N_eff ≈ 13` and are too generous to the CPU.
Apply, small-prime sieve, norm init and the threshold scan are **not yet measured**.

## Finding 1 — two-level fan-out loses, and the reason invalidates its premise

| variant | ms | L2 write sectors | sectors/record | DRAM % | occupancy | barrier stall |
|---|---:|---:|---:|---:|---:|---:|
| single-level atomic, 16K-way | **14.15** | 280,286,706 | 0.898 (**7.2× amplification**) | 17.7% | 85.7% | 0% |
| two-level L1 (128-way, staged) | 12.48 | 51,648,069 | 0.165 (1.3×) | 17.6% | 63.3% | 34.7% |
| two-level L2 (128-way, staged) | 25.38 | 51,025,495 | — | 17.1% | 64.9% | 35.3% |
| **two-level total** | **37.86** | | | | | |

Two-level does exactly what the design doc predicted at the transaction level:
it converts a 7.2× write amplification into 1.3×, a **5.5× reduction in L2 write
sectors**. And it still loses by 2.7×.

**All three kernels sit at the same ~17–18% of DRAM peak.** They are not
differentiated by efficiency — only by how many bytes they must move.
Single-level writes its output once (1.36 GB). Two-level writes it, reads it
back, and writes it again (3.75 GB). The time ratio 37.86/14.15 = 2.68×
is almost exactly the traffic ratio 3.75/1.36 = 2.76×.

**Why the premise fails on this card.** The doc argued: *"A single-level split
into ~16K buckets means ~16K simultaneously-open write streams; you cannot hold
16K partially-filled cache lines, so every record becomes a partial-line write."*
The first half is right — the amplification is real and measured. The second
half does not follow **because 16384 buckets × 128 B = 2 MB of open cache lines,
and this card has 48 MB of L2.** The L2 *is* the write-combining buffer. The
amplification happens between SM and L2 and is absorbed before DRAM, which is
why 280M write sectors and 51M write sectors produce identical DRAM throughput.

Two-level is spending 2.76× the DRAM traffic to solve a problem the L2 already
solves for free. Note per-pass it is genuinely better — L1 alone (12.48 ms)
beats single-level (14.15 ms) — but the second pass costs more than the win.

**This does not generalize.** It is a property of a 48 MB L2 against a 16K-way
fan-out. At I16e with 4× the regions, or on a card with a small L2, the open-line
footprint grows and the conclusion may flip. The rule to carry forward is
*compare open-line footprint to L2 size*, not "two-level wins."

## Finding 2 — record size: 4 B is optimal, and 2 B is worse

| record | ms | vs 4 B |
|---|---:|---|
| 2 B | 16.95 | **+40%** |
| **4 B** | **12.09** | — |
| 8 B | 14.09 | +17% |

The doc calls record size *"the single largest lever… bigger than bandwidth
efficiency"* and schedules 2 B as the measured upgrade. **It is a regression.**

8 B > 4 B is the expected byte-bound behaviour. 2 B > 4 B is not: the fill issues
the same 312M store instructions regardless of width, and a scattered 16-bit
store occupies a full sector transaction just as a 32-bit one does. Below 4 B you
stop saving bytes and only lose store efficiency. 4 B is the minimum record that
still uses a full-width store — which is also, not coincidentally, CADO's
`shorthint` size.

## Finding 3 — run-aggregation works, and is worth 2.3×

The doc's named lever: *"each (p,r) walk generates positions in monotone
increasing order… accumulate a run and emit it as one coalesced burst with a
single slot reservation."*

| L1 variant | ms | barrier stall |
|---|---:|---:|
| one record per thread per barrier | 26.00 | 30.4% |
| **runs of up to 8, one reservation each** | **11.19** | 34.7% |

**2.3×.** The first version was not memory-bound at all — it was barrier-bound at
7.5% of DRAM peak, which is why the profiler mattered more than the wall clock
here. Barrier stalls are still 35%, so more remains.

## Finding 4 — the root transform is a non-issue on GPU

**1.78 ms/special-q** for 6.84M modular inverses plus 6.84M FK reductions.

The CPU spends 373 ms/q on the same work (GGNFS "Sieve-Change", 12% of its wall
time). That is a **~200× speedup**, the largest single-stage win measured, and it
demotes the doc's "second per-q cost pillar" to 13% of the GPU sieve chain.

It is also *unoptimized*: `pl_transform` uses two 64-bit `%` operations and a
binary extended-Euclid inverse. The doc's recommended Hensel/REDC path should cut
it further. Not worth doing — it is already noise next to fill.

---

# Path 2 — apply kernel

**Date:** 2026-08-01. Same card, same factor base, same special-q.

> **Note on the record count.** Path 2 needed real norms, which forced the
> q-lattice reduction to become skew-aware (Finding 10). That changes the
> transformed roots, so the record count moved from 312,265,384 to
> **312,170,605** — 0.03%. Both are exact against their own CPU reference, and
> no Path 1 conclusion depends on the difference; fill volume is a property of
> the primes, and the transformed roots are uniform mod p either way. Path 1's
> timings above were not re-run.

The apply kernel owns one bucket region for its whole life: it initialises the
cells to the log-norm bound *in shared memory*, accumulates every log p that
landed in the region, scans for survivors, and writes back only the survivors.
The region never touches global memory in either direction. Apply's entire DRAM
footprint is the bucket read plus ~5 MB of survivors.

## Correctness

| gate | result |
|---|---|
| GPU records landed vs CPU reference, full I15e | **312,170,605 = 312,170,605** |
| Region 16384 (9,483 records) replayed on CPU: cells differing | **0** |
| Region 256 at I12/J512 (3,286 records): cells differing | **0** |

The cell gate covers the norm init, the shared-memory atomics, the log p
lookup and the threshold test together, against an independent CPU replay that
shares no device code.

## Headline

Best configuration: **region 2^14, 4 B records, 16-bit cells, 256 fill threads,
512 apply threads.**

| stage | ms / special-q |
|---|---:|
| transform + plattice (T) | 1.80 |
| fill, single-level 4 B | 12.30 |
| **apply (init + accumulate + scan)** | **3.08** |
| **total, algebraic side** | **17.4** |
| *(both sides ≈ 2×, extrapolated)* | *≈ 35* |

Against **182 ms** to tie the box and **56 ms** to reach the trial-division
*(both superseded — measured N_eff gives ~225 / ~71 ms, finding 43, and the
~71 ms is a hybrid's retained stage, not a floor for the GPU-resident target)*
floor. Still missing: the small-prime sieve (p < 2^15) and the rational side.

## Finding 5 — occupancy is the whole ballgame, and the ceiling table was right

| region | apply threads | smem | achieved occupancy | apply ms |
|---|---|---:|---:|---:|
| 2^15 | 512 | 64 KB | **32.4%** | 7.32 |
| 2^15 | 1024 | 64 KB | **64.2%** | 5.17 |
| **2^14** | **512** | **32 KB** | **95.7%** | **3.08** |
| 2^13 | 256 | 16 KB | — | 3.26 |

The doc's occupancy-ceiling table said a 64 KB region gets 1 block/SM, a block
caps at 1024 threads, the SM holds 1536, so 66.7% is a structural ceiling. The
measured numbers are 32.4% (512 threads × 1 block), 64.2% (1024 × 1) and 95.7%
(512 × 3). **A 2.4× swing in apply time from a table that could have been
written down before any code existed.**

Region 2^14 costs **+0.8 ms** in fill (more buckets) and saves **4.2 ms** in
apply. Take the trade. Region 2^13 gives back nothing in apply and costs 5.5 ms
in fill (65536 buckets), so 2^14 is the optimum, not a corner.

## Finding 6 — the safe cell scheme is also the fast one

| cells | region | apply ms |
|---|---|---:|
| **16-bit (safe)** | 2^14 | **3.077** |
| 8-bit (unsafe) | 2^15 | 3.047 |

Byte cells carry a real bug — accumulated logs do exceed 255 and `atomicAdd`
carries into the neighbouring position, which is why the doc rejected them.
Their only advantage is halving the shared memory a region needs. **That
advantage is worth 1%,** because shared memory is recoverable for free by
halving the region instead, and the region size is ours to choose. There is no
tradeoff here to agonise over.

Two related worries also priced out at nothing:

- **Shared-memory `atomicAdd` vs a racy plain `+=`: 3.08 vs 3.00 ms (~1%).**
  The "byte-atomic wrinkle" section spends a page resolving a problem that
  costs nothing to resolve correctly.
- **The sign convention works.** Atomics are add-only and a 16-bit half-word
  subtract would borrow into its neighbour, so cells start at `CINIT − T(x)`
  and *add*, with survivor ⟺ cell ≥ `CINIT`. Identical test to "start at the
  norm and subtract", no borrow, no CAS. Verified byte-exact above.

## Finding 7 — bank conflicts land inside the predicted band, and don't matter

| | shared wavefronts | bank conflicts | replay |
|---|---:|---:|---:|
| whole apply kernel | 66,515,509 | 25,831,468 | **1.63×** |
| init + scan alone (empty FB) | 17,486,146 | 20,839 | 1.001× |
| scatter (by difference) | 49,029,363 | 25,810,629 | ~2.5× |

The doc predicted *"roughly 1.5–2× on 32 banks. Acceptable, but confirm."*
Confirmed at **1.63×**. Init and scan are conflict-free by construction
(consecutive threads touch consecutive words); every conflict belongs to the
scatter, which makes two shared accesses per record — the log p lookup and the
atomic — for ~2.5× replay on its own.

It does not matter, because of the next finding.

## Finding 8 — apply is DRAM-bound at 58% of peak, a different regime from fill

| kernel | DRAM % of peak |
|---|---:|
| fill (all variants) | 17–18% |
| **apply** | **58.3%** |

Apply reads 1.25 GB of bucket records (312M × 4 B) in 3.08 ms = **406 GB/s**.
The floor — that same read at 100% of peak — is **1.86 ms**, so about 1.2 ms of
headroom remains and *no amount of shared-memory work can reach it*. Apply is
already within 1.7× of the only irreducible thing it does.

This also settles the record-size question from the other direction. A 2 B
record would halve apply's DRAM stream, but 2 B fill measured 40% slower
(Finding 2), and fill is 4× the cost of apply. 4 B stays.

## Finding 9 — norm init is cheaper than budgeted, and survivor emit is noise

| component | ms | how measured |
|---|---:|---|
| norm init, full fp32 degree-5 Horner + `__log2f` | **1.76** | `--norm horner` minus `--norm const` |
| accumulate + scan | ~5.2 | remainder at region 2^15 |
| survivor emit, 1.30M survivors | **0.41** | 1.30M vs 406 survivors |

The doc budgeted ~5 ms for norm init; the real per-position evaluation costs
**1.76 ms**. No piecewise approximation is needed on GPU — CADO's whole
piecewise-linear norm machinery exists to avoid a cost this card doesn't have.

## Finding 10 — the q-lattice must be reduced under the *skewed* norm

Not a performance result; a correctness one that Path 3 would have hit later
and harder. An unskewed Gauss reduction gives |a| ~ |b| ~ √q. The homogeneous
terms `c_k a^k b^(5−k)` then span **10³⁹**, `log2|F|` is set by `c₀b⁵` alone,
and the norms are simply wrong — the leading coefficient's term underflows fp32
entirely.

Reducing under `|(a,b)|² = (a/√s)² + (b√s)²` gives A = 2.33e12, B = 1.64e4 and
normalised coefficients `-0.0136, -0.0605, 1, -0.419, -0.0054, 0.381` — all
O(1). That balance is the entire purpose of the skew parameter, and it is also
what makes fp32 norm evaluation possible at all: without it there is no
scaling that keeps both the coefficients and the powers in float range.

---

# Small-prime sieve, rational side, and the CADO oracle

**Date:** 2026-08-01. `q = 120000053`, `rho = 112625526` — a **real** special-q
taken from las, not the synthetic one used above. Region 2^14, 4 B records,
16-bit cells, 128 fill threads, 512 apply threads.

## The whole sieve chain, both sides

| stage | side 1 (algebraic) | side 0 (rational) | both |
|---|---:|---:|---:|
| transform + plattice | 1.75 | 1.01 | 2.76 |
| bucket fill | 12.60 | 11.89 | 24.49 |
| apply — norm init + small sieve + bucket apply + scan | 11.95 | 7.91 | 19.86 |
| **total ms / special-q** | **26.30** | **20.81** | **47.11** |

**47.1 ms for the complete sieve, both sides**, against the then-used **182 ms**
CPU sieve-work line and **56 ms** optional-hybrid retained-TD line.

Correctness at full I15e, both sides: records landed equal the CPU reference
exactly (312,211,826 and 295,181,761), and a full region replayed independently
on the CPU gives **0 cells differing** on each side.

## Finding 11 — the small-prime sieve is 5× its budget, and it is now 28% of the chain

The doc budgeted *"small-prime sieve (~1.5e9 shared-mem updates) | ~1–3 ms"*.

| | side 1 | side 0 | both |
|---|---:|---:|---:|
| entries below `bkthresh` = 2^15 | 3,631 | 3,512 | 7,143 |
| updates | 2.957e9 | 1.398e9 | **4.36e9** |
| cost (apply with − without) | 8.77 ms | 4.40 ms | **13.17 ms** |
| rate | 337 G/s | 318 G/s | — |

**2.9× the updates and ~5× the time.** The update count is the surprise: those
7,143 entries are 0.1% of the factor base and produce **7.2× the entire
bucket-sieve volume**. On side 1, 84% of it comes from the 52 entries with
p < 64.

That skew is the whole engineering problem — a thread-per-prime loop would put
8,192 serial updates on whichever thread drew p = 2 while its neighbours did
one. Three tiers sized to each entry's hit count:

| tier | side-1 entries | threads per entry |
|---|---:|---|
| p < 64 | 52 (84% of updates) | the whole block |
| 64 ≤ p < 1024 | 165 | one warp |
| p ≥ 1024 (≤16 hits) | 3,414 | one thread |

At ~330 G updates/s against a card ceiling near 3.8 T conflict-free shared
atomics/s, this is ~9% of peak and there is real headroom left.

**Superseded by prime powers (finding 15).** At the time this was written fill
was the larger half (24.5 ms of 47.1) and the small sieve was not the place to
spend effort. Powers took the small sieve from 4.36e9 to 6.84e9 updates, and at
55.5 ms **apply (27.45) has overtaken fill (24.77) and the small-prime sieve is
now the largest single component of the chain.** If sieve milliseconds are ever
worth chasing again, this is where they are.

**Region choice is what makes this cheap.** With `log_region ≤ logI` a region
lies inside one j-row, so the entry point is one multiply and one remainder —
hits within a row are just the progression `i ≡ rt*j (mod p)`. No walk state
crosses regions, so every block is independent. CADO carries per-prime
positions forward between regions because it processes them in sequence; we
cannot, and do not need to.

## Finding 12 — the rational side is cheaper than the algebraic side

Side 0 costs **20.8 ms** against side 1's 26.3 ms, despite comparable bucket
volume (295.2M records vs 312.2M). The difference is entirely in the parts that
scale with the factor base and the polynomial: the transform is degree 1 and
every prime has exactly one root (1.01 vs 1.75 ms), and the small sieve carries
half the updates (1.40e9 vs 2.96e9).

Nothing on disk holds a rational factor base — GGNFS computes it on the fly and
CADO rebuilds it every run — so `rfb.c` builds it: sieve to `rlim`, then
`r = -Y0*Y1^-1 (mod p)`, with `p | Y1` as the projective case. Y0 and Y1 are 118
and 76 bits and are reduced by limb-wise Horner in base 2^32; **doubles are fine
for the norm (it only needs a logarithm) but give wrong roots.** Verified
against exact arithmetic: prime counts exact (1229 / 9592 / 78498 at 10^4 / 10^5
/ 10^6) and every root satisfies `Y1*r + Y0 ≡ 0 (mod p)`.

## Finding 13 — gated against CADO itself

`makefb` + `las -v -dumpfile` on the same polynomial. Full capture and
constants in `oracle/PARITY.md`.

| gate | las | ours | |
|---|---|---|---|
| q-lattice basis | `a0=7374527 b0=-1 a1=120000053 b1=0` | `(-7374527, 1, 120000053, 0)` | **same basis, first vector negated** |
| log2(maxnorm) side 0 | 131.86 | **131.86** | exact |
| log2(maxnorm) side 1 | 196.61 | 196.41 | 0.2 bits |
| one-sided survivors, side 0 | 23,952,829 | 30,043,786 | +25% |
| one-sided survivors, side 1 | 21,650,256 | 18,174,114 | −16% |

The survivor rates are close but not equal, and shouldn't be yet: we are missing
prime powers (CADO's small part has 3,839 ideals to our 3,631 — the gap is
exactly powers), and we run at scale 1.0 where las uses 1.28/1.93.

**We do not need las's `scale`.** It exists only so `scale × log2(maxnorm)` fits
in a byte — `1.28 × 196.61 = 251.7`. With 16-bit cells we can run at scale 1.0
or higher for free. las is discarding 0.39 bits of resolution per position that
we keep, which means fewer false survivors reaching cofactoring.

`-dumpfile` needed a one-line CADO patch to work at all: revision `0574bc39d`
has `ASSERT_ALWAYS(!las.dump_filename)` immediately before the only
`dumpfile.open()` call site in the tree. Original file and revert instructions
are in `oracle/`.

## Three bugs the oracle and the second side exposed

1. **`pl_transform` had CADO's field names on a differently-grouped struct.**
   las groups the basis by coordinate (`a = a0*i + a1*j`); `qlat_t` groups by
   vector (`a = i*a0 + j*b0`). Copying CADO's formula across silently swaps
   `a1` and `b0`, which **transposes the basis**. The result is a perfectly
   valid lattice of the right density — record counts, walk checks and every
   GPU/CPU cross-check still agree, because both sides used the same wrong
   lattice. It is simply not the lattice the norms describe. Nothing but a
   comparison against las would have caught it. `verify_transform()` now gates
   the transform against its definition (`a ≡ r*b mod p`) rather than against
   itself.

2. **The two-level fill livelocks past 16,384 regions.** Both levels stage into
   128 fixed shared buffers, so the split only fits when `nsuper ≤ 128` *and*
   `regions_per_super ≤ 128` — exactly I15e at region 2^15 and no further. At
   region 2^14 it indexed `cnt[]` out of bounds and spun forever in the retry
   loop. Path 1's two-level numbers were all taken at region 2^15, sitting
   exactly on that limit, so **Finding 1 stands**. But it sharpens the verdict:
   two-level cannot even express the operating point that won, without adding a
   third level. It now refuses the configuration instead of hanging.

3. **`log2(q)` was being divided out of both sides' norms.** Only the special-q
   side's norm carries a factor of q. On side 0 this made the threshold ~27 bits
   too generous — 171.7M survivors instead of 30.0M. Caught by comparing against
   las's per-side `log2(maxnorm)`, which is printed *after* the division on the
   sq side and *before* it on the other.

---

# Prime powers, las's log scale, and the attempt at a byte-diff

**Date:** 2026-08-01. Same q. The sieve now runs on **CADO's own factor base**
(`fb_cado.c` parses makefb's text format) at **las's own byte scale**, so both
the ideal set and the log units are las's rather than approximations of them.

## The chain with the correct factor base

| stage | side 1 (scale 1.28) | side 0 (scale 1.93) | both |
|---|---:|---:|---:|
| transform + plattice | 2.15 | 1.14 | 3.29 |
| bucket fill | 12.85 | 11.92 | 24.77 |
| apply (norm + small sieve + bucket apply + scan) | 17.82 | 9.63 | 27.45 |
| **ms / special-q** | **32.82** | **22.69** | **55.51** |

Up from 47.1 ms, entirely because prime powers add small-sieve work: side 1
goes 2.96e9 → **5.03e9** updates, side 0 1.40e9 → **1.81e9**. **6.84e9
small-sieve updates per special-q**, now half the chain. That puts the complete
two-sided sieve at **55.5 ms against the then-used 56 ms hybrid retained-TD
stage**.

Correctness holds: full I15e, both sides, a region replayed independently on
the CPU gives **0 cells differing**.

## Finding 14 — factor base and threshold now match las exactly

| | las | ours |
|---|---:|---:|
| side 1 small part (p < 2^15) | 3,839 ideals | **3,839** |
| side 1 bucketed | 7,601,777 ideals | **7,601,776** |
| side 0 small part | 3,589 | 3,586 |
| side 0 total | 3,958,485 | 3,957,371 |
| side 1 survivor bound | 143 | **143** |
| side 0 survivor bound | 141 | **141** |

The bounds fall out of `round(scale * lambda * lpb)` with no fitting:
`1.28 × 3.5 × 32 = 143.4` and `1.93 × 2.35 × 31 = 140.6`.

Two things had to be right to get here. **`fb_log_delta`, not `log(q)`**: a
prime power's log increment is `fb_log(p^nexp) − fb_log(p^oldexp)`, the
*marginal* cost of the extra valuation, so that powers telescope with their
base prime. And **the general prime-power root transform** (below), without
which the powers cannot be walked at all.

## Finding 15 — prime powers break two assumptions the prime-only code made

Both were silent until CADO's factor base supplied moduli that GGNFS's
`.afb.0` never does.

1. **`pl_invmod` requires an odd modulus.** It is binary extended Euclid, whose
   halving step `(x+p)>>1` is only exact for odd `p`. CADO's factor base
   carries the ladder 2, 4, 8, …, 32768. Feeding one to `pl_invmod` does not
   return a wrong answer — **it never terminates**. Powers of two now go to a
   2-adic Newton iteration (`pl_invmod_any`).

2. **With a prime power the denominator can be non-zero and still have no
   inverse.** Modulo a prime, `D = a0 − r·a1` is either invertible or zero;
   modulo `p^k` it can be divisible by `p` but not by `p^k`, and binary Euclid
   spins forever on `gcd > 1`. This hung on the first `q = 49` in the file.

   The fix generalises and *subsumes* the old whole-rows special case. With
   `g = gcd(D, q)` (D and N cannot both be divisible by p, or p would divide
   the determinant, which is the special-q):

   > solutions exist only when `g | j`, and then, writing `j = g·j'`,
   > `i ≡ rt·j' (mod m)` with `m = q/g` and `rt = N·(D/g)^-1 (mod m)`.

   `g = 1` is the ordinary affine case; `g = q` gives `m = 1`, meaning "every
   `i`, on every `q`-th row" — exactly the old `PL_ROWS`, no longer special.
   The small sieve now also tiers on `m` rather than `q`, because an entry with
   `q = 32768, g = 32768` has `m = 1` and hits every position in its rows;
   leaving it in the thread-per-entry tier would hand one thread a whole region.

## Finding 16 — the sieve is validated against the factor base analytically

This turned out to be a stronger gate than the byte-diff, and it needs no
oracle file. Expected sieved log per position is `Σ logp(ideal) / q(ideal)`
over the whole factor base — computable directly from `c183.fb1`:

| | log units / position |
|---|---:|
| analytic, whole factor base | **54.65** |
| **measured, ours** | **54.2** |
| analytic, small part only | 39.31 |
| measured, ours with the bucketed part removed | 40.4 |

**0.8% agreement.** That exercises the factor-base parse, the power handling,
the general root transform, the small sieve, the bucket fill and apply, and the
`fb_log_delta` scaling, in one number, against a quantity derived independently
from CADO's own file.

## Finding 17 — the `-dumpfile` oracle is not usable as captured

The byte-diff does **not** pass, and the evidence says the problem is the dump,
not the sieve.

| | mean sieved log / position |
|---|---:|
| analytic, whole factor base | 54.65 |
| ours, full | 54.2 |
| analytic, small part only | 39.31 |
| **las's dump** | **41.1** |

las's dump carries roughly the small-sieve contribution and **not** the
bucket-sieved one, even though `las-process-bucket-region.cpp` calls
`apply_buckets()` before `SminusS()` and the dump.

Worse, it does not line up positionally at all:

| | correlation |
|---|---:|
| ours (small-only) vs las, direct | +0.033 |
| ours (small-only) vs las, i-mirrored | +0.016 |
| *control:* ours (small-only) vs ours (full), same positions | **+0.636** |

The control shows the measurement works. Both orientations were checked because
our basis is las's with the first vector negated, which mirrors the i-axis.
Neither correlates.

This is consistent with how the flag was found: `-dumpfile` sat behind an
`ASSERT_ALWAYS(!las.dump_filename)` placed immediately before the only
`dumpfile.open()` call site in the tree, i.e. it has been dead code for long
enough to rot. **Do not treat these two 512 MB files as ground truth.** Options,
in order of expected value: `las_tracek`/`TRACE_K`, which follows one `(i,j)`
through las's pipeline and would localise this in one run; or `las -v`
checksums; or reporting the bug upstream.

Everything else from `oracle/PARITY.md` — the basis, the scales, the bounds,
the factor-base composition — came from the `-v` log and remains sound.

# Review round: two placement bugs, and the gate that would have caught them

*2026-08-02. Two external reviews (fable's, in `prototype.md`; codex's, quoted
in the session log). Everything below is measured or verified on the CPU — the
GPU was busy, so no timing in this section is new.*

## Finding 18 — projective roots with a nonzero reciprocal were placed wrong

**The bug.** Both CADO and GGNFS store a root of `q` in `[0, 2q)`: below `q` it
is affine (`a ≡ r·b mod q`), at or above it is projective with reciprocal
`rr = r − q`, meaning `a·rr ≡ b (mod q)`. For a **prime** `q` the only
projective root is `rr = 0`, the classical `b ≡ 0 (mod q)` — so code that
assumes `rr = 0` is correct on a prime-only factor base and wrong the moment
prime powers arrive. `bench_kernels.cu` passed `0` where the reciprocal
belonged, and `plattice.cuh` only implemented the `rr = 0` case.

`c183.fb1` has **35 such entries** — the projective ladder above the leading
coefficient `110880 = 2^5·3^2·5·7·11`, opening with `4:4,3: 6` (q=4, rr=2).
Between them:

| | |
|---|---:|
| updates per special-q | **469,482,034** |
| scaled log units per cell (scale 1.28) | **1.538** |

**Why every existing gate was blind to it.** The wrong lattice has the *same
density* as the right one — both are index-`q` sublattices. Measured directly
for q=4, rr=2 over a small box: 32 true hits, 32 predicted hits, **16 in
common**. So:

- `Σ logp/q` (finding 16) checks density. Same density → passes.
- the CPU replay (`verify_apply_region`) shares `plattice.cuh` with the GPU, so
  it made the identical error → passes.
- record counts vs the CPU reference → same count → passes.
- `verify_transform` as it stood checked only that each *predicted* hit
  satisfies the congruence, one direction, and it never exercised a nonzero
  reciprocal at all → passes.

This is the same failure class as the transposed-basis bug: **right volume,
wrong placement.** It is the third one this project has hit, which is the real
lesson — volume gates cannot catch placement bugs, and this codebase kept
building volume gates.

**The fix.** `pl_transform_gen` now takes the reciprocal and uses
`D = rr·a0 − a1, N = b1 − rr·b0` for the projective case (at `rr = 0` that is
the negation of the old `(a1, −b1)`, same ratio — a generalisation, not a
replacement). `pl_transform_enc` is the single place the encoding is decoded,
and every consumer now goes through it.

## Finding 19 — bucketed projective entries were dropped, on a wrong argument

`fb_restrict` discarded every entry with `root ≥ p`. Side 0 has two:
**38321** and **5746453**, both factors of
`Y1 = 59·101·127·281·1259·38321·5746453`.

fable's review recommended *keeping* the drop, reasoning that a bucketed
`p > J = 16384` means a projective ideal can only hit row `j = 0`. **That
argument is wrong, and the way it is wrong is worth recording**: the projective
condition constrains `b`, not `j`, and `b = i·a1 + j·b1`. For a special-q of
this size the reduced basis has b-components of order `sqrt(q/skew) ≈ 1`, and
on this lattice it is exactly

```
(a0,a1,b0,b1) = (-7374527, 1, 120000053, 0)   ->   b = i
```

so the condition is `i ≡ 0 (mod p)` — one hit per row, on **every** row.
**16,384 positions per entry, not one row's worth.** `fbtest` prints the
`b(i,j)` form at startup for exactly this reason.

(A consequence worth its own line: `b = i` means the entire `i = 0` column has
`b = 0` and can never yield a relation. That is a property of this q-lattice,
not a bug, but it caps what these two entries are worth in practice.)

`fb_restrict` now keeps them and the transform handles the encoding. What the
bucket walk still cannot express is `g > 1` — hits confined to every `g`-th row
— so `k_transform` emits an empty walk for those and **accumulates the number
of positions dropped**, which is printed. At the default `bkthresh = I ≥ J` it
is exactly zero.

## Finding 20 — the projective power ladder does lift

`rfb.c` broke out of the power loop at `p | Y1` with the comment "no lift".
Projective roots *do* lift: the point `(Y0 : −Y1)` of `P^1` normalises to
reciprocal `rr = −Y1/Y0 (mod p^e)`, which is divisible by `p` but not `p^e`.
Restoring it adds `59^2, 101^2, 127^2` — **exactly the −3 in side 0's small-part
count against las** (3,586 vs 3,589), which fable predicted and which now
closes. Side 0 is 3,957,374 ideals, up 3.

The affine and projective ladders are now the same loop: which of `Y0`, `Y1` is
the unit mod `p` decides the encoding, and exactly one of them is, because
`gcd(Y0,Y1) = 1`.

## Finding 21 — the device transform could hang, and would have

Three latent faults in `k_transform`, all of which CADO's factor base reaches
and GGNFS's does not, all now removed by routing through `pl_transform_enc`:

| | consequence |
|---|---|
| `q = 2^15` sits exactly at the default `bkthresh`, so an **even** modulus reaches the device | `pl_invmod` is binary Euclid; it needs an odd modulus. Silent wrong root. |
| projective entries reaching the affine formula | `r ≡ 0` gives a bogus affine root. Silent. |
| raising `--maxbits` puts odd prime powers in the bucket range | non-invertible denominator → binary Euclid **spins forever on the device**. |

Two silent wrong answers and one hang — none of which any gate would have
reported as a failure.

`pl_transform_gen` has a genuine precondition, now documented and enforced:
**`q` must be a prime power.** The step "solutions exist only when `g | j`"
needs `gcd(N,g) = 1`, which follows from "`p` divides at most one of `D`, `N`"
and holds only for a single prime. For `q = 200` (`D` even, `N` divisible by 5)
the solution set is a CRT combination and this form is wrong — the new
brute-force gate found this immediately, which is how it came to be documented.
`fb_check_prime_powers` keeps the assumption true for any factor base loaded,
including hand-edited or third-party ones.

## Finding 22 — the gates now check placement, not volume

New CPU-only binary `fbtest` (`make check`), 6 seconds, **no GPU** — so it runs
on a busy box and on machines without the card:

| gate | what it catches |
|---|---|
| transform vs definition by **set equality**, every prime power `q ≤ 200` and every root in `[0,2q)` | placement. Both directions, so a proper sublattice fails too. Covers primes, powers, even moduli, affine, projective with zero and nonzero reciprocal. |
| the same, driven by the **real** factor bases | a loader that mangles the encoding, even where the algebra is right |
| every modulus is a prime power | the transform's precondition |
| `Σ logp/q` per side | parse, powers, log deltas, scale — one number |

The first gate is the one that matters. The old `verify_transform` compared
predicted hits against the congruence in **one direction only**; a transform
naming a proper sublattice — right congruence, half the hits — passed it. Set
equality against the definition, with `(a,b)` computed from `(i,j)` and no
lattice algebra involved, is what closes that. Both of this session's bugs fail
it loudly.

`verify_count_updates` also gained an optional per-region count array, so the
fill can be gated on **placement across all 32K regions** rather than on a
global total (fable's R4).

Result:

```
PASS  synthetic moduli              primes, prime powers, even moduli, affine + projective
PASS  side 1 small part             3839 entries checked
PASS  side 1 moduli are prime powers 7605616 ideals
PASS  side 1 projective entries seen 41 projective, 35 with a NONZERO reciprocal
PASS  side 0 small part             3589 entries checked
PASS  side 0 moduli are prime powers 3957374 ideals
PASS  side 0 projective ladder      10 projective, 3 with a NONZERO reciprocal
```

## Finding 23 — the 16-bit cell's precision was being thrown away

`k_apply` clamped the initialised norm at **255**. las clamps there because its
cell *is* a byte — that is the entire reason `scale` exists
(`1.28 × 196.61 = 251.7`). Ours is 16 bits with `CINIT = 4096`, so the ceiling
is `CINIT`, and `scale` is a free parameter rather than a constraint. The clamp
meant we inherited las's 0.39 bits of lost resolution for nothing, and would
have silently flattened every norm above `255/scale` into one bucket the moment
anyone raised `scale`.

Now clamped at `CINIT` (16-bit cells) or 255 (8-bit), with the two real limits
validated up front and refused rather than saturated:

| limit | binds at |
|---|---|
| `scale × log2(maxnorm) ≤ CINIT` | scale ≈ 20 (side 1) |
| `scale × log2(p) ≤ 255` (the uint8 per-ideal log) | **scale ≈ 9** |

So production scales of 2, 4, 8 are all available and free on the GPU side.
Whether they reduce false survivors enough to matter for CPU cofactor work is
**unmeasured** — it needs the GPU, and it is the cheapest remaining experiment.

## Also fixed

- **CLI defaults were the losing configuration.** `--mode twolevel --region 15`
  were still the defaults after two-level lost by 2.7× and 2^15 lost to 2^14,
  so several commands in this file reproduced a path nobody would ship.
  Defaults are now `atomic` / `region 14`.
- **Two silent parameter failures now refuse.** `--region > 16` with 2 B or 4 B
  records overflows the 16-bit offset field (wrong cells, no error);
  `--region > logI` breaks the fused small sieve's one-row assumption. Both
  produced plausible-looking output.
- `qlat_build` moved to `poly.c` so the correctness gates link without CUDA.

## Finding 24 — the makefb parser read 81 as 9², and it cost log accuracy

`fb_cado.c`'s `is_power` searched the exponent **upward** from 2 and returned
the first that worked. That is k *minimal*, i.e. the *largest* base — it read
`81` as `9^2` and `729` as `27^2`, despite its own comment saying "k > 1
maximal".

The exponents in the file are relative to the base prime (`81:5,4: 114` means
3^5 over 3^4), so a wrong base inflates the log increment:

| q | read as | logp | correct | logp |
|---|---|---:|---|---:|
| 16 | 4² | 3 | **2⁴** | **1** |
| 64 | 8² | 4 | **2⁶** | **1** |
| 81 | 9² | 4 | **3⁴** | **2** |
| 729 | 27² | 6 | **3⁶** | **2** |

It hits every q whose exponent is composite. Placement was unaffected, which is
why nothing before gate 5 saw it — and note it is the mirror image of findings
18–19: those were right-value/wrong-place, this is right-place/wrong-value.
Between them they cover both ways a sieve can be wrong while looking healthy.

Fixed by trial-dividing for the least prime factor, which is exact and, with 97
long-form lines in the file, free.

## Finding 25 — las's printed `scale` is rounded, and the bound is a truncation

Two constants taken from las's `-v` log were subtly wrong, and both were
recorded in `oracle/PARITY.md` and used by every run since.

**The scale.** las prints `scale=1.28`, rounded to 2 dp. It also prints
`logbase` to 7 figures, and `logbase = 2^(1/scale)`, so:

| side | printed | exact |
|---|---:|---:|
| 0 | 1.93 | **1.925** |
| 1 | 1.28 | **1.275** |

That is enough to move `fb_log` by one unit for a band of primes: p = 25811171
gives 32 at 1.28 and **31**, which is what las applies, at 1.275. This was the
last surviving discrepancy in the gate-5 trace.

**The bound.** We used `round(scale * lambda * lpb)`. las (`las-norms.cpp:270`)
uses

```
bound = (unsigned char) (r * scale + LOGNORM_GUARD_BITS);     r = lambda * lpb
```

— a **truncating** cast plus a guard bit. With the exact scales,
`trunc(3.5*32*1.275 + 1) = 143` and `trunc(2.35*31*1.925 + 1) = 141`, both
exact. Our old formula agreed on these two only because the rounded scales
happened to compensate; it would have diverged on any other parameter set. The
agreement reported in finding 14 was therefore luckier than it looked.

## Finding 26 — GATE 5 PASSED: byte-exact sieve parity against las

`las_tracek` is a **stock CADO target** (`sieve/CMakeLists.txt:111`,
`TRACE_K=1`) — no patch needed, unlike `-dumpfile`. It follows one position
through the pipeline and prints every ideal applied to it with its log, which is
strictly more informative than a byte dump.

Use `-traceab a,b`, not `-traceij`: it is basis-independent, so the i-mirroring
never enters. It also confirmed the mirroring directly — our `(4999,8192)` is
las's `(-4999,8192)` for the same `(a,b)`.

Four positions, both sides, `q=120000053, rho=112625526`:

| (i,j) ours | side | las sum | ours sum | ideals |
|---|---|---:|---:|---:|
| (4999, 8192) | 1 | 22 | **22** | 5 |
| (4999, 8192) | 0 | 58 | **58** | 6 |
| (9237, 15022) | 1 | 94 | **94** | 15 |
| (9237, 15022) | 0 | 65 | **65** | 6 |
| (2978, 8393) | 1 | 46 | **46** | 14 |
| (2978, 8393) | 0 | 94 | **94** | 2 |
| (13198, 9151) | 1 | 51 | **51** | 13 |
| (13198, 9151) | 0 | 63 | **63** | 4 |

**8 of 8 exact — and the ideal lists agree entry by entry, not just the
totals.** Two positions were chosen because long projective power ladders hit
them (3,9,27,81,243,729,2187 and 2,4,8,16,32,64), making this the direct gate on
the nonzero-reciprocal bug rather than an aggregate one.

> **Correction, later the same day.** As first written this table came from
> `fbtest --trace`, which enumerates the factor base directly with
> `hits_def_pub` and **does not run the sieve** — not `pl_transform_enc`, not
> the walk, not the tiering, the bucket fill, the small sieve or the GPU apply.
> So it established that our *factor base, roots and logs* agree with las, which
> is what found the bugs above, but it did not establish sieve parity, and
> calling it "byte-exact sieve parity" was an overclaim. Finding 27 closes the
> gap properly; the numbers below stand as the ideal/log half of it.

**Norm initialisation still differs, and should.** las runs 0 to +2 above the
exact value, scattered: one unit is `LOGNORM_GUARD_BITS` (which las prints in
its own banner), and the rest is its norm *approximation* — it interpolates
log|F| along a row rather than evaluating at every position, computing the exact
norm only later for survivors. **Our fp32 Horner is the more accurate of the
two.** The gap is bounded and one-directional (las over-estimates, so it is
slightly more permissive), which also means a byte-for-byte region diff could
never have reached zero even with a working dumpfile. Gate 5 was the right
instrument, not merely an available one.

Reproduce:

```
cd ~/cado-nfs/build/<host> && make las_tracek
cd ~/code/cuda-sieve/oracle && las_tracek -poly c183.poly -fb1 c183.fb1 \
    -lim0 67100000 -lim1 134200000 -lpb0 31 -lpb1 32 -mfb0 60 -mfb1 92 \
    -lambda0 2.35 -lambda1 3.5 -A 29 -sqside 1 -q0 120000011 -nq 1 \
    -adjust-strategy 0 -B 14 -t 1 -traceab 946175173703,4999

cd ~/code/cuda-sieve/bench && ./fbtest --cadofb ../oracle/c183.fb1 --trace 4999,8192
```

## Finding 27 — gate 5, done properly: the *pipeline* reproduces las

Finding 26's table compared las against a direct enumeration of the factor base.
The thing we actually want to gate is the sieve. `bench --probe i,j` now reads
the cell back out of `k_apply` after the run, so the number it prints has been
through the root transform, the Franke-Kleinjung walk, the three-tier split, the
bucket fill, the small-prime line sieve and the GPU apply:

```
[gate 5] probe (i=4999, j=8192)  x=268456839  region 16385 offset 4999
         init norm S   = 241
         final cell    = 3877
         SIEVED LOG SUM = 22   <- produced by transform + walk + fill + small sieve + apply
```

Same four positions, both sides, `q=120000053, rho=112625526`:

| (i,j) | side 1 las / GPU | side 0 las / GPU |
|---|---|---|
| (4999, 8192) | 22 / **22** | 58 / **58** |
| (9237, 15022) | 94 / **94** | 65 / **65** |
| (2978, 8393) | 46 / **46** | 94 / **94** |
| (13198, 9151) | 51 / **51** | 63 / **63** |

**8 of 8, through the whole pipeline.** *This* is what closes Path 3 gate 2.

The probe costs one comparison per cell against a sentinel and is compiled in
unconditionally; it is off unless `--probe` is passed.

## Finding 28 — fp32 norms are wrong near root lines, in both directions

The normalisation gives fp32 ample dynamic *range* — that is what finding 10
bought. It does not give precision. Near a real root line of `F` the Horner
terms are O(1) and cancel to something tiny, so what survives is mostly rounding
noise. Measured against an fp64 evaluation over a band along the three real root
lines:

| | before | after |
|---|---:|---:|
| positions sampled | 63,497 | 63,497 |
| rounded-log mismatches | **144** | **1** |
| error range (sieve units) | **−3.31 .. +2.57** | −0.00 |
| worst relative error on the sum | 2.59 | 4.8e-4 |

A random control over 200,000 positions shows 1 mismatch either way, so this is
specific to the root lines and not a general precision problem.

**Both signs occur**, so it costs relations as well as wasting cofactor time —
and raising `scale` (finding 23) amplifies it in sieve units, which makes it a
blocker for that experiment rather than an independent issue.

The fix is an error bound, not more precision everywhere: run the same Horner on
`|.|` alongside the real one, and when `|acc| < 4.9e-4 * sum|terms|` recompute
that cell in fp64 with `(a,b)` formed exactly in int64. The bound costs one extra
FMA chain per cell; the fp64 path fires on well under one cell in a thousand, so
even at 1/64 double rate it is noise. The one residual mismatch is a genuine tie
at the rounding boundary.

Credit where due: this was codex's finding, and my first attempt to reproduce it
sampled too sparsely and came back with 1 mismatch instead of 144. Sampling every
row at closest approach to each root line reproduced it exactly.

## Finding 29 — factor-base preprocessing cost 6.7 s, 120x the sieve chain

Self-inflicted, on 2026-08-02: `fb_restrict` and `fb_split_small` called
`fb_is_proper_power` per entry, which runs Miller-Rabin, over 11.5M ideals.

| stage | before | after |
|---|---:|---:|
| load | 0.42 s | 0.25 s |
| `fb_split_small` | **4.63 s** | **0.005 s** |
| `fb_restrict` | **2.05 s** | **0.008 s** |

Every loader already knows which entries are powers — `fb_cado.c` literally
separates them into their own stream before merging, and `rfb.c` generates them
in a loop with the exponent in hand. `fb_t` now carries an `ispow` flag set at
load, and nothing downstream re-derives it. The independent primality check
survives where it belongs, in `fbtest`'s prime-power gate, where being
independent is the point.

## Finding 30 — the gates are now assertions, not printouts

Three things that were computed and printed but could not fail a run:

- **Per-region record counts.** `verify_count_updates` gained the array in the
  last round but `bench` passed `NULL`. It now compares all 32,768 regions
  against the GPU cursors and **returns non-zero** on any mismatch. A global
  total cannot distinguish "right total, wrong region" — and every placement bug
  this project has hit had exactly the right total. Currently: *all 32,768
  regions match the CPU reference exactly*, both sides.
- **`sum(logp/q)`.** Now a PASS/FAIL in `make check`, against expectations
  computed by **separate implementations** (a re-parse of `c183.fb1` that
  re-derives base primes independently; a fresh sieve to `rlim` plus the power
  ladder, which also reproduces the 3,957,374 ideal count exactly). Wiring it up
  immediately caught that the constant I had was the stale pre-finding-24 value:
  42.744 against the correct **42.2913**.
- **Parameter validation.** `--record-bytes` outside {2,4,8} used to fall through
  to the 8-byte kernel while allocating the requested size — an out-of-bounds
  write. Two-level mode accepted 8-byte records but launches the 4-byte level-2
  specialisation. Both now refuse, along with zero/negative `J`, `reps`,
  `threads`, and an area that does not divide evenly into regions.

## Finding 31 — the parity profile and the timed profile are not the same set

Worth stating plainly, because it limits what the gate-5 numbers cover:

| | timed `bench` run | las |
|---|---|---|
| side 1 upper bound | truncated at the special-q (GGNFS convention), **6,840,490** bucketed | runs to `alim`, **7,601,777** |
| side 0 powers | capped at `--maxbits 15` | `powlim = ULONG_MAX` |

The eight probes still agree because no ideal in `[q, alim)` happens to hit those
four positions — which the agreement itself demonstrates, since the GPU excludes
that band and would otherwise have come out low. That is evidence for these
positions, **not** a general equivalence.

Before any survivor-set comparison, pin one profile. The cheap direction is to
pin las *down* to ours (`-powlim0 32767 -powlim1 32767`, and a `q0` above `alim`
so no truncation applies) rather than chase it up; `bench --fbbound 134200000`
goes the other way but then re-includes the special-q ideal, which las divides
out.

## Finding 32 — `--verify` could exit 0 on a sieve that was demonstrably wrong

Two independent false-success paths, both of which made the verification suite
decorative rather than load-bearing.

**Cell mismatches printed but did not fail.** `verify_apply_region` counted
differing cells, printed the count, and returned normally. Demonstrated with the
deliberately racy `--apply-mode plain`:

```
[verify] region 16384: 9576 records replayed on CPU, 518 cells differ
                       (first at cell 0: gpu 3946 ref 3966)
exit=255      <- was 0
```

20 sieve-log units lost at cell 0 alone, and the old code reported success.

**Bucket overflow was invisible to the per-region gate, by construction.**
`cursor[b]` is incremented *before* the cap test, so it counts records
**attempted**, not stored — and `verify_count_updates` counts the same thing.
The two therefore agree exactly while `k_apply` truncates each bucket at `cap`
and silently drops the excess. Finding 30's per-region assertion, added
specifically to catch placement errors, cannot see this class at all. Overflow is
now failed explicitly under `--verify`, and the per-region OK line says what its
counts do and do not cover.

The general lesson is the one this project keeps re-learning: a gate that cannot
fail is not a gate. Findings 30 and 32 are the same mistake in two forms —
computing the right comparison and then not acting on it.

## Finding 33 — the CPU reference was not evaluating the kernel's expression

`k_apply` used `__log2f`, the fast intrinsic (~2 ulp, no host equivalent);
`norm_target_host` kept full `double` through `log2` with a `1e-300` clamp
against the device's `1e-30f`. So "0 cells differ" was comparing two slightly
different functions, and an emulation over 127k root-line samples found three
rounded disagreements between the two expressions.

The device now uses accurate `log2f` and the host mirrors it float-for-float,
including the clamp. Norm init was 1.76 ms of a 27 ms apply, so the intrinsic was
not worth the loss of a meaningful reference. **The reference must be the same
function, not a better one** — being more accurate on the host would make the
comparison meaningless in exactly the same way.

Re-measured after the change: still 1 residual mismatch in the 63,497-position
root-line band (the rounding tie from finding 28), and region 16384 still gives
0 cells differ — but now that number is a statement about the device path.

## Finding 34 — the default invocation never got finding 29's fix

`fb_load` (GGNFS `.afb.0`, the **default** `--fb` path) left `ispow` NULL, so
`FB_ISPOW` fell back to a primality test per entry — the exact 7.8 s that
finding 29 removed, still present on the one path not exercised while fixing it.

| stage | before | after |
|---|---:|---:|
| `fb_split_small` | 5.39 s | **0.069 s** |
| `fb_restrict` | 2.39 s | **0.024 s** |

`.afb.0` contains no prime powers at all — that is one of the two reasons the
CADO loader exists — so the fix is a zeroed flag array, stating that fact rather
than rediscovering it 7.6M times.

## Finding 35 — three more parameter combinations that failed silently

- **`--apply-threads` was never validated** (only `--threads` was). The small
  sieve strides its warp tier by `nwarps = threads >> 5`: below 32 that is
  **zero — an infinite loop on the device**, and a non-multiple of 32 leaves a
  partial warp whose lanes re-run warp-tier entries and double-add their logs.
  Now requires 0 or a multiple of 32 in [32,1024].
  > **SUPERSEDED 2026-08-25 by finding 75:** the upper bound is now **512**,
  > not 1024 — `k_apply` carries `__launch_bounds__(512, 3)` and that first
  > argument is a hard launch ceiling. The flag is also parsed with
  > `parse_int_range_arg` rather than `atoi` as of the same date, so malformed
  > values are rejected instead of silently becoming 0 (auto).
- **`1u << log_region` happened before `log_region` was bounded**; UBSan flags
  `--region 32`. Both `logI` and `log_region` are now bounded before anything
  shifts by them.
- **Probe coordinates were unchecked**, so `--probe 16384,0` aliased the real
  cell `(-16384,1)` and would have certified a position nobody asked about —
  the worst possible failure mode for a parity instrument.

## Finding 36 — the GPU half of Gate 1 now has joules; the CPU half cannot be measured on this box

**Superseded by finding 44 for the Gate-1 comparison.** WSL2 still cannot read
CPU energy directly, but a later same-session HWiNFO64 log measured CPU PPT,
GPU board power and both DIMM PMICs from Windows while the real CPU and GPU
workloads ran. The GPU-only measurements below remain valid historical data.

Gate 1's metric is updates/sec/**joule** and no joules existed. Board power
sampled at 10 Hz (`nvidia-smi --query-gpu=power.draw,utilization.gpu
--loop-ms=100`) across a 400-rep run of the chain, 2026-08-03:

| | chain ms/q | busy board power | J/q |
|---|---:|---:|---:|
| side 1 | 37.97 | 202.0 W (peak 222.7) | 7.67 |
| side 0 | 26.11 | 188.9 W (peak 220.3) | 4.93 |
| **both sides** | **64.08** | | **12.60** |

Idle is 27.65 W (108 samples, nothing on the card). 147 of 160 side-1 samples
sat at 100% utilisation, so this is a clean plateau, not an average over gaps.
The card's cap is 250 W and the chain draws 76–81% of it: this is a
power-limited, fully-occupied kernel with no headroom story to tell.

Three caveats, all of which push the same way:

- `power.draw` is **board** power, not wall. A 90%-efficient PSU puts this
  nearer **14 J/q** at the plug, and that is the number a relations/watt claim
  has to use.
- The box was running 14 `gnfs-lasieve4I1` workers throughout, so the
  milliseconds are contended and J/q is *over*-stated. The wattage itself is a
  device property and is unaffected.
- A GPU-sieving box still pays for its CPU and its idle draw. Per-device joules
  is the wrong comparison; whole-box wall watts in each configuration is the
  right one.

**The CPU side cannot be measured from inside WSL2.** There is no
`/sys/class/powercap` (no RAPL) and no `/dev/cpu/*/msr`, so package energy for
the 9800X3D is not readable here at all. It needs either a Windows-side tool
(HWiNFO64, LibreHardwareMonitor) or a wall meter. **A wall meter is the better
instrument** — Gate 1's real question is relations/sec/watt for the whole box in
each configuration, and a plug measures exactly that with no attribution
argument. Two readings settle it: the box sieving on CPU at 14 workers, and the
box sieving on GPU. Until one of those exists, Gate 1 has a numerator and no
denominator, and no relations/watt claim should be made.

## Finding 37 — the hybrid norm costs 1.4%; fp64-everywhere would have cost 2.7x

Finding 28 called the fp64 fallback "noise" on reasoning alone, and the
not-addressed list flagged that as reasoning rather than measurement. Measured
now. `NORM_CANCEL_TOL` became a compile-time override (`make
DEFS=-DNORM_CANCEL_TOL=...`) purely for this; the default is unchanged and
neither override is a correct setting. Norm-init cost is
apply(`--norm horner`) − apply(`--norm const`), side 1, 300 reps:

| build | apply ms | norm init | chain ms |
|---|---:|---:|---:|
| `--norm const` (no init at all) | 16.94 | — | 30.91 |
| tol = 0 — fp32 only, fallback dead | 23.32 | 6.34 | 37.21 |
| **tol = 4.9e-4 — default hybrid** | **23.82** | **6.89** | **37.87** |
| tol = 1 — every cell through fp64 | 86.87 | 69.94 | 100.82 |

The guard plus the fallback cost **0.55 ms on a 37.9 ms chain — 1.4%**. That is
an upper bound on the fallback alone, since nvcc may fold the now-dead `|·|`
error-bound chain out of the tol=0 build.

The interesting number is the last row. Fixing finding 28 the obvious way —
evaluate every norm in fp64 — costs **63 ms more per q** and takes the chain
from 37.9 ms to 100.8 ms. Against the then-used 182 ms tie-point (now ~225 ms,
finding 43) that is 1.8x instead of
4.8x. Consumer Blackwell runs fp64 at 1/64 rate so the cliff is expected; what
is worth recording is that the accuracy repair the probe needed turned out to
be nearly free, while the naive version of the same repair would have eaten
most of the margin the probe exists to establish.

## Finding 38 — RETRACTED: there is no measured survivor-count discrepancy

**Originally claimed** that our one-sided survivor counts (33.7M side 1, 32.8M
side 0) ran ~1.5x above "las's" 21,650,256 and 23,952,829. Those two numbers
came from `dumpcmp stat` on the `-dumpfile` capture — the same capture
**finding 17 in this document already ruled unusable**, because the dump
carries the small-sieve contribution and *not* the bucket-sieved one. Missing
bucket logs means less is subtracted, S is too high, and the survivor count
from that file is too low by construction. It was never las's survivor count.
The claim is withdrawn; no comparison was actually made.

Two things worth keeping from the attempt:

**The dump was re-confirmed dead, this time positionally.** Running side 1
pinned (`--scale 1.275 --allowance 112 --fbbound 134200000`) and dumping all
536,870,912 positions gives a `dumpcmp diff` delta histogram that is *flat* —
about 4.5M positions at every delta from -8 to +8, 86% of positions outside
that window entirely. Two arrays that share no position-level structure. Our
side of it is sound: at the four gate-5 positions our dump reads exactly
`init - sieved sum` = 219, 153, 195, 195 at `x = j*I + i + I/2`. las's file
reads none of those under either i orientation.

**Exact scale pins the bound with no fitting.** At the documented `scale =
1.275` (not the printed 1.28) the rule `(unsigned char)(scale * lambda * lpb) +
LOGNORM_GUARD_BITS` gives `(uchar)(142.8) + 1 = 143` — **las's bound exactly**.
At 1.28 it gives 144. Finding 25's exact scales and finding 31's bound rule
agree only at the exact value, which is a second independent check on both.

The real lesson is procedural: this document contained the refutation (finding
17) about 600 lines above the claim, and I quoted the discredited numbers
anyway. Numbers from a source marked unusable have to be deleted, not left
lying in a table where they read as data.

## Finding 39 — the only las survivor number that exists is two-sided

Chasing finding 38 turned up what the oracle can and cannot support. las's `-v`
log reports:

```
# survivors before_sieve: 536870912
# survivors after_sieve: 797028 (ratio 1.48e-03)
# survivors trial_divided_on_side[0]: 33355
# survivors enter_cofactoring: 1851
# survivors smooth: 37          -> 37 relations
```

`after_sieve: 797028` is **after intersecting both sides**. las never prints a
one-sided count, and the byte dump that would have given us one is broken. So
**there is no way to gate our one-sided survivor rate against las at all** —
not with a better comparison, but at all. The gate that does exist is the
two-sided one, and it requires the intersection we have not built.

That collapses three open items into one deliverable. The two-sided
intersection is simultaneously the missing pipeline stage, the only available
survivor parity gate (target: **797,028**), and the input to every
cofactor-cost projection. It should be built next.

Note also the funnel las reports: 797,028 survivors -> 1,851 entering
cofactoring -> 37 relations. Two orders of magnitude are removed by trial
division and leftover-norm checks *before* cofactoring, which is the part of
the CPU cost this probe keeps assuming is unchanged.

## Finding 40 — the two-sided survivor gate PASSES to within 0.25%

`bench --survbits FILE` writes a 1-bit-per-position survivor bitmap (64 MB);
`dumpcmp and A B logI` intersects two of them and applies the primitive-point
filter. Both sides pinned to las:

| | ours | las | |
|---|---:|---:|---|
| side-1 bound | **143** | 143 | exact |
| side-0 bound | **141** | 141 | exact |
| side-1 one-sided | 14,888,741 | 14,139,941 | +5.3% |
| side-0 one-sided | 18,936,923 | 18,663,976 | +1.5% |
| **two-sided** | **841,418** | **797,028** | **+5.6%** |

Both bounds land on las with no fitting, from `(unsigned char)(scale * lambda *
lpb) + LOGNORM_GUARD_BITS` at the **exact** scales 1.275/1.925 — 1.28 would
give 144. A second independent confirmation of findings 25 and 31.

Tightening only the side-1 bound by the already-understood one-unit norm
difference (`--allowance 111` -> bound 142) gives **795,037 against 797,028,
-0.25%**, and side 1 one-sided 14,062,732 against 14,139,941, -0.55%. That
one unit is diagnostic only — it must **not** be hardcoded, it is las's
approximation and ours is the more accurate side (PARITY.md).

las never prints a one-sided count; those two las numbers came from opening the
opposite side's bound to 253, which excludes only the 0.4% of positions whose
byte saturates. The two-sided figure is las's own `survivors after_sieve`
straight from its log, so this gate does not touch the broken `-dumpfile`.

**Bonus result, and it matters more than the parity.** las returns **37
relations at bound, bound+6 and bound+12** (`after_sieve` 797,028 -> 1,408,504
-> 2,438,956). Tripling the survivor count yields zero extra relations, so
every survivor past the tuned bound is pure cofactorisation waste. Survivor
*rate* is therefore a first-order cost input, not a detail.

## Finding 41 — CORRECTED: the "1.74x failure" was comparing two different populations

This finding first reported 1,386,939 against 797,028 and called the gate
failed. It was wrong, and the error was mine in a way worth recording.

**CADO's `after_sieve` counts PRIMITIVE points**: it has already dropped every
`(i,j)` with `gcd(i,j) > 1`, not merely the both-even ones. Our filter
(`--not-both-even`, `bench_kernels.cu`) removed only both-even. Applying the
right filter moves our two-sided count 1,386,939 -> **841,418** and closes
essentially the whole gap.

This convention is written down in `prototype.md`, in a list headed *"Known
parity gotchas, collected so nobody rediscovers them"*: "skip positions with
`gcd(i,j) > 1` conventions (`unsievethresh`)". I rediscovered it, from the
wrong end, after a day of measurement.

Two chains of reasoning built on the bad comparison, both now void:

- **The "clean six-unit shift on both sides".** Real, reproducible, and an
  artifact — two populations differing by ~35% of positions produce a smooth
  CDF offset that looks exactly like a calibration error. It reconciled with
  las at three bound settings across a 3x range of counts, which is precisely
  why it was convincing. *A consistent-looking offset between aggregates is not
  evidence of a constant; it is evidence the aggregates are comparable, which
  was the untested assumption.*
- **The p=2 ladder suspicion.** The "wrong sign" on both-even
  over-representation (33.7% of our survivors against a 25% baseline) is not
  anomalous at all. Removing the common power of two from a nonprimitive point
  leaves the cofactor of a *smaller* primitive point, so nonprimitive points
  are unusually likely to pass until they are explicitly filtered. Expected
  behaviour, not a bug.

The fifteen `las_tracek` traces were right all along: las is **+1**, and only
+1, everywhere. Per-position parity was never in question — the aggregate was
counting a different set. When per-position evidence and aggregate counts
disagree, the population definition is the thing to check first, before any
hypothesis about the arithmetic.

**Implementation note.** The right shape is the one now in `dumpcmp`: intersect
the two device bitmaps first, then compact and gcd-filter the ~1.4M
intersections — **not** 537M gcds over the whole area. `--not-both-even` stays
as the device-side pre-filter because parity is nearly free in the kernel
(`x`'s low bit and `x >> logI`'s low bit) while a gcd is not; the full
primitive-point test belongs on the compacted host-side set.

Still open: the exact survivor-*set* comparison. Matching counts to 0.25% is
not the same as matching membership, and nothing here has compared the sets
element by element.

## Finding 42 — a clean single-thread las number, on the parity profile only

With the CPU actually free (the GGNFS job finished), 31 special-q from
q=120000053, same flags as the parity capture minus `-dumpfile`:

```
# Total elapsed time 321.53s, per special-q 10.372, per relation 0.218581
# Total 1471 reports [0.182s/r, 47.5r/sq] in 322 elapsed s [83.1% CPU]
```

**10.37 s per special-q, 47.5 relations per special-q, single-threaded.**
Against our 66.9 ms for the two-sided sieve chain (39.0 + 27.9).

**This is not a baseline and must not be quoted as one.** `-B 14` forces las
into 2^14 bucket regions to match our region size, and `-adjust-strategy 0`
pins I and J — both are parity settings chosen to make the *comparison* valid,
and both cost las performance. `-t 1` is one thread of sixteen. The honest
CADO Gate 0 number needs production flags and the full box. At the time of
this parity run GGNFS `N_eff` was still unmeasured; finding 43 immediately
below measures it and supersedes the old 182 ms tie-point.

What it does establish: 47.5 relations per special-q, and the funnel
797,028 survivors -> 1,851 into cofactoring -> 37 smooth at the parity q.

## Finding 43 — N_eff measured at last: 10.24, not the assumed 13

Measured by codex on the quiet box, GGNFS `gnfs-lasieve4I15e` scaling sweep:

| workers | steady q/s | N_eff |
|---|---:|---:|
| 1 | 0.331 | 1.00 |
| 8 (physical cores) | 2.120 | 6.40 |
| 14 | 2.974 | 8.98 |
| **16** | **3.392** | **10.24** |

Sixteen workers win — SMT is worth 1.6x over the eight physical cores, which is
why 14 was leaving throughput on the table. `N_eff` is the divisor in every
target number in this document and it had never been measured; the assumed 13
was optimistic.

**Every guidepost moves, and against us.** Production-equivalent CPU time is
**~295 ms/q, not the assumed 238 ms**. The two lines this document has quoted
since the first page become approximately:

| | old (assumed N_eff 13) | **measured (N_eff 10.24)** |
|---|---:|---:|
| tie the box on sieve work | 182 ms | **~225 ms** |
| hybrid retained TD stage (if kept on CPU) | 56 ms | **~70 ms** |

Quiet-machine GPU timing, same session: side 1 38.056 ms, side 0 26.166 ms,
**64.2 ms/q** on the equal-work profile (65.0 ms on full-alim parity).

So the GPU sieve sits just below the **hybrid's retained CPU TD stage** — 64.2
ms against ~70 ms. That makes the 9800X3D + RTX 5070 a promising balanced
hybrid pair, but it does not create an irreducible project floor. Everything
that remains unbuilt—device intersection/compaction, resieve, factor recovery,
GPU cofactorization, and relation output—now determines the primary
GPU-resident result. Pause sieve-kernel optimization until those stages reveal
the actual critical path; do not infer that sieve milliseconds have no value
in an all-GPU or multi-GPU design.

## Finding 44 — Windows-side power closes the component-energy proxy

HWiNFO64 logged the Windows sensors every two seconds while codex drove the
actual workloads from WSL2. The saved run has 466 usable samples over 931 s.
The raw local capture is
`C:\Users\Kyle\OneDrive\Documents\loads.CSV` (not committed to this repo).
The comparison uses

```
P_proxy = CPU PPT + NVIDIA GPU Power + DIMM0 Total Power + DIMM1 Total Power
```

so both configurations pay for the CPU socket, the discrete GPU board and the
two DDR5 modules. It is a substantially better comparison than finding 36's
GPU-only joules, but it is still a **component proxy, not wall power**: it omits
the motherboard/chipset, storage, fans, VRM losses and PSU conversion losses.

Same-session idle and the full 16-worker CPU plateau:

| configuration | CPU PPT | GPU board | two DIMMs | **component proxy** |
|---|---:|---:|---:|---:|
| idle (45 samples) | 37.637 W | 28.669 W | 2.006 W | **68.311 W** |
| 16-worker GGNFS (27 samples, before the first worker exited) | 125.073 W | 28.436 W | 5.222 W | **158.731 W** |

The power batch covered 448 special-q and 20,445 relations. Summing each
persistent worker's steady rate gives **3.55825 q/s** (45.64 relations/q), or
281.0 ms of whole-box time per q. That makes the CPU configuration
**44.609 J/q** on the component proxy, or **25.411 J/q above idle**. This
short power batch does not replace finding 43's broader `N_eff` sweep; it is
the simultaneous throughput denominator for this power capture.

The benchmark times transform, fill and apply in separate repetition blocks,
and the HWiNFO trace resolves the resulting plateaus. Weighting each power
plateau by that stage's measured time is required; a single sample or a simple
mean over the invocation is wrong:

| stage | algebraic proxy | rational proxy |
|---|---:|---:|
| transform + plattice | 221.213 W | 228.283 W |
| fill | 237.330 W | 236.125 W |
| apply + norm + small sieve + scan | 281.775 W | 282.384 W |

| side | chain ms/q | stage-weighted proxy | proxy J/q | J/q above idle |
|---|---:|---:|---:|---:|
| algebraic | 38.177 | 264.183 W | 10.086 | 7.478 |
| rational | 26.194 | 259.172 W | 6.789 | 4.999 |
| **both sides** | **64.371** | **262.144 W** | **16.874** | **12.477** |

Of the two-sided 16.874 J/q, 13.306 J is GPU-board energy, 3.440 J is CPU
package energy and about 0.129 J is DIMM energy. Independent `nvidia-smi`
spot checks agreed with the HWiNFO watt trace: 227.16 W during algebraic apply
and 179.41 W during rational fill. Several unrelated HWiNFO utilisation and
temperature columns were frozen and were not used. The earlier standalone
`idle.CSV` reported about 14 W for the GPU while NVML read about 29 W, so it was
excluded; all baselines and deltas here come from the self-consistent load log,
whose idle GPU value (~28.7 W) agrees with NVML.

For the **measured portions only**, the GPU sieve requires 37.8% as much total
proxy energy per q as the complete CPU GGNFS q (a 2.64x advantage), or 49.1%
as much energy above idle (a 2.04x advantage). That is encouraging, but it is
deliberately **not an end-to-end perf/watt claim**: the CPU number includes its
complete q, while the GPU number stops after the two sieve sides. Device
intersection, primitive filtering, transfer, resieve/factor recovery, the
TD/cofactor feed and unique-relation accounting must be measured before Gate 2
can close. A wall meter is still the final instrument for economics, but CPU
power is no longer the binding unknown in Gate 1.

## Finding 45 — secondary hybrid whole-box projection

Finding 44 compares the GPU *sieve* against a complete CPU *q*, and says so.
This section asks a useful but secondary deployment question: what would the
measured 9800X3D + RTX 5070 do if the GPU sieved while GGNFS's factor recovery,
trial division, and cofactoring remained on the CPU? It is still a projection—the
glue is unbuilt—but its arithmetic inputs come from measurements rather than
the earlier planning assumptions. It is **not** the primary GPU-resident result.

**Cross-check first.** The GGNFS breakdown extrapolated to the measured
`N_eff = 10.24` predicts 302.1 ms/q. Directly measured: 294.8 ms/q (finding 43
sweep) and 281.0 ms/q (finding 44 power batch). Within 5–7%, by two independent
routes. For this hybrid split, **231.1 ms/q moves to the GPU and 71.1 ms/q is
retained on the CPU** for trial division and cofactoring.

A hybrid box pipelines: the GPU sieves q+1 while the CPU cofactors q, so
per-q time is the **max**, not the sum.

| | CPU-only | hybrid (projected) |
|---|---:|---:|
| ms per q | 281–295 | **71.1** (max of GPU 64.4, CPU floor 71.1) |
| component-proxy watts | 158.7 | **337.0** (125.1 CPU + 206.7 GPU board + 5.2 DIMM) |
| **J per q** | **44.6–46.8** | **23.96** |
| throughput | 1.00x | **3.95–4.15x** |
| **energy** | 1.00x | **1.86–1.95x** |

**Three things follow for this deployment mode.**

1. **The projected hybrid energy win is half its throughput win.** ~4x faster,
   ~1.9x more efficient, because the GPU adds 207 W while the CPU remains busy.
   For this hybrid, **1.9x is the relevant projection**, not finding 44's 2.64x
   stage comparison and not the 4x throughput number. The GPU-resident
   relations/watt result remains unmeasured.

2. **One RTX 5070 approximately fills one 9800X3D-class CPU in this split.**
   The ideal max() is pinned by 71.1 ms of retained CPU work versus a 64.4 ms
   sieve. Under perfect overlap and zero glue, further sieve optimization does
   not improve this particular hybrid. That statement does not apply to the
   GPU-resident path, to another CPU/GPU ratio, or if missing glue puts the GPU
   stage back on the critical path.

3. **The hybrid's next lever is its retained side.** Tightening the survivor
   bound did not reduce the 37-relation yield at bound, bound+6, or bound+12,
   but it also did not remove the retained work. Moving resieve/TD and
   cofactorization to the GPU is the primary roadmap; retaining them on the
   CPU remains the optional shortcut.

**Assumptions, stated so they can be attacked.** Perfect pipelining with no
stall; the CPU cofactor path unchanged and fully parallel at `N_eff = 10.24`;
GPU and CPU plateau powers additive; approximately one strong CPU available per
GPU. CPU cofactor-only scaling and power have not been isolated directly.

**For hybrid accounting, resieve is already charged to retained CPU TD.**
`oracle/ggnfs_timing_breakdown.txt` accounts for wall time in four categories
that close to within 0.01% at every q: Sieve (1683 ms), medsched (309) and
Sieve-Change (373) — all replaced — against **TD including MPQS (734 ms),
retained**. There is no separate resieve line because lasieve4 recovers a
survivor's prime factorisation *inside* trial division, which is the retained
phase. The 71.1 ms hybrid stage therefore already pays for it, and the hybrid
GPU need only deliver survivor `(i,j)` pairs. For the primary GPU-resident
design, resieve/factor recovery remains an unimplemented and unmeasured GPU
stage; CPU bookkeeping does not resolve that engineering work.

One caveat for the hybrid: this holds for the **GGNFS** path, which the 182/56 split
descends from. CADO's las does have an explicit resieve inside its sieve time,
so a CADO Gate 0 comparison must re-derive the split rather than reuse this
one. The residual risk is instead our **+5.6% survivor excess** (finding 40),
which raises the retained TD load by the same 5.6%.

The hybrid's survivor transfer is not a bandwidth risk: ~800K survivors/q at
4–8 B is 3–6 MB, and at 14 q/s that is 45–90 MB/s over an x16 link. The primary
path should keep those candidates resident and transfer final relations, so
this bandwidth observation is not an argument for choosing the hybrid.

## Finding 46 — the GPU-resident budget: what Path 4 has to hit

The primary target is GPU-resident, so the hybrid max() model does not apply —
a standalone box pays **sieve + every post-sieve stage, summed**. That makes
the verdict a budget question, and every input is now measured, so the budget
is derivable rather than notional.

Box power in GPU-resident operation is finding 44's stage-weighted proxy,
**262.1 W** (206.7 W GPU board + ~53 W host + ~2 W DIMM) — measured while the
GPU sieves and the CPU only orchestrates, which is exactly the target mode.
Against the CPU-only box at 44.6–46.8 J/q:

| goal | total ms/q allowed | **post-sieve budget** (after the 64.4 ms sieve) |
|---|---:|---:|
| throughput parity | 281–295 | ~220 ms — not the binding constraint |
| **energy parity** | **170–179** | **~106–114 ms** |
| **2x energy win** | **85–89** | **~21–25 ms** |

Read that carefully, because it is the whole project in three rows.
**Throughput parity is nearly free** — the sieve alone is 4.5x faster than the
CPU box, so almost any post-sieve path beats the CPU on relations/sec.
**Energy parity is comfortable**: ~106–114 ms for intersection, compaction,
primitive filtering, resieve/factor recovery, TD and cofactorisation, against
a CPU that needs 71 ms of *whole-box* time for the last three. **A 2x energy
win is tight**: ~21–25 ms for all of it.

So the honest statement of the remaining question is not "can it work" but
**"where in the 21–114 ms band does the post-sieve path land"** — and the
answer decides between a marginal result and a strong one. That is a far
better-posed question than this project had yesterday.

Two required rates follow, at the measured 15.5 q/s sieve pace:

| stage | volume | **required rate** |
|---|---:|---:|
| primitive candidates out of intersection | ~0.80M/q | **12.4M/s** |
| hard cofactors after TD funnel | ~1,900/q | **29,500/s** |

For reference the CPU produces 1.015–1.065 **relations/joule** at 47.5
relations/q. That is the number to beat and it is the first time it has been
written down.

**Caveat on the power figure.** 262.1 W was measured with the GPU running the
*sieve*. Post-sieve stages have different occupancy and instruction mixes —
ECM in particular is integer-heavy and branchy — so their board power may
differ. The budget should be re-derived once any post-sieve stage is measured;
until then treat 262 W as the sieve-plateau estimate, not a pipeline average.

## Finding 47 — preliminary YAFU 3LP cofactor benchmark clears the rate gate; a target run is still required

The owner supplied a 2026-02-05 log from Ben Buhrow's standalone
`nfs_3lp_batch_factor` suite. This was **not rerun in this results session**
because the CPU and GPU were occupied, and the raw log is not yet a frozen
artifact in this repository. It is nevertheless useful as a fail-fast result:
the RTX 5070's recurring 64/96-bit cofactor work is comfortably faster than
Path 4's estimated feed rate.

The source dataset is one million records from a **C164 GNFS job with
`lpbr=lpba=31`**. The project target is C183 with `lpbr=31, lpba=32`, so this is
not a target-equivalent benchmark. The CPU and GPU paths also use different
algorithms—batch GCD on the CPU versus staged ECM on the GPU—so the ratio is a
comparison of complete cofactor strategies, not the speedup of one identical
kernel.

The logged commands were `./cuda_3lp -m 1` on the CPU and
`./cuda_3lp -m 0 -b1 300 -b2 50 -c 100 -s 10` on the GPU. The latter printed
an effective 96-bit ECM run at `b1=300`, `b2=15000`, 100 requested curves and
a ten-curve no-success stopping rule. The CPU's 22.7166 s timer excludes both
its one-time 54,321,530-prime product build and the 0.55 s input parse; the GPU
outer timer excludes its 0.98 s parse but includes CUDA startup and teardown.

| path | reported time / 1M inputs | input rate | vs. 29,500/s Path-4 feed | equivalent time/q at 1,900 inputs/q |
|---|---:|---:|---:|---:|
| Ryzen 7 9800X3D batch solve (`-m 1`) | 22.7166 s | 44,021/s | 1.49x | 43.16 ms/q |
| RTX 5070 outer cold invocation (`-m 0`) | 11.9524 s | 83,665/s | 2.84x | 22.71 ms/q |
| RTX 5070 sum of printed recurring stages | **2.5901 s** | **386,084/s** | **13.09x** | **4.92 ms/q** |

The 2.5901 s estimate is the sum of the timers printed around the three work
stages: 140.4765 ms for 367,684 64-bit inputs, 2,431.7275 ms for 903,301
96-bit inputs, and 17.9066 ms for the final 19,352 64-bit inputs. The 96-bit
stage is 94% of that sum. Ben's correction that the useful GPU work was about
2.4 s is therefore directionally right. The more conservative **2.5901 s is
the provisional steady-work estimate**; it is not yet a clean persistent-mode
wall-clock measurement.

Source inspection explains much of the 11.9524 s cold time: the outer timer
includes GPU discovery, context creation, PTX load/JIT, kernel setup,
allocation/packing, and teardown. A production queue can retain the context,
module, and buffers. It is not valid, however, to label the entire
`11.9524 - 2.5901` s difference one-time until a persistent driver times the
full recurring path, including preparation, compaction, and validation.

The GPU run produced 10,246 complete relations versus 10,257 on the CPU:
**99.893% of the CPU count, 11 fewer**. That is encouraging but not a
correctness proof. The GPU run stopped after 95 of 100 requested curves after
ten curves without a valid factor, and the two strategies need not choose the
same successful subset. The controlled rerun must compare validated relation
sets and report yield as well as speed.

If the C164 stage mix and difficulty transfer to the target workload, the
hard-cofactor stage would consume about **4.9 ms/q**, leaving roughly
**16–20 ms/q** of finding 46's 21–25 ms post-sieve allowance for a 2x energy
win. Even the cold outer timing clears the raw 29,500/s rate by 2.84x, but at
22.7 ms/q it consumes essentially that entire aggressive allowance. The
correct conclusion is therefore:

> **The preliminary cofactor-rate fail-fast gate passes.** It does not yet
> close target coverage, steady-state latency, energy, correctness, or
> host-independence. If the target rerun holds near 4.9 ms/q, GPU
> resieve/factor recovery/TD becomes the largest unknown inside the 2x budget.

One million records represent about 526 target q at 1,900 records/q—roughly
34 seconds of arrivals at the 15.5 q/s sieve rate. The rerun must sweep smaller
batches and overlapping queues; a result that only saturates after buffering
half a minute is not sufficient evidence for the production pipeline.

### Harness audit before the controlled rerun

The freshly downloaded suite has changed since the February log (including
P-1 support and CPU TinySIQS/MPQS tails), and a source audit found several
prototype hazards to fix or explicitly control before trusting new numbers:

- The stack `relation_batch_t` is not zero-initialized. The CPU batch path
  resets its success count, but diagnostic counters are not initialized; this
  explains the nonsensical ~3.49-billion ECM/abort counters in the old CPU
  log. The current GPU path also needs an explicit success-count reset before
  it increments the field.
- The Makefile defines uppercase `TOOLKIT_VERSION`, while the source tests
  lowercase `toolkit_version`; the CUDA-version branch can therefore be the
  wrong one. Its CUDA-13 branch also passes an uninitialized
  `CUctxCreateParams *` to context creation.
- All devices with compute major >= 9 currently select `cuda_ecm90.ptx`; the
  RTX 5070 log consequently loaded sm_90 PTX on an sm_120 card. Add a native
  sm_120 artifact and loader selection before treating the rerun as a fair
  Blackwell result.
- The GPU path is not host-independent as written: the CPU parses and packs
  records, prepares Montgomery constants with GMP, validates and
  primality-checks results, compacts between curves, and in the newer tree may
  run CPU TinySIQS/MPQS tails. Record CPU utilization and component power, and
  distinguish required recurring host work from removable prototype setup.

The target rerun should therefore use a C183 `31/32` survivor/cofactor set,
zeroed accounting, native sm_120 PTX, a persistent context and reusable
buffers, a batch-size/queue-depth sweep, exact output validation, and
simultaneous CPU/GPU power sampling. Until that run exists, use 4.92 ms/q as a
promising sizing datum—not as a Path-5 result.

## Finding 48 — the standalone bench's `transform` line was measuring CUDA startup

k_transform is the first kernel of a standalone run (no `--pipeline`), so it
absorbed the entire one-time CUDA cost — module load for the fatbin, context
setup — and reported it divided by `--reps`. On WSL2 that fixed cost measures
~170–220 ms.

RTX 5070, c147, `--logI 14 --J 8192`, idle GPU:

| `--reps` | transform | fill | apply |
|---:|---:|---:|---:|
| 3 | **71.181** | 3.812 | 5.424 |
| 20 | 11.969 | 3.840 | 5.494 |
| 100 | 3.177 | 3.832 | 5.567 |
| 300 | 1.123 | 3.843 | 5.560 |
| 1000 | **0.728** | 3.846 | 5.583 |

Transform swings **98×** across a 333× range. Fill moves 0.9% (3.812 → 3.846)
and apply 2.9% (5.424 → 5.583) — apply's drift is small but monotone, so it is
not pure noise, and apply is the one stage that should be quoted with its reps
setting rather than treated as reps-free. The true transform cost is ~0.55 ms;
the pipeline, where it runs once per q against a warm context, independently
reports **0.954 ms** for both sides.

The damage was not hypothetical. Three GPUs were compared on this number at
three different `--reps` settings, and the resulting nonsense — an RTX 3090
appearing to transform 10× faster than a 5070 — was taken seriously for two
rounds before anyone checked whether the metric was stable.

**Fixed** by an untimed warm-up launch ahead of the timed loop, with the
`nproj`/`nlost` memsets moved after it (they are accumulators divided by reps).
That removes ~92% of the artifact — reps 3: 71.18 → 6.05, reps 100: 3.18 →
0.645 — but not all of it *on this box*, so **`--reps 100` is the floor for any
cross-machine comparison** and low-reps transform numbers stay untrustworthy.

**The residual is WSL2-specific.** The 5090 in finding 51 ran this same fixed
binary (commit `ad958cc`) on native Linux and reported transform at **0.149 ms
at `--reps 3`**. That is its true cost, not an inflated one: its pipeline
transform is 0.231 ms for both sides (3.13M ideals), and the standalone sieves
side 1 alone (2.06M), so the expected standalone figure is 0.231 × 2.06/3.13 =
**0.152 ms**. Measured 0.149.

So the warm-up fully solves the problem on native Linux, while WSL2 retains
~12× inflation at reps 3 (6.05 against a true ~0.50). The reps floor is a rule
for *this* box. Numbers other people took at low reps on native Linux were
probably fine; what actually broke was comparing across the two platforms.

The general rule this establishes: **a stage whose reported time depends on
`--reps` is not measuring the kernel.** Fill passes that test at reps 3; apply
passes to within 3%; transform fails it by two orders of magnitude.

The same gap existed in `run_pipeline`, which launched no kernel before
`k_transform` and so charged the whole one-time cost to the first q's transform
window before dividing by the band length. Fixed the same way. At 1340 q it was
worth ~0.15 ms on a 0.954 ms figure (+16%); on a 50-q band it would have been
~4 ms. Lazy module loading is per-kernel, so both warm-ups cover transform only
— fill and apply still pay their first-launch cost inside q0.

## Finding 49 — the grid width was hardcoded to this box's SM count

`blocks = cfg->blocks ? cfg->blocks : 48 * 6` appeared in three places, and no
`cudaGetDeviceProperties` call existed anywhere in the tree. The 48 is this
5070's SM count, so every other GPU ran the 6-blocks-per-SM tuning at whatever
occupancy 288 blocks happened to give it: 3.5 blocks/SM on an 82-SM 3090 (58%),
2.25 on a 128-SM 4090 (37%).

It reached `k_transform`, `k_fill_atomic`, `k_td`, `k_classify`,
`k_resieve_scatter` and the cofactor kernels. `k_apply` launches `nregion`
blocks and was never affected.

Now resolved from `multiProcessorCount * 6` and echoed on stdout at startup —
unconditionally, including when `--blocks` overrides it, since that is exactly
the A/B that wants the number. A failed device query is now fatal: it used to
leave `cfg.blocks` at 0, and the three `48 * 6` fallbacks then silently
reinstated this box's SM count with no diagnostic. Those three constants remain
as unreachable fallbacks; they are dead only so long as every entry point
routes through `main()`'s validation.

Unchanged on this box by construction. The reporter's 3090 saw the standalone
fill benchmark move from ~20 ms to ~14 ms, and ~6% end-to-end on the pipeline
(fill being ~21% of wall). **Those two figures are on the reporter's own
config, which was never recorded** — they are not comparable with finding 50's
3090 fill of 7.19 ms at `--logI 14 --J 8192 --reps 100`, and the ~1.4× ratio is
the only thing to take from them.

Related: `NVCC_ARCH` shipped an sm_120 cubin plus **compute_89 PTX**. The driver
only JITs PTX to a target ≥ the virtual arch, so that build could not load on
any Ampere card at all. Now sm_120 + sm_89 + sm_86 native, compute_80 PTX.

## Finding 50 — the design ports across architectures; fill is the whole gap

Three GPUs, c147 at `--logI 14 --J 8192`. Transform excluded per finding 48.

| | SMs × GHz | INT32 | FP32 | `--reps` | fill | apply |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 (sm_120) | 48 × 2.51 | 15.4 T | 15.4 T | 100 | **3.83** | 5.57 |
| RTX 3090 (sm_86) | 82 × 1.70 | 8.9 T | 17.8 T | 100 | 7.19 | **3.92** |
| A100 80GB (sm_80) | 108 × 1.41 | 9.8 T | 9.8 T | **3** | 9.42 | 6.18 |

INT32 is ops/s; FP32 is FMA/s; both are T. Consumer Ampere has 128 FP32 lanes
per SM but only 64 that accept INT32; Blackwell unified all 128. GA100 has 64
of each.

The A100 rows predate finding 48's `--reps 100` floor. Fill is reps-stable so
that row stands; apply carries up to ~3% of reps drift and should be re-taken.
Transform is omitted for all three because no reps setting makes it comparable
across machines.

**Apply's FP32 ranking is exact; its magnitudes are not.** Predicted ranking
3090 > 5070 > A100, measured 3.92 < 5.57 < 6.18 — three for three. But the
3090's predicted ratio (15.4/17.8 = 0.865 → 4.82 ms) misses the measured 3.92
by 19%, which is the same order of error as the fill miss flagged below as
unexplained. The ranking is evidence the stage is FP32-led; the model does not
predict its size. The A100 reaching near-parity on 0.63× the FP32 is consistent
with apply being 58% DRAM-bound (finding 8) on 2.9× the bandwidth, but that is
an explanation offered, not a fit tested.

**Fill tracks INT32 for the 3090** — predicted 1.73× slower than the 5070,
measured 1.88× — **and fails for the A100**, predicted 1.58×, measured 2.47×.
That miss is unexplained.

**REFUTED — see finding 51.** The hypothesis here was cursor contention: fill
scatters into 8192 cursors fixed by `--region` rather than by the GPU, so wider
cards were thought to pile more SMs onto the same contention points. `--region
13` on a 128-SM 4090 made fill *worse* (6.785 → 7.028) and left the 4090/5070
ratio unchanged, so cursor count is not the variable. The right axis is total
concurrency, and the INT32 model above is refuted with it — finding 51 has the
sweep. The 5070 control recorded here stands as data: fill is flat at **3.729 /
3.742 / 3.965 ms** for 8192 / 16384 / 32768 cursors, and apply degrades hard as
regions shrink (5.51 → 8.28 → 13.36 ms over the same sweep), so `--region` is
not a tuning knob in either direction.

**Whole-pipeline consequence.** Same command, same work (1340 q, ~159.8K
relations):

| | 5070 | A100 |
|---|---:|---:|
| wall/q | 25.10 | 37.02 |
| sieve, both sides | 17.74 | 29.73 |
| — transform / fill / apply | 0.954 / 7.459 / 9.322 | — |
| TD + classify, device | 3.121 | 4.420 |
| host per-q | 0.811 | 0.892 |

The sieve gap (11.99 ms) **is** the wall gap (11.92 ms). TD, cofactorisation
and host work contribute nothing net — the Amdahl risk the project was designed
around did not materialise here. Within the sieve, fill accounts for ~11.0 ms,
**92% of the total deficit**.

Two conclusions for the probe. The kernels are portable: nothing
architecture-specific broke, and the ranking is explained by published lane
counts rather than by anything in the design. And **fill is the only lever left
worth pulling** — 42% of sieve time on this box, and essentially the entire
difference against datacenter silicon. Production transform is 0.954 ms, 5% of
sieve; there is nothing there.

Caveat on the A100 rows: they were taken before finding 48's warm-up landed, at
`--reps 3`, so fill and apply are trustworthy (reps-stable) and transform is
not reported. A rerun at `--reps 100` on the current tree would tighten them.

## Finding 51 — fill saturates at 144 blocks on every card and does not scale with the GPU

> **SUPERSEDED by finding 52 on the geometry, 2026-08-06.** Every block sweep
> below holds `--threads` at 256 and varies blocks alone. Varying the block
> *width* moves the optimum to **1152 × 32** and dissolves the
> architecture-specific block response recorded here — the 4090's "+38% by 768"
> degradation reverses sign at 32 threads. `FILL_BLOCKS_DEFAULT` is 1152, not
> the 144 asserted below. What survives is the *scaling* result: fill still
> returns far less than the hardware ratio, on the corrected geometry too.

Adding the RTX 4090 (AD102, 128 SM × 2.52 GHz) and RTX 5090 (GB202, 170 SM ×
2.41 GHz) to the finding 50 set produced a result no hypothesis on the table
predicted. Same `--logI 14 --J 8192`:

| | transform | fill | apply | chain |
|---|---:|---:|---:|---:|
| RTX 5070 | 0.504 | **3.777** | 5.482 | 9.763 |
| RTX 4090 | 0.250 | 6.785 | **2.814** | 9.849 |
| RTX 5090 | 0.149 | 3.250 | 1.908 | 5.307 |

The 4090 tied the 5070 on the total from stages differing ~2× in both
directions. The 5090 rows are `--reps 3`, but on the warm-up-fixed binary
(`ad958cc`) and on native Linux, where that is enough — its 0.149 ms transform
matches the 0.152 predicted from its own pipeline figure. See finding 48 for
why the same reps setting is worthless on the WSL2 box.

A dead heat on the total, from stages that differ by ~2× in both directions.
The full pipeline was the same story: **25.97 ms/q on the 4090 against 25.28
on the 5070** — the wider, hotter, more expensive card losing by 2.7%.

### Three mechanisms, all refuted by measurement

1. **L2 capacity** (the original hypothesis). Dead: the 4090 has 1.5× the
   5070's L2, 1.5× the bandwidth and 2.67× the SMs, and its fill is 1.80×
   slower.
2. **INT32 lane count** (finding 50's model, which fit the 3090 to 8%). Dead:
   the 4090's INT32 peak is 20.6 T against the 5070's 15.4 — **1.34× more**
   integer throughput, 1.80× slower fill.
3. **Bucket-cursor contention** (finding 50's stated hypothesis, with the
   5070's flat `--region` sweep as its control). Dead: doubling cursors on the
   4090 via `--region 13` made fill *worse*, 6.785 → 7.028, and the
   4090/5070 ratio was unchanged at 1.82× vs 1.88×. Finding 50's contention
   note is withdrawn.

### What is actually true: an absolute concurrency optimum

Sweeping `--blocks` with `--stage fill`:

| blocks | threads | 5070 (48 SM) | 4090 (128 SM) | 5090 (170 SM) |
|---:|---:|---:|---:|---:|
| 32 | 8,192 | 5.383 | 6.680 | — |
| 48 | 12,288 | 4.827 | 6.384 | — |
| 64 | 16,384 | 4.828 | 5.392 | — |
| 96 | 24,576 | 4.778 | 6.039 | 4.362 |
| **144** | **36,864** | **3.726** | **4.806** | **3.156** |
| 288 | 73,728 | 3.854 | 5.638 | **3.113** |
| 576 | 147,456 | 3.906 | 6.473 | 3.145 |
| 768 | 196,608 | 3.955 | 6.636 | — |
| 1020 | 261,120 | — | — | 3.238 |
| 1536 | 393,216 | 3.901 | 6.524 | — |

**All three cards saturate at the same absolute 144 blocks** — 3 per SM on the
5070, 1.1 on the 4090, 0.85 on the 5090. Every one of them falls off sharply
below it (96 blocks costs 22–28%) and none gains anything above it. Three cards
spanning **3.5× in SM count** want the same ~37K threads in flight, so SM count
is the wrong axis for this kernel, not merely the wrong constant.

Above saturation the architectures split. **Blackwell is flat** — the 5070
varies 6% out to 1536 blocks, the 5090 4% out to 1020. **Ada degrades**, the
4090 climbing 38% by 768. So the earlier reading that both cards "degrade in
both directions" was wrong: only Ada degrades upward, and the 4090's 27% was a
penalty specific to it rather than a gain available everywhere.

144 is nonetheless the right default on all three: at or within noise of the
best point on every card, and it avoids Ada's penalty entirely.

The Ada/Blackwell split also fits the two fill misses finding 50 could not
explain — the A100 at 648 blocks and the 3090 at 492 were both far past 144 on
pre-Blackwell parts, though neither was swept to confirm it.

### Fill does not scale with the GPU

The 5090 has **3.5× the SMs, 2.7× the bandwidth and 3.4× the FP32/INT32** of
the 5070. Measured at each card's best:

| stage | 5070 | 5090 | speedup |
|---|---:|---:|---:|
| transform | 0.770 | 0.231 | 3.33× |
| apply | 9.385 | 4.681 | 2.01× |
| TD + classify (device) | 3.091 | 2.567 | 1.20× |
| **fill** | **3.726** | **3.113** | **1.20×** |

Every stage scales except fill, which returns 20% for 3.5× the hardware. Across
all five cards measured, fill spans only 3.11–9.42 ms while apply spans
1.91–6.18 tracking FP32 cleanly. **Fill is a near-fixed cost that GPU money
does not buy down**, and it is 40–56% of sieve time. A 5070 lands within 20% of
the best fill any card tested achieves.

The dominant sieve stage is insensitive to everything that makes a GPU
expensive. Note carefully what that does and does not imply: it bounds
relations per **dollar**, not relations per **joule**. A card that cannot use
its width also does not draw for it — the measured-power table below shows the
5090 idling at 47% of nameplate — so flat fill scaling and low draw are the
same fact, and it is only the throughput half of it that costs anything. This
paragraph previously ran on to call the result central *for a probe graded on
relations/sec/watt*, which had the sign backwards.

The surviving candidate mechanism is L2 write-combining decay: more concurrent
walks interleave each bucket's writes further apart, so 32 B sectors evict
before they fill. It is consistent with every observation including the mild
more-buckets-is-worse trend on both cards. **It is a candidate and nothing
more** — three mechanisms that also fit the data at the time have already been
refuted here, and confirming this one needs counters (`lts__t_sectors_op_write`
vs `dram__bytes_write` per card). `ncu` on a rented Vast box hits
ERR_NVGPUCTRPERM, which is a host module parameter and not fixable from inside
the container.

### The fix

`FILL_BLOCKS_DEFAULT = 144` in `bench.h`, an absolute count, with
`--fill-blocks N` to override. Fill's grid is now decoupled from the
`multiProcessorCount * 6` grid the other kernels use, and **`--blocks` no
longer moves fill** — the sweeps in this finding were taken with the old
binary where it did. Both grids are echoed at startup.

Measured at one job shape (8192 buckets, 77.4M records). The optimum plausibly
moves with bucket and record count; that is not measured, so characterise a new
job shape with `--fill-blocks` before trusting the default on it.

### Consequences

Projecting the 4090's 27.6% fill gain onto its pipeline: sieve loses ~3.5 ms,
wall goes 25.97 → **~22.5 ms/q, an 11% win over the 5070** rather than a 2.7%
loss. Projected, not measured. The 5070 and 5090 are unaffected — for them 144
is inside run-to-run noise of what they already ran (the 5070's 144 point
measured 3.726 and 3.850 in two sweeps, ~3%).

**RETRACTED: the efficiency conclusion was an artifact of nameplate TDP.**

This section previously read "the efficiency conclusion holds across four
cards" and concluded **"this design does not want wide expensive GPUs"** from a
rel/J table built on *nameplate* TDP. That table is withdrawn. Measured board
draw, sampled at 5 Hz through a running band with
`nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 200`,
removes the result:

| | ms/q | rel/s | nameplate | measured draw | % plate | old rel/J | **rel/J** |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 5070 | 25.15 | 4,742 | 250 W | **168.7 W** (590 samples) | 67% | 19.0 | **28.1** |
| **RTX 5090** | 14.95 | 7,978 | 575 W | **270.4 W** (369 samples) | 47% | 13.9 | **29.5** |
| RTX 4090 | 25.97 (→ ~22.5) | 4,593 | 450 W¹ | **254.2 W** (423 samples) | 64%¹ | 10.2 | **18.1** |
| A100 80GB | 37.02 | 3,222 | 300 W | *unsampled* | — | 10.7 | — |

¹ This particular 4090 is capped at **400 W**, so even the nameplate row was
wrong for the card that produced the number; the percentage is against 400.

**Every card draws far under nameplate, and not by the same factor** — 47% for
the 5090 against 67% for the 5070. TDP was therefore not a constant offset that
cancelled in the ratios, which is the assumption the old table's "treat the
ratios as indicative" hedge silently made. It penalised the widest card
hardest, precisely because the widest card is the one that idles.

**The corrected result is a tie, not a reversal.** The 5090 and 5070 land at
29.5 and 28.1 rel/J — within 5%, inside the run-to-run spread this project has
measured elsewhere. The 5090 is not more efficient in any way worth defending.
What it is, is **equally efficient while being 1.68× faster.**

That is a different claim from the retracted one and it points the opposite
way. If rel/J is flat across a 3.5× SM range, relations/joule stops
discriminating between these cards at all, and the decision falls to relations
per **dollar** and throughput per box.

**The split that does survive is architectural, and it is not width.** Both
Blackwell cards sit at 28–30 rel/J; the Ada 4090 sits at 18.1, worse by 36%
than a 5070 costing a third as much. That is the same division finding 51 found
in fill scaling — Blackwell flat above saturation, Ada degrading 38% — now
visible in the power domain. **Generation, not size, is what the measured data
separates.**

**Still not established.** Board draw excludes the host, and the metric of
record is whole-box relations/sec/watt; the A100 is unsampled; and all four
throughput figures are one-q-at-a-time. See "concurrent-q throughput" in the
open experiments — the 5090's 47% draw is headroom, and if that idleness is
schedulable its rel/J moves up from a tie, not down.

### Correction to finding 49

Finding 49 credits the grid fix with moving the reporter's 3090 standalone fill
from ~20 ms to ~14 ms. Given fill is flat from 144 to 1536 blocks, 288 → 492
cannot have produced that. The likely cause is that the "before" figure was a
`SIEVE CHAIN` total at low `--reps`, dominated by the transform startup artifact
of finding 48, which had not been found yet. The grid fix remains correct on
occupancy grounds for transform, TD, classify, resieve and the cofactor kernels;
**its effect on fill was nil and the ~20 → ~14 ms should not be attributed to
it.**

## Finding 52 — fill wants 1152 x 32, and finding 51's architecture split was an artifact of a fixed 256 threads

Finding 51 swept fill's **block count** at a fixed `--threads 256` and read the
resulting 144-block minimum as a hard saturation point, with an
architecture-specific response above it (Ada degrading 38% by 768 blocks,
Blackwell flat). Block **width** was never varied. It is not a free parameter.

### At constant total threads, narrower blocks always win

36,864 threads throughout, so every row is the same work differently cut up.
Standalone `--logI 14 --J 8192 --reps 100`, fill ms:

| T x B | 5070 | 4090 | 5090 |
|---|---:|---:|---:|
| 192 x 192 | 3.612 | 4.482 | 2.893 |
| 128 x 288 | 3.553 | 4.654 | 2.830 |
| 96 x 384 | 3.455 | 4.407 | 2.760 |
| 64 x 576 | 3.530 | 4.419 | 2.716 |
| **32 x 1152** | **3.454** | **4.384** | **2.636** |

### Thread count matters above 128, and not below it

At a constant 576 blocks the 5090 measures **2.716 / 2.711 / 3.147 / 4.289 ms**
at 64 / 128 / 256 / 512 threads. So 256 is **16% off** and 512 is 58% off,
while **64-128** is flat to 0.2%. Note 64, not 32 — 576 × 32 was never
measured, so flatness is established down to 64 and assumed below it.

At the shipped 1152 blocks the gap is **wider**, not narrower — 2.633 at 32
threads against **3.239 at 256, a 23% penalty**:

| 5090, fill ms | 32 thr | 128 thr | 256 thr |
|---|---:|---:|---:|
| 576 blocks | — | 2.711 | 3.147 |
| **1152 blocks** | **2.633** | — | 3.239 |

Note 1152 × 256 (3.239) is worse than 576 × 256 (3.147): **at 256 threads more
blocks still hurts** — the finding-51 behaviour — while at 32 threads more
blocks helps. The two axes interact strongly and neither can be swept alone.

Two consequences: a sweep pinned at 256 threads was 16-23% off the optimum
before it began, and fill's width cannot be tuned through `--threads`, which
also drives transform, intersect, TD, resieve and the cofactor kernels. Hence
`--fill-threads`. This cell was run specifically to test whether that flag
earns its keep rather than being an over-engineered alternative to raising
`FILL_BLOCKS_DEFAULT`; raising blocks alone would have landed on 3.239.

### The knee is 1152 blocks on all three cards

At `--threads 32`, fill ms:

| blocks | 5070 | 4090 | 5090 |
|---|---:|---:|---:|
| 288 | 3.909 | 5.089 | 3.629 |
| 576 | 3.487 | 4.626 | 2.983 |
| **1152** | **3.444** | **4.360** | **2.633** |
| 2304 | 3.410 | 4.407 | 2.668 |
| 4608 | 3.312 | 4.327 | 2.656 |
| 9216 | 3.312 | 4.285 | 2.632 |

Past 1152 the remaining movement is 1.7% (4090), 0.04% (5090) and 3.8% (5070,
whose own run-to-run spread is ~3%) — flat, not falling. Below it the cost is
steep. Overshooting is nearly free and undershooting is not, so the default
sits **at** the knee. Against the old 144 x 256: **7.5% / 8.8% / 16.7%**.

### The Ada/Blackwell split dissolves

Finding 51's headline architectural claim was that the 4090 *degrades* with
more blocks while Blackwell stays flat. At 32 threads the 4090 **improves**
monotonically over the same range. Both sweeps are correct; they differ only in
the fixed thread count. The block response is not a property of the
architecture, it is a property of the width you happen to hold fixed — and at
32 all three cards behave alike, which is why one geometry serves all of them.

### It is not L2

L2 capacity was already dead (finding 51's own list); write-combining decay was
that finding's surviving candidate. A single 1152 x 32 optimum across cards
with **48, 72 and 96 MB** of L2 argues against both. The behaviour it does fit
is **work granularity**: fine chunks balance the tail, and the effect saturates
once chunks are numerous enough — which is exactly the flat plateau above.
Still a candidate, not a conclusion; `ncu` remains blocked on the rented boxes.

### Verified end to end, not just in the microbenchmark

Same binary, same session, geometry the only variable. c147 band, 1340 q,
RTX 5070:

| | 144 x 256 | 1152 x 32 | delta |
|---|---:|---:|---:|
| wall clock/q | 25.00 ms | 24.47 ms | **-2.1%** |
| sieve, both sides | 17.94 | 17.48 | -2.6% |
| transform | 0.766 | 0.764 | -0.3% |
| **fill** | 7.576 | 7.113 | **-6.1%** |
| **apply** | 9.601 | 9.607 | **+0.06%** |
| relations | 159,837 | 159,837 | **byte-identical** |

Apply was the risk worth checking: it reads the bucket array fill writes, and
the write interleaving changed 32-fold. It did not move. The sorted relation
files compare equal, both reconstruction gates pass with the same 1747/1952
factor counts, and `cofcheck.sh` is 30/30. The pipeline gain (6.1%) is smaller
than the standalone predicted (7.5%), so expect the 5090's 16.7% to land nearer
13-14% in a band.

**Methodological note.** The first version of this A/B compared against a fill
figure captured in an earlier session and appeared to show apply regressing
2.4%. It had not: this box's apply drifts ~2.3% between sessions, and the
controlled same-binary run showed +0.06%. **Do not A/B against a captured
number from another session on this box** — rerun the control.

### Not yet measured

- The pipeline A/B is 5070-only; the 5090 is where the largest gain is claimed.
- `k_fill_l1` (twolevel path) has never been swept at any geometry and keeps its
  own 144 x 512 default.

## Finding 53 — host contention costs 29% of wall clock and every GPU counter we have is blind to it

Reported by the 3090 tester: a saturated CPU leaves the CUDA timings untouched
but moves wall clock. Reproduced here on the 5070, c147 band, 1340 q, same
binary, load applied with N spinning shells:

| | idle | 8/16 cores | 16/16 cores |
|---|---:|---:|---:|
| **wall clock/q** | **24.30 ms** | **29.79 (+22.6%)** | **31.27 (+28.7%)** |
| sieve, both sides | 17.48 | 18.94 | 17.41 |
| — transform | 0.764 | — | 0.822 |
| — fill | 7.113 | — | 7.085 |
| — apply | 9.607 | — | 9.505 |
| intersect + gcd | 0.074 | — | 0.105 |
| host per-q (tables, staging) | 0.699 | 1.249 | **1.994** |
| TD + classify, wall | 4.11 | 5.73 | **7.06** |
| host: small-prime tables | 0.467 | — | 0.833 |
| host: unaccounted | 0.528 | 1.594 | **2.869** |
| cofactorisation flushes | 0.56 | — | 0.97 |
| relations | 159,837 | 159,837 | 159,837 |

The sieve row at half load (18.94) is above both other columns, so it is
run-to-run noise, not a trend — the full-load column (17.41 against 17.48
idle) is the one to read for whether the GPU stages move. They do not.

**Replication status: each column is a single run.** That noise disclaimer
concedes an ~8% spread on the largest term, so the wall deltas carry an
uncertainty this finding does not quantify, and "every kernel within ~1%" rests
on the one column that happened to land close. A replication attempt was made
and had to be discarded: the box turned out to be running a production snfs236
sieve on the same GPU plus a 14-thread msieve, which put wall clock at ~53 ms —
so the contamination signature is unmistakable, and its absence from the 24-25
ms idle runs is the evidence that those were clean. **Re-run all three columns
n>=3 on a confirmed-idle box before quoting these as hardware constants.**

**The kernels are flat and the wall clock is not.** Every `cudaEvent`-timed
stage is within ~1% (`fill` -0.4%, `apply` -1.1%). Everything host-side scales
with contention, worst of all `unaccounted` — wall minus device time inside
TD/classify, i.e. launch and synchronisation overhead — at **+443%**.

### Why this is a measurement hazard and not just a scheduling tip

Our instrumentation is *structurally* unable to see the largest environmental
effect on throughput. A host-starved box reports flawless kernel times next to
a bad ETA, which reads as "the GPU is fine, the ETA is inexplicable" — the exact
shape of the original 3090 report that opened findings 48-52. Any wall-clock or
ETA comparison across boxes is invalid without knowing host load on both, and
the rented boxes used for findings 50-52 are shared machines.

The A100 is the one prior result worth re-examining on these grounds, and it is
**not cleared**. An earlier version of this paragraph said its host per-q of
0.892 ms beat "this box's 1.080", concluding its 37.02 ms wall stands. That
1.080 is unsourced and appears nowhere in this repo. Finding 50's own table —
same job, same 1340 q — records **5070 = 0.811 against A100 = 0.892**: the
A100's host was 10% *slower*, the opposite of what was claimed.

A 10% host gap is well inside what two different host CPUs produce, so it
neither establishes contention nor rules it out. **The A100 wall figure is
unverified** and stays so until a band is run there with the ratio line, on a
host confirmed idle.

### It is far larger for us than the reporter's numbers imply

Their host per-q rose **33%** (0.288 → 0.384 ms); ours rose **185%** (0.699 →
1.994). Quote the two percentages rather than a ratio of them — an earlier
heading here said "~4x", which matches no derivation: the relative growths
differ by 5.6×, the multipliers by 2.1×, the absolute milliseconds by 13.5×.
The difference is which harness was run. The standalone bench does almost no host work — a
transform, a sort and one H2D — while the pipeline carries TD tables, staging
and cofactor flushes. **The standalone structurally understates this effect**,
and the standalone is what testers are usually asked to run.

### There is no free headroom

Half the cores costs most of what all of them cost. State it as **throughput**,
not as percent-of-baseline-wall — an earlier version of this section did the
latter and inflated the decision rule:

| | wall/q | wall vs idle | **relation-rate loss** |
|---|---:|---:|---:|
| idle | 24.30 ms | — | — |
| 8/16 cores | 29.79 | +22.6% | **18.4%** |
| 16/16 cores | 31.27 | +28.7% | **22.3%** |

A +28.7% wall is a 22.3% throughput loss (1/1.287 = 0.777), not a 29% one. So
co-scheduling CPU-NFS work beside the GPU siever pays only if that work is
worth more than ~18% of the GPU's relation output at half load, or ~22% at
full — not the "more than a quarter" this section previously claimed, which
overstated the bar by 27% at half load.

### The pipeline now reports this directly

`GPU-accounted / wall (excl cofac)` prints on every band: event-timed device
time (sieve + intersect + the TD/classify device total) over wall clock, with
the cofactor queue removed from **both** sides.

**Values pending re-measurement.** A first version was measured at 0.824 idle /
0.661 loaded, but with the cofactor queue in the denominator only — which made
the ratio move with survivor density, so a candidate-dense band read as a
contended host on an idle box. That is the exact misdiagnosis the line exists
to prevent, so the expression was corrected and those two numbers no longer
describe what the code prints. Re-taking them needs a confirmed-idle box.

The numerator remains a **lower bound** on device time: `k_cof_enqueue`,
`k_cand_stats` and the flush's own kernels are real GPU work that no event
times. The ratio therefore understates utilisation by a small amount that is
roughly fixed for a given job — tolerable for comparing a box against its own
baseline, which is the only comparison it is for.

It is **not** the most sensitive signal available. An earlier version of this
section claimed it separates the two conditions "where every individual stage
timing does not"; that is false. Two already-printed timings separate them far
more sharply — `host per-q` at +185% and `host: unaccounted` at +443%, against
the ratio's −20%. Its merit is being one scale-free summary that needs no
baseline table to read, not being the sharpest.

No threshold and no warning text is attached, deliberately. The healthy value
depends on card and job — a faster GPU spends relatively more of its wall on
the same host work and therefore reads *lower* while perfectly healthy — so a
hardcoded "good" constant would repeat the 144-block mistake of promoting one
box's number to a universal one. Take an idle baseline per card, job **and band
length** and compare against it: `acc_wall` excludes the final cofactor flush,
so on a band shorter than one flush that tail is the entire cofactorisation and
a smoke run is not comparable to a production band.

### What to do about it: the host work is three problems, not one

The 1.694 ms does not have a single fix, because two thirds of it is *prep* and
one third is *launch overhead*:

| | idle | loaded | what it is |
|---|---:|---:|---|
| host per-q (tables, staging) | 0.699 | 1.994 | prep: tiers, staging, H2D |
| host: small-prime tables | 0.467 | 0.833 | prep: derived from the lattice |
| host: unaccounted | 0.528 | **2.869** | launch + sync overhead |

**Overlap beats threading for the prep, and not narrowly.** Both prep terms
depend only on the q-lattice and on no GPU result, so q+1's host work can be
done during q's kernels — double-buffering, not parallelism. The arithmetic is
lopsided: **1.166 ms of prep against ~20.7 ms of GPU work per q**, so it fits
inside the GPU's shadow with 18x room and perfect overlap takes it to zero on
the critical path. Threading the same work reaches perhaps 0.4 ms on four
threads and leaves it *on* the critical path — strictly worse on the arithmetic
alone.

**But the arithmetic is not the whole cost, and an earlier version of this
paragraph implied it was** by calling threading's downside "a synchronisation
problem that does not currently exist." The host does not sit idle through that
20.7 ms shadow: it blocks at six points per q — `cudaEventSynchronize(e3)` once
per side (`pipeline.cuh:240`), the intersect sync (`:975`), two inside
`pipe_td_perq` (`:671`, `:736`), and `cudaDeviceSynchronize` after
`k_cof_enqueue` (`:1021`). Single-threaded overlap requires splicing q+1's prep
in *ahead* of those syncs, i.e. restructuring them — which is most of the work
sub-item 2's graph capture wants anyway. Overlap is still ranked first because
it and (2) share that restructuring, not because it is free.

**`unaccounted` needs the opposite treatment.** It is wall minus device time
inside TD/classify: the CPU issuing launches and waiting on syncs. It is
interleaved with GPU execution by nature, so overlap cannot hide it, and it is
the term that grew **443%** under contention — it is what makes a box fragile
rather than merely slow. The per-q kernel sequence is fixed, so fewer and
larger launches, fewer sync points, or a captured CUDA graph replayed per q all
attack it directly.

**The micro-optimisation is the least valuable third.** Replacing the per-q
stable sort with a three-way partition and fusing the twice-done small-ideal
transform was previously the whole of open experiment 4. After overlap, that
work is hidden in the GPU shadow and its cost stops mattering. Order: overlap,
then graphs, then the partition.

### Consequences

- Open experiment 4 (host cost) is reordered on the strength of the above, and
  gets more valuable — though not on the idle-case arithmetic: 1.694 ms/q of a
  24.30 ms wall is only **7%**. Reducing it buys robustness on shared hardware
  more than throughput on a quiet box, and against open experiment 3's
  15.8-25% duplicate share it is the smaller prize.
- **It interacts with open experiment 1.** Running two concurrent special-q
  roughly doubles host work per unit time, so the concurrency experiment can
  come back negative for host reasons that have nothing to do with the GPU.
  Run it on a verified-idle box and check `GPU-accounted / wall` first.
- Correctness is unaffected: all three runs emit exactly 159,837 relations.

## Finding 54 — RTX 5060 Ti device timings on c147

**Date:** 2026-08-09. Reported externally on native Linux, RTX 5060 Ti
(36 SM, 32 MB L2), current **1152 x 32** fill geometry. The host was an
i5-2550K and was fully utilised during the pipeline run, so the device-event
timings are the portable result here; the pipeline wall clock is deliberately
not used for a cross-card comparison.

Standalone algebraic side, `--logI 14 --J 8192 --reps 3`, synthetic-root
`q=120000011`, 2,059,531 bucketed entries and 77,389,658 records:

| stage | ms / special-q |
|---|---:|
| transform + plattice | 0.687 |
| fill | 4.121 |
| apply | 7.211 |
| **sieve chain, algebraic side** | **12.019** |

The full pipeline used real algebraic special-q from 15,000,000, both sides,
and ran 1,340 q. Its event-timed device accounting was:

| stage | ms / special-q |
|---|---:|
| transform + plattice, both sides | 1.027 |
| fill, both sides | 7.980 |
| apply, both sides | 12.000 |
| **sieve, both sides** | **21.010** |

TD + classify breakdown (the eight component rows sum to the printed device
total):

| stage | ms / special-q |
|---|---:|
| rank scan | 0.189 |
| emit `(x,a,b)` in rank order | 0.021 |
| survivor filter | 0.099 |
| resieve + scatter, both sides | 2.291 |
| norms + trial division, both sides | 0.429 |
| classify, both sides | 0.159 |
| joint accept + compact | 0.023 |
| record candidate factorisations | 0.525 |
| **TD + classify, device total** | **3.736** |

Pipeline device-accounted totals:

| stage | ms / special-q |
|---|---:|
| sieve, both sides | 21.010 |
| intersect + gcd | 0.114 |
| TD + classify | 3.736 |
| cofactor queues + relation readback/emit, device-accounted | 1.390 |
| **device-accounted total excluding cofactorisation** | **24.860** |
| **device-accounted total including cofactorisation** | **26.250** |

The printed `GPU-accounted / wall (excl cofac)` was **0.743**. That low ratio
is consistent with finding 53: the CUDA stages remain measurable under host
contention while wall time grows. The standalone and pipeline sieve totals are
not a one-side/two-side scaling A/B: they use different q, roots, norm scales,
and the pipeline adds the rational side.

## Finding 55 — the item-0 geometry alarm is a false alarm; the factor-base convention is the real mismatch, and it is q-dependent

**Date:** 2026-08-16. No GPU: source reading, document archaeology, and one
direct count of the checked-in factor base. STATUS item 0 raised the
possibility that the GPU was timed on an `I14e` rectangle while being graded
against an `I15e` CPU baseline — a headline flattered by up to 4×, and the
largest single effect in that file. It is not what happened.

**Findings 43 and 44 are `I15e` on both sides.** The standalone benchmark
defaults to `cfg.logI = 15; cfg.J = 16384` (`bench_main.cu:426`), and `--J`
otherwise defaults to `2^(logI-1)` (`:977`). `git log -S'cfg.logI = 15'`
returns only the initial checkin, so the default has never moved. The
finding 43/44 reproduce commands pass neither flag, so they sieved
`2^15 × 2^14 = 5.369e8` positions — which is both the `A` printed in this
file's header and exactly the rectangle `gnfs-lasieve4I15e` sieves.
`RUNBOOK.md:101` fixes the mapping (`--logI N` ↔ `gnfs-lasieve4I{N}e`).

So **64.371 ms/q (38.177 + 26.194) is an `I15e` measurement**, finding 43's
`gnfs-lasieve4I15e` sweep is its matching comparator, and the CPU row to grade
it against is the 104.5 J/q one. The margin for that number is 4.3–5.5×, not
1.5–2.0×.

**The confusion is still live, in the other direction.** Every *pipeline* run on
record passes `--logI 14 --J 8192` — `README.md`, `RUNBOOK.md:74`, `:407`,
findings 50 and 54, and both c147 bands — while the c183 timings item 0 builds
its 70–90 ms/q projection on are standalone runs at the `logI 15` default.
Item 0 therefore projects a pipeline number from a rectangle no pipeline run has
ever used. The rule that follows: **a verdict band states its geometry and is
graded against the matching CPU row.** At defaults the pipeline sieves `I15e`;
at the documented invocation it sieves `I14e`. Both are defensible; mixing them
is worth 4×.

**The factor-base convention is the mismatch that survives, and it is not a
constant.** Standalone side 1 truncates the base at q by default
(`bench_main.cu:773-774`) — GGNFS's convention, and what the equal-work profile
was built to match. The pipeline does not: `fbbound = alim` (`:1360`), and
per-q truncation is item 3, unbuilt. Counted directly over `oracle/c183.fb1`
(bucketed `(p, root)` entries, `p >= 2^15`; `alim` 134.2M):

| special-q | entries at or below q | fraction of full base |
|---:|---:|---:|
| 50M | 2,997,498 | 0.394 |
| 120.000011M | 6,840,491 | 0.900 |
| 130M | 7,377,232 | 0.970 |
| >= 134.2M (`alim`) | 7,601,777 | 1.000 |

At the q the equal-work profile was measured, truncation removes 10% of the
base. At the verdict band's **low probe, q ≈ 50M, the pipeline's full base
carries 2.54× the bucketed entries GGNFS sieves at that same q** — and side-1
transform, fill, and bucket occupancy all scale with entry count. Consequences
for item 0's band as specified:

- The three probes will show a strong q-dependence in ms/q that is **not** yield
  drift and must not be read as such.
- The 70–90 ms/q projection is anchored at q=120M, truncated, standalone. It is
  roughly right at the 130M probe and **optimistic at 50M**.
- At the 190M probe truncation is a no-op — q is above `alim` — so the item-3
  full-vs-truncated A/B has no signal there at all.

Two corrections to older text while it was open. `prototype.md:255` says the
base reaches "the full 7.6M at q≥170M"; it reaches 7,601,777 at `alim` =
134.2M and cannot grow past it. And this file's header quotes 6,843,511 entries
at q=120,000,011 against a direct count of 6,840,491 at `bkthresh` 2^15 — 3,020
apart, unexplained, and not worth chasing except as a note that the header
figure is not reproducible to the digit.

## Finding 56 — the host is not a scheduling problem: renice buys nothing, and the idle box still leaves 11.5% of wall unaccounted

**Date:** 2026-08-17, RTX 5070, c147 band (`--logI 14`, 400 q), item 4's "free
experiment" as STATUS specifies it.

STATUS's decision rule was: *"If renicing moves that number toward the mid-90s,
the contention penalty is a scheduling problem and the code below is not
urgent."* **It does not move.**

| condition | runnable threads | wall ms/q | acc/wall | GPU util |
|---|---:|---:|---:|---:|
| **idle** (3 runs, 400 q) | 1 | **23.28** | **0.885** | **89.5%** |
| idle, 1500 q | 1 | 23.80 | 0.883 | 90% |
| 12 `gnfs-lasieve4I15e`, equal priority | 13 | 24.72 | 0.854 | 86% |
| 12 sievers, sieve favoured (them at nice 19) | 13 | 24.50 | 0.857 | 86% |
| 15 spinners, clean box (`ctl` below) | 16 | 24.76 | 0.835 | 86% |
| 12 sievers **+** 15 spinners, equal priority | 27 | 26.49 | 0.798 | 77% |
| 12 sievers + 15 spinners, sieve favoured | 27 | 26.28 | 0.812 | 81% |

**The penalty tracks oversubscription, not busyness.** At one runnable thread
per logical CPU the cost is **~6%** and the two independent measurements of it
agree (+6.2% with 12 real sievers, +6.4% with 15 spinners). Past that it
roughly doubles: the 27-thread rows are 1.7x subscribed and cost ~14%. The
first version of this finding quoted that 14% as "saturation", which was wrong
— those runs still had a draining siever queue underneath the spinners. The
operational lever this leaves is **worker count**, not any scheduler knob:
keep competing work at or below `nproc - 1`.

Each paired comparison is ~1%, against 2-10% spread *within* an arm. Method
note: renice is one-way for a non-root user, so the equal-priority arm was
reproduced by running `bench` itself at nice 19 — both parties at 19 is the
same CFS weighting as both at 0.

**Why priority cannot work here, which the item was not accounting for.** 15
spinners plus the feeder thread is exactly 16 runnable threads on 16 logical
CPUs, so nothing is waiting for a slot: the feeder is *sharing a physical core*
with a competitor over SMT. Nice values decide who is scheduled, not who you
share a core with. On 8 physical cores you cannot isolate the sieve by
subtraction (STATUS already said this) and it turns out you cannot isolate it
by priority either. The untested lever that follows is **affinity** — pin the
process and keep competitors off both its hyperthreads — which is still zero
code and has not been tried.

**The larger number is structural and needs no competitor at all.** On a
verified-idle box `acc/wall` is 0.885 and GPU utilisation 89.5%: ~11.5% of wall
with the card idle, which is host prep and launch/sync on the critical path,
i.e. items 4.1 and 4.2 exactly. Contention adds ~6% at 12 workers and ~14% at
saturation on top of that.

**But it amortises away with area, which decides how much item 4 is worth.**
Host work per special-q is roughly fixed while GPU work scales with the
rectangle:

| job | area | acc/wall | GPU util | host gap |
|---|---|---:|---:|---:|
| c147, `logI 14 J 8192` | `2^27` | 0.885 | 89.5% | 11.5% |
| c183, `logI 15 J 16384` | `2^29` | 0.946 | 97.5% | 5.4% |
| c183, `logI 15 J 32768` | `2^30` | 0.962 | 98.5% | 3.8% |

At the geometry a C195 would deploy at, item 4's entire prize is under 4% and
part of that is interleaved launch/sync that overlap cannot recover. It is a
small-job concern, not a prerequisite for anything.

Even the 1.7x-oversubscribed 14% is well short of finding 53's 28.7%. The
likely difference is the competitor: pure-ALU spinners contend for scheduling
slots and core resources, while finding 53's real workloads also contend for
memory bandwidth and L3. Finding 53 remains canonical for the memory-bound
case.

**`SCHED_FIFO` and `--blocking-sync` fail too, closing the list.** Run with
root 2026-08-17, 15 spinners, four arms interleaved within each rep so the
ordering survives drift (RT throttling confirmed at 950000/1000000 first — the
feeder spins by default, so an unthrottled FIFO task could hold a CPU):

| arm | what it changes | wall ms/q | vs ctl | acc/wall |
|---|---|---:|---:|---:|
| `ctl` | nothing | **24.76** | — | 0.835 |
| `rt` | `chrt -f 1` | 25.15 | +1.6% | 0.827 |
| `blk` | `--blocking-sync` | 25.35 | +2.4% | 0.824 |
| `rtblk` | both | **25.88** | **+4.5%** | 0.809 |

`rt` lost in 3 of 3 reps and `rtblk` in 3 of 3. Two mechanisms, both the same
shape as the `taskset` failure: `chrt` sets the policy for the **whole
process**, so the CUDA runtime's helper threads become real-time and can invert
priority against the normal-priority kernel work they depend on (under WSL2
that includes the GPU virtualisation path); and `--blocking-sync` hands back
the core only to pay wake-up latency into a busy runqueue at every sync.
`rtblk` -- sleep, then wake at real-time priority, the textbook low-latency
recipe -- is the **worst** arm, not the best.

**Every zero-code lever is now exhausted: nice, taskset, chrt, blocking sync.**
The ~6% one-to-one contention cost is not a scheduling problem in any sense a
scheduler knob can reach; it is contention for core resources (SMT siblings,
cache, memory bandwidth). The remedies that remain are to run fewer competing
threads, to make the GPU less dependent on the host (items 4.1 and 4.2, worth
under 4% at C195 geometry per the area table above), or to accept it.

**Affinity fails too, and pinning alone is actively harmful.** Same harness,
15 spinners, three arms interleaved (siblings are adjacent pairs here, so
physical core 0 is cpus 0-1):

| arm | spinners | bench | wall ms/q | acc/wall | GPU util |
|---|---|---|---:|---:|---:|
| none | 0-15 | anywhere | 25.27 | 0.823 | 83% |
| pin only | 0-15 | 0-1 | **26.73** | 0.789 | 76% |
| isolated | 2-15 | 0-1 | 25.76 | 0.811 | 80% |

`taskset` pins the **whole process**, and `bench` is not one thread — the CUDA
runtime carries its own helper threads, so pinning crams them onto two
hyperthreads instead of sixteen and costs 5.8%. Reserving the core for the
process only recovers that, to roughly the unpinned baseline. Extracting value
this way would need per-thread affinity on the feeder thread alone
(`pthread_setaffinity_np`), which is code — and therefore competes with
double-buffering, which is better code for the same problem.

**Both zero-code levers for item 4 are now exhausted.** Note also what neither
tool addresses: `taskset` constrains only the caller and never keeps others
off a CPU. Real reservation is cgroup v2 cpusets (`cpuset.cpus.partition`),
`isolcpus=`, or `nohz_full=`; the only lever that changes preemption *latency*
rather than placement is `SCHED_FIFO` via `chrt`, which needs privileges and
is the one plausible remaining test. On this box all of them sit above a
second scheduler regardless: WSL2's vCPUs are dispatched by the Windows
host.

## Finding 57 — the CPU baseline is per PRIME q and ours is per (q, rho) PAIR; item 0's margin was overstated 1.53x

**Date:** 2026-08-17, both sides measured in one session on one box, idle.
This is the second time a unit mismatch has flattered the GPU (finding 55 was
the first), and it is the same shape: two conventions living unlabelled in one
document.

### The control

`gnfs-lasieve4I15e` on the c183 (`oracle/input.job`, its own `.afb.0`),
algebraic special-q, default `J = I/2`, over q in `[120000000, 120005000)` —
the same interval and the same `2^15 x 2^14` rectangle the GPU had just
sieved. Four workers over disjoint subranges, 235 s wall on an idle box.

| | GGNFS `I15e` | cuda-sieve `logI 15 --J 16384` |
|---|---:|---:|
| relations | 12,398 | 13,941 |
| distinct primes q | 176 | — |
| **(q, rho) pairs** | **269** | **300** |
| **relations per (q, rho)** | **46.09** | **46.47** |

**Yield matches to 0.8%** — and ours is achieved while sieving 11% *more*
factor base, since GGNFS trimmed to `FB_bound 119999999` (6,844,120 of
7,605,406 entries) and the pipeline runs the full base to `alim`. There is no
yield gap on this job at this geometry. Any future claim of one needs a control
like this, not a cross-session comparison.

### The unit mismatch

GGNFS averages **1.528 roots per prime** here (269 pairs over 176 primes), and
it reports per *prime*. Our `--nq`, `ms/q` and `rel/q` are per *(q, rho) pair*.
Three independent numbers confirm that the CPU rows under item 6 are per prime:

| STATUS I15e row | today, per prime |
|---|---|
| 68.9 rel/q (q=130M) | 70.44 (q=120M; yield falls with q) |
| 474.8 ms/q | 3.138 core-s x 1.528 / N_eff 10.24 = **468 ms** |
| 104.5 J/q | 468 ms x 220 W = **103.0 J** |

All three reconcile within 1.5%, so the CPU baseline is sound — it is only
mislabelled. Item 0's margin table then put a per-pair GPU energy
("70-90 ms/q x 270 W = 18.9-24.3 J/q") against that per-prime 104.5 J and
claimed 4.3-5.5x.

### The corrected verdict, both sides in one unit

| | per (q, rho) pair | per prime q |
|---|---:|---:|
| GPU wall (idle box, `--cofactor`) | 98.5 ms | 150.5 ms |
| CPU whole box, N_eff 10.24 | 306 ms | 468 ms |
| **time advantage** | **3.11x** | **3.11x** |
| GPU energy at 270 W whole-box | 26.6 J | 40.6 J |
| CPU energy at 220 W | 67.4 J | 103.0 J |
| **energy advantage** | **2.53x** | **2.53x** |

**~2.5x on whole-box relations per joule at matched yield, not 4.3-5.5x.** It
still clears item 0's "beats the CPU outright" bar with margin. Caveats: one
q interval at q=120M rather than the 50M/130M/190M drift probes; relations not
deduplicated, which is immaterial on a band covering 0.004% of `lim`; N_eff
10.24 is finding 43's, not re-measured; the 270 W whole-box GPU figure is
derived (item 6) while the 183.6 W board figure was measured today; and the
single-core CPU rate comes from a 4-worker run, so it may slightly understate a
truly unloaded core.

**The rule this leaves behind: state the unit.** A relation count or a time
"per q" is meaningless in this project unless it says *prime* or *pair*, and
the factor between them is 1.528 on this polynomial and different on every
other one.

### The J sweep

Same c183, same band, `--logI 15`, geometry the only variable, two repeats
each (rel/q identical to three decimals — the pipeline is deterministic):

| J | area | ms/q | rel/q | acc/wall | GPU util | board W | J/q | **rel/J** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `2^14` | `2^29` (15e) | 98.5 | 46.47 | 0.946 | 97.5% | 183.6 | 18.1 | **2.57** |
| `2^15` | `2^30` | 200.3 | 68.49 | 0.962 | 98.5% | 182.9 | 36.6 | **1.87** |

Doubling J costs **2.03x the time for 1.47x the relations**, so rel/J falls
27%. `J = 2^14` is the better energy choice and `J = 2^15` the better
relations-per-q-range choice — the same trade the CPU shows between I14e and
I15e (2.9x time for 2.2x relations), but **worse for us**: the CPU buys big
rectangles to amortise its per-q root transform, which is 12% of its wall and
about 1% of ours. Having little per-q overhead to amortise, the GPU should
prefer *smaller* areas than the CPU does. Pick the larger J only when the q
range, not the energy, is the binding constraint.

## Finding 58 — equal-work control on the C194: 3.3x, not 4.9x, and a 14% yield hole at a 1:1 rectangle

> **PARTIALLY RETRACTED 2026-08-17 — the `15e -J 15` row is not a 1:1
> rectangle *in our coordinates* and there is no yield hole (finding 65).**
> GGNFS orders the reduced q-lattice basis longer-vector-first and we order it
> shorter-first, so its `2^15 x 2^15` square is our `2^16 x 2^14` -- the same
> region under swapped axes, with one halved and the other doubled because
> their j is non-negative where our i is signed. Compare that row against our `2^16 x 2^14`.
> The "-14.4%" compared our square against GGNFS's wide rectangle. At matched
> rectangles we are flat at 0.979-0.981 on four of them. **Everything else in
> this finding stands**, including the equal-work 3.31x time / 2.74x energy and
> the 15e and 16e rows, which are ordinary 2:1 geometries. Ignore the "do not
> deploy `-J 15`" instruction at the end of the yield section.


**Date:** 2026-08-17, RTX 5070 against `gnfs-lasieve4I15e`/`I16e` on the
NFS@Home C194 (`rlim 160M, alim 240M, lpbr 32, lpba 33, mfbr 63, mfba 95`,
algebraic special-q). Both sievers over the same q window, same `(q, rho)`
pairs, four GGNFS workers on four physical cores.

### The control, and why q = 250M

GGNFS truncates the special-q side's factor base at q (`Warning: lowering
FB_bound to ...`), which finding 55 showed is worth 2.54x on the c183 and is
worse here: at q = 20M with `alim` 240M we carry ~13.16M entries against its
~1.27M, a **10x** work difference that no like-for-like claim survives.

**Above `alim` the truncation cannot bite.** Sieving at q = 250M > 240M leaves
GGNFS on the full base — verified by the absence of any `lowering FB_bound`
line in the logs — so both sievers carry the identical factor base. That is the
first genuinely equal-work comparison this project has, and it is the owner's
suggestion, not a planned one.

### Yield: parity at 2:1, a hole at 1:1

99 pairs each side, q window `[250000000, 250002000)`:

| geometry | rectangle | GPU rel/pair | GGNFS rel/pair | ours vs theirs |
|---|---|---:|---:|---:|
| 15e | `2^15 x 2^14` (2:1) | 34.28 | 34.88 | **-1.7%** |
| 15e `-J 15` | ~~`2^15 x 2^15` (1:1)~~ **actually `2^16 x 2^14`** | 46.54 | 54.39 | ~~-14.4%~~ * |
| 16e | `2^16 x 2^15` (2:1) | 77.40 | 79.14 | **-2.2%** |

\* **That row is not a 1:1 rectangle and the -14.4% is not real** — see the
retraction above and finding 65. Our `2^15 x 2^15` is being compared with
GGNFS's `2^16 x 2^14`; at the matched rectangle we run 0.9786 of GGNFS.

Yield parity at both 2:1 geometries, independently confirming finding 57's
0.8% agreement on the c183. ~~**The 1:1 rectangle is the outlier, and the aspect
ratio is what separates it from the other two, not the area**~~ — 16e is twice
the area of `-J 15` and matches. Yield scaling by area says the same thing:
ours 2.258x against GGNFS's 2.269x going to 16e (agreement), but 1.358x against
1.560x going to `-J 15`.

**The mechanism is NOT the survivor gate, and is currently unknown.** The
obvious explanation was known defect #1 -- the largest-term norm approximation,
~2 bits off the true rectangle maximum, degrading as the rectangle grows taller
-- which would mis-scale the survivor bound. Tested directly on 2026-08-17 by
loosening the gate at both aspect ratios, since "a looser gate finds more" is
true everywhere and proves nothing on its own:

| allowance | 2:1 rel/pair | 1:1 rel/pair |
|---|---:|---:|
| derived | 34.28 | 46.54 |
| +2 bits | 34.29 | 46.56 |
| +4 bits | 34.34 | 46.61 |
| +6 bits | 34.34 (+0.2%) | 46.63 (+0.2%) |

**+6 bits recovers 0.2% at both.** The missing relations are not sitting just
outside the threshold, so the gate is not what is losing them, and item 5 gets
no credit for this. The funnel says the shortfall begins early -- doubling the
area multiplies survivors by 1.48, candidates by 1.43 and relations by 1.36 --
so it originates at or before the survivor stage rather than in
cofactorisation.

Candidates that remain, none of them tested: the byte **scale** rather than the
allowance being mis-derived for a tall rectangle (`log2(maxnorm)` moves 203.59
-> 207.45 and the scale 1.225 -> 1.200 between these two runs); a difference in
what region GGNFS's `-J 15` actually covers; or the small-prime line sieve
behaving differently when J exceeds I/2. **Do not deploy `-J 15` until this is
understood** -- 14% of the relations are unaccounted for, and the cause is not
the one that was assumed.

### Throughput: the 4.9x was the factor base

| geometry | GPU ms/pair | CPU ms/pair (N_eff 10.24) | time | GPU rel/s | CPU rel/s | rel/s |
|---|---:|---:|---:|---:|---:|---:|
| 15e | 109.6 | 369.0 | **3.37x** | 313 | 94.5 | **3.31x** |
| 15e `-J 15` | 211.8 | 634.5 | 3.00x | 220 | 85.7 | 2.57x |
| 16e | 460.0 | 1133.2 | 2.46x | 168 | 69.8 | 2.41x |

The same measurement at **q = 20M**, where GGNFS was on a base 10x smaller,
reads 4.91x on rel/s — decomposing as 2.01x on time and 2.44x on yield, the
yield half being the convention rather than the siever. **At equal work it is
3.31x**, which lands on finding 57's independently measured 3.11x for the c183.
Energy agrees too: 29.6 J/pair against 81.2 J/pair whole-box is **2.74x**,
against the c183's 2.53x — two jobs, two q, within 8%.

### The GPU's area penalty is real and larger than the CPU's

Relative to its own 15e rate, GGNFS keeps 0.91 / 0.74 of its relations/s at
`-J 15` / 16e. We keep **0.70 / 0.54**, so our margin decays from 3.31x to
2.41x across the same span. Time scaling: 1.93x / 4.20x for us against 1.72x /
3.07x for GGNFS. This is finding 57's conclusion — the CPU buys large
rectangles to amortise a per-q root transform that is 12% of its wall and ~1%
of ours — now measured at matched factor base rather than inferred.

**For a C194 on this hardware, run 15e.** Best throughput, best energy, largest
margin over the CPU, and the one alternative that looked good on area scaling
is currently paying a fixable 14% yield penalty.

Caveats. `N_eff` 10.24 is finding 43's, measured on the **c183 at I15e**; a
C194 at 16e on the full base has a much larger working set, so real 16-worker
scaling there is probably worse than 10.24, which flatters the CPU in the 16e
row. One q window of 99 pairs per configuration. The GPU whole-box 270 W is
derived (item 6) while its board figure (175.9 W at 15e) was measured here.

## Finding 59 — q-truncation is worth 10-15%, not 25%: the factor base halves but the bucket records do not

**Date:** 2026-08-17, no GPU. Offline replay over the c151 corpus in
`work/c151/msieve.dat` — 1,000,609 relations sampled (every 15th of 15.0M),
band `[15M, 33.5M]`, algebraic special-q, `alim` 33.5M, `mfba` 59.

### What truncation would and would not find

For each relation, the sq-side primes decide where it can be found. Under the
FULL base every in-band sq-side prime is a q that re-finds it. Under GGNFS's
truncation at q, everything above q stays in the cofactor, so the relation is
findable at q only while that residual still fits `mfb` and the splitter's
three-prime budget.

| | unique | raw finds | inflation |
|---|---:|---:|---:|
| full base (potential) | 1,000,609 | 1,520,319 | **1.5194** |
| truncated at q | 1,000,609 | 1,239,961 | **1.2392** |
| **actually emitted** (msieve dedup, RUNBOOK:461) | | | **1.1877** |

**Zero unique relations lost** — every relation keeps at least one valid q.
That is item 3's "yield-neutral if the band reaches `lim`" precondition
holding, not a general result: this band ends exactly at `alim`.

**The replay computes POTENTIAL re-finds, not emissions**, and the gap between
1.5194 and the 1.1877 msieve actually observed is the sieve region — a relation
is re-found at q only if it also lies inside *that* q's rectangle, which this
replay cannot model. So the duplicate prize is smaller than the 18.4% of
potential finds truncation removes, and item 3's older 1.34x assumption is too
pessimistic in the other direction.

### The prize is small because `Σ1/p` is flat, not because the base is

This is the part that changes the ranking. Truncating the c151's base from
33.5M to 15M cuts the **entry count by 53%**. Bucket records go as `Σ1/p` over
the bucketed range, which grows like `ln ln p`:

    Σ(15M)   = ln ln 15.0e6 - ln ln 16384 = 0.5322
    Σ(33.5M) = ln ln 33.5e6 - ln ln 16384 = 0.5797   ->  8.2% fewer records

**The model is validated by measurement**, not asserted: the C194 lim sweep
(finding 58's session) quadrupled both lims and moved the bucket array +12%,
against +12.6% predicted by the same formula. Entry count is the wrong axis for
sieving cost; it is the right one only for the root transform.

So truncation buys roughly half the **transform** (a few percent of the sieve),
**8-11% of fill and apply** (the bulk), and some fraction of an 18% duplicate
reduction downstream. **Call it 10-15% overall.** An earlier estimate of ~25%
in session was reasoned from entry count and was wrong in exactly the way this
finding documents.

### Which is why the full base may simply be the right choice

Against 10-15%, truncation carries a real cost the c151 cannot show. A C194
band of `[20M, ~130M]` against `alim` 240M does **not** reach `lim`, so
relations whose largest sq-side prime lies above the band survive only if their
residual fits `mfba 95`, and the rest are lost outright. The favourable case
measured here is the exception rather than the rule.

**Item 3 is therefore re-ranked below item 11**, where the milliseconds
actually are. The owner's own observation drove this measurement: duplicates in
practice have been much lower than the documentation assumed, which is
confirmed at 1.19x actual against a 1.34x assumption.

Item 7 is unaffected and remains separable — our gate admits ~46% more
cofactors than GGNFS for comparable yield, which is pure downstream cost
whatever the factor-base convention is.

## Finding 60 — item 11 answered: apply is issue-limited, not memory-limited, and there is no cheap win in it

**Date:** 2026-08-17, RTX 5070, C194 at its deployment geometry (`--logI 15`,
`J` default 16384, full base to `alim` 240M, `--reps 100`). Item 11 asked
whether a second material win exists in apply or whether the stage is at its
memory-system limit. **Neither.**

### The stage, and what it is made of

| | ms |
|---|---:|
| transform + plattice | 3.45 |
| fill | 15.47 |
| **apply** | **21.94** |

Apply is **54% of the sieve chain** here, well above fill -- the reverse of the
c147 ordering that motivated findings 48-52, and a reason to stop tuning fill.

Decomposed with the harness switches that already exist:

| variant | apply ms | delta | reads as |
|---|---:|---:|---|
| baseline | 21.94 | — | |
| `--norm const` | 15.54 | **-6.40** | norm init is **29% of apply** |
| `--apply-mode plain` | 21.20 | -0.74 | smem atomics are **3.4%** |
| `--cells 8` | 21.88 | -0.06 | cell traffic is **0.3%** |

The cell result is the informative one: halving the cell array's memory traffic
moves apply by 0.3%, so the stage is not limited by the thing its shared-memory
layout was designed around.

### The profile: not memory, and not one pipe

`ncu`, one `k_apply` launch, against finding 56's fill profile on the same card:

| | k_apply | k_fill (finding 56) |
|---|---:|---:|
| DRAM throughput | **9.03%** | 12.83% |
| L2 bandwidth | **4.70%** | 52.11% |
| L1 hit rate | 64.12% | 18.91% |
| sectors per global load request | **1.43** | 8.7 B per 32 B sector |
| waves per SM | **341.33** | 1.00 |
| SM throughput | **71.00%** | 3.75% |

Opposite characters. Fill is a latency-bound scatter with the SMs idle; apply
is well-coalesced (1.43 sectors/request), massively oversubscribed (341 waves),
and the SMs are busy. **The prior-art warning that scatter tuning on this GPU
dies at the L2 transaction ceiling does not describe apply any more than it
described fill.**

But no single pipe is saturated either:

| pipe | utilisation |
|---|---:|
| LSU | 32.3% |
| ALU | 30.3% |
| XU (transcendental) | 23.2% |
| FMA | 18.5% |
| **instructions issued** | **0.70 / cycle** |

Largest stall is short-scoreboard at 1.19 per issue-active -- shared-memory
dependency from the cell accumulation. So apply is **issue-limited with a
balanced mix**: a lot of instructions, spread evenly, with no hot spot to
attack.

### Both micro-levers priced, both small

Built behind `-D` switches so the numbers are reproducible rather than
argued (`NORM_FAST_LOG2` was added for this; `NORM_CANCEL_TOL` already
existed):

| change | apply ms | saving |
|---|---:|---:|
| `__log2f` instead of `log2f` | 21.32 | **0.62 (2.8%)** |
| cancellation guard + fp64 fallback removed | 21.56 | **0.39 (1.8%)** |

**The accurate-`log2f` decision stands.** Its comment justifies the cost as
affordable for host-replay reproducibility, and at 2.8% of apply -- 1.5% of the
sieve chain -- it is. Likewise the `aabs` cancellation guard, which evaluates
the polynomial a second time on every position to catch a case the comment says
fires on under one cell in a thousand: 1.8%, which is a fair price for a norm
that is right.

### The answer

Apply is not at its memory-system limit, **and there is no second material win
available from tuning it either.** Everything identifiable is small: 2.8% from
a less accurate log, 1.8% from dropping a correctness guard, 3.4% from the
atomics, 0.3% from the cell width. A material improvement would have to cut
instructions per position or per record -- an algorithmic change to what apply
computes, not a tuning pass -- and that is a much larger piece of work than
this item was scoped as. **Item 11 closes with a direction rather than a
patch.**

Caveat: one job, one geometry, one launch profiled. The decomposition switches
are standalone-only (`--pipeline` refuses them), so these are single-side
numbers at a synthetic root, which is what they have always been.

## Finding 61 — the power cap buys 5% whole-box rel/J, and the card will not let us reach the knee

**Date:** 2026-08-17, RTX 5070, C194 at its deployment geometry (`--logI 15`,
J 16384, full base), q window `[250000000, 250002000)`, 99 (q, rho) pairs.
Power limit set from the Windows side with MSI Afterburner, as item 10 requires
— the WSL-side `nvidia-smi` can read limits but not set them.

### First, a premise correction: the sieve is not power-limited at stock

    power.limit 250 W (stock)   power.min_limit 175 W   power.max_limit 275 W

and the sieve draws **~195 W steady** at 97% utilisation. So the card sits at
**78% of its own stock limit while running flat out**, and a cap only begins to
bind below that. Item 10 assumed the 60–80% range was available and worth
15–30% of rel/J; on this card and this workload the entire bindable range is
**70–77%**, about 20 W. The cap is not the lever the item assumed it was.

### The measurement

Three runs per setting, relations byte-identical (3,394) in all six, so the
only variable is time and power:

| | stock (250 W) | **70% (175 W)** | change |
|---|---:|---:|---:|
| ms/pair | 108.27 | 110.25 | **+1.83%** |
| board watts | 195.2 | 175.7 | **−9.99%** |
| SM clock | 2917 MHz | 2902 MHz | −0.5% |
| board J/pair | 21.13 | 19.37 | **rel/J +9.1%** |
| whole-box J/pair (+105 W host) | 32.50 | 30.95 | **rel/J +5.0%** |

Run spreads were 1.2% (stock) and 0.4% (capped) with no overlap between the
sets, so the 1.83% throughput cost is real rather than noise.

**Nine percent of the power for half a percent of the clock.** That is the
signature of operating past the efficiency knee — voltage scales superlinearly
with frequency, so the last few MHz cost disproportionate power — and it is
why the trade is favourable at all.

### A note on the host constant, which every whole-box figure here rests on

Item 6 measures it at ~105 W idle and ~115 W with the sieve's own core, and
finding 58's derived 270 W whole-box against a 175.9 W board implies ~94 W. The
tables here use **105 W**. At 115 W the cap result is +4.8% rather than +5.0%
and **the undervolt is +13.8% rather than +14.6%**; at 94 W the undervolt is
+15.1%. The headline therefore moves about a point across the range of
defensible values -- which changes no conclusion, but the constant should travel
with the number. A wall meter on both configurations would remove the
assumption and is the honest way to finish item 10.

### The board sensor overstates the win, exactly as item 10 warned

Board says **+9.1%**, the whole box says **+5.0%** — a factor of 1.8. The
mechanism is item 10's: capping lengthens the run by 1.8% and the ~105 W host
constant is paid for that extra time too. The item's worked example predicted a
4× discrepancy for a larger cap; this is a small cap, and the direction and
mechanism are confirmed. **Grading a power sweep on the board sensor would pick
a cap that is too low.** The figure is robust to the host constant: using item
6's ~115 W loaded value instead of ~105 W gives +4.8% rather than +5.0%.

### The cap is a floor, not a knee — so the undervolt was measured too

175 W is `power.min_limit`, and the curve is **still improving** where the card
stops letting us follow it. The lever that reaches further is an **undervolt**:
a V/F curve holding clocks near stock at lower voltage attacks the voltage term
directly, rather than waiting for a cap to throttle clocks reactively.

Operating point at stock, read from Afterburner under sustained load:
**2910 MHz at 1080 mV**. Curve edited to ~2900 MHz at **950 mV** and flattened
to the right so the card cannot boost past it.

**Correctness first.** An unstable undervolt does not crash, it computes wrong
answers, which for a sieve means plausible-looking corrupt relations. Both
gates were run before any timing: `cofcheck.sh`'s 30 exact relation counts, and
the 1500-q c147 band, which came back byte-identical (`47bb45b9...`). Only then
the measurement, three runs, 3,394 relations in every one:

| | stock | 70% cap | **950 mV** |
|---|---:|---:|---:|
| ms/pair | 108.27 | 110.25 | **115.55** (+6.7%) |
| board watts | 195.2 | 175.7 | **140.5** (−28.0%) |
| temperature | 69 °C | — | **50–56 °C** |
| board J/pair | 21.13 | 19.37 | **16.24** |
| board rel/J | — | +9.1% | **+30.2%** |
| **whole-box rel/J** | — | +5.0% | **+14.6%** |

**The undervolt is worth about 3x the power cap** on the metric of record, and
it confirms the diagnosis: the card was never power-limited, it was sitting far
past its efficiency knee. 28% of the board power was buying 6.7% of throughput.

Two second-order effects. The clock settled at ~2850 rather than the 2900 asked
for, so a small clock reduction rides along with the voltage drop and the 6.7%
is not purely voltage. And the card now runs at **50–56 °C against 69 °C**,
which matters over days: at 140 W it will hold clocks indefinitely, where the
stock point was still warming when measured. The sustained cost over a real run
is probably below 6.7%.

**Where the knee actually is remains unmeasured** — 900 mV was not tried. What
is established is that 950 mV is well past it and comfortably stable on this
card.

### The item-0 verdict at the floor

Applying this to finding 58's equal-work comparison at 15e:

| | stock | at 175 W | **at 950 mV** |
|---|---:|---:|---:|
| time advantage over the CPU box | 3.31× | 3.25× | **3.10×** |
| whole-box energy advantage | 2.74× | 2.88× | **3.14×** |

Caveat: whole-box GPU power is derived (board + item 6's host constant), not a
wall-meter reading; only the board figure was measured here.

## Finding 62 — item 7's surplus does not exist at shipping defaults: cofactor volume matches GGNFS within 0.4% on two jobs

**Date:** 2026-08-17, no code change. Item 7 records that we submit **1,426
cofactors per special-q against GGNFS's 978** — ~46% more, "almost all
unproductive — time, not relations" — and treats that as an open cost. Measured
directly, at each siever's own default gate, on both jobs:

| job / geometry | q window | pairs | our cofactors/pair | GGNFS COF/pair | ratio |
|---|---|---:|---:|---:|---:|
| c183, I14e (item 7's own config) | `[120000000, 120001000)` | 67 | **784.85** | **781.7** | **1.004** |
| C194, 15e | `[250000000, 250002000)` | 99 | **1594.12** | **1594.8** | **1.000** |

Relations from those submissions, same windows: 21.04 against 20.85 per pair on
the c183 (**we are 0.9% ahead**) and 34.28 against 34.88 on the C194 (**1.7%
behind**). Opposite signs on the two jobs, so there is no systematic yield
deficit either — the residual is job-level noise, not a gate defect.

**This is not a refutation of item 7's measurement.** That comparison was
explicitly *at matched lambda* — both sievers pinned to the same nominal bit
threshold — and it found that a given bit value means different things to the
two gates: GGNFS drops 17.3% of its yield going 91.8 → 87.5 bits where we drop
0.07% going to 88.0. That asymmetry may well be real. What this finding shows
is narrower and more useful: **at the defaults we actually ship, the asymmetry
does not produce a surplus.** Our derived allowance (`mfb` + ~1.5 bits from our
own quantisation) lands on the same cofactor volume GGNFS's lambda rule does,
on two jobs three digits apart in size and at two different geometries.

That also answers item 7's operational worry — *"until the cause is known the
bound can only be set by sweeping, never derived for a new job"*. The bound
**is** derived, and the derivation now has two independent checks against an
external siever showing it lands within half a percent. A new job does not need
a sweep.

Method note: GGNFS's `COF: N tests` is its count of cofactorisation attempts
and matches the last stage of its own `reports:` funnel (1,601/pair against
1,594.8 on the C194), so it is the same quantity as our band summary's
`cofactorisation candidates/q` — records handed to the splitter. The c183 rows
carry finding 55's usual caveat that GGNFS truncates its base at q while we run
the full one, worth 1.11x at q=120M; the C194 rows are at q > `alim` and so
carry no such difference.

## Finding 63 — RETRACTED: the 1:1 deficit is in the gate, not the sieve: four hypotheses eliminated

> **RETRACTED 2026-08-17 by finding 65. The deficit it analyses does not
> exist**: it compared our `2^15 x 2^15` against GGNFS's `-J 15`, which is
> `2^16 x 2^14`. Do not quote the 12.7% cofactor gap, the 1.480-against-1.624
> survivor scaling, or the conclusion that the scale/bound derivation is at
> fault — all three are that mismatch. The `log2(maxnorm)` rise of 3.86 bits
> is real and is simply what a taller rectangle costs.
>
> **The four eliminations below are sound and worth keeping**: the FK walk
> passes at every aspect ratio, the enumeration loses nothing in the shared
> region (finding 65 strengthens this to *all 6,743* of our 2:1 relations
> reappearing at `2^15 x 2^15`), the splitter gains equally at both, and the
> survivor list cap is not on the yield path. So is the side result that
> `--cof-rounds 2 --cof-budget 65536` is well chosen.


**Date:** 2026-08-17. Finding 58 measured our yield 14.4% below GGNFS at
`2^15 x 2^15` while both 2:1 geometries sit at parity, and finding 58's own
correction recorded the survivor gate as ruled out and the cause unknown. This
narrows it to one stage and kills four candidate mechanisms.

### Where it is NOT

**Not the Franke-Kleinjung walk.** The shipped gate calls
`verify_walk(8, 128, 24)` — I=256, J=128, i.e. **2:1, the only shape it has
ever been run at**, and every geometry the project uses is 2:1. Called directly
at other aspect ratios it passes everywhere:

    logI 8  J 64   4:1  PASS      logI 9  J 256  2:1  PASS
    logI 8  J 128  2:1  PASS      logI 9  J 512  1:1  PASS
    logI 8  J 256  1:1  PASS      logI 10 J 512  2:1  PASS
    logI 8  J 512  1:2  PASS      logI 10 J 1024 1:1  PASS

The walk enumerates the rectangle exactly at 1:1 and even at 1:2. Worth keeping
as a permanent case rather than a one-off.

**Not the sieve's ENUMERATION** -- and the distinction matters, because the
survivor *threshold* is implicated. Survivor bitmaps dumped at both geometries
and histogrammed by row: over the 16,384 rows the two runs share, densities are
identical to **0.3%** (2798.3 vs 2798.4 at j<1024; 1077.7 vs 1080.4 at
j~16000). The rows the 1:1 run adds are simply less productive -- 1035 falling
to 778 per row -- as larger norms at larger j predict. So nothing is lost in the
region the two geometries have in common, and no position goes unvisited.

That is *not* the same as the survivor count being right. Two different
measurements are in play and they must not be compared to each other:

| quantity | 2:1 | 1:1 | scaling |
|---|---:|---:|---:|
| pipeline two-sided survivors, real q | 71,754.8 | 106,172.1 | **1.480** |
| standalone one-sided survivors, synthetic q | 27.12M | 41.79M | 1.541 |
| GGNFS funnel stage 6 (comparable to the first row) | 47,356 | 76,900 | **1.624** |

**Our two-sided survivors scale 1.480 against GGNFS's 1.624**, so the gate is
already behind at the survivor stage -- consistent with the cofactor gap below
and inconsistent with an earlier draft of this finding, which compared our
one-sided 1.541 against GGNFS's *relation* scaling of 1.559 and read it as
agreement. Those are different quantities.

**Not the splitter giving up on harder cofactors.** Raising the budget from
`rounds 2 / 65536` to `rounds 4 / 262144` gains **+1.9% at 2:1 and +2.2% at
1:1** — the same at both, and saturating (rounds 6 adds nothing).

**Not the survivor list cap.** `maxsurv` is `1<<22`, but the pipeline
intersects through the uncapped **bitmap**; the list is not on the yield path.
The standalone's "list truncated" notice is its own display cap.

> **2026-08-24:** the conclusion outlived the mechanism. Finding 73 deleted the
> survivor list outright — it was write-only in both callers — so there is no
> longer a cap or a truncation notice to rule out. `maxsurv` now names only the
> intersect/compaction cap.

### Where it is

Both funnels at q=250M, per (q, rho) pair:

| | GGNFS 2:1 | GGNFS 1:1 | GGNFS scaling | **our scaling** |
|---|---:|---:|---:|---:|
| cofactors submitted | 1,594.8 | 2,606.9 | **1.635** | **1.428** |
| relations | 34.88 | 54.39 | 1.559 | 1.358 |
| relations per cofactor | 0.02187 | 0.02086 | −4.6% | **−4.9%** |

The deficit decomposes exactly:

    relations ratio 0.8557  =  cofactor ratio 0.8732  x  rel-per-cofactor ratio 0.9799

so **we admit 12.7% fewer cofactors at 1:1** (equivalently, GGNFS admits 14.5%
more -- an earlier draft quoted that figure as our shortfall, which inverts the
ratio) and get **2.0% less out of each one**. The second term is *not* specific
to 1:1: at 2:1 it is 1.7%, so a roughly constant ~2% downstream gap exists at
both aspect ratios and is the same small residual finding 62 already recorded.

**What is specific to 1:1 is the cofactor count**, which matches GGNFS to 0.04%
at 2:1 (finding 62) and falls 12.7% short at 1:1. That, not the downstream
term, is the additional deficit and where the work is.

So it is the gate -- as a **scaling failure, not an offset**, which is exactly
why finding 58's uniform allowance sweep could not move it: +6 bits recovers
0.2% because the threshold is not uniformly too tight. The survivor scaling
(1.480 against 1.624) and the cofactor scaling (1.428 against 1.635) agree that
the loss is already present when survivors are counted.

**The remaining suspect is the scale/bound derivation's response to a wider
norm range.** Between these two runs `log2(maxnorm)` rises 203.59 -> 207.45
(+3.86 bits) while the derived scale *falls* 1.225 -> 1.200 and the bound falls
119 -> 117. A norm range that grows while the byte threshold shrinks is the
shape of the observed loss; whether the derivation is wrong or merely coarse is
not established. That is where the next session should start —
`sieve_bound_checked` and `sieve_allowance`, at fixed q, sweeping only J.

### Side result: the splitter defaults are right

The +1.9% above costs **+68.8% of wall** (116.78 -> 197.09 ms/pair), so
`--cof-rounds 2 --cof-budget 65536` is well chosen and should not change. Note
also that with the larger budget our 2:1 yield reaches **34.94 against GGNFS's
34.88** — exact parity, confirming finding 62 is not an artifact of splitter
settings.

## Finding 64 — VRAM sizing: area buys the bucket array, lim buys the factor base, and they cross at area ~ 6.7 x lim

**Date:** 2026-08-17, C194, measured from the startup allocation report. Recorded
because sizing a job to a card was previously guesswork and the entry count --
the obvious axis -- is the wrong one for the largest allocation.

### Geometry, lims fixed at rlim 160M / alim 240M

| config | rectangle | area | bucket | steady state | ms/pair | rel/pair | rel/ms |
|---|---|---|---:|---:|---:|---:|---:|
| 15e | `2^15 x 2^14` | `2^29` | 1.46 GB | 3.63 GB | 113.6 | 65.3 | **0.575** |
| 15e `--J 24576` | `2^15 x 24576` | 1.5x`2^29` | 2.18 GB | 4.47 GB | 159.8 | 82.4 | 0.516 |
| 15e `--J 32768` | `2^15 x 2^15` (square) | `2^30` | 2.91 GB | 5.32 GB | 224.3 | 96.1 | 0.428 |
| **`--logI 16 --J 16384`** | **`2^16 x 2^14` (wide)** | **`2^30`** | **2.60 GB** | **4.98 GB** | 228.6 | 53.4 | 0.234 |
| 16e | `2^16 x 2^15` | `2^31` | 5.20 GB | 8.06 GB | 477.1 | 154.8 | 0.324 |

The `2^16 x 2^14` row was **added 2026-08-18** — it was the one cell finding 65
exercised for yield and this finding never sized.

**Its `rel/pair` and `rel/ms` do not compose with rows 1-3.** Those were taken
on a lower q band; the new row is on finding 65's `[250000000, 250004000]`, and
yield falls with q. The size of that effect is not small: the *same* square
re-measured on finding 65's window gives **45.88 rel/pair against the 96.1 in
its own row above**. Read down the memory columns freely -- they depend on
geometry and lims only, not on the band -- but compare relations only within a
band. For the `2^30` pair that means this block, measured back to back:

| rectangle | bucket | steady state | ms/pair | rel/pair | rel/ms |
|---|---:|---:|---:|---:|---:|
| `2^15 x 2^15` (square) | 2.91 GB | 5.28 GB | 223.0 | 45.88 | 0.206 |
| **`2^16 x 2^14` (wide)** | **2.60 GB** | **4.98 GB** | 228.6 | **53.38** | **0.234** |

**+16.3% rel/pair and +13.5% rel/ms**, at -0.30 GB (5.28 against 4.98, both
from this block; the 2026-08-17 row above records the square at 5.32). `ms/pair` here is wall
clock, and the box was under a 14-thread `msieve` throughout, so the wide arm's
+2.5% wall is host contention rather than a real cost: finding 65 times the same
two shapes on the device at **-0.3%**, which puts the honest throughput gain at
**+16.7% relations per device-second** (260.6 against 223.4). Per finding 53,
prefer the device figure whenever the host is loaded.

**At equal area the wide rectangle is also the cheaper one: 4.98 GB against the
square's 5.32 GB, and 2.60 GB of bucket array against 2.91 GB (-10.7%).** So
buying area with `I` rather than `J` wins on *both* axes at once — +16.3%
relations on a matched window (finding 65: 10,676 against 9,176, both
reproduced exactly on the 2026-08-18 re-run) for -0.30 GB of VRAM — and for the
same reason. Raising `logI` raises `bkthresh` with it, which simultaneously shortens the bucketed
prime range and, because `i` multiplies the shorter lattice vector, grows the
norms at half the rate (+1.86 bits against +3.86). There is no trade to make
here; the wide shape is simply better.

The factor base is **not** where the difference lives, which is worth stating
because `--maxbits` does have to move with `logI`: the m16 file carries 26 more
ideals than the m15 file (13,160,671 against 13,160,645 — 211 prime-powers
against 185), i.e. under a kilobyte at 25.4 B per entry. Both `2^30` rows load
0.83 GB of factor bases + bitmaps. **The entire geometry delta is the bucket
array.**

Re-run conditions, for the two 2026-08-18 rows: both started from an identical
10.76 GB free, so they are comparable to each other and to the 2026-08-17 rows.
The square re-measured at **5.28 GB** against the 5.32 GB recorded above -- a
0.04 GB spread that is the practical accuracy of this column, not drift.

**A desktop Windows box does not contaminate this measurement the way it looks
like it should**, and it is worth knowing before someone discards a run for the
wrong reason. `nvidia-smi` reported ~5.1 GB in use before both runs, all of it
the Windows desktop (dwm.exe alone holds 1.35 GB, plus browsers, VS Code and
the terminal; a WSL process such as msieve would not appear in Task Manager's
list at all, so do not read its absence as evidence). Under WDDM those
allocations are **evictable**: the moment `bench` asked for the bucket array,
Windows demoted them to system RAM and the card read 10.76 GB free, snapping
back to ~5 GB after exit. So `total - free` at sizing time reflects a nearly
cleared card regardless of desktop load -- which is also why 16e's 8.06 GB fit
here on 2026-08-17. **Do not close applications before sizing a job, and do not
subtract a desktop baseline; both would be corrections for an effect that does
not occur.** What this does *not* license is running against another CUDA
process -- those allocations are pinned, not evictable.

16e fits a 12 GB card with 3.9 GB spare, so the `2^31` area limit is reachable
on a 5070 for a C194. `J` need not be a power of two -- the only constraints
are `I*J <= 2^31` and `I*J` a multiple of the region size, so at `logI 15` any
integer J works and the 1.5x row above is a real measurement, not an estimate.

**The bucket array is linear in J at fixed I, and SUB-linear when I grows.**
1.46 -> 2.91 GB is exactly 2x (J doubled), but 2.91 -> 5.20 is 1.79x, because
that step raises `logI` and **`bkthresh` defaults to `1 << logI`**
(`bench_main.cu`). Moving the bucketed range's start from 2^15 to 2^16 drops
`Σ1/p` from 0.6184 to 0.5539, a factor 0.896 -- and 5.82 x 0.896 = 5.21 against
the measured 5.20. Anything that reasons about "area" without separating the
two ways of growing it will be wrong by ~11% on the 15e-versus-16e decision.

**The 2026-08-18 row isolates that factor directly**, because it holds area
fixed at `2^30` and moves `bkthresh` alone: 2.91 -> 2.60 GB is **0.893** against
the 0.896 predicted from `Σ1/p`. The `bkthresh` effect is therefore confirmed on
its own, not just inferred from a step that changes two things at once.

### Factor-base bounds, area fixed at `2^29`

| lims r/a | fb1 entries | fb0 entries | bucket | steady | ms/pair | rel/pair | rel/ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 80/120M | 6.84M | 4.67M | 1.37 GB | 3.27 GB | 105.8 | 54.6 | 0.516 |
| **160/240M** | 13.16M | 8.97M | 1.46 GB | 3.63 GB | 113.6 | 65.3 | **0.575** |
| 320/480M | 25.36M | 17.27M | 1.54 GB | 4.23 GB | 128.6 | 73.8 | 0.574 |

**Doubling both lims doubles the factor base (1.93x) but moves the bucket array
+5.5%**, because bucket records go as `Σ1/p ~ ln ln p`. Entry count is the
right axis only for the root transform.

**The job's own lims look well chosen**: doubling is break-even on throughput
(+13% relations for +13% time) and costs 0.6 GB, while halving loses 16.5% of
the relations to save 6.9% of the time. The sensitivity is asymmetric -- **too
small hurts, too large is merely wasteful** -- so an uncertain lim should err
upward.

### The model, and which knob costs what

Separating the two terms in the rows above gives **25.4 B per factor-base
entry** consistently (25.5 and 25.4 across the two steps), with entries
`~ pi(lim)` on *both* sides: 13.16M against `pi(240M) ~ 13.15M`, and 8.97M
against `pi(160M) ~ 8.97M`.

An earlier draft wrote this as `1.06 x pi(lim)` and read the 1.06 as evidence
that "the algebraic side's extra roots and the rational side's simplicity
cancel". **That was wrong twice.** The 1.06 is entries divided by the crude
`lim/ln(lim)`, i.e. the second-order term of the prime counting function, and
it appears on both sides because `pi` does -- it carries no information about
roots at all. The underlying claim survives on other grounds (a polynomial of
any degree averages one root mod p, so both sides have ~one entry per prime),
but these numbers are not what establishes it. Use `pi(lim)`, or
`1.06 x lim/ln(lim)` if a closed form is wanted.

    VRAM ~ bucket(max side, ~flat in lim) + 25.4 B x (entries_r + entries_a)
           + 3 x area/8 + fixed

The bucket array is sized by `max(side0, side1)` records and **not** their sum
(`pipeline.cuh`: `est = est1 > est0 ? est1 : est0`), because the two sides run
sequentially through one shared allocation. So raising the *smaller* side's lim
costs factor-base bytes only, until it overtakes the other side.

Differentiating both terms, the `ln L` cancels:

    bucket growth / entry growth  ~  area / (6.7 x lim)

At `2^29` and lim 240M that is **0.33** -- raising `alim` costs about 1.33x
what raising `rlim` by the same amount does -- and it matches the measured 0.33
over the 120M -> 240M step. The useful form is the crossover: the two terms are
equal at **area ~ 6.7 x lim**, so at 15e a C194 is comfortably factor-base
dominated and at 16e it has crossed into bucket dominated. Area is the
expensive axis; lim only becomes expensive once the rectangle is already large.

**Caveat on that crossover: it is derived treating the bucket term as linear in
area, which is true only when the area grows through J.** When it grows through
`logI`, `bkthresh` rises with it and the bucket term grows ~11% less than
linearly (see above), so the crossover sits correspondingly higher. Use it as
an axis-comparison rule at fixed `logI`, not as a prediction across sievers.

## Finding 65 — the 1:1 deficit was never real: `I15e -J 15` is our `2^16 x 2^14`, and at matched rectangles we are flat at 98% on four of them

**Date:** 2026-08-17, RTX 5070 against `gnfs-lasieve4I14e`/`I15e` on the
NFS@Home C194, algebraic special-q, q window `[250000000, 250004000)` — 130
special-q / 200 `(q, rho)` pairs, both sievers over the identical window.

**This retracts finding 58's 14% hole and the whole of finding 63.** They
compared two different rectangles.

### `gnfs-lasieve4I15e -J 15` does not sieve `2^15 x 2^15`

The help text says `-J   specify J bits`, and every measurement in findings 58
and 63 took that to mean the j-height doubles to `2^15`. It does not. Recovered
from the relations themselves — invert the q-lattice, `(i,j) = M^-1 (a,b)`,
with `M` built by the same `qlat_build` the sieve uses — each run reports the
rectangle it *actually* covered:

| run | i range | j range | actual rectangle |
|---|---:|---:|---|
| ours `--logI 15 --J 16384` | ±16383 | 1 .. 16381 | `2^15 x 2^14` |
| `I15e`, no `-J` | ±16383 | 1 .. 16381 | `2^15 x 2^14` (identical) |
| ours `--logI 15 --J 32768` | ±16383 | 1 .. 32761 | `2^15 x 2^15` |
| **`I15e -J 15`** | **±32738** | **4 .. 16380** | **`2^16 x 2^14`** |

So in **our** coordinates `-J n` widens the i axis and leaves j alone.
Confirmed independently on the other binary: `I14e` with no flag gives
`2^14 x 2^13`, and `I14e -J 14` gives `2^15 x 2^13`. Same area as the square in
each case, different shape.

**This does not mean GGNFS's flag misbehaves — see the basis-ordering section
below, added 2026-08-18.** `-J n` really does set `J_bits` exactly as
documented; the axis swap is ours. The operational rule is unaffected: **compare
`I15e -J 15` against our `2^16 x 2^14`.**

The mapping is not an inference. Applied to our own 2:1 output it places
**6,743 of 6,743 relations inside the rectangle, none outside and none
unmapped**, which is what makes it usable as an instrument. Two details are
load-bearing: a relation reported with `b > 0` may map to the antipodal point,
so `j < 0` is negated; and the lattice must be cached per `(q, rho)`, not per
`q` — this polynomial averages 1.53 roots per prime, and caching on `q` alone
hands a third of the relations the wrong basis.

### At matched rectangles there is no aspect-ratio effect at all

Set difference on `(a,b)`, restricted to the special-q both sievers covered:

| rectangle | ours | GGNFS | ours/theirs | recall |
|---|---:|---:|---:|---:|
| `2^14 x 2^13` (`I14e` default) | 2,864 | 2,924 | **0.9795** | 0.9771 |
| `2^15 x 2^13` (`I14e -J 14`) | 4,531 | 4,621 | **0.9805** | 0.9794 |
| `2^15 x 2^14` (`I15e` default) | 6,743 | 6,883 | **0.9797** | 0.9784 |
| `2^16 x 2^14` (`I15e -J 15`) | 10,676 | 10,910 | **0.9786** | 0.9772 |

**Flat at 0.979–0.981 across a 4x span of area and every aspect ratio from 2:1
to 4:1.** That constant ~2% is the same residual finding 62 recorded and
finding 63 correctly identified as geometry-independent; it is all that was
ever there. Per-`j` the share is flat too — at `2^15 x 2^14` it runs 0.976,
0.999, 0.985, 0.987, 0.977, 0.982, 0.983, 0.992 over eighths of the rectangle.

Our own sieve is internally exact across the change finding 63 was built on:
**our `2^15 x 2^15` run contains every one of our `2^15 x 2^14` relations,
6,743 of 6,743**, and all 2,433 extras lie in `j > 16384`. Nothing was lost in
the shared region, so the scale moving 1.225 -> 1.200 costs nothing.

#### Why the shapes differ: basis ordering, not a GGNFS bug (2026-08-18)

**Corrected.** An earlier version of this section read the `lasieve5_64` source,
found that `-J n` sets `J_bits` in the obvious way, could not reconcile that
with the recovered rectangles, and concluded the binary must diverge from its
source somewhere in `asm/lasched*`/`medsched*`. **That was wrong.** GGNFS parses
`-J` exactly as documented and sieves exactly the square its source says. The
entire discrepancy is a coordinate convention, and it is ours as much as theirs.

**The two sievers order the reduced q-lattice basis oppositely.**

- `redu2.c:61` branches on `if(a0sq > a1sq)` and puts the vector with the
  *larger* skewed norm into slot 0. GGNFS is **longer-vector-first**, so its `i`
  multiplies the long vector and its `j` the short one.
- `bench/poly.c:818` states the opposite as an explicit invariant -- "`(a0,a1)`
  is the SHORTER vector" -- so our `i` multiplies the short vector and our `j`
  the long one.

Slot for slot, their `i` is our `j` and their `j` is our `i`. Folding in the
antipodal canonicalisation `(a,b) ~ (-a,-b)` that keeps the second coordinate
positive gives

    our i = +/- (their j)        our j = |their i|

which turns their *non-negative* j range of length `n_J` into our *signed* i
range of width `2*n_J`, and their signed i range of width `n_I` into our
non-negative j range of height `n_I/2`. In closed form:

    our rectangle  =  2^(J_bits+1)  x  2^(I_bits-1)

**That reproduces every geometry this project has measured, with no free
parameters:**

| GGNFS invocation | its own region | predicted ours | measured ours |
|---|---|---|---|
| `I14e`, no `-J` | `2^14 x 2^13` | `2^14 x 2^13` | `2^14 x 2^13` |
| `I14e -J 14` | `2^14 x 2^14` (square) | `2^15 x 2^13` | `2^15 x 2^13` |
| `I15e`, no `-J` | `2^15 x 2^14` | `2^15 x 2^14` | `2^15 x 2^14` |
| `I15e -J 15` | `2^15 x 2^15` (square) | `2^16 x 2^14` | `2^16 x 2^14` |
| `I16e`, no `-J` | `2^16 x 2^15` | `2^16 x 2^15` | `2^16 x 2^15` |

**Five for five**, against an "asm override" story that explained none of them.
The rule also predicts `I16e -J 16` -> `2^17 x 2^15`; that one is untested here,
being past our `I*J <= 2^31` limit, so it is not counted above. The binaries were also confirmed byte-identical to the
`lasieve5_64` build (`cmp`, all three of `I14e`/`I15e`/`I16e`), so the source
above is the source that ran.

**Where the earlier argument went wrong.** It claimed a basis difference "could
not manufacture 97.7% agreement on the `(a,b)` pairs, which are
basis-independent". The `(a,b)` agreement is real and basis-independent -- but
that is the point: *both sievers swept the same region*, so of course the pairs
agree. What differs is only the axes each one names it in. The companion
"73.5% maximum intersection" bound made the same mistake in sharper form: it
computed the overlap of our `2^16 x 2^14` with a square **drawn on our axes**,
when the square in question is drawn on GGNFS's. Two clean rectangles related by
a coordinate swap also explain the recovered bounds landing on exact powers of
two, which had been offered as evidence *against* a basis explanation -- a swap
is unimodular *and* axis-preserving, unlike the general shear that objection
assumed.

**A useful consequence.** Since their `j` and our `i` both multiply the short
vector, **GGNFS's `-J` and our `--logI` are the same knob**: both extend the
cheap axis, the one costing 1.86 bits of `log2(maxnorm)` per doubling rather
than 3.86. The long-standing GGNFS practice of raising `-J` for more area and
this project's "buy area with I, not J" are one result in two coordinate
systems, not two competing pieces of advice.

**Nothing downstream moves.** `I15e -J 15` is still to be compared against our
`2^16 x 2^14`; the 0.979-0.981 parity stands; and finding 64's wide-versus-square
result never routed through GGNFS's flag at all, being measured end to end on
our own siever in a single basis.

### The real result: buy area with I, not J

The comparison findings 58 and 63 were reaching for — equal area, different
shape — is worth having, and it is large. Ours only, same window, same job,
same factor base and `--maxbits` on both arms:

| area | rectangle | relations | device ms/q | rel per device-second |
|---|---|---:|---:|---:|
| `2^27` | `2^14 x 2^13` | 2,864 | 36.66 | 390.6 |
| `2^28` | `2^14 x 2^14` (square) | 4,056 | 60.36 | 336.0 |
| `2^28` | **`2^15 x 2^13` (wide)** | **4,531** | 58.51 | **387.2** |
| `2^29` | `2^15 x 2^14` | 6,743 | 104.11 | 323.9 |
| `2^30` | `2^15 x 2^15` (square) | 9,176 | 205.39 | 223.4 |
| `2^30` | **`2^16 x 2^14` (wide)** | **10,671** | 204.74 | **260.6** |

At equal area the wide rectangle wins **+11.7% relations at `2^28`** and
**+16.3% at `2^30`**, for **-0.3% device time** — so **+15.2% and +16.7% on
relations per device-second**. Device time, not wall: the box was running the
owner's `msieve` at ~4 cores, and per finding 53 wall clock is not comparable
under load while `cudaEvent` totals stay flat within 1%.

**The mechanism is printed by our own banner.** Doubling the area costs norm
bits, but not the same number of them either way:

    2^15 x 2^14   log2(maxnorm) 203.59
    2^16 x 2^14   log2(maxnorm) 205.45   (+1.86 bits, doubling I)
    2^15 x 2^15   log2(maxnorm) 207.45   (+3.86 bits, doubling J)

`i` multiplies the **shorter** vector of the reduced q-lattice and `j` the
longer one (`qlat_build` keeps `(a0,a1)` short by construction, and
`norm_setup` forms `A = I/2*|a0| + J*|b0|`). Extending `i` therefore grows the
norms at roughly half the rate extending `j` does, for the same extra positions.
Finding 63 saw the +3.86 bits and read it as a gate pathology; it is the honest
signal that a taller rectangle has genuinely bigger norms.

### What this costs and what it does not

Finding 58's throughput and energy conclusions **survive**: its `15e` row is the
default geometry and is unaffected, and its 16e row (`2^16 x 2^15`, no `-J`) is
also a real 2:1 rectangle. The equal-work 3.31x time / 2.74x energy stands. What
changes is that the `-J 15` row was labelled `2^15 x 2^15` when it was
`2^16 x 2^14`, and its "-14.4%" was our square against their wide rectangle.

**"Do not deploy `-J 15` until this is understood" is withdrawn**, and so is
"deploy 2:1 geometries only". The correct rule is about which axis buys the
area, not about the ratio: at fixed area prefer the wider `I`.

### Why the earlier eliminations pointed the wrong way

Finding 63 was right in every particular and wrong in aggregate, which is worth
recording. It eliminated the FK walk, the enumeration, the splitter and the
list cap — all correctly. It then had one suspect left and no reason to doubt
the comparison itself, so a real 3.86-bit norm difference became evidence for a
scale-derivation defect. The check that would have caught it is the one this
finding opens with: **ask each siever's output which rectangle it covered**,
rather than trusting the flag. It is now `bench/relgeom.py`:

    ./relgeom.py --band 250000000:250004000 --skew 38612712.90 \
                 extent out.mine out.theirs          # what did each cover?
    ./relgeom.py --band ... --skew ... \
                 compare out.mine out.theirs J logI  # set difference by j

`extent` is the one to run before any cross-siever yield claim; `compare`
reports a nonzero "OUTSIDE the stated rectangle" count precisely when the
geometry is wrong, which is the signature this finding chased for two sessions.

The band and skew are required arguments, not defaults, because both fail
silently: a wrong band filters every relation's candidate list to empty and the
tool would otherwise print a clean-looking zero, and a wrong skew produces a
different lattice reduction and therefore different `(i,j)`. The reported I is
rounded up from observed coverage and is a **lower** bound — the widest
column may simply have yielded nothing on a short window.

## Finding 66 — 10-13% of survivors are quantisation noise, and removing them buys nothing: survivors are not a cost driver

**Date:** 2026-08-17, C194, same window as finding 65.

`las_scale` (`poly.c:584`) returns `floor(254/maxlog2 * 40)/40` — the 255-cell
rule las needs because *its* cell is a byte. Ours is 16 bits with `CINIT` 4096
(`bench_kernels.cu:360`, and the comment there already says the scale "becomes
a free parameter rather than a constraint"). The derivation was never updated,
so the sieve runs at ~1/16 of the resolution the cell can carry: one byte unit
is `1/scale` = 0.82 bits at the derived 1.225.

Holding the gate fixed **in bits** and raising only the scale:

| geometry | scale | survivors/q | relations |
|---|---:|---:|---:|
| `2^15 x 2^14` | 1.225 (derived) | 71,579.6 | 6,743 |
| `2^15 x 2^14` | 4.000 | **64,241.0** (-10.3%) | 6,740 (-0.04%) |
| `2^15 x 2^15` | 1.200 (derived) | 105,628.1 | 9,176 |
| `2^15 x 2^15` | 4.000 | **92,141.4** (-12.8%) | 9,171 (-0.05%) |

So 10-13% of the survivor list is positions admitted purely by rounding error,
and they yield nothing.

### And it is worth nothing, measured on an idle box

The obvious inference -- survivors drive resieve, trial division and the
cofactor queue, so cutting 10% of them should show up -- is **wrong**, and the
A/B says so unambiguously. C194, `2^15 x 2^14`, 200 pairs, alternated to guard
against drift, host load 0.5:

| | wall ms/q | survivors/q | candidates/q | relations |
|---|---:|---:|---:|---:|
| derived scale 1.225 | 112.24, 112.64 | 71,579.6 | 1590.68 | 6,743 |
| scale 4.000 | 111.17, 113.68 | 64,241.0 | **1590.08** | 6,740 |

**112.44 against 112.43 ms/q.** Nothing, against a run-to-run spread of ~1.3%.

The stage breakdown says why, and the answer is that **the survivor count is
not a cost driver at all**:

| stage | derived | scale 4.0 |
|---|---:|---:|
| fill | 31.375 | 31.204 |
| apply | 38.619 | 38.149 |
| resieve + scatter | 10.079 | 9.998 |
| norms + trial division | 1.885 | 1.786 |
| classify | 0.627 | 0.557 |
| **TD device total** | **14.044** | **13.797** |

Fill and apply are ~70 ms of the 112 and are **per position**, not per
survivor, so they cannot respond. The whole trial-division block is 14 ms and
moves 0.25 ms -- 0.22% of wall. And `candidates/q` is flat to **0.04%**
(1590.68 -> 1590.08), so the noise survivors never reach cofactorisation
either: they are killed by the rank scan and survivor filter, which together
cost 0.5 ms.

**So `las_scale` deriving from a 255-value cell is harmless, not a defect.**
The resolution is unused, and using it changes neither throughput nor yield
(relations move 6,743 -> 6,740, i.e. -0.04%, the wrong way and inside noise).
Recorded so the inference "fewer survivors must be faster" is not made again;
on this pipeline the survivor list is nearly free downstream.

Two notes for anyone re-running it. The runs above pass `--allowance`
explicitly to hold the gate at the derived bits; simply raising `scale` also
moves `sieve_allowance`'s `2/scale` slack term, which is a different
experiment. And the binding ceiling on scale is not `CINIT` but the guard at
`bench_main.cu:1862` (`scale * log2(lim) <= 255`, the 8-bit per-ideal
factor-base log), which caps it near 9.1 at `alim` 240M.

The 1:1/2:1 relation ratio is **1.3607 at scale 4.0 against 1.3608 at the
derived scale**, which is the control that rules quantisation out of finding 65
entirely.

## Finding 67 — q-truncation on a 275M-relation corpus: lossless in every configuration an operator would actually run, for a 1.8% prize

**Date:** 2026-08-18. Corpus: SNFS `17327^61-1` (difficulty 258.56, `rlim
67.1M, alim 134.2M, lpbr 31, lpba 32, mfbr 61, mfba 92`), **41.5 GB /
339,445,456 relation lines**, rational special-q. The largest dataset this
project has replayed, and it is the one STATUS item 3 was waiting for.

### The band, recovered from the corpus rather than assumed

NFS@Home returns an aggregate of many workers, so the file is not q-ordered and
its head is not a random sample (a `head -c 2G` slice reads as 100% covered at
every band top — the file *starts* at q = 40000003). Recovered from a uniform
random sample instead, by the property that **every** relation must contain at
least one band prime:

| band top | coverage | | band bottom | coverage |
|---:|---:|---|---:|---:|
| 170M | 0.98796 | | 39M | 1.00000 |
| 174M | 0.99869 | | 40M | **1.00000** |
| **175M** | **1.00000** | | 41M | 0.99516 |

Sharp at both ends: **q in [40M, 175M), rational side**. Independently, the
share of primes seen more than once collapses from 8.1% in 160–180M to 0.4% in
180–200M.

### Counting unique relations without deduplicating 275M of them

The corpus holds one line per (relation, q) pair and **the line does not record
q** — a relation found at two special-q produces two byte-identical lines. So
line counts weight a relation by how often it was re-found, and deduplicating
outright needs several GB. Instead relations are selected by a hash of their
`(a,b)` text: selection is deterministic, so every copy of a 1-in-200 subset is
kept and its multiplicity is exact, at 1/200 of the memory.

    lines read       339,445,456   (0 malformed)
    unique relations   1,376,424   sampled  ->  ~275.3M in the corpus
    mean copies/uniq        1.2325 <- raw; 1.2096 once a hand-stitched
                                      restart's re-sieved window is removed
                                      (see the last section)

### On the band as actually run, truncation loses nothing — structurally

**Zero unique relations lost**, and this is provable rather than merely
measured. Under truncation a relation is found at its largest band prime `q`;
any unsieved factor-base prime `p > q` with `p <= lim` would also satisfy
`p >= band_bottom`, so `p` would itself be a band prime *larger* than `q`,
contradicting maximality. Hence no unsieved FB prime remains, the cofactor
equals the full-base one, and the relation is found.

**Truncation is lossless exactly when `band_top >= lim`.** That single
condition explains every corpus this project has replayed: the c151's band
ended precisely at `alim` (zero loss), and this one runs to 175M against `rlim`
67.1M (zero loss). Neither was evidence that truncation is safe in general —
both were the favourable case, which is what item 3 suspected and could not
test.

### The counterfactual, and why its loss column is NOT a production risk

For a candidate band `[40M, B)`, count the unique relations a full-base run over
that band would find, and ask how many a truncated run finds at no q at all.

**Read the `B < lim` rows as test bands, not as configurations anyone runs.**
The owner's point, and it is decisive: if `lim` is above the top of the band
then truncation caps the sq-side base at `q < lim` for *every* q in the run, so
the `lim` that was set is never the `lim` that is used — an operator would
notice and lower it. Only two shapes occur in practice:

| | truncation | loss |
|---|---|---|
| `lim <= band_bottom` | never binds | zero, trivially |
| `band_bottom < lim <= band_top` (**this job**) | binds for `q < lim` | **zero, by the argument above** |

and the maximality argument covers the second: any unsieved FB prime `p > q`
has `p >= band_bottom` and `p <= lim <= band_top`, so it is in the band. So the
honest conclusion is stronger than a measurement — **in every configuration an
operator would actually run, q-truncation loses nothing at all.** The sweep
below exists to show what the loss regime *would* cost and to confirm the
boundary is exactly at `B = lim`, not to price a real option.

**Unique relations LOST to truncation (%)**

| `B`/lim | band top | mfb 61 (this job) | mfb 80 | mfb 92 |
|---:|---:|---:|---:|---:|
| 0.62 | 41.6M | **12.725** | 10.625 | 2.373 |
| 0.70 | 47.0M | 9.641 | 8.182 | 1.689 |
| 0.80 | 53.7M | 6.023 | 5.255 | 0.987 |
| 0.90 | 60.4M | 2.835 | 2.525 | 0.435 |
| 0.97 | 65.1M | 0.814 | 0.736 | 0.127 |
| **1.00** | 67.1M | **0.000** | 0.000 | 0.000 |
| 2.61 | 175M | 0.000 | 0.000 | 0.000 |

**Duplicate finds REMOVED by truncation (%)**, same grid

| `B`/lim | mfb 61 | mfb 80 | mfb 92 |
|---:|---:|---:|---:|
| 0.62 | 13.14 | 10.96 | 2.48 |
| 0.80 | 9.45 | 7.99 | 1.67 |
| 1.00 | 6.35 | 5.43 | 1.10 |
| **2.61 (as run)** | **1.78** | 1.53 | 0.35 |

Read the two together:

- **On the band as run, the prize is 1.78%** of the duplicate finds. Nearly
  nothing, for a per-q factor-base change — and that, not the loss column, is
  the number that decides this item.
- **Below `lim` the prize grows, but the loss grows faster** — at 0.70 it
  removes 11.4% of finds while losing 9.6% of the relations. This is the
  regime a *test* band lands in, which is exactly how the c147's 22.68% figure
  arose (below), and it is worth knowing so that a narrow probe is never
  mistaken for evidence about a production run.
- **Generous `mfb` collapses both.** At `mfb 92` — the c183's `mfba 92`, and
  close to the C195's `mfba 95` — loss and prize are each under 2.5%
  everywhere. The unsieved primes simply fit in the cofactor, so truncation
  becomes very nearly a no-op. This is the mechanism item 3 already predicted
  from `mfb` headroom, now measured on both sides.

Caveat in our favour, and it means these losses are **lower bounds**: the fit
test is `mfb` alone. A truncated run also has to get those larger cofactors
past the survivor threshold, which will refuse some of them.

### This reconciles every corpus replayed so far

The rule `loss = 0 iff band_top >= lim` is not new evidence contradicting the
earlier replays -- it is what all four of them already said, once each band is
compared against its own `lim` rather than against the others:

| corpus | sq-side lim | band as run | `band_top` vs lim | unique relations lost |
|---|---:|---|---|---:|
| c147 | `alim` 33.5M | [15.00M, 15.15M] | **0.45x** | **22.68%** |
| snfs236 (partial) | `rlim` 134.2M | [30.0M, 36.97M] | **0.28x** | loss regime |
| c151 (complete) | `alim` 33.5M | [15.0M, 33.5M] | **1.00x** | **0** |
| snfs2 (this) | `rlim` 67.1M | [40M, 175M) | **2.61x** | **0** |

The two that lost relations are the two whose band stops short of `lim`, and
neither is a production configuration: the c147's band is **0.15M wide**, a
test probe, and the snfs236 corpus is explicitly a *partial* slice of its job's
band. The two complete, realistically-shaped bands both lose exactly nothing.

So the c147's 22.68% -- quoted under item 3 as the alarming number -- is a
property of a 0.15M-wide probe, not of the design. Nothing here shows loss
inside a band anyone would sieve.

### A restart's re-sieved window, located from the corpus alone

Conditioning on `k`, the number of band q a relation could have been found at,
`E[copies | copies >= 1] = kP/(1-(1-P)^k)` solves for P directly. The **`k = 1`
row is a control**: a relation with one possible q cannot be re-found, so our
convention predicts exactly 1.0000 copies. It read **1.0141**.

That 1.41% is duplication the special-q convention did not cause. The
duplicated lines are byte-identical — a relation line does not record which q
produced it — so the lines themselves cannot say why. Their *q distribution*
can, and it is unambiguous:

| q range | k=1 relations | duplicated | rate |
|---|---:|---:|---:|
| 40-50M | 83,308 | 0 | 0.000% |
| **50-60M** | **77,354** | **12,735** | **16.463%** |
| 60-70M ... 160-175M (11 buckets) | 679,164 | 0 | 0.000% |

Narrowed, the duplicated relations occupy `q` in **[55,469,851, 57,129,679]**,
and inside that window **12,735 of 12,736** single-q relations are duplicated --
100.0%, with copies capped at exactly 2. Outside it, **0 of 827,090**.

That is a **stop/restart overlap: a ~1.66M-wide q window sieved twice**, which
the owner confirms (this run predates the `.part`/`.ckpt` mechanism, so the two
halves were stitched by hand). It is *not* BOINC re-issuing workunits, which an
earlier draft of this finding claimed -- that would have been spread across the
band, and the measured rate outside the window is exactly zero.

Excluding the window, the control comes out clean and P falls into line:

| k | relations | mean copies | implied P |
|---:|---:|---:|---:|
| 1 | 889,785 | **1.0000** | -- |
| 2 | 389,277 | 1.5450 | 0.7055 |
| 3 | 62,085 | 2.0322 | 0.6478 |
| 4 | 3,674 | 2.4747 | 0.6034 |

**P(re-found) = 0.6968** weighted over k>=2, against 0.702 (snfs236) and 0.723
(c151) -- three jobs, one number. Duplicate inflation is **1.2096x** with the
window excluded, against 1.2325x raw.

Two things worth keeping. **The `k = 1` stratum is a free integrity check on
any corpus**: it must read 1.0000, and whatever it reads above that is
duplication from outside the siever, localisable by histogramming q. And
**item 12a's checkpoint/resume exists precisely to prevent this** -- a
hand-stitched restart cost ~1.7M q of duplicate sieving here, which a
byte-exact resume removes.

## Finding 68 — a shipped correctness bug: non-primitive relations when q is small relative to the sieve area

**Date:** 2026-08-18. Reported from the field: an SNFS job
(`14193046087661928916151319601^7-1`, degree 6 with `F = (x^7-1)/(x-1)`,
`alim 3.5M, rlim 6M, lpba 28, lpbr 29, mfba 53, mfbr 57`, CADO-derived
parameters, rational special-q from q = 400000, `--logI 14`) produced hundreds
of `error -6 reading relation N` in msieve's `-nc1`.

**Reproduced exactly, root-caused, fixed, and gated.**

### What error -6 is

`gnfs/relation.c:161` returns -6 when `gcd(a, b) != 1` (`:167` for a zero
rational norm). msieve is refusing **non-primitive** relations: `(a,b)` a
multiple of a smaller pair, carrying no new information.

Sieving 154,810 relations with the reported command reproduced it: **154 of
them (0.0995%) had `gcd(a,b) != 1`**, my file's line numbers matching the
reporter's error indices at a constant offset, and the reporter's own msieve
binary reporting exactly 154 errors on the reproduction.

### Where they come from, and why the existing filter missed them

Every gcd was **the special-q itself** — 400009, 400051, 400093, ... never 2 or
3. `k_intersect_compact` dropped positions with `gcd(i,j) != 1` and its comment
asserted that this made `(a,b)` primitive. **It does not.** `(a,b) = M(i,j)`
with `det M = +-q`, and a non-unimodular map can destroy primitivity — on
exactly one sublattice, the one where `q | b` (hence `q | a`, since
`a = rho*b` mod q on the lattice), giving `(a,b) = q*(a',b')`.

No other prime can do it: `p | a` and `p | b` with `p != q` implies
`(a/p, b/p)` is still on the q-lattice, so `(i,j) = p*(i',j')` and the gcd test
already catches it. **So `gcd(a,b)` is 1 or q**, which is what the measured
distribution shows and what makes `q % b` an exact test.

### Why they survive the sieve — they look perfectly smooth

A position with `q | a` and `q | b` satisfies `a = r*b (mod q)` for **every**
root r of q, so it sits on the plattice line of all of them and the sieve
subtracts `log(q)` once per root. When the algebraic polynomial splits
completely mod q that is `deg * log(q)` — precisely the `q^deg` in
`F(a,b) = q^deg F(a',b')`. The rational side is `G(a,b) = q*G(a',b')` and the
special-q bias removes its one `log(q)`. Both sides then look exactly as smooth
as the much smaller pair `(a',b')`, the position survives, and trial division
*confirms* the factorisation — of a relation that is q times a smaller one.

The prediction that follows is sharp, and it holds: `F = (x^7-1)/(x-1)` splits
completely mod q **iff q = 1 (mod 7)**, and **154 of 154** offending relations
had `q = 1 (mod 7)`. The reduced points are tiny — `(-5,4)`, `(-3,11)`,
`(1,18)`, with `max |a/q| = 152`.

### Why it took a small job to expose it

The number of affected positions per special-q is about `I*J / q`. This job
sieves `2^27` positions against q from 400009 — **~335 candidates per q**.
A C194 at `2^29` against q = 250M has **~2**. The bug has been present the
whole time and is invisible at production q; it needs q small relative to the
sieve area, which is what a small SNFS with CADO parameters and `--logI 14`
produces. The c147 gate band contains **zero** non-primitive relations, which
is why every existing test passed.

### The fix, and the gate that should have caught it

One extra test in `k_intersect_compact`, one 64-bit modulo per two-sided
survivor (~1 position in 400):

    if (bgcd(ai, j) != 1) continue;
    if (q > 1 && ((int64_t)i * a1 + (int64_t)j * b1) % q == 0) continue;

Verified: relations 154,810 -> **154,656**, exactly the 154 removed, **zero**
non-primitive remaining, and the reporter's msieve reports **0 errors**. The
c147 1500-q band is **byte-identical** (`47bb45b982d6d1c7f983a02981d001ab`),
so the change is a no-op wherever q is large relative to the area, and
`make check` passes 72 gates.

**`--check-relations` did not catch this, and that is the more important
defect.** It verified that every factor divides, is prime and is within lpb,
and that both norms rebuild to exactly 1 — all of which a non-primitive
relation passes. It now also tests `gcd(a,b) == 1`, reported under its own
name rather than folded into the composite-factor count, because a
non-primitive relation is perfectly factored and calling it composite sends the
reader after the cofactoriser instead of after the survivor filter. Checked
both ways: 5 deliberately doubled relations (`F(2a,2b) = 2^6 F(a,b)`,
`G(2a,2b) = 2 G(a,b)`, so the factorisation is exact and only primitivity is
wrong) are rejected, and 3,000 real ones pass.

**The lesson worth keeping: a relation can be arithmetically perfect and still
be invalid.** Every check we had was about the factorisation; none was about
whether the point should have been sieved at all.

## Not addressed in this round

> **This section is a snapshot of one round, not a current status list**, and it
> sits directly after finding 51 where it reads like one. Items superseded since
> are struck through with the date. For what exists *now*, read `STATUS.md`.

- ~~**The survivor-set gate passes on counts**~~ **DONE — 2026-08-03/04.** This
  item is superseded; see `../prototype.md`, "The gate was built and run" and
  "The gate at scale". Corrections to what is written above:
  - `las -batch-print-survivors` **does not** dump the `after_sieve` set. It
    emitted 1,851 records, not 797,028, because `needs_resieving()`
    (`las-siever-config.hpp:116`) returns false when any side has `lim == 0`,
    and the flag's output is the post-TD cofactor list. The set comparison
    required a 3-line CADO patch instead (`../oracle/cado-after-sieve-survdump.patch`).
  - **Strict set containment fails and that is expected**, not a defect:
    2,162 of las's survivors are absent from ours, attributed to `powlim`
    pinning, a boundary column at `b = I/2` where the two conventions sieve
    different half-open intervals, and an unexplained residue (side 1: 722,
    side 0: 266) that remains open as a diagnostic.
  - **Relation containment is the operative gate and it passes:** over the whole
    frozen band, **3,026 of 3,026** in-region las relations are survivors of
    ours, zero misses. The remaining 136 fall outside our sieve region because
    20 of the band's 67 lattices use a different (equally valid, in fact
    better-reduced) basis — see `../oracle/PARITY.md`.
  - What this establishes is **in-region sieve correctness, not yield
    equivalence.** Final relation yield must come from cofactoring our own
    region.
- ~~**The primitive-point filter is host-side only.**~~ **DONE.** The device
  intersection, compaction and gcd filter are built and gated.
- ~~**The primary post-sieve GPU path is unbuilt.** No GPU resieve/factor
  recovery, regular TD, or hard-cofactor stage consumes the compacted
  candidates.~~ **DONE — 2026-08-05/06.** Intersection, GPU trial division,
  classification, resieve and cofactorisation (rho and ECM) all exist, emit
  relations, and are gated by `cofcheck.sh` (28 cases) plus the post-cofactor
  reconstruction gate. The whole-band runs quoted in findings 48–51 are of that
  complete path. (The struck text went on to weigh borrowing YAFU's CUDA/OpenCL
  ECM kernels; we wrote our own instead, so that option is moot.)
- **Per-region offset hashes** are still not done — counts cannot see a
  permutation within a region, and the verify gate compares counts.
- **`--maxbits > 15` is now safe but untested**: the transform handles bucketed
  odd prime powers, and `g > 1` losses are reported, but no run has exercised
  it. The reported-loss counter is the thing to watch when someone does.
- The `g > 1` bucketed case emits nothing rather than routing to the small
  tier. Correct-and-reported, not correct-and-complete. Zero at the default
  `bkthresh`.
- The remaining items under "What is not yet measured" below.

## What is not yet measured

**This is a sieve measurement, not a relation-collection measurement, and the
distinction is load-bearing.** What runs end-to-end is: transform → fill →
apply → threshold → survivor bitmap, per side. What does not exist at all is the
two-sided survivor intersection/compaction, GPU resieve/factor recovery/TD,
GPU cofactorization, final relation output, and unique-relation accounting.
~~The survivor *list* is also capped at 2^22 entries against one-sided sets of
18–30M — the count is exact and truncation is reported, but nothing consumes
the list yet.~~ **REMOVED 2026-08-24 (finding 73):** nothing ever consumed it,
so the list and its cap are gone; the per-side survivor *count* remains, and
the bitmap it was never a substitute for is what the pipeline actually reads.

So: **kernel feasibility is demonstrated; GPU-resident relation-collection
feasibility is not.** Any "3–4× whole-box speedup" is specifically the optional
hybrid projection with a strong CPU cofactor path assumed unchanged, not a
measured relation rate and not an all-GPU estimate.

The remaining comparison constants and scope limits are:

| constant | status |
|---|---|
| GGNFS `N_eff` at 14–16 workers | **measured 2026-08-03: 10.24 at 16 workers** (finding 43). Was assumed 13. |
| GPU watts during the chain | **measured cleanly** in finding 44: 206.7 W stage-weighted board average, 13.306 board J/q for both sides. |
| CPU and DRAM watts under 16-worker GGNFS | **measured from Windows** in finding 44: 125.073 W CPU PPT + 5.222 W DIMMs; 158.731 W full component proxy including the idle GPU. |
| component energy comparison | **measured** in finding 44: CPU full q 44.609 J/q; GPU two-side sieve 16.874 J/q. Scope differs, so this is not yet end-to-end. |
| resieve/factor recovery | **accounted for only in the hybrid projection** inside GGNFS's retained TD phase; unimplemented and unmeasured on GPU for the primary path. |
| GPU hard cofactor stage | **preliminary only**: finding 47's external C164 `31/31` printed stages imply 386,084 inputs/s and 4.92 ms/q at the projected feed. C183 `31/32`, persistent latency, power, yield and host demand remain unmeasured. |
| whole-box wall watts | **unmeasured** — motherboard, drives, fans, VRM and PSU losses remain outside the HWiNFO component proxy. Required for the final economics verdict. |
| CADO post-sieve share (Gate 0) | unmeasured; useful for oracle workload and a CADO hybrid projection, not a prerequisite for the GPU-resident architecture. |
| the "200× root-transform speedup" (finding 4) | GGNFS's Sieve-Change timer also covers small-sieve setup, transformed-polynomial work and report-bound setup, so this is an **upper bound**, not a like-for-like stage comparison. |

Also unmeasured: target-equivalent persistent YAFU-derived GPU cofactor
throughput, coverage, power and recurring host demand; weak-host and multi-GPU
scaling; `bkthresh` sweep; I16e slabbing; throughput mode; production scales
2/4/8 (finding 23); and the per-q host-side small-FB transform and sort
(`bench_kernels.cu`), which runs outside every timed number.

Byte-exact parity is blocked on a trustworthy oracle, not on our side of the
comparison.

**Caveats on the numbers above.** *(Written against the Path-2 configuration;
the sections from "Small-prime sieve, rational side" onward run both sides on
the real `q=120000053, rho=112625526`, so "one side only, synthetic rho" no
longer applies to those. It still applies to findings 1–10.)* The 1-in-413
survivor rate is one-sided with a generous 112-bit allowance and is *not* a
claim about las's survivor count — the real rate comes from intersecting both
sides, which is not implemented.

## Reproduce

**Current clean equal-work profile** — exact las scales, algebraic factor base
truncated at the special-q (GGNFS convention), and the cheap device parity
filter enabled. This is the profile used by findings 43–44:

```
cd bench && make
./bench --cadofb ../oracle/c183.fb1 --side 1 --scale 1.275 \
        --fbbound 120000053 --q 120000053 --rho 112625526 \
        --allowance 112 --not-both-even                    # 38.177 ms
./bench --side 0 --scale 1.925 \
        --q 120000053 --rho 112625526 --allowance 72.85 \
        --not-both-even                                    # 26.194 ms
```

For finding 44's power plateaus, add `--reps 2500` to side 1 and `--reps
3600` to side 0 while HWiNFO logs at two-second intervals. The benchmark times
transform, fill and apply as separate repetition blocks, so compute joules by
weighting each plateau by its printed stage time.

Defaults are `--mode atomic --record-bytes 4 --region 14 --apply-threads 512`
as of 2026-08-02; before that they were `twolevel` and `--region 15`, so
commands below that omit those flags reproduced a path that had already lost.

**`--reps 100` is the floor for any cross-machine comparison** (finding 48).
Below that the transform line reports amortized CUDA startup rather than
kernel time — it swings 98× between reps 3 and 1000 while fill and apply move
under 1%. The grid width now comes from `multiProcessorCount` and is echoed at
startup, so confirm the `grid: N SMs x 6` line matches the card before
comparing anything (finding 49).

**Cross-GPU profile** — the command all three cards in finding 50 ran:

```
./bench --poly ../oracle/c147.job --cadofb ../oracle/c147.roots1 \
        --logI 14 --J 8192 --reps 100
./bench --pipeline --cofactor --poly ../oracle/c147.job \
        --cadofb ../oracle/c147.roots1 \
        --logI 14 --qrange 15000000: --target-rels 100000 --relations OUT.dat
```

**This finding's workload is the C147, not the C183** — as are findings 48 and
54, the 144×256 vs 1152×32 geometry result, and the host-load experiment. Not
every timing in this file: findings 43–44 above profile the C183 via `--cadofb
../oracle/c183.fb1`. The two jobs are not interchangeable, so check which one a
command names before reusing its numbers.

`../oracle/c147.job` is tracked in git. `../oracle/c147.roots1` is 29 MB and
git-ignored — regenerate it with the `fbgen` command in `../oracle/README.md`,
which needs no CADO and reproduces the manifest-pinned file byte for byte.
**Pass `--maxbits 14`** to match this `--logI 14`; `fbgen` on its own defaults to
15 and produces a different factor base, and `bench` downgrades the mismatch to a
`note:` you will scroll past.

The standalone bench (no `--pipeline`) sieves one side at a fixed
`q=120000011`; the pipeline sieves both sides across a real band. They agree
once transform is excluded: A100/5070 on standalone fill+apply is **1.69×**
comparing like reps (both at `--reps 3`: 15.605/9.236), or 1.66× against the
5070's `--reps 100` row in finding 50's table (15.60/9.40). Pipeline sieve is
**1.68×**. The reconciliation holds either way; the residual spread is apply's
reps drift, not a disagreement between the two harnesses.

**The one CPU-only gate.** `fbtest` is the only thing here that touches
neither the GPU nor `nvcc`:

```
./fbtest --cadofb ../oracle/c183.fb1
```

`make check` is **not** a substitute and is **not safe alongside a running
job**. It used to be exactly the line above; it is now `check: all cofcheck`
(`Makefile:76`), so it compiles the whole CUDA path and then runs `cofcheck`
*on the GPU*. That change was deliberate — the CUDA path could previously fail
to compile while the gates still reported "all gates passed" — but it means
`make check` now contends for the card.

Nor is "CPU-only" the same as "safe on a busy box". Finding 53 measures host
contention at an **18.4–22.3% relation-rate loss** with every `cudaEvent`
timer still flat within 1%, so a parallel `nvcc` build both slows the running
job and silently corrupts any timing that job reports. On a box that is
sieving, run neither.

**GPU gates** (these run kernels — do not use them to check a busy box):

```
./bench --verify --logI 12 --J 512 --region 12   # correctness vs CPU reference
./bench --verify --region 14 --mode atomic --record-bytes 4 --apply-threads 512
./bench --mode atomic --record-bytes 4 --region 14 --apply-threads 512   # best
./bench --mode twolevel                          # the fill variant that loses
./bench --region 14 --norm const                 # isolates norm-init cost
./bench --region 14 --apply-mode plain           # isolates smem atomic cost
./bench --region 15 --cells 8                    # prices the unsafe byte cell
```

## Finding 69 — the C208 validated against a 1.5B-relation GGNFS corpus: 99.97% recall, 1.6% genuinely new — and the default rho budget was silently losing 13.7% of it

**Date:** 2026-08-19, RTX 5070 against the AS276 (C208) production corpus in
`~/code/ggnfs-distributed/AS276/` — 283,364 work units, ~1.5B relations,
`gnfs-lasieve4I16e -J 16`, already filtered and known good. Our side is the
first run of the 4-limb cofactor path (finding: STATUS "Four-limb cofactors").

AS276 is `lpbr 33 / lpba 35`, `mfbr 64 / mfba 101` — the first job whose
algebraic side does not fit 96 bits, and so the first real exercise of `mz<4>`.

### The comparison is exact, not approximate

Their work units are contiguous, non-overlapping 1000-wide q blocks:
**file index `N` = `floor((q - 80000000)/1000)`**, verified at N = 0, 1, 30, 60
and 283364. So `wu-38370f06-000000.dat.zst` is the *complete* GGNFS output for
`q in [80000023, 80000939]` — 30 special-q, 4,564 relations — and nothing for
those q lives anywhere else. Confirmed by probing 60 neighbouring work units
for 40 of our relations: zero hits.

`relgeom.py extent` recovers both rectangles from the relations themselves:

| run | i range | j range | rectangle |
|---|---:|---:|---|
| theirs, `I16e -J 16` | ±65477 | 1 .. 32761 | `2^17 x 2^15` |
| ours, `--logI 17 --J 16384` | ±65449 | 1 .. 16375 | `2^17 x 2^14` |
| ours, `--logI 16 --J 32768` | ±32721 | 1 .. 32761 | `2^16 x 2^15` |

confirming finding 65's rule `our rectangle = 2^(J_bits+1) x 2^(I_bits-1)` at
`n = 16`. **Their shape is `A = 32`, which we still refuse**, so both of our
runs are nested sub-rectangles — one halving `j`, the other halving `i`. Two
independent nestings, which is what makes the agreement below meaningful.

> **Correction 2026-09-01: "which we still refuse" expired on 2026-08-24.** The
> slab merge removed the total-area cap under `--pipeline` (the `I*J <= 2^31`
> refusal is guarded by `!cfg.pipeline`), and `A = 32` has
> since been sieved — `I16 J65536` in finding 74 and `I=J=2^16` in finding 72.
> What is still unrun is *this* aspect ratio, `2^17 x 2^15`, which is what a
> like-for-like rerun of this comparison would need.

### The rho budget was the whole story

At the **default** `--cof-rounds 2 --cof-budget 65536`, in the identical
`2^17 x 2^14` region over the identical 30 special-q:

| | relations | of theirs, missed |
|---|---:|---:|
| default budget | 3,205 | **389 of 2,846 (13.7%)** |
| `--cof-rounds 6 --cof-budget 262144` | 3,823 | **2 of 2,846 (0.07%)** |

**+19.3% relations from the budget alone**, and the missing 13.7% was not
geometry, not the gate, and not the width — it was `mz_split` running out of
rho iterations and returning `CF_INCOMPLETE`. That is exactly what a 4-limb job
should do to a budget tuned on 3-limb ones: rho's expected iteration count
scales as the square root of the factor sought, so a 35-bit large prime costs
`sqrt(2^(35-30)) = 5.7x` what a 30-bit one does. The default was calibrated on
the c183's `lpba 32`.

**This is the operational finding of the 4-limb work.** The width change is
correct and cheap; the *schedule* around it is what needed retuning, and a run
at the old default would have looked like a 14% yield hole in the siever.

### Agreement with GGNFS, at the retuned budget

Second nesting (`2^16 x 2^15`, the default-`16e` shape), same 30 q, all
4,089 of our relations passing `--check-relations` (every factor divides, is
prime, is within `lpb`, both norms rebuild to exactly 1):

| | count |
|---|---:|
| in both | 3,044 |
| theirs only | **1** |
| ours only | 1,045 |

**Recall 3,044 / 3,045 = 99.97%**, flat across all eight `j` bands (our share
1.26 to 1.44, no geometry artifact).

The 1,045 "ours only" are *mostly not new*. A relation carries every band prime
dividing its norm, so GGNFS may have found the same `(a,b)` under a different
special-q. Using the exact WU mapping above to look each one up in the work
unit that would have produced it:

| | count |
|---:|---:|
| re-found elsewhere in their corpus | **981** |
| had a re-find opportunity, absent anyway | 10 |
| no other q in their swept range `[80000023, 363364957]` — cannot be anywhere | 54 |

So **64 of our 4,089 relations (1.6%) exist nowhere in their 1.5B corpus**, and
all 64 reconstruct both norms exactly.

### What this establishes

- The 4-limb cofactor path is **correct on a real 4-limb job**, cross-validated
  against an independently filtered production corpus rather than against
  ourselves.
- The two sievers agree to **~2% in both directions** — 0.03% of theirs missed,
  1.6% of ours novel — which is the same geometry-independent residual finding
  65 measured on the C194 at 0.979. Neither siever is a subset of the other,
  and that was never the expectation once relations can be re-found under
  several q.
- **`--cof-rounds`/`--cof-budget` must be re-derived per job class**, not
  inherited. See RUNBOOK "Cofactor width".

**No timing is claimed here.** An ECM job had the GPU throughout; only counts
and set membership are reported, and both are timing-independent.

## Finding 70 — ECM beats rho ~2-4x at 3LP and LOSES at 2LP; the crossover is the large-prime count, not lpb. (Supersedes this finding's own first version.)

**Date:** 2026-08-19, RTX 5070, GPU idle (the box's ECM job suspended with
`kill -STOP`). Prompted by "might ECM beat rho at lpb 35?"

### Correction to the first version of this finding

The first version reported **15.3x** (c183) and **17.8x** (AS276) for ECM. Those
numbers were wrong, and wrong by the *same methodological error this finding was
written to expose*: I tuned one method and not the other. rho was priced at the
first budget I happened to try that saturated yield (`b=262144`, `b=2097152`),
never swept **downward**. Swept properly, rho saturates far cheaper:

| job | rho as first reported | rho actually swept | same relations |
|---|---:|---:|---|
| c183 `lpb 32 / mfb 92` | 239.09 ms/q (`b=262144`) | **30.83** (`b=8192`) | 9,394 |
| AS276 `lpb 35 / mfb 101` | 2,197.49 ms/q (`b=2097152`) | **332.63** (`b=16384`) | 4,089 |

rho's cost is nearly linear in the budget once the budget is past saturation, so
an over-large budget is pure waste and makes any comparison against it
meaningless. **Both methods must be swept from below.** Corrected:

| job | rho (cheapest saturating) | ECM (cheapest saturating) | ECM speedup |
|---|---:|---:|---:|
| c183 `lpb 32 / mfb 92`, 200 q | 30.83 ms/q (`r6 b=8192`) | **15.38** (`B1=250 c12 r4`) | **2.00x** |
| AS276 `lpb 35 / mfb 101`, 30 q | 332.63 ms/q (`r6 b=16384`) | **123.22** (`B1=300 c12 r4`) | **2.70x** |

Real, reproducible, and byte-identical output — but 2-3x, not 15-18x.

### The crossover is 2LP vs 3LP

Sweep on the c183 polynomial, 25 q, `--lpb` and `--mfb` varied together at
shapes an operator would actually run (2LP at `mfb = 2*lpb` for `lpb <= 30`;
3LP at `mfb = 3*lpb - 3` above, since three exact-`lpb` factors out of a
`3*lpb`-bit composite is vanishingly rare). Cheapest configuration of each
method reaching **saturated relation yield**, both swept from below:

| lpb | shape | mfb | rel | ECM best | rho best | winner |
|---:|---|---:|---:|---:|---:|---|
| 29 | 2LP | 58 | 237 | 2.61 (`B1=70`) | **2.42** | **rho 1.08x** |
| 30 | 2LP | 60 | 373 | 3.21 (`B1=200`) | **2.81** | **rho 1.14x** |
| 31 | 3LP | 90 | 724 | **12.61** (`B1=200`) | 28.15 | ECM 2.23x |
| 32 | 3LP | 93 | 1,171 | **19.71** (`B1=200`) | 42.62 | ECM 2.16x |
| 33 | 3LP | 96 | 1,828 | **29.26** (`B1=200`) | 93.34 | ECM 3.19x |
| 34 | 3LP | 99 | 2,686 | **135.00** (`B1=300`) | 367.49 | ECM 2.72x |
| 35 | 3LP | 102 | 3,837 | **213.66** (`B1=300`) | 880.36 | ECM 4.12x |
| 36 | 3LP | 105 | 5,498 | **394.30** (`B1=500`) | 1,225.73 | ECM 3.11x |

**The discriminant is the number of large primes, not lpb.** At 2LP rho wins
narrowly and the gap is inside run-to-run noise; the moment the shape becomes
3LP, ECM wins by 2-4x and stays there. That is mechanism, not coincidence: 2LP
needs one split of a semiprime with `~lpb`-bit factors, which rho does in
`~sqrt(2^lpb)` iterations on a cost that is *data-dependent* (easy lanes exit
early). 3LP needs two successive splits of a much larger composite, and rho pays
`sqrt` of the whole thing twice while ECM's `B1` barely moves.

Optimal `B1` tracks the factor size gently — 200 at `lpb 31-33`, 300 at 34-35,
500 at 36 — and 12 curves over 4 rounds covers the whole range.

### `stuck == 0` is NOT the saturation criterion

Worth stating because it cost a whole sweep. A record left `CF_INCOMPLETE` is of
*unknown* status, and at 3LP most of them are composites whose factors **all**
exceed `lpb` — never relations, but only provable dead by factoring them. At
`lpb 29 / mfb 87` (an unrealistic shape) rho needed `b=1048576` and 382 ms/q to
reach `stuck == 0` while relation yield had saturated at `b=65536` and 84 ms/q.
**Sweep to saturated yield, and read the stuck count as "unknown", not "lost".**

### The old "rho beats ECM 2.32x" was still an untuned-B1 artifact

That part of the first version survives. On the c183, `B1=1000 B2=10000 c16`
costs 36.85 ms/q against `B1=250`'s 15.38 for the same 9,394 relations — 2.4x,
and `B1=1000` is ~4x above optimal for this job's ~30-bit factors. The original
comparison also priced rho below full yield (the shipped default `r2/b65536`
returns 9,363, not 9,394). Correcting both still inverts the conclusion, just to
2x rather than 15x.

### Applied 2026-08-19

Shipped as the default: method chosen **per side** from `ceil(mfb/lpb)`, rho at
2LP and ECM at 3LP, `B1` derived from `lpb`, and the requeue round default
raised 2 -> 4 because ECM escalates in curves per round and 2 rounds reached
only 4,050 of AS276's 4,089. Zero flags, against the previous default:

| job | stage ms/q | wall ms/q | relations |
|---|---|---|---|
| c183 | 17.21 -> **14.30** | 113.35 -> **109.86** | 9,363 -> **9,394** |
| C194 | 15.48 -> **13.95** | 116.30 -> **109.36** | unchanged |
| AS276 | — | — | 3,443 -> **4,089** |

Cheaper and higher-yielding on all three, and on AS276 the automatic choice
(124.07 ms/q) matches the hand-tuned optimum (123.22) without being told
anything. `--cof-rho` / `--cof-ecm` force one method on both sides.

Three defects surfaced while wiring it, all from `0` being an existing sentinel
that the new "0 means derive" logic overrode: `--ecm-b2 0` (documented as
"disable stage 2") was silently re-enabled, `--ecm-curves 0` stopped being
refused, and an explicit small `--ecm-b1` synthesised an illegal `B2 = 30*B1`.
Fixed with explicit `_set` flags rather than value tests. All three were caught
by `cofcheck.sh`, which is what that suite is for.


## Finding 71 — item 0's verdict band, run at last: 2.99x time and 2.94x whole-box relations per joule, with EVERY term measured on this box in one session

**Date:** 2026-08-20, RTX 5070 **undervolted** (finding 61's 950 mV curve, so not
comparable to findings below 61 without the 6.7% correction) against 16
`gnfs-lasieve4I15e` workers on the 9800X3D. c183 `oracle/input.job`, `I15e`
geometry (`--logI 15 --J 16384` / GGNFS default `J = I/2`), algebraic special-q,
`--cofactor`. Both sides ran the **same q bands** — same first q, width
`10000 * ln(q)` so each covers ~10,000 (q, rho) pairs — and neither ran while
the other was on the box.

This is the run item 0 was chartered to produce. Every prior end-to-end number
was a proxy job or a single q interval, and **every term in the comparison was
previously derived from something measured on a different day**: the CPU's
throughput from finding 43's `N_eff` 10.24, both sides' power from item 6's
constants. None of them is derived here.

### The verdict, at q=190M — the band where the two sievers do identical work

| | GPU, undervolted | CPU, 16 workers | **advantage** |
|---|---:|---:|---:|
| (q, rho) pairs | 10,000 | 10,054 | |
| wall ms/pair | 100.95 | 301.47 | **2.99x** |
| unique relations/pair | 41.98 | 41.95 | 1.001 |
| factor base | 7,605,406 entries | 7,605,406 entries | identical |
| whole box, at the wall | 240 W | 236 W | |
| J/pair | 24.23 | 71.15 | 2.94x |
| **J per unique relation** | **0.5771** | **1.6958** | **2.94x** |

**This is the number to quote: 2.99x time, 2.94x energy.** q=190M is above
`alim` (134.2M), so GGNFS *cannot* truncate its base — the "Trimmed cached aFB"
line is absent from all 16 worker logs — and both sievers run the identical
7.6M-entry base. Yield then agrees to **0.07%** (41.98 against 41.95), which is
a tighter matched control than finding 57's 0.8% and leaves no room for a
yield-accounting dispute in either direction.

### The q=50M band, and why its bigger margin is not the answer

| | GPU | CPU | ratio |
|---|---:|---:|---:|
| wall ms/pair | 105.99 | 271.95 | 2.57x |
| unique relations/pair | 57.43 | 46.44 | 1.237 |
| factor base | 7,605,406 | **3,001,128** | 2.53x |
| J per unique relation | 0.4429 | 1.3819 | **3.12x** |

**Do not quote the 3.12x.** GGNFS trims to `FB_bound 49999999` and keeps 39% of
the base; we sieve all of it. Our resulting 24% yield surplus is not extra
unique output — per finding 67 those relations are *duplicates* that a
truncating siever re-finds later at their own larger q. The 50M row measures a
convention difference, not a hardware one.

It is worth having anyway, because it shows the convention costs us: at 50M we
pay 5% more wall than at 190M for a base GGNFS gets to cut by 61%.

### The GPU's margin GROWS with q, and the mechanism is the factor base

| | q=50M | q=190M | change |
|---|---:|---:|---:|
| GPU ms/pair | 105.99 | 100.95 | **−4.8%** |
| CPU ms/pair | 271.95 | 301.47 | **+10.9%** |
| time advantage | 2.57x | 2.99x | |

GGNFS slows with q because its FB trim stops helping — 3.0M entries at 50M, all
7.6M at 190M. We sieve 7.6M at both, so our cost barely moves. **The GPU's
advantage is smallest exactly where the CPU gets its discount**, and a real job
spends most of its q range above the point where that discount is small.

Note also that GGNFS absorbed **2.53x the entries for 10.9% more wall** — the
CPU side independently reproducing finding 67's `sum 1/p ~ ln ln p` model,
which until now had only been measured on our own bucket array.

### Both sides' power, measured at the wall in this session

Monitors measured at 45 W (item 6) and subtracted from both readings.

| state | on-monitor | **monitors subtracted** | GPU board |
|---|---:|---:|---:|
| GPU sieving, pipeline | 285 W | **240 W** | 133.5 W |
| CPU sieving, 16 x `I15e` | 280–282 W | **~236 W** | 30 W (idle) |

The implied host constant during the GPU run is **106.5 W**, which lands 1.4%
from item 6's independently measured ~105 W idle figure.

**Item 6's 220 W for the CPU side is superseded by 236 W**, and the difference
is not the cores. This box idles its GPU at **30 W**, not the ~16 W P8 figure
item 6 assumed; that accounts for 14 of the 16 W. Per item 6's own rule the
idle-GPU draw belongs on both sides of the comparison, so 236 W is correct and
the CPU cores are drawing what item 6 said they were.

**Sensitivity.** The UPS idle reading is jumpy at +/-7%. At 190M, the energy
margin is 2.94x at (240, 236) W, 3.01x at (235, 236), and 2.88x at (245, 236).
The conclusion is stable across the range.

### Deduplication, measured rather than assumed

| band | side | raw | unique | dup |
|---|---|---:|---:|---:|
| q=50M | GPU | 574,861 | 574,329 | 1.0009 |
| q=50M | CPU | 464,685 | 464,335 | 1.0008 |
| q=130M | GPU | 461,144 | 460,953 | 1.0004 |
| q=190M | GPU | 419,946 | 419,845 | 1.0002 |
| q=190M | CPU | 421,899 | 421,797 | 1.0002 |

The RUNBOOK's 1.19–1.34x is a **band-scale** figure and does not apply at probe
width; item 0 predicted this and it is confirmed on both sievers at once. The
ratio falls with q on both sides, as it must — sparser primes mean fewer
relations with both re-finding q inside a fixed window.

### The third GPU probe, and the drift

The GPU also ran q=130M (101.96 ms/pair, 46.10 rel/pair, dup 1.0004, 24.47
J/pair). Across the three probes yield falls **57.43 -> 46.10 -> 41.98**
rel/pair, 27%, while GPU cost falls 4.8% — so **GPU rel/J falls 23% across the
band**. A whole-job figure is an integral over that curve, not any single probe.
There is no matched CPU control at 130M; it was not run.

### What this confirms, and what it retires

- **Finding 43's `N_eff` 10.24 holds up.** It implied 306 ms/pair whole-box;
  measured here at 190M is **301.47**, 1.5% away. The weakest link in item 0's
  chain turns out to have been sound.
- **Retire finding 57's 2.53x** — stock card, derived 270 W, single q interval.
- **Retire item 10's 3.14x for the c183** — that is a C194 equal-work figure
  carried across jobs.
- The measured c183 undervolted figure is **2.94x energy, 2.99x time**.

### Correctness

All **1,455,951** GPU relations across the three probes pass
`--check-relations`: every factor divides, both norms rebuild to 1 exactly,
every prime within its lpb. This is the gate that matters on an undervolted
card, where the failure mode is wrong answers rather than a crash.

### Caveats

- Host load was 1.0–1.9 during the GPU probes (six rate-limited Python
  workers), not a silent box. Per finding 53 that direction costs the GPU, so
  the margin is a floor.
- One geometry (`I15e`). Finding 65's `2^16 x 2^14` is the better rel/J shape
  for us but was not run on the CPU, so it cannot be graded.
- The CPU bands used 16 workers over 16 equal-width disjoint subranges; workers
  finish unevenly, so a few core-seconds of tail idle are charged to the CPU.
  At 45 min/band that is well under 1%.
- No 130M CPU control.
## Finding 72 — L40 moves the slab-speed optimum to `2^30`, while `2^29` remains the better generic memory/performance default

A matched `I=J=2^16` cofactor run on an NVIDIA L40 gives a different optimum
from the RTX 3090 and RTX 5070: four `2^30`-position slabs are fastest, not
eight `2^29` slabs.

| local slab area | slabs | steady VRAM | fill | TD + classify | complete time/q | relations/q |
|---:|---:|---:|---:|---:|---:|---:|
| `2^31` | 2 | 7.76 GB | 235.029 ms | 68.51 ms | 564.13 ms | 110.58 |
| **`2^30`** | **4** | **4.72 GB** | **212.778 ms** | **70.39 ms** | **531.16 ms** | **110.58** |
| `2^29` | 8 | 3.20 GB | 223.320 ms | 84.40 ms | 555.70 ms | 110.58 |
| `2^28` | 16 | 2.43 GB | 267.659 ms | 121.06 ms | 628.78 ms | 110.58 |
| `2^27` | 32 | 2.05 GB | 373.371 ms | 206.37 ms | 819.63 ms | 110.58 |

The L40 therefore makes two points at once. First, `2^29` is **not** a
universal maximum-throughput slab size: `2^30` is 4.6% faster here, and fill
itself reaches its minimum at `2^30` before rising again at `2^29`. Second,
`2^29` is still a sound generic default: it is 1.5% faster than the former
`2^31` behavior and reduces steady VRAM by 59% (7.76 -> 3.20 GB). Relative to
the L40's speed optimum, it trades 4.6% of throughput for another 32% reduction
in steady VRAM (4.72 -> 3.20 GB).

This is a reason to **investigate large-L2 GPUs, not to add an L40 special
case**. The RTX 3090 and RTX 5070 both prefer `2^29`, while the L40 prefers
`2^30`. More matched data from large-L2 Ada/Hopper/Blackwell cards, ideally
including `k_fill_atomic` L2 hit/write-miss counters, is needed before an
L2-informed automatic slab target can be justified. Until then, the production
planner keeps `2^29` as the performance/memory compromise and `--slab-j`
remains the explicit per-device tuning override.

## Finding 73 — the 32-bit saturating walk is the SLOW one, and the apply threshold scan was paying an atomic per survivor

**Date:** 2026-08-24, RTX 5070, **card otherwise idle**. `oracle/c147.job` + `c147.roots1`
(maxbits 14), `--pipeline --cofactor --logI 14 --qrange 15000000: --nq 5`,
J 8192 (area `2^27`, unslabbed) and J 65536 (area `2^30`, 2 slabs). Reference
binary built from the merge commit in a detached worktree; both binaries run
back to back, three paired repeats unless stated.

**Methodological note, worth more than the numbers.** These were first measured
with an unrelated job holding the card at 100%. Pairing ref against new
controls for a *stationary* competing load, and the sieve-total figures did
survive it almost exactly (−9.8% then, −9.7% now). **Individual stage
percentages did not.** Contention inflated the unslabbed fill win (−9.1% under
load, −6.1% idle) and *deflated* the unslabbed apply win (−9.8% under load,
−13.4% idle) — it moved them in **opposite directions**, because the two
changes have different sensitivities to L2 pressure. A contended run is not a
scaled idle run. Absolutes were ~2.4x inflated. Take stage-level A/B numbers on
an idle card or do not quote them.

**Relations are byte-identical in all six pairs** (648 and 1297 relations), and
output is deterministic run-to-run, so the byte comparison is a real gate
rather than one that passes by accident.

### The headline

ms/q, n=3 paired repeats, every comparison non-overlapping:

| | unslabbed `2^27` | slabbed `2^30`, 2 slabs |
|---|---:|---:|
| fill | 8.718 → **8.190** (**−6.1%**) | 57.697 → 57.288 (−0.7%) |
| apply | 10.540 → **9.125** (**−13.4%**) | 83.317 → **72.543** (**−12.9%**) |
| **sieve, both sides** | 20.09 → **18.14** (**−9.7%**) | 141.83 → **130.65** (**−7.9%**) |
| wall clock/q, complete | 44.69 → 42.01 (−6.0%) | 204.47 → 190.48 (−6.8%) |

Idle-card repeats are tight — sd ~0.09 ms against ~0.75 under contention — so
n=3 separates cleanly where the contended data needed n=8 and a Welch t.

### The same change on c194 — a third of the win, and the reason is structural

**Do not quote the c147 numbers as the result.** Repeating the identical paired
A/B on `c194.job` + `c194.roots1.m16` at the production shape
(`--logI 16 --J 32768`, which auto-plans to 4 slabs), n=7 paired, idle card:

| stage | ref | new | | sd ref / new |
|---|---:|---:|---:|---|
| fill | 107.62 | 107.91 | **+0.3%** | 0.34 / 0.83 |
| apply | 158.06 | 149.56 | **−5.4%** | 0.60 / 1.25 |
| sieve, both sides | 271.79 | 263.61 | **−3.0%** | 0.71 / 2.15 |
| **wall clock/q** | **408.34** | **400.66** | **−1.9%** | 0.82 / 4.01 |

Relations byte-identical on all seven reps (1125 each) — and c194 is
`lpba 33 / mfba 95`, so this is the 4-limb-plus-ECM cofactor path, which the
c147 verification never touched.

**The fill result is not job-dependent, it is GEOMETRY-dependent, and that part
is fully explained.** The `pl_next64` conversion only changed the *unslabbed*
path; the slabbed path was already 64-bit before this work. c147 at J 8192 is
unslabbed, so it gains 6.1%. c194 at J 32768 is slabbed, so it gains nothing —
as did c147 slabbed (−0.7%). **The fill win exists only below the `2^30`
slabbing trigger.** Every production geometry at I16 and above is slabbed and
sees no fill benefit at all.

**The apply gap is NOT explained, and the obvious guesses are wrong.** −12.9% on
c147 slabbed against −5.4% on c194, and the absolute saving falls (10.78 ms to
8.00 ms) even though c194 has twice the area and 7x the two-sided survivors
(237,256 against 34,769). So the win tracks neither positions nor survivors.
Untested candidates: the per-side survivor counts the scan actually branches on
rather than the two-sided figure, the small-prime line sieve's share of apply at
`alim` 240M against 33.5M, or L2 behaviour at twice the region count.

**Headline, stated honestly: wall clock −1.9% (c194) to −6.8% (c147 slabbed),
sieve −3.0% to −9.7%.** c194 is the more representative job and it is the low
end. Plan against −2%, not −7%.

### `pl_next` costs a branch per increment; `pl_next64` costs an add

`pl_next` routes every increment through `pl_add32_sat`:

```c
if (hi || x > UINT32_MAX - lo) return UINT32_MAX;
```

That is a **branch per increment, up to two per walk step**, in the hottest
loop in the program. `pl_next64` is a plain 64-bit add — two IADDs, no control
flow. On this hardware the wide add wins, and register pressure never became
the binding constraint.

**This was found by getting it backwards.** The first attempt converted the
*slabbed* walk down to 32-bit on the theory that a 64-bit `x` plus two 64-bit
increments crowded the register budget. It cost **+6.0% of fill** (116.8 →
123.8 ms/q, n=4 vs n=5, non-overlapping ranges, I14/J65536). The move that pays
is the opposite one: convert the *unslabbed* path up to 64-bit, which is where
the −6.1% above comes from. The slabbed path was already correct and is
correspondingly flat.

**Every fill and resieve kernel was converted except one — see the
`k_fill_l1` note below, which reverts it.**
`k_fill_l1`, `k_fill_segmented` and `k_resieve_rewalk` are the A/B partners for
`k_fill_atomic` and `k_resieve_scatter`; leaving them on `pl_next` would have
charged ~6% of walk overhead to whichever bucketing strategy was under test.
**Consequence: absolute fill numbers recorded before this finding are stale for
every strategy.** What the conversion establishes is narrower than it first
looks: the strategies now *call the same walk function*, which is a
precondition for comparing them, not a demonstration that they compare cleanly.
Only `k_fill_atomic` was A/B'd. `k_fill_l1`, `k_fill_segmented` and
`k_resieve_rewalk` were converted unmeasured — and `k_fill_l1` in particular
carries `__launch_bounds__(512, 3)`, which caps it near 40 registers, so
widening its locals is not free by inspection. Findings 1-3 should not be
re-derived against post-conversion numbers until those kernels are measured.

**`k_fill_l1` was converted and then REVERTED — it is the one kernel that must
keep the 32-bit walk.** It carries `__launch_bounds__(512, 3)`, which caps it at
40 registers; HEAD fits in exactly 40 with no spill, and two extra 64-bit live
values push it over, at which point ptxas honours the bound by spilling instead
of dropping to 2 blocks/SM:

| | registers | stack | spill st | spill ld |
|---|---:|---:|---:|---:|
| HEAD | 40 | 0 | 0 | 0 |
| converted | 40 | 8 | 12 | 8 |

The converted version *would have been* the only kernel in the build with any
spill traffic — the shipped tree has zero spills anywhere, `k_fill_l1` included,
because it was reverted. The conversion existed purely to keep the
fill-strategy A/B honest, and buying that with local-memory traffic in one
strategy defeats its own purpose. So `k_fill_l1`
stays on `pl_next`, and the strategies are **not** all on the same walk after
all. Say so when quoting them against each other.

`pl_next`/`pl_first`/`pl_add32_sat` are therefore still a live device path, and
`verify_walk_slabs`' SLAB32 block still guards a shipping kernel rather than
only a CPU oracle.

### The apply threshold scan: one store per 32 positions, not one atomic per survivor

The scan gave every surviving position an `atomicOr` into `survbits` and an
`atomicAdd` on a single global counter — and that `atomicAdd`'s **return value**
was consumed by `surv[at] = x`, forcing a serialising `atom.global.add` on one
address. `surv[]` was **write-only in both callers**: allocated, passed, stored
into, never read, in the pipeline and in the standalone harness alike.

A warp now covers 32 consecutive positions, which is exactly one `survbits`
word, and stores it outright:

```c
const uint32_t mask = __ballot_sync(0xFFFFFFFFu, keep);
if (lane == 0) { if (survbits) survbits[x >> 5] = mask; nlocal += __popc(mask); }
```

`nlocal` is committed once per warp as a non-returning add. The list is gone,
with 32 MB of allocation. Adjacent lanes read the same shared word (CPW cells
per word), which broadcasts; the alternative thread-per-word shape that
`k_intersect_compact` uses would give a 16-way bank conflict here, so the
ballot is not merely a stylistic choice.

The store is only race-free across blocks when a block owns whole `survbits`
words — `log_region >= 5` — so smaller regions keep the `atomicOr` form.

### Two traps this left behind, both now guarded in the source

1. **The `survbits` memset is load-bearing; the `d_two` one next to it is not.**
   `k_intersect_compact` writes every word of `d_two` unconditionally, so
   clearing it first is redundant (~268 MB/q at I16 × 4 slabs, below noise).
   `survbits` is different: the small-region fallback still ORs into it.
   Dropping that clear by analogy yields stale survivor bits from the previous
   slab — extra two-sided survivors that look entirely plausible.
2. **A slab-continuation gate that compares a value with itself.** The obvious
   way to check the boundary step is `pl_next64(last) == x`, but the walk loop
   assigns `x = pl_next64(x)` from exactly that predecessor, so it compares
   `pl_next64(v)` with itself and cannot fail. `verify_walk_slabs` now checks
   the 32-bit result against the wide one instead, which is the property that
   can actually break.

## Finding 74 — the 5070 slab sweep completed, and a second sweep at double the area proves slab SIZE controls: `2^29` is the optimum at both areas, so the L40 split is real after all

**Date:** 2026-08-24, RTX 5070, **card otherwise idle**. `oracle/c194.job` +
`c194.roots1.m16`, `--pipeline --cofactor --logI 16 --J 32768 --qrange
80000023: --nq 10`, one binary, `--slab-j` swept. Two repeats per point, which
agree to better than 0.2% throughout.

Finding 72 left an open question: an L40 preferred `2^30` positions per slab
while the 3090 and 5070 were reported at `2^29`, and the 5070 evidence covered
only points at or below `2^29` — the disputed `2^30` and the old `2^31`
behaviour had never been measured on this card. Both ends are now filled in.

| `--slab-j` | slabs | local area | bucket array | fill | apply | sieve | **complete** | vs best |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32768 | 1 | `2^31` | 5.20 GB | 190.65 | 154.42 | 351.31 | **521.16** | +18.1% |
| 16384 | 2 | `2^30` | 2.60 GB | 138.35 | 156.17 | 300.80 | **465.92** | +5.6% |
| 10923 | 3 | `2^29.42` | 1.73 GB | 117.37 | 156.31 | 279.94 | **445.69** | +1.0% |
| **8192** | **4** | **`2^29`** | **1.30 GB** | **111.68** | **156.61** | **274.58** | **441.30** | **best** |
| 4096 | 8 | `2^28` | 0.65 GB | 112.22 | 156.39 | 274.99 | **454.44** | +3.0% |

**The default is right for this job, on both axes at once:** `2^29` is the
fastest point *and* uses a quarter of the `2^31` bucket array (1.30 vs
5.20 GB). The old monolithic behaviour is not merely memory-hungry, it is
**18% slower**.

> **Re-swept 2026-08-25 (finding 75) after `k_apply` got 12% faster, and
> `2^29` still wins.** The re-sweep is paired, extends down to `2^27` — which
> finding 74 never sampled — and confirms the optimum is an interior minimum
> bracketed on both sides. Its absolute numbers sit ~7% below this table's;
> see the note there before comparing the two.

> **RETRACTION AND RESOLUTION.** This finding first concluded that `2^30` being
> 5.6% slower here against 4.6% faster on the L40 made finding 72 "a genuine
> architecture split". **That reasoning was confounded** and was retracted:
> finding 72's L40 run is `I=J=2^16`, area `2^32`, while this sweep is area
> `2^31`, so the two were compared at equal slab SIZE and unequal slab COUNT —
> and this finding's own mechanism makes size and count separate terms. By
> count, both cards appeared to optimise at four slabs, which suggested the
> parameterisation itself was wrong.
>
> **The discriminating run was then done: the same 5070, same job, at `J 65536`
> — area `2^32`, matching the L40's.** If count controlled, the optimum should
> move to `2^30` (4 slabs). It did not.
>
> | `--slab-j` | slabs | local area | fill | apply | complete |
> |---:|---:|---|---:|---:|---:|
> | 32768 | 2 | `2^31` | 368.51 | 295.63 | 892.78 |
> | 16384 | 4 | `2^30` | 265.65 | 296.60 | 795.78 |
> | **8192** | **8** | **`2^29`** | **213.25** | **297.61** | **766.04** |
> | 4096 | 16 | `2^28` | 216.32 | 297.50 | 801.33 |
>
> **The optimum stayed at `2^29` POSITIONS while the count doubled from 4 to 8.
> Slab size controls; slab count does not.** The count hypothesis is dead and
> the existing `2^29` parameterisation is right.
>
> Normalising fill to cost per `2^29` of area makes it unambiguous — two sweeps
> at different total areas trace the same curve in slab size:
>
> | slab size | from area `2^31` | from area `2^32` |
> |---|---:|---:|
> | `2^31` | 47.66 | 46.06 |
> | `2^30` | 34.59 | 33.21 |
> | **`2^29`** | **27.92** | **26.66** |
> | `2^28` | 28.05 | 27.04 |
>
> **And that restores finding 72's conclusion on sound evidence.** At the
> matched area `2^32` the L40 optimises at `2^30` and this 5070 at `2^29`, so
> the two cards really do differ and neither optimum generalises — the original
> claim was right by a wrong route. One caveat survives: finding 72 does not
> name the L40's job and this sweep is c194, so a job difference between the two
> cards has not been excluded. Card-vs-card now needs the same job, not the same
> area.

### The curve has two mechanisms and they cross at 4 slabs

**Fill is the locality term and it saturates.** 190.65 → 138.35 → 117.37 →
111.68 → 112.22: a 41% improvement from `2^31` down to `2^29`, then nothing.
Whatever working set fill is thrashing at `2^31` fits by `2^29`.

**Apply does not care at all** — 154.42 to 156.61 across the entire 8x range,
a 1.4% drift in the *wrong* direction. Slab size is not an apply parameter.

**The per-slab tax is in trial division, and it is one stage.** Device TD
totals are 57.30, 56.71, 57.71, 58.59, 66.61 ms — flat to 4 slabs, then +8 ms.
Breaking that out:

| stage | 1 slab | 2 | 3 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| resieve + scatter | 40.09 | 38.93 | 38.25 | 37.34 | 37.08 |
| norms + trial division | 11.19 | 10.22 | 10.49 | 10.89 | 13.66 |
| **record candidate factorisations** | **1.39** | **2.66** | **3.99** | **5.35** | **10.64** |

`record candidate factorisations` is **linear in slab count at ~1.33 ms per
slab** — it is the per-slab fixed cost, not resieve, which actually gets
*faster* with smaller slabs (locality again, 40.09 → 37.08). It is the entire
reason `2^28` loses.

> **Mechanism, run down 2026-08-24 — and it is NOT what it looks like.** The
> obvious reading is per-launch overhead that batching across slabs would
> remove. Measured per *launch* (two per slab, one per side):
>
> | slabs | 1 | 2 | 3 | 4 | 8 |
> |---|---:|---:|---:|---:|---:|
> | ms per launch | 0.677 | 0.654 | 0.650 | 0.651 | 0.648 |
>
> **Dead flat at ~0.65 ms whether a launch carries 5,342 candidates or 577.**
> The cause is that `k_td` gives one thread per candidate and each thread
> marches the *entire* small-prime list — `nsm` is 6,726 / 6,542 on this job —
> so a launch costs one thread's march regardless of how many candidates ride
> along. It is latency-bound on `nsm`, not throughput-bound on `nacc`.
>
> Two things follow. **Sizing the k_td grid to `ceil(nacc/threads)` does not
> help** — tried, byte-identical, zero measured effect, because it scales the
> thread count and not the per-thread work. And **batching across slabs is the
> weaker of the two available fixes**: it removes `nslab-1` launches (3.9 ms/q
> at 4 slabs, 9.1 at 8), whereas splitting the `nsm` march across a warp per
> candidate would cut every launch ~32x (~5.0 ms/q at 4 slabs, ~10.1 at 8) with
> no cross-slab state, no per-candidate `j_base`, and no change to when
> `cof_enqueue` runs. They compose.
>
> Neither moves the optimum: projecting the batch across the sweep leaves 4
> slabs best at 421.6 ms/q against 8 slabs' 429.5. **This is ~0.9% at the
> operating point.** Its value is decoupling slab size from cost so that small
> slabs become affordable for memory reasons, which is an A=32-on-12-GB
> concern, not a throughput one. Neither fix was built.

**So the optimum is not a cache-size coincidence, it is a crossing.** Fill's
locality gain runs out at `2^29`; the recording pass's per-slab cost starts to
dominate at `2^28`. A card whose fill term keeps improving past `2^29` — a
larger L2, say — would cross later, which is exactly the L40's `2^30`
behaviour. That is a mechanism for finding 72's split rather than a
restatement of it, and it predicts the right knob: batch the recording pass and
the crossing moves to smaller slabs on every card.

## Finding 75 — `k_apply` was register-limited to 66.67% occupancy: one `__launch_bounds__` is worth −12.6% apply and −4.6% wall clock, uniformly across jobs

**Date:** 2026-08-25, RTX 5070 (sm_120), **card otherwise idle**. Paired A/B,
three repeats per geometry, relations byte-identical in nine of nine pairs.
Reference binary is `8d9d77b` (finding 73/74 as committed); the only source
difference is the annotation below plus the `--apply-threads` bound it forces.

Finding 74 left `apply` as the largest sieve stage and the one nothing had
moved: across the whole slab sweep it sat at 154–157 ms while `fill` fell 41%.
Finding 73's ballot scan then took 13% off it on c147 but only 5.4% on c194,
the job that matters. This finding is what `ncu` said when asked why.

### It was never memory-bound

Three geometries, `-k regex:k_apply`, HEAD binary:

| | c147 unslab | c147 slab | c194 |
|---|---:|---:|---:|
| Compute (SM) throughput | 73.55% | 73.49% | 72.94% |
| **DRAM throughput** | **8.52%** | **8.71%** | **8.90%** |
| L2 throughput | 3.31% | 3.34% | 6.01% |
| Executed IPC (of 4.0) | 2.90 | 2.88 | 2.91 |
| Achieved occupancy | 66.33% | 66.44% | 66.45% |

DRAM is idle. The kernel is issuing instructions at ~2.9 of 4 per cycle and
that is the entire story. Any plan that reorganises the bucket record format
or the read path is aimed at a bottleneck that does not exist here.

### The limiter is registers, and `ncu` names it

```
Block Limit Registers            2      <-- binding
Block Limit Shared Mem           3
Block Limit Warps                3
Theoretical Occupancy        66.67%
OPT  Est. Local Speedup: 33.33% ... limited by the number of required registers
```

At 45 registers x 512 threads only two blocks fit per SM. Shared memory would
allow three (33 KB x 3 = 99 KB of 100 KB — the budget `k_fill_l1` already
exploits).

**The budget is 40 registers, not the 42 the division gives.** Three blocks
need `65536/(512*3) = 42.67` per thread, but registers are allocated per warp
in units of 256 — 8 per thread — so a 42-register kernel is charged 48 and
gets `65536/(48*512) = 2.67` → **two** blocks and no gain whatsoever. 40 is the
first value that actually yields three, and it is where `ptxas` lands. Round
DOWN to a multiple of 8 before believing this arithmetic on another kernel.

`ptxas -v`, all nine instantiations the build emits:

| instantiation | HEAD | `__launch_bounds__(512,3)` |
|---|---:|---:|
| `<16,1,1,1>` **production, slabbed** | 45 | **40** |
| `<16,1,1,0>` production, unslabbed | 45 | **40** |
| `<8,1,1,0>` / `<8,0,1,0>` | 46 | 40 |
| `<16,0,1,0>` | 45 | 40 |
| `<16,1,0,0>` / `<16,0,0,0>` | 40 | 40 |
| `<8,1,0,0>` / `<8,0,0,0>` | 34 | 34 |
| **total spill bytes, whole TU** | **0** | **0** |

This is the exact mirror of `k_fill_l1`, where the same annotation had to be
**reverted** because a widened walk pushed it past the cap into 12 spill
stores. Same annotation, opposite sign. `-Xptxas -v` before and after, in both
directions, is the only thing that distinguishes the two cases.

Confirmed in the running binary — `ncu`, c194, before and after:

| | HEAD | patched |
|---|---:|---:|
| Registers per thread | 45 | **40** |
| Block Limit Registers | 2 | **3** |
| Theoretical occupancy | 66.67% | **100%** |
| Achieved occupancy | 66.45% | **99.66%** |
| Compute (SM) throughput | 72.94% | **84.71%** |

### End-to-end, paired

| geometry | stage | ref | new | delta |
|---|---|---:|---:|---:|
| c147 unslabbed | apply | 8.72 | 7.67 | **−12.1%** |
| | sieve | 17.57 | 16.50 | −6.1% |
| | wall | 40.89 | 38.69 | −5.4% |
| c147 slabbed | apply | 69.27 | 60.84 | **−12.2%** |
| | sieve | 125.23 | 116.75 | −6.8% |
| | wall | 176.01 | 167.16 | −5.0% |
| **c194 (production)** | **apply** | **148.60** | **129.89** | **−12.6%** |
| | **sieve** | **262.09** | **243.46** | **−7.1%** |
| | **wall** | **402.47** | **383.76** | **−4.6%** |

**The property finding 73 lacked is uniformity.** The ballot scan gave −13% on
c147 and −5.4% on c194; this gives its full −12.6% on c194, the largest of the
three. Repeat spread is under 0.05% (129.886 / 129.925 / 129.870). One
annotation is worth more wall clock on the production job than everything in
the finding 73/74 commit combined (−4.6% against −1.9%).

`--apply-threads` is now capped at 512 rather than 1024: `__launch_bounds__`'s
first argument is a hard launch ceiling, not a hint, so 1024 would fail the
launch outright. The bound is validated in `bench_main.cu` with a message
naming the cause.

### What this does NOT explain

Finding 73's c147-vs-c194 gap remains open and is now stranger, not clearer.
The three profiles are nearly identical, and per-launch instruction counts
differ by 3% for the same `2^29` positions (8.55G vs 8.80G warp instructions).
The obvious causes are ruled out; no replacement is offered.

One thing the instruction counts do settle: the threshold scan is only about
**3% of the kernel's instructions**, yet removing its atomics bought 5–13%.
That win was contention on global atomics, never instruction count — so there
is no second helping of it available by shaving the scan further.

### The build hid this for an hour, and would have buried it

The first A/B of this change came back **dead flat on all three geometries**,
with an `ncu` reading that still showed 45 registers. Both were artifacts of a
build that never ran. `make` in `bench/` had no `.DEFAULT_GOAL`, and the first
explicit rule in the Makefile is

```make
$(OBJS) $(TEST_OBJS): $(DEFS_STAMP) $(CF_LMAX_STAMP)
```

so make's default goal was the first word of `$(OBJS)` — **`bench_main.o`**. A
bare `make` compiled one object file, printed nothing alarming, and exited 0,
while `bench` kept the binary already on disk. The A/B then compared the
reference binary against itself and reported, correctly, that nothing had
changed.

`.DEFAULT_GOAL := all` is now set immediately above that rule. Two lessons
worth more than the fix:

1. **A zero exit from `make` is not evidence a build happened.** The binary's
   mtime is. `MAKE_EXIT=0` was checked and was true; the thing it stood proxy
   for was false. This is the same shape as the `pgrep -f` self-match that
   wasted an overnight run — trusting a status signal over the state it
   represents.
2. **A flat A/B deserves the same scrutiny as a surprising one.** A null result
   feels like the safe conclusion and gets less checking, which is exactly what
   makes it dangerous: a real −12.6% was one sentence away from being written
   off as "measured, no effect."

### The slab optimum is unchanged — re-swept paired, 2026-08-25

`apply` was the flat stage across finding 74's sweep, so a 12% cut to it could
in principle have moved the fill/apply balance that picked `2^29`. It does not.
Same job and geometry as finding 74 (`c194`, `--logI 16 --J 32768 --qrange
80000023: --nq 10`), both binaries interleaved at every point, two repeats
each, relations byte-identical at all six points. `2^27` is new — finding 74
stopped at `2^28` and never showed the lower turn.

| `--slab-j` | slabs | local area | fill ref → new | apply ref → new | **complete ref** | **complete new** |
|---:|---:|---|---:|---:|---:|---:|
| 32768 | 1 | `2^31` | 186.85 → 185.93 | 149.86 → 130.80 | 486.55 (+18.3%) | 467.77 (+19.2%) |
| 16384 | 2 | `2^30` | 136.48 → 135.19 | 150.83 → 131.13 | 436.48 (+6.1%) | 414.84 (+5.7%) |
| 10923 | 3 | `2^29.42` | 113.34 → 115.17 | 149.25 → 132.72 | 412.17 (+0.2%) | 398.29 (+1.5%) |
| **8192** | **4** | **`2^29`** | **108.89 → 109.40** | **150.52 → 132.44** | **411.32 best** | **392.28 best** |
| 4096 | 8 | `2^28` | 109.10 → 109.60 | 149.76 → 131.21 | 424.00 (+3.1%) | 404.39 (+3.1%) |
| 2048 | 16 | `2^27` | 125.31 → 125.39 | 150.70 → 131.84 | 473.18 (+15.0%) | 454.12 (+15.8%) |

**`2^29` wins on both binaries, and the curve is barely perturbed** — the
"vs best" column is within 1.3 points at every position. `SLAB_PERF_TARGET_LOG2
29` stands unchanged.

The reason is visible in the table: **`apply` is flat in slab size both before
and after** (149.25–150.83 ref, 130.80–132.72 new, no trend). Slabbing has
never moved `apply` and still does not; it moves `fill`, and `fill`'s shape is
untouched. A stage that is constant in the swept parameter cannot relocate that
parameter's optimum however much its constant changes.

The `2^27` point also fills in the lower half of the curve for the first time:
`fill` turns back up (109.60 → 125.39, +15%) once slabs get small enough that
per-slab fixed costs dominate. The optimum is a genuine interior minimum
bracketed on both sides, not the edge of the sampled range.

The per-point deltas confirm the A/B independently at six slab sizes of this
one geometry — not six geometries; the three-geometry claim above is the one
that spans jobs. `apply` −11.1% to −13.1%, `fill` within ±1.6% of zero (it is a
different kernel and should not move), wall −3.4% to −5.0%.

> **Do not compare these absolute numbers against finding 74's table.** That
> sweep reports 441.30 ms complete at `--slab-j 8192` where this one measures
> 411.32 for the same binary lineage — about 7% apart, with the gap
> concentrated in cofactorisation rather than in `fill` or `apply`. The two
> sweeps were taken under different machine conditions. Within either sweep the
> points are paired and comparable; across them only the *shape* is.

## Finding 76 — `k_fill_atomic` is L2-bound, its 50% occupancy ceiling is benign, and the block-count default was tuned for the wrong job: `1152 → 4608` is −8.6% fill, −5.7% wall

**Date:** 2026-08-25, RTX 5070, **card otherwise idle**. `oracle/c194.job`,
`--logI 16 --J 32768 --qrange 80000023: --nq 10`, one binary, `--fill-blocks`
swept, `--fill-threads 32` throughout unless stated. Relations byte-identical
at every point of every sweep below.

Finding 75 left `fill` as 27.9% of wall and the only large stage never
occupancy-profiled — `k_fill_l1` and `k_fill_l2` carry `__launch_bounds__`
while the shipping kernel carries none.

### The hypothesis was wrong: occupancy is not the lever

`ncu` on `k_fill_atomic`, c194:

| | value |
|---|---:|
| L2 cache throughput | **68.69%** |
| DRAM throughput | 24.66% |
| L2 hit rate | 93.60% |
| Compute (SM) throughput | **9.41%** |
| Executed IPC | **0.22** / 4.0 |
| Warp cycles per issued instruction | **101.07** |
| Avg. active threads per warp | 20.59 / 32 |
| **Block Limit SM** | **24** ← binding |
| Block Limit Warps / Registers / Shared | 48 / 64 / 32 |
| Theoretical occupancy | **50%** |

The opposite character to `k_apply`: memory-bound with the SMs idle, against
issue-bound with memory idle. Occupancy is capped at 50% because 32-thread
blocks are one warp each and at most 24 blocks are resident per SM — nothing to
do with registers (64) or shared memory (32).

**That ceiling was predicted to be the bottleneck, and is not.** The kernel is a
pure grid-stride over the global thread index (`bench_kernels.cu:109-110`), so
`576x64` assigns every thread exactly the same primes as `1152x32` at double
the occupancy. It is **slower**. Holding total threads at 73728:

| block width | resident warps/SM | occupancy | fill |
|---:|---:|---:|---:|
| **32** | 24 | **50%** | **101.84** |
| 64 | 48 | 100% | 106.70 |
| 128 | 48 | 100% | 105.67 |
| 256 | 48 | 100% | 107.30 |

Every 100%-occupancy configuration loses to the 50% one, and the same holds at
147456 threads (32thr 99.85 against 64thr 102.64). On an L2-bound kernel at
68.7% of its bandwidth ceiling, extra resident warps add contention on the
constrained resource rather than hiding the 101-cycle stall. **The 50% ceiling
is benign — `FILL_THREADS_DEFAULT 32` is correct and should not be "fixed".**

### The real lever is total threads, and the default was tuned on the wrong job

n=4, paired:

| `--fill-blocks` | fill | ± | Δfill | sieve | Δsieve | wall | Δwall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1152 (old default) | 108.34 | 1.02 | — | 244.62 | — | 419.51 | — |
| 2304 | 101.84 | 0.37 | −6.0% | 238.13 | −2.7% | 404.19 | −3.7% |
| **4608 (new default)** | **99.01** | 0.36 | **−8.6%** | **235.11** | **−3.9%** | **395.67** | **−5.7%** |
| 9216 | 99.85 | 0.16 | −7.8% | 235.78 | −3.6% | 396.12 | −5.6% |
| 18432 | 101.87 | 0.39 | −6.0% | 237.82 | −2.8% | 405.37 | −3.4% |

A bracketed interior minimum. Wall clock is the noisy column here — 1152 spans
35 ms across four repeats of *identical* configuration, against 13 ms for the
others — but the arms do not overlap: 4608's worst wall (400.67) beats 1152's
best (402.91), so the win survives the cofactorisation variance.

**Confirmed with the new default compiled in**, n=3 paired against an explicit
`--fill-blocks 1152`, relations byte-identical, arms again non-overlapping
(new max 401.06 against old min 410.80):

| stage | old 1152 | new 4608 | delta |
|---|---:|---:|---:|
| fill | 108.77 | 99.22 | **−8.8%** |
| sieve | 246.14 | 235.48 | −4.3% |
| wall | 416.60 | 399.21 | **−4.2%** |

**Quote the wall figure as −4 to −6%, not −5.7%.** The two runs disagree by
1.5 points on wall (−5.7% swept, −4.2% confirmed) because the 1152 arm draws a
long cofactorisation tail at different rates — its spread is 2-3x the 4608
arm's in both runs. `fill` is the stable measurement at −8.6% / −8.8%, and
`sieve` at −3.9% / −4.3%; wall inherits noise neither of them has.

`bench.h` predicted this in the sentence that shipped with the old constant:
*"The optimum plausibly moves with bucket and record count, which is not yet
measured."* 1152 was measured at 77.4M records; production runs far more.

### What the retune does NOT establish

| geometry | 1152 | 2304 | 4608 | 9216 | best |
|---|---:|---:|---:|---:|---|
| c147 unslabbed | — | −1.3% | −2.0% | **−2.9%** | 9216, still improving |
| c147 slabbed | — | +1.2% | +0.8% | +0.9% | **1152** |
| **c194 (production)** | — | −6.0% | **−8.6%** | −7.8% | **4608** |

**There is no single right constant.** c147 is nearly flat across an 8x block
range, but its two geometries disagree in *direction* despite sharing a factor
base: unslabbed wants more blocks, slabbed wants fewer (+0.8% at 4608 — small,
but ~2.7σ against a 0.15 ms spread). No mechanism explains why, and no
primes-per-thread formula fits, so none is offered.

The old constant was also validated on three cards (5070/4090/5090); **4608 is
measured on a 5070 only.** It is a better default for the I16 production class,
chosen deliberately at the cost of ~0.8% on c147-slabbed, and it is an argument
FOR the startup autotuner in STATUS item 2 rather than against it — the
autotuner is what gets every geometry its own optimum instead of trading one
hardcoded number for another.

> **Method note.** The hypothesis that motivated this finding — that the 50%
> occupancy ceiling was fill's bottleneck — was falsified by the first cell of
> the sweep. The win came from the axis that was being held fixed as a control.
> This is the second time in two days that the mechanism was wrong and the
> measurement was still worth taking (finding 73's 32-bit walk was the first).

## Finding 77 — the per-slab recording tax was `nsm`, not launches: one warp per candidate cuts the fixed cost 11.5x and turns the pass from `O(nsm)` into `O(nacc)`

**Date:** 2026-08-26, RTX 5070 (sm_120), **card otherwise idle**. `oracle/c194.job`
+ `c194.roots1.m16`, `--pipeline --cofactor --logI 16 --J 32768
--qrange 80000023: --nq 10`, `--slab-j` swept. One binary throughout: the old
path is `--td-record-scalar`, the new one is the default, so every pair below is
the same build, the same q, and the same factor base. **Relations are
byte-identical in every pair taken, at every slab count and on every job.**

Finding 74 diagnosed this and did not build it. The diagnosis held.

### The fix, and what it actually replaces

`k_td` gives one thread per candidate and each thread marches the entire
small-prime list. For the two dense passes that is right — they run over ~233K
survivors and are throughput-bound on `n`. For the **recording** pass it is not:
that pass runs `SELECT=1` over the ~5,300 joint candidates, so a launch costs
one thread's march of `nsm` (6,726 / 6,542 here) no matter how many candidates
ride along. Finding 74 measured exactly that — a dead-flat 0.65 ms per launch
from 5,342 candidates down to 577 — and predicted a ~32x win from splitting the
march across a warp per candidate.

`k_td_record_warp` (`td.cuh`) does that. One warp owns one candidate; the 32
lanes stride the entry list; `__ballot_sync` returns the hit mask in ascending
lane order, which **is** ascending entry order, so lane 0 walks it with `__ffs`
and divides in exactly the order `k_td` produces.

> **CORRECTION, 2026-08-26 (code review).** The sentence that stood here — "that
> ordering is the whole correctness argument, and it is why the relations stay
> byte-identical rather than merely equivalent" — **was wrong.** `td_divide_out`
> loops until `p` no longer divides, so an entry consumes the whole power of its
> prime whichever entry reaches it first, and distinct primes commute: the
> recorded multiset is order-invariant. Both emitters sort before writing
> (`cf_emit_sorted`, cofac.cuh:1613; `std::sort`, pipeline.cuh:1002), so the
> relation text cannot see the order either. Reversing the ballot walk would
> still produce byte-identical files. Order matters in exactly one place — when
> `nf` exceeds `fmax` the tail is dropped — and that is a hard error the host
> reports as "raise TD_FMAX". **The byte-identical result below stands and is
> still strong evidence** (it pins the factor multiset, the multiplicities, the
> predicate and the slab indexing across 3,349 relations and six configurations);
> it simply does not test ordering, and nothing needs it to. The practical
> consequence: **a lanes-per-item rewrite of the dense pass (item 18) is not
> constrained to preserve entry order**, which the old claim would have forced.

`bn_t` never leaves lane 0 — the divisions are sequential on
N and shuffling 256 bits between lanes would cost more than it saves — so the
other 31 lanes only ever evaluate the congruence. Their idling is free: the
serial part was one thread's work before the change too.

The shared `TD_TILE` staging is unchanged and still block-wide, which is what
keeps the entry read at one L2 hit per block rather than one per warp. The
`hitp[TD_MAXHIT]` buffer is gone — the ballot has already done the compaction —
and with it 192 bytes of stack frame. 68 registers, **0 spill stores, 0 spill
loads**, 32 B stack against `k_td<1,1,1>`'s 224 B.

### The prediction was 32x. The measured number is 11.5x, and the reason is worth more than the number

Per **launch** (two per slab, one per side). The candidates/launch column is
`nacc / nslab` with `nacc` the band mean of 5,342.5/q — candidates are not
distributed exactly evenly across slabs, so it is the right axis but not an
exact per-launch count; the fit residuals below bound how much that matters:

| slabs | candidates/launch | scalar | warp |
|---:|---:|---:|---:|
| 1 | 5,342 | 0.673 ms | 0.351 ms |
| 2 | 2,671 | 0.661 ms | 0.206 ms |
| **4** | **1,336** | **0.653 ms** | **0.131 ms** |
| 8 | 668 | 0.652 ms | 0.091 ms |
| 16 | 334 | 0.651 ms | 0.076 ms |

The scalar column is flat, reconfirming finding 74 on this binary. **The warp
column is not flat, and that is the result.** Least squares over those five
points:

```
warp launch = 56.6 us  +  55 ns per candidate      (residuals <= 2.5 us)
scalar launch = 653 us +  ~0 per candidate
```

So the pass stopped being `O(nsm)` and became `O(nacc)` with a small floor. The
32x applies to the *march*, which is no longer what a launch mostly costs; what
remains is the serial lane-0 work — norm, resieve divisions, small-prime
divisions — plus staging the tile, and that part is per candidate, real, and
was always there.

**The number that matters for slabbing is the FIXED term, because that is what
multiplies by slab count.** Two launches per slab:

| | per-slab tax |
|---|---:|
| scalar | 1,306 us |
| warp | **113 us** |
| | **11.5x** |

### The paired sweep: the optimum holds at `2^29` and the curve below it flattens

Three repeats per cell, means; `fill` is the control and does not move.

| local area | slabs | record scal | record warp | Δ | fill scal | fill warp | wall scal | wall warp | Δwall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2^31` | 1 | 1.347 | 0.702 | −47.8% | 180.57 | 180.54 | 425.57 | 424.80 | −0.18% |
| `2^30` | 2 | 2.643 | 0.825 | −68.8% | 126.25 | 126.42 | 373.57 | 371.87 | −0.46% |
| **`2^29`** | **4** | **5.222** | **1.047** | **−80.0%** | 98.87 | 98.84 | **349.69** | **345.87** | **−1.09%** |
| `2^28` | 8 | 10.425 | 1.459 | −86.0% | 99.46 | 99.43 | 361.71 | 351.86 | −2.72% |
| `2^27` | 16 | 20.838 | 2.443 | −88.3% | 115.16 | 115.02 | 405.25 | 387.58 | −4.36% |

At the production operating point the arms do not overlap: warp's worst wall
(346.14) beats scalar's best (348.93). `fill` agrees to 0.2% in every row, which
is the control working.

**`2^29` is still the optimum on both binaries.** What changed is the shape below
it — measured against each mode's own best:

| local area | scalar | warp |
|---|---:|---:|
| `2^31` | +21.7% | +22.8% |
| `2^30` | +6.8% | +7.5% |
| **`2^29`** | **best** | **best** |
| `2^28` | +3.4% | **+1.7%** |
| `2^27` | +15.9% | **+12.1%** |

Halving the slab now costs 1.7% instead of 3.4%, which is the point: it is the
memory decoupling item 8 wants, not a throughput win.

### The tax changed hands, and the new holder is the DENSE trial-division pass

This is the part that was not predicted. Warp-path stage totals, 4 → 8 → 16
slabs:

| stage | 4 | 8 | 16 | per slab |
|---|---:|---:|---:|---|
| record candidate factorisations | 1.04 | 1.43 | 2.43 | **0.12 ms** |
| **norms + trial division, both sides** | **10.34** | **13.18** | **21.83** | **~0.96 ms** |
| fill | 98.90 | 99.59 | 115.20 | turns up at 16 |

`norms + trial division` is `k_td<1,0,0>` — the same kernel, dense over
survivors — and it has the same disease. But the arithmetic is one step subtler
than "it goes latency-bound below `2^29`", and getting it right is what names the
fix.

`k_td` puts the `nsm` march **inside** its grid-stride loop, so a launch costs
`iters` marches, and `iters = ceil(n / (blocks*threads))`. With the default
`288 x 256` = 73,728 threads and 237,256 two-sided survivors per q:

| slabs | survivors/launch | `iters` | total `nsm` marches | measured TD |
|---:|---:|---:|---:|---:|
| 1 | 237,256 | 4 | 8 | 10.71 ms |
| 2 | 118,628 | 2 | 8 | 9.76 ms |
| 4 | 59,314 | **1** | 8 | 10.34 ms |
| 8 | 29,657 | 1 | **16** | 13.18 ms |
| 16 | 14,828 | 1 | **32** | 21.83 ms |

**The cost is flat exactly while the march count is, and rises exactly when it
does.** From 1 to 4 slabs the split is free because `iters` falls in lockstep
with it — twice as many launches, each doing half as many marches. At 4 slabs
`iters` hits its floor of **1** and has no further give, so from there every
halving of the slab doubles the total march count. That is the whole shape of
the column, and it is why the turn-up lands at 8 slabs rather than wherever the
cache happens to run out.

Finding 74 saw this column turn up at 8 slabs (11.19 / 10.22 / 10.49 / 10.89 /
13.66) and attributed the whole per-slab effect to the recording pass; it was
two passes with one mechanism, and removing the smaller one is what made the
larger one legible.

#### The model's own prediction, tested — confirmed one way, falsified the other

The model says the knee sits at `blocks*threads`, not at a slab size. Raising the
grid should make `iters` bottom out sooner, shorten the flat region, and — above
237,256 threads — remove it entirely. That is two runs rather than a kernel
rewrite, so it was run before building anything: three grids x five slab counts,
n=2, idle card, `--blocks` sweeping the whole pipeline's grid while the recording
pass (whose grid is derived from `nacc`) rides along as a control.

| slabs | predicted marches b288 / b512 / b1024 | TD b288 | b512 | b1024 | spread |
|---:|---|---:|---:|---:|---:|
| 1 | 8 / **4** / **2** | 10.85 | 9.56 | 9.15 | **−15.6%** |
| 2 | 8 / **4** / 4 | 9.87 | 9.43 | 9.32 | −5.5% |
| 4 | 8 / 8 / 8 | 10.49 | 10.38 | 10.43 | −0.6% |
| 8 | 16 / 16 / 16 | 13.39 | 13.31 | 13.38 | −0.0% |
| 16 | 32 / 32 / 32 | 21.82 | 21.96 | 22.36 | +2.5% |

**Confirmed, and strongly: where the march count is held equal, the grid does not
matter.** Across a 3.5x range of `--blocks`, the three bottom rows agree to 0.6%,
0.0% and 2.5%. Thread count is emphatically not the lever once `iters` is 1 —
which is the part of the model that identifies what the fix has to change.

**Falsified: the cost is not proportional to the march count.** The model has
b1024 doing a quarter of b288's marches at one slab and predicts roughly a
quarter of the cost; the measurement gives **−15.6%**, not −75%. Compare that
against the other direction, where doubling the marches 8 → 16 (4 → 8 slabs) cost
**+28%**. A march is not a fixed quantum of work, and the growth from 4 to 16
slabs is only partly the march count — per-launch efficiency falls as well, since
14,828 survivors leave 80% of a 73,728-thread grid idle.

No replacement model is offered. The march count predicts the *shape* — flat
while it is flat, rising where it rises, knee in the right place — and does not
predict magnitudes, and inventing a second model to fit five points would be
fitting noise.

**The decision this settles is the useful part.** `--blocks` is NOT a cheap
substitute for item 18. At one slab it is worth a real but small −1.7 ms on this
stage (−0.4% of wall); at every slabbed geometry — which is the entire case item
18 exists for — it is worth **nothing measurable**, because `iters` is already 1
and there is nothing left for a bigger grid to remove. The lanes-per-item kernel
change is the only thing that moves this, or nothing does.

> **Method note, the second in three days.** Finding 76's hypothesis was killed
> by the first cell of its sweep; this one was half killed by a sweep run
> specifically to try to kill it, before any code was written on the strength of
> it. The two-run version of "test the prediction first" cost 30 runs and saved a
> kernel rewrite aimed at the wrong quantity.

The fix is not the same, though, and should not be applied blind: warp-per-item
would *lose* on the dense pass wherever survivors exceed the thread count, which
is the common case at large slabs. What it wants is lanes-per-item chosen from
`n` against the grid — 1 lane when `n >= nthreads`, up to 32 when it is far
below. That is a bigger change to a hotter kernel than this one was, and it is
worth doing only for the same reason: it buys small slabs, not throughput.

The second term is `fill`, which stops improving at `2^29`, is flat to `2^28`
and is **16% worse at `2^27`** (115.20 against 98.90). Finding 74 called fill
saturated; over a wider sweep it reverses. That is the factor-base re-entry
every slab pays, and it is a floor on how small a slab can usefully get no
matter what happens to trial division.

### Correctness

Byte-identical relations, `--td-record-scalar` against default, one binary:

| job | geometry | slabs | relations | result |
|---|---|---:|---:|---|
| c194 | I16 J32768, 30 q | 1 | 3,349 | identical |
| c194 | I16 J32768, 30 q | 4 | 3,349 | identical |
| c194 | I16 J32768, 30 q | 16 | 3,349 | identical |
| c147 | I14 J8192, 5 q | 1 | 648 | identical |
| c147 | I14 J65536, 5 q | 2 | 1,297 | identical |
| **AS276** | **I17 J16384, 3 q** | **4** | **246** | **identical** |

(An earlier version of that row labelled the AS276 geometry "(A=32)". It is
not: `2^17 x 2^14` is `2^31`. Corrected 2026-09-01 — no `2^32` run has been
through the relation-identity gate.)

The three c194 runs also agree with each other as sorted sets, so slab count
does not move the relation population either. `--check-relations` rebuilds
**3,349 of 3,349** norms exactly from the warp path's factor lists. `make check`
passes end to end, `cofcheck.sh` included.

AS276 is the motivating case and shows the biggest absolute saving, because its
`nsm` is larger: **record 9.234 → 1.256 ms/q (−86%)** at 4 slabs, 4-limb
algebraic side, 3-limb rational, no width flags. That is ~8 ms/q returned on the
job that item 8 exists for.

### Block size: 256 threads, and 128 is worse

`TD_RECORD_THREADS` is a compile-time constant because the grid is derived from
it and `nacc`, so there is nothing a runtime flag could choose that the
candidate count does not already fix. It is a **bracketed interior minimum**,
n=2 at the 4-slab operating point, `DEFS=-DTD_RECORD_THREADS=N` pricing builds:

| threads/block | warps (candidates) per block | record |
|---:|---:|---:|
| 128 | 4 | 1.220 ms (+16.5%) |
| **256** | **8** | **1.047 ms** |
| 512 | 16 | 1.240 ms (+18.4%) |

Both directions lose and the two arms are within 0.013 ms of each other, which
is the shape of two costs crossing rather than a plateau. Below 256 the 16 KB
`TD_TILE` staging is shared across too few candidates and its per-candidate cost
rises; above it, 68 registers x 512 threads admits only one block per SM (an
**sm_120-only** figure -- sm_86 compiles this kernel to 60 registers and 64 B of
stack against k_td<1,1,1>'s 62 / 256 B, where two blocks of 512 do fit, so the
occupancy argument does not transfer to the other gencode targets), so the
staging is amortised better but there is far less to hide its latency behind.
The value either way is small — the whole pass is 0.3% of wall at the operating
point now — so this was swept once and left at 256.

## Finding 78 — `fill`'s per-slab floor is bandwidth on the factor base, not "re-entry": 42 B/entry x 22.1M entries = 929 MB per slab, and the slope the docs implied is 1.7x too steep

**Date:** 2026-08-26, RTX 5070 (sm_120), **card otherwise idle** (verified 0%,
29.1 W, no compute apps before the sweep; a first attempt was discarded, see the
method note). `oracle/c194.job` + `c194.roots1.m16`, `--pipeline --cofactor
--logI 16 --J 32768 --qrange 80000023: --nq 10`, **total area held fixed at
`2^31`** and `--slab-j` swept so that only the slab COUNT moves. Three repeats.

Findings 74 and 77 both attributed `fill`'s reversal below `2^29` to "the
factor-base re-entry every slab pays" (STATUS:219, STATUS:2144, RESULTS:5275).
That phrase was never measured — it was inferred from the shape of three points.
It is directionally right and quantitatively wrong.

### The sweep

| `--slab-j` | slabs | local area | fill | apply | dense TD | **complete** |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 1 | `2^31` | 184.17 | 131.19 | 10.91 | 439.63 |
| 16384 | 2 | `2^30` | 130.02 | 132.87 | 9.96 | 386.25 |
| 8192 | **4** | **`2^29`** | **99.85** | 131.63 | 10.39 | **355.65** |
| 4096 | 8 | `2^28` | 100.17 | 131.99 | 13.34 | 361.11 |
| 2048 | 16 | `2^27` | 115.97 | 132.29 | 22.03 | 396.27 |
| 1024 | 32 | `2^26` | 128.07 | 135.54 | 41.83 | 460.30 |
| 512 | 64 | `2^25` | 166.15 | 137.22 | 82.05 | 611.05 |

`apply` is flat across a 64x range in slab count and is the control. `2^29`
remains the optimum, now bracketed four doublings below instead of one.

### The prediction, written down first, and what happened to it

**H1, "fixed per-slab re-entry":** once locality saturates, `fill = A*n + C`.
Fitting on the only points the docs had, n=8 and n=16, gives **A = 1.975 ms per
slab**, C = 84.37 — and predicts 147.57 at 32 slabs and 210.77 at 64.

**H2, "it bends over":** the excess tracks entries that contribute nothing to
this slab, that count is bounded by `|FB|`, so growth decelerates.

| | n=32 | n=64 |
|---|---:|---:|
| H1 predicted | 147.57 | 210.77 |
| **actual** | **128.07** | **166.15** |
| error | **−13.2%** | **−21.2%** |

**H1 as stated is falsified and H2 is the right shape.** The local slope in the
4/8/16 window is 1.975 ms/slab; the asymptotic slope over 8..64 is **1.178
ms/slab**. Anyone extrapolating the floor from the three points the docs had
overestimates it by about 1.7x.

### The mechanism, with a number that lands

`k_fill_atomic` (bench_kernels.cu:134) is launched with `fb->n` **every slab**
and, before it can know whether a prime hits, unconditionally streams per entry:

| | bytes |
|---|---:|
| `plat[k]` (`plat_t` = 2x`uint64` + 2x`uint32`) | 24 |
| `slice[k]` (`uint16`) | 2 |
| `walk_cur[k]` read | 8 |
| `walk_next[k]` write | 8 |
| **total** | **42** |

x **22,121,650** bucketed entries (13,153,734 algebraic + 8,967,916 rational)
= **929 MB per slab**, independent of slab size. At 85% of the 5070's 672 GB/s
that is **1.63 ms/slab gross**. Measured net slope is 1.178, the difference
being that the area term keeps improving slightly as the slab shrinks and
partly cancels the fixed cost — which is also why the point-to-point slope
wobbles (0.08 / 1.98 / 0.76 / 1.19) while the endpoints fit cleanly.

So it is **bandwidth on the factor base, not launch overhead and not a "walk
re-entry" cost.** `--blocks` could never have touched it, and neither can
batching.

### The obvious fix does not pay, and the arithmetic is why

The kernel reads `plat[k]` and `slice[k]` *before* it reads `walk_cur[k]`, yet
the hit test is `walk_cur[k] < xmax` and a missing prime's entire body is
`walk_next[k] = walk_cur[k] - xmax`. Reordering — read the 8-byte walk state
first, early-out, and touch the 26 bytes of `plat`+`slice` only on a hit — costs
a non-hitting entry 16 B instead of 42. `k_transform` already writes
`walk_cur[k] = UINT64_MAX` for invalid entries (bench_kernels.cu:118), so even
the `PL_INVALID` test is available without reading `plat`.

It is not worth building. A prime `p > xmax` does not skip the slab, it hits
with probability `xmax/p`, so the skip fraction is far below the naive
`1 - pi(xmax)/pi(L)`:

| slabs | `xmax` | entries with no hit | MB/slab now | if reordered | saved |
|---:|---:|---:|---:|---:|---:|
| 8 | `2^28` | 0.0% | 929 | 929 | 0.00 ms |
| 16 | `2^27` | 7.1% | 929 | 888 | 0.07 ms |
| 32 | `2^26` | 29.3% | 929 | 761 | 0.29 ms |
| 64 | `2^25` | 52.0% | 929 | 630 | 0.52 ms |

At 64 slabs that is 33 ms of the 66 ms excess — half — but at **16 slabs, the
worst case a 12 GB card actually reaches for an A=32 job, it is 1.1 ms of 16.1**,
i.e. nothing. The floor is bandwidth on entries that genuinely do hit, and it is
close to irreducible without narrowing `plat_t`, which the 64-bit increment
requirement (plattice.cuh:30) forbids.

### What this does to item 18's ceiling

The two terms are near-equal partners below `2^29`, and item 18 can only touch
one of them:

| slabs | fill excess | dense TD excess | complete vs `2^29` | item 18 best case |
|---:|---:|---:|---:|---:|
| 8 | +0.31 | +2.95 | +1.5% | −0.8% |
| 16 | +16.11 | +11.64 | +11.4% | **−2.9%** |
| 32 | +28.21 | +31.44 | +29.4% | −6.8% |
| 64 | +66.30 | +71.65 | +71.8% | −11.7% |

Item 18's own TD numbers are confirmed — it projected ~2.8 ms at 8 slabs and
~11.5 at 16, against 2.95 and 11.64 measured. What is new is the other column:
**at 16 slabs, fill's irreducible excess is larger than the entire prize item 18
is chasing**, so the ceiling on lanes-per-item at the realistic A=32 geometry is
about 3% of wall, not the open-ended "buys small slabs" the item implies.

### Method note, the third in four days

The first attempt at this sweep produced fill=306 ms at one slab against 182 on
a clean re-run. Cause: a backgrounded script had been launched with a trailing
`&` *and* the harness's own background flag, so it survived a reported "exit 0"
and a second copy ran on top of it. Every number in that log was contended and
all of it was discarded rather than reported. The tell was the ordering — rows
interleaved out of sequence in a single output file. **A sweep whose rows do not
arrive in the order the loop emits them is not a slow sweep, it is two sweeps.**

## Finding 78b — `--blocks` is not a default worth changing, on slabbed OR unslabbed geometries

**Date:** 2026-08-26, same session and same idle card. Finding 77 closed off
`--blocks` for slabbed geometries (0.0–2.5% across a 3.5x range) but left one
cell open: the unslabbed row moved −15.6% of the dense TD stage. Whether that is
worth taking as a default was untested. Three repeats, two jobs.

**c194 I16 J32768 forced to 1 slab** (`--slab-j 32768`):

| `--blocks` | dense TD | wall | complete |
|---:|---:|---:|---:|
| 288 (auto) | 10.929 | 384.24 | 440.39 |
| 512 | 9.681 (−11.4%) | 380.85 (−0.88%) | 437.75 (−0.60%) |
| 1024 | 9.291 (**−15.0%**) | 382.27 (−0.51%) | 438.57 (−0.41%) |

**c147 I14 J8192** (natively unslabbed, `--nq 20`):

| `--blocks` | dense TD | wall | complete |
|---:|---:|---:|---:|
| 288 (auto) | 0.501 | 21.45 | 24.73 |
| 512 | 0.512 (+2.1%) | 21.37 (−0.39%) | 24.63 (−0.42%) |
| 1024 | 0.529 (**+5.6%**) | 21.76 (+1.45%) | 25.11 (+1.52%) |

The stage win on c194 is real and reproduces exactly (−15.0% against finding
77's −15.6%, and the three reps per arm do not overlap: 10.889–10.972 against
9.154–9.446). **It does not survive into wall time.** 1.6 ms of 384 is −0.51%,
against a rep-to-rep wall spread of 1.8–2.9% within a single arm — and the sign
flips between reps. On the small unslabbed job it **reverses**: b1024 is worst
on every column, because 1024x256 = 262K threads already exceeds the survivor
count and the extra blocks are idle.

**Verdict: leave `--blocks` on auto (6/SM).** This is the third measurement
pointing the same way and it strengthens item 18's core claim rather than
weakening it — **threads are not the lever.** Adding threads helps only where
`iters > 1` and hurts where `n` is already below the grid, which is exactly the
regime lanes-per-item exists to serve.

## Finding 79 — the `2^29` slab target is not a slab-size constant at all: it is 32,768 bucket regions, and `SLAB_PERF_TARGET_LOG2` is silently coupled to `log_region`

**Date:** 2026-08-26, RTX 5070, **card otherwise idle**. Same job and harness as
findings 77/78: `oracle/c194.job` + `c194.roots1.m16`, `--pipeline --cofactor
--logI 16 --J 32768 --qrange 80000023: --nq 10`, total area fixed at `2^31`.
`--region` x `--slab-j` swept jointly, two repeats.

Findings 74–78 all treat `2^29` positions/slab as the tuned quantity, and
`slab.h:38` hardcodes it as `SLAB_PERF_TARGET_LOG2`. The question this answers is
whether the optimum is really a *slab area*, or whether it is a *bucket region
count* that only looks like an area because `log_region` has never moved.

### The discriminator, stated before the runs

`nregion = slab_area >> log_region`. At the known optimum (region 14, area
`2^29`) that is **32,768**.

- **H_nregion** — the optimum is a fixed region count, so the optimal *area*
  halves with each halving of the region: `2^29` / `2^28` / `2^27` for regions
  14 / 13 / 12.
- **H_area** — the optimum is a property of the slab area, and `--region` does
  not move it: `2^29` for all three.

Judged on `fill`, not on `complete`: `--region` also resizes `k_apply`'s shared
tile, so `complete` carries a confounder that `fill` does not.

### fill (ms), mean of two reps — the minimum is the diagonal

| region | `2^31` | `2^30` | `2^29` | `2^28` | `2^27` | `2^26` |
|---:|---:|---:|---:|---:|---:|---:|
| **14** | 183.37 | 130.06 | **100.20** | 100.47 | 116.69 | 128.28 |
| **13** | 327.57 | 188.52 | 133.75 | **104.20** | 107.99 | 133.26 |
| **12** | 663.47 | 336.35 | 194.74 | 140.73 | **115.71** | 125.28 |

| region | optimal area | **nregion at the optimum** |
|---:|---:|---:|
| 14 | `2^29` | **32,768** |
| 13 | `2^28` | **32,768** |
| 12 | `2^27` | **32,768** |

**H_nregion confirmed 3 for 3, as predicted in advance.** The tuned quantity is
`32,768 bucket regions`. `2^29` is simply what that means when `log_region` is
14 — it has never been a slab-size constant, and the two findings that named it
one (74, 78) were reading a derived number as a primitive.

### But region 14 stays the default, because apply pulls the other way

At each region's own fill optimum:

| region | area | fill | apply | **complete** |
|---:|---:|---:|---:|---:|
| **14** | `2^29` | 100.20 | **131.81** | **360.96** |
| 13 | `2^28` | 104.20 | 200.81 (+52%) | 443.35 (+23%) |
| 12 | `2^27` | 115.71 | 338.83 (+157%) | 618.25 (+71%) |

`k_apply` launches one block per region, so halving the region doubles the block
count and re-enters the fused small-prime line sieve twice as often. Its cost
tracks `nregion` directly, in the opposite direction to fill's preference for
holding `nregion` fixed by shrinking the slab. **Region 14 is the joint optimum
on this card and the current defaults are right** — which is precisely why the
coupling below went unnoticed: at the default the two constants agree.

### The latent defect this exposes

`slab_perf_jmax` (slab.h:69) computes `rows = 2^29 / I` with no reference to
`cfg->log_region`. Verified directly — auto planning picks **8192 rows at every
region**:

```
--region 14: j-slabbing: 4 slabs, up to 8192 rows/slab   bucket array 32768 x 10658
--region 13: j-slabbing: 4 slabs, up to 8192 rows/slab   bucket array 65536 x 5457
--region 12: j-slabbing: 4 slabs, up to 8192 rows/slab   bucket array 131072 x 2856
```

so anyone who moves `--region` gets a slab target that is wrong by the same
factor:

| region | auto picks | fill | correct target | fill | penalty |
|---:|---:|---:|---:|---:|---:|
| 14 | `2^29` | 100.20 | `2^29` | 100.20 | — |
| 13 | `2^29` | 133.75 | `2^28` | 104.20 | **+28.4%** |
| 12 | `2^29` | 194.74 | `2^27` | 115.71 | **+68.3%** |

**This is latent, not a live production regression** — `--region` defaults to 14,
production never moves it, and moving it loses on `complete` anyway. But the
constant is written as though it were independent when it is not.
`SLAB_PERF_TARGET_LOG2` should be expressed as a region count and the row target
derived as `(SLAB_PERF_REGIONS << log_region) / I`, so the two cannot drift.
**Not built** — it is a behaviour-preserving refactor at the default and should
be taken with the autotune work (item 2), not on its own.

### What caps it at 32,768 is still not identified

Two candidates, and this sweep does not separate them:

- **L2 write frontier.** 32,768 append streams x one 128 B line = ~4 MB live.
  That fits the 3090's 6 MB L2 and would *not* fit at 65,536 streams (8 MB),
  which would explain the 3090. It does not explain the 5070, which has **48 MB**
  of L2 and still stops at 32,768.
- **Cursor atomic contention.** fill launches `FILL_BLOCKS_DEFAULT x
  FILL_THREADS_DEFAULT` = 4608 x 32 = **147,456 threads** against 32,768
  cursors — 4.5 threads per cursor. This is a launch-geometry property and is
  independent of L2, which would explain the 3090/5070 agreement across an 8x
  L2 range.

**The separating test is a `--fill-blocks` sweep**, which changes the thread
count without changing the cache footprint: if the optimal `nregion` tracks
thread count, it is contention; if it does not move, it is the cache. That is
one sweep and it is the obvious next one.

It also reframes the open "cache-aware automatic target" question in STATUS:
the card-dependent quantity to autotune is **`nregion`**, not slab size. The
L40's preference for `2^30` at region 14 is a preference for **65,536 regions**,
and stated that way it is a single number to measure per card rather than a
per-job geometry decision.

## Finding 80 — it is not cursor contention: a 16x sweep of fill threads does not move the optimum at all, and `fill = L(nregion) + 1.178 x nslab` collapses both sweeps

**Date:** 2026-08-26, RTX 5070, **card otherwise idle**. Same job/harness as
77–79. **Slab area pinned at `2^29`** (4 slabs) so finding 78's per-slab
factor-base re-stream is constant across the entire sweep; `nregion` moved by
`--region` alone; `--fill-threads` left at 32 so threads = `blocks x 32`.
Two repeats. `--region 16` is not testable at `logI 16`: `k_apply` asks for
131,328 B of shared memory against the device's 101,376 B opt-in ceiling.

Finding 79 left two candidate causes for the `nregion` optimum. This separates
them: `--fill-blocks` changes the thread count without changing the cache
footprint.

### fill (ms) — the argmin does not move

| blocks | threads | r15 (nr 16,384) | r14 (nr 32,768) | r13 (nr 65,536) | r12 (nr 131,072) |
|---:|---:|---:|---:|---:|---:|
| 1152 | 36,864 | **103.62** | 107.90 | 138.81 | 197.81 |
| 2304 | 73,728 | **98.07** | 101.52 | 135.95 | 196.64 |
| 4608 | 147,456 | **95.39** | 99.40 | 134.15 | 193.67 |
| 9216 | 294,912 | **96.11** | 100.02 | 135.26 | 195.49 |
| 18432 | 589,824 | **98.25** | 101.86 | 136.89 | 197.18 |

**Contention is falsified.** Across a **16x** range in thread count — 4.0 down
to 0.25 threads per cursor — the argmin is region 15 in all five arms. Threads
move the *level* of the curve but not its *position*. The 4.5-threads-per-cursor
coincidence in finding 79 was a coincidence.

The level effect is a shallow interior minimum at **4608 blocks in every region
arm** (1152/2304/**4608**/9216/18432 loses, wins, loses in all four columns), so
blocks and region are separable with no interaction — an independent
confirmation of the `FILL_BLOCKS_DEFAULT 4608` retune, on a geometry it was not
tuned against.

### The collapse

Subtracting finding 78's re-stream term and grouping by `nregion` alone — points
drawn from **two independent sweeps**, differing in region size, slab area and
slab count:

| `nregion` | `fill − 1.178 x nslab`, each measurement | mean | spread |
|---:|---|---:|---:|
| 8,192 | 95.6 97.8 | 96.70 | 2.4% |
| **16,384** | 87.6 89.1 90.7 91.0 | **89.61** | 3.9% |
| 32,768 | 94.7 94.8 95.5 96.9 | 95.45 | 2.3% |
| 65,536 | 127.7 129.0 129.4 131.3 | 129.37 | 2.8% |
| 131,072 | 182.2 186.2 189.0 190.0 | 186.83 | 4.2% |
| 262,144 | 326.4 334.0 | 330.19 | 2.3% |
| 524,288 | 662.3 | 662.29 | — |

**`fill = L(nregion) + 1.178 x nslab` holds to 2.3–4.2%** over configurations
whose slab counts differ by 32x. The locality term depends on the region count
and on nothing else; the rest is the re-stream. Above 32,768 regions `L`
roughly doubles per doubling of `nregion` — each additional open write stream
costs a full extra memory transaction.

### Correction to finding 79

Finding 79 reported the optimum as `nregion = 32,768`. That was the argmin of a
grid that did not contain the cell `(area 2^29, region 15)`. **`L` is minimised
at `nregion = 16,384`** (89.61 against 95.45). At `--region 14` the two are a
near-tie in *system* terms — 100.20 vs 100.47 in finding 79's table — because
reaching 16,384 regions there requires doubling the slab count, and the 5.8 ms
of locality gain is cancelled by 4.7 ms of extra re-stream. That near-tie is not
a coincidence, it is the two terms of the model crossing.

The finding-79 diagonal was real but was reading the *reachable* optimum. The
mechanism statement stands and is strengthened: the tuned quantity is a region
count, not a slab area.

### What is still open, and it is not answerable by timing

**L2 capacity is also falsified.** The write frontier at the optimum is
`16,384 x 128 B = 2 MB`, and at the old figure `4 MB` — against this card's
**48 MB** of L2. Capacity is not the binding resource here, and an 8x L2 range
(3090's 6 MB against this 5070's 48 MB) reportedly produces the same optimum,
which capacity cannot explain either.

What remains is **L2 set and sector geometry** — associativity-bound rather than
capacity-bound. Append streams that collide in the same sets evict each other's
partial 32 B sectors and force read-modify-write, and the number of streams that
survive is a function of ways and sectoring, which is roughly generation-
invariant while capacity is not. That fits the 3090/5070 agreement.

**It is not resolvable with more timing sweeps** — three have now been spent
narrowing it. The next step is `ncu` counters on `k_fill_atomic` at
`nregion` 16,384 / 32,768 / 65,536: L2 write hit rate, and sectors per write
request. If sectors/request climbs at the knee, it is RMW on evicted partial
lines and the mechanism is settled.

**Best fill measured anywhere in either sweep is `(area 2^29, region 15)` at
95.39 ms**, against the default's 99.40. It is not worth taking: `k_apply` costs
+41% at region 15 (184 vs 131 ms) and `complete` is 417 vs 367. The defaults
remain correct.

## Finding 81 — settled by `ncu`: the `nregion` knee is read-modify-write on partially-filled bucket lines. L2 reads are flat, DRAM reads rise 135%

**Date:** 2026-08-26, RTX 5070 (sm_120, CC 12.0), Nsight Compute 2026.1.1.
`k_fill_atomic` profiled directly, 8 launches per configuration. Slab area
pinned at `2^29` (4 slabs), `--fill-blocks 4608`, `--region` swept 15/14/13/12
= `nregion` 16,384 / 32,768 / 65,536 / 131,072. `--cofactor` dropped: it does
not touch fill and it tripled profiling time.

**The controlled comparison is exact.** Across the four configurations the
kernel reads the same `plat`/`slice`/`walk_cur` arrays over the same `fb->n`
at the same slab count, and writes the same total number of bucket records.
Only the number of open append streams changes. `dram__sectors_op_read` was
`n/a` under its Volta-era name; the CC 12.0 spelling is `dram__sectors_op_read`
(not `dram__sectors_read`), which is worth knowing before assuming a counter is
unsupported.

### The result

| `nregion` | L2 read sectors *(requested)* | DRAM read sectors *(fetched)* |
|---:|---:|---:|
| 16,384 | 23,449,366 | 18,518,388 |
| 32,768 | 23,450,025 (+0.0%) | 23,349,104 (**+26.1%**) |
| 65,536 | 24,210,058 (+3.2%) | 33,036,496 (**+78.4%**) |
| 131,072 | 23,413,638 (−0.2%) | 43,520,750 (**+135.0%**) |

**The kernel requests identical data at every point — L2 read sectors are flat
within 3% — while DRAM read traffic more than doubles.** Those extra reads are
lines nobody asked for. In a stream that only writes the bucket array, DRAM
reads caused by writing are read-modify-write, and there is no other
explanation available: the read side of the kernel is provably constant.

**H_RMW confirmed.** Bucket records are 4 B, a sector is 32 B, so eight
consecutive appends fill one sector. Past the knee a region's write frontier is
evicted before those eight arrive, and the next append to that line has to
re-fetch it.

### Write traffic against the theoretical floor

583M records/slab x 4 B = **2.333 GB** is the minimum bucket-write traffic per
slab (both sides, `Σ xmax/p` over both factor bases):

| `nregion` | measured DRAM writes | vs the floor |
|---:|---:|---:|
| **16,384** | 2.679 GB | **1.15x** |
| 32,768 | 2.996 GB | 1.28x |
| 65,536 | 3.578 GB | 1.53x |
| 131,072 | 4.569 GB | **1.96x** |

At the locality optimum the scatter is within 15% of perfect. At 131,072 regions
it costs nearly double, because evicted partial lines are written back and
re-fetched repeatedly.

### It accounts for the timing, quantitatively

Per q (4 slabs x 2 sides), against finding 80's locality term:

| `nregion` | DRAM read | DRAM write | total | x16k | fill (ms) | `L` (ms) | `L` x16k | eff GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 4.12 | 10.72 | 14.84 | 1.000 | 95.39 | 89.61 | 1.000 | 155.5 |
| 32,768 | 5.39 | 11.98 | 17.37 | 1.171 | 99.40 | 95.45 | 1.065 | 174.8 |
| 65,536 | 7.77 | 14.31 | 22.08 | 1.488 | 134.15 | 129.37 | 1.444 | 164.6 |
| 131,072 | 10.44 | 18.28 | 28.71 | 1.935 | 193.67 | 186.83 | 2.085 | 148.3 |

**Traffic ratio 1.00 / 1.17 / 1.49 / 1.94 against locality ratio 1.00 / 1.07 /
1.44 / 2.09**, with effective bandwidth flat within ±9%. fill is DRAM-traffic-
bound at every point measured, so the cost *is* the traffic, and `L(nregion)`
from finding 80 is a traffic curve rather than a latency or occupancy one.
~150–175 GB/s against the card's 672 GB/s peak is ~25%, which is what a
scattered atomic-append stream costs.

### Why the 3090 and the 5070 agree

Confirmed as an associativity/sectoring effect, not a capacity one. What binds
is how many append frontiers can stay resident without evicting each other,
which is a function of ways and sector geometry — roughly generation-invariant.
Total L2 capacity is not involved: the frontier at the optimum is ~2 MB against
this card's 48 MB. That is why a 6 MB 3090 and a 48 MB 5070 land in the same
place, and it means **the L40's preference for 65,536 regions needs its own
explanation** — capacity was never the reason, so "the L40 has more L2" is not
one. That remains open and is now the only open part of this question.

### No action follows at the operating point

The default (`region 14`, `nregion 32,768`) already runs at 1.28x the write
floor, and the best reachable point — `region 15`, 1.15x — is worth ~4 ms of
fill against **+53 ms of apply**. Confirmed again: the defaults are correct.

What this does change is the autotuner's objective (item 2). `L(nregion)` is a
DRAM-traffic curve, and traffic is predictable from `nregion` alone without
timing anything — so a startup autotune can price a candidate geometry from
counters, or from the model in finding 80, rather than by running trial fills.

**Three sweeps and one profile, and the mechanism is closed.** The timing
sweeps could bound it (contention out, capacity out) but could not identify it;
one `ncu` run on a controlled pair did, because it separates what the kernel
asked for from what the memory system actually moved.

## Finding 82 — `A = 32` at NFS@Home's own aspect ratio runs, slabs 8 ways, and validates: 1,322 of 1,322 relations rebuild both norms exactly

Item 8's remaining doubt was never the area — it was that nobody had run the
*shape*. Both prior `2^32` runs (finding 74's `I16 J65536`, finding 72's L40 at
`I=J=2^16`) were performance sweeps at an aspect ratio of our own choosing, and
neither went through the relation gate. Finding 69's rerun target — NFS@Home's
`I16e -J 16`, which in our coordinates is `2^17 x 2^15` — had never been sieved
at all, and finding 69's text still said we refused it.

**Run 2026-09-01, AS276 (C208), 10 special-q from q=80000023, `--logI 17
--J 32768 --maxbits 17`, `as276.roots1.m17`, `--cofactor`:**

```
  j-slabbing: 8 slabs, up to 4096 rows/slab
  bucket array 32768 x 9629 x 4 B = 1.18 GB, shared by both sides
  first-q validation  side 1: 9996 of 9996 PASS   side 0: 3618 of 3618 PASS
  1,322 relations, 132.20 per q
```

`bench --check-relations`: **1,322 of 1,322 rebuild both norms exactly, PASS.**

**The rectangle was confirmed from the relations, not from the flags** — the
whole point of `relgeom.py`, and the mistake findings 58/63 spent two sessions
on:

```
1322 mapped, 0 unmapped     i in [-65477, 65239]   j in [1, 32761]
rectangle: I = 2^17 x J = 2^15   (observed coverage, so a lower bound)
```

That is the same extent finding 69 recovered from GGNFS's own output for this
job (`±65477`, `1 .. 32761`). **`A = 32` at their shape is sieved, gated and
geometrically confirmed.**

`8 x 2^29` is what the planner picks unaided: `slab_perf_jmax` gives 4096 rows
at `logI 17`, and the safety bounds do not reduce it. No flag beyond the
geometry, no `--slab-j`, and the walk state carries across all eight slabs.

**No timing is quoted from this run.** The card was running someone else's job
at 96-100% throughout, which per finding 53 inflates wall clock without moving
the `cudaEvent` figures, and the device-wide memory headline is not ours to
read either. The per-stage setup allocation IS ours and is contention-free:
**bucket array 1.18 GB, factor bases + bitmaps 1.29, trial division context
0.09, cofactor queue 0.15 — 2.71 GB of setup.** Sizing this geometry properly
wants a rerun on an idle card, per the RUNBOOK's own rule about headline versus
breakdown.

**What this closes and what it does not.** It closes the last correctness
question in item 8: nothing in the `2^32` path is unexercised, and a 12 GB card
runs the NFS@Home geometry today. It does not make a like-for-like *performance*
comparison against NFS@Home — that needs an idle card and a matched q band, and
it is a different piece of work.

## Finding 83 — item 0's geometry hole closed: the margin is rectangle-invariant at ~3x, and the bigger rectangle is a rel/J loss for BOTH sievers

**Date:** 2026-09-01, RTX 5070 on finding 61's 950 mV curve, 16 `gnfs-lasieve4I15e`
workers on the 9800X3D, c183 `oracle/input.job`, algebraic special-q, band
`q = 190M` (above `alim`, so GGNFS cannot trim its base -- both run the identical
7,605,406 entries). Whole-box watts metered on the UPS, monitors (45 W)
subtracted. Neither siever ran while the other was on the box; the box was
otherwise idle apart from a decaying load average in the first minute.

Item 0 recorded three holes: only one geometry had a CPU comparator, there was
no control at 130M, and the GPU probes ran at host load 1.0-1.9. This closes the
first and third. **It also refutes the item's own premise.**

### The control reproduced August byte for byte

Arm A re-ran finding 71's exact geometry on today's binary:

```
md5  bc3fb27c93fd9affec8c50d4927b395f   q190M.rels      (2026-08-20)
md5  bc3fb27c93fd9affec8c50d4927b395f   gA_logI15.rels  (2026-09-01)
```

Same 419,946 raw, 419,845 unique, 41.984 per pair. **Every change since
2026-08-20 -- 12-limb norms, the 4608 fill default, `k_apply`'s launch bounds,
the warp recorder -- is performance-only on this job**, which nothing had
verified end to end before.

Time improved 100.95 -> 97.46 ms/pair (-3.5%), which is well short of what
findings 75, 76 and 77 compound to (-4.6%, -5.7%, -1.09%). Those were all
measured on **c194 at I16**; this is **c183 at I15e**, half the area and a
different factor base. Either the gains do not transfer, or the hotter card ate
them -- the board drew 152.8 W today against 133.5 W in August on the same
curve, in summer heat. Not separable from one session; do not assume those
speedups are universal.

### The 2x2

Finding 65's rule -- our rectangle = `2^(J_bits+1) x 2^(I_bits-1)` -- makes
`gnfs-lasieve4I15e -J 15` the same region as our `--logI 16 --J 16384`, so each
rectangle has both sievers on it.

| cell | ms/pair | uniq/pair | J/pair | **J/uniq rel** |
|---|---:|---:|---:|---:|
| GPU `2^15 x 2^14` | 97.46 | 41.984 | 24.61 | **0.5861** |
| GPU `2^16 x 2^14` | 180.26 | 62.729 | 46.87 | **0.7471** |
| CPU `2^15 x 2^14` (`-J 14`) | 301.47 | 41.953 | 71.15 | **1.6959** |
| CPU `2^16 x 2^14` (`-J 15`) | 546.66 | 62.681 | 129.83 | **2.0713** |

Whole box: GPU 252.5 W and 260 W, CPU 236 W and 237.5 W.

**Yield agrees to 0.07% and 0.08%** -- the same tight matched control finding 71
achieved, now at two rectangles instead of one. Both sievers find the same
relations.

| | time | energy |
|---|---:|---:|
| advantage at `2^15 x 2^14` | **3.09x** | **2.89x** |
| advantage at `2^16 x 2^14` | **3.03x** | **2.77x** |

### Doubling the area loses for both, and slightly more for us

| | time | yield | rel/J |
|---|---:|---:|---:|
| GPU | 1.850x | 1.494x | **0.785x** |
| GGNFS | 1.813x | 1.494x | **0.819x** |

**So item 0's assumption was wrong in two ways.** The bigger rectangle is not
better for us in absolute terms, and we pay slightly *more* than GGNFS to grow
it. The margin to quote remains each siever at its own best-known shape for this
job: **~3x time, ~2.8-2.9x energy**, now confirmed on two rectangles.

The identical 1.494x yield on both sievers says the rel/J loss from a bigger
rectangle is intrinsic to the mathematics -- larger norms, lower yield density --
not to either implementation.

**This does NOT test item 5's rule**, which is an EQUAL-AREA claim
(`2^16 x 2^14` against `2^15 x 2^15`, both `2^30`). Item 0's phrase "finding
65's `2^16 x 2^14` is our better rel/J shape" is sloppy and was read wrongly
here at first: it is the better of the `2^30` shapes, not better than the `2^29`
we deploy.

### Still open

No CPU control at 130M. And the GPU's 2.99x in finding 71 was measured on an
undervolted card at 133.5 W board; the same curve drew 152.8 W today, so
**ambient temperature moves the energy figure by several percent** and any rel/J
comparison across sessions needs the board draw quoted with it.

## Finding 84 — item 1 answered on the 5070: fill's knee is per-KERNEL, not per-device. Two concurrent fills run in 84% of serial time, and the gain saturates at two

**Date:** 2026-09-01, RTX 5070, c183 `oracle/input.job` at `--logI 15 --J 16384`,
`--reps 20`, idle box. Built as `--fill-streams N` in the standalone benchmark
(`bench_kernels.cu`); the production pipeline is untouched.

Every card swept plateaus at the same ABSOLUTE block count, which looks like a
device limit -- but the 4090 is **1.80x slower at fill than a 5070 with 1.5x its
bandwidth** (finding 51's table), which no device limit explains. The `ncu`
profile showed the mechanism candidate: `waves per SM = 1.00`, so the whole grid
is resident at once and a block that draws a heavy chunk has no queued block to
backfill its slot -- **SMs idle for 26.5% of elapsed cycles**.

Three arms, one binary, one geometry, **interleaved with each arm keeping its
minimum over three passes** -- see the drift note below. Each workspace gets its
own `plat`, cursor and bucket arrays; `slice` stays shared, as two real
special-q on one factor base would share it. (`primes` and `roots` are not read
by `k_fill_atomic` at all; an earlier draft of this finding listed them.)

| arm, N=2 | 3 invocations | per workspace |
|---|---|---:|
| CONCURRENT 2 x 4608 on 2 streams | 23.087 / 23.050 / 23.040 ms | **11.53 ms** |
| SERIAL 2 x 4608 on one stream | 27.194 / 27.073 / 27.202 ms | 13.58 ms |
| WIDE 1 x 9216, ONE workspace | 12.709 / 12.752 / 12.722 ms | 12.73 ms |
| single 4608, the standing control | 13.62 / 13.64 / 13.64 ms | 13.63 ms |

**concurrent/serial = 0.8490 / 0.8514 / 0.8470, mean 0.849, spread +-0.2%.**

**The knee is per-kernel.** Doubling ONE kernel's grid recovers 6.6% (13.63 ->
12.73); running TWO kernels concurrently recovers 15.4%. A single fill kernel
cannot saturate even this narrow card.

### Arm order was a real confound, worth about one point

The first version of this experiment ran the arms **once each in a fixed order,
CONCURRENT first**, and read 0.8418 / 0.8219 / 0.8477 / 0.8504, mean 0.840.
Boost clocks decay across the ~1.3 s each arm spends under load, so the first
arm is measured at the highest clocks -- a bias pointing exactly at the
conclusion. Interleaving the arms and taking each one's minimum over three
passes moves the ratio **0.840 -> 0.849** and tightens the spread from +-1.5%
to +-0.2%. **The effect is real and the correction is small**, but the original
number was flattering and every other A/B in this file uses best-of-N for
precisely this reason.

### It saturates at two streams

| N | concurrent/serial | per workspace |
|---:|---:|---:|
| 2 | 0.849 (3 invocations) | 11.53 ms |
| 4 | 0.843 | 11.53 ms |
| 8 | refused: needs 10.64 GB, 9.17 GB free | -- |

Four streams give **the same per-workspace time as two**, and WIDE gets *worse*
past 9216 blocks (18432 costs 13.68 ms against 9216's 12.73). So the idle
capacity is real but bounded: **one extra independent kernel fills it, a third
does not exist** -- on this card. N=8 needs a 24 GB+ card to test, which is the
same card the extrapolation below needs anyway.

### The 5090 answers the question the item was chartered to ask

Same binary sources, same job, same geometry, `GPU_ARCH=native CF_LMAX=3`
(which cannot touch `k_fill_atomic`), rented 5090, 2026-09-01.

| card | single 4608 | wide 9216 | best concurrent | saturates at | vs best single |
|---|---:|---:|---:|---|---:|
| RTX 5070, 48 SM, 12 GB | 13.63 | 12.73 | **11.53** (N=2) | N=2 | 9.4% |
| RTX 5090, 170 SM, 32 GB | 8.42 | 8.04 | **5.83** (N=4) | N=4 | **27.4%** |

5090 concurrent/serial: **0.7654 at N=2, 0.6959 at N=4, 0.6957 at N=8** -- it
saturates at four streams where the 5070 saturates at two, and N=8 reproduces
N=4 to 0.1% (5.827 against 5.832 ms per workspace). **The number of streams a
card wants is a property of the card**, not a constant.

**The mechanism is now unambiguous, because only fill fails to scale:**

| stage, 5070 -> 5090 | | ratio |
|---|---|---:|
| transform | 1.936 -> 0.514 ms | **3.77x** |
| apply | 21.827 -> 6.080 ms | **3.59x** |
| fill, one kernel | 13.63 -> 8.42 ms | **1.62x** |
| fill, concurrent | 11.53 -> 5.83 ms | **1.98x** |

Transform and apply both track the SM ratio (170/48 = 3.54x). Fill returns
1.62x for 3.54x the SMs, and concurrency lifts it to 1.98x -- roughly 40% of
the way to the 2.67x nameplate bandwidth ratio (1792 / 672 GB/s). **The
plateau was never a device limit.** It is one kernel failing to keep a wide
card busy, which is what `waves per SM = 1.00` and 26.5% idle SM cycles said.

This also retires the standing puzzle in the "Measured" section of STATUS: "the
5090 has 3.5x the SMs of a 5070 and returns far less than that on fill" now has
a cause and a partial remedy, rather than a list of refuted hypotheses.

### What it is worth, stated honestly

**On the 5070 it is not worth building.** Best single-kernel per workspace is
12.73 ms (9216 blocks) against concurrent 11.53 -- 9.4% off fill, and fill is
~23% of wall there, so roughly **2% of wall**.

**On the 5090 it is a different proposition, and the pipeline run settles the
wall-clock share.** 2,000 q of the same c183 band at `I15e`, `--cofactor`,
both sides:

| stage | 5070 | 5090 | ratio |
|---|---:|---:|---:|
| wall clock per q | 97.46 | **44.19** | 2.21x |
| transform + plattice | 3.108 | 0.928 | 3.35x |
| apply | 32.898 | 9.453 | **3.48x** |
| **fill** | 25.962 | **15.607** | **1.66x** |
| resieve + scatter | 8.997 | 5.964 | 1.51x |
| TD + classify | 14.51 | 7.84 | 1.85x |
| cofactorisation | 12.88 | 5.89 | 2.19x |

**Apply and transform scale with the SM count (3.54x). Fill returns 1.66x.**
Because everything else scaled and fill did not, fill's share of wall *grows*
with the card: **26.6% on the 5070, 35.3% on the 5090.**

At the measured 27.4% cut against the best single-kernel configuration, that is
**4.28 ms of 44.19 = 9.7% of wall** on a 5090 (10.8% against the shipped 4608
default). That is a production-sized number, and it is the one this item needed.

**A second stage is now visible in the same shape:** `resieve + scatter` scales
1.51x, worse than fill, and is 5.96 ms = 13.5% of the 5090's wall. It is
bucket-structured work like fill. Nobody has looked at it under this lens; it
is not part of this item, but it is the next place to look.

### The anomaly geometry makes the effect BIGGER

`c147` at `--logI 14 --J 8192` -- the geometry where the 4090 came out 1.80x
slower than a 5070 -- on the 5090, N=4:

```
CONCURRENT 4 x 4608, 4 streams :  6.451 ms  =  1.613 ms per workspace
SERIAL     4 x 4608, one stream: 10.798 ms  =  2.699
WIDE       1 x 18432            :  2.677 ms
concurrent/serial 0.5975
```

**39.7% off fill**, against 27.4% at the larger `c183 I15e` geometry. The
underfeeding gets WORSE as the per-kernel work gets smaller -- 8,192 regions
here against 32,768 -- which is the regime small jobs and heavily slabbed
geometries live in. A wide card running a small geometry is the worst case for
one kernel, and the best case for this change.

**The prediction that motivated this held.** It was: if one kernel leaves 15%
idle on the narrowest card, a wide card should show more. It shows twice as
much, and it wants twice the streams. **Wide cards are underfed, not poorly
suited** -- which is the opposite of what the flat rel/J between a 5070 and a
5090 (finding 47) implied, and it means the per-joule case for the wide card
was being made against a handicapped configuration.

**Still worth running on a 4090**, which is the remaining anomaly: it is 1.80x
SLOWER at fill than a 5070 despite 1.5x the bandwidth, and this result predicts
it should recover a large fraction of that under concurrency. If it does not,
Ada has a second mechanism that Blackwell does not.

### Which pairing production would use

The measurement is pairing-agnostic: it shows independent fills overlap, not
which two. **The two SIDES of one special-q are the cheaper candidate** -- they
already share the factor bases and run sequentially through one bucket
allocation today, so pairing them needs no second special-q's host state. Two
special-q is the symmetric option and doubles more. Neither is worth designing
until the wide-card numbers exist.

### What this does not model, stated more carefully than the first draft

The workspaces hold **identical plat values**, and the values *are* the walk:
`pl_first64`/`pl_next64` derive every hit position from them, so two real
special-q would write different per-region distributions while these march their
bucket frontiers in lockstep. Finding 81 established that fill is bound by
read-modify-write on partly filled bucket lines -- i.e. by exactly that
distribution. So this measures the **saturation** question honestly (N kernels,
N x 42 B/entry from distinct addresses, no shared-read advantage) and does
**not** predict how two real special-q interleave. That needs the production
pipeline, and it is a reason to re-measure there before shipping, not a reason
to doubt the 0.849.

## Finding 85 — the `--fill-blocks` default is confirmed in-band on c194, and the ladder's WRONG-ANSWER failure is pinned to its repeat-fill regime rather than to sample count: a 10-q band ranks the axis correctly on both jobs

**Date:** 2026-09-02, RTX 5070, **card idle but the CPU busy throughout** (see
the spread discussion below — this matters, and not uniformly). One binary,
`57480cd`. Arms `{2304, 4608, 9216, 18432}` interleaved *within* each rep so
drift hits all four equally. Decision metric is the pipeline's own `fill` line,
which is `cudaEventElapsedTime(ev[1], ev[2])` (`pipeline.cuh:566`) with no host
sync inside the window.

STATUS item 2 left two questions open. This settles both.

### 1. Is 4608 right for the job it was chosen on? Yes.

`oracle/c194.job` + `c194.roots1.m16`, `--logI 16 --J 32768
--qrange 80000023: `, i.e. finding 76's configuration. Slabbed, 4 slabs.

| `--fill-blocks` | in-band `--nq 200`, n=3 | spread | `--nq 10`, n=4 | spread |
|---:|---:|---:|---:|---:|
| 2304 | 102.343 | 0.72% | 103.233 | 0.67% |
| **4608 (shipped)** | **99.872** | 0.57% | **100.535** | 0.92% |
| 9216 | 102.200 | 0.28% | 102.277 | 1.71% |
| 18432 | 103.492 | 0.18% | 103.653 | 1.04% |

**4608 wins both regimes and the arms do not overlap**: in-band its worst rep
(100.104) beats the runner-up's best (102.032) by 1.9 ms. Margin over 9216 is
2.3% in-band. The full *ranking* is also identical across regimes —
`4608 < 9216 < 2304 < 18432`.

Finding 76's `--nq 10` sweep therefore predicted the in-band optimum exactly.
**The "4608 is unvalidated in-band" worry in item 2 is closed: it is validated.**

Absolute fill is above finding 76's August numbers on every arm, but **not
uniformly**: +1.37% (2304), +1.54% (4608), +2.43% (9216), +1.75% (18432).
Ambient explains the level — this box drew 152.8 W board in September against
133.5 W in August at the same curve — but not the spread across arms.

**That non-uniformity is the strongest caveat in this finding, and it bounds
what any single tune is worth.** Between two runs of the *identical* `--nq 10`
protocol on the same card, job and geometry, the 4608-vs-9216 gap went from
**0.84% in August to 1.70% in September** — a factor of two. Rankings held both
times, so the ordering is robust; the *margin* is not. Any tuner that compares
its measured margin against a threshold is comparing a quantity that has been
observed to double run-to-run.

### 2. Was the ladder's failure about sample count, or about its regime? Its regime.

This is the controlled version of the question. `oracle/input.job` +
`c183.fb1`, `--logI 15 --J 16384 --maxbits 15 --qrange 190000000: ` —
unslabbed: area `2^29` against a split trigger of
`(SLAB_PERF_REGIONS << log_region) * 2`, which is `2^30` **at the default
`--region 14` only** — stating that trigger as an absolute `2^30` is precisely
the bug finding 79 fixed (`slab.h:64-69`), so replaying this configuration at
`--region 12` would slab it. **The same job,
geometry and axis on which the deleted ladder picked 18432 and the band wanted
9216.**

| `--fill-blocks` | in-band `--nq 200`, n=4 | spread | `--nq 10`, n=5 | spread |
|---:|---:|---:|---:|---:|
| 2304 | 26.097 | 2.34% | 25.960 | 3.19% |
| 4608 (shipped) | 26.438 | 3.35% | 26.456 | 3.97% |
| **9216** | **25.799** | 2.35% | **25.301** | 4.09% |
| 18432 | 26.443 | 1.46% | 25.946 | 1.68% |

**Both regimes pick 9216**, reproducing yesterday's independent band result
(2.4% here against 2.1% then). The 10-q band gets the right answer on the very
case the ladder got wrong.

**Two limits on that, both load-bearing for the tuner design.** First, only
*first place* agrees here: `--nq 10` ranks 9216 < 18432 < 2304 < 4608 while
in-band ranks 9216 < 2304 < 4608 < 18432. The full-ranking agreement in the
c194 table is a c194 result, not a general one. Second, the **margin** the two
regimes report differs by ~1.8x — 4.37% at `--nq 10` against 2.42% in-band —
and on c194 it goes the other way (1.70% against 2.28%). A short band is a
reliable *ranker* and an unreliable *estimator of effect size*.

That eliminates sample count as the explanation and leaves the ladder's own
structure — repeated fills of ONE lattice, back-to-back, cache-warm — as the
cause. Item 2 hypothesised exactly this; it is now supported by a controlled
comparison rather than by inference. **A short band is not a bad measurement.
A repeat-fill proxy is.**

Note also that 4608, the shipped default, is at the bottom of the four arms on
this job: **last outright at `--nq 10`, and in-band indistinguishable from
18432 for last** (26.438 against 26.443 — a 0.02% gap, far inside these arms'
1.46-3.35% spreads). It is not third-and-comfortable; it is tied for worst.
That is not a defect: finding 76 established the
optimum is geometry-dependent, and c194/I16 is the production class the default
serves. It is a reason to tune per geometry, not to move the constant.

### Interleaving is doing the heavy lifting, and that is the design lesson

Ranking every rep individually — 16 interleaved reps across two jobs and two
regimes — **15 of 16 picked the in-band winner outright.** The one miss (c183,
in-band, rep 4) took 2304 over 9216 by 0.05 ms, 0.19%, between the two best
arms.

Single reps rank correctly despite within-arm spreads of up to 4.09% because
interleaving cancels correlated drift: whatever the host or the clocks are
doing during a rep, all four arms see it. **A tuner that interleaves its
candidates needs far fewer reps than one that runs each candidate to
completion in turn.** The redesign in item 2 — first N q of the band at each
candidate grid — is validated at N=10, half the 20 it proposed.

### Contention hit the small geometry and not the large one

Within-arm spreads across **both** regimes were **0.18–1.71% on c194** and
**1.46–4.09% on c183**, on the same busy box in the same session. Yesterday's
idle-box c183 band measured 0.18%, so contention inflated c183 by **8x to
23x** — and the worst of those (4.09%) is 9216 at `--nq 10`, an arm a tuner
would rely on.

**The mechanism is NOT established, and the obvious explanation does not
survive its own arithmetic.** The tempting story is finding 4's: host work per
q is roughly fixed while GPU work is not, so a small geometry is more exposed
(c183/I15e runs 26 ms of fill and 104 ms of wall per q; c194/I16 runs 100 ms
and 385 ms, a 3.8x ratio). And the fill event window is genuinely not pure
kernel time — `ev[1]` is recorded, then two `cudaMemset`s and the launch are
issued (`pipeline.cuh:554-560`), so a host stall between them lands inside the
measurement.

But that window is entered **once per side per slab**
(`pipe_side_sieve_slab`, called at `pipeline.cuh:1790,1794`), so slabbed c194
has 8 host-issued windows per q against unslabbed c183's 2. A 4x increase in
exposure against a 3.8x increase in GPU work per window roughly cancels, and
the ratio argument predicts no difference at all. The observation is solid;
the causal story is not. **Do not use it to justify skipping an idle box.**

**Operationally, as an observation rather than a rule:** `fill` was
contention-tolerant on c194/I16 and contention-sensitive on c183/I15e in this
session. On c183 the effect being measured (2.4%) was smaller than a single
arm's spread (3.35%); only interleaving made it readable. Since the mechanism
is unresolved, treat this as "measured on these two geometries", not as a
property that transfers to a third.

### Method note, and a correction to how this was framed

The framing that opened this session — "the shipped default came from the same
regime that produced yesterday's wrong answers" — was wrong, and worth
recording because it nearly aimed the experiment at the wrong target. Finding
76's sweep used `--nq 10`, which is a *band*: ten distinct special-q, each with
a freshly transformed `plat`, each filled once. The ladder ran one lattice
repeatedly in-process. Those differ in kind, not in length, and the tables
above show the length axis was never the problem.

### Output-equivalence evidence — weaker than byte-identity, and incomplete

**No byte-identity check was run in this experiment.** What was captured is
`COMPLETE RELATIONS/q`, identical across all four arms within each regime on
c183 (7.410 in-band, 6.200 at `--nq 10`). Equal relation counts are much weaker
than equal relations.

The c194 column was lost to a bad field extraction, so **the in-band c194 arms
— the ones carrying the "do not move the constant" conclusion — have no
output-equivalence evidence from this run at all.** The fallback is finding
76's byte-identity check on c194, which was taken at `--nq 10`; since this
finding's whole point is that regimes can differ, that fallback is an argument
by analogy rather than a check. `k_fill_atomic`'s output is block-count-
invariant by construction (a pure grid-stride over the same primes), which is
why this is a documentation gap rather than a suspected defect — but it should
be closed the next time these arms are run.

## Finding 86 — item 9's two defects no longer exist in the pipeline path, and startup on the client's job class is 1.9 s

**Date:** 2026-09-02, RTX 5070, idle box, commit `57480cd` + doc edits.
Whole-process wall timed with `date +%s.%N` around `./bench --pipeline
--cofactor`, `--nq 1` against `--nq 21`, three reps, first rep discarded as
cold page cache (the c194 factor base is a 205 MB file).

Item 9 asserted two specific defects and estimated 15-20 s of startup on
snfs236, and was **promoted 2026-09-01** on the reasoning that the distributed
client spawns a fresh `bench` per work unit and so pays them every ~15 minutes.
Both assertions are stale.

### The code: one parse, one `rfb_build`

In a `--pipeline` run the whole factor-base setup lives inside
`if (cfg.pipeline) {` (`bench_main.cu:1873-2871`), and inside it:

The `fb1` load is a **three-way** branch (`:2638-2660`), and `--fb1` is an
alias for `--cadofb` (`:1079`), so the runs measured here — and the
distributed client, which passes an explicit `--fb1` — take the first arm:

| branch | what | taken here |
|---|---|---|
| `cfg.cadofb` | `fb_load_cado(...)` `:2640` — fills logp internally (`fb_cado.c:449-465`) | **yes** |
| `fbpath_set` | `fb_load(...)` `:2645` + `fb_fill_logp(&fb1)` `:2646`, legacy `--fb` | no |
| else | `afb_build_gpu(...)` `:2656`, in-process GPU generation | no |

and then, unconditionally:

| what | where | times |
|---|---|---:|
| `rfb_build(&POLY, rlim, ...)` | `:2726` | 1 |
| `fb_fill_logp(&fb0)` | `:2727` | 1 |

**No factor base is loaded anywhere before line 1873**, so the "throwaway first
parse that used to supply the q list" is gone — the `used to` in item 9's own
wording turns out to be the whole story.

The second `rfb_build` at `:2889` is real but unreachable here: it sits in the
`else` of `if (cfg.side == 1)` (`:2876`), a **sibling** of the pipeline branch
at the same brace depth, i.e. the standalone non-pipeline path. It is not a
second call in one run; it is the same call in the other run mode.

### The measurement agrees, including the part that would have shown the defect

Fitting `T(n) = S + n*p` from the two points:

| config | `T(1)` | `T(21)` | fitted per-q | **startup `S`** | of a 900 s work unit |
|---|---:|---:|---:|---:|---:|
| c183, `--sq-side 1` | 2.003 s | 3.930 s | 96.4 ms | **1.91 s** | 0.21% |
| c183, `--sq-side 0` | 1.978 s | 3.847 s | 93.5 ms | **1.88 s** | 0.21% |
| c194, `--sq-side 1` | 3.710 s | 8.952 s | 262 ms | **~3.45 s** | 0.38% |

The c183 fit is tight and self-checking: reps reproduce to 4 ms at `--nq 1` and
21 ms at `--nq 21`, and the fitted 96.4 ms/q lands on the band's own
97.5 ms/q.

**The `--sq-side` pair is the direct test.** Item 9's dead parse was specific
to `--sq-side 0`, so that column should have been the expensive one. It is
0.022 s *cheaper*. That is not "noise" — reps reproduce to 4 ms at `T(1)`, so
22 ms is outside the rep spread, and with n=2 per point after discarding the
cold rep there is no uncertainty on `S` to quote. It is **immaterial**: both
values are ~1.9 s and the sign is the wrong way for the defect. There is no
asymmetry to remove.

**And the pair was run on c183 only.** Item 9's cost was described as scaling
with factor-base size, so an asymmetry would show most strongly on c194's
`rlim` 160M — the arm not run. The closure rests on the code reading, which is
independent of base size; the `--sq-side` pair corroborates it on one job
rather than establishing it on all.

**c194's numbers are approximate.** Its `T(21)` spans 8.19-11.10 s across three
reps, so the fitted 262 ms/q disagrees with the band's 385 ms/q and the fit is
not trustworthy. What survives is a floor from the reproducible `T(1)`:
startup is ~3.3-3.5 s. Enough to bound the item; not a number to quote further.

### Disposition

Startup grows with the factor-base bounds, not with ms/q — c183 (`rlim` 67.1M)
pays 1.91 s and c194 (`rlim` 160M) ~3.45 s. It is **not** proportional: 2.39x
the `rlim` buys 1.81x the startup, so there is a large `rlim`-independent
intercept (CUDA context creation, allocations) that two points cannot
separate from the slope. Only monotonicity is needed here, and it holds: the
client's job class is *smaller* than c183, so it pays **at most ~1.9 s of a
~900 s work unit, 0.2%**.

The 15-20 s figure was measured on **snfs236, which was not re-measured
here** — nothing in this finding refutes it for that base, and an
snfs236-class job should be timed rather than assumed. What it is not is a
number the client pays: its base is far larger than anything the client
sieves. **Item 9 is closed: the defects are gone and the residual on the
client's job class is 0.2-0.4%.** Nothing here is worth engineering.

**Method note.** This item survived in the list for weeks describing code that
had been restructured underneath it, and was *promoted* on that stale reading.
The measurement took eight minutes; the code reading that made sense of it took
about the same. Both were cheaper than the promotion.

## Finding 87 — item 0's last hole closed: the matched CPU control at q=130M gives 2.90x time and 2.77x whole-box relations per joule, both sides measured this session

**Date:** 2026-09-02, RTX 5070 + 9800X3D, idle box, GPU and CPU never running
together. Whole-box watts on the UPS, monitors (45 W) subtracted throughout.

Finding 71 ran GPU probes at 50M / 130M / 190M and matched CPU controls at 50M
and 190M only. Finding 83 closed two of the three remaining holes. **This
closes the third.**

### The row

| q = 130M | GPU | CPU, 16 workers | advantage |
|---|---:|---:|---:|
| wall ms/pair | 103.13 | 298.98 | **2.90x** |
| unique rel/pair | 46.095 | 45.927 | 1.0037 |
| whole box, at the wall | 252.5 W | 240 W | |
| J/pair | 26.04 | 71.76 | 2.76x |
| **J per unique relation** | **0.5649** | **1.5624** | **2.77x** |

Yield agrees to **0.37%**; our dedup is 1.0004 (461,143 raw, 460,952 unique).
The CPU band ran 10,053 pairs against the GPU's 10,000 — the width formula
`10000 * ln q` overshoots the GPU's actual span by ~5%, as it does for the 50M
and 190M controls, and both columns are normalised per pair.

Against finding 83's `2^15 x 2^14` row (**3.09x / 2.89x** at 190M), 130M sits
slightly lower, which is the direction finding 71's mechanism predicts: GGNFS
truncates its base at q, so at 130M it still sieves only 97% of the base and at
190M all of it. The CPU control now shows that drift directly —
**271.95 -> 298.98 -> 301.47 ms/pair** across 50M / 130M / 190M while its yield
falls **46.44 -> 45.93 -> 41.95**.

### The August GPU probe was re-run, and the reason it seemed necessary was wrong

`q130M.log` (2026-08-20) ran gate scale **1.2750/1.9250** against September arm
A's **1.2500/1.9000**, and `fill=1152` against today's 4608. Pairing it with a
September CPU control looked unsafe, so it was re-run on the current binary
with arm A's exact flags.

**The gate difference turned out not to be one.** Scale is a byte-scale
representation choice; the quantity that gates is the derived allowance, and
that is **93.57 against 93.60**. The two runs returned 461,144 / 460,953
(August) and 461,143 / 460,952 (September) — one relation apart in 461,143. The
re-run cost 17 minutes and converted an assumption into a check.

### A REAL 7.33 ms/q REGRESSION — ENVIRONMENTAL, not our code (finding 88)

**Corrected 2026-09-02, same day.** This section first claimed the accounting
boundary had moved between August and September. It has not. That claim came
from computing August's unaccounted as `wall - sieve - td` = 15.23 ms, which
silently dropped three line items the breakdown prints. Both runs wrote a full
breakdown to `.stdout`; reading them settles it:

| stage | Aug `4b581b33` | Sep `57480cd` | delta |
|---|---:|---:|---:|
| transform + plattice | 3.123 | 3.268 | +0.15 |
| fill | 28.131 | 26.106 | **-2.03** |
| apply | 40.987 | 33.731 | **-7.26** |
| **sieve, both sides** | 72.24 | 63.11 | -9.13 |
| intersect + gcd | 0.416 | 0.412 | -0.00 |
| host per-q | 1.046 | 1.143 | +0.10 |
| TD + classify, wall | 14.49 | 15.99 | +1.50 |
| cofactorisation, in-loop flushes | 13.27 | 14.66 | +1.39 |
| **unaccounted** | **0.50** | **7.83** | **+7.33** |
| **wall** | **101.96** | **103.13** | **+1.17** |

The deltas sum to +1.18 against a +1.17 wall change, so the table closes and
nothing moved between buckets.

**What actually happened is that two real wins were eaten.** `apply` is 17.7%
faster and `fill` 7.2% faster -- 9.3 ms/q of genuine improvement, the fill
retune and the apply work -- and every millisecond of it was consumed by
**+7.33 ms of unaccounted wall**, plus 1.50 on TD and 1.39 on cofactor flushes.

`unaccounted` is `acc_wall - acc_sieve - acc_isect - acc_host - acc_td -
tm.join - tm.cofac` (`pipeline.cuh:2452-2454`), i.e. wall that no timer in the
breakdown sees. It went from **0.50 ms to 7.83 ms/q, a 15x increase worth 7.1%
of wall**, and that is the whole of the `acc/wall` drop from 0.967 to 0.878.

**RESOLVED THE SAME DAY -- see finding 88.** The bisect exonerated every
commit: rebuilding `4b581b33`, the exact commit the August probe was built
from, reproduces `unaccounted` 0.50 -> **7.86** with no source change at all.
The variable is **environmental**, narrowed to the 2026-08-27 `apt upgrade`;
the CUDA toolkit line and the Windows driver are both ruled out by direct
test. It costs ~10.6 ms/q, and it has been masking 9.44 ms/q of real
apply+fill work.

The thermal hypothesis below is **withdrawn**; finding 88 closes the
August-to-September arithmetic to 0.01 ms without it.

### One unexplained discrepancy, which does not touch the row

Whole wall went **101.96 -> 103.13 (+1.1%)** at 130M while at 190M it went
**100.95 -> 97.46 (-3.5%)** — opposite directions for the same binary pair.

**Explained by finding 88: an environmental penalty of ~10.6 ms/q from the
2026-08-27 upgrade, partly offset by our own 9.44 ms/q of apply+fill work.** At 190M
the kernel win is larger and wall improves; at 130M it is smaller and wall
worsens. The rest of this section records what was ruled out first, and the
thermal suspicion it ends on is superseded.

It is not contention: today's host load ran **1.27-1.71** against August's
**1.93-2.35**, so today was the quieter box.

**It is not a clock or voltage difference either — checked 2026-09-02.** The
question was raised directly (was the card undervolted in August?) and the
answer is yes: finding 61 applied the 950 mV curve on **2026-08-17**, three
days before the August probes, and finding 71's header records those probes as
undervolted. The board draw confirms it independently — finding 61 measured
stock at **195.2 W** and the 950 mV curve at **140.5 W**, and every August
probe sits at a mean of **138.3-141.0 W** (q50M / q130M / q190M), nowhere near
stock. September's runs mean **147.0-150.8 W** on the same curve.

So both sides of this comparison are undervolted and the remaining suspect is
thermal — 147-150 W today against 140-143 W in August, the September ambient
finding 83 already flagged — but that is a hypothesis, not a measurement.

*(Unrelated but adjacent: anything measured BEFORE 2026-08-17 does need
finding 61's +6.7% ms/pair correction before comparison. That boundary is
three days earlier than this one and does not touch these runs.)* It does not affect the verdict
row, whose two sides were both measured today.

### What item 0 has left

Nothing on this axis. Three probes, three matched controls, both rectangles,
metered watts on both sides. The remaining caveat is finding 83's: quote the
board draw alongside any rel/J, because the same curve drew 133.5 W in August
and 147-152.8 W in September.

## Finding 88 — the 7.33 ms/q "regression" is ENVIRONMENTAL, not our code: it survives a rebuild of the August commit. Narrowed to the 2026-08-27 system upgrade; the CUDA 13.2/13.3 toolkit line is RULED OUT

**Date:** 2026-09-02, RTX 5070, idle box. c183 `oracle/input.job`, `--logI 15
--qrange 130000000: `, the item-0 130M geometry.

Finding 87 recorded a +7.33 ms/q blowup in `unaccounted` between the August and
September 130M runs and set out to bisect it. **The bisect exonerates every
commit** -- that part is solid and is what this finding is for. The *cause* is
still open; an earlier version of this finding named the CUDA 13.2 -> 13.3
toolkit and that has since been refuted by direct test (see "What is ruled
out").

### The decisive test: one commit, rebuilt today

`4b581b33` is the exact commit the August probe was built from. Rebuilt and
re-run today, unchanged:

| `4b581b33` | runtime | nq | wall | sieve | td | **unaccounted** |
|---|---|---:|---:|---:|---:|---:|
| 2026-08-20 | **13.2** | 10000 | 102.01 | 72.24 | 14.49 | **0.50** |
| 2026-09-02 | **13.3** | 200 | 113.45 | 73.14 | 16.59 | **7.86** |

Same source, same box, same card, same undervolt curve. `unaccounted` is
band-length insensitive (HEAD gives 7.96 at nq=200 against 7.83 at nq=10000),
so **+7.36 ms/q of pure GPU idle** is exact. Wall carries a ~0.82 ms nq
artifact, measured the same way, giving a penalty of **~10.6 ms/q, +10.4%**.

**This is the load-bearing result: nothing we wrote caused it.** The same
source that measured 0.50 in August measures 7.86 today.

### What is RULED OUT

**The CUDA toolkit line, 13.2 vs 13.3.** Tested directly by building HEAD
against both (`make NVCC=/usr/local/cuda-13.2/bin/nvcc`; `CUDART_LINK ?= static`
at `Makefile:217`, so the runtime is baked in at build time):

| HEAD, c183/130M, `--nq 200` | wall | sieve | **unaccounted** |
|---|---:|---:|---:|
| CUDA **13.3** | 104.01 | 63.74 | **7.96** |
| CUDA **13.2** | 102.58 | 63.21 | **7.87** |

The toolkit line is worth ~1.4 ms of wall and **nothing** of the idle. An
earlier version of this finding attributed the whole regression to
13.2 -> 13.3, reasoning from it being the one difference that could be named
rather than from an isolated variable. **That attribution is withdrawn.**

**The Windows driver / WSL passthrough.** Every real binary in
`/usr/lib/wsl/lib` -- `libcuda.so` (187,984 B), `libnvidia-ml.so.1`,
`nvidia-smi` -- is dated **2026-07-22**, a month before either session. The
only 2026-08-27 entries there are two symlinks of 15 and 20 bytes plus the
directory mtime, created by that day's `ldconfig`. Independently: the August
logs already reported `driver 13.3`, so the driver already spoke that API
before the Windows-side toolkit install; and a newer driver has been available
since 2026-08-26 that is still not installed (the box runs 610.88 from July).

### What it is NARROWED to

`/var/log/dpkg.log` shows a broad `apt upgrade` on **2026-08-27 17:51**, between
the August probes and the September runs. It installed CUDA 13.3, bumped the
13.2 line (cudart 13.2.75 -> 13.2.86, compiler 13.2.1 -> 13.2.2), **and
upgraded 59 non-CUDA packages** including apparmor, PAM, OpenSSL, Python and
iproute2.

So "CUDA 13.3" was merely the most visible item in a system-wide upgrade. Note
that the toolkit test above compares two *post-upgrade* toolkits against each
other, which is why they agree -- the 13.2 on the box today is **not** the
13.2.75 that produced the August numbers.

**CUDA IS NOW FULLY ELIMINATED.** The August-era `libcudart_static.a`
(13.2.75-1, byte-different from 13.2.86) was fetched from NVIDIA's repo,
extracted without installing, and linked through a symlink-farm shadow
toolkit:

| static `libcudart` linked into HEAD | `unaccounted` |
|---|---:|
| **13.2.75** -- the one August actually used | **7.83** |
| 13.2.86 | 7.87 |
| 13.3.29 | 7.96 |
| *August's own measurement* | *0.50* |

Linking the exact runtime August ran changes nothing. Not the toolkit, not the
static runtime, not the driver. The trigger is elsewhere in the 08-27 upgrade
(59 non-CUDA packages: apparmor, PAM, OpenSSL, Python, snapd, `wsl-setup`,
procps) or outside apt entirely. No microcode, kernel image or libc6 was
touched, and the CPU mitigation set is unremarkable for Zen 5.

### An assumption of this finding is WRONG: `unaccounted` is not simply GPU idle

An Nsight Systems trace (`--trace=cuda`, 6 q, steady state after dropping the
first third) says:

| | |
|---|---:|
| GPU kernel-busy fraction of span | **97.6%** |
| total inter-kernel gap | **3.44 ms/q** |
| `unaccounted` on the same geometry | 7.83 ms/q |

**The GPU is 97.6% busy, and the gaps do not add up to `unaccounted`.** Some of
that 7.83 ms is therefore device work falling OUTSIDE the event brackets, not
the GPU waiting. Every statement in this finding calling `unaccounted` "GPU
idle" is an assumption that this trace does not support -- treat the quantity
as *wall not attributed to a bracketed stage* and nothing more. (Profiling
also inflated the run to ~141 ms/q against 103 unprofiled, so the 3.44 is not
directly comparable either; what the trace ranks reliably is which transitions
dominate, not their absolute size.)

**The gap ranking, steady state:**

| transition | ms/q |
|---|---:|
| `k_transform` -> `k_fill_atomic` | **0.800** |
| `k_cof_enqueue` -> `k_transform` | 0.524 |
| `k_transform` -> `k_transform` | 0.487 |
| `k_scatter_sel` -> `k_gather_ab` | 0.271 |
| `k_apply` -> `k_intersect_compact` | 0.204 |

**The top entry is the `cudaEventSynchronize(S->ev[1])` bubble** described
below, now measured rather than inferred from a diff. It is a real and bounded
win, independent of whatever caused the 08-27 regression.

### The bisect, and why it pointed nowhere

Every commit built TODAY lands in the same band regardless of its content:

| commit | what it is | unaccounted |
|---|---|---:|
| `4b581b33` | the August baseline itself | **7.86** |
| `0c412a2` | Windows code-review fixes | 7.73 |
| `899f19b` | one `setvbuf` argument | 7.77 |
| `556a631` | the >31-bit / slab architecture | 9.03 |
| `9398fff` | slab cleanups | 8.71 |
| `57480cd` | HEAD | 7.96 |

`4b581b33` and `899f19b` differ by **118 lines**: a `.gitignore` block, a
`setvbuf` size argument, and a deleted `wintest.bat`. Nothing that can touch
sieve performance — and they measured 0.50 and 7.77 before the rebuild, 7.86
and 7.77 after it.

**Two methodological errors are worth recording, because both cost time.**

1. **The bisect was mis-framed: `4b581b33` is NOT an ancestor of HEAD.** It is
   the tip of a side branch whose Windows commits were re-applied onto
   mainline as different SHAs (`040953d` -> `899f19b`, `9175cb1` -> `0c412a2`).
   `git log 4b581b33..HEAD` therefore lists commits that are not "between" the
   two states in any useful sense. Check ancestry with
   `git merge-base --is-ancestor` BEFORE bisecting.
2. **A mechanism was proposed from a diff before the data supported it.**
   `556a631` does add a `cudaEventSynchronize(S->ev[1])` between the transform
   and the fill, splitting a single async chain into two synced halves, and
   that IS a real GPU bubble invisible to every timer. But `899f19b` predates
   it, still has the old single-sync structure, and is equally "bad" -- so the
   sync cannot be this regression. **It remains a genuine inefficiency worth
   fixing on its own** (the sync exists only to read `t_transform`, and
   deferring that read to after `ev[3]`'s sync would remove it), just not this
   one.

### What it means for everything measured across the boundary

At **matched runtime and matched nq**, our own work between the two commits is
worth **9.44 ms/q** (113.45 -> 104.01, -8.3%) -- the apply and fill
improvements, real and previously invisible. The whole August-to-September
picture closes to 0.01 ms:

```
102.01  August, pre-upgrade
+10.62  the 2026-08-27 environmental change (cause still open)
 -9.44  our apply + fill work
------
103.19  September, post-upgrade   (measured: 103.19)
```

**This also explains finding 87's unexplained sign flip.** Two forces of
similar magnitude pulling opposite ways: at 190M the kernel win (72.56 ->
61.97, 10.6 ms) exceeds the environmental penalty and wall improves 3.5 ms; at 130M
the win is 9.13 against ~10.6 and wall worsens 1.17 ms. Same mechanism, sign
set by which term is larger. Nothing thermal is needed, and the thermal
hypothesis recorded in finding 87 is withdrawn.

**Confounded, and needing a caveat rather than a re-run:** any timing compared
across 2026-08-20 -> 2026-09-01, which includes finding 83's attribution of the
190M speedup to our work. The *verdict rows themselves are unaffected* --
finding 83's and finding 87's GPU and CPU arms were each measured within one
session on one runtime.

### Not diagnosed

**Where the 7.4 ms goes is unknown**, and per the section above it is NOT
simply GPU idle -- treat it as wall not attributed to a bracketed stage. The
instrument that can rank the inter-kernel component is **Nsight Systems**
(gaps *between* kernels), not `ncu` (what happens *inside* one); finding 89
records that Nsight ranks reliably but cannot quantify. At ~10% of wall this is worth more than
every item on the open list except item 1. Item 19 carries the plan.

**A note on how this was nearly mis-closed.** Two attributions were made and
withdrawn in one session: first thermal (finding 87), then CUDA 13.2 -> 13.3
(this finding). Both were reached the same way -- by naming the most visible
difference between two sessions rather than by isolating a variable. The one
claim that has survived every test is the one backed by a controlled
experiment: **rebuild the August commit and the regression reappears.** Prefer
that shape of evidence here; the environment on 2026-08-27 changed in at least
61 ways and any of them can be named plausibly.

## Finding 89 — two serialisation fixes in the per-q path: -1.40 ms/q (-1.36%), relations unchanged. And a reminder that un-interleaved A/B doubled the apparent win

**Date:** 2026-09-02, RTX 5070, idle box (owner away, nothing else running).
c183 `oracle/input.job`, `--logI 15 --qrange 130000000: --nq 200`. Both arms
built from the same source tree with the same toolkit; only the two changes
below differ.

Finding 88 left the code exonerated for the 08-27 environmental regression but
identified one thing we own: `k_transform -> k_fill_atomic` was the largest
inter-kernel gap in the Nsight trace. Fixing it exposed a second serialisation
point behind it.

### Fix 1 — the transform event sync existed only to read a timer

`pipe_side_prepare_q` ended with `cudaEventRecord(ev[1])` +
`cudaEventSynchronize(ev[1])` + `*t_transform = time_kernel(ev[0], ev[1])`.
Nothing after that sync touches device data: it is `cudaGetLastError`, an
elapsed-time read, and five host `free()`s. So the block bought nothing except
the timer, and cost a drained GPU while the host prepared the other side and
the TD tables.

The transform end moved to a **dedicated `ev[4]`** -- `pipe_side_sieve_slab`
re-records `ev[1]` once per slab and would clobber a shared one -- and the
elapsed time is now read after the slab loop, where the `ev[3]` sync has
already guaranteed `ev[4]` completed. Same events, same subtraction, same
number: `transform + plattice` reads 3.304 against 3.305 across the change.

### Fix 2 — which exposed that the H2D uploads were the real block

Fix 1 alone moved 2.83 ms/q out of `unaccounted` and put **2.04 ms of it into
`host per-q`** (1.179 -> 3.220). The time had not gone away; it had become
visible. `*t_host` is computed at `pipeline.cuh:525`, *after* four
**synchronous** `cudaMemcpy` H2D calls. A synchronous copy on the legacy
default stream cannot begin until prior stream work drains, so once the event
sync no longer drained it, side 0's uploads blocked on side 1's still-running
transform -- and that wait was charged to host time.

Those four, plus the equivalent in `pipe_td_small`, became
`cudaMemcpyAsync`. **Safe on two independently checked grounds:** the source
buffers are already pinned (`cudaHostAlloc`, `pipe_side_init`), and they are
only rewritten by the next q's call, by which point the stream has provably
drained -- each side's copy is queued before its transform, and
`pipe_side_sieve_slab` synchronises on `ev[3]` after apply. A comment at each
site records that the second condition is load-bearing.

### Result: interleaved A/B, four paired reps

| | PRE | POST | delta |
|---|---:|---:|---:|
| wall/q (COMPLETE) | 102.57 | 101.17 | **-1.40 ms (-1.36%)** |
| host per-q | 1.150 | 0.833 | -0.317 |
| `unaccounted` | 7.80 | 6.67 | -1.14 |

Per-pair wall deltas -1.45 / -1.51 / -1.38 / -1.25; **every pair favours the
fix and the arms do not overlap.**

**Correctness gate:** 9,053 relations from a 200-q band, and `comm -23` against
finding 87's pre-fix 10,000-q reference returns **0** -- every relation
produced is present in the reference. Run twice, once after each fix.

### Method note: the un-interleaved measurement was wrong by 2x

Run as three PRE reps followed by three POST reps, the same comparison gave
**-2.8 ms/q**. The PRE arm was drifting downward across its own reps
(103.48 -> 103.12 -> 102.52) as the box settled after a compile, and running
one arm to completion before the other credited that drift to the fix.
Interleaving removes it and gives -1.40.

This is finding 85's lesson arriving from the other direction: there,
interleaving let *one* rep rank four candidates correctly; here, failing to
interleave doubled a two-arm result. **Interleave, even when the box is idle
and the arms look stable.**

### A defect this introduced, found in review the same day and fixed

The async uploads' safety argument -- "only rewritten by the NEXT q's call, by
which point the stream has drained" -- **was false on the `PIPE_Q_SKIP`
path**. The four `cudaMemcpyAsync` are issued at `pipeline.cuh:424-430`; the
norm-width check raises `PIPE_Q_SKIP` at `:497`, *after* them; and
`run_pipeline_impl` `continue`s on a skip without entering the slab loop, so
`cudaEventSynchronize(ev[3])` never runs. Up to `PIPE_SKIP_MAX` (100)
consecutive skips could queue with no synchronisation at all, and the next q
rewrites the pinned staging buffers underneath them. Side 1's `k_transform` is
left in flight on the same path, so an execution fault there would have been
reported against the *next* q's first CUDA call.

Today's consequence was benign -- the next stream-ordered copy is the same size
and overwrites the torn data, so device state ends correct -- but that is luck,
not design, and the comment instructed future maintainers to rely on a
precondition that did not hold.

**Fixed** with a `cudaStreamSynchronize(0)` at the skip site, which drains both
the uploads and the in-flight transform; skips are rare and capped, so the cost
is irrelevant. The comment at the upload site now names *both* draining paths
and says a new early return between them needs the same treatment.

Two smaller hardenings from the same review: the deferred `time_kernel` reads
now `cudaEventSynchronize` on `ev[4]` first -- no-ops today, since the slab
loop's `ev[3]` sync already covers them, but `time_kernel`
(`bench_kernels.cu:869`) discards `cudaEventElapsedTime`'s status, so a
not-ready event would silently report 0 ms *and* latch `cudaErrorNotReady` for
the next unrelated `cudaGetLastError` to report as fatal. And
`pipe_side_prepare_q`'s `float *t_transform` out-parameter, which after this
change only ever wrote `0.0f`, is gone.

**NOT verified at runtime.** No gate in the repo exercises `PIPE_Q_SKIP`:
c183's norms are ~197 bits and AS276's fit 384, so nothing available skips
under a 12-limb build. To exercise it, `make BN_LIMBS=6` (192 bits, permitted
by the Makefile's `4 6 8 10 12 14 16` filter) and run any c183 band -- most q
will skip. Worth adding as a standing gate; the path is now load-bearing for
the async uploads' safety.

### Confirmed on the timeline

A post-fix Nsight trace against finding 88's pre-fix one, same geometry:

| transition | pre-fix | post-fix |
|---|---:|---:|
| `k_transform -> k_fill_atomic` | 0.800 | **0.120** |
| `k_transform -> k_transform` | 0.487 | *out of the top 8* |
| GPU kernel-busy | 97.6% | **98.2%** |
| total inter-kernel gap | 3.44 ms/q | **2.42 ms/q** |

The new largest gap is `k_cof_enqueue -> k_transform` at 0.583 ms/q -- the q
boundary, where the host runs per-q preparation.

### What is left, and why the trace does NOT answer it

`unaccounted` is still **6.67 ms/q**, ~6.6% of wall, the residue of finding
88's environmental regression. These two fixes are ours and were worth taking
regardless; they do not touch it.

**Nsight cannot quantify it, only rank it.** Including memcpy and memset
activity (separate CUPTI tables from kernels -- an earlier gap count that
omitted them was measuring the wrong thing) the steady-state GPU is **98.5%
busy with 2.06 ms/q idle**. But the profiled run costs **134.8 ms/q against
~101 unprofiled**: CUPTI inflates device time about 46%, roughly 1.2 ms per
kernel across ~35 kernels per q. **That distortion is larger than the quantity
being measured**, so 2.06 ms/q does not transfer to an unprofiled run and must
not be quoted as the true idle.

What would answer it is a whole-q event bracket -- record one event before the
first GPU operation of a q and one after the last, and compare that span
against the sum of the stage device times. The difference is true idle,
measured without a profiler in the loop. That is measurement scaffolding in
the production path and is left for the owner to approve rather than added
alongside two functional changes.
