# Running a job

Every command here was run end to end on a c123 (factored) and a c151 (sieved
and verified), 2026-08-05. Nothing in it is from memory.

## What you need

1. **A polynomial file.** A GGNFS `.job` works directly — `n:`, `skew:`,
   `c0:` through `c8:`, `Y0:`, `Y1:`. The extra keys (`rlim:`, `lpbr:`, `alambda:` …)
   are ignored by `fbgen`, so `result.job` needs no
   conversion. A CADO `.poly` works too.

2. **No algebraic factor-base file is required for pipeline runs.** When
   `--fb1` is omitted, `bench` generates the complete algebraic factor base on
   the assigned GPU before allocating the persistent sieve buffers. Ordinary
   prime roots use the fixed-capacity CUDA generator; the small set of
   prime-power, ramified, and projective branches is completed by the same
   exact CPU Hensel code used by native `fbgen`.

   A pre-generated native factor base remains supported and is useful for
   validation or repeated bands that should not repay GPU generation at every
   `bench` startup.  The preferred fast cached-file path is:

   ```sh
   make -C /path/to/cuda-sieve/bench GPU_ARCH=native fbgen_gpu
   /path/to/cuda-sieve/bench/fbgen_gpu --poly JOB.job --lim ALIM \
       --maxbits LOGI --out JOB.roots1
   ```

   That file is generated with the same GPU ordinary-root + exact CPU Hensel
   hybrid used in-process and is serialized by the same writer as native
   `fbgen`.  `--out` stages to `JOB.roots1.part` and renames only after success.
   `make -C bench fbgpucheck` requires byte identity with native `fbgen` and
   then compares the loaded `fb_t` contents as a second gate.

   The CPU-only generator remains available as an independent oracle/fallback:

   ```sh
   make -C /path/to/cuda-sieve/bench fbgen
   /path/to/cuda-sieve/bench/fbgen --poly JOB.job --lim ALIM \
       --maxbits LOGI --threads 6 --out JOB.roots1
   ```

   `--lim` defaults to `alim` when the `.job` carries it. `--maxbits` should be
   the sieve's `logI`. Both generators emit the same native text format.
   Supplying `--fb1` bypasses in-process GPU generation and loads that file
   exactly as before; the legacy GGNFS `.afb.0` `--fb` format is still
   sieve/debug only because it lacks the required prime-power metadata.

## Where to put things

`fbgen` and `bench` take paths and do not care about the working directory.
**msieve does** — it looks for `cub/`, the lanczos kernels and its default
`msieve.dat` / `msieve.fb` / `worktodo.ini` relative to wherever you launch it.
So: one directory per job, and run msieve from inside it.

```
~/nfs/c151/
  c151.job                 copy of result.job — the poly
  c151.roots1              optional native fbgen output (only with --fb1)
  msieve.dat               relations: sieve STRAIGHT into this, no copy step
  msieve.fb                N, poly and bounds (see below)
  worktodo.ini             N on one line
  cub -> ~/msieve_nfsathome_basement/cub
  lanczos_kernel.ptx    -> ~/msieve_nfsathome_basement/lanczos_kernel.ptx
  lanczos_kernel.fatbin -> ~/msieve_nfsathome_basement/lanczos_kernel.fatbin
  sieve.log
```

Run `fbgen`, `fbgen_gpu`, and `bench` from anywhere with `--out` /
`--relations` pointing into the job directory; run `msieve` from inside it.

Two practical notes:

- **`--relations msieve.dat` writes directly**, saving a full duplicate of the
  relation file — ~8 GB on a c151 at 65M relations (~128 bytes each). Budget
  ~15 GB per job including msieve's `.mat` / `.cyc` / `.chk` / `.dep`.
- **`bench` writes `NAME.part` and renames on success.** The `.part` is the
  durable artifact, not scratch: after every cofactor flush it is fsynced and
  `NAME.part.ckpt` records the resume point. **Rerunning the same command
  resumes there** — it does not overwrite, and it does not need the several
  files and a `cat` this note used to prescribe. See "Stopping and resuming"
  below.

## Sieving

```sh
bench --pipeline --cofactor \
      --poly JOB.job \
      --logI 14 \
      --qrange 15000000: --target-rels 65000000 \
      --relations ~/nfs/c151/msieve.dat
```

That is the whole command. With no `--fb1`, startup first generates the complete
algebraic factor base in-process. If the same polynomial will be reused across
many bands, generate the native roots file once with `fbgen_gpu --out` and pass
it back with `--fb1` to avoid that startup cost:

```sh
bench/fbgen_gpu --poly JOB.job --lim ALIM --maxbits LOGI --out JOB.roots1
bench/bench --pipeline --cofactor --poly JOB.job --fb1 JOB.roots1 ...
```

The standalone GPU file is required by `fbgpucheck` to be byte-identical to
CPU `fbgen` output. The in-process path does not serialize or write this file.
On Windows, `build_windows.bat fbgen_gpu` additionally builds
`fbgen_gpu.exe`; the default Windows build still compiles the library form into
`bench.exe`.

**`rlim`, `alim`, `lpbr`, `lpba`, `mfbr`, `mfba` and both lambdas are read from
the `.job` file**, and the byte scale and survivor allowance are derived from
the polynomial. Each one is printed with its source, so a run says where every
number came from:

```
  job file: rlim 16700000, alim 33500000, lpbr 29, lpba 30, mfbr 56, mfba 59
  job file: alambda 2.5 -> 62.50 bits (side 1), reported only; the allowance is derived below
  job file: rlambda 2.4 -> 57.60 bits (side 0), reported only; the allowance is derived below
params from q=15000017: side 1 log2(maxnorm)=... scale=... allowance=<derived> bound=...
```

**The lambda lines and the allowance are unrelated numbers** — the example
above used to print `allowance=62.50`, the same value as `alambda`'s, which
read as the job file's lambda flowing through into the bound. It does not; see
"Lambda: we ignore it, on purpose" below. The `reported only` suffix is what
the binary actually prints, and it is there to stop exactly that misreading.

Anything you state explicitly wins over the file; anything the file carries
wins over the derivation. A CADO `.poly` carries none of these keys, so there
you either state them or let them derive.

- **`--logI`** is the siever width: `gnfs-lasieve4I14e` → 14. **`--J` defaults
  to `2^(logI-1)`** (CADO's convention), so it no longer needs passing.
- **`--region`** must be ≤ `--logI`; the default is 14, so only pass it for
  I=13 or smaller (`--region 13`).
- **`--qrange MIN:`** with the upper end omitted generates prime special-q and
  their polynomial roots upward, independently of `rlim`/`alim`. It requires
  **`--target-rels N`** or **`--nq N`** as its stopping condition. Relation
  targets are checked at flush boundaries, so they overshoot by well under 1%
  on any real target.
- Side 1 is the algebraic side and carries the special-q. Side 0 is rational.

### Current hard size limits and j-slabbing

The production binary keeps **each local sieve slab at or below `2^31`
positions** as a correctness limit, but the full pipeline rectangle may be
larger. Automatic planning also has a performance policy: once the full sieve
area reaches `2^30` positions, it targets slabs of at most `2^29` positions.
Areas below `2^30` are not split for performance alone; in particular, `2^29`
and smaller geometries retain the ordinary unslabbed path. The hot
bucket/bitmap/TD positions remain slab-local while exact global `j` is restored
where norms, primitivity, and relation coordinates need it. With default
`J=2^(logI-1)` and default `bkthresh=I`, the plans are I15 -> 1 slab, I16 -> 4,
I17 -> 16, I18 -> 64, I19 -> 256, and I20 -> 1024.

The `2^29` target is a performance heuristic, not an arithmetic requirement.
RTX 3090 and RTX 5070 runs both minimized complete time at `2^29` positions per
slab. A later L40 run showed that the speed optimum can move on a large-L2 GPU:
for the same `I=J=2^16` geometry it preferred `2^30` positions per slab.

| L40 local slab area | slabs | steady VRAM | fill | TD + classify | complete time/q |
|---:|---:|---:|---:|---:|---:|
| `2^31` | 2 | 7.76 GB | 235.029 ms | 68.51 ms | 564.13 ms |
| **`2^30`** | **4** | **4.72 GB** | **212.778 ms** | **70.39 ms** | **531.16 ms** |
| `2^29` | 8 | 3.20 GB | 223.320 ms | 84.40 ms | 555.70 ms |
| `2^28` | 16 | 2.43 GB | 267.659 ms | 121.06 ms | 628.78 ms |
| `2^27` | 32 | 2.05 GB | 373.371 ms | 206.37 ms | 819.63 ms |

Thus `2^29` is not claimed to be the universal maximum-throughput setting. On
the L40 it is 4.6% slower than the `2^30` optimum, but it remains 1.5% faster
than the former `2^31` default while using 59% less steady VRAM (3.20 vs
7.76 GB). That performance/memory tradeoff is why `2^29` remains the generic
default.

**What `2^29` actually tunes is a bucket-region COUNT, not a slab area**
(findings 79-81, 2026-08-26). `fill` is minimised at a fixed number of regions,
so the optimal area halves with `--region`: `2^29` is what 32,768 regions means
at the default `--region 14`, and the L40's preference above is a preference for
65,536 regions. The mechanism is read-modify-write on partially-filled bucket
lines — DRAM read sectors rise 135% past the knee while L2 reads stay flat — so
it is traffic, not cache capacity, which is why a 6 MB 3090 and a 48 MB 5070
agree. **No operator action:** the default already runs at 1.28x the write floor,
`--region` 14 is the joint optimum (`k_apply` costs +52% at 13), and an
automatic per-card target is still open work.

`--slab-j N` remains a regression/tuning override and may select a larger or
smaller slab subject to the mandatory safety bounds.
The parser rejects shapes that cannot contain whole TD rank groups, and the
runtime planner also enforces the `2^31` local-position bound and the arithmetic
bound implied by the **actual largest direct-tested small prime**. Raising
`--bkthresh` can therefore reduce the automatically chosen slab height. A
forced height that violates either bound is rejected rather than used
unsafely.

The remaining representation limits are:

- `lpbr` or `lpba` above 64;
- `mfbr` or `mfba` above 128 (above 96 in a `CF_LMAX=3` build — see
  [Cofactorisation](#cofactorisation-width-and-method-both-per-side) below);
- a ratio `ceil(mfb/lpb)` above 3 on either side;
- `lim^2 <= 2^lpb` on either side, which makes the "prime by size" test in the
  splitter unsound (it binds around `lpb 55` at an `alim` of 240M); or
- an exact norm wider than this build's `BN_LIMBS * 32` bits — 384 by default
  since 2026-08-27, previously 256.

These are checked limits, not tuning advice, and there is no unsafe override.
The non-pipeline harness still requires its whole `I*J` area to fit in `2^31`;
**`--pipeline` has no total-area cap** and slabs anything through `logI 20`.

**The norm-width limit is the one that bites mid-band, so survey it first.**
A q-lattice whose exact norm does not fit is *skipped* with a warning rather
than wrapped, and the band stops after `PIPE_SKIP_MAX` of them — so a job that
needs more width than the binary has bleeds special-q until it dies, and the
fix is a rebuild (`make BN_LIMBS=N`, even limbs 4..16) that every client would
have to carry. The width therefore has to be decided centrally, before work
units go out. `normscan` answers it; see [Sizing a job
first](#sizing-a-job-first-testsievesh). A wider build is byte-identical to a
narrower one on any job the narrower one could run, so widening is safe to ship
mid-project. Cost measured on AS276: +0.45 ms on a 90 ms special-q, and
`sizeof(bn_t)` 32 -> 48 bytes per survivor and per candidate.

**Read the side in the warning — it names the cause.** Each skip prints
`exact side-N degree-D norm may require ... bits`. Side 1 is the algebraic form
and side 0 the degree-1 rational one `G = Y1*x + Y0`, and *which one overflows
is a property of the job, not a constant*. A side-1 skip is telling you about
the algebraic degree and the sieve area; a side-0 skip is telling you about the
size of `Y0`, and no change of geometry will help much, because side 0 gains
only ~1 bit per logI step against side 1's ~5. `normscan` now prints the same
split up front (`per side:` and `over N bits: side1 only / side0 only / both`),
so the cause is visible before any work unit goes out. Measured examples in
finding 93: the c194 quintic is algebraic-driven by 79–91 bits, while a degree-4
SNFS with `Y0 ~ 2^171` is rational-driven by 73 — both entirely ordinary jobs.

**A capped band resumes, but slowly.** `nqskip` is not checkpointed, so a run
that stops at `PIPE_SKIP_MAX` writes a checkpoint and the next invocation starts
counting again — advancing roughly `PIPE_SKIP_MAX` special-q per run and
producing no relations while it does. That is progress, not a deadlock, but it
is not a way to finish a band: it means the build is too narrow, and the answer
is still a rebuild. `make skipcheck` (needs a card, and a build at
`BN_LIMBS=4`) is the gate over all of this.

The p-lattice **increments** are 64-bit, and were so even when positions were
still 32-bit. This is required for correctness: realistic large factor-base
primes can produce a reduced `(j << logI) +/- i` increment above `2^32`.
The slabbed walk carries an exact 64-bit continuation between slabs. This
avoids the latent uint32 wrap that older monolithic walks could turn into
spurious sieve hits.

Since 2026-08-24 the walk **position** is 64-bit in the production fill and
resieve kernels too — `k_fill_atomic`, `k_resieve_scatter`, `k_fill_segmented`
and `k_resieve_rewalk`. That is a performance choice, not a correctness one:
the 32-bit form reaches its bound by saturating, which costs a branch per
increment, and removing it was worth −6.1% of fill on an *unslabbed* geometry
(`bench/RESULTS.md` finding 73; slabbed geometries were already 64-bit and gain
nothing). Sieve positions still fit in 31 bits, so nothing about record layout,
`--region`, or the `2^31` slab bound changes.

**`k_fill_l1` is the exception and still walks in 32 bits**, so `pl_next`,
`pl_first` and `pl_add32_sat` remain live device code. Its
`__launch_bounds__(512, 3)` caps it at 40 registers, it fits in exactly 40
today, and widening its walk made `ptxas` spill to local memory rather than
lower occupancy. Do not delete the 32-bit walk helpers or the SLAB32 block in
`verify_walk_slabs` that gates them.

Since 2026-08-25 `k_apply` carries the same `__launch_bounds__(512, 3)` for the
opposite reason: uncapped it compiled to 45 registers, which fits only two
512-thread blocks per SM and held occupancy at 66.67% while shared memory would
have allowed three. Capped it reaches 40 registers with no spill and ~100%
occupancy, worth **−12.6% apply and −4.6% wall clock** on the C194
(`bench/RESULTS.md` finding 75). **The 512 is a hard launch ceiling, not a
hint**, which is why `--apply-threads` now refuses anything above 512; it
accepted up to 1024 before this change. The default (512) is unaffected.

`--no-td-verify` disables the dense TD reconstruction gate. In the pipeline the
gate normally runs once on the first slab of the first q; in the standalone TD
harness it disables that harness reconstruction check as well. It saves a
transient host/device allocation and is intended for production after the
correctness gates are established. `--cofgate` requires TD verification, so
`--cofgate` and `--no-td-verify` are rejected together during argument
validation.

**`lpb` above 32 is supported as of 2026-08-17** — an NFS@Home C194 asks for
`lpba 33` with `mfba 95`, and that runs today. What it did *not* require is any
change to the cofactor arithmetic: `mfb <= 96` still means three limbs, and
`ceil(96/33) = 3` still fits the splitter.

LPB and MFB should not be conflated. LPB is the bound on each resulting prime;
raising it above 32 widened the split-prime, emission, and validation paths to
64 bits (done 2026-08-17). MFB is the maximum composite residual sent to
cofactorisation and sets the arithmetic width: up to 64 bits can use two 32-bit
limbs, up to 96 three, and up to 128 four. Thus `lpb 33` with `mfb 64/95` does
not require four-limb rho or ECM, and is exactly the configuration the C194
runs.

### Cofactorisation: width and method, both per side

**Four-limb cofactors landed 2026-08-18.** The width is now chosen **per side**,
at run time, as the narrowest instantiation that holds that side's `mfb`:

| `mfb` | limbs | bits | example |
|---|---|---|---|
| ≤ 96 | 3 | 96 | SNFS `mfbr 88`; C194 `mfba 95`; AS276 `mfbr 64` |
| 97–128 | 4 | 128 | AS276 `mfba 101` |

**AS276** (the C208, `~/code/ggnfs-distributed/AS276.job`: `lpbr 33 / lpba 35`,
`mfbr 64 / mfba 101`) therefore runs **4/3** — wide on the hard side, narrow on
the easy one — without any flag. Nothing needs setting for it to be picked, and the choice is
printed at the head of every cofactoring run:

    cofactor width: side 0 3 limbs (96 bits), side 1 4 limbs (128 bits); this build carries 3..4

Two knobs exist, and neither is tuning advice:

- **`--cof-limbs N` / `--cof-limbs0 N`** force a side wider than its `mfb`
  needs. This exists so the *same* job can be timed at 3 limbs and at 4 —
  otherwise the only jobs that exercise the 4-limb splitter are ones no 3-limb
  run can be compared against. Forcing a side *narrower* than its `mfb` needs
  is refused, because that is exactly the silent truncation the width machinery
  removes.
- **`make CF_LMAX=3`** builds a binary carrying only the 3-limb splitter,
  reproducing the pre-2026-08-18 executable. The default carries both and picks
  per job. This is the knob for the "separate executables" option; it takes
  `bench_kernels.o`'s device code from 8.99 MB to 4.02 MB and roughly halves the
  ptxas work on `k_cofac` and its ECM variants. A `CF_LMAX=3` binary is
  **shippable** — its relations are byte-identical to a `CF_LMAX=4` binary's, and
  it is a first-class Make variable rather than a `DEFS` value precisely so that
  it is not branded a pricing build and refused `--relations`.

#### Method: ECM for 3LP, rho for 2LP — AUTOMATIC since 2026-08-19

**This is the default; nothing needs passing.** The method is chosen **per
side** from `ceil(mfb/lpb)`: 2 parts (2LP) runs rho, 3 parts (3LP) runs ECM.
Per side because a real job is usually one of each — AS276 is 2LP rational and
3LP algebraic, so a per-job choice is wrong on one side of nearly every job.
Every run prints what it picked:

    cofactor method: side 0 rho, side 1 ECM

ECM parameters are derived from the lpb of whichever side runs ECM: **B1 = 200**
at `lpb <= 33`, **300** at 34–35, **500** at 36+; `B2 = 30 * B1`; 12 curves;
4 rounds. Override any of them with `--ecm-b1 / --ecm-b2 / --ecm-curves /
--cof-rounds`.

Measured effect of the default change, zero flags, against the previous
default (rho on both sides, `--cof-rounds 2 --cof-budget 65536`):

| job | stage ms/q | wall ms/q | relations |
|---|---|---|---|
| c183 | 17.21 → **14.30** | 113.35 → **109.86** | 9,363 → **9,394** |
| C194 | 15.48 → **13.95** | 116.30 → **109.36** | unchanged |
| AS276 | — | — | 3,443 → **4,089** |

Cheaper *and* higher-yielding on all three. AS276's +18.8% is the old rho
budget having been far too small for a 3LP side at `lpb 35` — see finding 69.

To force one method on both sides, for A/B work or to reproduce the old
behaviour:

```sh
--cof-rho     # rho on both sides (the pre-2026-08-19 default)
--cof-ecm     # ECM on both sides
```

The `B1 = 1000-2000` values in `cofcheck.sh` are correctness fixtures, not
tuning advice — `B1=1000` costs 2.4x what `B1=250` does for the same relations.

#### Sweep the budget from BELOW, both methods

The single most expensive mistake available here, and it has now been made
twice in this repo's history in both directions. rho's cost is nearly linear in
its budget once past saturation, so pricing it at the first budget that happens
to saturate overstates it wildly — on AS276 that was 332 ms/q reported as
2,197. Always walk the budget **up** from a value that clearly under-yields, and
take the first one where the relation count stops moving.

Do **not** use `stuck == 0` as the criterion. A stuck record's status is
unknown, and at 3LP most are composites whose factors all exceed `lpb` — never
relations, but only provable dead by factoring them. Yield saturates long before
the stuck count does.

#### If you force rho (`--cof-rho`), its budget is NOT inherited across width classes

Historical but load-bearing: this is how the 4-limb work's one real defect was
found, and it is still what happens to anyone who forces rho on a 3LP side.
Measured on
AS276 against the GGNFS corpus (RESULTS finding 69), same region, same 30
special-q:

| setting | relations | of GGNFS's, missed |
|---|---:|---:|
| `--cof-rounds 2 --cof-budget 65536` (the default until 2026-08-19) | 3,205 | 389 of 2,846 (**13.7%**) |
| `--cof-rounds 6 --cof-budget 262144` | 3,823 | 2 of 2,846 (0.07%) |

**+19.3% relations from the budget alone.** The default was calibrated on the
c183's `lpba 32`. Pollard-Brent rho's expected iteration count grows as the
square root of the factor being sought, so a 35-bit large prime costs
`sqrt(2^5) = 5.7x` what a 30-bit one does; an unchanged budget simply returns
`CF_INCOMPLETE` and the relation is lost. It does not fail, warn, or look
wrong — it looks like a 14% yield hole in the siever.

Rule of thumb: scale the budget by `2^((lpb - 30)/2)` off the c183 default, and
confirm on a short band against a known corpus if one exists. The automatic
method avoids this entirely on a 3LP side by not using rho there.

#### What the fourth limb costs — MEASURED, decision made

Both jobs, 200 q and 100 q, all width combinations byte-identical:

| job | 3/3 | 3/4 (the wide-algebraic shape) | wall cost |
|---|---:|---:|---:|
| c183 | 16.88 ms/q stage, 112.17 wall | 26.38, **121.99** | **+8.8%** |
| C194 | 15.48, 116.30 | 24.18, **125.82** | **+8.2%** |

A side's queue costs **×1.72** at four limbs. **The automatic per-side width is
what ships**, because 8–9% of wall is too much to give away for the simplicity
of always running wide, and the selector costs nothing at run time. Forcing a
width with `--cof-limbs` is for A/B work only.

To reproduce, add `--cof-limbs 4 --cof-limbs0 4` to any run and `cmp` the
relation file against the unforced one — it must be byte-identical.

#### Running AS276 (the C208)

```sh
cd work/as276
../../bench/fbgen --poly AS276.job --maxbits 16 --threads 12 --out as276.roots1.m16
../../bench/bench --pipeline --poly AS276.job --fb1 as276.roots1.m16 \
    --logI 16 --J 32768 --qrange 80000023: --nq 200 --cofactor \
    --relations as276.rels.txt --log as276.log
```

No width or method flags: the job resolves to **3 limbs rational / 4 algebraic**
and **rho rational / ECM algebraic** on its own, which is the measured optimum.
`--maxbits` should track `--logI`.

That is `A = 31`. NFS@Home sieve this job at `I16e -J 16`, which in our
coordinates is `2^17 x 2^15` — `A = 32` (finding 65's rule: our rectangle =
`2^(J_bits+1) x 2^(I_bits-1)`). The production pipeline runs that geometry
through j-slabbing rather than refusing it:

```sh
../../bench/fbgen --poly AS276.job --maxbits 17 --threads 12 --out as276.roots1.m17
../../bench/bench --pipeline --cofactor --poly AS276.job --fb1 as276.roots1.m17 \
    --logI 17 --J 32768 --maxbits 17 --qrange 80000023: --nq 10 \
    --relations as276.a32.rels.txt
```

The planner picks **8 slabs of 4096 rows** on its own — no `--slab-j` — and
setup allocates 2.71 GB, so this fits a 12 GB card. Validated 2026-09-01
(finding 82): every relation rebuilds both norms exactly, and `relgeom.py
extent` recovers `2^17 x 2^15` from the relations themselves, matching the
extent GGNFS produced for the same job.

`--logI 17 --J 16384` is a smaller job, not an alternative nesting of this one:
same `i` range, half the area (`2^17 x 2^14` = `2^31`). It is what findings 69
and 77 ran, and it is the right choice only if you want the smaller rectangle.

**Check the rectangle from the relations, never from the flags**, before any
cross-siever claim:

```sh
./relgeom.py --band 80000023:80000200 --skew 51059252.11 extent as276.a32.rels.txt
```

Note the ceiling that actually binds first is **not** the width. `CF_MAXFAC`
caps a split at 3 large primes, so `mfb <= 3 * lpb` regardless — and at
`lpb 32` that is 96 bits, exactly what 3 limbs already hold. **A side only
needs 4 limbs once its `lpb` is 33 or more.** Widening past 128 bits would mean
4LP, which is a different change (`CF_MAXFAC` and `mz_split`'s stack guard),
not another limb.

For the current status, A=32 memory/slabbing options, and why the end metric is
time to a filterable matrix rather than raw relations/s, see [Current size
limits and
j-slabbing](bench/STATUS.md#current-size-limits-and-j-slabbing).

### Will it fit? VRAM sizing

**`testsieve.sh` reports measured memory per geometry** (see [Sizing a job
first](#sizing-a-job-first-testsievesh)), which is the better number whenever
the geometry actually runs. **The formula below is still what you need for the
case it exists for** — deciding whether the *next size up* will fit. A geometry
too large for the card aborts at the bucket-array admission check, before any
memory line is printed, so `testsieve.sh` reports a bare `FAILED:` row and no
numbers precisely when you most want them.

Two knobs move device memory and they are not interchangeable
(`bench/RESULTS.md` finding 64):

    VRAM ~ bucket(larger side, nearly flat in lim) + 33.4 B x (entries_r + entries_a)
           + 3 x local_area/8 + ~1.45 GB fixed     where entries ~ pi(lim)

A slabbed run additionally carries two 64-bit walk-continuation values per
full-FB entry (16 B/entry). The 33.4 B coefficient is 8 B above the historical
measurements below because `plat_t` now stores overflow-safe 64-bit increments.

- **Area is the expensive axis.** On a C194 the bucket array is 1.46 GB at
  `2^29`, 2.91 GB at `2^30` and 5.20 GB at `2^31`. It is linear in `J` at fixed
  `I` (1.46 → 2.91 is exactly 2×) but **sub-linear when `logI` rises**, because
  `bkthresh` defaults to `1 << logI` and a higher start cuts `Σ1/p` — which is
  why 2.91 → 5.20 is 1.79×, not 2×.
- **Lim is the cheap one.** Doubling both lims doubles the factor base but
  moves the bucket array only ~5%, because records go as `Σ1/p ~ ln ln p`. It
  costs ~33 bytes per entry, summed over both sides, before slab-only walk state.
- The bucket array is sized by the **larger** side, not the sum — the two sides
  share one allocation and run sequentially — so raising the smaller side's lim
  is nearly free until it overtakes the other.
- With the widened walk representation, the same first-order model moves the
  unslabbed crossover from the historical ~6.7 x lim to about **8.8 x lim**;
  slab-only continuation state moves it higher still (about **13 x lim**).
  Treat these as sizing estimates until fresh GPU measurements replace the
  pre-wide-walk data below.

Measured device-in-use totals for a C194 (`rlim 160M, alim 240M`): **3.63 GB**
at `2^15 x 2^14`, 5.32 GB at the `2^15 x 2^15` square, **4.98 GB** at the
`2^16 x 2^14` wide rectangle of the same `2^30` area, and **8.06 GB** at 16e
(`2^16 x 2^15`) — so 16e fits a 12 GB card with 3.9 GB to spare. These are what
`bench` reports as *device* memory in use (total − free); two runs agree to
~0.04 GB, which is the practical accuracy.

**A busy Windows desktop does not need to be cleared first.** `nvidia-smi` will
show several GB in use by dwm.exe, browsers and editors, but under WDDM those
allocations are evictable — asking for the bucket array demotes them to system
RAM, and the figures above were taken with ~5 GB apparently in use and the card
reading 10.76 GB free a moment later. Do not close applications before sizing,
and do not subtract a desktop baseline. **Another CUDA process is different**:
its memory is pinned, and it really does come off your budget.

**The wide rectangle is cheaper as well as higher-yielding.** At equal `2^30`
area it costs 4.98 GB against the square's 5.28 GB measured back to back (2.60
GB of bucket array against 2.91 GB) *and* returns +16.3% relations. Buying area
with `I` rather than `J` is not a trade — both effects come from the same place, since raising
`logI` raises `bkthresh` with it. The factor base is not involved: `--maxbits
16` carries only 26 more ideals than `--maxbits 15` on this job, so the whole
geometry difference is the bucket array.

**Do not size a job from an aborted startup.** The startup table lists only the
bucket array, factor bases, bitmaps, trial-division context and cofactor queue
— roughly 2.2 GB of the 3.63 GB above at 15e — and per-q buffers grow on demand
afterwards. The "device memory, steady state" line is the complete figure and
prints only on a *successful* run, so budget from the formula and confirm with a
short completed band.

### Sizing a job first: `testsieve.sh`

Before committing days to a band, test-sieve it. `bench/testsieve.sh` samples
the yield curve at several q, normalizes each sample, integrates it, and
projects the whole run. Run `./testsieve.sh` with no arguments for interactive
prompts, or pass flags for a repeatable run:

```sh
cd bench
./testsieve.sh --poly c194.job --fb1 c194.roots1 \
    --qmin 20 --qmax 200 --points 5 \
    --target-rels 300 --geom 15,16384 --geom 15,32768
```

**`--width` is a q-INTERVAL WIDTH, not a count of special-q.** The default
`--width 2000` sieves a 2,000-*integer*-wide window, which at the default
`--qmin 20` holds `2000 / ln(2e7)` ~= **119 (q, rho) pairs** -- so the default
5-point run sieves about **600 pairs in total**, not 10,000. On the c183 at
`15e` (97.46 ms/pair, finding 83) that is ~58 s of sieving plus ~15-20 s of
per-geometry startup: **about a minute and a half**. Reading "5 points x 2000"
as 10,000 special-q over-estimates the run by 17x, and the same misreading in
reverse makes a real band look impossibly fast. Widen `--width` to buy
precision; it is the knob that costs time.

**`--qmin`, `--qmax` and `--target-rels` are in millions.** That run sieves
`[2e7, 2e8)` and reports where 3e8 relations are met. Fractions work
(`--qmin 2.5`), and a value that still looks like an absolute count is
*rejected* with the translation rather than silently multiplied — the old
`--qmin 20000000` would otherwise have meant 2e13 and projected a band nobody
asked for. The units stop there: `--width`, `--rlim` and `--alim` are absolute,
and so is `bench`'s own `--target-rels`, which is a different flag on a
different program and unchanged.

**Each geometry's block begins with a `normscan` verdict on the exact-norm
width.** It surveys the whole projected band for that poly and geometry and
reports a *projected* band maximum from a fit to the upper tail, not the sample
maximum — on the 2,1139+ job, 2,500 samples said 242 bits and "256 is fine",
while 160,018 samples found a lattice at 273.08. The answer moves with the
geometry (250 bits at 15e, 257 at 16e on that job), which is why it runs per
geometry. Exit codes are verdicts: **0 pass, 2 will overflow, 3 too little
margin, 1 the survey could not run**. A non-zero verdict is recorded and
repeated in the summary but does **not** abort the sweep — you still want the
yield numbers that say whether this is even the geometry to rebuild for. Act on
it before distributing work: see [Current hard size
limits](#current-hard-size-limits-and-j-slabbing).

Each geometry's block ends with its measured device memory, so the sizing
question and the yield question are answered by the same run. On the **c183** at
`15,16384`:

```
  memory: 3.28 GB in use of 11.94 GB, 8.66 free   <- size from this
          setup 2.06 GB = bucket array 1.38, factor bases + bitmaps 0.44, trial division context 0.09, cofactor queue 0.15
          (+1.22 GB of per-q buffers and CUDA context after these marks; do not size from the setup figure)
```

**Size from the headline, not the breakdown.** This is the same rule as [Do not
size a job from an aborted startup](#will-it-fit-vram-sizing) above: the
per-stage marks cover setup only, and per-q buffers plus CUDA's lazily reserved
per-kernel local memory arrive afterwards. Here that is 1.22 of 3.28 GB — 37%
of the job, and mostly real job memory rather than context. The breakdown is
there to show **which knob moves memory**, not how much to budget: at `15,8192`
the same job reports `bucket array 0.69` against `1.38`, while the cofactor
queue stays at 0.15 either way.

**Disagreement between samples means the card was busy.** Every sample of a
geometry allocates identically, so the script compares them and refuses to
quote a single figure when they differ by more than 0.05 GB. That check exists
because each stage figure is a difference of two free-memory probes: a
neighbour allocating or freeing between them lands in whichever stage straddled
it — observed at 1.57 GB for a cofactor queue that is a fixed 0.15 GB
regardless of job, geometry or cofactor width. The breakdown names the knob: at
`15,8192` the same job reports `bucket array 0.69` against `15,16384`'s `1.38`,
while the cofactor queue stays at 0.15 either way.

That makes the formula in [Will it fit?](#will-it-fit-vram-sizing) a
cross-check rather than the primary method — test-sieve the geometries you are
considering and read the number off.

When `--fb1` is omitted, `fbase` is the managed cache stem. Before sieving,
the script compares its embedded polynomial, `lim`, and `maxbits` metadata with
the selected job and geometry, rebuilding it when it is missing or stale.
Geometry sweeps use stable per-`logI` names such as `fbase.m15`, because
prime-power depth is part of the factor base. Supplying `--fb1`, even as
`--fb1 fbase`, makes that path caller-managed and the script never overwrites
it.

**Rebuilds go to the GPU when `fbgen_gpu` is built.** `--fb-backend` selects
`gpu`, `cpu`, or `auto` (the default). The two generators are intended to write
byte-identical files, so the backend is deliberately *not* part of the cache
identity: the header check is unchanged, and a cache built by either generator
validates and is reused by the other.

**Know what gates that.** `make fbgpucheck` sweeps polynomial degrees 1..8 and
the CAP=6/CAP=8 boundary, but only at `--maxbits` 1, 14 and 15. `testsieve.sh`
accepts `logI` 2..20 and names its caches `fbase.m$logI`, and `oracle/README.md`
documents `.m16` in active use — so above 15 the byte identity is expectation,
not gate. Spot checks at maxbits 16, 17 and 20 on c147 came back identical, and
the header check would not catch it if they ever stopped being. If you are
mixing generators across a geometry sweep above `logI` 15, build all the caches
with one backend, or extend `fbgpucheck.sh` past 15. Measured on a 5070, c147 at its job
`alim` of 3.35e7 and `maxbits` 15 — 2,061,655 entries, 29.4 MB:

| generator | wall | CPU |
| --- | --- | --- |
| `fbgen_gpu` | 0.52 s | 0.27 s |
| `fbgen --threads 16` | 2.75 s | 33.5 s |

and `cmp` on the two files reported no difference. A five-times wall-clock
saving is not what decides a campaign — this is a one-off of seconds against a run of
days, and the *point* of moving it is that it stops being a reason to hand-manage
`--fb1` caches at all — but 33.5 CPU-seconds becoming 0.27 does matter on a
shared box, and it scales: BOINC-sized limits are 600e6 and up.

`auto` will not start the build itself, because `make fbgen_gpu` is a
multi-architecture CUDA compile of minutes and this script's unit of work is a
few seconds a sample point. If the binary is absent it says how to get it
(`make GPU_ARCH=native fbgen_gpu`) and uses the CPU generator. `--fb-backend
gpu` is the explicit request: it builds the binary if needed, and if generation
fails it stops rather than quietly answering with the CPU path. Under `auto` a
GPU failure — a busy card, no card — falls back and says so, and stays on the
CPU for every later factor base in the run rather than retrying a card already
known to be unavailable. `--fb-device N` picks the CUDA device for generation
(default 0); note it is separate from the device `bench` sieves on, which comes
through `--extra "--device N"`. `--fb-threads N` caps the CPU generator's
threads and has no effect on the GPU one; it defaults to the online CPU count
capped at 256, and a value outside [1,256] is rejected.

Neither backend runs `make`. Build the generator once with `make
GPU_ARCH=native fbgen_gpu`: a bare `make fbgen_gpu` would rewrite `.arch.stamp`
to the fat gencode list at Makefile-parse time and silently invalidate every
CUDA object in a tree built with `GPU_ARCH=native`.

The displayed `n-yield` and every projected quantity use the same
normalization as `~/code/test-sieve`: expected pairs are `width / ln(q0)`, and
`n-yield = raw relations * expected pairs / observed pairs`. The GPU-day and
energy projections likewise replace the observed pair count with the expected
count. This keeps a short interval that happens to contain unusually many
special-q roots from masquerading as a higher-yielding sieve configuration.
Each sample after the first also has an `exp-rel` value: the trapezoidal
estimate between the preceding q point and that row, using the two normalized
yields. At the end, the source job is displayed with the selected side's
`lss: 0` and any command-line overrides called out separately; the source file
is not modified.

The final table reports normalized relations, GPU days, energy, relations per
kilojoule, and (when requested) the q where the relation target is reached.

**When comparing geometries, compare at equal area — and prefer the wider
one.** The two axes are not interchangeable. At equal area a wide rectangle
beats a square by **+11.7% relations at `2^28` and +16.3% at `2^30`, for −0.3%
device time** (`bench/RESULTS.md` finding 65): `2^16 × 2^14` returns 10,671
relations where `2^15 × 2^15` returns 9,176 on the same q window. Doubling I
costs 1.86 bits of `log2(maxnorm)`; doubling J costs 3.86, because `i`
multiplies the shorter vector of the reduced q-lattice and `j` the longer one.
So a `J = I` row is a genuinely worse configuration than a wider rectangle of
the same size — but it is not *broken*, and the sweep can be trusted as
measured. The wider shape is also **~0.3 GB cheaper** in device memory at that
area (see "Will it fit?" above), so nothing is being traded away for the yield.

An earlier version of this note said our yield at `2^15 × 2^15` was "14% low"
against GGNFS and blamed the survivor gate. **Both claims are withdrawn**:
`gnfs-lasieve4I15e -J 15` covers what we call `2^16 × 2^14`, so that comparison
was our square against GGNFS's wide rectangle. (GGNFS's own coordinates call
that a `2^15 × 2^15` square: it orders the reduced basis longer-vector-first
where we order it shorter-first, so its axes are our axes swapped. Same region,
different names — see `bench/RESULTS.md` finding 65.) At matched rectangles we run 0.979–0.981
of GGNFS at every aspect ratio and area tested. See `bench/STATUS.md` item 5.

**Before comparing yields against another siever, confirm the rectangles
match** with `bench/relgeom.py --band QLO:QHI --skew S extent FILE...`, which recovers
the region a run actually covered by inverting the q-lattice on its own
relations. Two sessions
went into explaining a "deficit" that was a `-J` flag meaning something other
than assumed; the check takes seconds.

It differs from GGNFS's `test_sieve.sh` in three ways. It can sweep
**geometry** (`--geom logI,J`, repeatable) and the **factor-base bounds**
(`--rlim`/`--alim`), which a per-I compiled siever cannot. It reports
**energy**, taken from the run log's board-watt samples. And its observed and
expected counts are **(q, rho) pairs**, rather than GGNFS's per-prime-q count;
the distinction is stated in its own header because confusing them has cost
this project one wrong headline already (`bench/RESULTS.md` finding 57).

Two cautions. The default `--width 2000` sieves ~100 pairs per point, which is
a few seconds but visibly noisy: on the C194 the projected total moved 11%
between a 2,000-wide and a 20,000-wide sample. Widen it when the number
matters. And the projected days are **GPU-busy** days with startup excluded —
a real run adds the factor-base load once and whatever the host steals.

### Progress output

One `\r` line, updated every 30 s, reported against whichever goal is in force:

```
q=15020857  1222 q  78251 rel +111390 cand  2608 rel/s  26.1%  ETA 0h 01m
```

With `--target-rels` the percentage and ETA track the relation target. Without
one they track an explicit `--nq`, a finite numeric q range, or the q-list
length instead.

**`+N cand` is the cofactorisation queue's occupancy, not pending relations.**
Roughly two thirds of those records become relations on the SNFS job, and the
band summary reports the two separately. It is on the line because it is the
only field that advances on *every* q: `rel` cannot move until a flush, and a
flush is 131,072 candidates — about 67 q on the c183, but 686 q (~40 s) on the
SNFS job, which enqueues 191 records per q instead of 1,956. Before the first
flush the rate and ETA print as `-- rel/s` and `ETA --h --m`, because they are
unknown rather than zero; `+N cand` climbing is how you tell a healthy run from
a stalled one in that window. `--relations` also stages to `NAME.part` until the
band ends, so `ls` shows nothing during it either.

The relation figure advances at queue flush boundaries rather than per q, so it
moves in steps and the rate is slightly understated early in a run. It settles.

**Redirected, that line is written as whole lines instead**, every five minutes
rather than every 30 s, without the carriage return and erase that only mean
something on a terminal. That cadence is fixed and independent of
`--log-every`: the run log is a separate, richer stream on its own clock.

### Stopping and resuming

Relations stage to `NAME.part`; the rename to `NAME` is the "band completed"
marker. After every cofactor flush — the one instant the file holds a whole
number of special-q — the `.part` is fsynced and `NAME.part.ckpt` records the
next `(q, rho)`, the byte offset, the relation count, the derived scale and
allowance, and a job fingerprint.

- **Rerunning the same command resumes**, truncating the `.part` to the
  recorded offset, which also discards any torn final line from a `kill -9`.
- An **empty `.part` with no checkpoint is discarded automatically** and the
  band starts from the beginning. It contains no relations to preserve and can
  be left when a process is killed between creating the staging file and its
  first checkpoint. In a standalone launch, a nonempty uncheckpointed `.part`
  is still refused. This unambiguous empty-file cleanup does not consume a
  BOINC automatic-recovery attempt.
- Under a **BOINC-managed launch**, unusable resume artifacts are discarded and
  that workunit is recomputed from the beginning. This covers a missing or
  malformed checkpoint, mismatched fingerprints or file lengths, and sampled
  relations that fail reconstruction. BOINC volunteers should not have to edit
  private project files to unstick a task. A persistent `.part.recover` counter
  caps this at three automatic recoveries per workunit and is removed only when
  the final result commits, so a deterministic defect cannot restart forever.
  Unreadable files, active locks, other filesystem errors, bad inputs,
  band/command mismatches and compute failures remain fatal because deleting
  output cannot repair them.
  Checkpoints that include candidate output also record the candidate staging
  pathname and file identity. This lets a relaunch that omitted `--candidates`
  remove the matching private staging file without trusting checkpoint text as
  authority to delete some other file. Older checkpoints without that identity
  are preserved rather than partially discarded.
- **`SIGINT`/`SIGTERM` stop cleanly** at the next special-q, draining the queue
  first, so a planned stop loses nothing. A second signal exits at once and
  falls back to the previous checkpoint.
- **`--stop-file PATH`** stops cleanly once that path exists — the same thing
  for a run with no terminal to press `^C` in.
- **`--restart`** discards an existing `.part` and its checkpoint, and clears
  the BOINC automatic-recovery counter.
- `NAME.lock` refuses a second writer and clears itself if the recorded pid is
  gone.
- In standalone mode, a `.part` whose fingerprint disagrees with the current
  command is **refused, not appended to**; managed BOINC recomputes it as
  described above. The polynomial, both sides' bounds, `logI`/`J`, `--sq-side`,
  `--maxbits` and the whole cofactor configuration are all in the fingerprint,
  because every one of them changes which relations a run emits while leaving
  each individual line verifiable.

### Logging an unattended run

```sh
bench --pipeline --cofactor ... --relations msieve.dat \
      --log msieve.runlog --log-every 300
```

`--log` appends; `--log-every` is seconds and defaults to 300 (~864 records
over three days, so there is nothing to rotate). A resumed session appends a
second header block to the same file, which is what happened and is more useful
than a directory of fragments.

The header block records the commit, the full argv, the job fingerprint, the
card and driver, the geometry **labelled `I14e`/`I15e`**, the factor-base
convention with entry counts, the derived gate, and the resume point. Quoting a
band's ms/q without those is how the same number gets compared against two
different CPU baselines — see `bench/RESULTS.md` finding 55.

Each record carries the progress numbers plus the four that say whether they
can be compared to anything:

```
2026-08-16T21:03:44-0400 +2s  q=15000523 nq=42 rel=0 cand=7972 rel/s=0.0 \
    pct=28.00 eta=-- ms/q=41.38 acc/wall=0.800 gpu=98% board=182.4W load=3.26
```

- **`acc/wall`** is `GPU-accounted / wall (excl cofac)`, the running form of
  the band summary's line. There is no "good" value: a faster card sits lower
  when perfectly healthy, so compare it against *your own* idle baseline on the
  same card, job and band length. Falling during a run means the host started
  competing (finding 53).
- **`gpu=` and `board=`** come from NVML, bound to the card by PCI ID. They are
  absent (`n/a`) on a host with no driver library. Board watts are **not** the
  metric of record; whole-box watts need a meter.
- **`load=`** is the one-minute load average — the thing finding 53 says must
  be known on both boxes before any wall-clock comparison between them means
  anything.
- **`+2s` is monotonic elapsed time.** Compute rates from it, not from the
  timestamps: under WSL2 the clock resyncs to the Windows host and can step
  backwards mid-run.

Mid-band warnings — bucket overflow, norm overflow, the candidate cap, a target
not reached — are written to both stderr and the log.

### Special-q on the rational side (SNFS)

An SNFS polynomial often has tiny algebraic coefficients — `x^5 + x^4 - 4x^3 -
3x^2 + 3x + 1` for a `p^11 - 1` job — so the difficulty sits on the *rational*
side, and the special-q and the 3LP `mfb` go there with it. Pass `--sq-side 0`:

```sh
bench --pipeline --cofactor --poly snfs236.job \
      --sq-side 0 --logI 14 --qrange 30000000: --target-rels 150000000 \
      --relations msieve.dat
```

The factor base is still **algebraic** regardless of where the special-q lives.
With no `--fb1`, bench generates the complete algebraic base in-process on the
assigned GPU and fills the rare prime-power/ramified branches with the exact
CPU Hensel implementation.  Passing a native `--fb1` file remains supported.

One thing that looks like an error and is not:

- **"f has no root mod 2"**. That polynomial has `f(0) = 1`, `f(1) = -1` and an
  odd leading coefficient, so 2 can never divide the algebraic norm and fbgen
  correctly emits no `p = 2`. The run says so and continues. The `p = 2` check
  only fires when f *does* have a root mod 2 and the entry is missing anyway.

Special-q generation on the rational side is especially cheap: for every prime
`q` not dividing `Y1`, the root is simply `-Y0/Y1 (mod q)`. It continues above
`rlim`; the rational factor base remains bounded by `rlim` for sieve and TD.

### Lambda: we ignore it, on purpose

**The survivor allowance is derived, not imported.** A `.job` file's lambda is
reported and *not applied*, because it is calibrated to GGNFS's survivor gate
and that calibration does not transfer — see the measurements below. CADO's
automatic rule is not a better source either: on the SNFS job it gives 97.3
bits where GGNFS's gives 91.8, looser still.

What we use instead is `mfb` plus the slack *our own* approximation needs: one
byte unit is `1/scale` bits, so the slack is `max(1.5, 2/scale)`. Measured:

| job / side | job file says | we derive | survivors | relations |
|---|---:|---:|---:|---:|
| SNFS, both sides | 61.10 / 91.80 | **60.50 / 89.50** | −20% | −0.04% |
| c183, side 0 | 68.1 (CADO rule) | **61.5** | −42% | −0.19% |

A fifth to two fifths of the trial-division input, for two hundredths of a
percent of the relations.

Overrides, in precedence order: `--allowance` / `--allowance0` in bits wins
outright; `--lambda0` / `--lambda1` opts back into CADO's rule; otherwise the
derivation above. `bench` warns if an override is more than 2 bits looser than
what it would have derived.

One trap if you calibrate by hand: **a single special-q is not enough of a
sample.** On the c183's parity q the tighter bound looks like it costs 1
relation in 37 (2.7%); over 120 special-q the real rate is 0.19%.

### For reference: GGNFS and CADO do not mean the same thing either

You should not need this now that the lambda is not applied, but the two
conventions differ by which quantity they multiply:

| | lambda is in units of | cofactor bits allowed |
|---|---|---|
| **GGNFS** | `log2(lim)` | `lambda * log2(lim)` |
| **CADO** | `lpb` | `lambda * lpb` |

The GGNFS test-sieve's own suggestion confirms it: *"Suggested rlambda: 2.32
(mfbr=56 / log2(rlim=16700000))"* — i.e. GGNFS's lambda is `mfb / log2(lim)`.
CADO's automatic is `0.3 + mfb / lpb`.

**To state one yourself, use `--allowance` / `--allowance0`, in BITS.** That is
the quantity both conventions are really expressing, and it is what the
survivor bound is computed from:

```
bound = (uint32_t)(allowance * scale + 1)
```

(CADO writes this as `(unsigned char)`, which is right for its 8-bit cells. We
sieve 16-bit cells with CINIT 4096, so bounds above 255 are legal here and the
cast has to be wider — an earlier version of this file, and the banner, both
printed the 8-bit truncation.)

Worked example, the c151 above (`alambda 2.5`, `alim 33.5M`, `lpba 30`):

```
GGNFS:  2.5 * log2(33500000) = 2.5 * 25.00 = 62.5 bits   -> --allowance 62.5
CADO :  the same 62.5 bits would be lambda = 62.5/30 = 2.083
CADO's automatic (0.3 + 59/30 = 2.267) would give 68.0 bits — more generous
```

and the rational side (`rlambda 2.4`, `rlim 16.7M`, `lpbr 29`):

```
GGNFS:  2.4 * log2(16700000) = 2.4 * 23.99 = 57.6 bits   -> --allowance0 57.6
```

The byte scale is always derived from the polynomial (CADO's formula,
`las-norms.cpp:237`) and **an explicit `--allowance` / `--scale` overrides
it**, so the two compose: derive what you don't know, state what you do.

If you would rather work in CADO's units, `--lambda1` / `--lambda0` take
CADO-style lambdas and 0 means "use CADO's automatic". They override a `.job`
file's GGNFS-unit lambdas, and say so when they do.

> `--auto-params` used to gate the derivation. It is now the default and the
> flag is accepted and ignored, so old scripts keep working; drop it when you
> next touch them.

## Running on a different GPU

The build ships native cubins for sm_120, sm_89, sm_86 and sm_80, plus
compute_80 PTX. Anything from Ampere onward runs without editing the Makefile.

Two cases still want a Makefile line. **Older than sm_80** needs its own
`compute_XX`, because the driver only JITs PTX to a target **at or above** the
virtual arch — a PTX fallback never reaches down. **Newer than sm_120, or
sm_90 (H100)**, runs from PTX, and that JIT costs seconds on a cold
`~/.nv/ComputeCache` and repeats every process start if the cache is disabled
or evicted. It is far larger than the ~200 ms of module load the `--reps 100`
floor below is calibrated against, so add a native `-gencode` for any card you
intend to publish numbers for.

Grid width comes from `multiProcessorCount`. Every run opens with the line it
derived, on stdout, including when `--blocks` overrides it:

```
grid: 48 SMs x 6 = 288 blocks (dev 0: NVIDIA GeForce RTX 5070, 48 MB L2)
grid: 4608 x 32 for fill (absolute, not per SM)
```

**Check that line first**, and check it names the card you meant. The SM count
is queried at runtime, so it always matches the device actually in use — a
wrong number means the wrong GPU was selected, not a stale binary. A *stale*
binary shows up as the line being **absent**.

The `dev N:` field is the CUDA ordinal actually in use, which is the one thing
that tells two identical cards apart in a log. It is the BOINC client's
`gpu_device_num` assignment under a BOINC build, `--device N` when there is no
assignment, and CUDA's default device otherwise.

For an ordinary (non-BOINC) run, `CUDA_VISIBLE_DEVICES=1` also still works and
renumbers the chosen card to `dev 0`. Do **not** combine it with a BOINC
build: masking hides the very ordinal the client assigned, so `cudaSetDevice`
fails and the task errors out. Use `--device`, or the client's own
`cc_config.xml` `<exclude_gpu>`, instead.

**Fill has its own grid AND its own block width, and neither scales with the
card.** `--blocks` is 6 per SM; fill instead uses a fixed **4608 blocks of 32
threads** (`--fill-blocks` / `--fill-threads` to override). Above the knee every
card is flat, so overshooting is nearly free; below it the cost is steep (288
blocks is 15–38% worse), so undershoot is the expensive mistake.

**The default was 1152 until finding 76 retuned it to 4608** on the c194
production shape (fill −8.6%, wall −5.7%). Read the two numbers with different
confidence: the "same absolute knee on a 48-SM 5070, a 128-SM 4090 and a 170-SM
5090" result was measured **at 1152**, and 4608 was swept **on a 5070 only**.
So the cross-card claim has not been re-established at the shipped default —
which is one reason the banner prints the number.

**`--fill-threads` is separate from `--threads` on purpose.** Fill wants many
narrow blocks; transform, intersect, TD, resieve and the cofactor kernels are
all tuned at 256. At a constant 576 blocks the 5090 measures 2.711 ms at 128
threads against 3.147 at 256 — **16%** — so tuning fill through `--threads`
would have cost five other stages to buy one.

Don't expect a faster card to fix a slow fill: a 5090 with 3.5× the 5070's
hardware still returns far less than that ratio on this stage. The geometry was
measured at one job shape (8192 buckets, 77.4M records) and plausibly moves
with bucket and record count, so sweep it when characterising a new job — and
sweep **both** axes, because holding one fixed is exactly how the old 144 × 256
default was arrived at:

```sh
for t in 32 64 128 256; do
  for b in 576 1152 2304 4608; do
    printf "fill %4s x %-3s " $b $t
    bench --poly JOB.job --fb1 JOB.roots1 --logI 14 --J 8192 \
          --reps 100 --stage fill --fill-blocks $b --fill-threads $t | grep "fill:"
  done
done
```

Note `--blocks` and `--threads` do **not** move fill — the grids are fully
separate, and the startup lines print both. See RESULTS.md finding 52 (and 51,
which it supersedes on the geometry).

Three rules for comparing cards, all learned the hard way (RESULTS.md
findings 48–53):

- **Run on an IDLE host, and say whether you did.** GPU kernel times come from
  `cudaEvent`, so they are blind to host contention: saturating this box's 16
  cores left `fill` and `apply` flat within 1% while wall clock went **24.30 →
  31.27 ms/q**. As throughput that is a **22.3% relation-rate loss**, and half
  the cores already costs **18.4%**, so there is no safe headroom. A busy box therefore reports *perfect* kernel numbers and a
  bad ETA — which is exactly what a card looks like when it is fine and the
  host is not. Never compare a wall-clock or ETA figure across boxes without
  knowing the host load on both; rented and shared boxes are the risk.

  **The pipeline** prints `GPU-accounted / wall (excl cofac)` for this — the
  standalone does not, so a wall-clock or ETA claim has to come from a pipeline
  run. There is no universal "good" value: a faster card spends relatively more
  of its wall on the same host work and so reads *lower* while perfectly
  healthy. Take an idle baseline per card, job and band length, and compare
  against that; a drop against your own baseline is the signal, the absolute
  number is not. (Published values are pending a re-measurement — the first
  pair was taken with a formula since corrected. See RESULTS.md finding 53.)

- **Use `--reps 100` or more.** Below that, the standalone bench's `transform`
  line reports amortized CUDA startup rather than kernel time — it moves 98×
  between `--reps 3` and `--reps 1000` on the same hardware, while `fill` moves
  0.9% and `apply` 2.9%. A stage whose time depends on `--reps` is not
  measuring the kernel. Quote `apply` with its reps setting; `fill` is safe
  without one.
- **Compare stages, never the total.** `fill` and `apply` respond to completely
  different limits — fill to integer throughput, apply to FP32 and memory
  bandwidth — and cards rank differently on each. A 3090 loses `fill` to a 5070
  and wins `apply` against it. `SIEVE CHAIN` averages that away into a number
  that cannot be reasoned about.

The pipeline prints the same three stages under `sieve, both sides`, which is
the honest source for transform: there it runs once per q against a warm
context, the way production does.

## Checking the output

```sh
bench --check-relations ~/nfs/c151/msieve.dat --poly JOB.job --lpb 30 --lpb0 29
```

Rebuilds both norms from `(a,b)` and the polynomial for every relation and
requires each recorded factor to divide exactly, both norms to reduce to 1, and
every prime to sit within its side's lpb. Run it before spending hours on
filtering — it catches a truncated factor, a dropped power of two, or a
composite emitted as prime, none of which any other check sees.

## Post-processing with msieve

Relations come out in GGNFS/msieve format already — `a,b:rational:algebraic`,
hex factors, rational side first, special-q included. So:

```sh
cd ~/nfs/c151          # relations are already here, sieved straight into msieve.dat
cat > msieve.fb <<EOF
N <decimal N>
SKEW <skew>
R0 <Y0>
R1 <Y1>
A0 <c0>
...
A5 <c5>
FRMAX <rlim>
FAMAX <alim>
SRLPMAX <2^lpbr>
SALPMAX <2^lpba>
EOF
echo "<decimal N>" > worktodo.ini

# msieve looks for these RELATIVE TO THE WORKING DIRECTORY
ln -s /path/to/msieve/cub .
ln -s /path/to/msieve/lanczos_kernel.ptx .
ln -s /path/to/msieve/lanczos_kernel.fatbin .

msieve -s msieve.dat -l msieve.log -nc1 -v    # filtering
msieve -s msieve.dat -l msieve.log -nc2 -v    # linear algebra (GPU)
msieve -s msieve.dat -l msieve.log -nc3 -v    # square root -> factors
```

There is **no `N` header line** in `msieve.dat`; N comes from `msieve.fb`.

We generate **no free relations**. CADO made 122,390 for the c123 and msieve's
filtering still cleared its target without them, so they are not load-bearing at
that size. Watch for it on larger jobs.

If `-nc1` reports *"filtering wants N more relations"* and singleton removal
collapses the set (e.g. 8.6M relations against 15.0M unique ideals reducing to
279), that is simply not enough relations — sieve more, do not tune.

## Measured

| | c123 | c151 |
|---|---:|---:|
| logI | 13 | 14 |
| lpb (alg / rat) | 29 / 28 | 30 / 29 |
| **ms per special-q** | **10.14** | **24.49** |
| relations / q | 272.1 | 70.62 |
| relations / sec | 26,839 | 2,883 |
| special-q used | — | 949,331 of 1,088,865 |
| relations | — | 67,043,952 |
| reconstruction gate | all pass | 48,420 / 48,420 |

The c151 column is a **complete band**, not a sample. The earlier 24.75 ms and
79.8 rel/q came from the first few thousand special-q; over the full run to
`--target-rels 67000000` the yield falls to 70.62 rel/q, because yield decays
as q climbs and the band consumed 87% of the available special-q. Size a job
off a full-band figure, not off a test sieve at the bottom of the range.

The c123 factored end to end: 1,093 s sieving, 171 s filtering, 61 s linear
algebra, 153 s square root. The c151's relations built a matrix in msieve.

### SNFS, special-q on the rational side

`376364081347875370546831^11 - 1`, difficulty 235.76, `--sq-side 0`, I14,
`mfbr 88 / lpbr 31` (3LP on the rational side). Against `gnfs-lasieve4I14e` at
the same q = 30M, with the rational factor base matched to GGNFS's truncated
one (1.92M vs its 1.86M primes):

| | GGNFS, 1 core | this, GPU |
|---|---:|---:|
| ms per special-q | 919 | 38.84 |
| relations / q | 40.8 | 46.14 |
| **ms per relation** | **22.52** | **0.842** |

**26.7× per relation against one core**, so roughly 3.3× against eight. Same
order as the c151's 2.03×, which is the reassuring part — the rational-side
path is not a special case that happens to look good.

**Relation sets compared directly**, same q range `[30000000, 30001003]`,
GGNFS run locally rather than quoted (`gnfs-lasieve4I14e -v -n0 -c 1000 -f
30000000 -o rels.out -r result.job`, 62 special-q, 2531 relations, 1.127 s/q):

| | count |
|---|---:|
| GGNFS relations | 2,531 |
| ours | 3,357 |
| **in both** | **2,531** |
| **GGNFS found, we missed** | **0** |
| ours only | 826 |

We are a strict superset: **zero misses against a real GGNFS run**, and 3,357
of 3,357 pass `--check-relations`. That is the correctness result and it
stands.

**The 826 extras are mostly duplicates, not extra yield.** GGNFS's `lowering
FB_bound to 29999999` is not a limitation — it is deliberate duplicate
avoidance. A relation whose special-q-side norm has primes p₁ < p₂ is found at
q = p₁ *and* again at q = p₂. Truncating the sq-side base at the current q
means only the largest of those primes ever finds it, so GGNFS collects each
relation exactly once. We do not truncate, so we collect it repeatedly.

Measured, not assumed:

| | |
|---|---:|
| mean sq-side primes in [30M, rlim] per relation | 1.82 |
| P(re-found when that prime is sieved as q) | 71.8% (n = 1,023) |
| **finds per unique relation** | **1.34** |
| **duplicate share of raw output** | **~25%** |

Calibrated against ground truth: msieve's filtering of the **c151** found
10,594,292 duplicates in 67,165,877 relations — **15.8%**, 1.187 finds per
unique. Back-solving that run gives P(re-find) = **0.728**, against the 0.718
measured directly on the SNFS job, so the model holds across two independent
jobs.

**Confirmed directly, 2026-08-07**, by replaying both corpora off disk instead
of back-solving — no GPU. On the c151 the replay finds **10,594,292**
duplicates, digit-for-digit the count above, at 1.1877 finds per unique.

**But over a different total, and that is not yet explained.** The paragraph
above quotes msieve's count over 67,165,877 relations; the file on disk holds
67,043,952, which is also what the run record at the top of this document says.
Those 121,925 lines make the duplicate share 15.77% by msieve's pair of numbers
and 15.80% by the replay's. An exact match on the *count* across different
totals is what you would see if the extra lines were all unique — free
relations being the obvious candidate — which would validate the
duplicate-detection logic and leave only the denominator open. Until someone
checks that, this is a strong agreement, not an exact reproduction:

| | measured here | n | this table |
|---|---:|---:|---:|
| mean sq-side primes, [30M, rlim], SNFS | 1.7918 | 19.6M rel | 1.82 |
| P(re-found), SNFS job | **70.2%** | 1,089,564 | 71.8% (n=1,023) |
| P(re-found), c151 | **72.3%** | 13,709,863 | 0.728 (back-solved) |

Two traps, both of which bit on the way to those numbers. **The raw
slot-take rate is biased low** — a relation is in the corpus only because one
slot already hit, so subtracting that forced hit from both numerator and
denominator understates P; at k = 2 the correction is exact, `P = 2r/(1+r)`,
turning a raw 54.1%/56.6% into 70.2%/72.3%. And **the band must come from the
run log, not from the corpus**: an ordinary prime p divides ~N/p relations, so
on a 67M-line corpus every prime near 1M looks like a special-q. Taking the
c151's band as [1M, 33.5M] rather than [15M, 33.5M] drops P to 18.6%, with
whole separation buckets reading exactly 0.0% — the tell that the low-end
"q" were never sieved at all. The 1,088,865 count above is the cross-check.

P(re-found) also decays with separation, gently: 74% at q₂/q₁ ≈ 1.04 down to
61% by q₂/q₁ ≈ 2 (bias-corrected, c151). Treating it as one constant is fine
for a whole-band estimate and wrong for a narrow A/B.

(An earlier version of this table said 37%, from the naive
`1 + (mean-1) x P`. That is wrong: a relation with k eligible primes still only
needs ONE hit to appear at all, so the correct form is
`P / sum_k (f_k/k)(1-(1-P)^k)` with the population recovered as `n_k ~ f_k/k` —
sampling relation LINES over-weights relations that were found often.)

The re-find probability is 72% rather than 100% because (a,b) must also land
inside the sieve rectangle of the *other* prime's lattice, which is a different
lattice. Confirmed directly: relations found at q ≈ 30.0M reappear when
[31004593, 31107961] is sieved as special-q.

So over a full run we do ~1.34× the trial division and cofactorisation for the
same unique yield, and our raw relation counts are inflated by the same factor.
**Quote unique relations, or quote raw and say so.** GGNFS's 40.8 rel/q are
unique; the 46.14 rel/q in the table above are not, so the honest comparison is
~34.4 unique/q against GGNFS's 40.8 — at matched factor bases GGNFS collects
*more* unique relations per special-q than we do, because it collects each one
only once. What we buy for that is not needing to sieve the full q range to
find a relation whose largest sq-side prime sits above `qmax`.

That last clause is the whole risk in truncating, and it is now measured. Under
truncation a relation is found at its **largest** sq-side factor-base prime, so
if the band stops below that prime it is not deduplicated — it is never found.
On the c147's as-run band ([15.00M, 15.15M], 0.4% of `alim`) the `mfb` ceiling
alone says truncation would find **nothing at all for 22.68%** of unique
relations. Truncation is yield-neutral only over a band reaching `lim`.

And the dedup it buys is capped by `mfb` headroom, because an unsieved prime in
`(q, lim]` does not disappear — it lands in the cofactor. Measured floors over
a band reaching `lim`: **−15.0%** of emissions on the c151 and **−17.3%** on
the c147 (both `mfb` 59), but only **−6.1%** on the SNFS job (`mfbr` 88). The
rest of the way down to 1.00× is the survivor gate's to give, which makes this
one experiment with "our gate is looser than GGNFS's" below, not two.

The `ms per relation` row above is therefore **raw**, not unique. Against
unique relations it is ~1.13 ms, so ~20× one core rather than 26.7×. And the
GPU was shared with another job when the 38.84 ms/q was taken, so treat the
timings here as indicative until re-measured on a quiet card.

### Our survivor bound is looser than GGNFS's at the same lambda

Do not carry GGNFS lambda intuition across. Sweeping the SAME q range on the
SAME job, `gnfs-lasieve4I14e` loses 17.3% of its yield going from 91.8 to 87.5
bits; we lose 0.07% going from 91.8 to 88.0:

| bits | GGNFS | ours |
|---|---:|---:|
| 91.8 | 2,531 | baseline |
| 87.5 / 88.0 | 2,092 (−17.3%) | −0.07% |
| 83.7 / 85.0 | 1,952 (−22.9%) | −6.25% |

GGNFS submits 978 cofactors per special-q (`COF: 60664 tests`, 62 q); we submit
1,426 for comparable unique yield. So at a nominally identical bound we admit
noticeably more, and the surplus is almost all unproductive — which is why
tightening is free for us and expensive for GGNFS. **We are paying in time, not
in relations.** Why the two bounds differ at matched nominal bits is not
understood yet; see the tasks. Until it is, tune `--allowance` by measuring
this tool, not by translating a lambda that worked in GGNFS.

**Corroborated on a second job and a real corpus (2026-08-19).** Against
AS276's own 1.5B-relation GGNFS output, over identical special-q and an
identical rectangle, we recover **3,044 of their 3,045** relations and emit
**64 of 4,089 (1.6%) that exist nowhere in their corpus** — all reconstructing
their norms exactly. Same direction, same rough size: a looser gate, paid for
in time rather than in relations. RESULTS finding 69.

3LP is real and load-bearing here — over a 178-q band at q = 20M, large primes
per relation on the rational side came out 0LP 1,261 / 1LP 3,803 / 2LP 4,009 /
**3LP 1,183**, i.e. 11.5% of relations need the third prime. The algebraic side
tops out at 2LP exactly as `mfba 59 / lpba 30` predicts. All 10,256 relations
passed `--check-relations`.
