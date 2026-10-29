# oracle/ — frozen inputs for the cuda-sieve feasibility probe

Inputs in this directory are **frozen**. The original C183 copies were taken
on **2026-08-01** from the owner's live `../test-sieve/` directory so the target
job survives polynomial changes there. Later inputs and captures have their
own dates below.

These files are correctness oracles and CPU performance baselines for the
project's primary **GPU-resident relation-collection** goal. Their presence does
not imply that CADO or GGNFS cofactorization remains in the production data
path. A GPU-sieve + CPU-cofactor hybrid is tracked separately as an optional
deployment mode.

## The target job

**C183, GNFS, degree 5.** This is the number every graded result in
`../prototype.md` refers to.

| file | what it is |
|---|---|
| `input.job` | GGNFS-format job: poly + `rlim`/`alim`/`lpbr`/`lpba`/`mfbr`/`mfba`/lambdas |
| `result.job` | same, as written back by `test_sieve.sh` (adds `lss: 0`) |
| `c183.poly` | **CADO-format poly, generated here** — `../test-sieve/cado_tmp.poly` was for a *different* number (a C203) and was never regenerated for this job |
| `input.job.afb.0` | GGNFS algebraic factor base for this poly, `alim=134.2M` (61 MB) |
| `ggnfs_test_sieve_15e.txt` | the `gnfs-lasieve4I15e` test sieve this project's per-q budget is derived from |
| `test_sieve.sh`, `cado_test_sieve.sh`, `estimate_norms.py` | the owner's tooling as it stood, for reproducing the above |
| `MANIFEST.sha256` | checksums for all of it, including the git-ignored large binaries |

## Correctness gates and captures (added 2026-08-03/04)

These are review artifacts, not frozen inputs — they are reproducible, and the
commands that produce them are in `../prototype.md` under "Artifacts and how to
reproduce". They are checksummed here because the gates they support are cited
as passed.

| file | what it is |
|---|---|
| `cado-after-sieve-survdump.patch` | 3-line CADO patch dumping the `after_sieve` survivor `(a,b)` set, which `-batch-print-survivors` does **not** do (it emits the post-TD cofactor list — 1,851 records, not 797,028). Apply, rebuild `las`, keep the binary **outside** the CADO tree, revert the source. Enabled by `CUDASIEVE_SURVDUMP=<path>`; inert when unset. |
| `c183.q120000053.after_sieve_ab.powlim.txt.gz` | 795,845 `(a,b)` at `-powlim0/1 32767` — **the oracle to gate against** |
| `c183.q120000053.after_sieve_ab.nopowlim.txt.gz` | 797,028 `(a,b)` at default `powlim`; kept only to reproduce the `powlim` finding |
| `c183.q120000053.relations_ab.txt` | las's 37 relations at this q |
| `c183.q120000053.cofac_candidates.txt` | 1,851 `a b cofac0 cofac1` from `-batch-print-survivors` |
| `las_q120000053_after_sieve_powlim.log`, `las_q120000053_batchsurv.log` | capture logs |
| `relcontain_band.c` | the relation-containment harness: inverts `(a,b)` through **our** basis, attributes relations to a lattice by `a ≡ rho·b (mod q)`, classifies contained / outside-region / in-region-miss |
| `relation_containment_band.txt` | its result over the frozen band at the default bound: **3,026 of 3,026** in-region relations contained, zero misses, over 67 lattices |
| `relation_containment_bound128.txt` | the same gate rerun at the tightened bound 128 — identical result, which is what licenses adopting it |
| `c183.q120000000-120001000.relations.default.txt` | the 3,162 las relations that harness runs against |
| `bound_sweep.sh` | the coarse survivor-bound sweep, `lambda1` in steps of 0.1. **Note the resolution:** one step is 4.08 units of bound, so it never sampled the cliff edge |
| `bound_sweep_fine.sh` | the 1-unit refinement that did |
| `bound_sweep_results.txt` | both passes, merged. **The zero-loss floor is bound 128** — 2.46× fewer survivors than the default 143 for zero relations lost. Use **130** in the GPU path until the containment gate is rerun at a candidate bound; the floor gives no margin against our two-directional one-unit norm error |
| `c183.q120000000-120010000.cofac_candidates.tar.gz` | **1,062,811 cofactor candidates over 548 special-q** in [120000000, 120010000] — the real C183 `lpb 31/32` post-sieve population, 1,939 records/q, for the controlled cofactorization rerun (31 MB) |

**A caution on the relation artifacts.** Containment establishes that las's
in-region relations are all survivors of ours. It does **not** establish yield
equivalence: 20 of the band's 67 lattices use a different (in fact
better-reduced) basis and therefore cover a different `(a,b)` region. Final
relation yield has to come from cofactoring our own region — see `PARITY.md`.

## Parameters (C183)

```
rlim  67100000        lpbr 31        mfbr 60        rlambda 2.35
alim 134200000        lpba 32        mfba 92        alambda 3.5
```

`mfba=92` with `lpba=32` means **three large primes on the algebraic side**. It
makes this a useful stress case for GPU cofactorization and a demanding test of
YAFU's existing GPU path; it does not make CPU cofactorization an architectural
requirement. See the post-sieve section of `../prototype.md`.

## Operating point (C183, locked)

- CADO: `-A 29` (≡ `-I 15`), `-adjust-strategy 0`, `-sqside 1`
- Job q-range: `[50M, 190M]` — to be sieved for real on NFS@Home
- Frozen benchmark band: `-q0 120000000 -q1 120001000`
- Sensitivity points: `q0 = 50M` and `q0 = 190M`

## What is NOT here, and why

- **CADO factor bases.** `../test-sieve/cado_roots1.gz` belongs to the C203, not
  this number. Both sides must be generated fresh:
  ```
  makefb -poly c183.poly -side 0 -lim  67100000 -out cado_roots0.gz -t 16
  makefb -poly c183.poly -side 1 -lim 134200000 -out cado_roots1.gz -t 16
  ```
  Side 0 has never existed at all — `las` computes the rational side on the fly.
- **`gnfs-lasieve4I15e`.** Lives in `../test-sieve/`; the owner is changing the
  poly, not the binaries.

## Provenance of the test sieve

`ggnfs_test_sieve_15e.txt` was produced by `test_sieve.sh`, which launches **one**
siever process with no backgrounding (`test_sieve.sh:187`). All of its rates are
therefore **single-core**, and its "ETA: 315d" is a one-core ETA. The derived
constant that matters — **~3.1 core-seconds per special-q, flat across the whole
q-range** — is per core.

## The timing workload (added 2026-08-11)

**C147, GNFS, degree 5.** A different number from the C183 above, and easy to
conflate. The C183 is the *parity* oracle — factor bases, survivor containment,
cofactor candidates — and the **C147 is the timing workload**: findings 48, 50
and 54, the 144×256 vs 1152×32 geometry result, and the host-load experiment in
`../bench/RESULTS.md` were all measured on it. (Not *every* device timing in that
file: findings 43–44 profile the C183 via `--cadofb ../oracle/c183.fb1`. Check
which job a command names before reusing its numbers.) Neither C147 file was in
the repo before this entry, so RESULTS.md's reproduction commands referred to
paths a fresh clone did not have.

Its parameters are **not** the ones under "Parameters (C183)" above:

```
rlim 16700000        lpbr 29        mfbr 56        rlambda 2.4
alim 33500000        lpba 30        mfba 59        alambda 2.5
```

| file | what it is |
|---|---|
| `c147.job` | GGNFS-format job: poly + `rlim`/`alim`/`lpbr`/`lpba`/`mfbr`/`mfba`/lambdas. Tracked in git — it is 543 bytes |
| `c147.roots1` | algebraic-side CADO-format factor base, `alim=33.5M` (29 MB). **Git-ignored**, manifest-pinned, regenerated by the command below |

```
cd ../bench && make fbgen
./fbgen --poly ../oracle/c147.job --maxbits 14 \
        --threads 8 --out ../oracle/c147.roots1
```

`--lim` is omitted deliberately: it defaults to `alim` from the job file
(`fbgen.c`, `if (!lim) lim = P.alim`), so spelling 33500000 again here would just
be a second copy to drift. `--threads` affects runtime only, not output.

**`--maxbits` must equal the `--logI` you sieve at**, and the frozen file is
`logI=14`. This is not a magic constant: `bench_main.cu` defaults `maxbits` to
`cfg.logI` (`if (!maxbits_set) maxbits = cfg.logI;`), and `fbgen`'s own usage
text recommends logI. `fbgen` alone defaults to 15, and at 15 the output differs
from the frozen file in 21 lines — 19 prime-power ladder entries between 2^14 and
2^15, plus the two-line header change. With 14 it reproduces
`546a78a1…423a9745c`, the hash in `MANIFEST.sha256`.

Because that header is written into the file, `head -4 c147.roots1` reports the
`maxbits` a stale copy was built with — cheaper than hashing 29 MB when you just
want to know which one you have. Note that a mismatch against `--logI` is only a
`note:` from `bench`, not an error, so it will scroll past in a band run.

This needs no CADO: `fbgen` is the project's own generator, gated byte-for-byte
against CADO `makefb` revision `0574bc39d` by `../bench/fbgencheck.sh`, which
includes a C147 case at `--maxbits 14` pinned to the manifest hash.

## The geometry / VRAM-sizing workload (added 2026-08-18)

**C194, GNFS, degree 5, NFS@Home.** A *third* number, and the one to reach for
when reproducing anything about sieve-region shape or device memory: findings
64, 65 and 66 in `../bench/RESULTS.md` are all C194, at algebraic special-q over
the window `[250000000, 250004000]` (200 `(q, rho)` pairs). It was not in the
repo when those findings were written, so their commands named a `c194.job` a
fresh clone did not have; that is what this entry fixes.

Its parameters are not the C183's or the C147's:

```
rlim 160000000       lpbr 32        mfbr 63        rlambda 2.4
alim 240000000       lpba 33        mfba 95        alambda 3.55
```

| file | what it is |
|---|---|
| `c194.job` | GGNFS-format job: poly + bounds + lambdas + `lss: 0`. Tracked in git — it is 644 bytes |
| `c194.roots1.m15` | algebraic factor base at `--maxbits 15`, `alim=240M` (197 MB). **Git-ignored**, manifest-pinned |
| `c194.roots1.m16` | the same at `--maxbits 16` (197 MB). **Git-ignored**, manifest-pinned |

```
cd ../bench && make fbgen
./fbgen --poly ../oracle/c194.job --maxbits 15 --threads 12 --out ../oracle/c194.roots1.m15
./fbgen --poly ../oracle/c194.job --maxbits 16 --threads 12 --out ../oracle/c194.roots1.m16
```

**Two files, because `--maxbits` must equal the `--logI` you sieve at** and the
geometry findings sieve at both 15 and 16. The per-`logI` suffix matches the
`fbase.m15` convention `../bench/testsieve.sh` already uses. The difference
between them is tiny — 13,160,671 ideals at m16 against 13,160,645 at m15, i.e.
**26** extra prime-power ladder entries in `(2^15, 2^16]` (211 against 185
prime-powers), and 322 bytes of on-disk size. That is worth knowing when
costing a `logI` change. In VRAM those 26 entries are 26 x 25.4 B ~ 660 B by
finding 64's per-entry rate -- that rate is device bytes per factor-base entry
and has nothing to do with the 322 on-disk bytes, which work out to a different
number because the text format is a different thing. Either way it rounds to
nothing, so **essentially all of the VRAM difference between two geometries is
the bucket array** (`../bench/RESULTS.md` finding 64).

Both carry `maxbits` in the header, so `head -4 c194.roots1.m16` identifies a
stale copy without hashing 197 MB. A mismatch against `--logI` is only a `note:`
from `bench`, not an error.

## C143 cado-gpu comparison (2026-10-12)

[`c143.job`](c143.job) preserves the degree-5 GNFS polynomial and bounds from
the owner's `~/cado-gpu/c143-649044015427` campaign.
[`c143.poly`](c143.poly) is its original CADO polynomial file. This is a
**performance comparison input**, with two large primes on both sides, not a
new relation-containment oracle. Findings [105](../bench/RESULTS.md#finding-105)
and [106](../bench/RESULTS.md#finding-106) record the results and proposed CPU
work; the current plan is [STATUS item 4](../bench/STATUS.md#low-cpu-plan).

| setting | value |
|---|---|
| `rlim` / `alim` | 14,000,000 / 18,000,000 |
| `lpbr` / `lpba` | 30 / 30 |
| `mfbr` / `mfba` | 57 / 58 |
| CADO `lambda0` / `lambda1` | 1.905 / 1.94, supplied on the command line |
| special-q side / start | algebraic (1) / 2,200,000 |
| first comparison band | `[2200000, 2300000)`, 6,983 prime/root pairs |
| full campaign target | 78,000,000 **raw** relations |
| CADO geometry | `A=26`, `logI=13`, q-dependent `J <= 8192` |
| cuda-sieve campaign geometry | `logI=14`, `J=4096`, `region=13`, `maxbits=13` |

The lambdas use CADO units. The job deliberately omits GGNFS
`rlambda`/`alambda`; do not substitute those for the command-line overrides.
The I14 experiment deliberately keeps `maxbits=13` to match the saved CADO
factor base's prime-power convention.

The comparison's small evidence was in `c143-comparison-20261012/`: our I13,
I14 and campaign outputs with their run logs and `/usr/bin/time` results, and
the CADO campaign's parameter snapshot, dispatch log, per-workunit band
summary, relation-file samples and duplicate-removal summaries. It left the
tree with the other dated folders ([below](#dated-evidence-folders)). The
multi-gigabyte relation files remain local in their original directories.

To repeat the I14 first band, from the repository root, use a fresh output
directory so an earlier run cannot resume. **Since 2026-10-13 this command
runs the new fill defaults** (finding 107: about 68.9 s, not the captured
~77-81 s). To reproduce the configuration of the captures above, add
`--fill-rows 0 --no-fill-gate`.

```sh
mkdir -p work
run_dir=$(mktemp -d work/c143.XXXXXX)
/usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S max_rss_kb=%M' \
  -o "$run_dir/wall-time.txt" \
  ./bench/bench --pipeline --cofactor --poly oracle/c143.job \
    --logI 14 --J 4096 --region 13 --maxbits 13 \
    --sq-side 1 --qrange 2200000:2299999 \
    --lambda0 1.905 --lambda1 1.94 \
    --relations "$run_dir/relations.dat" --log "$run_dir/run.log" \
    >"$run_dir/output.txt" 2>&1
```

For I13, change `--logI 14 --J 4096` to `--logI 13 --J 8192`. For the full
campaign, replace the bounded q range with
`--qrange 2200000: --target-rels 78000000`. Our upper q endpoint is inclusive;
the saved CADO command uses the exclusive `-q1 2300000`.

With the compared cado-gpu checkout built at `~/code/cado-nfs-public`, the
corresponding first-band commands are:

```sh
cado_dir="$HOME/code/cado-nfs-public/gpu-siever"
mkdir -p work
cado_run=$(mktemp -d work/c143-cado.XXXXXX)
"$cado_dir/build/cado/sieve/makefb" -poly oracle/c143.poly \
  -lim 18000000 -maxbits 13 -side 1 -t 16 \
  -out "$cado_run/roots1.gz"
"$cado_dir/scripts/las-glas" -poly oracle/c143.poly \
  -q0 2200000 -q1 2300000 -A 26 \
  -lim0 14000000 -lim1 18000000 -lpb0 30 -lpb1 30 \
  -mfb0 57 -mfb1 58 -lambda0 1.905 -lambda1 1.94 \
  -ncurves0 13 -ncurves1 14 -fb1 "$cado_run/roots1.gz" \
  -out "$cado_run/relations.gz" -t 16 -stats-stderr \
  >"$cado_run/output.txt" 2>"$cado_run/stderr.txt"
```

The launcher replaces `-t` with its host CPU allocation unless
`GLAS_THREADS` overrides it; record that setting when comparing CPU budgets.
`GLAS_GPU_STATS=1` enables additional diagnostics for a new measurement.
These are reproduction recipes, not new measurements. Compare fresh,
interleaved runs and record host load and aggregate process CPU time.

## Dated evidence folders

Findings 105-131 kept their small evidence here, one folder per finding:
captured outputs, the gate, timing and trace scripts that produced them,
profiles and diffs. The folders were removed from the tree on 2026-10-29
(548 of its 665 files). `../bench/RESULTS.md` still names their files by path,
and commit `cadcf3c` has every one of them: `git show
cadcf3c:oracle/<folder>/<file>` prints one, and `git checkout cadcf3c --
oracle/<folder>` restores a folder. That commit's `MANIFEST.sha256` pins
them, and this file as it stood there describes each one. Evidence since
finding 132 has stayed in the git-ignored `../work/`.

The four tools later work still runs moved to `../bench/tools/`: `analyze.py`
and `gaps.py` (kernel concurrency and idle gaps from an `nsys` SQLite export,
finding 113), `kernsum.py` (per-kernel ms per special-q, finding 115) and
`relcmp.py` (two relation sets compared, finding 107).

| folder | finding | what it held |
|---|---|---|
| `c143-comparison-20261012/` | [105](../bench/RESULTS.md#finding-105) | C143 against cado-gpu (above) |
| `c143-fill-20261013/` | [107](../bench/RESULTS.md#finding-107) | fill's row tier and side-0 gate: fillcheck, interleaved sweeps, cado-gpu ablations |
| `c143-qpipe-20261013/` | [108](../bench/RESULTS.md#finding-108) | two special-q in flight |
| `qpipe-slabs-20261013/` | [109](../bench/RESULTS.md#finding-109) | `--q-pipeline` across slabs |
| `apply-i16-20261014/` | [110](../bench/RESULTS.md#finding-110) | `k_apply` profiled at I16 |
| `ss-two-step-20261014/` | [111](../bench/RESULTS.md#finding-111) | the two-step small-sieve tail, with `sass-structure.txt` (YIELD/BSSY per `k_apply` instance, cited in `bench_kernels.cu`) |
| `c143-h2h-20261014/` | [112](../bench/RESULTS.md#finding-112) | C143 against cado-gpu at equal yield |
| `nsys-c143-20261014/` | [113](../bench/RESULTS.md#finding-113) | Nsight Systems at C143 |
| `hostwait-20261014/` | [114](../bench/RESULTS.md#finding-114) | two host waits removed |
| `i16-h2h-20261016/` | [115](../bench/RESULTS.md#finding-115) | the NFS@Home I16 job against cado-gpu at equal yield |
| `resieve-transform-20261016/` | [116](../bench/RESULTS.md#finding-116) | the resieve filter and the transform's fast path |
| `scan-clears-20261016/` | [117](../bench/RESULTS.md#finding-117) | apply's threshold scan and the bitmap clears |
| `recscan-20261018/` | [118](../bench/RESULTS.md#finding-118) | side 0's record scan, the counter build |
| `rscan-20261018/` | [119](../bench/RESULTS.md#finding-119) | side 0's resieve from its records |
| `tdwait-20261020/` | [120](../bench/RESULTS.md#finding-120) | card idle re-traced after findings 116-119 |
| `qprep-20261020/` | [121](../bench/RESULTS.md#finding-121) | the q boundary at small jobs (`--q-prep`) |
| `prof-c135-20261020/` | [122](../bench/RESULTS.md#finding-122) | c135 profiled against cado-gpu |
| `fillahead-20261020/` | [123](../bench/RESULTS.md#finding-123) | fill ahead |
| `skipoff-20261020/` | [124](../bench/RESULTS.md#finding-124) | cado-gpu at equal yield, its `p < 96` skip off |
| `slicecut-20261021/` | [128](../bench/RESULTS.md#finding-128) | apply's slice cut on large factor bases |
| `tinyfold-20261021/` | [129](../bench/RESULTS.md#finding-129) | `--tiny-fold` |
| `logrec-20261022/` | [130](../bench/RESULTS.md#finding-130) | log p in the bucket records |
| `tinytable-20261022/` | [131](../bench/RESULTS.md#finding-131) | `--tiny-table` |
