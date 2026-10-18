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

The small evidence files in [`c143-comparison-20261012/`](c143-comparison-20261012/)
are tracked and pinned in `MANIFEST.sha256`:

- `cuda-i13.*`, `cuda-i14.*`, `cuda-campaign.*`: verbatim output, run log,
  and `/usr/bin/time` result, copied respectively from root-level local
  `c143-timing.fHQfNX`, `c143-timing.FinIL9`, and `c143-timing.7GYZN2`.
  Paths and command lines in these historical captures are unchanged.
- `cado.parameters.txt`: the campaign's `c143.parameters_snapshot.1`.
- `cado.campaign.txt`: original first-workunit dispatch and target-reached
  log lines, plus the original first sieve command.
- `cado.bands.tsv`: summary fields from the 165 accepted-range workunit
  stderr files, with source filenames. It excludes later surplus workunits.
- `cado.samples.txt`: non-per-q comment lines from four compressed relation
  files, followed by explicitly labelled geometry aggregates derived from
  their per-q headers. Relation rows are not copied.
- `cado.dup2.slice*.txt`: verbatim duplicate-removal summaries.

The multi-gigabyte relation files remain local in their original directories;
they are not part of this capture. The temporary root `c143-timing.sh` was
removed after preserving its effective commands here and in the run logs.

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

## C143 fill tier, fill gate and cado-gpu ablations (2026-10-13)

[`c143-fill-20261013/`](c143-fill-20261013/) holds the small evidence for
[finding 107](../bench/RESULTS.md#finding-107), pinned in `MANIFEST.sha256`.
Paths and command lines inside the captured scripts are verbatim, and the
runs they name live in the git-ignored `../work/`.

- `fillcheck.txt`: the identity gate with the new defaults, 44 of 44
  byte-identical.
- `sweep-*.txt`, `confirm*.txt`: interleaved A/B lines (one per run, host load
  on the confirmation lines), from `fillab.sh` and `confirm.sh`. `summ.py`
  turns them into the median tables.
- `firstband-*`: the 6,983-q first band, old fill against the new defaults at
  I13 and I14, with `--relations` and `/usr/bin/time` (`firstband.sh`).
- `cado-<arm>.{stderr,time,threads}.txt`: the six `las-glas` arms of
  `cado-ablation.sh`. `threads` is a `ps -L` snapshot 15 s into the run.
- `relcmp.py` and `cado-*_vs_cuda-i13.txt`: the first band's relation sets,
  CADO default and CADO with `GLAS_SKIP_BOUND=0`, each against cuda-sieve I13.

The CADO arms run the launcher with `-fb1` pointing at a copy of the
campaign's `c143.roots1.gz` in `work/`, so glas writes its factor-base cache
there and not into the campaign directory. The environment variables are
the only difference between arms.

## C143 two special-q in flight (2026-10-13)

[`c143-qpipe-20261013/`](c143-qpipe-20261013/) holds the small evidence for
[finding 108](../bench/RESULTS.md#finding-108), pinned in `MANIFEST.sha256`.
Captured scripts keep their verbatim paths; the runs live in the git-ignored
`../work/qconc/`.

- `qspan-i13.txt`, `twoproc.*`, `overlap.*`: the measurements that sized the
  prize before the build: idle time per q, two processes sharing the card,
  and the two sides overlapped (`--fill-concurrent`).
- `qpab.*`: the interleaved A/B, single-queued against `--q-pipeline`, on
  C143 I13/I14 and c183.
- `fillcheck.txt`, `fillcheck-c183-single-slab.txt`: the identity gate,
  60 of 60 byte-identical; `identity-first-runs.txt`: the first per-job
  checks; C194 stayed single-queued on its four-slab plan until finding 109.
- `stop-resume.*`: a band stopped with a q in flight and resumed;
  `stop-resume-after-review.*` and `qpab-after-review.txt` repeat the stop
  test and the I13 A/B after the review fixes.
- `cpu.txt`: host CPU with and without the pipeline.
- `firstband-*`: the 6,983-q first band with the new defaults.
- `vram-holder.cu`, `wsl-vram-holder.c183.txt`: the attempt to force the
  VRAM fallback. Under WSL2 the driver evicted the holder's memory, so the
  run still saw ~9 GB free.

## Multi-slab --q-pipeline (2026-10-13)

[`qpipe-slabs-20261013/`](qpipe-slabs-20261013/) holds the small evidence for
[finding 109](../bench/RESULTS.md#finding-109), pinned in `MANIFEST.sha256`.
`tests.sh` is the plan that produced `tests.txt` (identity gate, the
NFS@Home lpba-34 job single-queued against pipelined, and the interleaved
C194/NFS@Home timing); `fillcheck.txt` is that gate's log. **Section 3 of
`tests.txt` is vacuous:** the stop file is read every 64 q and that 120-q
band ended before a check saw it, so its IDENTICAL (13,487) compares two
uninterrupted runs. The real test was run by hand with the same C194
arguments at 400 q: `stop-resume-c194.base.txt` is the uninterrupted
single-queued band (44,846 relations), and `stop-resume-c194.stopped.txt` /
`.resumed.txt` are the pipelined run stopped after 192 q at q=80003687 with
work in flight and its resume, byte-identical to the base. `tdskip.*` is
the `PIPE_K=8` run in which slabs are dropped for truncated trial division,
pipelined against single-queued.

After the code review: `review-rerun.sh` produced `review-rerun.txt`. It
runs `skipslab.sh` on a `BN_LIMBS=4` build (log `skipslab.txt`: norm-width
skips on a four-slab plan, both sides, single-queued against pipelined),
rebuilds the default, runs `fillcheck.sh` (log
`fillcheck-after-review.txt`, now with the issued-ahead checks), and repeats
the 400-q stop and resume against the same base, asserting that the band
really stopped (`stop-resume-c194.after-review.*`: stopped after 128 q at
q=80002567). `nfs-after-review.txt` is the NFS@Home job's 40-q identity run
on the fixed build. These reruns shared the card with an NFS@Home client,
so their timings mean nothing. The scripts' paths point into the git-ignored
`work/slabpipe/`. The NFS@Home job's polynomial and factor base lived in the
client's `/tmp` workdir and are not kept here.

## k_apply at I16 (2026-10-14)

[`apply-i16-20261014/`](apply-i16-20261014/) holds the small evidence for
[finding 110](../bench/RESULTS.md#finding-110), pinned in `MANIFEST.sha256`.
`prof.sh` took the source-level ncu profiles of `k_apply` (the NFS@Home
lpba-34 I16 job and c183, q index 3, single-queued). `launches.txt` is the
per-launch duration, instructions and issue figures for base and the
experiment. `stages.txt` is the per-stage shares and stall mix, plus the
64-bit-fallback share, produced by `cat.py`, `cat2.py` (line numbers
remapped for the experiment build), `stall.py` and `catdef.py`; `agg.py`
lists the top source lines. `exp.diff` is the experiment: the thread-tier
load hoist, applied to a `git archive` copy of `5c1ab93` (`build.txt`).
`ab.sh` produced `ab.txt`: relations identity on both jobs, then
single-queued ABBA × 2 and a pipelined ABBA on the NFS@Home job. The ncu
reports (~140 MB) and both builds stay in the git-ignored `work/applyprof/`,
along with copies of the NFS@Home job's polynomial and factor base, which
are not kept here. Landing it: `land-gate.sh` produced `land-gate.txt`
(default fat build, `--verify-only`, `make check`, `fillcheck.sh` with log
`land-fillcheck.txt`, and identity against the base copy's relation files).
`land-recheck.txt` is the landed binary against the base copy, pipelined,
ABBA, at host load ~15.

## Two-step small-sieve tail (2026-10-14)

[`ss-two-step-20261014/`](ss-two-step-20261014/) holds the evidence for
[finding 111](../bench/RESULTS.md#finding-111), pinned in `MANIFEST.sha256`.
`landed-run.sh` produced `landed-run.txt`: the default fat build's resources,
`--verify-only`, apply-parity `--verify` at I16, I17 and I15 (with the new
`ss_first` device gate), `make check`, `fillcheck.sh`, identity on the
NFS@Home job and c183, and fat-vs-fat timing against the saved hoist
binary. `landed-calls.txt` is the 64-bit remainder call count before and
after. The variants that lost: `two-range-run.sh`/`.txt` (two loops back to
back, gates passing and apply +23%) with `two-range-stalls.txt` (the barrier
stall that explains it); `variant2-ab.sh`/`.txt` and `variant2-c183-ab.txt`
(one loop with a per-entry select, +0.8% at c183); `variant3-ab.txt` (a
runtime branch between two loops, stopped). `sass-structure.txt` counts each
variant's `YIELD`/`BSSY` per `k_apply` instance. After the xhigh review:
`review-run.sh` produced `review-run.txt` (build, per-instance SASS with the
trap, `--verify` at four geometries, `make check` with slabtest's new tail
gate, fillcheck, identity on all four pipeline instances against `5c1ab93`,
ncu and timing against the hoist), and `slabtest-mutants.txt` shows two
broken two-step proofs failing that gate. The scripts' paths point into the
git-ignored `work/applyprof/`, where the builds and ncu reports stay.

## C143 against cado-gpu, equal yield (2026-10-14)

[`c143-h2h-20261014/`](c143-h2h-20261014/) holds the evidence for
[finding 112](../bench/RESULTS.md#finding-112), pinned in `MANIFEST.sha256`.
`run.sh` produced `run.txt`: two interleaved end-to-end rounds of CADO
(default and `GLAS_SKIP_BOUND=0`) and cuda-sieve (I13, I14), then
per-kernel arms (CADO one slot with and without the skip, CADO 8 slots with
GPU stats, cuda-sieve `--no-q-pipeline` at I13 and I14). Each arm's CADO
stderr and `/usr/bin/time`, or cuda-sieve band output and time, is kept
here as `cado-<arm>.*` / `cuda-<arm>.*`. `build.txt` names both builds. The
relation files stay in the git-ignored `work/h2h-20261014/`.

## Nsight Systems at C143 (2026-10-14)

[`nsys-c143-20261014/`](nsys-c143-20261014/) holds the evidence for
[finding 113](../bench/RESULTS.md#finding-113), pinned in `MANIFEST.sha256`.
`trace.sh` captured the four timelines (`trace.txt`); `analyze.py` produced
`analysis.txt` (busy, idle, concurrency depth and per-kernel overlap for each);
`gaps.py` produced `gaps-*.txt` (idle charged to kernel transitions, with the
host CUDA calls inside); `gap-detail.txt` breaks down the three cuda-sieve
gaps. The `.nsys-rep` captures and their SQLite exports (~40 MB) stay in the
git-ignored `work/nsys-20261014/`. Recorded at host load 9.5-10.6.

## Host waits removed (2026-10-14)

[`hostwait-20261014/`](hostwait-20261014/) holds the evidence for
[finding 114](../bench/RESULTS.md#finding-114), pinned in `MANIFEST.sha256`.
`pipeline.diff` is the change. `ab.sh` produced `ab.txt` (identity on four
jobs and interleaved timing of `3e79628`, A and B). `gates.sh` produced
`gates.txt` (fat build, `make check`, fillcheck, stop/resume on C194 and C143,
skipslab with its log `skipslab.txt`, the `PIPE_K=8` TD-skip identity).
`final-fat.txt` is fat build against fat build with the SM clock logged.
`trace-quiet.txt` is the quiet-host Nsight Systems comparison (analyze.py and
gaps.py from `nsys-c143-20261014/`). `build.txt` names the binaries.
After the xhigh review: `pipeline-final.diff` is the change as landed
(`pipeline.diff` is the first version). `review-gates.sh` produced
`review-gates.txt`: fat build, `make check`, fillcheck, identity on four
jobs, stop/resume on C194 and C143, relative timing with the card shared
with GPU ECM, `skipslab` and the `PIPE_K=8` TD-skip test. Its forced-overflow
block used whole-percent capacities that either overflowed everything (85%) or
nothing (95-99%). `ovf2.sh` produced `ovf2.txt`, the forced-overflow
comparison at the per-mille settings where only some slabs overflow. `real-timing.txt`
is the clean fat-vs-fat timing of the landed build, with the card free.

## The NFS@Home I16 job against cado-gpu, equal yield (2026-10-16)

[`i16-h2h-20261016/`](i16-h2h-20261016/) holds the evidence for
[finding 115](../bench/RESULTS.md#finding-115), pinned in `MANIFEST.sha256`.
`nfs.poly` is the job in CADO's format and `makefb.log` its factor-base
build. `run.sh` produced `run.txt`: end-to-end CADO (skip off twice,
default once) and cuda-sieve (twice), per-kernel arms (CADO one slot with GPU
stats, cuda-sieve `--no-q-pipeline`), then `relcmp.py` (from
`c143-fill-20261013/`) on the relation sets. `round2.sh` produced
`round2.txt`, a second interleaved round with SM clock and power logged.
Each arm's CADO stderr and `/usr/bin/time`, or cuda-sieve band output and
time, is kept as `cado-<arm>.*` / `cuda-<arm>.*`. `cado-phases.txt` holds
CADO's startup/steady split and host/device cofactor counts, read from the
relation files. `nsys-trace.sh` captured three timelines (`nsys-trace.txt`).
`nsys-analysis.txt` and `nsys-gaps-qp.txt` are `analyze.py` and `gaps.py`
from `nsys-c143-20261014/`, and `kernsum.py` produced `nsys-kernsum.txt`
(ms per q per kernel). `qpcheck.sh`, `qprel.sh` and `full.sh` (`.txt`)
check the q-pipeline against single-queued without nsys. `build.txt` names
both builds. The relation files, `.nsys-rep` captures and SQLite exports
stay in the git-ignored `work/cado-nfs-i16/` and `work/nsys-i16-20261016/`.
Recorded at host load 10-22 (nine CPU ECM processes throughout).

## Resieve filter and transform fast path (2026-10-16)

[`resieve-transform-20261016/`](resieve-transform-20261016/) holds the
evidence for [finding 116](../bench/RESULTS.md#finding-116), pinned in
`MANIFEST.sha256`. `change.diff` is the code change.
- **Profiles:** `prof.sh` captured the ncu reports (one q of the NFS@Home
  job: eight resieve launches, two transform launches). `metrics.py`
  produced `*-metrics.txt`, and `agg.py` from `work/applyprof/` produced
  `*-lines.txt`, the per-source-line split.
- **Identity:** `ident.sh` (`ident.txt`) on four jobs and three arms.
- **Experiments:** `ab.sh`, `ab2.sh` and `ab3.sh` (`.txt`) are the free-card
  A/Bs at I16, on the other geometries, and at intermediate slab areas.
- **Gates:** `gates.sh` (`gates.txt`) ran them on the final source.
  `check-summary.txt` and `skipslab.txt` are their detail.
- **Final timing:** `final-timing.sh` and `c143-recheck.sh` (`.txt`) ran
  with the card time-sliced against the NFS@Home client.
  `final-timing-clean.sh`, `c143-clean.sh` and `c143-more.sh` (`.txt`) are
  the same on a free card.
- **Landed binary:** `fat-check.sh` (`fat-check.txt`) checked the rebuilt fat
  `bench/bench`.

`build.txt` names the binaries. The ncu reports and snapshot builds stay in
the git-ignored `work/resieve-i16/`.

## Threshold scan and bitmap clears (2026-10-16)

[`scan-clears-20261016/`](scan-clears-20261016/) holds the evidence for
[finding 117](../bench/RESULTS.md#finding-117), pinned in
`MANIFEST.sha256`. `change.diff` is the code change.
- **Code shape:** `scan-sass.txt` is the threshold-scan loop of the
  production I16 `k_apply` instance (sm_120) in the base, first and final
  versions. `gates.txt` lists the registers of every instance.
- **Profiles:** `prof.sh` (`prof-base.csv`, `prof-new.csv`,
  `prof-summary.txt`) is the per-launch ncu run on one NFS@Home q. `split.csv`
  and `split-summary.txt` are the per-kernel durations by side.
- **Gates:** `sbgate.sh` (`sbgate.txt`) compares the harness's survivor
  bitmaps, base against new. `ident.sh` (`ident.txt`) checks relation
  identity on four jobs. `gates.txt` covers `--verify-only`, `make check` and
  fillcheck.
- **Timing:** `timing.sh` (`timing.txt`) is the three-round free-card A/B.
  `c183-single.txt` is the single-queued intersect check.
- **Per-side acceptance:** `sidestats.diff` is the test-only counter and
  `sidestats.txt` its four bands. It was never landed.
- **Review fixes (2026-10-17):** `review.diff` is the change.
  `review-gates.sh` (`review-gates.txt`) ran the gates, and `verify-c183.txt`
  is its `--verify` run. `timing-review.txt` is the A/B that a GPU ECM job
  time-sliced, so only its host timers (`host-timers.txt`) are used.
  `timing.sh` gained the `B=` override for it. `timing-retime.sh`
  (`timing-retime.txt`) is the free-card rerun of 2026-10-18 against the
  committed build. Its rows also record each run's exit status and the
  card's utilization just before it.
- **Landed binary:** `fat-check.sh` (`fat-check.txt`) checked the rebuilt fat
  `bench/bench`, after the review fixes.

`build.txt` names the binaries. The snapshot builds and SASS dumps stay in
the git-ignored `work/scan-i16/`.

## Side-0 record scan, measured (2026-10-18)

[`recscan-20261018/`](recscan-20261018/) holds the evidence for
[finding 118](../bench/RESULTS.md#finding-118), pinned in `MANIFEST.sha256`.
`recscan.diff` is the test-only counter build on `9b9f314`. It was never
landed. `run.sh` (`run.txt`) ran it on five jobs single-queued, and on two
of them pipelined. Its final exit-status line was added after the run, in
finding 119's review. `build.txt` names the binary, which stays in the
git-ignored `work/recscan/`.

## Side-0 resieve from its records (2026-10-18)

[`rscan-20261018/`](rscan-20261018/) holds the evidence for
[finding 119](../bench/RESULTS.md#finding-119), pinned in `MANIFEST.sha256`.
`change.diff` is the code change (`bench/abcheck.sh` is in the tree).
- **Gates:** `gates.sh` (`gates.txt`) ran `abcheck.sh` against the committed
  `9b9f314` build: the default arm, mixed record/walk arms, every fallback,
  the refusals, `make check` and fillcheck. `gates-first.txt` is the first
  run, with the knob-refusal artefact described in the finding.
- **Timing:** `timing.sh` (`timing.txt`) is the three-round free-card A/B.
- **Landed binary:** `fat-check.sh` (`fat-check.txt`, `fat-abcheck.txt`)
  checked the fat `bench/bench`, rebuilt after the review.
- **Review:** `growfail.sh` (`growfail.txt`) drove the failed-growth branch
  with the `PIPE_RS_TEST_GROW_FAIL` test build. `gates.sh` and
  `fat-check.sh` gained exit statuses that count every check after their
  recorded runs. Those runs' outputs are unchanged.

`build.txt` names the binaries. The snapshot builds stay in the git-ignored
`work/rscan/`.
