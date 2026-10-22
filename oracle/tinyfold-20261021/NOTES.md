# Tiny-prime tables (STATUS 8b): pre-measurements, 2026-10-21

Host only; the card was running the other window's head-to-head and was not
touched (the device query uses cuInit/cuDeviceGetAttribute, no context).
Source: a copy of the working tree at HEAD 3537cdd plus its uncommitted
finding 125-126 edits (`tree/`, `tree.status`).

## 1. Hit census by effective modulus (`hitcount.c`, `census.sh`, `census.*.txt`)

Builds each side's line-sieved table per real special-q exactly as
`pipe_side_prepare_q` does (fb_split_small, pl_transform_fast -> m, rt, g)
and counts sieve_small's hits under ss_setup's rules (g | j, first hit per
region, the --sieve-skip even-row rule). Checked against the harness's
"updates" estimate on c135 q=7000033 with the skip off: 254.7M / 108.4M
against 2.546e8 / 1.084e8.

T420 = effective m in {1,2,3,4,5,7}; pow2 fold = m in {1,2,4,8,16,32}.

| job | side | block hits/pos | T420 of block hits | T420 of block warp-iters | T420 of block entries | pow2 fold of block hits |
|---|---|---:|---:|---:|---:|---:|
| c135 | 1 | 4.32 | 67.7% | 66.8% | 32.6% | 37.9% |
| c135 | 0 | 1.56 | 61.0% | 59.8% | 26.3% | 39.3% |
| C143 | 1 | 3.98 | 61.6% | 60.3% | 25.8% | 27.4% |
| C143 | 0 | 1.57 | 64.4% | 63.1% | 29.8% | 42.3% |
| c183 | 1 | 5.41 | 61.7% | 61.2% | 30.7% | 35.9% |
| c183 | 0 | 1.56 | 62.7% | 62.1% | 29.5% | 38.7% |
| I16 | 1 | 5.07 | 66.0% | 65.5% | 30.7% | 44.1% |
| I16 | 0 | 1.56 | 58.6% | 58.1% | 25.0% | 34.0% |
| C194 | 1 | 4.58 | 61.7% | 61.3% | 29.4% | 38.5% |
| C194 | 0 | 1.56 | 59.9% | 59.4% | 25.0% | 38.6% |
| AS276 | 1 | 5.44 | 62.0% | 61.6% | 30.2% | 39.7% |
| AS276 | 0 | 1.57 | 66.4% | 65.8% | 32.7% | 45.4% |

(entries column = T420 entries / block entries per q, from `entries/q`.)
Whole-row entries (m = 1, any g) alone are 0.44-0.95 hits/position on side 1.

### From hits to instructions (calibration, not a forecast)

Finding 110's per-line export (`work/applyprof/c183_apply_L0.csv`, c183
side 1, one launch): the block tier's per-entry lines (setup, shuffle loop,
decode) total 765M warp instructions over 48.2M (entry, warp) shuffles, ~16
each including entries with no hit in the row; the hit loop is ~11-12 per
warp-iteration (lines 636-637 plus ss_add's body). Its 95.6M warp-iterations
match the census's 91.8M for c183 side 1. So per-entry overhead is ~40% of the
block tier, and a set holding ~62-67% of block hits but ~30% of its entries
removes roughly half the block tier's instructions. Against finding 110's
launch totals: T420 ~14% of side-1 apply, ~6-7% of side-0 apply, ~10% of
apply before the replacement's own cost, at c183 and I16. Hypothesis to test.

## 2. Headroom (`devattr.txt`, `occ.cpp`/`occ.txt`, `ptxas.log`)

RTX 5070, cc 12.0: 102,400 B shared memory per SM, 1,024 B reserved per block,
128 B allocation granule (NVIDIA's cuda_occupancy.h). Three 512-thread blocks
need dynamic smem + 1,024 rounded up to 128 to be <= 34,133 -> <= 33,024 B.

| job | region | nslice_pow2 (s1/s0) | apply smem/block | spare at 3 blocks |
|---|---|---|---:|---:|
| c135, C143 | 2^13 | 32/32 | 16,448 | ~16.6 KB (threads limit, not smem) |
| c183 | 2^14 | 64/64 | 32,896 | 128 B (one granule) |
| I16, C194, AS276 | 2^14 | 128/64 | 33,024 | **0** |

At I16, 33,024 + 72 B already drops to 2 blocks, and so does an 840 B table in
place of the lookup table (32,768 + 840). The comment above k_apply
(bench_kernels.cu, "allows nslice_pow2 <= 512 ... C194 sits today at 33792")
omits the reserve: the limit at --region 14 is 128 slices, and C194 is at 128.

Registers: the pipeline's k_apply instances use 40 of 40 (launch_bounds 512,3),
0 spill, sm_120. Any design must compile at 40 with no new spill.

## 3. What follows for the design

- Moduli dividing 2*nth (1024 at 512 threads) give each thread the SAME
  contribution word on every step of the init loop (its cells are 2t + 1024k),
  so m in {1,2,4,8,16,32} fold into one per-thread constant: no table, no
  shared memory, no barrier. 27-45% of block-tier hits.
- 3, 5, 7 rotate per step and need a table. Room is free at c135/C143; c183
  has one granule; I16-class needs the bucket lookup table as bytes (logp is
  u8 in fb_t; frees 128 B) or the table inside the lookup table's space
  (+2 barriers: build -> init, lookup load -> bucket apply).
- Apply's existing barriers: after init (line 1029) and after bucket apply
  (1051); the lookup table loads with no barrier before init.

## 4. Omission ceilings (`omit.sh`, `omit.txt`, `omit.summary.txt`; card free, host load ~14)

`omit/` drops entries by effective modulus under TINY_OMIT (wrong output,
timing only); `base/` is the same source unpatched; both sm_120. Single-queued
apply, median of 3 rotated rounds:

| job | base | without pow2 set | without T420 |
|---|---:|---:|---:|
| c135 | 1.151 ms | 1.041 (−9.6%) | 0.980 (−14.9%) |
| c183 | 16.402 | 14.772 (−9.9%) | 13.909 (−15.2%) |
| I16 | 70.086 | 62.890 (−10.3%) | 60.019 (−14.4%) |

Wall is not comparable across these arms: the missing hits also remove
survivors (I16 110,558 -> 69,016 -> 45,106 per q), so TD shrinks too.

## 5. Stage A: --tiny-fold (`fold/`, `fold.build.log`, `foldgate.sh`, `foldgate.txt`)

`ss_fold_front` (host, both callers) moves entries with m a power of two
<= 32 to the front of the sorted table; `ss_fold_word` gives each thread the
word those entries add to every word it initialises (setup by ss_setup,
lane-parallel, shuffled, as the block tier does); k_apply adds it as init
stores each word, and the block tier starts after them. `--no-tiny-fold`
leaves the table and kernel path exactly as before. No shared memory, no
barrier; every k_apply instance still 40 registers (38 for two 8-bit ones),
no spill.

Gates (`foldgate.txt`, 0 failed): build clean; --verify-only 3 OK; apply cell
parity (--verify, one region replayed on the CPU from every small-sieve entry)
0 cells differ in all 8 cases -- both sides, I13/I15/I16, region 2^10, 96 and
160 threads, 8-25 entries folded, and a --no-tiny-fold control; abcheck 36/36
identical fold on and 36/36 with --no-tiny-fold on NEW's pipeline runs;
make check 117 PASS, 0 FAIL.

## 6. Stage A timing (`foldtime.sh`, `foldtime.txt`, `foldtime.summary.txt`)

Median of 3 rotated rounds, host load ~14, relations identical in every arm:
apply single-queued c135 −4.3%, C143 −3.8%, c183 −6.5%, I16 −7.8%, C194
−6.9%; pipelined wall c183 −2.8%, I16 −3.5%, C194 −2.8% (every round
negative on all three), c135 −1.3%, C143 −1.1% (noise).

## 7. Larger jobs: 2,1147+ (`../c1147/`, `cliff.sh`, `cliff.txt`, `rs.*`)

Greg's Cunningham job (SNFS degree 7, alim = rlim = 600M, lpb 36/37, mfb
72/111): 31.3M ideals per side, 138/137 slices (`hitcount --slices-only`), so
both log tables pad to 256 and apply needs 33,280 B per block: two blocks per
SM. Slices = entries/262,144 + ~14-20 log changes on every job measured; the
128 limit is ~30M entries on a side, a bound of ~580M.

`nocut/` (fb_build_slices' 262,144 forced cut off, ~20 slices on any factor
base), 10 q at 300M, median of 3, relations identical:

| | base | nocut | |
|---|---:|---:|---:|
| 2,1147+ apply, single-queued | 73.0 ms | 59.6 | −18.4% |
| 2,1147+ wall, pipelined | 265.4 | 251.5 | −5.2% (−4.6, −5.3, −4.5) |
| I16 apply (control, three blocks either way) | 70.9 | 71.5 | +0.8% |
| I16 wall | 169.3 | 168.2 | −0.6% |

The only production reader of the slice ID found is apply's log lookup
(bucket records' high 16 bits); the side-0 resieve uses its own index, the
harness's layout B its own slicing; fbtest tests only the 16-bit ID limit.

Second limit on this job: the side-0 record index (finding 119) is sized
cap/2 = 5,787 per region and is full in 28 of 40 slabs, which walk instead.
`--resieve-scan-cap 11574` (= cap): 40/40 from records, resieve 29.0 -> 18.6
ms/q, wall 252.1 -> 245.3 (−2.7%), relations identical, but device memory
8.48 -> 10.13 GB of 11.91 pipelined. Wants a better-sized index, not cap.
Memory otherwise: 6.6 GB single-queued, 8.5 GB pipelined.

## 8. The >= 9 GB NFS@Home queue (Kyle: Greg runs NFS@Home; this queue needs >= 9 GB)

Apply's blocks per SM by architecture (`occarch.cpp`, `occarch.txt`, NVIDIA's
cuda_occupancy.h with each architecture's documented limits):

| architecture | 32,896 B | 33,024 B | 33,280 B (2,1147+) |
|---|---:|---:|---:|
| sm_75 Turing (2080 Ti 11 GB) | 1 | 1 | 1 |
| sm_80 A100 | 3 | 3 | 3 |
| sm_86 Ampere, sm_89 Ada, sm_120 Blackwell | 3 | 3 | **2** |

So every consumer Ampere/Ada/Blackwell card in the queue runs 2,1147+'s apply
at two blocks per SM today; Turing is at one block whatever the slices.
Memory: 2,1147+ needs 6.6 GB single-queued, 8.5 GB pipelined; the second q is
granted only when cudaMemGetInfo shows it fits with PIPE_VRAM_MARGIN, else the
band runs single-queued. A 10 GB card keeps the pipeline with ~1 GB spare, so
a fix that adds memory (the full-size side-0 index: +1.65 GB) is ruled out.
