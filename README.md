# cuda-sieve

`cuda-sieve` is an experimental, standalone CUDA implementation of the
lattice-sieving relation-collection pipeline used by the Number Field Sieve.
It builds factor bases, sieves both sides of a special-q lattice, performs
trial division and GPU cofactorisation, and emits relations for msieve.

The implementation has been exercised on NVIDIA Ampere, Ada, and Blackwell
GPUs. It is research software: the correctness gates pass, but the current
limitations and unfinished experiments in [`bench/STATUS.md`](bench/STATUS.md)
are part of its release status.

The production path has a hard local-slab safety ceiling of `2^31` positions.
For performance, automatic planning now starts j-slabbing once the full sieve
reaches `2^30` positions and targets at most `2^29` positions per slab. Sieve
areas below `2^30` are not split for performance alone, so `2^29` and smaller
retain the ordinary unslabbed path. With default `J=2^(logI-1)`, this means
I15 -> 1 slab, I16 -> 4, I17 -> 16, I18 -> 64, I19 -> 256, and I20 -> 1024.
`--slab-j` remains an explicit tuning override subject to the safety bounds.
The `2^29` target is a robust performance/memory default, not a universal
speed optimum: an L40 benchmark preferred `2^30` positions per slab by 4.6%
in complete time, while the `2^29` default still beat the former `2^31`
behavior by 1.5% and reduced steady VRAM from 7.76 GB to 3.20 GB. The quantity
that target really tunes is a bucket-region count rather than an area, so it
moves with `--region`; a per-card automatic policy is still open work.
There is no cap on the total pipeline area: `--pipeline` slabs any geometry
through `logI 20`, and `2^32` has been sieved on a 5070 and an L40.
`lpb <= 64`,
`mfb <= 128`, at most three large primes per side, and an exact norm within
this build's `BN_LIMBS * 32` bits (384 by default) remain checked limits.
Cofactor arithmetic width and
factorisation method are both chosen **per side, automatically**, from that
side's `mfb` and `lpb`: 3 limbs (96-bit residuals) or 4 (128-bit), and
Pollard-Brent rho for a two-large-prime side or ECM for a three-large-prime
one. The separate representation, memory,
cofactor-performance, and filtering implications of lifting them are laid out
in [Current size limits and j-slabbing](bench/STATUS.md#current-size-limits-and-j-slabbing).

## Build and test

You need a Linux build environment, GNU Make, a C11 compiler, a C++17
compiler, and an NVIDIA CUDA toolkit. The supplied Makefile targets compute
capability 8.0 and newer.

```sh
make -C bench

# Generate the large factor base used by the GPU golden tests. It is ignored
# by Git and can be regenerated at any time.
./bench/fbgen --poly oracle/c183.poly --lim 134200000 \
    --maxbits 15 --threads 6 --out oracle/c183.fb1

# Same deal for the C147, which is the workload behind the device timings in
# bench/RESULTS.md. --maxbits must match the --logI you sieve at; 14 is what
# the frozen file uses, and --lim defaults to alim from the .job file.
./bench/fbgen --poly oracle/c147.job --maxbits 14 \
    --threads 6 --out oracle/c147.roots1

make -C bench check
```

Both files are pinned in [`oracle/MANIFEST.sha256`](oracle/MANIFEST.sha256);
`cd oracle && sha256sum -c MANIFEST.sha256` confirms a regeneration matched.

The default build produces a portable binary covering compute capability 8.0
through 12.0, which takes a few minutes because `ptxas` is slow on the newest
target. `make -C bench GPU_ARCH=native` builds only for the card in this machine
— much faster on an Ampere or Ada host, and the resulting binary will not run
anywhere else.

`make -C bench CF_LMAX=3` builds a narrower binary carrying only the 3-limb
cofactor kernels: `mfb` is then capped at 96 rather than 128, the device code
halves (8.99 MB to 4.02 MB), and `ptxas` has roughly half as much to do. Its
relations are byte-identical to a full build's on any job both can run, so it
is a shippable variant rather than a debug one.

For a real job, the complete algebraic factor base is generated in-process
on the assigned GPU when `--fb1` is omitted.  Rare prime-power, ramified and
projective branches are completed by the same exact CPU Hensel code used by
`fbgen`:

```sh
./bench/bench --pipeline --cofactor --poly JOB.job \
    --logI 14 --qrange 15000000: --target-rels 65000000 \
    --relations msieve.dat
```

A pre-generated native factor base remains supported by passing
`--fb1 JOB.roots1`; this is useful for A/B validation or campaigns that reuse
one polynomial often enough to avoid paying the startup generation cost on
every band.  The GPU generator can create that cache directly in the exact
native `fbgen` format:

```sh
make -C bench GPU_ARCH=native fbgen_gpu
./bench/fbgen_gpu --poly JOB.job --lim ALIM --maxbits LOGI --out JOB.roots1
```

The standalone writer uses a staging `.part` file and atomic rename.  Its
correctness gate requires byte-identical output against CPU `fbgen`; the normal
in-process `bench` path still bypasses text serialization entirely.

The job's `rlim`, `alim`, large-prime bounds, cofactor bounds, and polynomial
are read from the `.job` file. Adding `--log msieve.runlog` appends a run
record: a header naming the commit, argv, job fingerprint, card, geometry and
factor-base convention, then a timestamped line every five minutes carrying
progress alongside GPU-accounted/wall, GPU utilisation, board watts and host
load. See [`RUNBOOK.md`](RUNBOOK.md) before starting a long run; it documents
supported inputs, output handling, parameter precedence, stopping and resuming,
logging, verification, and msieve handoff.

## Optional BOINC application build

BOINC integration is compiled only when `HAVE_BOINC=1`. The normal build has no
BOINC header or library dependency; its wrappers compile to no-ops and the
recurring per-q progress path is omitted. With distribution-provided BOINC
development files, a typical build is:

```sh
make -C bench bench HAVE_BOINC=1 GPU_ARCH=all \
    BOINC_CPPFLAGS="-I/usr/include/boinc"
```

When building against a BOINC source tree instead, point the compiler and
linker at that tree explicitly:

```sh
make -C bench bench HAVE_BOINC=1 GPU_ARCH=all \
    BOINC_CPPFLAGS="-I/path/to/boinc/api -I/path/to/boinc/lib" \
    BOINC_LIBS="-L/path/to/boinc/api -L/path/to/boinc/lib -lboinc_api -lboinc -lpthread"
```

Every build links the CUDA runtime statically, not just this one: pass
`CUDART_LINK=shared` to opt out, at the cost of a binary that will not start
unless the toolkit's library directory is on the loader path. In addition, this
path passes `-static-libstdc++ -static-libgcc` to the host linker by default —
those apply only under `HAVE_BOINC=1`, so a non-BOINC build with
`CUDART_LINK=static` still carries a dependency on the build host's libstdc++.
Override `BOINC_HOST_STATIC` only when a toolchain cannot provide those static
runtime archives. This does not by itself make a completely static Linux
executable:
BOINC libraries and system libraries follow the archives and linker policy
provided by the build environment.

The BOINC build:

- initialises and finishes through the BOINC API;
- requests normal host-thread priority for the CUDA feeder thread;
- runs on the GPU the client assigned to the task, read from `init_data.xml`
  (`<gpu_device_num>`) through `boinc_get_init_data()`. This is what keeps
  concurrent tasks on a multi-GPU host off a single card, and it is the only
  non-deprecated way to learn the assignment: the client also appends
  `--device N` to the command line, but only while the **app version** declares
  an `<api_version>` below 7.5, so that argument disappears the moment a
  project declares a modern one. The assignment therefore outranks `--device`,
  which selects the card only when there is no assignment — standalone runs,
  and app versions the client treats as CPU-only. Each task's stderr records
  which of the two happened and which card was used;
- resolves every explicitly supplied input or output filename through
  `boinc_resolve_filename_s()` and follows native BOINC output links before
  staging and renaming result files; and
- detects whether it is actually managed by a BOINC core client (as opposed to
  a BOINC build launched standalone) and, only in managed mode, discards
  unusable staging/checkpoint artifacts so a volunteer task recomputes instead
  of requiring manual repair. A persistent counter limits this to three
  recoveries per workunit; and
- reports a nondecreasing fraction done at special-q boundaries, normally no
  more than once per second, with an immediate end-of-band update. The
  denominator is, in priority order, `--target-rels`, `--nq`, a bounded
  generated q range, or the length of `--qlist`.

The sieve reserves the final one percent for final cofactor flushing, output
close/rename, and cleanup. A successful BOINC workunit reports 100 percent just
before `boinc_finish(0)`.

A BOINC workunit command line should therefore name all files using the logical
names declared in its workunit template, for example:

```text
--pipeline --cofactor --poly job_file --logI 14 \
--qrange 15000000:16000000 --relations result_file
```

Do not put `--device` in a workunit or app version command line: it is shared
by every task on every host, so it cannot express a per-task assignment. A
`--device` that disagrees with the client is ignored, with a line in stderr.

If a host's tasks all report the same card, read the task's stderr first:

- `BOINC: client assigned CUDA device N` — the assignment arrived. Compare `N`
  across concurrent tasks on that host.
- `BOINC: no GPU assignment in init_data.xml` — the client sent none, so it
  does not consider this a GPU app version. That is a project-side plan class
  or `<coproc>` declaration to fix; no application change can work around it.

This integration reports progress but does not use BOINC's checkpoint API.
`bench_boinc_fraction_done` (`boinc_support.cpp:276`) calls `boinc_fraction_done`
on a ~1 s cadence, so the client's progress bar is live; what is never called is
`boinc_time_to_checkpoint`/`boinc_checkpoint_completed`, so BOINC's *scheduler*
still treats a restart as starting over. The application's **own** sidecar checkpointing is
live and independent of that: a task that exits and is restarted recovers from
the `.part` and its sidecar rather than re-sieving the band, subject to the
staging-identity gate described under "Relations, candidate files and
checkpoints" below. On Windows, note that a `TerminateProcess` stop cannot
write one on the way out — use `--stop-file` for a clean stop there.

## Repository map

- [`bench/`](bench/) contains the implementation, build, and test programs.
- [`bench/testsieve.sh`](bench/testsieve.sh) test-sieves a job at several q,
  normalizes each yield to the expected special-q-pair count, and projects the
  whole run — relations, days, energy, and where a relation target is met —
  across geometries and factor-base bounds. Its `--qmin`, `--qmax` and
  `--target-rels` are given in millions (`--qmin 20` is q = 2e7); its managed
  factor-base cache is rebuilt on the GPU when `fbgen_gpu` is built. Run it
  without arguments for interactive setup.
- [`bench/STATUS.md`](bench/STATUS.md) is the current architecture, validation,
  known-defect, and open-experiment summary.
- [`bench/RESULTS.md`](bench/RESULTS.md) is the chronological benchmark lab
  notebook, including superseded and refuted findings.
- [`oracle/`](oracle/) contains the small, checked-in CADO/GGNFS parity data and
  a manifest for the larger reproducible artifacts that Git ignores.
- [`prototype.md`](prototype.md) records the design and review history.

## License

Original material in this repository is dedicated to the public domain under
[CC0 1.0 Universal](LICENSE), including its as-is warranty and liability
disclaimers. A few files derived from CADO-NFS remain under
LGPL-2.1-or-later; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### Native Windows build (MSVC)

The production executable can also be built natively on Windows with Visual
Studio `cl.exe` as nvcc's host compiler. From an **x64 Native Tools Command
Prompt for Visual Studio** with the CUDA Toolkit on `PATH`:

```bat
cd bench
set GPU_ARCH=120
build_windows.bat
```

Leave `GPU_ARCH` unset for the same portable fat-binary targets as the Linux
Makefile, or set it to `native` to target the card in this machine, or to a
bare compute capability such as `86`, `89`, or `120`. It is validated the same
way the Makefile validates it: `sm_86` and `8.6` are rejected with an
explanation rather than passed through to an opaque ptxas error.

`CF_LMAX` (`3` or `4`, default `4`) and `DEFS` work as they do in the
Makefile, and the build stamp is set the same way, so a Windows run log names
the commit it was built from rather than `unknown`. Objects are compiled
`/MT`, so `bench.exe` does not need the Visual C++ redistributable on a
volunteer host. The Windows `bench.exe` includes the same in-process GPU
algebraic factor-base generator as the Linux build, so omitting `--fb1` has the
same semantics on both platforms.

The default script target builds `bench.exe`. To also build the standalone GPU
cache generator, which can pay factor-base startup once and write a reusable
byte-identical roots file:

```bat
build_windows.bat fbgen_gpu
fbgen_gpu.exe --poly JOB.job --lim 600000000 --maxbits 16 --out JOB.roots1
bench.exe --pipeline --cofactor --poly JOB.job --fb1 JOB.roots1 ...
```

This extra target is opt-in because it recompiles `fbgen_gpu.cu` with the
standalone CLI/text writer enabled; an ordinary production build needs only the
library form already linked into `bench.exe`. `build_windows.bat clean` removes
the objects plus both possible executables. The remaining helper/test tools
still use the Makefile.

Relations, candidate files and checkpoints are written in binary mode on both
platforms, so a Windows build produces byte-identical relation output to a
Linux one — the same bytes to hand to msieve or CADO, not a CRLF variant of
them. Note that the checkpoint additionally records the staging file's
identity (inode on POSIX, volume plus file index on Windows) to gate automatic
recovery, and that identity is meaningful only on the host that wrote it;
moving a `.part` and its sidecar between a Linux and a Windows machine
mid-band is not a supported resume.

One platform difference is worth knowing: on Windows a task stopped with
`TerminateProcess` — which is how the BOINC client and most job queues stop
one — runs no handler at all, so it cannot checkpoint on the way out. Use
`--stop-file` to ask for a clean, checkpointed stop there. Ctrl-C, Ctrl-Break
and closing the console window are handled.
