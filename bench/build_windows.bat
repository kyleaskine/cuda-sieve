@echo off
setlocal EnableExtensions

rem Native Windows build for the main CUDA sieve executable and, optionally,
rem the standalone GPU factor-base cache generator.
rem Run from an "x64 Native Tools Command Prompt for VS" with CUDA on PATH.
rem
rem Knobs, matching bench/Makefile so a Windows binary is the same build:
rem   GPU_ARCH  all (default) | native | full | a bare compute capability,
rem             e.g. 86 -- full needs CUDA <= 12.8 on PATH, see :arch_full
rem   CF_LMAX   4 (default, 128-bit cofactors) | 3 (96-bit)
rem   DEFS      extra -D's for pricing experiments, e.g. -DNORM_FAST_LOG2
rem             (dash form: these reach nvcc too, which rejects /D)
rem
rem   set GPU_ARCH=86 ^&^& set CF_LMAX=3 ^&^& build_windows.bat
rem   build_windows.bat fbgen_gpu   rem also build fbgen_gpu.exe
rem
rem Default builds bench.exe only.  The fbgen_gpu argument additionally builds
rem the reusable roots-file utility without making every ordinary build compile
rem fbgen_gpu.cu twice.  "build_windows.bat clean" removes both executables.

if /I "%~1"=="clean" goto :do_clean
set "BUILD_FBGEN_GPU="
if "%~1"=="" goto :arg_done
if /I "%~1"=="fbgen_gpu" (
    set "BUILD_FBGEN_GPU=1"
    goto :arg_done
)
echo error: unknown target "%~1". Use no argument, fbgen_gpu, or clean.
exit /b 1
:arg_done

where cl >nul 2>nul || (
    echo error: cl.exe not found. Run this from an x64 Visual Studio Native Tools prompt.
    exit /b 1
)
where nvcc >nul 2>nul || (
    echo error: nvcc.exe not found. Add the CUDA Toolkit bin directory to PATH.
    exit /b 1
)

rem ---- GPU_ARCH ----------------------------------------------------------
rem Validated rather than pasted straight into the gencode string. An
rem unchecked value produces "compute_sm_86" or "compute_8.6" and an opaque
rem ptxas diagnostic; the Makefile rejects the same three spellings by hand
rem and says why, so this does too.
rem
rem Case-sensitive comparisons, on purpose: the Makefile's `ifeq` has no
rem case-insensitive mode, so GPU_ARCH=Full/ALL/Native would build here but
rem get rejected there. Matching that (rather than the friendlier /I) keeps
rem one spelling giving one answer on both platforms.
if not defined GPU_ARCH set "GPU_ARCH=all"

rem Shared SASS list and PTX-fallback suffix, set unconditionally and
rem BEFORE the dispatch below (a plain string assignment, no process spawn)
rem so :arch_full composes from the same six real targets :arch_all uses
rem instead of re-spelling them -- one string to update when a target joins
rem or leaves the fat binary, not two. Must come before the `goto`s: each
rem one jumps straight past anything below it, so setting these after the
rem dispatch would leave them undefined by the time :arch_all/:arch_full
rem read them -- caught by testing this change against real cmd.exe.
set "NVCC_ARCH_ALL_SASS=-gencode arch=compute_120,code=sm_120 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75"
set "NVCC_ARCH_PTX_FALLBACK=-gencode arch=compute_80,code=compute_80"

if "%GPU_ARCH%"=="all" goto :arch_all
if "%GPU_ARCH%"=="native" goto :arch_native
if "%GPU_ARCH%"=="full" goto :arch_full
echo %GPU_ARCH%| findstr /r /c:"^[0-9][0-9]*$" >nul || goto :arch_bad
set "NVCC_ARCH=-gencode arch=compute_%GPU_ARCH%,code=sm_%GPU_ARCH%"
goto :arch_done

:arch_bad
echo error: GPU_ARCH must be all, native, full, or a bare compute capability
echo        like 86 or 120 -- got "%GPU_ARCH%". Not sm_86, not 8.6.
exit /b 1

:arch_all
rem Same fat-binary set as the Makefile's default, including the compute_80
rem PTX floor so a newer card than this list still runs. sm_75 joined this
rem list 2026-09-09: it is QUALIFIED on real RTX 2080 Ti hardware
rem (STATUS.md item 15) and still compiles under this project's CUDA
rem 13.2/13.3 canonical toolchain. sm_50/52/60/61/70 do NOT -- CUDA 13 drops
rem them outright -- so they live only in :arch_full below, not here.
set "NVCC_ARCH=%NVCC_ARCH_ALL_SASS% %NVCC_ARCH_PTX_FALLBACK%"
goto :arch_done

:arch_full
rem sm_50-through-sm_120 fat binary -- the Makefile's GPU_ARCH_full twin.
rem Every target in :arch_all plus sm_50/52/60/61/70. Needs CUDA <= 12.8:
rem CUDA 13 hard-rejects sm_50/52/60/61/70 with "Unsupported gpu
rem architecture", confirmed 2026-09-09 against real nvcc 13.0; nvcc
rem 12.8.93 accepts all of them (deprecation warning only). Only sm_61 and
rem sm_75 are hardware-qualified (STATUS.md item 15) -- sm_50/52/60/70 are
rem untested, included because nvcc accepts them, not because a card
rem confirmed them.
rem
rem The version check is inlined here rather than a separate `call`ed
rem subroutine: a `call`ed `exit /b` only returns to the caller with an
rem errorlevel set, it does not stop the calling script by itself -- that
rem exact gotcha cost a real bug during development (this error printed,
rem then the build continued anyway). A plain `goto` to an error label
rem below, matching :arch_bad/:native_bad's own convention, has no such
rem trap and needs no caller-side errorlevel check.
set "NVCC_VER_LINE="
for /f "tokens=*" %%L in ('nvcc --version ^| findstr /c:"release"') do if not defined NVCC_VER_LINE set "NVCC_VER_LINE=%%L"
set "NVCC_MAJOR="
if not defined NVCC_VER_LINE goto :arch_full_ver_bad
rem Search for the word "release", not a fixed field position: findstr
rem confirms it is present, then %VAR:*release =% strips everything up to
rem and including it, wherever it falls in the line. A previous version of
rem this used a fixed "tokens=2 delims=," position, which would silently
rem break if nvcc's banner ever gains or loses a field before "release" --
rem this is the same "search for the keyword" approach the Makefile's own
rem `grep -oE 'release [0-9]+'` already uses, not a fixed-position guess.
set "NVCC_REST=%NVCC_VER_LINE:*release =%"
for /f "tokens=1 delims=, " %%V in ("%NVCC_REST%") do set "NVCC_VER=%%V"
for /f "tokens=1 delims=." %%A in ("%NVCC_VER%") do set "NVCC_MAJOR=%%A"
echo %NVCC_MAJOR%| findstr /r /c:"^[0-9][0-9]*$" >nul || goto :arch_full_ver_bad
if %NVCC_MAJOR% GEQ 13 goto :arch_full_too_new
set "NVCC_ARCH=%NVCC_ARCH_ALL_SASS% -gencode arch=compute_70,code=sm_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_50,code=sm_50 %NVCC_ARCH_PTX_FALLBACK%"
goto :arch_done

:arch_full_ver_bad
echo error: GPU_ARCH=full: could not read nvcc's version (got "%NVCC_MAJOR%" from `nvcc --version`^).
exit /b 1

:arch_full_too_new
echo error: GPU_ARCH=full needs CUDA ^<= 12.8 -- nvcc reports release %NVCC_MAJOR%.x,
echo        and CUDA 13 dropped Maxwell/Pascal/Volta outright ^(verified
echo        2026-09-09: nvcc 13.0 refuses sm_50/52/60/61/70; nvcc 12.8.93
echo        accepts all of them^). Put a CUDA ^<= 12.8 Toolkit's bin directory
echo        on PATH to build this target.
exit /b 1

:arch_native
set "GPU_CC="
for /f "usebackq delims=" %%G in (`nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul`) do if not defined GPU_CC set "GPU_CC=%%G"
if not defined GPU_CC goto :native_bad
rem "8.6" -> "86"
set "GPU_CC=%GPU_CC:.=%"
set "GPU_CC=%GPU_CC: =%"
echo %GPU_CC%| findstr /r /c:"^[0-9][0-9]*$" >nul || goto :native_bad
set "NVCC_ARCH=-gencode arch=compute_%GPU_CC%,code=sm_%GPU_CC%"
goto :arch_done

:native_bad
echo error: GPU_ARCH=native: no usable compute capability from nvidia-smi.
echo        Pass GPU_ARCH=^<cc^>, e.g. GPU_ARCH=86, or GPU_ARCH=all.
exit /b 1

:arch_done

rem ---- CF_LMAX -----------------------------------------------------------
rem A build-shape knob, not a pricing define: it changes which splitter widths
rem exist, and a CF_LMAX=3 binary is shippable. See bench/Makefile.
if not defined CF_LMAX set "CF_LMAX=4"
if "%CF_LMAX%"=="3" goto :cflmax_ok
if "%CF_LMAX%"=="4" goto :cflmax_ok
echo error: CF_LMAX must be 3 (96-bit cofactors) or 4 (128-bit) -- got "%CF_LMAX%".
exit /b 1
:cflmax_ok
rem Dash form deliberately: cl.exe accepts -D as happily as /D, and nvcc
rem accepts only -D. One spelling that works for both keeps the CUDA and host
rem compiles from drifting apart on the one define that changes build shape.
set "CF_LMAX_DEF=-DCF_LMAX=%CF_LMAX%"

rem ---- build stamp -------------------------------------------------------
rem What makes a run log traceable to a source tree, and what keeps a pricing
rem build's relations distinguishable from production output. Without these
rem two defines runlog.c falls back to "unknown", and every Windows run log
rem claims an unknown commit whether or not that is true.
set "GIT_RAW="
for /f "usebackq delims=" %%G in (`git -C . describe --always --abbrev^=8 2^>nul`) do if not defined GIT_RAW set "GIT_RAW=%%G"
if not defined GIT_RAW goto :stamp_unknown
set "GIT_DIRT="
for /f "usebackq delims=" %%G in (`git -C . status --porcelain 2^>nul`) do set "GIT_DIRT=1"
if defined GIT_DIRT set "GIT_RAW=%GIT_RAW%-dirty"
set "GIT_DESC=%GIT_RAW%"
goto :stamp_done
:stamp_unknown
set "GIT_DESC=unknown"
:stamp_done

rem ---- flags -------------------------------------------------------------
rem /MT, not the default /MD: bench/Makefile carries its C++ and GCC support
rem runtimes into distribution binaries for the same reason. A /MD bench.exe
rem hard-depends on VCRUNTIME140.dll, and a BOINC volunteer without the
rem matching Visual C++ redistributable fails the task with a loader error
rem before main() runs.
set "CFLAGS=/nologo /O2 /W3 /MT -D_CRT_SECURE_NO_WARNINGS %CF_LMAX_DEF% %DEFS%"
set "CXXFLAGS=/nologo /O2 /W3 /MT /EHsc -D_CRT_SECURE_NO_WARNINGS %CF_LMAX_DEF% %DEFS%"

echo Building host C objects with cl.exe... (GPU_ARCH=%GPU_ARCH% CF_LMAX=%CF_LMAX% build=%GIT_DESC%)
for %%F in (fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c platform.c) do (
    cl %CFLAGS% /std:c11 /c %%F || exit /b 1
)

rem runlog.c alone carries the build stamp, exactly as in the Makefile.
cl %CFLAGS% /std:c11 -DBENCH_GIT_DESC=\"%GIT_DESC%\" -DBENCH_DEFS=\"%DEFS%\" /c runlog.c || exit /b 1

rem The main executable needs only fbgen's streaming special-q generator.
rem FBGEN_LIBRARY excludes the standalone pthread/open_memstream writer.
cl %CFLAGS% /std:c11 -DFBGEN_LIBRARY /c fbgen.c /Fofbgen_lib.obj || exit /b 1
cl %CXXFLAGS% /std:c++17 /c boinc_support.cpp || exit /b 1

echo Building CUDA objects with cl.exe as nvcc's host compiler...
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c bench_main.cu -o bench_main.obj || exit /b 1
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c bench_kernels.cu -o bench_kernels.obj || exit /b 1

rem Production bench needs the same in-process algebraic factor-base generator
rem as the Makefile build.  FBGEN_GPU_LIBRARY removes the standalone CLI, timers
rem and text writer, so omitting --fb1 has no extra serialization path.
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -DFBGEN_GPU_LIBRARY ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c fbgen_gpu.cu -o fbgen_gpu_lib.obj || exit /b 1

echo Linking bench.exe...
nvcc %NVCC_ARCH% --cudart static -ccbin cl -Xlinker "/OPT:REF" -o bench.exe ^
    bench_main.obj bench_kernels.obj fbgen_gpu_lib.obj fb_load.obj verify_cpu.obj poly.obj ^
    primes.obj rfb.obj fb_cado.obj platform.obj runlog.obj fbgen_lib.obj ^
    boinc_support.obj || exit /b 1

echo Built %CD%\bench.exe (GPU_ARCH=%GPU_ARCH% CF_LMAX=%CF_LMAX% build=%GIT_DESC%)

if not defined BUILD_FBGEN_GPU exit /b 0

echo Building standalone fbgen_gpu.exe...
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c fbgen_gpu.cu -o fbgen_gpu.obj || exit /b 1
nvcc %NVCC_ARCH% --cudart static -ccbin cl -Xlinker "/OPT:REF" -o fbgen_gpu.exe ^
    fbgen_gpu.obj fbgen_lib.obj fb_load.obj fb_cado.obj poly.obj primes.obj platform.obj || exit /b 1
echo Built %CD%\fbgen_gpu.exe
exit /b 0

:do_clean
del /q *.obj bench.exe fbgen_gpu.exe 2>nul
echo Cleaned Windows build products in %CD%
exit /b 0
