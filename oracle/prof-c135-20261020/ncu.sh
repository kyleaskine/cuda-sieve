#!/bin/sh
# c135, isolated kernel profiles (ncu serialises launches): duration, SM and
# memory throughput, occupancy and launch shape for each siever's main
# kernels, a few launches each after warm-up. Ours from q 7000000 (bench/bench);
# cado-gpu from the same q as its client ran it. The question: does
# cado-gpu's 1.8 ms per special-q come from lighter kernels, or from running
# several special-q's kernels at once (its trace: 2.5x summed/busy)?
R=/home/kylea/code/cuda-sieve; W=$R/work/prof-c135
D=/home/kylea/cado-gpu/c136-934760359018/client/download
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
SEC="--section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis"
cd $W || exit 1
fail=0
ncu --target-processes all $SEC \
    -k regex:'^k_(fill_atomic|fill_rows|apply|transform|resieve_scatter|resieve_rows|td|classify|td_record_warp)$' \
    --launch-skip 150 --launch-count 45 -f -o $W/ours-ncu \
    $R/bench/bench --pipeline --cofactor --poly $R/work/c135/c135.job --logI 13 --J 4096 \
    --region 13 --maxbits 13 --sq-side 1 --lambda0 1.89 --lambda1 1.94 --qrange 7000000: --nq 40 \
    > $W/ours-ncu.out 2>&1 || fail=$((fail + 1))
echo "ours ncu: $(tail -1 $W/ours-ncu.out)"
ncu --target-processes all $SEC \
    -k regex:'^(fill_ahead|region|resieve_small|filter|filter_prime|fill_row|transform|tiny_mask)_kernel$' \
    --launch-skip 120 --launch-count 40 -f -o $W/cado-ncu \
    $L -poly $D/c135.poly -q0 7000000 -I 13 -q1 7002000 \
    -lim0 9000000 -lim1 12000000 -lpb0 29 -lpb1 30 -mfb0 55 -mfb1 58 -lambda0 1.89 -lambda1 1.94 \
    -ncurves0 12 -ncurves1 14 -fb1 $D/c135.roots1.gz -out $W/cado-ncu.rel.gz -t 16 -stats-stderr \
    > $W/cado-ncu.out 2>&1 || fail=$((fail + 1))
echo "cado ncu: $(tail -1 $W/cado-ncu.out)"
for f in ours-ncu cado-ncu; do
  ncu --import $W/$f.ncu-repz --csv --page raw \
      --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__waves_per_multiprocessor,lts__t_sector_hit_rate.pct \
      > $W/$f.csv 2>&1 || fail=$((fail + 1))
done
echo "failures: $fail"
[ $fail -eq 0 ]
