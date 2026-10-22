// Apply's blocks per SM on the architectures a >= 9 GB NFS@Home queue sees,
// from NVIDIA's cuda_occupancy.h with each architecture's documented limits
// (shared memory per SM, reserved per block, opt-in per block, threads per SM).
#include <cuda_occupancy.h>
#include <stdio.h>
struct A { const char *name; int maj, min, smem_sm, rsv, optin, thr_sm; };
int main()
{
    const A a[] = {
        { "sm_75 Turing (2080 Ti 11 GB)", 7, 5, 65536, 0, 65536, 1024 },
        { "sm_80 A100", 8, 0, 167936, 1024, 166912, 2048 },
        { "sm_86 Ampere (3060 12, 3080 10/12, 3090)", 8, 6, 102400, 1024, 101376, 1536 },
        { "sm_89 Ada (4060 Ti 16, 4070-4090)", 8, 9, 102400, 1024, 101376, 1536 },
        { "sm_120 Blackwell (5070-5090)", 12, 0, 102400, 1024, 101376, 1536 },
    };
    const int sz[] = { 32896, 33024, 33280, 16448 };
    for (unsigned i = 0; i < sizeof a / sizeof a[0]; i++) {
        cudaOccDeviceProp p;
        p.computeMajor = a[i].maj; p.computeMinor = a[i].min;
        p.maxThreadsPerBlock = 1024; p.maxThreadsPerMultiprocessor = a[i].thr_sm;
        p.regsPerBlock = 65536; p.regsPerMultiprocessor = 65536; p.warpSize = 32;
        p.sharedMemPerBlock = 49152; p.sharedMemPerMultiprocessor = a[i].smem_sm;
        p.numSms = 48; p.sharedMemPerBlockOptin = a[i].optin; p.reservedSharedMemPerBlock = a[i].rsv;
        cudaOccFuncAttributes f;
        f.maxThreadsPerBlock = 512; f.numRegs = 40; f.sharedSizeBytes = 0;
        f.partitionedGCConfig = PARTITIONED_GC_OFF; f.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
        f.maxDynamicSharedSizeBytes = a[i].optin; f.numBlockBarriers = 1;
        cudaOccDeviceState s; s.cacheConfig = CACHE_PREFER_SHARED; s.carveoutConfig = SHAREDMEM_CARVEOUT_MAX_SHARED;
        printf("%-42s", a[i].name);
        for (unsigned k = 0; k < sizeof sz / sizeof sz[0]; k++) {
            cudaOccResult r;
            cudaOccMaxActiveBlocksPerMultiprocessor(&r, &p, &f, &s, 512, sz[k]);
            printf("  %5d B: %d", sz[k], r.activeBlocksPerMultiprocessor);
        }
        printf("\n");
    }
    return 0;
}
