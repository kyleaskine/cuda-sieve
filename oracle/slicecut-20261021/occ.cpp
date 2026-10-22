// Apply's blocks per SM by shared memory, from NVIDIA's host-only occupancy
// calculator (cuda_occupancy.h) with this card's attributes (devattr.txt):
// cc 12.0, 102400 B per SM, 1024 B reserved per block, 1536 threads per SM.
#include <cuda_occupancy.h>
#include <stdio.h>
int main()
{
    cudaOccDeviceProp p;
    p.computeMajor = 12; p.computeMinor = 0;
    p.maxThreadsPerBlock = 1024; p.maxThreadsPerMultiprocessor = 1536;
    p.regsPerBlock = 65536; p.regsPerMultiprocessor = 65536; p.warpSize = 32;
    p.sharedMemPerBlock = 49152; p.sharedMemPerMultiprocessor = 102400;
    p.numSms = 48; p.sharedMemPerBlockOptin = 101376; p.reservedSharedMemPerBlock = 1024;
    cudaOccFuncAttributes f;
    f.maxThreadsPerBlock = 512; f.numRegs = 40; f.sharedSizeBytes = 0;
    f.partitionedGCConfig = PARTITIONED_GC_OFF; f.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
    f.maxDynamicSharedSizeBytes = 101376; f.numBlockBarriers = 1;
    cudaOccDeviceState s; s.cacheConfig = CACHE_PREFER_SHARED; s.carveoutConfig = SHAREDMEM_CARVEOUT_MAX_SHARED;
    const int sz[] = { 16448, 32896, 33024, 33024 + 85, 33024 + 86, 33024 + 72, 33024 + 124, 33024 + 840, 33792, 32768 + 840 };
    for (unsigned i = 0; i < sizeof sz / sizeof sz[0]; i++) {
        cudaOccResult r;
        cudaOccMaxActiveBlocksPerMultiprocessor(&r, &p, &f, &s, 512, sz[i]);
        printf("dynamic smem %6d B -> %d blocks/SM (limited by 0x%x)\n", sz[i], r.activeBlocksPerMultiprocessor, r.limitingFactors);
    }
    return 0;
}
