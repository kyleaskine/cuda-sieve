/* Device limits that decide apply's blocks per SM, read through the driver API
 * with cuInit and cuDeviceGetAttribute only: no context is created and nothing
 * runs on the card, so it is safe beside another process's timing run. */
#include <cuda.h>
#include <stdio.h>
int main(void)
{
    CUdevice d; int v;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&d, 0) != CUDA_SUCCESS) return 1;
    struct { CUdevice_attribute a; const char *n; } A[] = {
        { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "cc major" },
        { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "cc minor" },
        { CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "max smem per SM (B)" },
        { CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "max smem per block, opt-in (B)" },
        { CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, "reserved smem per block (B)" },
        { CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "registers per SM" },
        { CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "max threads per SM" },
        { CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, "max blocks per SM" },
        { CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "SMs" },
    };
    for (unsigned i = 0; i < sizeof A / sizeof A[0]; i++)
        if (cuDeviceGetAttribute(&v, A[i].a, d) == CUDA_SUCCESS) printf("%-34s %d\n", A[i].n, v);
    return 0;
}
