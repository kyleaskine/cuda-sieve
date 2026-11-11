/* xftest -- the device half of the transform's fast-path gate (finding 155).
 *
 * pl_transform_fast_try (plattice.cuh) is one source for host and device, but
 * two of its primitives are spelled per platform: PL_MULHI32 (__umulhi on the
 * device, pl_redc's high word) and PL_CLZ32 (__clz, the exponent's top
 * window). fbtest's verify_transform_fast and every other CPU gate compile the
 * host spelling only. And a device-only fault is SILENT in a run: a wrong
 * inverse fails the fast path's own self-check, every entry falls through to
 * pl_transform_enc, and the relations come out byte-identical, only slower.
 * So this runs the device's code against the host's:
 *
 *   1. PL_MULHI32 and pl_redc on structured words: {0, 1, 2, 3, 2^31 - 1,
 *      2^31, 2^31 + 1, ~1, ~0} and random, against odd moduli at both ends
 *      (3, 2^31 +- 1, 2^32 - 1, 2^32 - 3) and random, T at 0, 1, q*2^32 - 1
 *      and random. A signed __mulhi fails here at once.
 *   2. pl_transform_fast_try's (taken, rt): exhaustive over odd q < 256, every
 *      root in [0, 2q) and every d (a0 in [1, q], a1 = 0); random odd moduli of
 *      every bit length 2..32 with coefficients inside, at and past the 2^32
 *      guard; the largest odd moduli. The fast path must be taken on some
 *      inputs and refused on others, or the comparison proves nothing.
 *
 * work/twin/eqv compares the host path with the bit-at-a-time version it
 * replaced (92M inputs), so host == old and device == host close the chain.
 * Rerun work/twin/xfmutants.py after changing either primitive. Exit 0 only
 * if every check ran and matched. Needs a CUDA device.
 *
 *   make xfcheck         (builds xftest, runs it)
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include "plattice.cuh"

static int fails, checks;
static void check(const char *what, long bad, long n)
{
    checks++;
    if (bad || n == 0) fails++;
    printf("  %-4s %-44s %ld of %ld differ\n", (bad || n == 0) ? "FAIL" : "ok", what, bad, n);
}
static int cuda_ok(const char *what)
{
    const cudaError_t e = cudaDeviceSynchronize();
    const cudaError_t l = cudaGetLastError();
    if (e == cudaSuccess && l == cudaSuccess) return 1;
    printf("  FAIL %s: %s\n", what, cudaGetErrorString(e != cudaSuccess ? e : l));
    fails++; checks++;
    return 0;
}

static uint64_t s = 0x243f6a8885a308d3ull;
static uint64_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

/* ---- 1. the primitives ------------------------------------------------- */

struct pc { uint32_t a, b, q, qinv; uint64_t T; };
__global__ void k_prim(const pc *in, uint32_t *hi, uint32_t *rd, long n)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    hi[i] = PL_MULHI32(in[i].a, in[i].b);
    rd[i] = pl_redc(in[i].T, in[i].q, in[i].qinv);
}
static void run_prim()
{
    static const uint32_t W[] = { 0u, 1u, 2u, 3u, 0x7FFFFFFFu, 0x80000000u, 0x80000001u,
                                  0xFFFFFFFEu, 0xFFFFFFFFu };
    static const uint32_t Q[] = { 3u, 0x7FFFFFFFu, 0x80000001u, 0xFFFFFFFFu, 0xFFFFFFFDu };
    const int NW = (int)(sizeof W / sizeof W[0]), NQ = (int)(sizeof Q / sizeof Q[0]);
    std::vector<pc> v;
    for (int k = 0; k < 1 << 20; k++) {
        pc c;
        c.a = (k < NW * NW) ? W[k / NW] : (k & 1) ? W[rnd() % NW] : (uint32_t)rnd();
        c.b = (k < NW * NW) ? W[k % NW] : (uint32_t)rnd();
        c.q = (k % 4 == 0) ? (uint32_t)(rnd() >> 32) | 1u : Q[rnd() % NQ];
        if (c.q < 3u) c.q = 3u;
        c.qinv = c.q;
        c.qinv *= 2u - c.q * c.qinv; c.qinv *= 2u - c.q * c.qinv;
        c.qinv *= 2u - c.q * c.qinv; c.qinv *= 2u - c.q * c.qinv;
        const uint64_t top = ((uint64_t)c.q << 32) - 1u;          /* q*2^32 - 1 */
        switch (k % 5) {
        case 0:  c.T = 0; break;
        case 1:  c.T = 1; break;
        case 2:  c.T = top; break;
        case 3:  c.T = top - (rnd() & 0xFFFFu); break;
        default: c.T = rnd() % (top + 1u); break;
        }
        v.push_back(c);
    }
    const long n = (long)v.size();
    pc *din; uint32_t *dhi, *drd;
    if (cudaMalloc(&din, n * sizeof(pc)) || cudaMalloc(&dhi, n * 4) || cudaMalloc(&drd, n * 4)) {
        cuda_ok("cudaMalloc (primitives)"); return; }
    cudaMemcpy(din, v.data(), n * sizeof(pc), cudaMemcpyHostToDevice);
    k_prim<<<(unsigned)((n + 255) / 256), 256>>>(din, dhi, drd, n);
    std::vector<uint32_t> hi(n), rd(n);
    cudaMemcpy(hi.data(), dhi, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(rd.data(), drd, n * 4, cudaMemcpyDeviceToHost);
    if (!cuda_ok("primitives kernel")) return;
    cudaFree(din); cudaFree(dhi); cudaFree(drd);
    long bhi = 0, brd = 0;
    for (long i = 0; i < n; i++) {
        const uint32_t h = PL_MULHI32(v[i].a, v[i].b), r = pl_redc(v[i].T, v[i].q, v[i].qinv);
        if (h != hi[i] && bhi++ < 5)
            printf("    mulhi(%08x, %08x): host %08x device %08x\n", v[i].a, v[i].b, h, hi[i]);
        if ((r != rd[i] || r >= v[i].q) && brd++ < 5)
            printf("    redc(%llx, q=%u): host %u device %u\n", (unsigned long long)v[i].T, v[i].q, r, rd[i]);
    }
    check("PL_MULHI32, device == host", bhi, n);
    check("pl_redc, device == host, in [0, q)", brd, n);
}

/* ---- 2. the fast path ---------------------------------------------------- */

struct cs { uint32_t q, r; int64_t c[4]; };
__global__ void k_xf(const cs *in, uint32_t *rt, int *tk, long n)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = 0xdeadbeefu;
    tk[i] = pl_transform_fast_try(in[i].q, in[i].r, in[i].c[0], in[i].c[1], in[i].c[2], in[i].c[3], &x);
    rt[i] = x;
}
static int64_t coef()
{   /* inside, at and just past the 2^32 guard, both signs */
    const uint64_t u = rnd(); int64_t v;
    switch (u & 3) {
    case 0:  v = (int64_t)(rnd() >> 32); break;
    case 1:  v = (int64_t)(rnd() >> (33 + (int)(rnd() % 31))); break;
    case 2:  v = ((int64_t)1 << 32) - 1 - (int64_t)(rnd() % 4); break;
    default: v = ((int64_t)1 << 32) + (int64_t)(rnd() % 3); break;
    }
    return (u & 4) ? -v : v;
}
static std::vector<cs> B;
static long ncase, ntaken, nbad;
static int flush()
{
    const long n = (long)B.size();
    if (!n) return 1;
    cs *din; uint32_t *drt; int *dtk;
    if (cudaMalloc(&din, n * sizeof(cs)) || cudaMalloc(&drt, n * 4) || cudaMalloc(&dtk, n * 4))
        return cuda_ok("cudaMalloc (fast path)");
    cudaMemcpy(din, B.data(), n * sizeof(cs), cudaMemcpyHostToDevice);
    k_xf<<<(unsigned)((n + 255) / 256), 256>>>(din, drt, dtk, n);
    std::vector<uint32_t> rt(n); std::vector<int> tk(n);
    cudaMemcpy(rt.data(), drt, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(tk.data(), dtk, n * 4, cudaMemcpyDeviceToHost);
    if (!cuda_ok("fast-path kernel")) return 0;
    cudaFree(din); cudaFree(drt); cudaFree(dtk);
    for (long i = 0; i < n; i++) {
        uint32_t h = 0xdeadbeefu;
        const int ht = pl_transform_fast_try(B[i].q, B[i].r, B[i].c[0], B[i].c[1], B[i].c[2], B[i].c[3], &h);
        ncase++; ntaken += ht;
        if (ht != tk[i] || (ht && h != rt[i]))
            if (nbad++ < 5)
                printf("    q=%u r=%u a=(%lld,%lld) b=(%lld,%lld): host %d/%u device %d/%u\n",
                       B[i].q, B[i].r, (long long)B[i].c[0], (long long)B[i].c[1],
                       (long long)B[i].c[2], (long long)B[i].c[3], ht, h, tk[i], rt[i]);
    }
    B.clear();
    return 1;
}
static int add(uint32_t q, uint32_t r, int64_t a0, int64_t a1, int64_t b0, int64_t b1)
{
    B.push_back(cs{q, r, {a0, a1, b0, b1}});
    return B.size() >= (1u << 22) ? flush() : 1;
}
static void run_fast()
{
    /* exhaustive: odd q < 256, every root in [0, 2q), every a0 in [1, q] */
    for (uint32_t q = 3; q < 256; q += 2)
        for (uint32_t r = 0; r < 2 * q; r++)
            for (uint32_t a = 1; a <= q; a++) {
                const int64_t b0 = (int64_t)(rnd() >> 33), b1 = (int64_t)(rnd() >> 33) - ((int64_t)1 << 30);
                if (!add(q, r, a, 0, b0, b1)) return;
            }
    /* random odd moduli, every bit length 2..32 */
    for (long i = 0; i < 4000000; i++) {
        const int bits = 2 + (int)(rnd() % 31);
        uint32_t q = (uint32_t)(rnd() >> (64 - bits)) | (1u << (bits - 1)) | 1u;
        if (q < 3) q = 3;
        const uint32_t r = (uint32_t)(rnd() % (2ull * q));
        const int64_t a0 = coef(), a1 = coef(), b0 = coef(), b1 = coef();
        if (!add(q, r, a0, a1, b0, b1)) return;
    }
    /* the largest odd moduli */
    for (uint32_t q = 0xFFFFFFFFu; q > 0xFFFFF000u; q -= 2)
        for (int k = 0; k < 64; k++) {
            const uint32_t r = (uint32_t)(rnd() % (2ull * q));
            const int64_t a0 = coef(), a1 = coef(), b0 = coef(), b1 = coef();
            if (!add(q, r, a0, a1, b0, b1)) return;
        }
    if (!flush()) return;
    check("pl_transform_fast_try, device == host", nbad, ncase);
    checks++;
    if (ntaken == 0 || ntaken == ncase) {
        fails++;
        printf("  FAIL the fast path must be both taken and refused (%ld of %ld taken)\n", ntaken, ncase);
    } else {
        printf("  ok   fast path taken on %ld of %ld inputs, refused on the rest\n", ntaken, ncase);
    }
}

int main()
{
    printf("[xftest] the transform's fast path and its device primitives, device against host\n");
    run_prim();
    run_fast();
    printf("%d of %d checks failed\n", fails, checks);
    return (checks > 0 && fails == 0) ? 0 : 1;
}
