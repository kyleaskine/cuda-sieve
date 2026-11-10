/* multest -- the gate for cofac_arith.cuh's PTX products (finding 153).
 *
 * mz_mul<3> and mz_mul<4> as k_cofac runs them, against an independent
 * reference: the full product a*b, then 32L halvings mod n (add n when odd,
 * shift right), then one conditional subtract. It shares nothing with the
 * word-level CIOS under test but the operands, and it is checked on the host
 * against itself as compiled for the device.
 *
 * WHAT IT DRAWS, and why random moduli are not enough. The top word's carries
 * fire about once in 2^32 rows on random input, so a product that drops one
 * passes random moduli -- and moduli just under the width's limit -- alike
 * (work/eomul/mutants.py: 2 of 10 single-carry mutants did). So each width
 * runs three kinds of modulus:
 *     random      odd n of exactly B bits, every B from 34 to the limit
 *     near max    n = 2^B - (random < 2^20), operands near n half the time
 *     structured  words drawn from {~0, ~1, 1, 0, random}: where a carry
 *                 chains through a whole accumulator
 * and every kind takes edge operands {0, 1, 2, n-1, n-2, x} in all 36 pairings
 * plus a chain of dependent products. VEC3 holds six (n, a, b) that
 * work/eomul/search.cu found to drive the reduction step's odd-chain carry
 * into an all-ones E3, the case that got past the first version of this test.
 * With all of it, every one of mutants.py's ten mutants fails here: rerun them
 * after any change to the chains.
 *
 * The limits are CF_WIDTH_BITS: 96 at 3 limbs, 127 at 4. Exit 0 only if every
 * check ran and matched. Needs a CUDA device.
 *
 *   make mulcheck        (builds multest, runs it)
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cofac_arith.cuh"

#define NT    32768      /* moduli per case */
#define CHAIN 128        /* dependent products per chain */
#define NE    6          /* edge operands */

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "multest: CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

/* ---- the reference ------------------------------------------------------ */

/* r = a*b*2^(-32L) mod n in [0, n), for odd n < 2^(32L) and a, b < n.
 * t = a*b < n^2; each halving keeps t < max(t, (t + n) / 2) and divides by 2
 * mod n exactly (n is odd), so after 32L of them t = (ab + kn) / R for some
 * k < R, which is < n^2/R + n < 2n: one subtract finishes it. 2L+1 words hold
 * every intermediate. */
template <int L>
__host__ __device__ void ref_mul(mz<L> *r, const mz<L> *a, const mz<L> *b, const mz<L> *n)
{
    uint32_t t[2 * L + 1];
    for (int k = 0; k < 2 * L + 1; k++) t[k] = 0;
    for (int i = 0; i < L; i++) {
        uint64_t c = 0;
        for (int j = 0; j < L; j++) {
            c += (uint64_t)a->v[j] * b->v[i] + t[i + j];
            t[i + j] = (uint32_t)c; c >>= 32;
        }
        t[i + L] = (uint32_t)c;
    }
    for (int s = 0; s < 32 * L; s++) {
        if (t[0] & 1) {
            uint64_t c = 0;
            for (int k = 0; k < 2 * L + 1; k++) {
                c += (uint64_t)t[k] + (k < L ? n->v[k] : 0);
                t[k] = (uint32_t)c; c >>= 32;
            }
        }
        for (int k = 0; k < 2 * L; k++) t[k] = (t[k] >> 1) | (t[k + 1] << 31);
        t[2 * L] >>= 1;
    }
    /* t < 2n: subtract n once if t >= n (t[L] is the only word above n's) */
    uint32_t d[L + 1]; int64_t bw = 0;
    for (int k = 0; k <= L; k++) {
        int64_t x = (int64_t)t[k] - (k < L ? n->v[k] : 0) - bw;
        d[k] = (uint32_t)x; bw = x < 0;
    }
    for (int k = 0; k < L; k++) r->v[k] = bw ? t[k] : d[k];
}

/* ---- the kernels ---------------------------------------------------------- */

template <int L>
__device__ void edges(mz<L> *e, const mz<L> &n, const mz<L> &x)
{
    for (int i = 0; i < 3; i++) for (int k = 0; k < L; k++) e[i].v[k] = 0;
    e[1].v[0] = 1; e[2].v[0] = 2;
    e[3] = n; e[3].v[0] -= 1;                 /* n odd: no borrow */
    e[4] = n;                                 /* n - 2, borrowing as needed */
    { uint32_t bw = 2; for (int k = 0; k < L && bw; k++) { uint32_t o = e[4].v[k]; e[4].v[k] = o - bw; bw = o < bw; } }
    e[5] = x;
}

/* out: per modulus, NE*NE products, then 2 chain results; the REF arm writes
 * the same layout from ref_mul */
template <int L, bool REF>
__global__ void k_run(const mz<L> *nn, const mz<L> *xx, mz<L> *out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NT) return;
    const mz<L> n = nn[t];
    const uint32_t ni = mz_n0inv(n.v[0]);
    mz<L> e[NE], *o = out + (size_t)t * (NE * NE + 2);
    edges<L>(e, n, xx[t]);
    for (int i = 0; i < NE; i++)
        for (int j = 0; j < NE; j++) {
            if (REF) ref_mul<L>(&o[i * NE + j], &e[i], &e[j], &n);
            else     mz_mul<L>(&o[i * NE + j], &e[i], &e[j], &n, ni);
        }
    mz<L> x = xx[t], y = xx[t ^ 1], s;
    if (mz_cmp<L>(&y, &n) >= 0) y = e[3];     /* t^1's x need not be < this n */
    for (int c = 0; c < CHAIN; c++) {
        if (REF) { ref_mul<L>(&s, &x, &y, &n); x = s; ref_mul<L>(&s, &y, &x, &n); y = s; }
        else     { mz_mul<L>(&s, &x, &y, &n, ni); x = s; mz_mul<L>(&s, &y, &x, &n, ni); y = s; }
    }
    o[NE * NE] = x; o[NE * NE + 1] = y;
}

/* ---- inputs ----------------------------------------------------------------- */

static uint64_t rs = 0x2545F4914F6CDD1Dull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }

enum { RANDOM, NEARMAX, STRUCTURED };
static const char *kind_name[] = { "random", "near max", "structured" };

/* odd n of exactly `bits` bits, x < n */
template <int L>
static void gen(int bits, int kind, std::vector<mz<L>> &n, std::vector<mz<L>> &x)
{
    static const uint32_t W[4] = { 0xffffffffu, 0xfffffffeu, 1u, 0u };
    n.resize(NT); x.resize(NT);
    for (int t = 0; t < NT; t++) {
        for (int k = 0; k < L; k++) {
            n[t].v[k] = rnd(); x[t].v[k] = rnd();
            if (kind == STRUCTURED) {
                uint32_t r = rnd() & 7; if (r < 4) n[t].v[k] = W[r];
                if ((rnd() & 3) == 0) x[t].v[k] = 0xffffffffu;
            }
            if (kind == NEARMAX) n[t].v[k] = 0xffffffffu;
            const int lo = 32 * k, hi = 32 * k + 32;
            if (bits <= lo) { n[t].v[k] = 0; x[t].v[k] = 0; }
            else if (bits < hi) {
                const uint32_t mask = (1u << (bits - lo)) - 1;
                n[t].v[k] &= mask; n[t].v[k] |= 1u << (bits - lo - 1);
                x[t].v[k] &= mask >> 1;               /* x < 2^(bits-1) <= n */
            } else if (bits == hi) { n[t].v[k] |= 0x80000000u; x[t].v[k] &= 0x7fffffffu; }
        }
        if (kind == NEARMAX) n[t].v[0] ^= rnd() & 0xffffeu;   /* 2^B - small */
        n[t].v[0] |= 1;
        if (kind != RANDOM && (t & 3) == 1 && n[t].v[0] >= 5) {  /* x near n */
            x[t] = n[t]; x[t].v[0] -= 2 + 2 * (rnd() & 0xff);
            if (x[t].v[0] > n[t].v[0]) x[t].v[0] = n[t].v[0] - 2;  /* no borrow */
        }
    }
}

static int checks = 0, fails = 0;

template <int L>
static void run_case(int bits, int kind)
{
    std::vector<mz<L>> n, x;
    gen<L>(bits, kind, n, x);
    const size_t per = NE * NE + 2, nout = (size_t)NT * per;
    std::vector<mz<L>> got(nout), want(nout);
    mz<L> *dn, *dx, *dout;
    CK(cudaMalloc(&dn, NT * sizeof(mz<L>))); CK(cudaMalloc(&dx, NT * sizeof(mz<L>)));
    CK(cudaMalloc(&dout, nout * sizeof(mz<L>)));
    CK(cudaMemcpy(dn, n.data(), NT * sizeof(mz<L>), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), NT * sizeof(mz<L>), cudaMemcpyHostToDevice));
    k_run<L, false><<<NT / 256, 256>>>(dn, dx, dout); CK(cudaGetLastError());
    CK(cudaMemcpy(got.data(), dout, nout * sizeof(mz<L>), cudaMemcpyDeviceToHost));
    k_run<L, true><<<NT / 256, 256>>>(dn, dx, dout); CK(cudaGetLastError());
    CK(cudaMemcpy(want.data(), dout, nout * sizeof(mz<L>), cudaMemcpyDeviceToHost));
    cudaFree(dn); cudaFree(dx); cudaFree(dout);

    /* the reference as compiled for the device, against itself on the host,
     * on the first moduli's edge products */
    size_t bad_ref = 0;
    for (int t = 0; t < 64; t++)
        for (int i = 0; i < NE * NE; i++) {
            const mz<L> *e0 = &want[(size_t)t * per + i];
            /* recompute this pair on the host from the same operands */
            mz<L> ops[NE], r;
            for (int q = 0; q < 3; q++) for (int k = 0; k < L; k++) ops[q].v[k] = 0;
            ops[1].v[0] = 1; ops[2].v[0] = 2; ops[3] = n[t]; ops[3].v[0] -= 1;
            ops[4] = n[t]; { uint32_t bw = 2; for (int k = 0; k < L && bw; k++) { uint32_t o = ops[4].v[k]; ops[4].v[k] = o - bw; bw = o < bw; } }
            ops[5] = x[t];
            ref_mul<L>(&r, &ops[i / NE], &ops[i % NE], &n[t]);
            if (memcmp(&r, e0, sizeof r)) bad_ref++;
        }
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < nout; i++)
        if (memcmp(&got[i], &want[i], sizeof(mz<L>))) { if (!bad) first = i; bad++; }
    checks++;
    const bool ok = bad == 0 && bad_ref == 0;
    if (!ok) fails++;
    printf("%-4s L=%d n %3d bits %-10s  %s", ok ? "PASS" : "FAIL", L, bits, kind_name[kind],
           bad ? "" : "products identical to the reference");
    if (bad) printf("%zu of %zu products differ (first: modulus %zu, %s %zu)", bad, nout,
                    first / per, first % per < NE * NE ? "edge pair" : "chain", first % per);
    if (bad_ref) printf("; device reference differs from host on %zu", bad_ref);
    printf("\n");
}

/* (n, a, b) from work/eomul/search.cu, words high to low */
static const uint32_t VEC3[][9] = {
    { 0xfffffffe, 0xffffffff, 0xffffffff,  0xfffffffe, 0xffffffff, 0xfffffffd,  0x50e0af2f, 0x69c9b398, 0xffffffff },
    { 0x80000001, 0x3c746a14, 0x00000001,  0x7fffffff, 0xffffffff, 0xffffffff,  0x1863ef33, 0x490f33ab, 0xffffffff },
    { 0x80000001, 0xffffffff, 0x00000001,  0x7fffffff, 0xffffffff, 0xffffffff,  0x7fffffff, 0xffffffff, 0xffffffff },
    { 0x80000001, 0xffffffff, 0x00000001,  0x7fffffff, 0xffffffff, 0xffffffff,  0x0f9755f1, 0xffffffff, 0xffffffff },
    { 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xfffffffd,  0x7fffffff, 0xce14752e, 0x5111215e },
    { 0xffffffff, 0xffffffff, 0xffffffff,  0x7fffffff, 0xe18ea193, 0x26c91679,  0xffffffff, 0xffffffff, 0xfffffffb },
};
#define NVEC3 (int)(sizeof VEC3 / sizeof VEC3[0])

__global__ void k_vec3(const mz<3> *nab, mz<3> *out)
{
    const int t = threadIdx.x;
    if (t >= NVEC3) return;
    const mz<3> n = nab[3 * t], a = nab[3 * t + 1], b = nab[3 * t + 2];
    mz_mul<3>(&out[t], &a, &b, &n, mz_n0inv(n.v[0]));
}

static void run_vec3()
{
    std::vector<mz<3>> h(3 * NVEC3), got(NVEC3);
    for (int i = 0; i < NVEC3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) h[3 * i + j].v[k] = VEC3[i][3 * j + 2 - k];
    mz<3> *d, *o;
    CK(cudaMalloc(&d, h.size() * sizeof(mz<3>))); CK(cudaMalloc(&o, NVEC3 * sizeof(mz<3>)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(mz<3>), cudaMemcpyHostToDevice));
    k_vec3<<<1, 32>>>(d, o); CK(cudaGetLastError());
    CK(cudaMemcpy(got.data(), o, NVEC3 * sizeof(mz<3>), cudaMemcpyDeviceToHost));
    cudaFree(d); cudaFree(o);
    int bad = 0;
    for (int i = 0; i < NVEC3; i++) {
        mz<3> r; ref_mul<3>(&r, &h[3 * i + 1], &h[3 * i + 2], &h[3 * i]);
        if (memcmp(&r, &got[i], sizeof r)) bad++;
    }
    checks++;
    if (bad) fails++;
    printf("%-4s L=3 %d search vectors (rare top-word carries)%s\n", bad ? "FAIL" : "PASS", NVEC3,
           bad ? ": differ from the reference" : "");
}

int main()
{
    printf("[multest] mz_mul<3> up to %u bits, mz_mul<4> up to %u bits, against a bit-serial reference\n",
           CF_WIDTH_BITS(3), CF_WIDTH_BITS(4));
    /* every size a 3- or 4-limb side can be handed: the split stack passes
     * composites far below the side's mfb, so the small sizes run too */
    for (int b : { 34, 50, 63, 64, 65, 80, 92, 94, 95, 96 }) run_case<3>(b, RANDOM);
    for (int b : { 34, 64, 96, 97, 101, 111, 120, 126, 127 }) run_case<4>(b, RANDOM);
    for (int b : { 64, 95, 96 }) { run_case<3>(b, NEARMAX); run_case<3>(b, STRUCTURED); }
    for (int b : { 96, 111, 126, 127 }) { run_case<4>(b, NEARMAX); run_case<4>(b, STRUCTURED); }
    run_vec3();
    static_assert(CF_WIDTH_BITS(3) == 96u && CF_WIDTH_BITS(4) == 127u,
                  "multest's sizes stop at the widths' limits; move them with CF_WIDTH_BITS");
    printf("%d of %d checks failed\n", fails, checks);
    return (checks > 0 && fails == 0) ? 0 : 1;
}
