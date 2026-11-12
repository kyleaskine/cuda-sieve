/* cofac_arith.cuh -- the L-limb Montgomery arithmetic of the cofactoriser:
 * mz<L>, the generic templates (host builds and any L), and the PTX
 * specialisations for the two widths that run, L = 3 and L = 4, including the
 * even/odd products (finding 153).
 *
 * Split out of cofac.cuh, unchanged, so that mulcheck.cu can test exactly the
 * code k_cofac runs without the rest of that header. cofac.cuh includes it
 * at the point where this code used to begin. */
#ifndef CUDA_SIEVE_COFAC_ARITH_CUH
#define CUDA_SIEVE_COFAC_ARITH_CUH

#include <stdint.h>
#include "bench.h"          /* CF_WIDTH_BITS: the bound each product needs */

#if defined(__CUDACC__)
#define CF_FN __device__ __forceinline__
#define CF_NOINLINE __device__ __noinline__
#define CF_HD static __host__ __device__ __forceinline__
#else
#define CF_FN static inline
#define CF_NOINLINE static
#define CF_HD static inline
#endif

/* ---- L-limb arithmetic ------------------------------------------------- *
 *
 * Templated on the limb count so the rational side pays for 64 bits and the
 * algebraic side for 96, rather than both paying prp.cuh's fixed 128. At three
 * limbs the CIOS inner loop is 9 multiply-adds against 16 -- the modmul is the
 * whole cost of this stage, so the limb count is not a detail.
 */

template <int L> struct mz { uint32_t v[L]; };

template <int L> CF_FN int mz_cmp(const mz<L> *a, const mz<L> *b)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
    return 0;
}

template <int L> CF_FN int mz_is_zero(const mz<L> *a)
{
    uint32_t o = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> CF_FN int mz_is_one(const mz<L> *a)
{
    uint32_t o = a->v[0] ^ 1u;
    #pragma unroll
    for (int i = 1; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> CF_FN uint32_t mz_sub(mz<L> *a, const mz<L> *b)
{
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t d = (uint64_t)a->v[i] - b->v[i] - borrow;
        a->v[i] = (uint32_t)d;
        borrow = (d >> 32) & 1u;
    }
    return (uint32_t)borrow;
}

/* a = a - b mod n, given a, b < n */
template <int L> CF_FN void mz_sub_mod(mz<L> *a, const mz<L> *b, const mz<L> *n)
{
    if (mz_sub<L>(a, b)) {
        uint64_t c = 0;
        #pragma unroll
        for (int i = 0; i < L; i++) {
            uint64_t s = (uint64_t)a->v[i] + n->v[i] + c;
            a->v[i] = (uint32_t)s;
            c = s >> 32;
        }
    }
}

/* a = a + b mod n, given a, b < n */
template <int L> CF_FN void mz_add_mod(mz<L> *a, const mz<L> *b, const mz<L> *n)
{
    uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t s = (uint64_t)a->v[i] + b->v[i] + c;
        a->v[i] = (uint32_t)s;
        c = s >> 32;
    }
    if (c || mz_cmp<L>(a, n) >= 0) mz_sub<L>(a, n);
}

/* a = 2a mod n */
template <int L> CF_FN void mz_dbl_mod(mz<L> *a, const mz<L> *n)
{
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint32_t hi = a->v[i] >> 31;
        a->v[i] = (a->v[i] << 1) | carry;
        carry = hi;
    }
    if (carry || mz_cmp<L>(a, n) >= 0) mz_sub<L>(a, n);
}

template <int L> CF_FN void mz_rshift1(mz<L> *a)
{
    #pragma unroll
    for (int i = 0; i < L - 1; i++)
        a->v[i] = (a->v[i] >> 1) | (a->v[i + 1] << 31);
    a->v[L - 1] >>= 1;
}

template <int L> CF_FN int mz_bits(const mz<L> *a)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i]) return i * 32 + (32 - __clz(a->v[i]));
    return 0;
}

/* -n^{-1} mod 2^32, Newton. n odd. */
CF_FN uint32_t mz_n0inv(uint32_t n0)
{
    uint32_t inv = n0;
    for (int k = 0; k < 5; k++) inv *= 2u - n0 * inv;
    return (uint32_t)(0u - inv);
}

/* r = a*b*R^-1 mod n, R = 2^(32L). CIOS. */
template <int L>
CF_FN void mz_mul(mz<L> *r, const mz<L> *a, const mz<L> *b,
                  const mz<L> *n, uint32_t n0inv)
{
    uint32_t t[L + 2];
    #pragma unroll
    for (int i = 0; i < L + 2; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t C = 0;
        #pragma unroll
        for (int j = 0; j < L; j++) {
            uint64_t p = (uint64_t)a->v[j] * b->v[i] + t[j] + C;
            t[j] = (uint32_t)p;
            C = p >> 32;
        }
        {
            uint64_t p = (uint64_t)t[L] + C;
            t[L] = (uint32_t)p;
            t[L + 1] = (uint32_t)(p >> 32);
        }
        {
            uint32_t m = t[0] * n0inv;
            uint64_t p = (uint64_t)m * n->v[0] + t[0];
            C = p >> 32;
            #pragma unroll
            for (int j = 1; j < L; j++) {
                p = (uint64_t)m * n->v[j] + t[j] + C;
                t[j - 1] = (uint32_t)p;
                C = p >> 32;
            }
            p = (uint64_t)t[L] + C;
            t[L - 1] = (uint32_t)p;
            t[L] = t[L + 1] + (uint32_t)(p >> 32);
        }
    }
    #pragma unroll
    for (int i = 0; i < L; i++) r->v[i] = t[i];
    if (t[L] || mz_cmp<L>(r, n) >= 0) mz_sub<L>(r, n);
}

/* The bits of a W-limb value's top word that width W's ARITHMETIC does not
 * accept (CF_WIDTH_BITS): none at 3 limbs, bit 31 at 4, whose product needs
 * n < 2^127. The one copy of that shift, for every fit test: k_cof_enqueue's,
 * per side, and cf_over_width's. */
template <int W> CF_HD uint32_t cf_top_over(uint32_t top)
{
    if constexpr (CF_WIDTH_BITS(W) < 32u * W)
        return top >> (CF_WIDTH_BITS(W) - 32u * (W - 1));
    else
        return 0;
}

/* ---- PTX carry chains for the two widths that run: L = 3 and L = 4 ------ *
 *
 * Specialisations of mz_mul, mz_add_mod, mz_sub_mod and mz_dbl_mod above, bit
 * for bit the same results FOR REDUCED OPERANDS, a, b < n (and n < 2^127 for
 * mz_mul<4>, see THE BOUND below) -- the only kind any caller passes, since
 * every value here lives in [0, n). They are NOT the generic code on
 * unreduced input: with a >= n, or a + b >= R + n, the final reduce keeps the
 * truncated value where the generic `t[L] || cmp >= 0` subtracts n. Reduce
 * first before feeding anything raw to them. Device only: CF_FN is __device__
 * under nvcc, and a host build of this header keeps the generic templates.
 *
 * The evidence. mz_add_mod, mz_sub_mod and mz_dbl_mod are the PTX of finding
 * 125: work/gpuecm-probe/mul/arithcheck.cu, 262,144 random moduli per width
 * from 2 to 32L bits, 256 chained steps, identical to the generic code, and an
 * injected borrow bug and madc->mad bug each make them differ. The products
 * are finding 153's, and their gate is in the tree: mulcheck.cu (`make
 * mulcheck`) runs these very functions against an independent bit-serial
 * reference.
 *
 * WHY. The generic add/sub end in mz_cmp, a loop with an early return. Inlined
 * into every x_add/x_dbl of the ECM ladder and unrolled, that is one divergent
 * region per add or sub: k_cofac<3, ECM>'s SASS was 293,632 instructions with
 * 4,965 BSSY regions, ~4.7 MB, far past the instruction cache. Here the reduce
 * is a borrow mask (sub) or a select on the borrow of sum - n (add): 18,224
 * instructions, 76 BSSY. The 64-bit C products also lowered to IADD.64 plus a
 * MOV/HFMA2 per zeroed high half; mad/madc with carry is about 2/3 of that.
 * Registers unchanged under __launch_bounds__(256, 2). Measured on the I16
 * NFS@Home batch: ECM 0.37x and rho 0.56x the time, I16 end to end 187.5 ->
 * 170.4 ms/q with byte-identical relations (work/ptx-ab/NOTES.md).
 *
 * THE PRODUCT (2026-11-09), even/odd accumulators -- the layout of Greg
 * Childers' one-thread GPU ECM product (gmp-ecm, branch gpu-stage1-dynamic,
 * GPLv3), reimplemented here, not copied. The CIOS rows they replaced (still
 * the shape of the generic mz_mul above) wrote the low words of a row's
 * products as one carry chain and the high words as another, and ptxas
 * lowered every product to IMAD + IMAD.HI + IADD.X. Here the CIOS
 * accumulator T is two arrays:
 *     E[k] at word position k     products a_j*x, j even: (E[j], E[j+1])
 *     O[k] at word position k+1   products a_j*x, j odd:  (O[j-1], O[j])
 * so the low and high word of every product share a register pair, and
 * mad.lo.cc + madc.hi.cc on that pair is ONE IMAD.WIDE.U32(.X). Dividing by
 * W moves every position down one, which flips each pair's parity: the next
 * row's E is this row's O, the next row's O is this row's E two words down
 * (taken as the addends of the next odd chain, so the shift is free), and
 * E[1], the one word left at position 0, is added into the new E[0] with its
 * carry consumed by the odd chain's first product. sm_120, CUDA 13.4, per
 * product: 95 -> 46 SASS instructions at L = 3, 158 -> 59 at L = 4.
 *
 * Random moduli are not a test of these. The top word's carries fire about
 * once in 2^32 rows on random input, so a dropped one passes random and
 * near-limit moduli alike (work/eomul/mutants.py: 2 of 10 single-carry
 * mutants did). mulcheck.cu therefore also draws words from {~0, ~1, 1, 0}
 * and keeps the vectors work/eomul/search.cu found; with them it catches all
 * ten. Run its mutants again after any change to these chains.
 *
 * THE BOUND, and why the two widths differ. With a < n, every T after a row
 * is < a + n, so every sum before a division is < (a + n) W. If n < R/2 that
 * is < R W and no chain ever carries out of position L: the L = 4 product
 * relies on this, so it REQUIRES n < 2^127 -- cf_width_bits(4) is 127 and
 * mfb 128 is refused upstream (no job is near it; 3*lpb at lpb 37 is 111).
 * The L = 3 product does not: it keeps one carry word h for position 4, so it
 * takes any odd n < 2^96, and mfb 96 -- the I16 NFS@Home job's mfba -- stays
 * on 3 limbs. h costs 6 instructions a product (40 without it), against
 * moving mfb 96 to the 4-limb kernel.
 *
 * OUTPUTS ARE "=r", NOT "=&r". Most blocks write an output before their last
 * input is read, which under GCC's rules would want the early-clobber "&".
 * nvcc 12.0 rejects it ("an asm operand may specify only one constraint
 * letter", a forum report against ad05510), and NVIDIA's inline-PTX docs
 * define only "=" and "+". Nor does nvcc need it: the asm's operands are PTX
 * virtual registers and ptxas allocates after the block is spliced in. With
 * and without "&", the PTX is byte-identical on 12.8 and 13.5, and
 * work/earlyclobber/aliascheck.py finds no output sharing a register with
 * another operand on 12.0, 12.8 or 13.5, every arch (its 20 mutants, one per
 * block template, are all caught). Rerun it after adding or changing a block
 * here. */
#if defined(__CUDACC__)
template <>
CF_FN void mz_mul<3>(mz<3> *r, const mz<3> *a, const mz<3> *b,
                      const mz<3> *n, uint32_t n0inv)
{
    const uint32_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2];
    const uint32_t n0 = n->v[0], n1 = n->v[1], n2 = n->v[2];
    const uint32_t z = 0;
    uint32_t e0, e1, e2, e3, o0, o1, h, m;

    {
        const uint32_t x = b->v[0];
        asm("mul.lo.u32 %0, %6, %9;\n\t"
            "mul.hi.u32 %1, %6, %9;\n\t"
            "mul.lo.u32 %2, %8, %9;\n\t"
            "mul.hi.u32 %3, %8, %9;\n\t"
            "mul.lo.u32 %4, %7, %9;\n\t"
            "mul.hi.u32 %5, %7, %9;"
            : "=r"(e0), "=r"(e1), "=r"(e2), "=r"(e3), "=r"(o0), "=r"(o1)
            : "r"(a0), "r"(a1), "r"(a2), "r"(x));
    }
    m = e0 * n0inv;
    asm("mad.lo.cc.u32  %0, %7, %8, %0;\n\t"
        "madc.hi.cc.u32 %1, %7, %8, %1;\n\t"
        "madc.lo.cc.u32 %2, %7, %10, %2;\n\t"
        "madc.hi.cc.u32 %3, %7, %10, %3;\n\t"
        "addc.u32       %6, 0, 0;\n\t"           /* position 4 */
        "mad.lo.cc.u32  %4, %7, %9, %4;\n\t"
        "madc.hi.cc.u32 %5, %7, %9, %5;\n\t"
        "addc.cc.u32    %3, %3, 0;\n\t"
        "addc.u32       %6, %6, 0;"
        : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3), "+r"(o0), "+r"(o1), "=r"(h)
        : "r"(m), "r"(n0), "r"(n1), "r"(n2));

    #pragma unroll
    for (int i = 1; i < 3; i++) {
        const uint32_t x = b->v[i];
        uint32_t E0, E1, E2, E3, O0, O1, H;
        asm("add.cc.u32     %0, %7, %9;\n\t"
            "madc.lo.cc.u32 %4, %14, %16, %10;\n\t"
            "madc.hi.cc.u32 %5, %14, %16, %11;\n\t"
            "addc.u32       %3, %12, 0;\n\t"         /* old h (now position 3) + carry */
            "mad.lo.cc.u32  %0, %13, %16, %0;\n\t"
            "madc.hi.cc.u32 %1, %13, %16, %8;\n\t"
            "madc.lo.cc.u32 %2, %15, %16, %17;\n\t"
            "madc.hi.cc.u32 %3, %15, %16, %3;\n\t"
            "addc.u32       %6, 0, 0;"
            : "=r"(E0), "=r"(E1), "=r"(E2), "=r"(E3), "=r"(O0), "=r"(O1), "=r"(H)
            : "r"(o0), "r"(o1), "r"(e1), "r"(e2), "r"(e3), "r"(h),
              "r"(a0), "r"(a1), "r"(a2), "r"(x), "r"(z));
        m = E0 * n0inv;
        asm("mad.lo.cc.u32  %0, %7, %8, %0;\n\t"
            "madc.hi.cc.u32 %1, %7, %8, %1;\n\t"
            "madc.lo.cc.u32 %2, %7, %10, %2;\n\t"
            "madc.hi.cc.u32 %3, %7, %10, %3;\n\t"
            "addc.u32       %6, %6, 0;\n\t"
            "mad.lo.cc.u32  %4, %7, %9, %4;\n\t"
            "madc.hi.cc.u32 %5, %7, %9, %5;\n\t"
            "addc.cc.u32    %3, %3, 0;\n\t"
            "addc.u32       %6, %6, 0;"
            : "+r"(E0), "+r"(E1), "+r"(E2), "+r"(E3), "+r"(O0), "+r"(O1), "+r"(H)
            : "r"(m), "r"(n0), "r"(n1), "r"(n2));
        e1 = E1; e2 = E2; e3 = E3; o0 = O0; o1 = O1; h = H;
    }

    {
        uint32_t t0, t1, t2, t3, s0, s1, s2, bw;
        asm("add.cc.u32  %0, %4, %6;\n\t"
            "addc.cc.u32 %1, %5, %7;\n\t"
            "addc.cc.u32 %2, %8, 0;\n\t"
            "addc.u32    %3, %9, 0;"
            : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
            : "r"(o0), "r"(o1), "r"(e1), "r"(e2), "r"(e3), "r"(h));
        asm("sub.cc.u32  %0, %4, %7;\n\t"
            "subc.cc.u32 %1, %5, %8;\n\t"
            "subc.cc.u32 %2, %6, %9;\n\t"
            "subc.u32    %3, %10, 0;"               /* t3 is 0 or 1 */
            : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(bw)
            : "r"(t0), "r"(t1), "r"(t2), "r"(n0), "r"(n1), "r"(n2), "r"(t3));
        const bool keep = bw != 0;
        r->v[0] = keep ? t0 : s0; r->v[1] = keep ? t1 : s1; r->v[2] = keep ? t2 : s2;
    }
}

template <>
CF_FN void mz_sub_mod<3>(mz<3> *a, const mz<3> *b, const mz<3> *n)
{
    uint32_t d0, d1, d2, m;
    asm("sub.cc.u32  %0, %4, %7;\n\t"
        "subc.cc.u32 %1, %5, %8;\n\t"
        "subc.cc.u32 %2, %6, %9;\n\t"
        "subc.u32    %3, 0, 0;"            /* m = 0 or 0xffffffff (borrow) */
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(m)
        : "r"(a->v[0]), "r"(a->v[1]), "r"(a->v[2]), "r"(b->v[0]), "r"(b->v[1]), "r"(b->v[2]));
    asm("add.cc.u32  %0, %0, %3;\n\t"
        "addc.cc.u32 %1, %1, %4;\n\t"
        "addc.u32    %2, %2, %5;"
        : "+r"(d0), "+r"(d1), "+r"(d2)
        : "r"(n->v[0] & m), "r"(n->v[1] & m), "r"(n->v[2] & m));
    a->v[0] = d0; a->v[1] = d1; a->v[2] = d2;
}

template <>
CF_FN void mz_add_mod<3>(mz<3> *a, const mz<3> *b, const mz<3> *n)
{
    uint32_t s0, s1, s2, s3, r0, r1, r2, bw;
    asm("add.cc.u32  %0, %4, %7;\n\t"
        "addc.cc.u32 %1, %5, %8;\n\t"
        "addc.cc.u32 %2, %6, %9;\n\t"
        "addc.u32    %3, 0, 0;"
        : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3)
        : "r"(a->v[0]), "r"(a->v[1]), "r"(a->v[2]), "r"(b->v[0]), "r"(b->v[1]), "r"(b->v[2]));
    asm("sub.cc.u32  %0, %4, %7;\n\t"
        "subc.cc.u32 %1, %5, %8;\n\t"
        "subc.cc.u32 %2, %6, %9;\n\t"
        "subc.u32    %3, %10, 0;"          /* borrow out of (s3:s) - n */
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(bw)
        : "r"(s0), "r"(s1), "r"(s2), "r"(n->v[0]), "r"(n->v[1]), "r"(n->v[2]), "r"(s3));
    const bool keep = bw != 0;              /* s < n: keep the sum */
    a->v[0] = keep ? s0 : r0; a->v[1] = keep ? s1 : r1; a->v[2] = keep ? s2 : r2;
}

template <>
CF_FN void mz_dbl_mod<3>(mz<3> *a, const mz<3> *n)
{
    const mz<3> b = *a;
    mz_add_mod<3>(a, &b, n);
}

template <>
CF_FN void mz_mul<4>(mz<4> *r, const mz<4> *a, const mz<4> *b,
                      const mz<4> *n, uint32_t n0inv)
{
    const uint32_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2], a3 = a->v[3];
    const uint32_t n0 = n->v[0], n1 = n->v[1], n2 = n->v[2], n3 = n->v[3];
    const uint32_t z = 0;
    uint32_t e0, e1, e2, e3, o0, o1, o2, o3, m;

    {
        const uint32_t x = b->v[0];
        asm("mul.lo.u32 %0, %8, %12;\n\t"
            "mul.hi.u32 %1, %8, %12;\n\t"
            "mul.lo.u32 %2, %10, %12;\n\t"
            "mul.hi.u32 %3, %10, %12;\n\t"
            "mul.lo.u32 %4, %9, %12;\n\t"
            "mul.hi.u32 %5, %9, %12;\n\t"
            "mul.lo.u32 %6, %11, %12;\n\t"
            "mul.hi.u32 %7, %11, %12;"
            : "=r"(e0), "=r"(e1), "=r"(e2), "=r"(e3),
              "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(x));
    }
    m = e0 * n0inv;
    asm("mad.lo.cc.u32  %0, %8, %9, %0;\n\t"
        "madc.hi.cc.u32 %1, %8, %9, %1;\n\t"
        "madc.lo.cc.u32 %2, %8, %11, %2;\n\t"
        "madc.hi.cc.u32 %3, %8, %11, %3;\n\t"
        "addc.u32       %7, %7, 0;\n\t"          /* even chain's carry, position 4 */
        "mad.lo.cc.u32  %4, %8, %10, %4;\n\t"
        "madc.hi.cc.u32 %5, %8, %10, %5;\n\t"
        "madc.lo.cc.u32 %6, %8, %12, %6;\n\t"
        "madc.hi.u32    %7, %8, %12, %7;"        /* nothing past position 4 */
        : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3), "+r"(o0), "+r"(o1), "+r"(o2), "+r"(o3)
        : "r"(m), "r"(n0), "r"(n1), "r"(n2), "r"(n3));

    #pragma unroll
    for (int i = 1; i < 4; i++) {
        const uint32_t x = b->v[i];
        uint32_t E0, E1, E2, E3, O0, O1, O2, O3;
        /* new E = old O (+ old E[1] at position 0); new O = old E >> 2 words,
         * taken as the odd chain's addends; old O[3] is the new E[3] */
        asm("add.cc.u32     %0, %8, %12;\n\t"
            "madc.lo.cc.u32 %4, %16, %19, %13;\n\t"
            "madc.hi.cc.u32 %5, %16, %19, %14;\n\t"
            "madc.lo.cc.u32 %6, %18, %19, %20;\n\t"
            "madc.hi.u32    %7, %18, %19, %20;\n\t"   /* nothing past position 4 */
            "mad.lo.cc.u32  %0, %15, %19, %0;\n\t"
            "madc.hi.cc.u32 %1, %15, %19, %9;\n\t"
            "madc.lo.cc.u32 %2, %17, %19, %10;\n\t"
            "madc.hi.cc.u32 %3, %17, %19, %11;\n\t"
            "addc.u32       %7, %7, 0;"               /* even chain's carry, position 4 */
            : "=r"(E0), "=r"(E1), "=r"(E2), "=r"(E3),
              "=r"(O0), "=r"(O1), "=r"(O2), "=r"(O3)
            : "r"(o0), "r"(o1), "r"(o2), "r"(o3), "r"(e1), "r"(e2), "r"(e3),
              "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(x), "r"(z));
        m = E0 * n0inv;
        asm("mad.lo.cc.u32  %0, %8, %9, %0;\n\t"
            "madc.hi.cc.u32 %1, %8, %9, %1;\n\t"
            "madc.lo.cc.u32 %2, %8, %11, %2;\n\t"
            "madc.hi.cc.u32 %3, %8, %11, %3;\n\t"
            "addc.u32       %7, %7, 0;\n\t"
            "mad.lo.cc.u32  %4, %8, %10, %4;\n\t"
            "madc.hi.cc.u32 %5, %8, %10, %5;\n\t"
            "madc.lo.cc.u32 %6, %8, %12, %6;\n\t"
            "madc.hi.u32    %7, %8, %12, %7;"
            : "+r"(E0), "+r"(E1), "+r"(E2), "+r"(E3), "+r"(O0), "+r"(O1), "+r"(O2), "+r"(O3)
            : "r"(m), "r"(n0), "r"(n1), "r"(n2), "r"(n3));
        e1 = E1; e2 = E2; e3 = E3; o0 = O0; o1 = O1; o2 = O2; o3 = O3;
    }

    {
        uint32_t t0, t1, t2, t3, s0, s1, s2, s3, bw;
        asm("add.cc.u32  %0, %4, %8;\n\t"
            "addc.cc.u32 %1, %5, %9;\n\t"
            "addc.cc.u32 %2, %6, %10;\n\t"
            "addc.u32    %3, %7, 0;"
            : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
            : "r"(o0), "r"(o1), "r"(o2), "r"(o3), "r"(e1), "r"(e2), "r"(e3));
        asm("sub.cc.u32  %0, %5, %9;\n\t"
            "subc.cc.u32 %1, %6, %10;\n\t"
            "subc.cc.u32 %2, %7, %11;\n\t"
            "subc.cc.u32 %3, %8, %12;\n\t"
            "subc.u32    %4, 0, 0;"
            : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3), "=r"(bw)
            : "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(n0), "r"(n1), "r"(n2), "r"(n3));
        const bool keep = bw != 0;
        r->v[0] = keep ? t0 : s0; r->v[1] = keep ? t1 : s1;
        r->v[2] = keep ? t2 : s2; r->v[3] = keep ? t3 : s3;
    }
}

/* The fit tests and mfb bounds (CF_WIDTH_BITS, bench.h) and these products
 * must agree on what each width takes; changing either alone fails here. */
static_assert(CF_WIDTH_BITS(3) == 96u,  "mz_mul<3> takes any n < 2^96 (its carry word h)");
static_assert(CF_WIDTH_BITS(4) == 127u, "mz_mul<4> needs n < 2^127 (THE BOUND)");

template <>
CF_FN void mz_sub_mod<4>(mz<4> *a, const mz<4> *b, const mz<4> *n)
{
    uint32_t d0, d1, d2, d3, m;
    asm("sub.cc.u32  %0, %5, %9;\n\t"
        "subc.cc.u32 %1, %6, %10;\n\t"
        "subc.cc.u32 %2, %7, %11;\n\t"
        "subc.cc.u32 %3, %8, %12;\n\t"
        "subc.u32    %4, 0, 0;"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3), "=r"(m)
        : "r"(a->v[0]), "r"(a->v[1]), "r"(a->v[2]), "r"(a->v[3]),
          "r"(b->v[0]), "r"(b->v[1]), "r"(b->v[2]), "r"(b->v[3]));
    asm("add.cc.u32  %0, %0, %4;\n\t"
        "addc.cc.u32 %1, %1, %5;\n\t"
        "addc.cc.u32 %2, %2, %6;\n\t"
        "addc.u32    %3, %3, %7;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(n->v[0] & m), "r"(n->v[1] & m), "r"(n->v[2] & m), "r"(n->v[3] & m));
    a->v[0] = d0; a->v[1] = d1; a->v[2] = d2; a->v[3] = d3;
}

template <>
CF_FN void mz_add_mod<4>(mz<4> *a, const mz<4> *b, const mz<4> *n)
{
    uint32_t s0, s1, s2, s3, s4, r0, r1, r2, r3, bw;
    asm("add.cc.u32  %0, %5, %9;\n\t"
        "addc.cc.u32 %1, %6, %10;\n\t"
        "addc.cc.u32 %2, %7, %11;\n\t"
        "addc.cc.u32 %3, %8, %12;\n\t"
        "addc.u32    %4, 0, 0;"
        : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3), "=r"(s4)
        : "r"(a->v[0]), "r"(a->v[1]), "r"(a->v[2]), "r"(a->v[3]),
          "r"(b->v[0]), "r"(b->v[1]), "r"(b->v[2]), "r"(b->v[3]));
    asm("sub.cc.u32  %0, %5, %9;\n\t"
        "subc.cc.u32 %1, %6, %10;\n\t"
        "subc.cc.u32 %2, %7, %11;\n\t"
        "subc.cc.u32 %3, %8, %12;\n\t"
        "subc.u32    %4, %13, 0;"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(bw)
        : "r"(s0), "r"(s1), "r"(s2), "r"(s3),
          "r"(n->v[0]), "r"(n->v[1]), "r"(n->v[2]), "r"(n->v[3]), "r"(s4));
    const bool keep = bw != 0;
    a->v[0] = keep ? s0 : r0; a->v[1] = keep ? s1 : r1;
    a->v[2] = keep ? s2 : r2; a->v[3] = keep ? s3 : r3;
}

template <>
CF_FN void mz_dbl_mod<4>(mz<4> *a, const mz<4> *n)
{
    const mz<4> b = *a;
    mz_add_mod<4>(a, &b, n);
}
#endif

#endif /* CUDA_SIEVE_COFAC_ARITH_CUH */
