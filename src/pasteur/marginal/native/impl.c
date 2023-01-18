#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>

#define ALG 32

#define INSPECT(__var, __msg)               \
    {                                       \
        uint16_t k;                         \
        k = _mm256_extract_epi16(__var, 0); \
        printf("%s: %d\n", __msg, k);       \
    }

static inline void sum_inline_u32(
    uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16,
    int n_u32, int *mul_u32, uint32_t **arr_u32)
{
    uint32_t exp_u8[200] __attribute__((aligned(ALG)));
    for (int i = 0; i < n_u8; i += 1)
    {
        for (int j = 0; j < 1 << 3; j += 1)
        {
            exp_u8[(i << 3) + j] = (uint32_t)mul_u8[i];
        }
    }
    uint32_t exp_u16[200] __attribute__((aligned(ALG)));
    for (int i = 0; i < n_u16; i += 1)
    {
        for (int j = 0; j < 1 << 3; j += 1)
        {
            exp_u16[(i << 3) + j] = (uint32_t)mul_u16[i];
        }
    }
    uint32_t exp_u32[200] __attribute__((aligned(ALG)));
    for (int i = 0; i < n_u32; i += 1)
    {
        for (int j = 0; j < 1 << 3; j += 1)
        {
            exp_u32[(i << 3) + j] = (uint32_t)mul_u32[i];
        }
    }

    for (uint64_t i = 0; i < l - 7; i += 1 << 3)
    {
        __m256i tmp, mul;
        __m256i idx = _mm256_setzero_si256();

        for (int j = 0; j < n_u8; j += 1)
        {
            __m128i load = _mm_loadl_epi64((__m128i *)&arr_u8[j][i]);
            tmp = _mm256_cvtepu8_epi32(load);
            mul = _mm256_load_si256((__m256i *)&exp_u8[j << 3]);
            tmp = _mm256_mullo_epi32(tmp, mul);
            idx = _mm256_add_epi32(tmp, idx);
        }

        for (int j = 0; j < n_u16; j += 1)
        {
            __m128i load = _mm_lddqu_si128((__m128i *)&arr_u16[j][i]);
            tmp = _mm256_cvtepu16_epi32(load);
            mul = _mm256_load_si256((__m256i *)&exp_u16[j << 3]);
            tmp = _mm256_mullo_epi32(tmp, mul);
            idx = _mm256_add_epi32(tmp, idx);
        }

        for (int j = 0; j < n_u32; j += 1)
        {
            tmp = _mm256_lddqu_si256((__m256i *)&arr_u32[j][i]);
            mul = _mm256_load_si256((__m256i *)&exp_u32[j << 3]);
            tmp = _mm256_mullo_epi32(tmp, mul);
            idx = _mm256_add_epi32(tmp, idx);
        }

        uint32_t k;
        k = _mm256_extract_epi32(idx, 0);
        out[k] += 1;
        k = _mm256_extract_epi32(idx, 1);
        out[k] += 1;
        k = _mm256_extract_epi32(idx, 2);
        out[k] += 1;
        k = _mm256_extract_epi32(idx, 3);
        out[k] += 1;
        k = _mm256_extract_epi32(idx, 4);
        out[k] += 1;
        k = _mm256_extract_epi32(idx, 5);
        out[k] += 1;
        k = _mm256_extract_epi32(idx, 6);
        out[k] += 1;
        k = _mm256_extract_epi32(idx, 7);
        out[k] += 1;
    }
}

static inline void sum_inline_u16(
    uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16)
{
    uint16_t exp_u8[200] __attribute__((aligned(ALG)));
    for (int i = 0; i < n_u8; i += 1)
    {
        for (int j = 0; j < 1 << 4; j += 1)
        {
            exp_u8[(i << 4) + j] = (uint16_t)mul_u8[i];
        }
    }
    uint16_t exp_u16[200] __attribute__((aligned(ALG)));
    for (int i = 0; i < n_u16; i += 1)
    {
        for (int j = 0; j < 1 << 4; j += 1)
        {
            exp_u16[(i << 4) + j] = (uint16_t)mul_u16[i];
        }
    }

    for (uint64_t i = 0; i < l - 15; i += 1 << 4)
    {
        __m256i tmp, mul;
        __m256i idx = _mm256_setzero_si256();

        for (int j = 0; j < n_u8; j += 1)
        {
            __m128i load = _mm_lddqu_si128((__m128i *)&arr_u8[j][i]);
            tmp = _mm256_cvtepu8_epi16(load);
            mul = _mm256_load_si256((__m256i *)&exp_u8[j << 4]);
            tmp = _mm256_mullo_epi16(tmp, mul);
            idx = _mm256_adds_epu16(tmp, idx);
        }

        for (int j = 0; j < n_u16; j += 1)
        {
            tmp = _mm256_lddqu_si256((__m256i *)&arr_u16[j][i]);
            mul = _mm256_load_si256((__m256i *)&exp_u16[j << 4]);
            tmp = _mm256_mullo_epi16(tmp, mul);
            idx = _mm256_adds_epu16(tmp, idx);
        }

        uint16_t k;
        k = _mm256_extract_epi16(idx, 0);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 1);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 2);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 3);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 4);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 5);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 6);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 7);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 8);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 9);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 10);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 11);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 12);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 13);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 14);
        out[k] += 1;
        k = _mm256_extract_epi16(idx, 15);
        out[k] += 1;
    }
}

static inline void sum_inline_u16_nonsimd(
    uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16)
{
    uint16_t idx;
    for (uint64_t i = 0; i < l; i++)
    {
        idx = 0;
        for (int j = 0; j < n_u8; j++)
        {
            idx += (uint16_t)arr_u8[j][i] * (uint16_t)mul_u8[j];
        }
        for (int j = 0; j < n_u16; j++)
        {
            idx += arr_u16[j][i] * (uint16_t)mul_u16[j];
        }
        out[idx] += 1;
    }
}

static inline void sum_inline_u32_nonsimd(
    uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16,
    int n_u32, int *mul_u32, uint32_t **arr_u32)
{
    unsigned int idx;
    for (uint64_t i = 0; i < l; i++)
    {
        idx = 0;
        for (int j = 0; j < n_u8; j++)
        {
            idx += (unsigned int)arr_u8[j][i] * mul_u8[j];
        }
        for (int j = 0; j < n_u16; j++)
        {
            idx += (unsigned int)arr_u16[j][i] * mul_u16[j];
        }
        for (int j = 0; j < n_u32; j++)
        {
            idx += (unsigned int)arr_u32[j][i] * mul_u32[j];
        }
        out[idx] += 1;
    }
}

#define SUM_INLINE(_i, _j, _k)                                                                  \
    if (dom > 65536 || n_u32 > 0)                                                               \
    {                                                                                           \
        sum_inline_u32(l, out, _i, mul_u8, arr_u8, _j, mul_u16, arr_u16, _k, mul_u32, arr_u32); \
    }                                                                                           \
    else                                                                                        \
    {                                                                                           \
        sum_inline_u16(l, out, _i, mul_u8, arr_u8, _j, mul_u16, arr_u16);                       \
    }

#define SUM_CASE(i, j, k) (400 * i + 20 * j + k)
#define SUM_INLINE_CASE(_i, _j, _k) \
    case SUM_CASE(_i, _j, _k):      \
        SUM_INLINE(_i, _j, _k);     \
        break;

#define SUM_RECUR_K(i, j)    \
    SUM_INLINE_CASE(i, j, 0) \
    SUM_INLINE_CASE(i, j, 1) \
    SUM_INLINE_CASE(i, j, 2) \
    SUM_INLINE_CASE(i, j, 3) \
    SUM_INLINE_CASE(i, j, 4)

#define SUM_RECUR_J(i) \
    SUM_RECUR_K(i, 0)  \
    SUM_RECUR_K(i, 1)  \
    SUM_RECUR_K(i, 2)  \
    SUM_RECUR_K(i, 3)  \
    SUM_RECUR_K(i, 4)

#define SUM_RECUR  \
    SUM_RECUR_J(0) \
    SUM_RECUR_J(1) \
    SUM_RECUR_J(2) \
    SUM_RECUR_J(3) \
    SUM_RECUR_J(4)

#ifndef _DEBUG

void sum_non_simd(
    uint64_t dom, uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16,
    int n_u32, int *mul_u32, uint32_t **arr_u32)
{
    if (dom > 65536 || n_u32 > 0)
    {
        sum_inline_u32_nonsimd(l, out, n_u8, mul_u8, arr_u8, n_u16, mul_u16, arr_u16, n_u32, mul_u32, arr_u32);
    }
    else
    {
        sum_inline_u16_nonsimd(l, out, n_u8, mul_u8, arr_u8, n_u16, mul_u16, arr_u16);
    }
}

#ifdef __AVX2__
void sum(
    uint64_t dom, uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16,
    int n_u32, int *mul_u32, uint32_t **arr_u32)
{
    // SUM_INLINE(n_u8, n_u16, n_u32)
    switch (SUM_CASE(n_u8, n_u16, n_u32))
    {
        SUM_RECUR
    default:
        SUM_INLINE(n_u8, n_u16, n_u32)
        break;
    }
}
#else
#warning AX2 not supported by the current platform. Code will be compiled without SIMD.
void sum(
    uint64_t dom, uint64_t l, uint32_t *out,
    int n_u8, int *mul_u8, uint8_t **arr_u8,
    int n_u16, int *mul_u16, uint16_t **arr_u16,
    int n_u32, int *mul_u32, uint32_t **arr_u32)
{
    sum_non_simd(dom, l, out, n_u8, mul_u8, arr_u8, n_u16, mul_u16, arr_u16, n_u32, mul_u32, arr_u32);
}
#endif

#else

#define N 10000000

static uint32_t out[N] __attribute__((aligned(ALG)));
static uint8_t a1[N] __attribute__((aligned(ALG)));
static uint16_t a2[N] __attribute__((aligned(ALG)));

int main(int argc, char *argv[])
{
    for (int i = 0; i < N; i++)
    {
        a1[i] = i % 255;
        a2[i] = (i + 5) % 255;
        out[i] = 0;
    }

    int mul_u8[] = {1};
    uint8_t *arr_u8[] = {a1};
    int mul_u16[] = {256};
    uint16_t *arr_u16[] = {a2};

    sum_inline_u32(N, out, 1, mul_u8, arr_u8, 1, mul_u16, arr_u16, 0, NULL, NULL);

    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            printf("%d ", out[256 * i + j]);
        }
        printf("\n");
    }
    return 1;
}
#endif