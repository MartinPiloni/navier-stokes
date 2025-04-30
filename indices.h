#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <stdio.h>
#include <immintrin.h>

static int color_offset(int n) {
    return (8 - ((n/2 + 1)%8)) % 8;
}

static inline size_t rb_idx(size_t x, size_t y, size_t dim) {
    size_t base = ((x % 2) ^ (y % 2)) * dim * (dim / 2 + color_offset(dim-2));
    size_t offset = (x / 2) + y * ((dim / 2) + color_offset(dim-2));
    return base + offset;
}

static inline __m256i rb_idx_m256i(__m256i x ,__m256i y, size_t dim) {
    __m256i dimV = _mm256_set1_epi32(dim);
    __m256i dim2V = _mm256_set1_epi32(color_offset(dim-2));
    dim2V = _mm256_add_epi32(dim2V, _mm256_srai_epi32(dimV, 1));
    __m256i one = _mm256_set1_epi32(1);
    __m256i color = _mm256_xor_si256(_mm256_and_si256(x, one), _mm256_and_si256(y, one));
    __m256i base = _mm256_mullo_epi32(dim2V, _mm256_mullo_epi32(color, dimV));
    __m256i aux = _mm256_mullo_epi32(y, dim2V);
    __m256i offset = _mm256_add_epi32(_mm256_srai_epi32(x, 1), aux);
    return _mm256_add_epi32(base, offset);
}

static inline size_t idx(size_t x, size_t y, size_t stride) {
    return x + y * stride;
}

#pragma GCC diagnostic pop
