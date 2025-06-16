#pragma once

__host__ __device__ static inline size_t rb_idx(size_t x, size_t y, size_t dim) {
    size_t base = ((x % 2) ^ (y % 2)) * dim * (dim / 2);
    size_t offset = (x / 2) + y * (dim / 2);
    return base + offset;
}

__host__ __device__ static inline size_t idx(size_t x, size_t y, size_t stride) {
    return x + y * stride;
}

