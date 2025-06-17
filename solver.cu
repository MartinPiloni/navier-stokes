#include <stddef.h>
#include <stdio.h>

#include "solver.h"
#include "indices.h"

#define IX(x,y) (rb_idx((x),(y),(n+2)))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;


__global__ void add_source_kernel(unsigned int n, float* x, float* s, float dt)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (n + 2) * (n + 2)) {
        return;
    }
    x[tid] += dt * s[tid];
}

void add_source(unsigned int n, float* x, float* s, float dt)
{
    size_t size = (n + 2) * (n + 2);
    size_t threads_per_block = 256; 
    size_t blocks = (size + threads_per_block - 1) / threads_per_block;
    add_source_kernel<<<blocks, threads_per_block>>>(n, x, s, dt);
}

__global__ void set_bnd_kernel(unsigned int n, boundary b, float * x)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (tid <= n) {
        x[IX(0, tid)]     = b == VERTICAL ? -x[IX(1, tid)] : x[IX(1, tid)];
        x[IX(n + 1, tid)] = b == VERTICAL ? -x[IX(n, tid)] : x[IX(n, tid)];
        x[IX(tid, 0)]     = b == HORIZONTAL ? -x[IX(tid, 1)] : x[IX(tid, 1)];
        x[IX(tid, n + 1)] = b == HORIZONTAL ? -x[IX(tid, n)] : x[IX(tid, n)];
    }

    __syncthreads();
    if (blockIdx.x + threadIdx.x > 0) return;
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void set_bnd(unsigned int n, boundary b, float * x)
{
    size_t threads_per_block = 256;
    size_t blocks = (n + threads_per_block - 1) / threads_per_block;
    set_bnd_kernel<<<blocks, threads_per_block>>>(n, b, x);
}

__global__ void lin_solve_rb_step_kernel(grid_color color,
                                         unsigned int n,
                                         float a,
                                         float c,
                                         const float* same0,
                                         const float* neigh,
                                         float* same)
{
    unsigned int width = (n + 2) / 2;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // shift
    // red, y impar -> 1
    // negro, y impar -> -1
    // red, y par -> -1
    // negro, y par -> 1
    int shift = (y & 1) ? ((color == RED) ? 1 : -1) : ((color == RED) ? -1 : 1);

    // start
    // red, y impar -> 0
    // red, y par -> 1
    // negro, y impar -> 1
    // negro, y par -> 0
    int start = (color == RED) ^ (y & 1);

    if (y > n) return;
    if (x < start || x >= width - (1 - start)) {
        return;
    }
    int index = idx(x, y, width);
    same[index] = (same0[index] + a * (neigh[index - width] +
                                       neigh[index] +
                                       neigh[index + shift] +
                                       neigh[index + width])) / c;
}

void lin_solve(unsigned int n, boundary b,
                    float* x, float* x0,
                    float a, float c)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1 )/ threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    float* red0 = x0;
    float* blk0 = x0 + color_size;
    float* red  = x;
    float* blk  = x + color_size;
    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step_kernel<<<numBlocks, threadsPerBlock>>>(RED, n, a, c, red0, blk, red);
        cudaDeviceSynchronize(); 

        lin_solve_rb_step_kernel<<<numBlocks, threadsPerBlock>>>(BLACK, n, a, c, blk0, red, blk);
        cudaDeviceSynchronize();

        set_bnd(n, b, x);
        cudaDeviceSynchronize();	
    }
}

static void diffuse(unsigned int n, boundary b, float* x, float* x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

__global__ void advect_kernel(unsigned int n, boundary b, float* d, float* d0, float* u, float* v, float dt)
{
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;

    float dt0 = dt * n;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > n || j > n) return;
    x = i - dt0 * u[IX(i, j)];
    y = j - dt0 * v[IX(i, j)];
    if (x < 0.5f) {
        x = 0.5f;
    } else if (x > n + 0.5f) {
        x = n + 0.5f;
    }
    i0 = (int)x;
    i1 = i0 + 1;
    if (y < 0.5f) {
        y = 0.5f;
    } else if (y > n + 0.5f) {
        y = n + 0.5f;
    }
    j0 = (int)y;
    j1 = j0 + 1;
    s1 = x - i0;
    s0 = 1 - s1;
    t1 = y - j0;
    t0 = 1 - t1;
    d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
}


void advect(unsigned int n, boundary b, float* d, float* d0, float* u, float* v, float dt) {
    dim3 threads_per_block(16,16);
    dim3 blocks((n + threads_per_block.x - 1) / threads_per_block.x, (n + threads_per_block.y) / threads_per_block.y);
    advect_kernel<<<blocks, threads_per_block>>>(n, b, d, d0, u, v, dt);
    cudaDeviceSynchronize();
    set_bnd(n, b, d);
    cudaDeviceSynchronize();
}

__global__ void project_step1_kernel(unsigned int n, float* u, float* v, float* p, float* div) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i > n || j > n) return;

    div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / n;
    p[IX(i, j)] = 0;
}

__global__ void project_step2_kernel(unsigned int n, float* u, float* v, float* p) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i > n || j > n) return;

    u[IX(i, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
    v[IX(i, j)] -= 0.5f * n * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
}


static void project(unsigned int n, float* u, float* v, float* p, float* div)
{
    dim3 threads_per_block(16, 16);
    dim3 blocks((n + threads_per_block.x - 1) / threads_per_block.x, (n + threads_per_block.y - 1) / threads_per_block.y);

    project_step1_kernel<<<blocks, threads_per_block>>>(n, u, v, p, div);
    cudaDeviceSynchronize();

    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);
    cudaDeviceSynchronize();

    lin_solve(n, NONE, p, div, 1, 4);

    project_step2_kernel<<<blocks, threads_per_block>>>(n, u, v, p);
    cudaDeviceSynchronize();

    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
    cudaDeviceSynchronize();
}

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    cudaDeviceSynchronize();
    SWAP(x0, x)
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
    cudaDeviceSynchronize();
    SWAP(u0, u);
    diffuse(n, VERTICAL, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(n, HORIZONTAL, v, v0, visc, dt);
    project(n, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(n, VERTICAL, u, u0, u0, v0, dt);
    advect(n, HORIZONTAL, v, v0, u0, v0, dt);
    project(n, u, v, u0, v0);
}
