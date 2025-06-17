/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

	This code is a simple prototype that demonstrates how to use the
	code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
	for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <stdio.h>
#include <stdlib.h>

#include "indices.h"
#include "solver.h"
#include "wtime.h"

/* macros */

#ifndef GRID_SIZE
#define GRID_SIZE 128
#endif

#define IX(x,y) (rb_idx((x),(y),(N+2)))

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;
static float max_cells_per_ms;

static float *h_u, *h_v, *h_u_prev, *h_v_prev;
static float *d_u, *d_v, *d_u_prev, *d_v_prev;
static float *h_dens, *h_dens_prev;
static float *d_dens, *d_dens_prev;


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


static void free_data(void)
{
    if (h_u) free(h_u);
    if (h_v) free(h_v);
    if (h_u_prev) free(h_u_prev);
    if (h_v_prev) free(h_v_prev);
    if (h_dens) free(h_dens);
    if (h_dens_prev) free(h_dens_prev);
    if (d_u) cudaFree(d_u);
    if (d_v) cudaFree(d_v);
    if (d_u_prev) cudaFree(d_u_prev);
    if (d_v_prev) cudaFree(d_v_prev);
    if (d_dens) cudaFree(d_dens);
    if (d_dens_prev) cudaFree(d_dens_prev);
}

static void clear_data(void)
{
    int i, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        h_u[i] = h_v[i] = h_u_prev[i] = h_v_prev[i] = h_dens[i] = h_dens_prev[i] = 0.0f;
    }
}

static int allocate_data(void)
{
    int size = (N + 2) * (N + 2);

    h_u = (float*)malloc(size * sizeof(float));
    h_v = (float*)malloc(size * sizeof(float));
    h_u_prev = (float*)malloc(size * sizeof(float));
    h_v_prev = (float*)malloc(size * sizeof(float));
    h_dens = (float*)malloc(size * sizeof(float));
    h_dens_prev = (float*)malloc(size * sizeof(float));

    if (!h_u || !h_v || !h_u_prev || !h_v_prev || !h_dens || !h_dens_prev) {
        fprintf(stderr, "cannot allocate data\n");
        return (0);
    }

    cudaError_t err = cudaMalloc((void **)&d_u, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc((void **)&d_v, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc((void **)&d_u_prev, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc((void **)&d_v_prev, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc((void **)&d_dens, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc((void **)&d_dens_prev, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    return (1);
}


static void react(float* d, float* u, float* v)
{
    int i, size = (N + 2) * (N + 2);
    float max_velocity2 = 0.0f;
    float max_density = 0.0f;

    max_velocity2 = max_density = 0.0f;
    for (i = 0; i < size; i++) {
        if (max_velocity2 < u[i] * u[i] + v[i] * v[i]) {
            max_velocity2 = u[i] * u[i] + v[i] * v[i];
        }
        if (max_density < d[i]) {
            max_density = d[i];
        }
    }

    for (i = 0; i < size; i++) {
        u[i] = v[i] = d[i] = 0.0f;
    }

    if (max_velocity2 < 0.0000005f) {
        u[IX(N / 2, N / 2)] = force * 10.0f;
        v[IX(N / 2, N / 2)] = force * 10.0f;
    }
    if (max_density < 1.0f) {
        d[IX(N / 2, N / 2)] = source * 10.0f;
    }

    return;
}

static void one_step(void)
{
    static int times = 1;
    static double start_t = 0.0;
    static double one_second = 0.0;
    static double react_ms_p_cell = 0.0;
    static double vel_ms_p_cell = 0.0;
    static double dens_ms_p_cell = 0.0;
    static double init_memcpy_ms_p_cell = 0.0;
    static double end_memcpy_ms_p_cell = 0.0;
    size_t size = (N + 2) * (N + 2);

    start_t = wtime();
    react(h_dens_prev, h_u_prev, h_v_prev);
    react_ms_p_cell += 1.0e3 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    cudaMemcpy(d_u, h_u, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_prev, h_u_prev, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_prev, h_v_prev, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens, h_dens, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens_prev, h_dens_prev, size * sizeof(float), cudaMemcpyHostToDevice);
    init_memcpy_ms_p_cell += 1.0e3 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    vel_step(N, d_u, d_v, d_u_prev, d_v_prev, visc, dt);
    vel_ms_p_cell += 1.0e3 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    dens_step(N, d_dens, d_dens_prev, d_u, d_v, diff, dt);
    dens_ms_p_cell += 1.0e3 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    cudaMemcpy(h_u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_u_prev, d_u_prev, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v_prev, d_v_prev, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dens, d_dens, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dens_prev, d_dens_prev, size * sizeof(float), cudaMemcpyDeviceToHost);
    end_memcpy_ms_p_cell += 1.0e3 * (wtime() - start_t) / (N * N);

    if (1.0 < wtime() - one_second) { /* at least 1s between stats */
        max_cells_per_ms = max(max_cells_per_ms, 
                                times / (react_ms_p_cell + vel_ms_p_cell + dens_ms_p_cell + 
                                         init_memcpy_ms_p_cell + end_memcpy_ms_p_cell));
        //printf("cells per ms: %lf\n", 
        //       times / (react_ms_p_cell + vel_ms_p_cell + dens_ms_p_cell));
        one_second = wtime();
        react_ms_p_cell = 0.0;
        init_memcpy_ms_p_cell = 0.0;
        vel_ms_p_cell = 0.0;
        dens_ms_p_cell = 0.0;
        end_memcpy_ms_p_cell = 0.0;
        times = 1;
    } else {
        times++;
    }
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char** argv)
{
    int i = 0;

    if (argc != 1 && argc != 7) {
        fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
        fprintf(stderr, "where:\n");
        fprintf(stderr, "\t N      : grid resolution\n");
        fprintf(stderr, "\t dt     : time step\n");
        fprintf(stderr, "\t diff   : diffusion rate of the density\n");
        fprintf(stderr, "\t visc   : viscosity of the fluid\n");
        fprintf(stderr, "\t force  : scales the mouse movement that generate a force\n");
        fprintf(stderr, "\t source : amount of density that will be deposited\n");
        exit(1);
    }

    if (argc == 1) {
        N = GRID_SIZE;
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        fprintf(stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
                N, dt, diff, visc, force, source);
    } else {
        N = atoi(argv[1]);
        dt = atof(argv[2]);
        diff = atof(argv[3]);
        visc = atof(argv[4]);
        force = atof(argv[5]);
        source = atof(argv[6]);
    }

    if (!allocate_data()) {
        exit(1);
    }
    clear_data();
    float start_time = wtime();

    printf("llegue aca\n");
    for (i = 0; i < 2048; i++) {
        one_step();
    }
    printf("%lf\n", max_cells_per_ms);
    printf("%lf\n", wtime() - start_time);
    free_data();

    exit(0);
}
