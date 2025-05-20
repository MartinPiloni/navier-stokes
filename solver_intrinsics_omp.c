#include <stddef.h>
#include <immintrin.h>

#include "solver.h"
#include "indices.h"

#define IX(x,y) (rb_idx((x),(y),(n+2)))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

static void add_source(unsigned int n, float* x, const float* s, float dt)
{
    for (unsigned int i = 0; i < (n+2)*(n+2 + 2*color_offset(n)); i+=8) {
        __m256 xv = _mm256_load_ps(&x[i]);
        __m256 sv = _mm256_load_ps(&s[i]);
        __m256 avx_dt = _mm256_set1_ps(dt);
        xv = _mm256_fmadd_ps(avx_dt, sv, xv);
        _mm256_store_ps(&x[i], xv);
    }
}

static void set_bnd(unsigned int n, boundary b, float * x)
{
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)]     = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)]     = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void lin_solve_rb_step(grid_color color,
                              unsigned int n,
                              float a,
                              float c,
                              const float * restrict same0,
                              const float * restrict neigh,
                              float * restrict same)
{
    
    unsigned int y = 1;
    #pragma omp parallel for private(y)
    for (y = 1; y <= n; ++y) {
        const __m256 avx_a = _mm256_set1_ps(a);
        const __m256 avx_c = _mm256_set1_ps(c);
        
        unsigned int width = (n + 2) / 2;
        
        unsigned int coff = color_offset(n);
        unsigned int start = ((color == RED ? 0 : 1)+y+1)%2;
        unsigned int x = 0;
        int index = idx(x, y, width + coff);
        __m256 weast  = _mm256_load_ps(&neigh[index]);   // west & east
        __m256 next   = _mm256_load_ps(&neigh[index+8]); // next
        __m256 s = _mm256_load_ps(&same[index]);
        
        // Cargar datos
        __m256 s0   = _mm256_load_ps(&same0[index]);
        __m256 north   = _mm256_load_ps(&neigh[index - (width + coff)]);   // north
        __m256 south   = _mm256_load_ps(&neigh[index + (width + coff)]);   // south

        __m256 aux = _mm256_undefined_ps();
        __m256 sum = _mm256_undefined_ps();
        if (start) {
            // Operaciones
            __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
            aux = _mm256_permutevar8x32_ps(weast, perm_idx);
            sum = _mm256_add_ps(weast, aux);
            sum = _mm256_add_ps(sum, north);
            sum = _mm256_add_ps(sum, south);
            sum = _mm256_mul_ps(sum, avx_a);
            sum = _mm256_add_ps(sum, s0);
            sum = _mm256_div_ps(sum, avx_c);
            sum = _mm256_blend_ps(sum, s, 1);  // border case
        } else {
            aux  = _mm256_blend_ps(weast, next, 1);
            __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
            aux = _mm256_permutevar8x32_ps(aux, perm_idx);
            sum = _mm256_add_ps(weast, aux);
            sum = _mm256_add_ps(sum, north);
            sum = _mm256_add_ps(sum, south);
            sum = _mm256_mul_ps(sum, avx_a);
            sum = _mm256_add_ps(sum, s0);
            sum = _mm256_div_ps(sum, avx_c);
        }

        // Guardar resultado
        _mm256_store_ps(&same[index], sum);

        for (x = 8; x+8 < width + coff; x += 8) {
            index = idx(x, y, width + coff);
            
            __m256 s0   = _mm256_load_ps(&same0[index]);
            __m256 north   = _mm256_load_ps(&neigh[index - (width + coff)]);   // north
            __m256 south   = _mm256_load_ps(&neigh[index + (width + coff)]);   // south
            
            __m256 aux = _mm256_undefined_ps();
            if (start) {
                aux  = _mm256_blend_ps(next, weast, 1<<7);
                __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                weast   = next;
                next = _mm256_load_ps(&neigh[index+8]); // next
            } else {
                weast   = next;
                next = _mm256_load_ps(&neigh[index+8]); // next
                aux  = _mm256_blend_ps(weast, next, 1);
                __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                aux = _mm256_permutevar8x32_ps(aux, perm_idx);
            }

            __m256 sum = _mm256_add_ps(weast, aux);
            sum = _mm256_add_ps(sum, north);
            sum = _mm256_add_ps(sum, south);
            sum = _mm256_mul_ps(sum, avx_a);
            sum = _mm256_add_ps(sum, s0);
            sum = _mm256_div_ps(sum, avx_c);

            // Guardar resultado
            _mm256_store_ps(&same[index], sum);
        }

        if (x < width - (1-start)){
            index = idx(x, y, width + coff);
                
            __m256 s0   = _mm256_load_ps(&same0[index]);
            __m256 north   = _mm256_load_ps(&neigh[index - (width + coff)]);   // north
            __m256 south   = _mm256_load_ps(&neigh[index + (width + coff)]);   // south
            __m256 s = _mm256_load_ps(&same[index]);
            
            __m256 aux = _mm256_undefined_ps();
            if (start) {
                aux  = _mm256_blend_ps(next, weast, 1<<7);
                __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                weast   = next;
            } else {
                weast   = next;
                __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                aux = _mm256_permutevar8x32_ps(weast, perm_idx);
            }

            __m256 sum = _mm256_add_ps(weast, aux);
            sum = _mm256_add_ps(sum, north);
            sum = _mm256_add_ps(sum, south);
            sum = _mm256_mul_ps(sum, avx_a);
            sum = _mm256_add_ps(sum, s0);
            sum = _mm256_div_ps(sum, avx_c);
            switch (width-(1-start)-x){
                case 1:
                    sum = _mm256_blend_ps(s, sum, (1<<1)-1);
                    break;
                case 2:
                    sum = _mm256_blend_ps(s, sum, (1<<2)-1);
                    break;
                case 3:
                    sum = _mm256_blend_ps(s, sum, (1<<3)-1);
                    break;
                case 4:
                    sum = _mm256_blend_ps(s, sum, (1<<4)-1);
                    break;
                case 5:
                    sum = _mm256_blend_ps(s, sum, (1<<5)-1);
                    break;
                case 6:
                    sum = _mm256_blend_ps(s, sum, (1<<6)-1);
                    break;
                case 7:
                    sum = _mm256_blend_ps(s, sum, (1<<7)-1);
                    break;
                default:
                    break;
            }

            // Guardar resultado
            _mm256_store_ps(&same[index], sum);
        }
    }
}

static void lin_solve(unsigned int n, boundary b,
                      float * restrict x,
                      const float * restrict x0,
                      float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2 + color_offset(n));
    const float * red0 = x0;
    const float * blk0 = x0 + color_size;
    float * red = x;
    float * blk = x + color_size;

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step(RED,   n, a, c, red0, blk, red);
        lin_solve_rb_step(BLACK, n, a, c, blk0, red, blk);
        set_bnd(n, b, x);
    }
}

static void diffuse(unsigned int n, boundary b, float* x, const float* x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect(unsigned int n, boundary b, float* d, const float* d0, const float* u, const float* v, float dt)
{
    for (unsigned int half = 0; half <= 1; half++) {
        unsigned int y = 1;
        #pragma omp parallel for private(y)
        for (y = 1; y <= n; y++) {
            unsigned int start = (half+y+1)%2;
            unsigned int width = (n+2)/2;
            unsigned int coff = color_offset(n);
            unsigned int mid = (n+2)*(width + coff);
            __m256i i0, i1, j0, j1;
            __m256 xV, yV, s0, t0, s1, t1;

            __m256 dt0 = _mm256_set1_ps((-dt)*n);
            __m256 c1 = _mm256_set1_ps(0.5f);
            __m256 c2 = _mm256_set1_ps(n+0.5f);
            __m256i one = _mm256_set1_epi32(1);
            __m256 one_ps = _mm256_set1_ps(1.0f);
            unsigned int x = 0;
            unsigned int index = mid*half + idx(x, y, width + coff);

            __m256 jV = _mm256_set1_ps(y);
            __m256 iV = _mm256_set_ps(2.0f*(x+7)+(1-start), 2.0f*(x+6)+(1-start), 2.0f*(x+5)+(1-start), 2.0f*(x+4)+(1-start),
                                    2.0f*(x+3)+(1-start), 2.0f*(x+2)+(1-start), 2.0f*(x+1)+(1-start), 2.0f*x+(1-start));

            __m256 uV = _mm256_load_ps(&u[index]);
            __m256 vV = _mm256_load_ps(&v[index]);

            xV = _mm256_fmadd_ps(dt0, uV, iV);
            yV = _mm256_fmadd_ps(dt0, vV, jV);

            xV = _mm256_max_ps(xV, c1);
            xV = _mm256_min_ps(xV, c2);
            __m256 aux = _mm256_floor_ps(xV);
            i0 = _mm256_cvtps_epi32(aux);
            i1 = _mm256_add_epi32(i0, one);

            yV = _mm256_max_ps(yV, c1);
            yV = _mm256_min_ps(yV, c2);
            aux = _mm256_floor_ps(yV);
            j0 = _mm256_cvtps_epi32(aux);
            j1 = _mm256_add_epi32(j0, one);

            s1 = _mm256_sub_ps(xV, _mm256_cvtepi32_ps(i0));
            s0 = _mm256_sub_ps(one_ps, s1);
            t1 = _mm256_sub_ps(yV, _mm256_cvtepi32_ps(j0));
            t0 = _mm256_sub_ps(one_ps, t1);

            __m256 aux0 = _mm256_i32gather_ps(d0, rb_idx_m256i(i0, j0, n+2), 4);
            __m256 aux1 = _mm256_i32gather_ps(d0, rb_idx_m256i(i0, j1, n+2), 4);
            __m256 aux2 = _mm256_i32gather_ps(d0, rb_idx_m256i(i1, j0, n+2), 4);
            __m256 aux3 = _mm256_i32gather_ps(d0, rb_idx_m256i(i1, j1, n+2), 4);
            aux0 = _mm256_mul_ps(t0, aux0);
            aux0 = _mm256_fmadd_ps(t1, aux1, aux0);
            aux2 = _mm256_mul_ps(t0, aux2);
            aux2 = _mm256_fmadd_ps(t1, aux3, aux2);
            aux0 = _mm256_mul_ps(s0, aux0);
            aux0 = _mm256_fmadd_ps(s1, aux2, aux0);

            if (start) {
                __m256 dV = _mm256_load_ps(&d[index]);
                aux0 = _mm256_blend_ps(aux0, dV, 1);
            }

            _mm256_store_ps(&d[index], aux0);

            for (x = 8; x+8 < width + coff; x+=8) {
                index = mid*half + idx(x, y, width + coff);
                __m256 iV = _mm256_set_ps(2.0f*(x+7)+(1-start), 2.0f*(x+6)+(1-start), 2.0f*(x+5)+(1-start), 2.0f*(x+4)+(1-start),
                                        2.0f*(x+3)+(1-start), 2.0f*(x+2)+(1-start), 2.0f*(x+1)+(1-start), 2.0f*x+(1-start));

                __m256 uV = _mm256_load_ps(&u[index]);
                __m256 vV = _mm256_load_ps(&v[index]);

                xV = _mm256_fmadd_ps(dt0, uV, iV);
                yV = _mm256_fmadd_ps(dt0, vV, jV);

                xV = _mm256_max_ps(xV, c1);
                xV = _mm256_min_ps(xV, c2);
                __m256 aux = _mm256_floor_ps(xV);
                i0 = _mm256_cvtps_epi32(aux);
                i1 = _mm256_add_epi32(i0, one);

                yV = _mm256_max_ps(yV, c1);
                yV = _mm256_min_ps(yV, c2);
                aux = _mm256_floor_ps(yV);
                j0 = _mm256_cvtps_epi32(aux);
                j1 = _mm256_add_epi32(j0, one);

                s1 = _mm256_sub_ps(xV, _mm256_cvtepi32_ps(i0));
                s0 = _mm256_sub_ps(one_ps, s1);
                t1 = _mm256_sub_ps(yV, _mm256_cvtepi32_ps(j0));
                t0 = _mm256_sub_ps(one_ps, t1);

                __m256 aux0 = _mm256_i32gather_ps(d0, rb_idx_m256i(i0, j0, n+2), 4);
                __m256 aux1 = _mm256_i32gather_ps(d0, rb_idx_m256i(i0, j1, n+2), 4);
                __m256 aux2 = _mm256_i32gather_ps(d0, rb_idx_m256i(i1, j0, n+2), 4);
                __m256 aux3 = _mm256_i32gather_ps(d0, rb_idx_m256i(i1, j1, n+2), 4);
                aux0 = _mm256_mul_ps(t0, aux0);
                aux0 = _mm256_fmadd_ps(t1, aux1, aux0);
                aux2 = _mm256_mul_ps(t0, aux2);
                aux2 = _mm256_fmadd_ps(t1, aux3, aux2);
                aux0 = _mm256_mul_ps(s0, aux0);
                aux0 = _mm256_fmadd_ps(s1, aux2, aux0);

                _mm256_store_ps(&d[index], aux0);
            }

            if (x < width - (1-start)){
                index = mid*half + idx(x, y, width + coff);
                __m256 iV = _mm256_set_ps(2.0f*(x+7)+(1-start), 2.0f*(x+6)+(1-start), 2.0f*(x+5)+(1-start), 2.0f*(x+4)+(1-start),
                                        2.0f*(x+3)+(1-start), 2.0f*(x+2)+(1-start), 2.0f*(x+1)+(1-start), 2.0f*x+(1-start));

                __m256 uV = _mm256_load_ps(&u[index]);
                __m256 vV = _mm256_load_ps(&v[index]);

                xV = _mm256_fmadd_ps(dt0, uV, iV);
                yV = _mm256_fmadd_ps(dt0, vV, jV);

                xV = _mm256_max_ps(xV, c1);
                xV = _mm256_min_ps(xV, c2);
                __m256 aux = _mm256_floor_ps(xV);
                i0 = _mm256_cvtps_epi32(aux);
                i1 = _mm256_add_epi32(i0, one);

                yV = _mm256_max_ps(yV, c1);
                yV = _mm256_min_ps(yV, c2);
                aux = _mm256_floor_ps(yV);
                j0 = _mm256_cvtps_epi32(aux);
                j1 = _mm256_add_epi32(j0, one);

                s1 = _mm256_sub_ps(xV, _mm256_cvtepi32_ps(i0));
                s0 = _mm256_sub_ps(one_ps, s1);
                t1 = _mm256_sub_ps(yV, _mm256_cvtepi32_ps(j0));
                t0 = _mm256_sub_ps(one_ps, t1);

                __m256 aux0 = _mm256_i32gather_ps(d0, rb_idx_m256i(i0, j0, n+2), 4);
                __m256 aux1 = _mm256_i32gather_ps(d0, rb_idx_m256i(i0, j1, n+2), 4);
                __m256 aux2 = _mm256_i32gather_ps(d0, rb_idx_m256i(i1, j0, n+2), 4);
                __m256 aux3 = _mm256_i32gather_ps(d0, rb_idx_m256i(i1, j1, n+2), 4);
                aux0 = _mm256_mul_ps(t0, aux0);
                aux0 = _mm256_fmadd_ps(t1, aux1, aux0);
                aux2 = _mm256_mul_ps(t0, aux2);
                aux2 = _mm256_fmadd_ps(t1, aux3, aux2);
                aux0 = _mm256_mul_ps(s0, aux0);
                aux0 = _mm256_fmadd_ps(s1, aux2, aux0);

                __m256 dV = _mm256_load_ps(&d[index]);
                switch (width-(1-start)-x){
                    case 1:
                        aux0 = _mm256_blend_ps(dV, aux0, (1<<1)-1);
                        break;
                    case 2:
                        aux0 = _mm256_blend_ps(dV, aux0, (1<<2)-1);
                        break;
                    case 3:
                        aux0 = _mm256_blend_ps(dV, aux0, (1<<3)-1);
                        break;
                    case 4:
                        aux0 = _mm256_blend_ps(dV, aux0, (1<<4)-1);
                        break;
                    case 5:
                        aux0 = _mm256_blend_ps(dV, aux0, (1<<5)-1);
                        break;
                    case 6:
                        aux0 = _mm256_blend_ps(dV, aux0, (1<<6)-1);
                        break;
                    case 7:
                        aux0 = _mm256_blend_ps(dV, aux0, (1<<7)-1);
                        break;
                    default:
                        break;
                }

                _mm256_store_ps(&d[index], aux0);
            }
        }
    }
    set_bnd(n, b, d);
}

static void project(unsigned int n, float* u, float* v, float* p, float* div)
{
    for (unsigned int half = 0; half <= 1; half++) {
        unsigned int y = 1;
        #pragma omp parallel for private(y)
        for (y = 1; y <= n; y++) {
            unsigned int start = (half+y+1)%2;
            __m256 avx_d = _mm256_set1_ps(-2.0f*n);
            unsigned int width = (n+2)/2;
            unsigned int coff = color_offset(n);
            unsigned int mid = (n+2)*(width + coff);
            unsigned int x = 0;
            unsigned int index = idx(x, y, width + coff);
            unsigned int index_div = mid*half + index;
            unsigned int index_uv = mid*(1-half) + index;

            __m256 divV = _mm256_load_ps(&div[index_div]);
            __m256 pV    = _mm256_load_ps(&p[index_div]);
            __m256 curr = _mm256_load_ps(&u[index_uv]);
            __m256 next = _mm256_load_ps(&u[index_uv+8]);
            __m256 up = _mm256_load_ps(&v[index_uv - (width + coff)]);
            __m256 down = _mm256_load_ps(&v[index_uv + (width + coff)]);

            __m256 aux  = _mm256_undefined_ps();
            __m256 aux2  = _mm256_setzero_ps();
            __m256 sv = _mm256_sub_ps(down, up);
            if (start) {
                __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                aux = _mm256_permutevar8x32_ps(curr, perm_idx);
                aux = _mm256_sub_ps(curr, aux);
                aux = _mm256_add_ps(aux, sv);
                aux = _mm256_div_ps(aux, avx_d);
                aux = _mm256_blend_ps(aux, divV, 1);
                aux2 = _mm256_blend_ps(aux2, pV, 1);
            } else {
                aux  = _mm256_blend_ps(curr, next, 1);
                __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                aux = _mm256_sub_ps(aux, curr);
                aux = _mm256_add_ps(aux, sv);
                aux = _mm256_div_ps(aux, avx_d);
            }

            _mm256_store_ps(&div[index_div], aux);
            _mm256_store_ps(&p[index_div], aux2);

            for (x = 8; x+8 < width + coff; x+=8) {
                index = idx(x, y, width + coff);
                index_div = mid*half + index;
                index_uv = mid*(1-half) + index;

                __m256 up = _mm256_load_ps(&v[index_uv - (width + coff)]);
                __m256 down = _mm256_load_ps(&v[index_uv + (width + coff)]);

                __m256 aux  = _mm256_undefined_ps();
                __m256 sv = _mm256_sub_ps(down, up);
                if (start) {
                    aux  = _mm256_blend_ps(next, curr, 1<<7);
                    __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                    aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                    curr = next;
                    next = _mm256_load_ps(&u[index_uv+8]); // next
                    aux = _mm256_sub_ps(curr, aux);
                } else {
                    curr = next;
                    next = _mm256_load_ps(&u[index_uv+8]); // next
                    aux  = _mm256_blend_ps(curr, next, 1);
                    __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                    aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                    aux = _mm256_sub_ps(aux, curr);
                }
                aux = _mm256_add_ps(aux, sv);
                aux = _mm256_div_ps(aux, avx_d);

                _mm256_store_ps(&div[index_div], aux);
                _mm256_store_ps(&p[index_div], _mm256_setzero_ps());
            }

            if (x < width - (1-start)){
                index = idx(x, y, width + coff);
                index_div = mid*half + index;
                index_uv = mid*(1-half) + index;

                __m256 divV = _mm256_load_ps(&div[index_div]);
                __m256 pV   = _mm256_load_ps(&p[index_div]);
                __m256 up = _mm256_load_ps(&v[index_uv - (width+coff)]);
                __m256 down = _mm256_load_ps(&v[index_uv + (width+coff)]);
                
                __m256 aux = _mm256_undefined_ps();
                __m256 sv = _mm256_sub_ps(down, up);
                if (start) {
                    aux  = _mm256_blend_ps(next, curr, 1<<7);
                    __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                    aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                    curr = next;
                    aux = _mm256_sub_ps(curr, aux);
                } else {
                    curr = next;
                    __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                    aux = _mm256_permutevar8x32_ps(curr, perm_idx);
                    aux = _mm256_sub_ps(aux, curr);
                }
                aux = _mm256_add_ps(aux, sv);
                aux = _mm256_div_ps(aux, avx_d);
                
                __m256 aux2 = _mm256_setzero_ps();
                switch (width-(1-start)-x){
                    case 1:
                        aux = _mm256_blend_ps(divV, aux, (1<<1)-1);
                        aux2 = _mm256_blend_ps(pV, aux2, (1<<1)-1);
                        break;
                    case 2:
                        aux = _mm256_blend_ps(divV, aux, (1<<2)-1);
                        aux2 = _mm256_blend_ps(pV, aux2, (1<<2)-1);
                        break;
                    case 3:
                        aux = _mm256_blend_ps(divV, aux, (1<<3)-1);
                        aux2 = _mm256_blend_ps(pV, aux2, (1<<3)-1);
                        break;
                    case 4:
                        aux = _mm256_blend_ps(divV, aux, (1<<4)-1);
                        aux2 = _mm256_blend_ps(pV, aux2, (1<<4)-1);
                        break;
                    case 5:
                        aux = _mm256_blend_ps(divV, aux, (1<<5)-1);
                        aux2 = _mm256_blend_ps(pV, aux2, (1<<5)-1);
                        break;
                    case 6:
                        aux = _mm256_blend_ps(divV, aux, (1<<6)-1);
                        aux2 = _mm256_blend_ps(pV, aux2, (1<<6)-1);
                        break;
                    case 7:
                        aux = _mm256_blend_ps(divV, aux, (1<<7)-1);
                        aux2 = _mm256_blend_ps(pV, aux2, (1<<7)-1);
                        break;
                    default:
                        break;
                }

                _mm256_store_ps(&div[index_div], aux);
                _mm256_store_ps(&p[index_div], aux2);
            }
        }
    }

    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    __m256 avx_m = _mm256_set1_ps(-0.5f*n);
    for (unsigned int half = 0; half <= 1; half++) {
        unsigned int y = 1;
        #pragma omp parallel for private(y)
        for (y = 1; y <= n; y++) {
            unsigned int start = (half+y+1)%2;
            unsigned int width = (n+2)/2;
            unsigned int coff = color_offset(n);
            unsigned int mid = (n+2)*(width + coff);
            unsigned int x = 0;
            unsigned int index = idx(x, y, width + coff);
            unsigned int index_uv = mid*half + index;
            unsigned int index_p = mid*(1-half) + index;

            __m256 curr = _mm256_load_ps(&p[index_p]);
            __m256 next = _mm256_load_ps(&p[index_p+8]);
            __m256 uV = _mm256_load_ps(&u[index_uv]);
            __m256 vV = _mm256_load_ps(&v[index_uv]);
            __m256 up = _mm256_load_ps(&p[index_p - (width + coff)]);
            __m256 down = _mm256_load_ps(&p[index_p + (width + coff)]);

            __m256 aux  = _mm256_undefined_ps();
            __m256 su   = _mm256_undefined_ps();
            __m256 sv = _mm256_sub_ps(down, up);
            sv = _mm256_fmadd_ps(avx_m, sv, vV);
            if (start) {
                __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                aux = _mm256_permutevar8x32_ps(curr, perm_idx);
                su = _mm256_sub_ps(curr, aux);
                su = _mm256_fmadd_ps(avx_m, su, uV);
                su = _mm256_blend_ps(su, uV, 1);
                sv = _mm256_blend_ps(sv, vV, 1);
            } else {
                aux  = _mm256_blend_ps(curr, next, 1);
                __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                su = _mm256_sub_ps(aux, curr);
                su = _mm256_fmadd_ps(avx_m, su, uV);
            }

            _mm256_store_ps(&u[index_uv], su);
            _mm256_store_ps(&v[index_uv], sv);

            for (x = 8; x+8 < width + coff; x+=8) {
                index = idx(x, y, width + coff);
                index_uv = mid*half + index;
                index_p = mid*(1-half) + index;

                __m256 uV   = _mm256_load_ps(&u[index_uv]);
                __m256 vV   = _mm256_load_ps(&v[index_uv]);
                __m256 up   = _mm256_load_ps(&p[index_p - (width + coff)]);
                __m256 down = _mm256_load_ps(&p[index_p + (width + coff)]);

                __m256 aux  = _mm256_undefined_ps();
                __m256 su   = _mm256_undefined_ps();
                if (start) {
                    aux  = _mm256_blend_ps(next, curr, 1<<7);
                    __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                    aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                    curr = next;
                    next = _mm256_load_ps(&p[index_p+8]); // next
                    su = _mm256_sub_ps(curr, aux);
                } else {
                    curr = next;
                    next = _mm256_load_ps(&p[index_p+8]); // next
                    aux  = _mm256_blend_ps(curr, next, 1);
                    __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                    aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                    su = _mm256_sub_ps(aux, curr);
                }
                su = _mm256_fmadd_ps(avx_m, su, uV);
                
                __m256 sv = _mm256_sub_ps(down, up);
                sv = _mm256_fmadd_ps(avx_m, sv, vV);

                _mm256_store_ps(&u[index_uv], su);
                _mm256_store_ps(&v[index_uv], sv);
            }

            if (x < width - (1-start)){
                index = idx(x, y, width + coff);
                unsigned int index_uv = mid*half + index;
                unsigned int index_p = mid*(1-half) + index;
                    
                __m256 uV = _mm256_load_ps(&u[index_uv]);
                __m256 vV = _mm256_load_ps(&v[index_uv]);
                __m256 up = _mm256_load_ps(&p[index_p - (width+coff)]);
                __m256 down = _mm256_load_ps(&p[index_p + (width+coff)]);
                
                __m256 aux = _mm256_undefined_ps();
                __m256 su = _mm256_undefined_ps();
                if (start) {
                    aux  = _mm256_blend_ps(next, curr, 1<<7);
                    __m256i perm_idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
                    aux = _mm256_permutevar8x32_ps(aux, perm_idx);
                    curr = next;
                    su = _mm256_sub_ps(curr, aux);
                } else {
                    curr = next;
                    __m256i perm_idx = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                    aux = _mm256_permutevar8x32_ps(curr, perm_idx);
                    su = _mm256_sub_ps(aux, curr);
                }
                su = _mm256_fmadd_ps(avx_m, su, uV);
                
                __m256 sv = _mm256_sub_ps(down, up);
                sv = _mm256_fmadd_ps(avx_m, sv, vV);
                switch (width-(1-start)-x){
                    case 1:
                        su = _mm256_blend_ps(uV, su, (1<<1)-1);
                        break;
                    case 2:
                        su = _mm256_blend_ps(uV, su, (1<<2)-1);
                        break;
                    case 3:
                        su = _mm256_blend_ps(uV, su, (1<<3)-1);
                        break;
                    case 4:
                        su = _mm256_blend_ps(uV, su, (1<<4)-1);
                        break;
                    case 5:
                        su = _mm256_blend_ps(uV, su, (1<<5)-1);
                        break;
                    case 6:
                        su = _mm256_blend_ps(uV, su, (1<<6)-1);
                        break;
                    case 7:
                        su = _mm256_blend_ps(uV, su, (1<<7)-1);
                        break;
                    default:
                        break;
                }

                _mm256_store_ps(&u[index_uv], su);
                _mm256_store_ps(&v[index_uv], sv);
            }
        }
    }
    
    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
}

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
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
