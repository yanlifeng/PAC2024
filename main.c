
#define _GNU_SOURCE
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include "omp.h"
#include <stdint.h>
#include <pthread.h>

//#define my_print_error

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#include "polynomial_stencil.h"

const long MAX_NX = (8L * 1000L * 1000L * 1000L);
const double MAX_DIFF = 1e-8;
const int ITER_TIMES = 5;

#define THREAD_NUM_KP 576

static uint32_t state = 123456789;
#pragma omp threadprivate(state) 

static uint32_t xorshift32(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static inline double fast_rand(double max_num) {
    return xorshift32() / (double)(UINT32_MAX / max_num) + 2;
}

void set_thread_affinity_main(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}



__attribute__((optimize("O0")))
void my_init(long nx, double *f) {

    asm volatile("" ::: "memory"); 
    double x_init = (double)(rand() / (double)(RAND_MAX / 2.0));
    for (long i = 0; i < nx; i++) {
        f[i] = x_init;
    }
    asm volatile("" ::: "memory"); 
}


void polynomial_stencil_verify(double *fa, double *f, long nx, double p[], int term)
{
    int idx = term / 2;

#pragma omp parallel for
    for (long i = 0; i < nx; i++) {
        for (int j = 0; j < term; j++) {

            // 超出边界的点按零处理
            if ((i + j - idx) < 0 || (i + j - idx) >= nx) {
                continue;
            }
            double x = f[i + j - idx];
            for (int k = 1; k < abs(j - idx); k++) {
                x *= f[i + j - idx];
            }
            fa[i] += x * p[j];
        }
    }
}

void print_array(double *fa, long nx)
{
    long i;
    printf("start print array\n");
    for (i = 0; i < nx; i++) {
        printf("%f ", fa[i]);
    }
    printf("\n");
}


int FastCompareResults(double *se, double *seOpt, long size)
{

    int oks[THREAD_NUM_KP];
    for(int i = 0; i < THREAD_NUM_KP; i++) {
        oks[i] = 1;
    }
#ifdef my_print_error
#else
#pragma omp parallel for
#endif
    for (long i = 0; i < size; i++) {
#ifdef my_print_error
        int tid = 0;
#else
        int tid = omp_get_thread_num();
#endif
        double diff = se[i] - seOpt[i];
        if ((diff > MAX_DIFF) || (diff < -MAX_DIFF)) {
#ifdef my_print_error
            printf("compute error at %ld, result: %.20lf %.20lf, diff %.20lf\n", i, se[i], seOpt[i], diff);
            return -1;
#endif
            //return -1;
            oks[tid] = 0;
        }
    }
    
    for(int i = 0; i < THREAD_NUM_KP; i++) {
        if(oks[i] == 0) return -1;
    }
    return 0;
}

int CompareResults(double *se, double *seOpt, long size)
{
    long i;
    for (i = 0; i < size; i++) {
        double diff = se[i] - seOpt[i];
        if ((diff > MAX_DIFF) || (diff < -MAX_DIFF)) {
            printf("compute error at %ld, result: %.20lf %.20lf, diff %.20lf\n", i, se[i], seOpt[i], diff);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    //set_thread_affinity_main(0);
    long nx;
    long i, j;
    double max_num = 2.0;
    struct timeval start, stop;

    nx = MAX_NX;

    double *f;
    posix_memalign((void **)&f, 4096, nx * sizeof(double));
    double *fa;
    posix_memalign((void **)&fa, 4096, nx * sizeof(double));
    double *fb;
    posix_memalign((void **)&fb, 4096, nx * sizeof(double));
    //double *f = malloc(nx * sizeof(double));
    //double *fa = malloc(nx * sizeof(double));
    //double *fb = malloc(nx * sizeof(double));

    int rr = rand();
    memset(fa, rr, nx * sizeof(double));
    memset(fb, rr, nx * sizeof(double));
    memset(f, rr, nx * sizeof(double));

    FILE *fp = fopen(argv[1], "r");
    double compute_time = 0.0;
    double iter_time = 0.0;

    int test = 1;
    int term;
    double p[15];
    int missed = 0;
    int iter;
    while (fscanf(fp, "%ld", &nx) != EOF) {

        if(nx > MAX_NX || nx <=0 ) {
            printf("nx value error.\n");
            return -2;
        }

        srand(test);

        fscanf(fp, "%d", &term);

        for (j = 0; j < term; j++) {
            fscanf(fp, "%lf", &p[j]);
        }

        printf("test %d , nx %ld, term %d, p0: %lf, p[-1]: %lf\n", test, nx, term, p[0], p[term - 1]);

        for (iter = 1; iter <= ITER_TIMES; iter++) {

//#pragma omp parallel
//            {
//                int thread_id = omp_get_thread_num();
//                int threads_count = omp_get_num_threads();
//                long l_range = (nx / threads_count) * thread_id;
//                long r_range = (thread_id == threads_count - 1) ? nx : (nx / threads_count) * (thread_id + 1);
//                for(long i = l_range; i < r_range; i++) {
//                    fa[i] = 0;
//                    fb[i] = 0;
//                    f[i] = fast_rand(max_num);
//                }
//
//            }
//

            memset(fa, 0, nx * sizeof(double));
            memset(fb, 0, nx * sizeof(double));
            //my_init(nx, f);
            //my_init(nx, fa);
            //my_init(nx, fb);

            srand(iter);
//#pragma omp parallel for
            for (j = 0; j < nx; j++) {
                //fa[j] = 0;
                //fb[j] = 0;
                f[j] = fast_rand(max_num);
            }

            //printf("22222\n");

            long si1, si2;
            if(1) {
                int nodes[1];
                void *pages[1];
                int now_numa = 0;
                int cnt = 0;
                long ii = 0;
                int pre_node = -1;
                while(ii < nx) {
                    double *pp = &(f[ii]);
                    pages[0] = (void*)pp;
                    numa_move_pages(0, 1, pages, NULL, nodes, 0);
                    //printf("%lld %d\n", ii, nodes[0]);
                    if(pre_node == 3 && nodes[0] == 0) break;
                    pre_node = nodes[0];
                    ii++;
                }
                si1 = ii;
                //for(; ii < nx; ii += 512) {
                //    int tar_numa = 0;
                //    pages[0] = (void*)(&(f[ii]));
                //    numa_move_pages(0, 1, pages, NULL, nodes, 0);
                //    tar_numa = nodes[0];
                //    for(long jj = ii; jj < ii + 512; jj++) {
                //        double *pp = &(f[jj]);
                //        pages[0] = (void*)pp;
                //        numa_move_pages(0, 1, pages, NULL, nodes, 0);
                //        if(nodes[0] != tar_numa) {
                //            printf("GG11 for %lld, %d != %d\n", ii, nodes[0], tar_numa);
                //            exit(0);
                //        }
                //    }
                //    double *pp = &(f[ii]);
                //    pages[0] = (void*)pp;
                //    numa_move_pages(0, 1, pages, NULL, nodes, 0);
                //    //printf("f %lld on %d\n", ii, nodes[0]);
                //    if(nodes[0] != now_numa) {
                //        printf("GG12 for %lld, %d != %d\n", ii, nodes[0], now_numa);
                //        exit(0);
                //    }
                //    cnt++;
                //    now_numa++;
                //    now_numa %= 4;
                //}
            }



            if(1) {
                int nodes[1];
                void *pages[1];
                int now_numa = 0;
                int cnt = 0;
                long ii = 0;
                int pre_node = -1;
                while(ii < nx) {
                    double *pp = &(fa[ii]);
                    pages[0] = (void*)pp;
                    numa_move_pages(0, 1, pages, NULL, nodes, 0);
                    //printf("%lld %d\n", ii, nodes[0]);
                    if(pre_node == 3 && nodes[0] == 0) break;
                    pre_node = nodes[0];
                    ii++;
                }
                si2 = ii;
                //for(; ii < nx; ii += 512) {
                //    int tar_numa = 0;
                //    pages[0] = (void*)(&(fa[ii]));
                //    numa_move_pages(0, 1, pages, NULL, nodes, 0);
                //    tar_numa = nodes[0];
                //    for(long jj = ii; jj < ii + 512; jj++) {
                //        double *pp = &(fa[jj]);
                //        pages[0] = (void*)pp;
                //        numa_move_pages(0, 1, pages, NULL, nodes, 0);
                //        if(nodes[0] != tar_numa) {
                //            printf("GG21 for %lld, %d != %d\n", ii, nodes[0], tar_numa);
                //            exit(0);
                //        }
                //    }
                //    double *pp = &(fa[ii]);
                //    pages[0] = (void*)pp;
                //    numa_move_pages(0, 1, pages, NULL, nodes, 0);
                //    //printf("fa %lld on %d\n", ii, nodes[0]);
                //    if(nodes[0] != now_numa) {
                //        printf("GG22 for %lld, %d != %d\n", ii, nodes[0], now_numa);
                //        exit(0);
                //    }
                //    cnt++;
                //    now_numa++;
                //    now_numa %= 4;
                //}
            }
            printf("main si %lld %lld\n", si1, si2);



            gettimeofday(&start, (struct timezone *)0);
            polynomial_stencil(fa + si2, f + si1, nx - MAX(si1, si2), p, term);

            gettimeofday(&stop, (struct timezone *)0);
            //polynomial_stencil_verify(fb, f, nx, p, term);
            polynomial_stencil_verify(fb + si2, f + si1, nx - MAX(si1, si2), p, term);

            iter_time = (double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.e-6;
            compute_time += iter_time;
            printf("iteration %d compute time:%10.6f sec\n", iter, iter_time);

            missed += FastCompareResults(fa, fb, nx);
        }

        if (missed) {
            if (missed == -ITER_TIMES) {
                printf("compute error!\n");
            }
            printf("part of compute result varifed fail, optimizing instable!\n");
            return -1;
        } else {
            printf("compute results are verifed success!\n");
        }

        test++;
    }
    printf("all test cases compute time is %lf sec.\n", compute_time);

    return 0;
}
