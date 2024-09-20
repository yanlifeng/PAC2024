#include <stdlib.h>
#include <stdint.h>
//#define use_init_code
#define MAX_THREAD_NUM 576

//#define use_my_sve

#ifdef use_my_sve
#include <arm_sve.h>
#endif

#include <sys/time.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

#ifdef use_init_code
void polynomial_stencil(double *fa, double *f, long nx, double p[], int term) {
    int idx = term / 2;

#pragma omp parallel for
    for (long i = 0; i < nx; i++) {
        for (int j = 0; j < term; j++) {

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
#else
void polynomial_stencil(double *fa, double *f, long nx, double p[], int term)
{
    int idx = term / 2;

    if(term * MAX_THREAD_NUM * 2ll >= nx) {

#pragma omp parallel for
        for (long i = 0; i < nx; i++) {
            for (int j = 0; j < term; j++) {
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
        return;
    }

    double time1[MAX_THREAD_NUM] = {0};
    double time2[MAX_THREAD_NUM] = {0};
    //double pr[term];
    double *pr;

    if (posix_memalign((void**)&pr, 64, term * sizeof(double)) != 0) {
        perror("Memory allocation failed");
        exit(1);
    }

    for(int i = 0; i < term; i++) {
        pr[i] = p[term - i - 1];
    }

    double *X[MAX_THREAD_NUM];
    for(int i = 0; i < MAX_THREAD_NUM; i++) {
        if (posix_memalign((void**)&X[i], 64, (idx + 1) * 8 * sizeof(double)) != 0) {
            perror("Memory allocation failed");
            exit(1);
        }

    }


    double t_tot = GetTime();
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int threads_count = omp_get_num_threads();
        long l_range = (nx / threads_count) * thread_id;
        long r_range = (thread_id == threads_count - 1) ? nx : (nx / threads_count) * (thread_id + 1);

		double t0;
		t0 = GetTime();

        long si = l_range + term;
        while(((uintptr_t)&f[si]) % 64) si++;


		long ii;
		for(ii = si; ii + 8 <= r_range - term; ii += 8) {

#ifdef use_my_sve

			double *Xx;
			svbool_t pg = svwhilelt_b64(0, 8);

			// Initialize Xx with 1
			Xx = &(X[thread_id][0]);
			svfloat64_t ones = svdup_f64(1.0);
			svst1(pg, Xx, ones);

			// Load f values into vector
			double *ff = &(f[ii]);
			svfloat64_t ff_vec = svld1(pg, ff);

			// Compute Xx for each k
			for(int k = 1; k <= idx; k++) {
				Xx = &(X[thread_id][k * 8]);
				svfloat64_t prev_Xx = svld1(pg, &(X[thread_id][(k - 1) * 8]));
				svfloat64_t Xx_vec = svmul_f64_m(pg, prev_Xx, ff_vec);
				svst1(pg, Xx, Xx_vec);
			}

			// Copy ff values to the first Xx
			Xx = &(X[thread_id][0]);
			svst1(pg, Xx, ff_vec);

			double *ffa = &(fa[ii]);

			// First loop for ffa update
			for(int j = 0; j <= idx; j++) {
				double p_con = pr[term - idx + j - 1];
				Xx = &(X[thread_id][j * 8]);
				svfloat64_t Xx_vec = svld1(pg, Xx);
				svfloat64_t ffa_vec = svld1(pg, &(ffa[j]));
				ffa_vec = svmla_f64_m(pg, ffa_vec, Xx_vec, svdup_f64(p_con));
				svst1(pg, &(ffa[j]), ffa_vec);
			}

			// Second loop for negative indexing
			for(int j = term - idx - 1; j >= 1; j--) {
				double p_con = pr[term - idx - j - 1];
				Xx = &(X[thread_id][j * 8]);
				svfloat64_t Xx_vec = svld1(pg, Xx);
				svfloat64_t ffa_vec = svld1(pg, &(ffa[-j]));
				ffa_vec = svmla_f64_m(pg, ffa_vec, Xx_vec, svdup_f64(p_con));
				svst1(pg, &(ffa[-j]), ffa_vec);
			} 
#else
            double *Xx;
            Xx = &(X[thread_id][0]);
            for(int o = 0; o < 8; o++) Xx[o] = 1;
            double *ff = &(f[ii]);
            for(int k = 1; k <= idx; k++) {
                Xx = &(X[thread_id][k * 8]);
                for(int o = 0; o < 8; o++) {
                    Xx[o] = Xx[o - 8] * ff[o];
                }
            }
            Xx = &(X[thread_id][0]);
            for(int o = 0; o < 8; o++) Xx[o] = ff[o];

            double *ffa = &(fa[ii]);
            for(int j = 0; j <= idx; j++) {
                double p_con = pr[term - idx + j - 1];
                Xx = &(X[thread_id][j * 8]);
                for(int o = 0; o < 8; o++) {
                    ffa[o + j] += Xx[o] * p_con;
                }
            }
            //for(int j = term - idx - 1; j >= 1; j--) {
            for(int j = 1; j < term - idx; j++) {
                double p_con = pr[term - idx - j - 1];
                Xx = &(X[thread_id][j * 8]);
                for(int o = 0; o < 8; o++) {
                    ffa[o - j] += Xx[o] * p_con;
                }
            }
#endif
        }

        for(; ii < r_range - term; ii++) {
            double x = f[ii];
            X[thread_id][0] = 1;
            for(int k = 1; k <= idx; k++) {
                X[thread_id][k] = X[thread_id][k - 1] * x;
            }
            X[thread_id][0] = x;
            for(int j = 0; j <= idx; j++) {
                fa[ii + j] += X[thread_id][j] * pr[term - idx + j - 1];
            }
            for(int j = term - idx - 1; j >= 1; j--) {
                fa[ii - j] += X[thread_id][j] * pr[term - idx - j -1];
            }
        }

        time2[thread_id] = GetTime() - t0;

        //t0 = GetTime();
#pragma omp critical
        {

            for(long i = l_range; i < si; i++) {
                double x = f[i];
                X[thread_id][0] = 1;
                for(int k = 1; k <= idx; k++) {
                    X[thread_id][k] = X[thread_id][k - 1] * x;
                }
                X[thread_id][0] = x;
                for(int j = 0; j <= idx; j++) {
                    if(i + j < nx) fa[i + j] += X[thread_id][j] * p[idx - j];
                }
                for(int j = 1; j < term - idx; j++) {
                    if(i - j >= 0) fa[i - j] += X[thread_id][j] * p[idx + j];
                }
            }

            for(long i = r_range - term; i < r_range; i++) {
                double x = f[i];
                X[thread_id][0] = 1;
                for(int k = 1; k <= idx; k++) {
                    X[thread_id][k] = X[thread_id][k - 1] * x;
                }
                X[thread_id][0] = x;
                for(int j = 0; j <= idx; j++) {
                    if(i + j < nx) fa[i + j] += X[thread_id][j] * p[idx - j];
                }
                for(int j = 1; j < term - idx; j++) {
                    if(i - j >= 0) fa[i - j] += X[thread_id][j] * p[idx + j];
                }
            }

        }
        //time1[thread_id] = GetTime() - t0;




    }
    printf("time total cost %lf\n", GetTime() - t_tot);
    double time1_tot = 0;
    double time2_max = 0;
    for(int i = 0; i < MAX_THREAD_NUM; i++) {
        time1_tot += time1[i];
        time2_max = MAX(time2[i], time2_max);
    }
    //printf("time1 total cost %lf\n", time1_tot);
    printf("time2 max cost %lf\n", time2_max);

    for(int i = 0; i < MAX_THREAD_NUM; i++) {
        free(X[i]);
    }
    free(pr);

}
#endif
