#include <stdlib.h>
#include <stdint.h>
//#define use_init_code
#define use_fast_code
#define MAX_THREAD_NUM 576

#define use_my_sve

#ifdef use_my_sve
#include <arm_sve.h>
#endif

#include <sys/time.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

# ifdef use_fast_code
void polynomial_stencil(double *fa, double *f, long nx, double p[], int term) {
    int idx = term / 2;

    if (term * 2 >= nx) {
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

    for (long i = 0; i < term; i++) {
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
    for (long i = nx - term; i < nx; i++) {
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

    long nxx = nx - term * 2;
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int threads_count = omp_get_num_threads();
        long l_range = (nxx / threads_count) * thread_id;
		long r_range = (thread_id == threads_count - 1) ? nxx : (nxx / threads_count) * (thread_id + 1);
		l_range += term;
		r_range += term;

        long si = l_range;
        while (si < r_range && ((uintptr_t) &f[si]) % 64) si++;

        for (long i = l_range; i < si; i++) {
            for (int j = 0; j < term; j++) {
                double x = f[i + j - idx];
                for (int k = 1; k < abs(j - idx); k++) {
                    x *= f[i + j - idx];
                }
                fa[i] += x * p[j];
            }
        }

        l_range = si;


		long i;
#ifdef use_my_sve
		for (i = l_range; i + svcntd() * 8 <= r_range; i += svcntd() * 8) {
		//for (i = l_range; i + svcntd() <= r_range; i += svcntd()) {
			double* ff1 = &(f[i]);
			double* ff2 = &(f[i + 8]);
			double* ff3 = &(f[i + 16]);
			double* ff4 = &(f[i + 24]);
			double* ff5 = &(f[i + 32]);
			double* ff6 = &(f[i + 40]);
			double* ff7 = &(f[i + 48]);
			double* ff8 = &(f[i + 56]);

			double* ffa1 = &(fa[i]);
			double* ffa2 = &(fa[i + 8]);
			double* ffa3 = &(fa[i + 16]);
			double* ffa4 = &(fa[i + 24]);
			double* ffa5 = &(fa[i + 32]);
			double* ffa6 = &(fa[i + 40]);
			double* ffa7 = &(fa[i + 48]);
			double* ffa8 = &(fa[i + 56]);

			svbool_t pg = svptrue_b64();
			svfloat64_t vec_fa1 = svld1(pg, ffa1);
			svfloat64_t vec_fa2 = svld1(pg, ffa2);
			svfloat64_t vec_fa3 = svld1(pg, ffa3);
			svfloat64_t vec_fa4 = svld1(pg, ffa4);
			svfloat64_t vec_fa5 = svld1(pg, ffa5);
			svfloat64_t vec_fa6 = svld1(pg, ffa6);
			svfloat64_t vec_fa7 = svld1(pg, ffa7);
			svfloat64_t vec_fa8 = svld1(pg, ffa8);

			for (int j = 0; j < idx; j++) {
                svfloat64_t ff1_vec = svld1(pg, &ff1[j - idx]);
                svfloat64_t ff2_vec = svld1(pg, &ff2[j - idx]);
                svfloat64_t ff3_vec = svld1(pg, &ff3[j - idx]);
                svfloat64_t ff4_vec = svld1(pg, &ff4[j - idx]);
                svfloat64_t ff5_vec = svld1(pg, &ff5[j - idx]);
                svfloat64_t ff6_vec = svld1(pg, &ff6[j - idx]);
                svfloat64_t ff7_vec = svld1(pg, &ff7[j - idx]);
                svfloat64_t ff8_vec = svld1(pg, &ff8[j - idx]);

                svfloat64_t vec_x1 = ff1_vec;
                svfloat64_t vec_x2 = ff2_vec;
                svfloat64_t vec_x3 = ff3_vec;
                svfloat64_t vec_x4 = ff4_vec;
                svfloat64_t vec_x5 = ff5_vec;
                svfloat64_t vec_x6 = ff6_vec;
                svfloat64_t vec_x7 = ff7_vec;
                svfloat64_t vec_x8 = ff8_vec;

				for (int k = 1; k < idx - j; k++) {
					vec_x1 = svmul_f64_x(pg, vec_x1, ff1_vec);
					vec_x2 = svmul_f64_x(pg, vec_x2, ff2_vec);
					vec_x3 = svmul_f64_x(pg, vec_x3, ff3_vec);
					vec_x4 = svmul_f64_x(pg, vec_x4, ff4_vec);
					vec_x5 = svmul_f64_x(pg, vec_x5, ff5_vec);
					vec_x6 = svmul_f64_x(pg, vec_x6, ff6_vec);
					vec_x7 = svmul_f64_x(pg, vec_x7, ff7_vec);
					vec_x8 = svmul_f64_x(pg, vec_x8, ff8_vec);
				}

				vec_fa1 = svmla_f64_x(pg, vec_fa1, vec_x1, svdup_f64(p[j]));
				vec_fa2 = svmla_f64_x(pg, vec_fa2, vec_x2, svdup_f64(p[j]));
				vec_fa3 = svmla_f64_x(pg, vec_fa3, vec_x3, svdup_f64(p[j]));
				vec_fa4 = svmla_f64_x(pg, vec_fa4, vec_x4, svdup_f64(p[j]));
				vec_fa5 = svmla_f64_x(pg, vec_fa5, vec_x5, svdup_f64(p[j]));
				vec_fa6 = svmla_f64_x(pg, vec_fa6, vec_x6, svdup_f64(p[j]));
				vec_fa7 = svmla_f64_x(pg, vec_fa7, vec_x7, svdup_f64(p[j]));
				vec_fa8 = svmla_f64_x(pg, vec_fa8, vec_x8, svdup_f64(p[j]));
			}

            for (int j = idx; j < term; j++) {
                svfloat64_t ff1_vec = svld1(pg, &ff1[j - idx]);
                svfloat64_t ff2_vec = svld1(pg, &ff2[j - idx]);
                svfloat64_t ff3_vec = svld1(pg, &ff3[j - idx]);
                svfloat64_t ff4_vec = svld1(pg, &ff4[j - idx]);
                svfloat64_t ff5_vec = svld1(pg, &ff5[j - idx]);
                svfloat64_t ff6_vec = svld1(pg, &ff6[j - idx]);
                svfloat64_t ff7_vec = svld1(pg, &ff7[j - idx]);
                svfloat64_t ff8_vec = svld1(pg, &ff8[j - idx]);

                svfloat64_t vec_x1 = ff1_vec;
                svfloat64_t vec_x2 = ff2_vec;
                svfloat64_t vec_x3 = ff3_vec;
                svfloat64_t vec_x4 = ff4_vec;
                svfloat64_t vec_x5 = ff5_vec;
                svfloat64_t vec_x6 = ff6_vec;
                svfloat64_t vec_x7 = ff7_vec;
                svfloat64_t vec_x8 = ff8_vec;

				for (int k = 1; k < j - idx; k++) {
					vec_x1 = svmul_f64_x(pg, vec_x1, ff1_vec);
					vec_x2 = svmul_f64_x(pg, vec_x2, ff2_vec);
					vec_x3 = svmul_f64_x(pg, vec_x3, ff3_vec);
					vec_x4 = svmul_f64_x(pg, vec_x4, ff4_vec);
					vec_x5 = svmul_f64_x(pg, vec_x5, ff5_vec);
					vec_x6 = svmul_f64_x(pg, vec_x6, ff6_vec);
					vec_x7 = svmul_f64_x(pg, vec_x7, ff7_vec);
					vec_x8 = svmul_f64_x(pg, vec_x8, ff8_vec);
				}

				vec_fa1 = svmla_f64_x(pg, vec_fa1, vec_x1, svdup_f64(p[j]));
				vec_fa2 = svmla_f64_x(pg, vec_fa2, vec_x2, svdup_f64(p[j]));
				vec_fa3 = svmla_f64_x(pg, vec_fa3, vec_x3, svdup_f64(p[j]));
				vec_fa4 = svmla_f64_x(pg, vec_fa4, vec_x4, svdup_f64(p[j]));
				vec_fa5 = svmla_f64_x(pg, vec_fa5, vec_x5, svdup_f64(p[j]));
				vec_fa6 = svmla_f64_x(pg, vec_fa6, vec_x6, svdup_f64(p[j]));
				vec_fa7 = svmla_f64_x(pg, vec_fa7, vec_x7, svdup_f64(p[j]));
				vec_fa8 = svmla_f64_x(pg, vec_fa8, vec_x8, svdup_f64(p[j]));
			}



			svst1(pg, ffa1, vec_fa1);
			svst1(pg, ffa2, vec_fa2);
			svst1(pg, ffa3, vec_fa3);
			svst1(pg, ffa4, vec_fa4);
			svst1(pg, ffa5, vec_fa5);
			svst1(pg, ffa6, vec_fa6);
			svst1(pg, ffa7, vec_fa7);
			svst1(pg, ffa8, vec_fa8);
		}
#else
        for (i = l_range; i + 8 <= r_range; i += 8) {
            double* ff = &(f[i]);
            double* ffa = &(fa[i]);
            for (int j = 0; j < term; j++) {
                double x[8];
                for(int o = 0; o < 8; o++) x[o] = ff[o + j - idx];
                for (int k = 1; k < abs(j - idx); k++) {
                    for(int o = 0; o < 8; o++) {
                        x[o] *= ff[o + j - idx];
                    }
                }

                for(int o = 0; o < 8; o++) {
                    ffa[o] += x[o] * p[j];
                }
            }
        }
#endif
        for (; i < r_range; i++) {
            for (int j = 0; j < term; j++) {
                double x = f[i + j - idx];
                for (int k = 1; k < abs(j - idx); k++) {
                    x *= f[i + j - idx];
                }
                fa[i] += x * p[j];
            }
        }
    }
}
# else

void polynomial_stencil(double *fa, double *f, long nx, double p[], int term) {
    int idx = term / 2;

    if (term * MAX_THREAD_NUM * 2ll >= nx) {

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

    if (posix_memalign((void **) &pr, 64, term * sizeof(double)) != 0) {
        perror("Memory allocation failed");
        exit(1);
    }

    for (int i = 0; i < term; i++) {
        pr[i] = p[term - i - 1];
    }

    double *X[MAX_THREAD_NUM];
    for (int i = 0; i < MAX_THREAD_NUM; i++) {
        if (posix_memalign((void **) &X[i], 64, (idx + 1) * 8 * sizeof(double)) != 0) {
            perror("Memory allocation failed");
            exit(1);
        }
    }


    double t_tot = GetTime();
    long batch_size = MAX_THREAD_NUM * 1024 * 1024;
#pragma omp parallel
    {

        for (long batch_start = 0; batch_start < nx; batch_start += batch_size) {

            long batch_end = batch_start + batch_size;
            if (batch_end > nx) batch_end = nx;

            long nxx = batch_end - batch_start;

            int thread_id = omp_get_thread_num();
            int threads_count = omp_get_num_threads();
            long l_range = (nxx / threads_count) * thread_id;
            long r_range = (thread_id == threads_count - 1) ? nxx : (nxx / threads_count) * (thread_id + 1);

            l_range += batch_start;
            r_range += batch_start;

            double t0;
            t0 = GetTime();

            long si = l_range + term;
            while (((uintptr_t) &f[si]) % 64) si++;


            long ii;
            for (ii = si; ii + 8 <= r_range - term; ii += 8) {

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
                for (int o = 0; o < 8; o++) Xx[o] = 1;
                double *ff = &(f[ii]);
                for (int k = 1; k <= idx; k++) {
                    Xx = &(X[thread_id][k * 8]);
                    for (int o = 0; o < 8; o++) {
                        Xx[o] = Xx[o - 8] * ff[o];
                    }
                }
                Xx = &(X[thread_id][0]);
                for (int o = 0; o < 8; o++) Xx[o] = ff[o];

                double *ffa = &(fa[ii]);
                for (int j = 0; j <= idx; j++) {
                    double p_con = pr[term - idx + j - 1];
                    Xx = &(X[thread_id][j * 8]);
                    for (int o = 0; o < 8; o++) {
                        ffa[o + j] += Xx[o] * p_con;
                    }
                }
                //for(int j = term - idx - 1; j >= 1; j--) {
                for (int j = 1; j < term - idx; j++) {
                    double p_con = pr[term - idx - j - 1];
                    Xx = &(X[thread_id][j * 8]);
                    for (int o = 0; o < 8; o++) {
                        ffa[o - j] += Xx[o] * p_con;
                    }
                }
#endif
            }

            for (; ii < r_range - term; ii++) {
                double x = f[ii];
                X[thread_id][0] = 1;
                for (int k = 1; k <= idx; k++) {
                    X[thread_id][k] = X[thread_id][k - 1] * x;
                }
                X[thread_id][0] = x;
                for (int j = 0; j <= idx; j++) {
                    fa[ii + j] += X[thread_id][j] * pr[term - idx + j - 1];
                }
                for (int j = term - idx - 1; j >= 1; j--) {
                    fa[ii - j] += X[thread_id][j] * pr[term - idx - j - 1];
                }
            }

            time2[thread_id] += GetTime() - t0;

            //t0 = GetTime();
#pragma omp critical
            {

                for (long i = l_range; i < si; i++) {
                    double x = f[i];
                    X[thread_id][0] = 1;
                    for (int k = 1; k <= idx; k++) {
                        X[thread_id][k] = X[thread_id][k - 1] * x;
                    }
                    X[thread_id][0] = x;
                    for (int j = 0; j <= idx; j++) {
                        if (i + j < nx) fa[i + j] += X[thread_id][j] * p[idx - j];
                    }
                    for (int j = 1; j < term - idx; j++) {
                        if (i - j >= 0) fa[i - j] += X[thread_id][j] * p[idx + j];
                    }
                }

                for (long i = r_range - term; i < r_range; i++) {
                    double x = f[i];
                    X[thread_id][0] = 1;
                    for (int k = 1; k <= idx; k++) {
                        X[thread_id][k] = X[thread_id][k - 1] * x;
                    }
                    X[thread_id][0] = x;
                    for (int j = 0; j <= idx; j++) {
                        if (i + j < nx) fa[i + j] += X[thread_id][j] * p[idx - j];
                    }
                    for (int j = 1; j < term - idx; j++) {
                        if (i - j >= 0) fa[i - j] += X[thread_id][j] * p[idx + j];
                    }
                }

            }
            //time1[thread_id] += GetTime() - t0;


        }


    }
    printf("time total cost %lf\n", GetTime() - t_tot);
    double time1_tot = 0;
    double time2_max = 0;
    double time2_tot = 0;
    double time2_min = 1e9;
    int cnt = 0;
    for (int i = 0; i < MAX_THREAD_NUM; i++) {
        //if(time2[i] > 0) printf("thread %d time2 %lf\n", i, time2[i]);
        time1_tot += time1[i];
        time2_tot += time2[i];
        time2_max = MAX(time2[i], time2_max);
        if (time2[i] > 0) time2_min = MIN(time2[i], time2_min);
        if (time2[i] > 0) cnt++;
    }
    //printf("time1 total cost %lf\n", time1_tot);
    printf("time2 cost %lf %lf %lf\n", time2_tot / cnt, time2_max, time2_min);

    for (int i = 0; i < MAX_THREAD_NUM; i++) {
        free(X[i]);
    }
    free(pr);

}

# endif
#endif

