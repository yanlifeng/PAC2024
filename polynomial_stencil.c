#include <stdlib.h>
//#define use_init_code

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

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int threads_count = omp_get_num_threads();
        long l_range = (nx / threads_count) * thread_id;
        long r_range = (thread_id == threads_count - 1) ? nx : (nx / threads_count) * (thread_id + 1);


        if(r_range - l_range <= term * 2) {
        //if(1) {

            for (long i = l_range; i < r_range; i++) {
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

        } else {

            for (long i = l_range; i < l_range + term; i++) {
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

            for (long i = r_range - term; i < r_range; i++) {
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


            double X[term][idx + 1];
            long start_pos = l_range + term - idx;
            for(long i = start_pos; i < start_pos + term; i++) {
                double x = f[i];
                X[i - start_pos][0] = 1;
                for(int j = 1; j < idx + 1; j++) {
                    X[i - start_pos][j] = X[i - start_pos][j - 1] * x;
                }
                X[i - start_pos][0] = x;
            }

            int begin_pos = 0;

            for (long i = l_range + term; i < r_range - term; i++) {
                // need line [i-idx, i+term-idx)
                double sum = 0;
                int target_pos = begin_pos;
                for (int j = 0; j < idx; j++) {
                    sum += X[target_pos][idx - j] * p[j];
                    target_pos++;
                    if(target_pos == term) target_pos = 0;
                }
                for (int j = idx; j < term; j++) {
                    sum += X[target_pos][j - idx] * p[j];
                    target_pos++;
                    if(target_pos == term) target_pos = 0;
                }
                begin_pos++;
                if(begin_pos == term) begin_pos = 0;
                fa[i] = sum;
                // remove line 'i-idx', add line 'i+term-idx-1'
                int add_pos = target_pos;
                double x = f[i + term - idx];
                X[add_pos][0] = 1;
                for(int j = 1; j < idx + 1; j++) {
                    X[add_pos][j] = X[add_pos][j - 1] * x;
                }
                X[add_pos][0] = x;
            }

        }
    }
}
#endif
