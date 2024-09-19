#include <stdlib.h>

void polynomial_stencil(double *fa, double *f, long nx, double p[], int term)
{
    int idx = term / 2;

    long i;
    int j;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < term; j++) {
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
