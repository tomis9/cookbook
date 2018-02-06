#include <R.h> 
#include <Rmath.h> 
#include <math.h>
#include <stdio.h>

void fun(double *X, int *p, int *n, double *phi, double *w) {
    int i, j, k, l;
    double sos;
    k = 0;
    for (i=0; i<*n-1; i++) {
        for (j=i+1; j<*n; j++) {
            sos = 0.;
            for (l=0; l<*p; l++) {
                sos += pow(X[l + (*p) * i] - X[l + (*p) * j],2.);
            }
            w[k] = exp(-(*phi) * sos);
            k += 1;
        }
    }
}

