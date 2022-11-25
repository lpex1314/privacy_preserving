#include <cstring>
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>
#include <vector>
#include <math.h>
using namespace std;
extern "C"{
void dropout(double *input, double p, int shape[]){
    int m = 214748387;
    int a = 75;
    int c = 0;
    int N,C,H,W;
    int random = 10;
    double rand;
    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
    int k1=W,k2=H*W,k3=C*H*W;
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_c=0;idx_c<C;idx_c++){
            random = (a * random + c) % m; // linear PRNG
            rand = (double)random / (double)m;
            if(rand<p){
                for(int idx_h=0;idx_h<H;idx_h++){
                    for (int idx_w = 0; idx_w < W; idx_w++){
                        input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] *=0;
                    }
                }
            }
            else{
                for(int idx_h=0;idx_h<H;idx_h++){
                    for (int idx_w = 0; idx_w < W; idx_w++){
                        input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w]=input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] / (1-p);
                    }
                }
            }
        }
    }
    return;
}
void d_dropout(double *dy, double *y, int len, double p, int shape[], double *result){
    int i, j;
    int N, C, H, W;
    N = shape[0];
    C = shape[1];
    H = shape[2];
    W = shape[3];
    int k1=W,k2=H*W,k3=C*H*W;
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_c=0;idx_c<C;idx_c++){
            double eps = 1e-6;
            double test_eps1 = fabs(y[idx_n*k3+idx_c*k2]-0);
            double test_eps2 = fabs(y[idx_n*k3+idx_c*k2+1]-0);
            double test_eps3 = fabs(y[idx_n*k3+idx_c*k2+2]-0);
            if (test_eps1 < eps && test_eps2 < eps && test_eps3 < eps){
                for(int idx_h=0;idx_h<H;idx_h++){
                    for (int idx_w = 0; idx_w < W; idx_w++){
                        result[idx_n*k3+idx_c*k2+idx_h*k1+idx_w]=0;
                    }
                }
            }
            else{
                for(int idx_h=0;idx_h<H;idx_h++){
                    for (int idx_w = 0; idx_w < W; idx_w++){
                        result[idx_n*k3+idx_c*k2+idx_h*k1+idx_w]=(double)1 / (1-p) * dy[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                    }
                }
            }
        }
    }
    return;
}
}