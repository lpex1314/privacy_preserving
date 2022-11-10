#include<stdio.h>
#include<stdlib.h>
#include<math.h>
void softmax_e(double *input, int shape[], int dim){
    dim = 1;
    int N=shape[0], C=shape[1];
    int k2=C;
    double sum=0;
    double max=0;
    for(int idx_n=0;idx_n<N;idx_n++){
        sum = 0;
        max = -1e7;
        for(int idx_c=0;idx_c<C;idx_c++){
            max = input[idx_n*k2 + idx_c] > max? input[idx_n*k2 + idx_c]: max;
        }
        for(int idx_c=0;idx_c<C;idx_c++){
            input[idx_n*k2 + idx_c] -= max;
        }
        for(int idx_c=0;idx_c<C;idx_c++){
            sum += exp(input[idx_n*k2 + idx_c]);
        }
        for(int idx_c=0;idx_c<C;idx_c++){
            input[idx_n*k2 + idx_c] = exp(input[idx_n*k2 + idx_c]) / sum;
        }
    }
    return;
}
int main(){
    double a[640];
    int shape[2] = {64, 10};
    for(int i=0;i<640;i++){
        a[i] = (double) i / 10;
    }
    softmax_e(a, shape, 1);
    for(int i=0;i<640;i++){
        printf("%lf, ", a[i]);
    }
    return 0;
}