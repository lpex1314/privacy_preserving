#include <cstring>
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>
#include <vector>
#include <math.h>
using namespace std;
void relu_c(double *input,int len) {
    for(int j=0;j<len;j++){
        input[j]=input[j]>=0?input[j]:0;
        // printf("%f ",input[j]);
    }
//    printf("relu OK\n");
    return ;
}
// softmax
void softmax_c(double *input, int shape[], int dim, double *output){
    int N, C, H, W;
    N=shape[0];C=shape[1];H=shape[2];W=shape[3];
    int k1=W,k2=H*W,k3=C*H*W;
    vector<vector<vector<vector<double> > > > out(N, vector<vector<vector<double> > >(C, vector<vector<double> >(H, vector<double>(W,0))));
    int idx_n, idx_c, idx_h, idx_w;
    if (dim==0){
        for(idx_n=0;idx_n<N;idx_n++){
            for (idx_h = 0;idx_h< H; idx_h++){
                for(idx_w=0;idx_w<W;idx_w++){
                    double sum=0;
                    for(idx_c=0;idx_c<C;idx_c++){
                        sum += input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w];
                    }
                    for(idx_c=0;idx_c<C;idx_c++){
                        out[idx_n][idx_c][idx_h][idx_w]=input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w] / sum;
                    }
                }
            }
            
        }
    }
    else if (dim==1){
        for(idx_n=0;idx_n<N;idx_n++){
            for (idx_c = 0;idx_c< C; idx_c++){
                for(idx_w=0;idx_w<W;idx_w++){
                    double sum=0;
                    for(idx_h=0;idx_h<H;idx_h++){
                        sum += input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w];
                    }
                    for(idx_h=0;idx_h<H;idx_h++){
                        out[idx_n][idx_c][idx_h][idx_w]=input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w] / sum;
                    }
                }
            }
            
        }
    }
    else if (dim==2){
        for(idx_n=0;idx_n<N;idx_n++){
            for (idx_c = 0;idx_c< C; idx_c++){
                for(idx_h=0;idx_h<H;idx_h++){
                    double sum=0;
                    for(idx_w=0;idx_w<W;idx_w++){
                        sum += input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w];
                    }
                    for(idx_w=0;idx_w<W;idx_w++){
                        out[idx_n][idx_c][idx_h][idx_w]=input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w] / sum;
                    }
                }
            }
            
        }
    }
    int idx=0;
    for(int i=0;i<N;i++){
        for(int j=0;j<C;j++){
            for(int k=0;k<H;k++){
                for(int l=0;l<W;l++){
                    output[idx++]=out[i][j][k][l];
                }
            }
        }
    }
    return;
}
// sigmoid
void sigmoid_c(double *input, int len){
    for(int j=0;j<len;j++){
        input[j] = 1.0 / (1.0 + exp(-input[j]));
    }
    return;
}


void max_pool_2d_c(double *input, int shape[],int kernel_size, int stride, double* out)
{
    //shape是input的形状
    int N,C,H,W;
    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
    int H_out = 1 + (H - kernel_size) / stride;
    int W_out = 1 + (W - kernel_size) / stride;
    int k1=W,k2=H*W,k3=C*H*W;
    // out = (double*)malloc(N*C*H*W*sizeof(double));
    vector<vector<vector<vector<double> > > > output(N, vector<vector<vector<double> > >(C, vector<vector<double> >(H_out, vector<double>(W_out,0))));
    // 这是out的形状，但out的形状同样需要在python函数中计算出来并赋予相应的空间
    for(int idx_N=0;idx_N<N;idx_N++){
        for(int idx_c=0;idx_c<C;idx_c++){
            for(int idx_h=0;idx_h<H_out;idx_h++){
                for(int idx_w=0;idx_w<W_out;idx_w++){
                    int start_h = idx_h * stride;
                    int start_w = idx_w * stride;
                    int end_h = start_h + kernel_size;
                    int end_w = start_w + kernel_size;
                    double local_max=-1e7;
                    for(int idx_hh=start_h;idx_hh<end_h;idx_hh++){
                        for(int idx_ww=start_w;idx_ww<end_w;idx_ww++){
                            double tmpp = input[idx_N*k3+idx_c*k2+idx_hh*k1+idx_ww];
                            local_max = local_max>tmpp?local_max:tmpp;
                        }
                    }
                    output[idx_N][idx_c][idx_h][idx_w] = local_max;
                }
            }

        }
    }

    int idx=0;
    for(int i=0;i<N;i++){
        for(int j=0;j<C;j++){
            for(int k=0;k<H_out;k++){
                for(int l=0;l<W_out;l++){
                    out[idx++]=output[i][j][k][l];
                }
            }
        }
    }

    return;
}

// crossEntropyloss
double cross_entropy_c(double *y_true, double *y_pred, int length){
    double sum =0;
    for(int i=0;i<length;i++){
        y_pred[i] = y_pred[i] < 1? y_pred[i]:0.9999;
        y_pred[i] = y_pred[i] > 0? y_pred[i]:0.0001;
        sum += y_true[i] * log2(y_pred[i]) + (1 - y_true[i]) * log2(1-y_pred[i]);
    }
    return -sum;
}
// MSE loss
double MSE_c(double *a, double *b, int length, char size_average[]){
    char mode_mean[5] = "mean";

    double *loss = (double*)malloc(sizeof(double) * length);
    double result=0;
    for(int i=0;i<length;i++){
        result += (a[i]-b[i]) * (a[i] - b[i]);
    }
    if(strcmp(size_average, mode_mean)==0){
        return result / (double)length;
    }
    else{
        return result;
    }
}
// dropout p: discard rate
void dropout_c(double *input, double p, int *shape){
    int m = 2147483647;
    int a = 75;
    int c = 0;
    int N,C,H,W;
    int random = 10;
    double rand;

    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
////    printf("n:%d,c:%d,h:%d,w:%d",N,C,H,W);
    int k1=W,k2=H*W,k3=C*H*W;
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_c=0;idx_c<C;idx_c++){
            random = (a * random) % m; // linear PRNG
            rand = (double)random / (double)m;
            if(rand<p){
                for(int idx_h=0;idx_h<H;idx_h++){
                    for (int idx_w = 0; idx_w < W; idx_w++){
                        input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w]=0;
                    }
                }
            }
            else
                continue;
        }
    }
    return ;
}
extern "C"{

void relu(double *input,int len){
return relu_c(input,len);
}

void softmax(double *input, int shape[], int dim, double *output)
{
return softmax_c(input,shape, dim, output);
}

void sigmoid(double *input, int len)
{
return sigmoid_c(input, len);
}

void max_pool_2d(double *input,int shape[],int kernel_size,int stride,double* out)
{
return max_pool_2d_c(input,shape,kernel_size,stride,out);
}

double cross_entropy(double *y_true, double *y_pred, int length)
{
return cross_entropy_c(y_true, y_pred, length);
}
double MSE(double *a, double *b, int length, char size_average[])
{
return MSE_c(a, b, length, size_average);

}

void dropout(double *input, double p, int shape[])
{
return dropout_c(input, p, shape);
}





}