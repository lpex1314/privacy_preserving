#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<vector>
using namespace std;
double alpha=0;
int indexs[64+5]={11,28,5,45,39,6,23,1,37,32,12,4,24,12,58,12,61,2,29,60,5,44,6,36,20,27,13,16,39,61,49,51,25,54,32,0,60,55,1,33,23,14,37,47,26,31,60,24,34,25,20,39,5,26,11,25,53,25,42,28,22,27,15,47};

void d_softmax(double *y, double *result, int len,  int dim, int shape[]){
    //if dim = 0
    //then size_result = C*H*W*(N*N)(len_out)
    //size_y:len=N*C*H*W
    //only y is needed
    int N, C, H, W;
    N = shape[0];
    C = shape[1];
    H = shape[2];
    W = shape[3];
    int k1=W,k2=H*W,k3=C*H*W;
    int idx_c, idx_n, idx_h, idx_w, i, j;
    int glob_idx=0;
    
    if (dim==0){
        double *tmp = (double*)malloc(sizeof(double) * N * N);
        for(idx_c=0;idx_c<C;idx_c++){
            for(idx_h=0;idx_h<H;idx_h++){
                for(idx_w=0;idx_w<W;idx_w++){

                    for(idx_n=0;idx_n<N;idx_n++){
                        tmp[idx_n*N + idx_n] = y[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                        //diag[Y]
                    }
                    for(i=0;i<N;i++){
                        for(j=0;j<N;j++){
                            tmp[i*N + j] -= y[i*k3+idx_c*k2+idx_h*k1+idx_w]*y[j*k3+idx_c*k2+idx_h*k1+idx_w];
                            //DxY = diag(Y) - Y'*Y
                        }
                    }
                    //copy to result
                    for(i=0;i<N*N;i++){
                        result[glob_idx++] = tmp[i];
                        
                    }
                    //clear tmp
                    for(i=0;i<N*N;i++){
                        tmp[i] = 0;
                    }

                }
            }
        }
        free(tmp);
    }

    if (dim==1){
        double *tmp = (double*)malloc(sizeof(double) * C * C);
        for(idx_n=0;idx_n<N;idx_n++){
            for(idx_h=0;idx_h<H;idx_h++){
                for(idx_w=0;idx_w<W;idx_w++){

                    for(idx_c=0;idx_c<C;idx_c++){
                        tmp[idx_c*C+idx_c] = y[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                        //diag[Y]
                    }
                    for(i=0;i<C;i++){
                        for(j=0;j<C;j++){
                            tmp[i*C+j] -= y[idx_n*k3+i*k2+idx_h*k1+idx_w]*y[idx_n*k3+j*k2+idx_h*k1+idx_w];
                            //DxY = diag(Y) - Y'*Y
                        }
                    }
                    //copy to result
                    for(i=0;i<C*C;i++){
                        result[glob_idx++] = tmp[i];
                    }
                    //clear tmp
                    for(i=0;i<C*C;i++){
                        tmp[i] = 0;
                    }

                }
            }
        }
        free(tmp);
    }

    if (dim==2){
        double *tmp = (double*)malloc(sizeof(double) * H * H);
        for(idx_n=0;idx_n<N;idx_n++){
            for(idx_c=0;idx_c<C;idx_c++){
                for(idx_w=0;idx_w<W;idx_w++){

                    for(idx_h=0;idx_h<H;idx_h++){
                        tmp[idx_h*H+idx_h] = y[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                        //diag[Y]
                    }
                    for(i=0;i<H;i++){
                        for(j=0;j<H;j++){
                            tmp[i*H+j] -= y[idx_n*k3+idx_c*k2+i*k1+idx_w]*y[idx_n*k3+idx_c*k2+j*k1+idx_w];
                            //DxY = diag(Y) - Y'*Y
                        }
                    }
                    //copy to result
                    for(i=0;i<H*H;i++){
                        result[glob_idx++] = tmp[i];
                    }
                    //clear tmp
                    for(i=0;i<H*H;i++){
                        tmp[i] = 0;
                    }

                }
            }
        }
        free(tmp);
    }

    if (dim==3){
        double *tmp = (double*)malloc(sizeof(double) * W * W);
        for(idx_n=0;idx_n<N;idx_n++){
            for(idx_c=0;idx_c<C;idx_c++){
                for(idx_h=0;idx_h<H;idx_h++){

                    for(idx_w=0;idx_w<W;idx_w++){
                        tmp[idx_w*W+idx_w] = y[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                        //diag[Y]
                    }
                    for(i=0;i<W;i++){
                        for(j=0;j<W;j++){
                            tmp[i*W+j] -= y[idx_n*k3+idx_c*k2+idx_h*k1+i]*y[idx_n*k3+idx_c*k2+idx_h*k1+j];
                            //DxY = diag(Y) - Y'*Y
                        }
                    }
                    //copy to result
                    for(i=0;i<W*W;i++){
                        result[glob_idx++] = tmp[i];
                    }
                    //clear tmp
                    for(i=0;i<W*W;i++){
                        tmp[i] = 0;
                    }


                }
            }
        }
        free(tmp);
    }

    return;

}

void d_relu(double *y, int len, double *result){
    //y, result: both sizes are len
    for(int i=0;i<len;i++){
        if(y[i]>0){
            result[i] = 1;
        }
        else{
            result[i] = 0;
        }
    }
}

void d_sigmoid(double *y, int len, double *result){
    for(int i=0;i<len;i++){
        result[i] = y[i] * (1 - y[i]);
    }
    return;
}
extern "C"
{
void sigmoid(double *y, int len, double *result)
{
return d_sigmoid(y, len, result);
}
void relu(double *y, int len, double *result)
{
return d_relu(y, len, result);
}
void softmax(double *y, double *result, int len,  int dim, int shape[])
{
return d_softmax(y, result, len, dim, shape);
}
}