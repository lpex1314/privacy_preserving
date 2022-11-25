#include <cstring>
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>
#include <vector>
#include <math.h>
using namespace std;
extern "C"{
double alpha = 1;
int indexs[8+5]={1,11,8,9,4,1,4,2};
void softmax(double *input, int shape[], int dim){
    int N, C, H, W;
    if(dim==-1){
        dim=2;
    }
    N=shape[0];C=shape[1];H=shape[2];W=shape[3];
    int k1=W,k2=H*W,k3=C*H*W;
    int len = N * C * H * W;
    double out[N][C][H][W];
    int idx_n, idx_c, idx_h, idx_w;
    // if (dim==0){
    //     for (idx_h = 0;idx_h< H; idx_h++){
    //             for(idx_w=0;idx_w<W;idx_w++){
    //                 for(idx_c=0;idx_c<C;idx_c++){
    //                     double sum=0;
    //                     for(idx_n=0;idx_n<N;idx_n++){
    //                         sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
    //                     }
    //                     for(idx_n=0;idx_n<N;idx_n++){
    //                         out[idx_n][idx_c][idx_h][idx_w]=exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]) / sum;
    //                     }
    //                 }
                        
    //             }
    //         }
    // }
    if (dim==0){
        for(idx_n=0;idx_n<N;idx_n++){
            for (idx_h = 0;idx_h< H; idx_h++){
                for(idx_w=0;idx_w<W;idx_w++){
                    double sum=0;
                    for(idx_c=0;idx_c<C;idx_c++){
                        sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
                    }
                    for(idx_c=0;idx_c<C;idx_c++){
                        out[idx_n][idx_c][idx_h][idx_w]=exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]) / sum;
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
                        sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
                    }
                    for(idx_h=0;idx_h<H;idx_h++){
                        out[idx_n][idx_c][idx_h][idx_w] = exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w] )/ sum;
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
                        sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
                    }
                    for(idx_w=0;idx_w<W;idx_w++){
                        out[idx_n][idx_c][idx_h][idx_w] = exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w] )/ sum;
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
                    input[idx++]=out[i][j][k][l];
                }
            }
        }
    }
    return;
}
void d_softmax(double *y, double *result, int len, int len_out, int dim, int shape[]){
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
    if(dim==-1){
        dim=2;
    }
    // if (dim==0){
    //     double *tmp = (double*)malloc(sizeof(double) * N * N);
    //     for(idx_c=0;idx_c<C;idx_c++){
    //         for(idx_h=0;idx_h<H;idx_h++){
    //             for(idx_w=0;idx_w<W;idx_w++){

    //                 for(idx_n=0;idx_n<N;idx_n++){
    //                     tmp[idx_n*N + idx_n] = y[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
    //                     //diag[Y]
    //                 }
    //                 for(i=0;i<N;i++){
    //                     for(j=0;j<N;j++){
    //                         tmp[i*N + j] -= y[i*k3+idx_c*k2+idx_h*k1+idx_w]*y[j*k3+idx_c*k2+idx_h*k1+idx_w];
    //                         //DxY = diag(Y) - Y'*Y
    //                     }
    //                 }
    //                 //copy to result
    //                 for(i=0;i<N*N;i++){
    //                     result[glob_idx++] = tmp[i];
                        
    //                 }
    //                 //clear tmp
    //                 for(i=0;i<N*N;i++){
    //                     tmp[i] = 0;
    //                 }

    //             }
    //         }
    //     }
    //     free(tmp);
    // }

    if (dim==0){
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

    if (dim==1){
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

    if (dim==2){
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
                            //dy/dx = diag(Y) - Y'*Y
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
void d_batchnorm(double *x_hat, double *grad_out, double *running_mean, double *running_var, double *weight, double eps, double *delta, int len, int shape[], double *ret_grad_w, double *ret_grad_b){
    // x_hat, grad_out: len
    // weight,bias : C
    int N,C,H,W;
    int k1,k2,k3;
    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
    k3=C*H*W;k2=H*W;k1=W;
    // double * key = (double *)malloc(len*sizeof(double));
    // int i, j;
    // memset(key,0,sizeof(double)*len);
    // for(i=0;i<8;i++){
    //     for(j=0;j<len;j++){
    //         key[j]+=delta[indexs[i]*len+j];
    //     }
    // }
    // //key is f(r) now
    // for(j=0;j<len;j++){
    //     x_hat[j]=x_hat[j]-key[j]; // x_hat is decrypted now 
    //     grad_out[j] /= alpha;
    // }
    int i, j;
    double *grad_w = (double*)malloc(sizeof(double)*C*H*W);
    memset(grad_w,0,sizeof(double)*C*H*W);
    double tmp=0;
    for(int idx_c=0;idx_c<C;idx_c++){
        for(int idx_h=0;idx_h<H;idx_h++){
            for(int idx_w=0;idx_w<W;idx_w++){
                tmp = 0;
                for(int idx_n=0;idx_n<N;idx_n++){
                    
                    tmp += x_hat[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] * grad_out[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                }
                
                grad_w[idx_c*k2+idx_h*k1+idx_w] = tmp;
                
            }
        }
    }
    for(int idx_c=0;idx_c<C;idx_c++){
        tmp = 0;
        for(int idx_h=0;idx_h<H;idx_h++){
            for(int idx_w=0;idx_w<W;idx_w++){
                for(int idx_n=0;idx_n<N;idx_n++){
                    tmp += x_hat[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] * grad_out[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                }  
            }
        }
        ret_grad_w[idx_c] = tmp;
    }
    for(int idx_c=0;idx_c<C;idx_c++){
        tmp = 0;
        for(int idx_h=0;idx_h<H;idx_h++){
            for(int idx_w=0;idx_w<W;idx_w++){
                for(int idx_n=0;idx_n<N;idx_n++){
                    tmp += 1 * grad_out[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                }  
            }
        }
        ret_grad_b[idx_c] = tmp;
    }
    double *grad_b = (double*)malloc(sizeof(double)*C*H*W);
    memset(grad_b,0,sizeof(double)*C*H*W);
    for(int idx_c=0;idx_c<C;idx_c++){
        for(int idx_h=0;idx_h<H;idx_h++){
            for(int idx_w=0;idx_w<W;idx_w++){
                tmp = 0;
                for(int idx_n=0;idx_n<N;idx_n++){
                    tmp += 1 * grad_out[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                }
                grad_b[idx_c*k2+idx_h*k1+idx_w] = tmp;
            }
        }
    }
    double *coef_inp = (double*)malloc(sizeof(double)*C);
    for(int idx_c=0;idx_c<C;idx_c++){
        coef_inp[idx_c] = (double) 1 / N * weight[idx_c] / sqrt(running_var[idx_c] + eps);
    }    
    double *grad_w_expand = (double*)malloc(sizeof(double)*len);
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_c=0;idx_c<C;idx_c++){
            for(int idx_h=0;idx_h<H;idx_h++){
                for(int idx_w=0;idx_w<W;idx_w++){
                    grad_w_expand[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] = grad_w[idx_c*k2+idx_h*k1+idx_w];
                }
            }
        }
    }
    double *part1 = (double*)malloc(sizeof(double)*len);
    for(i=0;i<len;i++){
        part1[i] = -1 * grad_w_expand[i] * x_hat[i];
    }
    double *part2 = (double*)malloc(sizeof(double)*len);
    for(i=0;i<len;i++){
        part2[i] = N * grad_out[i];
    }
    double *part3 = (double*)malloc(sizeof(double)*len);
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_c=0;idx_c<C;idx_c++){
            for(int idx_h=0;idx_h<H;idx_h++){
                for(int idx_w=0;idx_w<W;idx_w++){
                    part3[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] = grad_b[idx_c*k2+idx_h*k1+idx_w];
                }
            }
        }
    }
    double *inp_expand = (double*)malloc(sizeof(double)*len);
    for(int idx_c=0;idx_c<C;idx_c++){
        for(int idx_h=0;idx_h<H;idx_h++){
            for(int idx_w=0;idx_w<W;idx_w++){
                for(int idx_n=0;idx_n<N;idx_n++){
                    inp_expand[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] = coef_inp[idx_c];
                }
            }
        }
    }
    double *grad_x= (double*)malloc(sizeof(double)*len);
    for(i=0;i<len;i++){
        x_hat[i] = inp_expand[i] * (part1[i] + part2[2] + part3[i]);
    }
    // for(int idx_c=0;idx_c<C;idx_c++){
    //     tmp=0;
    //     for(int idx_h=0;idx_h<H;idx_h++){
    //         for(int idx_w=0;idx_w<W;idx_w++){
    //             tmp += grad_w[idx_c*k2+idx_h*k1+idx_w];
                
    //         }
    //     }
    //     printf("%lf; ", tmp);
    //     ret_grad_w[idx_c] = tmp;
    // }
    // for(int idx_c=0;idx_c<C;idx_c++){
    //     tmp=0;
    //     for(int idx_h=0;idx_h<H;idx_h++){
    //         for(int idx_w=0;idx_w<W;idx_w++){
    //             tmp += grad_b[idx_c*k2+idx_h*k1+idx_w];
    //         }
    //     }
    //     ret_grad_b[idx_c] = tmp;
    // }
    return;
}
void d_softmax_easy(double *dy, double *y, double *result, int len, int dim, int shape[], double *delta){
    //dy:[N, classes] = d(loss)/dy
    //y:[N, classes] = softmax(x)
    //decrypt y first:
    //len = N * classes
    //result: Size of len
    double * key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        y[j]=y[j]-key[j];
    }
    for(j=0;j<len;j++){
        dy[j] = dy[j] / alpha;
    }
    int global_idx = 0;
    int N=shape[0], C=shape[1];
    double *tmp = (double*)malloc(sizeof(double) * C * C);
    for(int idx_n=0;idx_n<N;idx_n++){
        memset(tmp, 0, sizeof(double)*C*C);
        for(int idx_c=0;idx_c<C;idx_c++){
            tmp[idx_c*C+idx_c] = y[idx_n*C+idx_c]; // diag(y)
        }
        for(int i=0;i<C;i++){
            for(int j=0;j<C;j++){
                tmp[i*C+j] -= y[idx_n*C+i] * y[idx_n*C+j]; // dy/dx=diag(y)-y'.*y
            }
        }
        double temp=0;
        for(int i=0;i<C;i++){
            for(int j=0;j<C;j++){
                temp += dy[idx_n*C+j] * tmp[j*C+i]; // d(loss)/dx=d(loss)/dy .* dy/dx
            }
            result[global_idx++] = temp * alpha;
            temp=0;
        }
    }
    free(tmp);
    free(key);
    return;
}
void d_relu(double *dy, double*y, int len, double *result, double *delta2){
    //y, result: both sizes are len
    //dy=d(loss)/dy
    double *key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta2[indexs[i]*len+j];
        }
    }
    for(i=0;i<len;i++){
        dy[i] /= alpha;
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        y[j]=y[j]-key[j];
    }
    for(int i=0;i<len;i++){
        if(y[i]>0){
            result[i] = dy[i] * alpha;
        }
        else{
            result[i] = 0;
        }
    }
    // printf("%lf", alpha);
    free(key);
    return;
    
}

void d_sigmoid(double *y, int len, double *result){
    for(int i=0;i<len;i++){
        result[i] = y[i] * (1 - y[i]);
    }
    return;
}

void d_dropout(double *dy, double *y, int len, double p, int shape[], double *result, double *delta){
    double * key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        y[j]=y[j]-key[j];
        dy[i] /= alpha;
    }
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
                        result[idx_n*k3+idx_c*k2+idx_h*k1+idx_w]=(double)1 / (1-p) * dy[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] * alpha;
                    }
                }
            }
        }
    }
    free(key);
    return;
}
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
void d_crossEntropy(double *y_true, double *y_pred, int shape[], int len, double *result, int classes, double *delta1){
    double * key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta1[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        y_pred[j]=y_pred[j]-key[j];
    }
    double tmp=0;
    softmax_e(y_pred, shape, -1);
    int mean = len / classes; 
    for(int i=0;i<len;i++){
        result[i] = (y_pred[i] - y_true[i]) / mean; 
    }
    return;
}

void gen_alpha(double *delta){
    for(int i=0;i<64;i++){
        alpha += delta[indexs[i]];
    }
    return;
}
void d_max_pool_2d(double *dy, double *result, int len_y, int len_x, int argmax[]){
    //len_y是经过正向传播后的大小=N*C*H_out*W_out
    //len_x=N*C*H*W
    //y:len_y
    //result:len_x

    int N,C,H,W;
    for(int i=0;i<len_x;i++)
        result[i] = 0;
    for(int i=0;i<len_y;i++){
        dy[i] = dy[i] / alpha;
    }
    for(int i=0;i<len_y;i++){
        result[argmax[i]] = dy[i] * alpha;
    }
    return;
}
void encrypt(double *dy, double *result, int len){
    /*
        int indexs[64];
        sgx_status_t SGXAPI sgx_read_rand(indexs, 64);
    */
    //delta[256] double
    for(int i=0;i<len;i++){
        result[i] = dy[i] * alpha;
    }
    return;
}

void decrypt_conv(double *dw, double *delta, double *dy, int len_x, int shape_w[], int H_in, int H_out, int Batch_size){
    /*
        gout = d(loss) / dy y = w.*x
        if y:[*,r,p], X[*,q,p]
        len=N*C*q*p length of x
        shape = shape of gout[N,C,r,p]
    */
    int i,j;
    int kernel_num = shape_w[0], depth=shape_w[1], kernel_size = shape_w[2], N = Batch_size;
    double *key = (double*)malloc(sizeof(double)*len_x);
    memset(key,0,sizeof(double)*len_x);
    for(i=0;i<8;i++){
        for(j=0;j<len_x;j++){
            key[j]+=delta[indexs[i]*len_x+j];
        }
    }
    double grad_tmp = 0;
    double *weight = (double*)malloc(sizeof(double)*kernel_num*depth*kernel_size*kernel_size);
    int k3=kernel_num*H_out*H_out, k2=H_out*H_out, k1=H_out;
    int y3=depth*H_in*H_in, y2=H_in*H_in, y1=H_in;
    int x3=depth*kernel_size*kernel_size, x2=kernel_size*kernel_size, x1=kernel_size;
    memset(weight,0,sizeof(double)*kernel_num*depth*kernel_size*kernel_size);
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_k=0;idx_k<kernel_num;idx_k++){
            for (int idx_d = 0; idx_d < depth; idx_d++){
                for (int idx_h = 0; idx_h < kernel_size; idx_h++){
                    for (int idx_w = 0; idx_w < kernel_size; idx_w++){
                        grad_tmp = 0;
                        for (i = 0; i < H_out; i++){
                            for (j = 0; j < H_out; j++){
                                grad_tmp += dy[idx_n*k3+idx_k*k2+i*k1+j] * key[idx_n*y3+idx_d*y2+(i+idx_h)*y1+(j+idx_w)];
                            }
                            
                        }
                        weight[idx_k*x3+idx_d*x2+idx_h*x1+idx_w] += grad_tmp / (double)N;
                    }

                }
                
            }
            
        }
    }
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_k=0;idx_k<kernel_num;idx_k++){
            for (int idx_d = 0; idx_d < depth; idx_d++){
                for (int idx_h = 0; idx_h < kernel_size; idx_h++){
                    for (int idx_w = 0; idx_w < kernel_size; idx_w++){
                        dw[idx_k*x3+idx_d*x2+idx_h*x1+idx_w] -= weight[idx_k*x3+idx_d*x2+idx_h*x1+idx_w];
                        dw[idx_k*x3+idx_d*x2+idx_h*x1+idx_w] /= alpha;
                    }

                }
                
            }
            
        }
    }
    free(key);
    free(weight);
    return;
}

void decrypt_linear(double *delta, double *gout, double *dw, int len, int shape[]){
    /*
        gout = d(loss) / dy y = x.*w
        len=N*d length of x
        shape = shape of gout[N, c] c: num_of_classes
    */
    int i,j;
    int N = shape[0], C = shape[1];
    int D = len / N;
    double *key = (double*)malloc(sizeof(double)*len);
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];
        }
    }
    // vector<vector<vector<vector<float>>>> noise(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W,0))));
    double *w_update = (double*)malloc(sizeof(double)*D*C);
    memset(w_update,0,sizeof(double)*D*C);
    int k1,k2,k3;
    double tmp;
    for (int idx_n = 0; idx_n < N; idx_n++){
        for(i=0;i<D;i++){
            for(j=0;j<C;j++){
                tmp = key[idx_n*D + i] * gout[idx_n*C + j];
                w_update[i*C+j] += tmp / N;
            }
        }
    }
    
    for(i=0;i<D;i++){
        for(j=0;j<C;j++){
            dw[i*C+j] -= w_update[i*C+j];
            dw[i*C+j] /= alpha;
        }
    }
    free(key);
    free(w_update);
    return;
}
}