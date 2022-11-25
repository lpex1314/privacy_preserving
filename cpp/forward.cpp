#include <cstring>
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>
#include <vector>
#include <math.h>
using namespace std;
extern "C" {
int indexs[8+5]={1,11,8,9,4,1,4,2};
void relu(double *input,int len){
    for(int j=0;j<len;j++){
        input[j]=input[j]>=0?input[j]:0;
    }
    return;
}
// void softmax(double *input, int shape[], int dim){
//     int N, C, H, W;
//     N=shape[0];C=shape[1];H=shape[2];W=shape[3];
//     int k1=W,k2=H*W,k3=C*H*W;
//     int len = N * C * H * W;
//     double out[N][C][H][W];
//     int idx_n, idx_c, idx_h, idx_w;
//     if(dim==-1){
//         dim=2;
//     }
//     // if (dim==0){
//     //     for (idx_h = 0;idx_h< H; idx_h++){
//     //             for(idx_w=0;idx_w<W;idx_w++){
//     //                 for(idx_c=0;idx_c<C;idx_c++){
//     //                     double sum=0;
//     //                     for(idx_n=0;idx_n<N;idx_n++){
//     //                         sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
//     //                     }
//     //                     for(idx_n=0;idx_n<N;idx_n++){
//     //                         out[idx_n][idx_c][idx_h][idx_w]=exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]) / sum;
//     //                     }
//     //                 }
                        
//     //             }
//     //         }
//     // }
//     if (dim==0){
//         for(idx_n=0;idx_n<N;idx_n++){
//             for (idx_h = 0;idx_h< H; idx_h++){
//                 for(idx_w=0;idx_w<W;idx_w++){
//                     double sum=0;
//                     for(idx_c=0;idx_c<C;idx_c++){
//                         sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
//                     }
//                     for(idx_c=0;idx_c<C;idx_c++){
//                         out[idx_n][idx_c][idx_h][idx_w]=exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]) / sum;
//                     }
//                 }
//             }
            
//         }
//     }
//     else if (dim==1){
//         for(idx_n=0;idx_n<N;idx_n++){
//             for (idx_c = 0;idx_c< C; idx_c++){
//                 for(idx_w=0;idx_w<W;idx_w++){
//                     double sum=0;
//                     for(idx_h=0;idx_h<H;idx_h++){
//                         sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
//                     }
//                     for(idx_h=0;idx_h<H;idx_h++){
//                         out[idx_n][idx_c][idx_h][idx_w] = exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w] )/ sum;
//                     }
//                 }
//             }
            
//         }
//     }
//     else if (dim==2){
//         for(idx_n=0;idx_n<N;idx_n++){
//             for (idx_c = 0;idx_c< C; idx_c++){
//                 for(idx_h=0;idx_h<H;idx_h++){
//                     double sum=0;
//                     for(idx_w=0;idx_w<W;idx_w++){
//                         sum += exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w]);
//                     }
//                     for(idx_w=0;idx_w<W;idx_w++){
//                         out[idx_n][idx_c][idx_h][idx_w] = exp(input[idx_n*k3 + idx_c*k2 + idx_h*k1 + idx_w] )/ sum;
//                     }
//                 }
//             }
            
//         }
//     }
//     int idx=0;
//     for(int i=0;i<N;i++){
//         for(int j=0;j<C;j++){
//             for(int k=0;k<H;k++){
//                 for(int l=0;l<W;l++){
//                     input[idx++]=out[i][j][k][l];
//                 }
//             }
//         }
//     }
//     return;
// }
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
void ecall_batchnorm2d(double *input, double eps, double momentum, int mode, int shape[], double *running_mean, double *running_var, double *delta1, double *delta2, int len){
    //running
    // double * key = (double *)malloc(len*sizeof(double));
    // int i, j;
    // memset(key,0,sizeof(double)*len);
    // for(i=0;i<8;i++){
    //     for(j=0;j<len;j++){
    //         key[j]+=delta1[indexs[i]*len+j];
    //     }
    // }
    // //key is f(r) now
    // for(j=0;j<len;j++){
    //     input[j]=input[j]-key[j];
    // }
    /* 
        BN
    */
    int N,C,H,W;
    int k1,k2,k3;
    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
    k3=C*H*W;k2=H*W;k1=W;
    // double *mean = (double*)malloc(sizeof(double)*C);
    // double *var = (double*)malloc(sizeof(double)*C);
    // memset(mean, 0, sizeof(double)*C);
    // memset(var, 0, sizeof(double)*C);
    double mean_tmp=0;
    double mean_2_tmp = 0;
    double var_tmp=0;
    double std;
    if(mode==1){ // train
        for(int idx_c=0;idx_c<C;idx_c++){
            mean_tmp = 0;
            var_tmp = 0;
            mean_2_tmp = 0;
            std = 0;
            for(int idx_n=0;idx_n<N;idx_n++){
                for(int idx_h=0;idx_h<H;idx_h++){
                    for(int idx_w=0;idx_w<W;idx_w++){
                        mean_tmp += input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] / (double) (N*H*W);
                        mean_2_tmp += input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] * input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] / (double) (N*H*W);
                    }
                }
            }
            var_tmp = mean_2_tmp - mean_tmp * mean_tmp ;
            running_mean[idx_c] = momentum * running_mean[idx_c] + (1 - momentum) * mean_tmp;
            running_var[idx_c] = momentum * running_var[idx_c] + (1 - momentum) * var_tmp;
            std = sqrt(var_tmp + eps);
            for(int idx_n=0;idx_n<N;idx_n++){
                for(int idx_h=0;idx_h<H;idx_h++){
                    for(int idx_w=0;idx_w<W;idx_w++){
                        input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] = (input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] - mean_tmp) / std;
                    }
                }
            }
        }
    }
    if(mode==0){// test
        for(int idx_c=0;idx_c<C;idx_c++){
            var_tmp = running_var[idx_c];
            mean_tmp = running_mean[idx_c];
            std = sqrt(var_tmp + eps);
            for(int idx_n=0;idx_n<N;idx_n++){
                for(int idx_h=0;idx_h<H;idx_h++){
                    for(int idx_w=0;idx_w<W;idx_w++){
                        input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] = (input[idx_n*k3+idx_c*k2+idx_h*k1+idx_w] - mean_tmp) / std;
                    }
                }
            }
        }
    }
    /*
        Encrypt
    */
    // memset(key,0,sizeof(double)*len);
    // for(i=0;i<8;i++){
    //     for(j=0;j<len;j++){
    //         key[j]+=delta2[indexs[i]*len+j];
    //     }
    // }
    // //key is f(r) now
    // for(j=0;j<len;j++){
    //     input[j]=input[j]+key[j];
    // }
    return;
}
double cross_entropy(double *y_true, double *y_pred, int length, int classes, int shape[]){
    double sum =0;
    // softmax_e(y_pred, shape, -1);
    for(int i=0;i<length;i++){
        y_pred[i] = y_pred[i] < 1? y_pred[i]:0.9999;
        y_pred[i] = y_pred[i] > 0? y_pred[i]:0.0001;
        sum += y_true[i] * log(y_pred[i]);
    }
    int lens =  length / classes;
    return -sum / (double) lens;
}
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
void sigmoid(double *input, int len){
    for(int j=0;j<len;j++){
        input[j] = 1.0 / (1.0 + exp(-input[j]));
    }
    return;
}
double MSE(double *a, double *b, int length, int mode){
    double result=0;
    for(int i=0;i<length;i++){
        result += (a[i]-b[i]) * (a[i] - b[i]);
    }
    if(mode==0){
        return result / (double)length;
    }
    else{
        return result;
    }
}
void max_pool_2d(double *input, int shape[],int kernel_size, int stride, double* out, int* maxarg)
{
    //shape是input的形状
    //maxarg数组记录滑动窗口中最大值出现的位置，maxarg形状与out相同[N,C,H_out,W_out]
    int N,C,H,W;
//    printf("%f\n",input[0]);
    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
    int H_out = 1 + (H - kernel_size) / stride;
    int W_out = 1 + (W - kernel_size) / stride;
    int k1=W,k2=H*W,k3=C*H*W;
    int arg_ind=0;
    // out = (double*)malloc(N*C*H*W*sizeof(double));
    vector<vector<vector<vector<double>>>> output(N, vector<vector<vector<double>>>(C, vector<vector<double>>(H_out, vector<double>(W_out,0))));
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
                            if(tmpp > local_max){
                                local_max = tmpp;
                                maxarg[arg_ind] = idx_N*k3+idx_c*k2+idx_hh*k1+idx_ww;
                            }
                            
                        }
                    }
                    arg_ind++;
                    output[idx_N][idx_c][idx_h][idx_w] = local_max;
                }
            }
            // int idx_yy = 0;
            // for(int idx_y=0;idx_y<H_out;idx_y++){
            //     int idx_xx = 0;
            //     for(int idx_x=0;idx_x<W_out;idx_x++){
            //         // window = x[idx_N, idx_c, idx_yy:idx_yy+pool_height, idx_xx:idx_xx+pool_width];
            //         double local_max=-1e7;
            //         for(int tmp_i=0;tmp_i<kernel_size;tmp_i++){
            //             for(int tmp_j=0;tmp_j<kernel_size;tmp_j++){
            //                 double tmpp = input[idx_N*k3+idx_c*k2+(idx_yy+tmp_i)*k1+idx_xx+tmp_j];
            //                 // printf("%d ",idx_N*k3+idx_c*k2+(idx_yy+tmp_i)*k1+idx_xx+tmp_j);
            //                 // printf("%f ",tmpp);
            //                 local_max = local_max>tmpp?local_max:tmpp;
            //             }
            //         }
            //         // out[idx_N, idx_c, idx_y, idx_x] = np.max(window);
            //         output[idx_N][idx_c][idx_y][idx_x] = local_max;
            //         // printf("%f ",local_max);
            //         idx_xx += stride;
            //     }
            //     idx_yy += stride;
            // }  
        }
    }     
//    printf("\n");
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
//    printf("max_pool OK\n");
    return;
}

void ecall_encrypt(double *delta, double *input, int len){
    //delta is key_list with 256*len elements
    //input size : len
    int i, j;
    double * key = (double *)malloc(len*sizeof(double));
    memset(key,0,sizeof(double)*len);
    // make r
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];

        }
    }
    // key is r(noises) now
    for(j=0;j<len;j++){
        //x + r 
        input[j]=input[j]+key[j];
    }
    free(key);
    return;
}
void ecall_sigmoid(double *f_x_r, double *f_r, int len, double *input, double *delta){
    double * key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=f_r[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        input[j]=f_x_r[j]-key[j];
    }
    sigmoid(input, len);
    memset(key,0,sizeof(double)*len);
    // make r
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];

        }
    }
    // key is r(noises) now
    for(j=0;j<len;j++){
        //x + r 
        input[j]=input[j]+key[j];
    }
    // ecall_encrypt(delta, x, len);
    free(key);
    return;
}
void ecall_decrypt(double *f_x_r, double *f_r, int len) {
	double * key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=f_r[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        f_x_r[j]=f_x_r[j]-key[j];
    }
    free(key);
    return ;
}
void ecall_relu(double *f_x_r, double *f_r, double *input, int len, double *delta){
//    printf("fr[0]:%f x[0]:%f\n",f_r[0],x[0]);
    // ecall_decrypt(f_r, x, len);
    double *tmp_res = (double*)malloc(sizeof(double)*len);
    double *key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=f_r[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        tmp_res[j]=f_x_r[j]-key[j];
    }
//    printf("x[0]:%f\n",x[0]);
    // for(j=0;j<len;j++){
    //     //x + r 
    //     // input[j]=tmp_res[j]+key[j];
    //     printf("%lf, ", tmp_res[j]);
    // }
    relu(tmp_res,len);
    memset(key,0,sizeof(double)*len);
    // make r
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];

        }
    }
    // key is r(noises) now
    for(j=0;j<len;j++){
        //x + r 
        input[j]=tmp_res[j]+key[j];
        // printf("%lf, ", tmp_res[j]);
    }
    // ecall_encrypt(delta, x, len);
    free(key);
    free(tmp_res);
    return;
}
void ecall_max_pool_2d(double *f_x_r,int len, int len_out, double *f_r,int shape[],int kernel_size, int stride, double* input, double *delta, int *maxarg){
    //decrypt
    double *tmp_res = (double*)malloc(sizeof(double)*len);
    double *key = (double *)malloc(len*sizeof(double));
    int i, j;
    int N,C,H,W;
    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
    int H_out = 1 + (H - kernel_size) / stride;
    int W_out = 1 + (W - kernel_size) / stride;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=f_r[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        f_x_r[j]=f_x_r[j]-key[j];
    }

    max_pool_2d(f_x_r,shape,kernel_size,stride,input,maxarg);
    // store result to input
    double *key_ = (double *)malloc(len_out*sizeof(double));
    memset(key_,0,sizeof(double)*len_out);
    // make r
    for(i=0;i<8;i++){
        for(j=0;j<len_out;j++){
            key_[j]+=delta[indexs[i]*len_out+j];

        }
    }
    // key is r(noises) now
    for(j=0;j<len_out;j++){
        //x + r 
        input[j]=input[j]+key_[j];
    }
    free(tmp_res);
    free(key);
    free(key_);
    return;
}
void ecall_softmax(double *f_x_r, double *f_r, double *input, int len, int *shape, int dim, double *delta){
    double *tmp_res = (double*)malloc(sizeof(double)*len);
    double *key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=f_r[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        tmp_res[j]=f_x_r[j]-key[j];
    }
    // for(j=0;j<len;j++){
    //     //x + r 
    //     printf("%lf, ", tmp_res[j]);
    // }
    softmax_e(tmp_res, shape, dim);

    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];

        }
    }
    // key is r(noises) now
    for(j=0;j<len;j++){
        //x + r 
        // printf("%lf, ", tmp_res[j]);
        input[j]=tmp_res[j]+key[j];
    }
    free(tmp_res);
    free(key);
    return;
}
void ecall_MSE(double *f_x_r_a, double *f_x_r_b, double *f_r, int len, int mode, double *result){
    double *tmp_res = (double*)malloc(sizeof(double)*len);
    double *key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=f_r[indexs[i]*len+j];
        }
    }
    //key is f(r) now
    for(j=0;j<len;j++){
        f_x_r_a[j]=f_x_r_a[j]-key[j];
        f_x_r_b[j]=f_x_r_b[j]-key[j];
    }
    result[0] = MSE(f_x_r_a, f_x_r_b, len, mode);
    free(key);
    return;
}

void ecall_dropout(double *f_x_r, double *f_r, int len, double *input, double p, int *shape, double *delta){
    double *tmp_res = (double*)malloc(sizeof(double)*len);
    double *key = (double *)malloc(len*sizeof(double));
    int i, j;
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=f_r[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        tmp_res[j]=f_x_r[j]-key[j];
    }

    dropout(tmp_res, p, shape);
    memset(key,0,sizeof(double)*len);
    for(i=0;i<8;i++){
        for(j=0;j<len;j++){
            key[j]+=delta[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        input[j] = tmp_res[j] + key[j];
    }
    free(key);
    free(tmp_res);
    return;
}
void ecall_entropy(double *y_pred, double *y_true, int len, int classes, double *delta1, double *result, int shape[]){
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
        // printf("%lf, ", y_pred[j]);
    }
    // for(j=0;j<classes;j++){
    //     printf("%lf, ", y_pred[j]);
    // }
    // printf("\n");

    double res = cross_entropy(y_true, y_pred, len, classes, shape);
    result[0] = res;
    // printf("%lf", res);
    free(key);
    return;
}
}