#include <cstring>
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>
#include <vector>
#include <math.h>
using namespace std;
extern "C"{
double gaussrand_NORMAL() {
	static double V1, V2, S;
	static int phase = 0;
	double X;


	if (phase == 0) {
		do {
			double U1 = (double) rand() / RAND_MAX;
			double U2 = (double) rand() / RAND_MAX;


			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);


		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);


	phase = 1 - phase;


	return X;
}


double gaussrand(double mean, double stdc) {
	return mean + gaussrand_NORMAL() * stdc;
}
double alpha = 1;
int indexs[16+5]={6, 2, 2, 3, 0, 6, 6, 7, 1, 2, 2, 4, 5, 0, 8, 4};
int insert = 9;
int Ne = 10;
int Nt = 16;
// 缩放比，在进行浮点数加减（对应加密方式中的加解密）前先乘以缩放比，以减少精度损失，运算后再除以该缩放比
double cast = 5;
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
void d_batchnorm(double *x_h, double *dy, double *running_mean, double *running_var, double *weight, double eps, int len, int shape[], double *ret_grad_w, double *ret_grad_b){
    // x_h, dy: len * 10
    // x_h: the x after normalization and encrypted
    // weight,bias : C
    // dy: len * 10
    // decrypt grad_out/dy
    int i, j;
    double *temp = (double*)malloc(sizeof(double)*len);
    double *grad_out = (double*)malloc(sizeof(double)*len);
    double *x_hat = (double*)malloc(sizeof(double)*len);
    // decrypt dy
    memset(temp, 0, sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            temp[j] += dy[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        grad_out[j] = temp[j] + dy[insert*len+j];
    }
    //decrypt x_hat
    memset(temp, 0, sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            temp[j] += x_h[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        x_hat[j] = temp[j] + x_h[insert*len+j];
    }
    //compute gradient
    int N,C,H,W;
    int k1,k2,k3;
    N=shape[0],C=shape[1],H=shape[2],W=shape[3];
    k3=C*H*W;k2=H*W;k1=W;
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
    // 这里计算时到底应该用running_var 还是 var有待考究
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
        grad_x[i] = inp_expand[i] * (part1[i] + part2[2] + part3[i]);
    }
    // encrypt gradient
    memset(x_h, 0, sizeof(double)*len*10);
    double mean=0, std, mean_sqr=0;
    for(j=0;j<len;j++){
        mean += grad_x[j] / len;
        mean_sqr += grad_x[j] * grad_x[j] / len;
    }
    std = sqrt(mean_sqr - mean * mean );
    mean = mean / 16;
    for(i=0;i<len*9;i++){
        x_h[i] = gaussrand(mean, std);
        x_h[i] = grad_x[i % len] - x_h[i];
    }//generate new x1~x9
    for(i=0;i<16;i++){
        for(j=len*9;j<len*10;j++){
            x_h[j] += x_h[indexs[i]*len+(j-len*9)];
        }
    }
    for(j=len*9;j<len*10;j++){
        x_h[j] = grad_x[j-len*9] - x_h[j];
    }//compute x10
    free(temp);
    free(x_hat);
    free(grad_out);
    free(grad_b);
    free(grad_w);
    free(grad_x);
    free(part1);
    free(part2);
    free(part3);
    free(coef_inp);
    free(inp_expand);
    free(grad_w_expand);
    return;
}
void d_softmax_easy(double *dy_hat, double *y_hat, double *result, int len, int shape[]){
    //dy:[N, classes] = d(loss)/dy
    //y:[N, classes] = softmax(x)
    //decrypt y first:
    //len = N * classes
    //result: Size of len
    int i, j;
    double *tmp = (double*)malloc(sizeof(double)*len);
    double *res = (double*)malloc(sizeof(double)*len);
    memset(result, 0, sizeof(double)*len*Ne);
    // decrypt dy
    double *dy = (double*)malloc(sizeof(double)*len);
    memset(tmp, 0, sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            tmp[j] += dy_hat[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        dy[j] = tmp[j] + dy_hat[insert*len+j];
    }
    // decrypt y
    double *y = (double*)malloc(sizeof(double)*len);
    memset(tmp, 0, sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            tmp[j] += y_hat[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        y[j] = tmp[j] + y_hat[insert*len+j];
    }
    // compute grad
    int global_idx = 0;
    int N=shape[0], C=shape[1];
    double *grad_tmp = (double*)malloc(sizeof(double) * C * C);
    for(int idx_n=0;idx_n<N;idx_n++){
        memset(grad_tmp, 0, sizeof(double)*C*C);
        for(int idx_c=0;idx_c<C;idx_c++){
            grad_tmp[idx_c*C+idx_c] = y[idx_n*C+idx_c]; // diag(y)
        }
        for(int i=0;i<C;i++){
            for(int j=0;j<C;j++){
                grad_tmp[i*C+j] -= y[idx_n*C+i] * y[idx_n*C+j]; // dy/dx=diag(y)-y'.*y
            }
        }
        double temp=0;
        for(int i=0;i<C;i++){
            for(int j=0;j<C;j++){
                temp += dy[idx_n*C+j] * grad_tmp[j*C+i]; // d(loss)/dx=d(loss)/dy .* dy/dx
            }
            res[global_idx++] = temp;
            temp=0;
        }
    }
    double mean=0, std, mean_sqr=0;
    for(j=0;j<len;j++){
        mean += res[j] / len;
        mean_sqr += res[j] * res[j] / len;
    }
    
    std = sqrt(mean_sqr - mean * mean );
    mean = mean / 16;
    for(i=0;i<len*9;i++){
        result[i] = gaussrand(mean, std);
        result[i] = res[i % len] - result[i];
    }//generate new x1~x9
    for(i=0;i<16;i++){
        for(j=len*9;j<len*10;j++){
            result[j] += result[indexs[i]*len+(j-len*9)];
        }
    }
    
    for(j=len*9;j<len*10;j++){
        result[j] = res[j-len*9] - result[j];
    }//compute x10
    free(tmp);
    free(res);
    free(dy);
    free(y);
    free(grad_tmp);
    return;
}
void d_relu(double *dy, double *y, int len, double *result){
    //result: size of len * 10
    //y: size of len*10
    //dy=d(loss)/dy: same shape of y: [N*10,C,H,W] (10*len)
    int i, j;
    double *tmp = (double*)malloc(sizeof(double)*len);
    double *res = (double*)malloc(sizeof(double)*len);
    // decrypt dy
    memset(tmp, 0, sizeof(double)*len);
    memset(result, 0, sizeof(double)*len*Ne);
    double *dy_hat = (double*)malloc(sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            tmp[j] += dy[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        dy_hat[j] = tmp[j] + dy[insert*len+j];
    }
    //decrypt y
    memset(tmp, 0, sizeof(double)*len);
    double *y_hat = (double*)malloc(sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            tmp[j] += y[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        y_hat[j] = tmp[j] + y[insert*len+j];
    }
    //compute gradient
    for(int i=0;i<len;i++){
        // printf("%lf, ", dy_hat[i]);
        if(y_hat[i]>0){
            res[i] = dy_hat[i];
        }
        else{
            res[i] = 0;
        }
    }

    double mean=0, std, mean_sqr=0;
    for(j=0;j<len;j++){
        mean += res[j] / len;
        mean_sqr += res[j] * res[j] / len;
    }
    
    std = sqrt(mean_sqr - mean * mean );
    mean = mean / 16;
    for(i=0;i<len*9;i++){
        result[i] = gaussrand(mean, std);
        result[i] = res[i % len] - result[i];
    }//generate new x1~x9
    for(i=0;i<16;i++){
        for(j=len*9;j<len*10;j++){
            result[j] += result[indexs[i]*len+(j-len*9)];
        }
    }
    
    for(j=len*9;j<len*10;j++){
        result[j] = res[j-len*9] - result[j];
    }//compute x10
    free(tmp);
    free(y_hat);
    free(dy_hat);
    return;
}

void d_sigmoid(double *y, int len, double *result){
    for(int i=0;i<len;i++){
        result[i] = y[i] * (1 - y[i]);
    }
    return;
}

void d_dropout(double *dy, double *y, int len, double p, int shape[], double *result){
    int i, j;
    double *tmp = (double*)malloc(sizeof(double)*len);
    double *res = (double*)malloc(sizeof(double)*len);
    memset(result, 0, sizeof(double)*len*Ne);
    // decrypt dy
    memset(tmp, 0, sizeof(double)*len);
    double *dy_hat = (double*)malloc(sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            tmp[j] += dy[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        dy_hat[j] = tmp[j] + dy[insert*len+j];
    }
    //decrypt y
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
                        res[idx_n*k3+idx_c*k2+idx_h*k1+idx_w]=0;
                    }
                }
            }
            else{
                for(int idx_h=0;idx_h<H;idx_h++){
                    for (int idx_w = 0; idx_w < W; idx_w++){
                        res[idx_n*k3+idx_c*k2+idx_h*k1+idx_w]=(double)1 / (1-p) * dy_hat[idx_n*k3+idx_c*k2+idx_h*k1+idx_w];
                    }
                }
            }
        }
    }
    double mean=0, std, mean_sqr=0;
    for(j=0;j<len;j++){
        mean += res[j] / len;
        mean_sqr += res[j] * res[j] / len;
    }
    
    std = sqrt(mean_sqr - mean * mean );
    mean = mean / Nt;
    for(i=0;i<len*9;i++){
        result[i] = gaussrand(mean, std);
        result[i] = res[i % len] - result[i];
    }//generate new x1~x9
    for(i=0;i<Nt;i++){
        for(j=len*(Ne-1);j<len*Ne;j++){
            result[j] += result[indexs[i]*len+(j-len*(Ne-1))];
        }
    }
    
    for(j=len*(Ne-1);j<len*Ne;j++){
        result[j] = res[j-len*(Ne-1)] - result[j];
    }//compute x10
    free(res);
    free(tmp);
    free(dy_hat);
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
            // printf("%lf, ", input[idx_n*k2 + idx_c]);
            // 测试，某一个样本的输出值如下，可以看出区别并不大：
            // 0.100011, 0.099988, 0.099995, 0.100004, 0.100004, 0.100004, 0.099988, 0.100005, 0.099996, 0.100005
        }
        // printf("\n");
    }
    return;
}
// void d_crossEntropy(double *y_true, double *y_pred, int shape[], int len, double *result, int classes, double *delta1){
//     double * key = (double *)malloc(len*sizeof(double));
//     int i, j;
//     memset(key,0,sizeof(double)*len);
//     for(i=0;i<8;i++){
//         for(j=0;j<len;j++){
//             key[j]+=delta1[indexs[i]*len+j];
//         }
//     }
//     //key is f(r) now
//     for(j=0;j<len;j++){
//         y_pred[j]=y_pred[j]-key[j];
//     }
//     double tmp=0;
//     softmax_e(y_pred, shape, -1);
//     int mean = len / classes; 
//     for(int i=0;i<len;i++){
//         result[i] = (y_pred[i] - y_true[i]) / mean; 
//     }
//     return;
// }
void d_crossentropy(double *y_pred, double *y_true, int len, int classes, double *result, int shape[]){
    //decrypt
    //N=batch_size
    //result: Ne*len
    //y_pred: N*Ne*classes
    //y_true: N*classes = len
    int i, j;
    int N = shape[0];
    double max = 21;
    double *tmp = (double*)malloc(sizeof(double)*len);
    memset(tmp, 0, sizeof(double)*len);
    double *y_hat = (double*)malloc(sizeof(double)*len);
    memset(y_hat, 0, sizeof(double)*len);
    double *res = (double*)malloc(sizeof(double)*len);
    memset(result, 0, sizeof(double)*len*Ne);
    for(j=0;j<len*Ne;j++){
        // casting
        y_pred[j] *= cast;
    }
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            tmp[j] += y_pred[indexs[i]*len+j];
        }
    }//x_ = x1 + x5 + x3 + x6 + x3......
    for(j=0;j<len;j++){
        y_hat[j] = tmp[j] + y_pred[insert*len+j];
        // anti_casting
        y_hat[j] /= cast;
        // printf("%.2lf, ", y_hat[j]);
    }// x = x_ + x10

    softmax_e(y_hat, shape, 1);

    // printf("N: %d\n", N);
    // printf("len: %d\n", len);
    // compute grad
    for(i=0;i<len;i++){
        res[i] = (y_hat[i] - y_true[i]) / N; 
        // res[i] = y_hat[i] - y_true[i]; 
        // max = res[i] > max ? res[i] : max;
        // printf("%.2lf, ", res[i]);
    }

    // encryption
    double mean=0, std, mean_sqr=0;
    for(j=0;j<len;j++){
        mean += res[j] / len;
        mean_sqr += res[j] * res[j] / len;
    }
    std = sqrt(mean_sqr - mean * mean );
    mean = mean / 16;
    // printf("mean: %lf\n", mean);
    // printf("mean_sqr: %lf\n", mean_sqr);
    // printf("std: %lf\n", std);
    for(i=0;i<len;i++){
        res[i] *= cast;
    }

    //generate new x1~x8
    for(i=0;i<len*9;i++){
        result[i] = gaussrand(mean, std) * cast;
        result[i] = res[i % len] - result[i];
    }

    for(i=0;i<16;i++){
        for(j=len*9;j<len*10;j++){
            result[j] += result[indexs[i]*len+(j-len*9)];
        }
    }
    for(i=0;i<len*9;i++){
        result[i] /= cast;
    }
    for(j=len*9;j<len*10;j++){
        result[j] = (res[j-len*9] - result[j]) / cast;
    }//compute x9
    free(tmp);
    free(res);
    free(y_hat);
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
    //dy, y: len_y * Ne
    //result:len_x * Ne
    //y=maxpool(x)
    double *res = (double*)malloc(sizeof(double)*len_x);
    double *tmp = (double*)malloc(sizeof(double)*len_y);
    int N,C,H,W;
    int i, j;
    //decrypt dy
    memset(tmp, 0, sizeof(double)*len_y);
    memset(res, 0, sizeof(double)*len_x);//gradient is of the shape with x
    memset(result, 0, sizeof(double)*len_x*Ne);
    double *dy_hat = (double*)malloc(sizeof(double)*len_y);
    for(i=0;i<Nt;i++){
        for(j=0;j<len_y;j++){
            tmp[j] += dy[indexs[i]*len_y+j];
        }
    }
    for(j=0;j<len_y;j++){
        dy_hat[j] = tmp[j] + dy[insert*len_y+j];

    }
    //compute grad
    for(int i=0;i<len_y;i++){
        res[argmax[i]] = dy_hat[i];
        // printf("%.2lf, ", res[argmax[i]]);
    }
    //encrypt res(gradient)
    double mean=0, std, mean_sqr=0;
    for(j=0;j<len_x;j++){
        mean += res[j] / len_x;
        mean_sqr += res[j] * res[j] / len_x;
    }
    
    std = sqrt(mean_sqr - mean * mean );
    mean = mean / Nt;
    for(i=0;i<len_x*(Ne-1);i++){
        result[i] = gaussrand(mean, std);
        result[i] = res[i % len_x] - result[i];
    }//generate new x1~x(Ne-1)
    for(i=0;i<Nt;i++){
        for(j=len_x*(Ne-1);j<len_x*Ne;j++){
            result[j] += result[indexs[i]*len_x+(j-len_x*(Ne-1))];
        }
    }
    
    for(j=len_x*(Ne-1);j<len_x*Ne;j++){
        result[j] = res[j-len_x*(Ne-1)] - result[j];
    }//compute x(Ne)
    free(res);
    free(dy_hat);
    free(tmp);
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

void decrypt_conv(double *dw, double *x, double *grad_out, int len_x, int len_out, int shape_w[], int H_in, int H_out, int Batch_size){
    /*
        gout = d(loss) / dy y = w.*x
        if y:[*,r,p], X[*,q,p]
        len=N*C*q*p length of x
        shape = shape of gout[N,C,r,p]
    */
    int i,j;
    int kernel_num = shape_w[0], depth=shape_w[1], kernel_size = shape_w[2], N = Batch_size;
    // decrypt grad_out / gout
    double *tmp = (double*)malloc(sizeof(double)*len_out);
    memset(tmp, 0, sizeof(double)*len_out);
    double *dy = (double*)malloc(sizeof(double)*len_out);
    for(i=0;i<16;i++){
        for(j=0;j<len_out;j++){
            tmp[j] += grad_out[indexs[i]*len_out+j];
        }
    }
    for(j=0;j<len_out;j++){
        dy[j] = tmp[j] + grad_out[insert*len_out+j];
    }
    free(tmp);
    //decrypt x
    double *tmp_x = (double*)malloc(sizeof(double)*len_x);
    memset(tmp_x, 0, sizeof(double)*len_x);
    double *x_real = (double*)malloc(sizeof(double)*len_x);
    for(i=0;i<16;i++){
        for(j=0;j<len_x;j++){
            tmp_x[j] += x[indexs[i]*len_x+j];
        }
    }
    for(j=0;j<len_x;j++){
        x_real[j] = tmp_x[j] + x[insert*len_x+j];
    }
    //compute gradient
    memset(dw, 0, sizeof(double)*kernel_num*depth*kernel_size*kernel_size);
    double grad_tmp = 0;
    int k3=kernel_num*H_out*H_out, k2=H_out*H_out, k1=H_out;
    int y3=depth*H_in*H_in, y2=H_in*H_in, y1=H_in;
    int x3=depth*kernel_size*kernel_size, x2=kernel_size*kernel_size, x1=kernel_size;
    for(int idx_n=0;idx_n<N;idx_n++){
        for(int idx_k=0;idx_k<kernel_num;idx_k++){
            for (int idx_d = 0; idx_d < depth; idx_d++){
                for (int idx_h = 0; idx_h < kernel_size; idx_h++){
                    for (int idx_w = 0; idx_w < kernel_size; idx_w++){
                        grad_tmp = 0;
                        for (i = 0; i < H_out; i++){
                            for (j = 0; j < H_out; j++){
                                grad_tmp += dy[idx_n*k3+idx_k*k2+i*k1+j] * x_real[idx_n*y3+idx_d*y2+(i+idx_h)*y1+(j+idx_w)];
                            }
                        }
                        dw[idx_k*x3+idx_d*x2+idx_h*x1+idx_w] += grad_tmp ;
                    }

                }
                
            }
            
        }
    }
    free(tmp_x);
    free(x_real);
    free(dy);
    return;
}

void decrypt_linear(double *x, double *dy, double *dw, int len, int shape[]){
    /*
        dy = d(loss) / dy y = x.*w
        len = N*d length of real x
        shape = shape of dy[N, c] c: num_of_classes
        x : [N * 10, d]
    */
    int i,j;
    int N = shape[0], C = shape[1];
    int D = len / N;
    int len_out, len_x;
    len_out = len;
    len_x = N * D;
    double *tmp = (double*)malloc(sizeof(double)*len);
    // decrypt dy
    memset(tmp, 0, sizeof(double)*len);
    double *dy_real = (double*)malloc(sizeof(double)*len);
    for(i=0;i<16;i++){
        for(j=0;j<len;j++){
            tmp[j] += dy[indexs[i]*len+j];
        }
    }
    for(j=0;j<len;j++){
        dy_real[j] = tmp[j] + dy[insert*len+j];
    }
    free(tmp);
    //decrypt x
    double *tmp_x = (double*)malloc(sizeof(double)*len_x);
    memset(tmp_x, 0, sizeof(double)*len_x);
    double *x_real = (double*)malloc(sizeof(double)*len_x);
    for(i=0;i<16;i++){
        for(j=0;j<len_x;j++){
            tmp_x[j] += x[indexs[i]*len_x+j];
        }
    }
    for(j=0;j<len_x;j++){
        x_real[j] = tmp_x[j] + x[insert*len_x+j];
    }
    //compute gradient
    // vector<vector<vector<vector<float>>>> noise(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W,0))));
    memset(dw,0,sizeof(double)*D*C);
    int k1,k2,k3;
    double temp;
    for(i=0;i<D;i++){
        for(j=0;j<C;j++){
            temp = 0;
            for(int idx_n=0;idx_n<N;idx_n++){
                temp += x_real[idx_n*D+i] * dy_real[idx_n*C+j];
            }
            dw[i*C+j] = temp;
        }
    }
    free(tmp_x);
    return;
}
}