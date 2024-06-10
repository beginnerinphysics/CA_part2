#include "matmul.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>
#define block_t 4
#define block_oc 4//OC指的是QKV三個值, 所以應該還需要修改
#define block_c 4//初版完成之後有時間需要回來研究修改此block, 已達到更好的locality
using namespace std;
void matmul_forward_cpu(float* out, //這裡的matmul單純是把inp和W, b相乘相加線性投射而已, y=XW^T+b, X的矩陣是T*C, W的空間似乎是OC*C, b似乎是OC的空間, 同個token的轉換會使用同個值的b
                    const float* inp, const float* weight, const float* bias,
                    int T, int C, int OC) {
                    
for(int ocend=0; ocend!=OC;ocend=ocend+C){                    
    for(int ii = 0; ii < t; ii = ii + block_t){
        for(int jj = 0; jj < C; jj = jj + block_c){
            for(int kk =0; kk < C; kk = kk+block_oc){
                for(int i = ii; i < min(T,block_t+i); i = i +1){
                    for(int j = jj ; j < min(OC,block_c+j); j = j + 1){
                        float r = 0;
                        for(int k = kk; k < min(C,k+blcok_c); k = k + 1){
                            r = r + inp[i][k] * weight[k][j+ocend] + bias[i+ocend];//這樣寫的話應該就是QKV獨立了, 但還需驗證
                        }
                        out[i][j] = r;
                    }
                }
            }  
        }
    }
}
}
