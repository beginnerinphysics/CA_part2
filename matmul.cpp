#include "matmul.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>
#define block_t 4
#define block_oc 4
#define block_c 4//初版完成之後有時間需要回來研究修改此block, 已達到更好的locality
using namespace std;
void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int T, int C, int OC) {
    int iter=0 ;              
    for(int ocend=0; ocend!=OC;ocend=ocend+C){                  
        for(int ii = 0; ii < T; ii = ii + block_t){
            for(int jj = 0; jj < C; jj = jj + block_c){
                for(int kk =0; kk < C; kk = kk+block_oc){
                    for(int i = ii; i < T&& i<block_t+ii; i = i +1){
                        for(int j = jj ; j < OC&&j<block_c+jj; j = j + 1){
                            float r = 0;
                            for(int k = kk; k <C && k<kk+block_oc; k = k + 1){
                                r = r + inp[i*C+k] * weight[k*C+j+ocend*T] ;//這樣寫的話應該就是QKV獨立了, 但還需驗證
                            }
                            out[i*C+j+ocend*T] = r + bias[i+iter*T];
                        }
                    }
                }  
            }
        }
        iter = iter + 1;
    }
}
