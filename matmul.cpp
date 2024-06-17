#include "matmul.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>
#define block_t 4
#define block_oc 4
#define block_c 4
using namespace std;//如果最後沒用到可以刪掉

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int T, int C, int OC) {
/*
    for (int qkv = 0; qkv < OC; qkv += C) {

        const float* qkv_weight = weight + qkv*C ;


        float* qkv_out = out + qkv; // 如果out是一個矩陣,串連
        

        for (int i = 0; i < T; i++) {
            for (int j = 0; j < min(C, OC - qkv); j++) {
                float r = 0;
                for (int k = 0; k < C; k++) {
                    r += inp[i * C + k] * qkv_weight[k + j*C];//此種方式至的output是對的, 暗指已經轉置過了

                }
                qkv_out[i * OC + j] = r + bias[qkv + j]; // 如果out是一個矩陣, 串連
            }
        }
    }


*/

    for(int qkv =0; qkv < OC; qkv+=C){
        const float* qkv_weight= weight + qkv * C;//似乎Weight的時候qkv不是串連, 而是q先表達完再換k, k so on to v
        float* qkv_out= out + qkv;
       /// for 
        for(int jj = 0; jj < C;jj += block_oc){
            for(int kk = 0;kk < C;kk +=block_c){
                for(int i = 0; i < T; i=i+1){
                    for(int j=jj;j<min(jj+block_oc,C);j=j+1){//嘗試交換此compute to variable, 看會不會加速
                        float r = 0;
                        if(kk==0){qkv_out[i * OC + j] = bias[qkv+j];} 
                                           
                        for(int k = kk;k<min(kk+block_c,C);k=k+1){
                            r = r + inp[i * C + k] * qkv_weight[k + j*C];                
                        }
//                        cerr <<"r is"<<r<<endl;
                        qkv_out[i * OC + j] = qkv_out[i * OC + j] + r ;          
              //          cerr<<"now qkv out is"<< qkv_out[i * OC + j];  
                    
                    }
                }
            }
        }
    }



}



/*
const float* ptr_end = out + T * OC;//指向inp的最後面
const int bias_t = block_t * OC ;
const int bias_c = 
    for(const float* ptr_t = inp; ptr_t < ptr_end; ptr_t = ptr_t + bias_t){
        for(const float* ptr_c = ptr_t; ptr_c < OC; ptr_c = ptr_c + block_c){
            for(int kk =0; kk < C; kk = kk + block_c){
            
            
            
            
            
            
            
            }
        }
    }

*/






