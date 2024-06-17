#include "matmul.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>
#define block_t 4
#define block_oc 4
#define block_c 4
using namespace std;

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int T, int C, int OC) {
    for(int qkv =0; qkv < OC; qkv+=C){
        const float* qkv_weight= weight + qkv * C;//似乎Weight的時候qkv不是串連, 而是q先表達完再換k, k so on to v
        float* qkv_out= out + qkv;
        for(int jj = 0; jj < C;jj += block_oc){
  
            for(int kk = 0;kk < C;kk +=block_c){
                for(int i = 0; i < T; i=i+1){
                    for(int j=jj;j<min(jj+block_oc,OC);j=j+1){//嘗試交換此compute to variable, 看會不會加速
                        float r = 0;
                        if(kk==0){qkv_out[i * OC + j] = bias[qkv+j];} 
                                           
                        for(int k = kk;k<min(kk+block_c,C);k=k+1){
                            r = r + inp[i * C + k] * qkv_weight[k + j*C];                
                        }

//if((qkv_out+i*OC+j>= out+T*OC)&&(j==0||j>760)){cerr <<"i/j/ is"<<i<<"and"<<j<<endl;}
                        qkv_out[i * OC + j] = qkv_out[i * OC + j] + r ;          
              //          cerr<<"now qkv out is"<< qkv_out[i * OC + j];  
                    
                    }
                }
            }
        }
//    cerr<< "qkv is "<<qkv<<endl;
    }
}



