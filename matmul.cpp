#include "matmul.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>

using namespace std;//如果最後沒用到可以刪掉

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int T, int C, int OC) {
    for (int qkv = 0; qkv < OC; qkv += C) {

        const float* qkv_weight = weight + qkv*C ;//如果QKV weight是一個矩陣,串連,根據p20頁的說法, 且考慮轉置的效果, 所以QKV造成的影響要用轉置後的W去想


        float* qkv_out = out + qkv; // 如果out是一個矩陣,串連
        

        for (int i = 0; i < T; i++) {
            for (int j = 0; j < min(C, OC - qkv); j++) {
                float r = 0;
                for (int k = 0; k < C; k++) {
                    r += inp[i * C + k] * qkv_weight[k + j*C];//此種方式至的output是對的, 暗指已經轉置過了

                }
            //    qkv_out[i * C + j] = r + bias[qkv + j]; // 如果out是三個矩陣, 完全獨立
                qkv_out[i * OC + j] = r + bias[qkv + j]; // 如果out是一個矩陣, 串連
            }
        }
    }
}

//已知換成1Darray QKV是交錯的
//已知bias的內部參數應該放i才是rowwise,可是放j才符合3*C的malloc, 實驗顯示要使用j, 也許跟轉置有關
//另外有一個大重點:OC不一定只有qkv三個值, 也就是OC不一定是3倍的C
//注意矩陣的乘法a*b時, b的傳入是已經轉置了
