#include "matmul.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>

using namespace std;

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int T, int C, int OC) {
    for (int qkv = 0; qkv < OC; qkv += C) {
        const float* qkv_weight = weight + qkv*C ;
  //      const float* qkv_bias = bias + qkv*C ;
        float* qkv_out = out + qkv; // 修正 qkv_out 的計算方式

        for (int i = 0; i < T; i++) {
            for (int j = 0; j < C; j++) {
                float r = 0;
                for (int k = 0; k < C; k++) {
                    r += inp[i * C + k] * qkv_weight[k + j*C];
                }
                qkv_out[i * OC + j] = r + bias[qkv + j]; // 修正 qkv_out 的存取索引
            }
        }
    }
}

//已知換成1Darray QKV是交錯的
//已知bias的內部參數應該放i才是rowwise,可是放j才符合3*C的malloc, 實驗顯示要使用j
//另外有一個大重點:OC不一定只有qkv三個值, 也就是OC不一定是3倍的C
//注意矩陣的乘法a*b時, b的傳入是已經轉置了
