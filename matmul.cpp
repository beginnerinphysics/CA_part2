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
        float* qkv_out = out + qkv;
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < min(C, OC - qkv); j++) {
                float r = 0;
                for (int k = 0; k < C; k++) {
                    r += inp[i * C + k] * qkv_weight[k + j*C];
                }
                qkv_out[i * OC + j] = r + bias[qkv + j]; 
            }
        }
    }
}
