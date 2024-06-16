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
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < min(C, OC - qkv); j++) {
                float r = 0;
                for (int k = 0; k < C; k++) {
                    r += inp[i * C + k] * weight[qkv*C + k + j*C];
                }
                out[i * OC + qkv + j] = r + bias[qkv + j]; 
            }
        }
    }
}
