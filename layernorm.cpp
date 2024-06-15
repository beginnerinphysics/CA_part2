#include "math.h"
#include "layernorm.h"
using namespace std;
float EPSILON = 1e-5;
void layernorm_forward_cpu(float* out, const float* inp, const float* weight, const float* bias,
                       int T, int C) {
    float mu[T] = {0};
    for (int token = 0; token < T; token++) {
        float mean[T] = {0};
        float sig = 0;
        
        for (int ch = 0; ch < C; ch++) {
            mean[T] += inp[token * C + ch];
        }
        mean[T] /= C;
        
        
        for (int ch = 0; ch < C; ch++) {
            float val = inp[token * C + ch] - mean[T];
            sig += val * val;
        }

        sig = sqrt(sig / C + EPSILON);
        for (int ch = 0; ch < C; ch++) {
          
            out[token * C + ch] = (inp[token * C + ch] - mean[T]) / sig * weight[ch] + bias[ch];
        }
    }
}
