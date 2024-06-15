#include "math.h"
#include "layernorm.h"
using namespace std;
const float EPSILON = 1e-5;
void layernorm_forward_cpu(float* out, const float* inp, const float* weight, const float* bias,
                       int T, int C) {
    float mu[T] = {0}
    for (int token = 0; token < T; token++) {
        float mu[T] = 0;
        float sig = 0;
        
        for (int channel = 0; channel < C; channel++) {
            mu[T] += inp[token * C + channel];
        }
        mu[T] /= C;
        
        
        for (int channel = 0; channel < C; channel++) {
            float val = inp[token * C + channel] - mu[T];
            sig += val * val;
        }

        sig = sqrt(sig / C + EPSILON);
        for (int channel = 0; channel < C; channel++) {
          
            out[token * C + channel] = (inp[token * C + channel] - mu[T]) / sig * weight[channel] + bias[channel];
        }
    }
}
