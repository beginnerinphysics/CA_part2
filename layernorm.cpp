#include "math.h"
#include "layernorm.h"
using namespace std;
float EPSILON = 1e-5;
void layernorm_forward_cpu(float* out, const float* inp, const float* weight, const float* bias,
                       int T, int C) {
    for (int tk = 0; tk < T; tk++) {
        float mean[T] = {0};
        float sig = 0;
        
        for (int ch = 0; ch < C; ch++) {
            mean[T] += inp[tk * C + ch];
        }
        mean[T] /= C;
        
        
        for (int ch = 0; ch < C; ch++) {
            float val = inp[tk * C + ch] - mean[T];
            sig += val * val;
        }

        sig = sqrt(sig / C + EPSILON);
        for (int ch = 0; ch < C; ch++) {
          
            out[tk * C + ch] = (inp[tk * C + ch] - mean[T]) / sig * weight[ch] + bias[ch];
        }
    }
}
