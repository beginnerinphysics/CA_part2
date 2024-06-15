#include <cmath>
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; ++i) {
        float x = inp[i];
        out[i] = 0.5 * x * (1+tanh(GELU_SCALING_FACTOR * (x + 0.044715 * pow(x,3))));
    }
}
