#include <cmath>
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
void gelu_forward_cpu(float* out, const float* inp, int N) {
    
    for (int i = 0; i < N; ++i) {
        x = inp[i];//not sure if just use inp[i]will make it slower
        
        //first version
        out[i] = 0.5f * x * (tanhf(GELU_SCALING_FACTOR * (x *(1 + 0.044715 * x * x)));//tanhf, thef means float not double
        //second version
        //out[i] = x * sigmoid(1.702 * x)
    }
}