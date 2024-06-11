#include <cmath>
//float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

void gelu_forward_cpu(float* out, const float* inp, int N) {
    
    for (int i = 0; i < N; ++i) {
        float x = inp[i];//not sure if just use inp[i]will make it slower
        
        //first version
        out[i] = 0.5 * x * (1+tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x,3))));
        //second version, seems lead little miss
        //out[i] = x * sigmoid(1.702 * x);
    }
}
