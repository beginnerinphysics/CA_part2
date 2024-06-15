#include <cmath>
//float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
const float factor = GELU_SCALING_FACTOR;
void gelu_forward_cpu(float* out, const float* inp, int N) {
    
    for (int i = 0; i < N; ++i) {
        float x = inp[i];//not sure if just use inp[i]will make it slower
        
        //first version
		float x_squared = x * x;
		out[i] = 0.5f * x * (1.0f + tanhf(factor * (x *( 1 + 0.044715f * x_squared))));
        //second version, seems lead little miss
        //out[i] = x * sigmoid(1.702 * x);
    }
}
