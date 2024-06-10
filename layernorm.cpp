#include "math.h"
#include "layernorm.h"
using namespace std;
const float EPSILON = 1e-5;
void layernorm_forward_cpu(float* out, const float* inp, const float* weight, const float* bias,
                       int T, int C) {//公式內的epsilon是為了避免除以零用的, 通常值都很小
    for (int token = 0; token < T; token++) {// separate token
        float mu = 0,sigma = 0;
        
        for (int channel = 0; channel < C; channel++) {// cal mu
            mu += inp[token * C + channel];
        }
        mu /= C;
        
        
        for (int channel = 0; channel < C; channel++) {// cal sigma
            float val = inp[t * C + channel] - mu;
            sigma += val * val;
        }
//        sigma = sqrt(sigma / C); //cal this when we need a number which is not squre sigma
        sigma = sqrt(sigma / C + EPSILON)
        for (int channel = 0; channel < C; channel++) {///needfix: gamma and beta not define
            out[token * C + channel] = (inp[token * C + channel] - mu) / sigma * weight[c] + bias[c];
        }
    }
}
