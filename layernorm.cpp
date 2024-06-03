#include "math.h"
#include "layernorm.h"
using namespace std;
void layernorm_forward_cpu(float* out, const float* inp, const float* weight, const float* bias,
                       int T, int C) {
    for (int token = 0; token < T; token++) {// separate token
        float mu = 0,sqrsigma = 0;
        
        for (int channel = 0; channel < C; channel++) {// cal mu
            mu += inp[token * C + channel];
        }
        mu /= channel;
        
        
        for (int channel = 0; channel < C; channel++) {// cal sigma
            float val = inp[t * C + channel] - mu;
            sqrsigma += val * val;
        }
//        sigma = sqrt(sigma / C); //cal this when we need a number which is not squre sigma
        sqrsigma /= C;
        
        for (int channel = 0; channel < C; channel++) {///needfix: gamma and beta not define
            out[t * C + channel] = (inp[t * C + channel] - mu) / sigma * gamma[c] + beta[c];
        }
    }
}
