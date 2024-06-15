#include <math.h>
#include <iostream>
void attention_forward_cpu(float* out, const float* inp, int T, int C, int NH) {
using namespace std;
    for(int head = 0;head < NH;head = head + 1){
        int H = C / NH;
        float *input =  inp + head * H;
        float *out_qkv = out + head*H;
        float r[T][T] = {0};
        float MAX[T] = {0};
        for(int i = 0; i < T; i = i + 1){
            float max_val = 0;
            for(int j = 0; j < T; j = j + 1){
                for(int k = 0;k < H;k = k + 1){
                    r[i][j] = r[i][j] + input[i * C * 3 + k] * input[C + j * C * 3 + k];
                }
                r[i][j] = r[i][j] / sqrt(H);
                max_val = max(max_val, r[i][j]);
                
            }
            for(int j = 0; j < T; j = j + 1){
                r[i][j] = exp(r[i][j] - max_val);
                MAX[i] = MAX[i] + r[i][j];
            }
            for(int j = 0; j < T; j = j + 1){
                r[i][j] = r[i][j] / MAX[i];
            }
            
            for(int j = 0; j < H; j = j + 1){
                float o = 0;
                for(int h = 0;h < T;h = h + 1){
                    o = o + r[i][h] * input[C * 2 + h * C * 3 + j];
                }
                out_qkv[i * C + j] = o;
            }
        }
        
    }
}

