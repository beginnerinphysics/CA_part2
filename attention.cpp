#include <math.h>
#include <iostream>
using namespace std;
void attention_forward_cpu(float* out, const float* inp, int T, int C, int NH) {
//看main.cpp可以了解到, 此func對應的是第19頁的attention一個小塊, 而非MHSA block, 而前一個blockmatmul已經使channel變成3倍的長度
//multihead會降低的是每個head看的維度, 切塊後每個head看的token數量是不變的
    const int H = C / NH;
    const float SQRTH = sqrt(H);
    const int OC = C * 3;
    const int bias_v = 2 * C;
    for(int head = 0;head < NH;head = head + 1){
        const float *bias = inp + head * H;
        const float *inp_b_v = bias + bias_v;
        float *out_qkv_b = out + head * H;
        for(int i = 0; i < T; i = i + 1){
            float *out_qkv = out_qkv_b + i * C;
            const float *inp_q = bias + i * OC;
    
            float rsoftmax = 0;
			float r[T] = {0};
			const float *inp_b_k = bias + C;
            for(int j = 0; j < T; j = j + 1){
                const float *inp_k = inp_b_k + j * OC;
                for(int k = 0;k < H;k = k + 1){
                    r[j] = r[j] + inp_q[k] * inp_k[k];//如果用3就無法scale up, 希望不是大問題, 另外要注意轉置的問題, 畫圖會幫助理解
                }
                r[j] = exp(r[j] / SQRTH);
                rsoftmax = rsoftmax + r[j];
            }//Q K multiply done already
            for(int j = 0; j < T; j = j + 1){
                r[j] = r[j] / rsoftmax;
            }
            
            for(int j = 0; j < H; j = j + 1){
				float o = 0;
				const float *inp_v = inp_b_v + j;
				for(int h = 0;h < T;h = h + 1){
					o = o + r[h] * inp_v[h * OC];//記得output只有T*C的大小
				}
				out_qkv[j] = o;
			}
            
        }
        
    }
}


