#include <math.h>
#include <iostream>
void attention_forward_cpu(float* out, const float* inp, int T, int C, int NH) {
//看main.cpp可以了解到, 此func對應的是第19頁的attention一個小塊, 而非MHSA block, 而前一個blockmatmul已經使channel變成3倍的長度
//multihead會降低的是每個head看的維度, 切塊後每個head看的token數量是不變的
//注意!!T會比H大
using namespace std;
    int H = C / NH;
    for(int head = 0;head < NH;head = head + 1){
        const float *inp_qkv =  inp + head * H;
        float *out_qkv = out + head*H;
        float r[T][T] = {0};
        float rsoftmax[T] = {0};
        for(int i = 0; i < T; i = i + 1){
            float max_val = 0;
            for(int j = 0; j < T; j = j + 1){
                for(int k = 0;k < H;k = k + 1){
                    r[i][j] = r[i][j] + inp_qkv[i * C * 3 + k] * inp_qkv[C + j * C * 3 + k];//如果用3就無法scale up, 希望不是大問題, 另外要注意轉置的問題, 畫圖會幫助理解
                }
                r[i][j] = r[i][j] / sqrt(H);//H is 8
                max_val = fmax(max_val, r[i][j]);
                
            }
            for(int j = 0; j < T; j = j + 1){
                r[i][j] = exp(r[i][j] - max_val);
                rsoftmax[i] = rsoftmax[i] + r[i][j];
            }//Q and K have done already
            for(int j = 0; j < T; j = j + 1){
                r[i][j] = r[i][j] / rsoftmax[i];
            }
            
            for(int j = 0; j < H; j = j + 1){
                float o = 0;
                for(int h = 0;h < T;h = h + 1){
                    o = o + r[i][h] * inp_qkv[2 * C + j + h * C * 3];//記得output只有T*C的大小
                }
                out_qkv[i * C + j] = o;
            }
        }
        
    }
}
//全部的3都是因為QKV, 如果像是matmul一樣測資有OC=4C的話, 會無法用argument 去 scale up

