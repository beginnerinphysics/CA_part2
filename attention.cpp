#include <math.h>
#include <iostream>
using namespace std;
void attention_forward_cpu(float* out, const float* inp, int T, int C, int NH) {


    const int H = C / NH;
    const float SQRTH = 1.0f/sqrt(H);
    const int OC = C * 3;
    const int bias_v = 2 * C;
    for(int head = 0;head < NH;head = head + 1){
        const float *bias = inp + head * H;
        const float *inp_b_v = bias + bias_v;
        float *out_qkv_b = out + head * H;
        for(int i = 0; i < T; i = i + 1){
            float *out_qkv = out_qkv_b + i * C;
            const float *inp_q = bias + i * OC;
    
            double rsoftmax = 0.0;
			double r[T] = {0.0};
			double maxvalue = -INFINITY;
			const float *inp_b_k = bias + C;
            for(int j = 0; j < T; j = j + 1){
                const float *inp_k = inp_b_k + j * OC;
                for(int k = 0;k < H;k = k + 1){
                    r[j] = r[j] + inp_q[k] * inp_k[k]*SQRTH;
                }
                maxvalue = fmaxf(maxvalue,r[j]);
                
            }
            for(int j = 0; j < T; j = j + 1){
            
//                r[j] = exp(r[j] * SQRTH );
            
                rsoftmax = rsoftmax + exp(r[j]-maxvalue);
//                cerr <<"now j and rsoftmax is "<<j <<"and"<< rsoftmax<<endl;
            }
            for(int j = 0; j < T; j = j + 1){
             
                r[j] = exp(r[j]-maxvalue) / rsoftmax;
/*                              
                                if (isnan(r[j]) ) {
                cerr << "Error: rsoftmax prob" << rsoftmax <<"and r[j] is"<< r[j] <<"and j is" << j << endl;
                return;}
*/
            }
            
            for(int j = 0; j < H; j = j + 1){
				float o = 0.0f;
				const float *inp_v = inp_b_v + j;
				for(int h = 0;h < T;h = h + 1){
					o = o + r[h] * inp_v[h * OC];//記得output只有T*C的大小
				}
				out_qkv[j] = o;
			}
            
        }
        
    }
}
