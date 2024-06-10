void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
//normalize之前的值和attention之後的值相加, N是 T*C
//inp1是T*C, inp2應該也是T*C, 也許是因為是pointer所以可以用一維的方式實現, 原因待確認
//通常來說, residual完之後會在次normalize, attention完之後也會, 但最先embedding完的x不會
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}
