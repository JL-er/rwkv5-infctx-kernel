#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y, float *last_state, float *new_state);
void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu, float *last_state, float *last_sa, float *last_sb, float *last_sc, float *last_sd, float *new_sa, float *new_sb, float *new_sc, float *new_sd);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y, torch::Tensor &last_state, torch::Tensor &new_state) {
    cuda_forward(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>(), last_state.data_ptr<float>(), new_state.data_ptr<float>());
}


void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &ww, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &last_state,
    torch::Tensor &last_sa, torch::Tensor &last_sb, torch::Tensor &last_sc, torch::Tensor &last_sd,
    torch::Tensor &new_sa, torch::Tensor &new_sb, torch::Tensor &new_sc, torch::Tensor &new_sd) {
    cuda_backward(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), ww.data_ptr<float>(), u.data_ptr<bf16>(), gy.data_ptr<bf16>(), gr.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), last_state.data_ptr<float>(), 
    last_sa.data_ptr<float>(), last_sb.data_ptr<float>(), last_sc.data_ptr<float>(), last_sd.data_ptr<float>(),
    new_sa.data_ptr<float>(), new_sb.data_ptr<float>(), new_sc.data_ptr<float>(), new_sd.data_ptr<float>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv5 forward");
    m.def("backward", &backward, "wkv5 backward");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}