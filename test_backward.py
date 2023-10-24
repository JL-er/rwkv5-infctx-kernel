import torch
from torch.utils.cpp_extension import load



wkv5_cuda = load(name="wkv51", sources=["cuda/wkv_torch.cpp", f"cuda/wkv.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_=4"])

torch.manual_seed(42)

B, T, C, H = 1, 2, 4, 1
r = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
k = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
v = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
w = torch.randn(B, C, device='cuda', dtype=torch.float32).contiguous()
u = torch.randn(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
gy = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gr = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gk = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gv = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gw = torch.zeros(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
gu = torch.zeros(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
last_state = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
new_state = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()

# 调用backward函数
wkv5_cuda.backward(B, T, C, H, r, k, v, w, w, u, gy, gr, gk, gv, gw, gu, last_state, new_state)

# 打印梯度结果
print("gr:\n", gr)
print("gk:\n", gk)
print("gv:\n", gv)
print("gw:\n", gw)
print("gu:\n", gu)





