import torch
from torch.utils.cpp_extension import load

# cuda_module = load(name="v5_wkv",
#                    extra_include_paths=["include"],
#                    sources=["cuda/wkv_torch.cpp", "cuda/wkv.cu"],
#                    verbose=True)

wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op_state.cpp", f"cuda/wkv5_cuda_state.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_=64"])

B = 2
T = 3
C = 128
H = 2

# 创建测试输入张量
r = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
k = torch.randn(B, T, C,device='cuda', dtype=torch.bfloat16).contiguous()
v = torch.randn(B, T, C,device='cuda', dtype=torch.bfloat16).contiguous()
w = torch.randn(B, C,device='cuda', dtype=torch.float32).contiguous()
u = torch.randn(B, C,device='cuda', dtype=torch.bfloat16).contiguous()
y = torch.empty((B, T, C),device='cuda', dtype=torch.bfloat16).contiguous() # .uniform_(-1, 1)
print(y)
# 调用前向传播函数
y = wkv5_cuda.forward(B, T, C, H, r, k, v, w, u, y)

# 确保输出张量的形状和类型正确



