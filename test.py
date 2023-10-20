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
C = 4
H = 5

# 创建测试输入张量
r = torch.randn(B, T, C, H, dtype=torch.float32)
k = torch.randn(B, T, C, H, dtype=torch.float32)
v = torch.randn(B, T, C, H, dtype=torch.float32)
w = torch.randn(H, dtype=torch.float32)
u = torch.randn(B, T, C, H, dtype=torch.float32)

# 调用前向传播函数
y = wkv5_cuda.forward(B, T, C, H, r, k, v, w, u)

# 确保输出张量的形状和类型正确



