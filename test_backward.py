import torch
from torch.utils.cpp_extension import load



wkv5_cuda = load(name="wkv5", sources=["cuda/wkv_torch.cpp", f"cuda/wkv.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_=4"])

torch.manual_seed(42)

B, T, C, H = 1, 8, 4, 1
r = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
k = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
v = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
w = torch.randn(B, C, device='cuda', dtype=torch.float32).contiguous()
we = torch.randn(B, C, device='cuda', dtype=torch.float32).contiguous()

u = torch.randn(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
gy = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gr = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gk = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gv = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
gw = torch.zeros(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
gu = torch.zeros(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
last_state = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
new_state = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
last_sa = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
last_sb = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
last_sc = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
last_sd = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
new_sa = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
new_sb = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
new_sc = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
new_sd = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()



r2 = r[:, 4:, :].clone()
k2 = k[:, 4:, :].clone()
v2 = v[:, 4:, :].clone()
sa1 = new_sa.clone()
sb1 = new_sb.clone()
sc1 = new_sc.clone()
sd1 = new_sd.clone()
gy2 = gy[:, 4:, :].clone()
gr2 = gr.clone()
gk2 = gk.clone()
gv2 = gv.clone()
gw2 = gw.clone()
gu2 = gw.clone()



r1 = r[:, :-4, :].clone()
k1 = k[:, :-4, :].clone()
v1 = v[:, :-4, :].clone()

u1 = u.clone()
y1 = torch.zeros(B, 4, C, device='cuda', dtype=torch.bfloat16).contiguous()
ls = last_state.clone()
sa = new_sa.clone()
sb = new_sb.clone()
sc = new_sc.clone()
sd = new_sd.clone()
gy1 = gy[:, :-4, :].clone()
gr1 = gr[:, :-4, :].clone()
gk1 = gk[:, :-4, :].clone()
gv1 = gv[:, :-4, :].clone()
gw1 = gw.clone()
gu1 = gw.clone()
# 调用backward函数
#print(v)
# wkv5_cuda.forward(B, T, C, H, r, k, v, w, u, y, ls, ns)

wkv5_cuda.backward(B, T, C, H, r, k, v, w, we, u, gy, gr, gk, gv, gw, gu, last_state, 
                   last_sa, last_sb, last_sc, last_sd,
                   new_sa, new_sb, new_sc, new_sd)
# 打印梯度结果
# print("2222:\n", gr)
#print("1111:\n", gr)
print(gr)
print(gu)
# print("new_sa:\n", new_sa)
# print("new_sb:\n", new_sb)
# print("new_sc:\n", new_sc)
# print("new_sd:\n", new_sd)
wkv5_cuda.backward(B, 4, C, H, r1, k1, v1, w, we, u, gy1, gr1, gk1, gv1, gw1, gu1, last_state, 
                   last_sa, last_sb, last_sc, last_sd,
                   sa, sb, sc, sd)
#print(last_state)
print(gu1)
wkv5_cuda.forward(B, 4, C, H, r1, k1, v1, w, u, y1, last_state, new_state)
new1 = new_state.clone()
new1 = new1.transpose(-1, -2)
# new_state = new_state.t(dim=-1)
# print("new_sa:\n", sa)
# print("new_sb:\n", sb)
# print("new_sc:\n", sc)
# print("new_sd:\n", sd)


wkv5_cuda.backward(B, 4, C, H, r2, k2, v2, w, we, u, gy2, gr1, gk1, gv1, gw1, gu1, new_state, 
                   sa, sb, sc, sd,
                   sa1, sb1, sc1, sd1)
#print(gr1,gk1)
print(gr1)
print(gu1)
# print("new_sa:\n", sa1)
# print("new_sb:\n", sb1)
# print("new_sc:\n", sc1)
# print("new_sd:\n", sd1)

