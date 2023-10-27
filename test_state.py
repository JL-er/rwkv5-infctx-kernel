import torch
from torch.utils.cpp_extension import load



wkv_cuda = load(name="wkv5", sources=["cuda/wkv_torch.cpp", f"cuda/wkv.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_=4"])

torch.manual_seed(42)

class WKV_part(torch.autograd.Function):
    # def init_state(B, C):
    #     state = torch.zeros((B, C, 3), device='cuda')
    #     state[:,:,2] -= 1e38
    #     return state.cuda()

    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u, last_state):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        ctx.H = H
        #w = -torch.exp(w.contiguous())
        r = r.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ew = (-torch.exp(w.float())).contiguous()
        eew = (torch.exp(ew)).contiguous()
        last_state = last_state.contiguous()
        y = torch.empty(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
        ctx.save_for_backward(r, k, v, eew, ew, u, last_state)
        new_state = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
        wkv_cuda.forward(B, T, C, H, r, k, v, eew, u, y, last_state, new_state)
        return y, new_state

    @staticmethod
    def backward(ctx, gy, gnew_state):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        H = ctx.H
        r, k, v, eew, ew, u, last_state = ctx.saved_tensors
        gy = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
        gr = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
        gk = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
        gv = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
        gw = torch.zeros(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
        gu = torch.zeros(B, C, device='cuda', dtype=torch.bfloat16).contiguous()
        glast_state = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()

        wkv_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu, last_state)
        # gw = torch.sum(gw, dim=0)
        # gu = torch.sum(gu, dim=0)
        return (None, None, None, None, gr, gk, gv, gw, gu, None)
    

torch.manual_seed(41)
B = 1
T = 8
C = 4
H = 1

# 创建测试输入张量
r = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
k = torch.randn(B, T, C,device='cuda', dtype=torch.bfloat16).contiguous()
v = torch.randn(B, T, C,device='cuda', dtype=torch.bfloat16).contiguous()
w = torch.randn(B, C,device='cuda', dtype=torch.float32).contiguous()
u = torch.randn(B, C,device='cuda', dtype=torch.bfloat16).contiguous()
y = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
ls = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()
#ns = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()

y,state = WKV_part.apply(B, T, C, H, r, k, v, w, u, ls)
print(y)

r1 = r[:, :-4, :].clone()
k1 = k[:, :-4, :].clone()
v1 = v[:, :-4, :].clone()
w1 = w.clone()
u1 = u.clone()
y1 = torch.zeros(B, 4, C, device='cuda', dtype=torch.bfloat16).contiguous()
ls1 = ls.clone()
y1,state1 = WKV_part.apply(B, 4, C, H, r1, k1, v1, w, u, ls1)


r2 = r[:, 4:, :].clone()
k2 = k[:, 4:, :].clone()
v2 = v[:, 4:, :].clone()
w2 = w.clone()
u2 = u.clone()
y2 = torch.zeros(B, 4, C, device='cuda', dtype=torch.bfloat16).contiguous()

y2,state2 = WKV_part.apply(B, 4, C, H, r2, k2, v2, w, u, state1)

print(y2)
