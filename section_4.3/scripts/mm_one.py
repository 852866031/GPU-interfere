import torch, sys
N = int(sys.argv[1])
a = torch.randn(N, N, device="cuda")
b = torch.randn(N, N, device="cuda")
for _ in range(3):
    torch.mm(a, b)          # warm up / trigger cuBLAS load
torch.cuda.synchronize()
c = torch.mm(a, b)          # the launch to profile
torch.cuda.synchronize()
