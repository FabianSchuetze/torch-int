import torch
from torch_int._CUDA import (
        bmm_s8t_s8n_s8t,
        bmm_s8t_s8n_s32t,
        bmm_s8t_s8n_f32t,
        bmm_f32t_f32n_f32t)
from icecream import ic

# @torch.no_grad()
# def test_bmm_s8t_s8n_s8t():
    # # used by attn_prob x value
    # # B, M, K, N = 1, 512, 512, 32
    # B, M, N, K =  300, 196, 196, 64 #not divisible by 32! a sin...
    # # a = torch.randint(-128, 127, (B, M, N), dtype=torch.int8, device='cuda')
    # # breakpoint()
    # # a = torch.softmax(a, dim=2)
    # # b = torch.randint(-128, 127, (B, N, K), dtype=torch.int8, device='cuda')
    # a = torch.load('/tmp/attn.pt')
    # b = torch.load('/tmp/v.pt')
    # anew = torch.zeros((
    # # assert a.shape == a2.shape
    # # assert b.shape == b2.shape
    # breakpoint()
    # scale = 0.001
    # # c = bmm_s8t_s8n_s8t(b.transpose(-2, -1).contiguous(), a.transpose(-2, -1), scale).transpose(-2, -1)
    # c = bmm_f32t_f32n_f32t(b.float().transpose(-2, -1).contiguous(), a.float().transpose(-2, -1),scale).transpose(-2, -1)
    # c_gt = torch.bmm(a.float(), b.float().transpose(1,2)) * scale
    # c_gt = c_gt.clamp(-128, 127).round().to(torch.int8)
    # ic(torch.allclose(c_gt, c.cpu()))

@torch.no_grad()
def test_bmm_s8t_s8n_s8t():
    # used by attn_prob x value
    B, M, K, N = 300, 196, 64, 196
    a1 = torch.randint(-128, 127, (B, M, N), dtype=torch.int8)
    b1 = torch.randint(-128, 127, (B, N, K), dtype=torch.int8)
    anew = torch.zeros((B, K, 224), dtype=torch.int8)
    bnew = torch.zeros((B, 224, 224), dtype=torch.int8)
    anew[:, :, :196] = b1.transpose(-2, -1).contiguous()
    bnew[:, :196, :196] = a1
    scale = 0.001
    c1 = bmm_s8t_s8n_s8t(anew.cuda(), bnew.transpose(-2, -1).cuda(), scale).transpose(-1, -2)[:, :196, :]
    # c2 = bmm_s8t_s8n_s8t(anew.cuda(), bnew.cuda(), scale)
    c_gt = torch.bmm(a1.float(), b1.float()) * scale
    c_gt = c_gt.clamp(-128, 127).round().to(torch.int8)
    ic(torch.allclose(c_gt, c1.cpu()))


@torch.no_grad()
def test_bmm_s8t_s8n_f32t():
    breakpoint()
    # used by attn_prob x value
    # B, M, K, N = 1, 512, 512, 32
    B, M, K, N = 400, 196, 64, 196
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8).cuda()
    b = torch.randint(-128, 127, (B, N, K), dtype=torch.int8).cuda()
    bnew = torch.zeros((B, 224, 64), device='cuda', dtype=torch.int8)
    bnew[:, :196, :] = b
    scale = 0.001
    c = bmm_s8t_s8n_f32t(a, bnew, scale)
    # ic(c)
    c_gt = torch.bmm(a.float(), b.transpose(-2, -1).float()) * scale
    # ic(c_gt)
    ic(torch.mean((c_gt - c) ** 2))


@torch.no_grad()
def test_bmm_s8t_s8n_s8t_2():
    # used by attn_prob x value
    a = torch.tensor([[[7,  8,  6, 16,  3,  7, 10,  4, 11,  5,  4, 11, 11,  7,  6,  9],
                       [7,  7,  9,  8,  9,  9,  6,  6, 6,
                           9, 10, 10,  8,  9,  9,  6],
                       [7,  9,  8,  9,  7,  7,  6,  5, 7,
                           7,  8, 10, 11,  8,  9,  8],
                       [9,  7,  6, 12,  5,  7, 12,  6,
                           12, 6,  4,  9,  9,  8,  7, 10],
                       [7,  6,  6, 10, 10,  9, 12,  7, 8,
                           9,  7,  7,  5,  8,  8,  8],
                       [6,  9,  9,  6, 10,  9,  8,  9, 9,
                           6, 10,  8,  8,  6,  6,  9],
                       [6,  8,  7,  9, 12,  7,  7, 10, 10,
                           7,  8,  9,  5,  7,  8,  7],
                       [6,  9,  9,  7, 12,  9,  4,  5,
                           7,  5, 13, 10, 13,  4,  6,  9],
                       [8, 10, 11,  8, 12,  7,  5,  8,
                           6,  7,  9,  7, 11,  5,  7,  7],
                       [5,  9,  8,  8,  5,  8,  6,  4, 12,
                           5,  8, 10, 15,  6,  7, 11],
                       [7,  8,  8,  5,  7, 10,  7,  6,
                           7,  8, 11,  7, 10,  8,  8,  9],
                       [7,  7,  9,  5,  6, 11,  7,  8,
                           7,  9, 11,  7, 11,  8,  7,  8],
                       [9,  8,  7,  9,  3,  9, 10,  4,
                           9,  5,  5,  9, 13,  9,  8,  9],
                       [5,  8,  9,  8,  8,  8,  7,  8,
                           9,  9, 10,  9,  7,  9,  8,  7],
                       [6,  7,  7, 13,  3,  6, 11,  7, 12,
                           8,  4,  8,  8, 10,  9,  8],
                       [8,  7,  6, 14,  6,  6, 11,  5,  9,  7,  4, 10,  6, 10,  9,  9]]],
                     device='cuda:0', dtype=torch.int8)
    b = torch.tensor([[[ -37,   19,   26,  -45,   -6,   -3,   10,   97,   80,    4,  -31,
           -51,   58,  -83,   31, -100],
         [ -16,   10,   -9,  -24,   15,  -39,   22,   26,  -10,   50,  -38,
             1,  -25,  -26,    5,  -15],
         [ -76,  -19,   44,  -56,  -13,  -12,   21,   51,   77,   -8,  -28,
            14,  -12,   -6,   15,   -5],
         [ -53,    2,  -11,  -42,   26,   10,  -82,   43,   25,   54,   69,
           -25,   70,   -2,  -37,  -71],
         [  15, -119,    2,  -66,  -54,  -64,  115,  112,  -95,  -75,    7,
            44,    8,  -74,  110,  -52],
         [ -53,  -16,   38,  -35,   16,  -98,   35,   93,    7,   14,  -53,
           -90,   47,  -54,   48,  -11],
         [ -23,   10,  -74,  -30,   18,  -85,   45,   61,   35,   57,   17,
           -98,  -16,  -19,  -67,   63],
         [ -78,  -55,   44, -105,  -25,  -89,   80,   81,   71,    3,   29,
             8,  -19,   44,    7,  -55],
         [ -36,  -56,   62,  -13,   50,  -21,  -63,   12,  -58,   -2,  -19,
            39,   -4,  104,   26,  -96],
         [ -73,  -37,    4,  -94,   -7,  -24,   24,   23,   29,  -40,    7,
             5,    1,    1,    0,   -6],
         [ -28,  -44,   70,  -63,   -8, -102,  107,   59,    6,  -51,  -47,
           -10,   47,  -37,   98,  -32],
         [  -3,    3,   67,    3,   56,  -52,  -20,  -10,  -24,   75,  -26,
           -11,   87,   53,   23,  -60],
         [ -67,   38,   27,  -80,   56,  -75,  -18,   76,   48,   -8, -105,
           -94,   80, -127,   51,  -88],
         [  -8,   41,   -5,    0,   86,  -21,   44,  -21,  -20,   80,  -21,
           -30,    4,   29,  -38,   55],
         [ -47,   32,   53,  -54,   -4,   58,    9,   28,    6,  -32,  -11,
            13,   51,  -33,   75,   -1],
         [  63,  -60,  -74,  -27,   22,  -55,   16,   51,   -5,  -54,    1,
          -101,    2,  -58,    1,  -45]]], device='cuda:0', dtype=torch.int8)
    scale = 0.0186
    c_gt = torch.bmm(a.float(), b.float()) * scale
    c_gt = c_gt.round().clamp(-128, 127).to(torch.int8)
    # ic(c_gt)
    b1 = b.transpose(1, 2).contiguous()
    c1 = bmm_s8t_s8n_s8t(a, b1, scale)
    # ic(c1)
    ic(torch.mean((c_gt.float() - c1.float()) ** 2))


@torch.no_grad()
def test_bmm_f32t_f32n_f32t():
    B, M, K, N = 300, 196, 64, 196
    a = torch.load('/tmp/q.pt')
    b = torch.load('/tmp/k.pt')
    assert a.size(0) == B
    assert a.size(1) == M
    assert a.size(2) == K
    assert b.size(1) == N
    scale = 0.125
    c_gt = torch.bmm(a * scale, b.transpose(-2, -1))
    c = bmm_f32t_f32n_f32t(a, b,scale)
    r2 = ((c_gt - c)**2).mean() / (c_gt**2).mean()
    ic(r2)

@torch.no_grad()
def test_bmm_f32t_f32n_f32t():
    # breakpoint()
    B, M, K, N = 300, 196, 64, 196
    a = torch.load('/tmp/q.pt')
    b = torch.load('/tmp/k.pt')
    assert a.size(0) == B
    assert a.size(1) == M
    assert a.size(2) == K
    assert b.size(1) == N
    scale = 0.125
    c_gt = torch.bmm(a * scale, b.transpose(-2, -1))
    c = bmm_f32t_f32n_f32t(a, b,scale)
    r2 = ((c_gt - c)**2).mean() / (c_gt**2).mean()
    ic(r2)

# @torch.no_grad()
def test_bmm_f32t_f32n_f32t_attn():
    B, M, K, N = 300, 196, 196,  64
    a = torch.load('/tmp/attn.pt')
    b = torch.load('/tmp/v.pt')
    # a= torch.rand(a.shape, device=a.device)
    # b= torch.rand(b.shape, device=b.device)
    assert a.size(0) == B
    assert a.size(1) == M
    assert a.size(2) == K
    assert b.size(2) == N
    scale = 1.0
    c_gt = torch.bmm(a * scale, b)
    c = bmm_f32t_f32n_f32t(b.transpose(-2, -1).contiguous(), a.transpose(-2, -1),scale).transpose(-2, -1)
    r2 = ((c_gt - c)**2).mean() / (c_gt**2).mean()
    ic(r2)

#(Pdb) print(a.shape)
#torch.Size([300, 196, 196])
#(Pdb) print(b.shape)
#torch.Size([300, 196, 64])



# (A_row V_row)_row ^ T = (V_row ^T A_row ^T)_row = (V^T_row A_col)_row
# (A_row V_row)_row = (A_row V_col ^T)_row
@torch.no_grad()
def test_bmm_s8t_s8n_s32t():
    # used by query x key
    B, M, K, N = 1, 512, 512, 12288
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8)
    b = torch.randint(-128, 127, (B, N, K), dtype=torch.int8)
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)
                     ).round().to(torch.int32)
    c = bmm_s8t_s8n_s32t(a.cuda(), b.cuda())
    ic(torch.allclose(c_gt, c.cpu()))


if __name__ == '__main__':
    print('test_bmm_s8t_s8n_s8t')
    test_bmm_s8t_s8n_s8t()
    # print('test_bmm_s8t_s8n_s8t_2')
    # test_bmm_s8t_s8n_s8t_2()
    # print('test_bmm_s8t_s8n_s32t')
    # test_bmm_s8t_s8n_s32t()
    print('test_bmm_s8t_s8n_f32t')
    test_bmm_s8t_s8n_f32t()
    # print('test_bmm_f32t_f32n_f32t')
    # test_bmm_f32t_f32n_f32t()
    # print('test_bmm_f32t_f32n_f32t_attn')
    # test_bmm_f32t_f32n_f32t_attn()
