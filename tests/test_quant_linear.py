from multiprocessing import dummy
import torch
from torch_int.nn import Int8Linear
from icecream import ic


@torch.no_grad()
def test_quant_linear_1():
    linear = torch.nn.Linear(16, 4, bias=False).cuda().half()
    linear.weight.copy_(
        torch.tensor([[-0.0279, -0.2490,  0.0114, -0.0024,  0.0257, -0.0735,  0.0860, -0.2352,
                       -0.2261,  0.0398, -0.0443, -0.2264,  0.2310, -0.1953,  0.1742,  0.2457],
                      [0.1306, -0.1079,  0.1936,  0.2227,  0.0271, -0.0440, -0.0673, -0.1841,
                       -0.0316,  0.0075,  0.2137, -0.2415, -0.1388, -0.1835,  0.0658,  0.1228],
                      [0.0399, -0.1190,  0.1387,  0.0562,  0.1017, -0.1037,  0.0772, -0.1639,
                       0.1174, -0.1809, -0.1707, -0.1005,  0.1559, -0.1802, -0.0812,  0.1442],
                      [0.0134,  0.0653, -0.2432,  0.2156, -0.0065, -0.0313,  0.1425, -0.1692,
                       -0.0691, -0.2374, -0.0140,  0.1893,  0.0249,  0.0800,  0.1508,  0.1681]],
                     device='cuda:0', dtype=torch.float16, requires_grad=False)
    )
    ic(linear.weight)
    int8_linear = Int8Linear.from_float(linear).cuda()
    dummy_input = torch.tensor([[-1.3945,  0.2177,  1.9121,  1.7041, -1.4561,  0.4175, -0.0382,  1.6543,
                                 -3.8965, -0.1681, -0.9111,  0.2949,  1.2559,  0.0072,  1.8398,  0.3503],
                                [1.6016, -0.5054,  1.5723, -1.7490,  1.4561,  0.1853,  0.2102, -1.0557,
                                 -1.0879, -1.8994, -0.7266,  0.9199, -0.8955, -0.5356,  3.2734,  0.7412],
                                [-1.6309, -1.5459,  1.0361,  1.1416, -1.0771,  1.3262, -1.1162, -1.1025,
                                 0.2747, -0.6743,  0.7500, -0.7388, -0.2966, -2.2324, -0.1644, -0.4609],
                                [1.0801, -0.8916,  1.6084, -0.1173,  1.4580,  1.2402,  0.5781, -0.8926,
                                 1.6924, -0.3442,  0.5269, -0.2009,  1.3594, -1.4453, -1.2910,  0.3298]],
                               device='cuda:0', dtype=torch.float16)
    ic(dummy_input)
    y = linear(dummy_input)
    ic(y)
    q_y = int8_linear(dummy_input)
    ic(q_y)
    mse = (y - q_y).pow(2).mean()
    ic(mse)

@torch.no_grad()
def test_quant_linear_2():
    for _ in range(10):
        linear = torch.nn.Linear(12288, 12288)
        linear.weight.copy_(torch.randn_like(linear.weight))
        linear.bias.copy_(torch.randn_like(linear.bias))
        linear = linear.cuda().half()
        int8_linear = Int8Linear.from_float(linear).cuda()
        dummy_input = torch.randn(512, 12288, device='cuda:0', dtype=torch.float16)
        y = linear(dummy_input).float()
        q_y = int8_linear(dummy_input).float()
        print(y)
        print(q_y)
        r2 = 1 - (y - q_y).pow(2).mean() / y.pow(2).mean()
        ic(r2)


if __name__ == '__main__':
    # test_quant_linear_1()
    test_quant_linear_2()
