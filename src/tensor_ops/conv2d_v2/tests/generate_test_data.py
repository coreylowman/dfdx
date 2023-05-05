import torch
import numpy as np

def gen_data(x_shape, w_shape, stride, padding, dilation, groups):
    x = torch.randn(*x_shape, requires_grad=True)
    w = torch.randn(*w_shape, requires_grad=True)
    y = torch.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)
    y.square().mean().backward()
    return {
        'x': x.detach().numpy(),
        'w': w.detach().numpy(),
        'y': y.detach().numpy(),
        'x_grad': x.grad.detach().numpy(),
        'w_grad': w.grad.detach().numpy(),
    }

torch.manual_seed(0)
test_data = {
    "default": gen_data((3, 15, 15), (5, 3, 4, 4), 1, 0, 1, 1),
    "s2": gen_data((3, 15, 15), (5, 3, 4, 4), 2, 0, 1, 1),
    "p1": gen_data((3, 15, 15), (5, 3, 4, 4), 1, 1, 1, 1),
    "d2": gen_data((3, 15, 15), (5, 3, 4, 4), 1, 0, 2, 1),
    "g2": gen_data((6, 15, 15), (6, 3, 4, 4), 1, 0, 1, 2),
    "g3": gen_data((18, 15, 15), (9, 6, 4, 4), 1, 0, 1, 3),
    "all": gen_data((9, 15, 15), (6, 3, 4, 4), 2, 1, 2, 3),
}

for lbl, data in test_data.items():
    for name, value in data.items():
        print(lbl, name, value.shape)
        np.save(f"data/{lbl}_{name}.npy", value)