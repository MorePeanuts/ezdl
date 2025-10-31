import torch
from ezdl.models.llama import rotate_half


def test_rotate_half():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    y = rotate_half(x)
    assert y.tolist() == [-3, -4, 1, 2]
    