import torch
from mini_transformer.models.llama import rotate_half


def test_rotate_half():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    y = rotate_half(x)
    assert y.tolist() == [-3, -4, 1, 2]
    
    
def test_inference_forward():
    pass
    
    
def test_train_backward():
    pass
    