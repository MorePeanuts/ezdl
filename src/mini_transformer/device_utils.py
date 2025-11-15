import torch
from typing import Literal


def get_single_device(kind: Literal['cpu', 'gpu'] = 'gpu'):
    """
    Get a single device based on the specified kind.

    Args:
        kind (Literal['cpu', 'gpu']): The kind of device to get. Defaults to 'gpu'.

    Returns:
        torch.device: The device object.
    """

    if kind == 'cpu':
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        print('No GPU or MPS device available')
        return torch.device('cpu')
