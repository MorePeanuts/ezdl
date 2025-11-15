from typing import Literal
from ..configuration_utils import PreTrainedConfig


class LeNetConfig(PreTrainedConfig):
    """ """

    model_type = 'lenet'

    def __init__(
        self,
        num_classes: int = 10,
        in_features: list[int] = [1, 28, 28],
        c1_out_channels: int = 6,
        c1_kernel_size: int | list[int] = 5,
        c1_padding: int | list[int] = 2,
        c1_stride: int | list[int] = 1,
        p1_kernel_size: int | list[int] = 2,
        p1_padding: int | list[int] = 0,
        p1_stride: int | list[int] = 2,
        c2_out_channels: int = 16,
        c2_kernel_size: int | list[int] = 5,
        c2_padding: int | list[int] = 0,
        c2_stride: int | list[int] = 1,
        p2_kernel_size: int | list[int] = 2,
        p2_padding: int | list[int] = 0,
        p2_stride: int | list[int] = 2,
        fc1_out_features: int = 120,
        fc2_out_features: int = 84,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.c1_out_channels = c1_out_channels
        self.c1_kernel_size = c1_kernel_size
        self.c1_padding = c1_padding
        self.c1_stride = c1_stride
        self.p1_kernel_size = p1_kernel_size
        self.p1_padding = p1_padding
        self.p1_stride = p1_stride
        self.c2_out_channels = c2_out_channels
        self.c2_kernel_size = c2_kernel_size
        self.c2_padding = c2_padding
        self.c2_stride = c2_stride
        self.p2_kernel_size = p2_kernel_size
        self.p2_padding = p2_padding
        self.p2_stride = p2_stride
        self.fc1_out_features = fc1_out_features
        self.fc2_out_features = fc2_out_features
