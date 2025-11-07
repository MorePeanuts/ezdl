from ..configuration_utils import PreTrainedConfig


class NiNConfig(PreTrainedConfig):
    
    model_type = 'nin'
    
    def __init__(
        self,
        num_classes: int = 1000,
        in_features: list[int] = [3, 224, 224],
        nin_out_channels: list[int] = [96, 256, 384],
        nin_kernel_size: list[int] = [11, 5, 3, 3],
        nin_stride: list[int] = [4, 1, 1, 1],
        nin_padding: list[int] = [0, 2, 1, 1],
        dropout: float = 0.5,
        pool_kernel_size: int = 3,
        pool_stride: int = 2,
        pool_padding: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.nin_out_channels = nin_out_channels + [num_classes]
        self.nin_kernel_size = nin_kernel_size
        self.nin_stride = nin_stride
        self.nin_padding = nin_padding
        self.dropout = dropout
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        
        assert len(self.nin_out_channels) == len(self.nin_kernel_size) == len(self.nin_stride) == len(self.nin_padding)