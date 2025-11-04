from ..configuration_utils import PreTrainedConfig


class VGGConfig(PreTrainedConfig):
    
    model_type = 'vgg'
    
    def __init__(
        self,
        num_classes: int = 1000,
        in_features: list[int] = [3, 224, 224],
        fc1_out_features: int = 4096,
        dropout_fc1: float = 0.5,
        fc2_out_features: int = 4096,
        dropout_fc2: float = 0.5,
        vgg_pairs: list[tuple[int, int]] = [
            (1, 64),
            (1, 128),
            (2, 512),
            (2, 512)
        ],
        vgg_kernel_size: int = 3,
        vgg_padding: int = 1,
        vgg_stride: int = 1,
        pool_kernel_size: int = 2,
        pool_padding: int = 0,
        pool_stride: int = 2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.fc1_out_features = fc1_out_features
        self.dropout_fc1 = dropout_fc1
        self.fc2_out_features = fc2_out_features
        self.dropout_fc2 = dropout_fc2
        self.vgg_pairs = vgg_pairs
        self.vgg_kernel_size = vgg_kernel_size
        self.vgg_padding = vgg_padding
        self.vgg_stride = vgg_stride
        self.pool_kernel_size = pool_kernel_size
        self.pool_padding = pool_padding
        self.pool_stride = pool_stride