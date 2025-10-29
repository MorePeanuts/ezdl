from ..modeling_utils import PreTrainedConfig


class AlexNetConfig(PreTrainedConfig):
    """
    """
    
    model_type = 'alexnet'
    
    def __init__(
        self,
        num_classes: int = 1000,
        in_features: list[int] = [3, 224, 224],
        c1_out_channels: int = 96,
        c1_kernel_size: int | list[int] = 11,
        c1_padding: int | list[int] = 0,
        c1_stride: int | list[int] = 4,
        p1_kernel_size: int | list[int] = 3,
        p1_padding: int | list[int] = 1,
        p1_stride: int | list[int] = 2,
        c2_out_channels: int = 256,
        c2_kernel_size: int | list[int] = 5,
        c2_padding: int | list[int] = 2,
        c2_stride: int | list[int] = 1,
        p2_kernel_size: int | list[int] = 3,
        p2_padding: int | list[int] = 0,
        p2_stride: int | list[int] = 2,
        c3_out_channels: int = 384,
        c3_kernel_size: int | list[int] = 3,
        c3_padding: int | list[int] = 1,
        c3_stride: int | list[int] = 1,
        c4_out_channels: int = 384,
        c4_kernel_size: int | list[int] = 3,
        c4_padding: int | list[int] = 1,
        c4_stride: int | list[int] = 1,
        c5_out_channels: int = 256,
        c5_kernel_size: int | list[int] = 3,
        c5_padding: int | list[int] = 1,
        c5_stride: int | list[int] = 1,
        p5_kernel_size: int | list[int] = 3,
        p5_padding: int | list[int] = 0,
        p5_stride: int | list[int] = 2,
        fc1_out_features: int = 4096,
        dropout_fc1: float = 0.5,
        fc2_out_features: int = 4096,
        dropout_fc2: float = 0.5,
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
        self.c3_out_channels = c3_out_channels
        self.c3_kernel_size = c3_kernel_size
        self.c3_padding = c3_padding
        self.c3_stride = c3_stride
        self.c4_out_channels = c4_out_channels
        self.c4_kernel_size = c4_kernel_size
        self.c4_padding = c4_padding
        self.c4_stride = c4_stride
        self.c5_out_channels = c5_out_channels
        self.c5_kernel_size = c5_kernel_size
        self.c5_padding = c5_padding
        self.c5_stride = c5_stride
        self.p5_kernel_size = p5_kernel_size
        self.p5_padding = p5_padding
        self.p5_stride = p5_stride
        self.fc1_out_features = fc1_out_features
        self.dropout_fc1 = dropout_fc1
        self.fc2_out_features = fc2_out_features
        self.dropout_fc2 = dropout_fc2
