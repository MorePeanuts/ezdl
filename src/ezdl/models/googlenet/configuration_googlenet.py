from ..modeling_utils import PreTrainedConfig


class GoogLeNetConfig(PreTrainedConfig):
    model_type = "googlenet"

    def __init__(self, num_classes: int = 1000, in_features: list[int] = [1, 28, 28]):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
