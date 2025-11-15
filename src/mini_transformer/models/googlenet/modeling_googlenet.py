import torch
import torch.nn as nn
from ..modeling_utils import PreTrainedModel
from .configuration_googlenet import GoogLeNetConfig


class GoogLeNetPreTrainedModel(PreTrainedModel):
    config_class = GoogLeNetConfig
    base_model_prefix = 'cnn'

    def __init__(self, config: GoogLeNetConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        else:
            super()._init_weights(module)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        c1_out_channels,
        c2_1_out_channels,
        c2_2_out_channels,
        c3_1_out_channels,
        c3_2_out_channels,
        c4_out_channels,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.seq1 = nn.Sequential(nn.LazyConv2d(c1_out_channels, kernel_size=1), nn.ReLU())
        self.seq2 = nn.Sequential(
            nn.LazyConv2d(c2_1_out_channels, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c2_2_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.seq3 = nn.Sequential(
            nn.LazyConv2d(c3_1_out_channels, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c3_2_out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.seq4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv2d(c4_out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return torch.cat(
            [self.seq1(x), self.seq2(x), self.seq3(x), self.seq4(x)], dim=1
        )  # concat the channel dimension


class GoogLeNetModel(GoogLeNetPreTrainedModel):
    def __init__(self, config: GoogLeNetConfig):
        super().__init__(config)

        in_channels, in_size = config.in_features[0], config.in_features[1:]

        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LazyConv2d(64, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(64, 96, 128, 16, 32, 32),
            InceptionBlock(128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(192, 96, 208, 16, 48, 64),
            InceptionBlock(160, 112, 224, 24, 64, 64),
            InceptionBlock(128, 128, 256, 24, 64, 64),
            InceptionBlock(112, 144, 288, 32, 64, 64),
            InceptionBlock(256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(256, 160, 320, 32, 128, 128),
            InceptionBlock(384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        x = torch.randn(1, in_channels, *in_size)
        self.net(x)

    def forward(self, x):
        return self.net(x)


class GoogLeNetModelForClassification(GoogLeNetPreTrainedModel):
    def __init__(self, config: GoogLeNetConfig):
        super().__init__(config)
        self.model = GoogLeNetModel(config)
        self.head = nn.LazyLinear(out_features=config.num_classes)

        in_channels, in_size = config.in_features[0], config.in_features[1:]
        x = torch.randn(1, in_channels, *in_size)
        self.head(self.model(x))

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
