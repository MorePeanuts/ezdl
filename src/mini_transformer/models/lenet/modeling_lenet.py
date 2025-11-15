import torch
import torch.nn as nn
from ..modeling_utils import PreTrainedModel
from .configuration_lenet import LeNetConfig


class LeNetPreTrainedModel(PreTrainedModel):
    """ """

    config_class = LeNetConfig
    base_model_prefix = 'cnn'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        else:
            super()._init_weights(module)


class LeNetModel(LeNetPreTrainedModel):
    """ """

    def __init__(self, config: LeNetConfig):
        super().__init__(config)

        c1_in_channels, in_size = config.in_features[0], config.in_features[1:]

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=c1_in_channels,
                out_channels=config.c1_out_channels,
                kernel_size=config.c1_kernel_size,
                stride=config.c1_stride,
                padding=config.c1_padding,
            ),
            nn.Sigmoid(),
            nn.AvgPool2d(
                kernel_size=config.p1_kernel_size,
                stride=config.p1_stride,
                padding=config.p1_padding,
            ),
            nn.LazyConv2d(
                out_channels=config.c2_out_channels,
                kernel_size=config.c2_kernel_size,
                stride=config.c2_stride,
                padding=config.c2_padding,
            ),
            nn.Sigmoid(),
            nn.AvgPool2d(
                kernel_size=config.p2_kernel_size,
                stride=config.p2_stride,
                padding=config.p2_padding,
            ),
            nn.Flatten(),
            nn.LazyLinear(out_features=config.fc1_out_features),
            nn.Sigmoid(),
            nn.LazyLinear(out_features=config.fc2_out_features),
            nn.Sigmoid(),
        )

        self.net(torch.randn(1, c1_in_channels, *in_size))

    def forward(self, x):
        return self.net(x)


class LeNetModelForClassification(LeNetPreTrainedModel):
    def __init__(self, config: LeNetConfig):
        super().__init__(config)
        self.model = LeNetModel(config)
        self.head = nn.Linear(config.fc2_out_features, config.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
