import torch
from torch import nn
from mini_transformer.models.modeling_utils import PreTrainedModel
from mini_transformer.models.vgg.configuration_vgg import VGGConfig


class VGGPreTrainedModel(PreTrainedModel):
    config_class = VGGConfig
    base_model_prefix = 'cnn'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        else:
            super()._init_weights(module)


class VGGBlock(nn.Module):
    def __init__(self, num_convs, out_channels, config: VGGConfig):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_convs):
            self.layers.append(
                nn.LazyConv2d(
                    out_channels, config.vgg_kernel_size, config.vgg_stride, config.vgg_padding
                )
            )
            self.layers.append(nn.ReLU())
        self.layers.append(
            nn.MaxPool2d(
                config.pool_kernel_size,
                config.pool_stride,
                config.pool_padding,
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VGGModel(VGGPreTrainedModel):
    def __init__(self, config: VGGConfig):
        super().__init__(config)

        in_channels, in_size = config.in_features[0], config.in_features[1:]

        self.blocks = nn.ModuleList()
        for num_convs, out_channels in config.vgg_pairs:
            self.blocks.append(VGGBlock(num_convs, out_channels, config))

        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(config.fc1_out_features),
            nn.ReLU(),
            nn.Dropout(config.dropout_fc1),
            nn.LazyLinear(config.fc2_out_features),
            nn.ReLU(),
            nn.Dropout(config.dropout_fc2),
        )

        x = torch.randn(1, in_channels, *in_size)
        for block in self.blocks:
            x = block(x)
        self.fcs(x)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fcs(x)
        return x


class VGGModelForClassification(VGGPreTrainedModel):
    def __init__(self, config: VGGConfig):
        super().__init__(config)
        self.model = VGGModel(config)
        self.head = nn.Linear(config.fc2_out_features, config.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
