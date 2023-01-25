import torch
import torch.nn as nn

# (kernel_size, filters, stride, padding)
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.lr(self.bn(self.conv(x)))
    

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs) -> None:
        super().__init__()

        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(ConvBlock(
                            in_channels,
                            out_channels=x[1],
                            kernel_size=x[0],
                            stride=x[2],
                            padding=x[3],   
                    )
                )
                in_channels = x[1]
            elif type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                            ConvBlock(
                                in_channels,
                                out_channels=conv1[1],
                                kernel_size=conv1[0],
                                stride=conv1[2],
                                padding=conv1[3],
                            ),
                            ConvBlock(
                                conv1[1],
                                out_channels=conv2[1],
                                kernel_size=conv2[0],
                                stride=conv2[2],
                                padding=conv2[3],
                            ),
                    ]
                
                in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5))

        )
