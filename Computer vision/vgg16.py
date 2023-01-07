import torch
import torch.nn as nn

class VGG16(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # layer 2
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # layer 4
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 5
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # layer 6
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 7
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # layer 8
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # layer 9
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 10
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # layer 11
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # layer 12
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            # fc1
            nn.Dropout(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            # fc2
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # fc3
            nn.Linear(4096, num_classes)
        )


    def forward(self, x: torch.tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x