import torch
import torch.nn as nn
import torch.nn.functional as F

class FourColumnMDNN(nn.Module):
    def __init__(self):
        super(FourColumnMDNN, self).__init__()

        # ✅ Very Large Receptive Field Column
        self.column1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # ✅ Large Receptive Field Column
        self.column2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # ✅ Medium Receptive Field Column
        self.column3 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )

        # ✅ Small Receptive Field Column (fine details for dense crowds)
        self.column4 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        # ✅ Fusion Layer
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 64 + 80 + 96, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.column1(x)
        x2 = self.column2(x)
        x3 = self.column3(x)
        x4 = self.column4(x)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.fusion(x_cat)
        return out


# model = FourColumnMDNN()
# model.summary()

# dummy_input = tf.random.normal((1, 256, 256, 3))
# output = model(dummy_input)