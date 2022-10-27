import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        # 分类网络--全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # N*3*224*224
        x = torch.flatten(x, start_dim=1)  # 展平N*512*7*7
        x = self.classifier(x)  # N*512*7*7
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # 该方法也被称为glorot的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏值置为0
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 初始化
                nn.init.constant_(m.bias, 0)  # 偏值置为0
