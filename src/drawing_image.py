import torch
from torch import nn
from torchviz import make_dot

# 定义CapsuleNet模型
class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.primary_capsules = nn.Conv2d(256, 32 * 8, kernel_size=9, stride=2)
        self.bn2 = nn.BatchNorm2d(32 * 8)
        self.digit_capsules = nn.Linear(32 * 8 * 6 * 6, 10 * 16)
        self.branch_weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(0.5)

    def squash(self, x):
        squared_norm = (x ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.dropout(x)
        x = self.bn2(self.primary_capsules(x))
        x = x.view(x.size(0), 8, -1)
        x = self.squash(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        digit_caps = self.digit_capsules(x)
        digit_caps = digit_caps.view(digit_caps.size(0), 10, 16)
        digit_caps = self.squash(digit_caps)
        probs = torch.norm(digit_caps, dim=-1)
        weighted_probs = self.softmax(self.branch_weights)
        return probs, weighted_probs

# 创建一个模型实例
model = CapsuleNet()

# 创建一个假输入，用于生成计算图
x = torch.randn(1, 1, 28, 28)  # 假设输入大小为28x28的灰度图像

# 通过模型进行前向传播
y, weighted_probs = model(x)

# 使用torchviz绘制计算图
dot = make_dot(y, params=dict(model.named_parameters()))

# 保存并显示图
dot.render("capsulenet", format="png", cleanup=True)
