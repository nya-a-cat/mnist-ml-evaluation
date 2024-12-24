import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleNet(nn.Module):
    """
    实现了基于Capsule的神经网络模型
    包含多个分支的分类路径和可学习的分支权重
    """

    def __init__(self):
        super(CapsuleNet, self).__init__()
        # 第一个卷积层：输入通道1(灰度图),输出256通道,9x9卷积核
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.bn1 = nn.BatchNorm2d(256)  # 添加批归一化

        # Primary Capsules: 将特征转换为capsules
        self.primary_capsules = nn.Conv2d(256, 32 * 8, kernel_size=9, stride=2)
        self.bn2 = nn.BatchNorm2d(32 * 8)

        # Digit Capsules: 最终的数字分类capsules
        self.digit_capsules = nn.Linear(32 * 8 * 6 * 6, 10 * 16)

        # 分支权重：用于组合多个分类分支的输出
        self.branch_weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)

        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(0.5)

    def squash(self, x):
        """
        Squash激活函数: 将向量压缩到长度在[0,1]范围内
        保持方向不变,但调整大小以表示概率
        """
        squared_norm = (x ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        # 基础特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        # Primary capsules处理
        x = self.bn2(self.primary_capsules(x))
        x = x.view(x.size(0), 8, -1)
        x = self.squash(x)

        # Digit capsules处理
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        digit_caps = self.digit_capsules(x)
        digit_caps = digit_caps.view(digit_caps.size(0), 10, 16)
        digit_caps = self.squash(digit_caps)

        # 计算分类概率
        probs = torch.norm(digit_caps, dim=-1)

        # 加权分支输出
        weighted_probs = self.softmax(self.branch_weights)

        return probs, weighted_probs
