import torch as t
import torch.nn as nn
import torch.nn.functional as F
 
class Stem_v4_Res2(nn.Module):
    """
    stem block for Inception-v4 and Inception-RestNet-v2
    """

    def __init__(self):
        super(Stem_v4_Res2, self).__init__()
        self.step1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = nn.Conv2d(64, 96, 3, 2, 0, bias=False)
        self.step3_1 = nn.Sequential(
            nn.Conv2d(160, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, 3, 1, 0, bias=False),
            nn.BatchNorm2d(96)
        )
        self.step3_2 = nn.Sequential(
            nn.Conv2d(160, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, 3, 1, 0, bias=False),
            nn.BatchNorm2d(96)
        )
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = nn.Conv2d(192, 192, 3, 2, 0, bias=False)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = t.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = t.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        out = t.cat((tmp1, tmp2), 1)
        return out

class Inception_A(nn.Module):
    """
    Inception-A block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n3, b4_n1, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(in_channels, b1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b1)
        )
        self.branch2 = nn.Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b3_n1),
            nn.Conv2d(b3_n1, b3_n3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(b3_n3)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b4_n1),
            nn.Conv2d(b4_n1, b4_n3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(b4_n3),
            nn.Conv2d(b4_n3, b4_n3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(b4_n3)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return t.cat((out1, out2, out3, out4), 1)

class Reduction_A(nn.Module):
    """
    Reduction-A block for Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2 nets
    """

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch2 = nn.Conv2d(in_channels, n, 3, 2, 0, bias=False)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(k),
            nn.Conv2d(k, l, 3, 1, 1, bias=False),
            nn.BatchNorm2d(l),
            nn.Conv2d(l, m, 3, 2, 0, bias=False),
            nn.BatchNorm2d(m),
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return t.cat((out1, out2, out3), 1)

class Inception_B(nn.Module):
    """
    Inception-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x7, b3_n7x1, b4_n1, b4_n1x7_1,
                 b4_n7x1_1, b4_n1x7_2, b4_n7x1_2):
        super(Inception_B, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(in_channels, b1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b1),
        )
        self.branch2 = nn.Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b3_n1),
            nn.Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            nn.BatchNorm2d(b3_n1x7),
            nn.Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False),
            nn.BatchNorm2d(b3_n7x1),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b4_n1),
            nn.Conv2d(b4_n1, b4_n1x7_1, (1, 7), (1, 1), (0, 3), bias=False),
            nn.BatchNorm2d(b4_n1x7_1),
            nn.Conv2d(b4_n1x7_1, b4_n7x1_1, (7, 1), (1, 1), (3, 0), bias=False),
            nn.BatchNorm2d(b4_n7x1_1),
            nn.Conv2d(b4_n7x1_1, b4_n1x7_2, (1, 7), (1, 1), (0, 3), bias=False),
            nn.BatchNorm2d(b4_n1x7_2),
            nn.Conv2d(b4_n1x7_2, b4_n7x1_2, (7, 1), (1, 1), (3, 0), bias=False),
            nn.BatchNorm2d(b4_n7x1_2),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return t.cat((out1, out2, out3, out4), 1)

class Reduction_B_v4(nn.Module):
    """
    Reduction-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b2_n1),
            nn.Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False),
            nn.BatchNorm2d(b2_n3),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b3_n1),
            nn.Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            nn.BatchNorm2d(b3_n1x7),
            nn.Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False),
            nn.BatchNorm2d(b3_n7x1),
            nn.Conv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=False),
            nn.BatchNorm2d(b3_n3),
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return t.cat((out1, out2, out3), 1)

class Inception_C(nn.Module):
    """
    Inception-C block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x3_3x1, b4_n1,
                 b4_n1x3, b4_n3x1, b4_n1x3_3x1):
        super(Inception_C, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(in_channels, b1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b1),
        )
        self.branch2 = nn.Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3_1 = nn.Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False)
        self.branch3_1x3 = nn.Conv2d(b3_n1, b3_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch3_3x1 = nn.Conv2d(b3_n1, b3_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)
        self.branch4_1 = nn.Sequential(
            nn.Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(b4_n1),
            nn.Conv2d(b4_n1, b4_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            nn.BatchNorm2d(b4_n1x3),
            nn.Conv2d(b4_n1x3, b4_n3x1, (3, 1), (1, 1), (1, 0), bias=False),
            nn.BatchNorm2d(b4_n3x1),
        )
        self.branch4_1x3 = nn.Conv2d(b4_n3x1, b4_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch4_3x1 = nn.Conv2d(b4_n3x1, b4_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        tmp = self.branch3_1(x)
        out3_1 = self.branch3_1x3(tmp)
        out3_2 = self.branch3_3x1(tmp)
        tmp = self.branch4_1(x)
        out4_1 = self.branch4_1x3(tmp)
        out4_2 = self.branch4_3x1(tmp)
        return t.cat((out1, out2, out3_1, out3_2, out4_1, out4_2), 1)

class Inception(nn.Module):
    """
    implementation of Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2
    """

    def __init__(self, version, num_classes):
        super(Inception, self).__init__()
        self.version = version
        self.stem = Stem_Res1() if self.version == "res1" else Stem_v4_Res2()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        if self.version == "v4":
            self.fc = nn.Linear(1536, num_classes)
        elif self.version == "res1":
            self.fc = nn.Linear(1792, num_classes)
        else:
            self.fc = nn.Linear(2144, num_classes)

    def __make_inception_A(self):
        layers = []
        if self.version == "v4":
            for _ in range(4):
                layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        elif self.version == "res1":
            for _ in range(5):
                layers.append(Inception_A_res(256, 32, 32, 32, 32, 32, 32, 256))
        else:
            for _ in range(5):
                layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == "v4":
            return Reduction_A(384, 192, 224, 256, 384) # 1024
        elif self.version == "res1":
            return Reduction_A(256, 192, 192, 256, 384) # 896
        else:
            return Reduction_A(384, 256, 256, 384, 384) # 1152

    def __make_inception_B(self):
        layers = []
        if self.version == "v4":
            for _ in range(7):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                          192, 192, 224, 224, 256))   # 1024
        elif self.version == "res1":
            for _ in range(10):
                layers.append(Inception_B_res(896, 128, 128, 128, 128, 896))  # 896
        else:
            for _ in range(10):
                layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))  # 1152
        return nn.Sequential(*layers)

    def __make_reduction_B(self):
        if self.version == "v4":
            return Reduction_B_v4(1024, 192, 192, 256, 256, 320, 320)  # 1536
        elif self.version == "res1":
            return Reduction_B_Res(896, 256, 384, 256, 256, 256, 256, 256)  # 1792
        else:
            return Reduction_B_Res(1152, 256, 384, 256, 288, 256, 288, 320)  # 2144

    def __make_inception_C(self):
        layers = []
        if self.version == "v4":
            for _ in range(3):
                layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        elif self.version == "res1":
            for _ in range(5):
                layers.append(Inception_C_res(1792, 192, 192, 192, 192, 1792))
        else:
            for _ in range(5):
                layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)
        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out,dim=1)

def inception_v4(num_classes):
    return Inception("v4", num_classes)


def inception_resnet_v1(num_classes):
    return Inception("res1", num_classes)


def inception_resnet_v2(num_classes):
    return Inception("res2", num_classes)







