"""
by zk
用于部件分类的推理函数调用接口
最后修改：20220301
"""

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn as nn
import math
import cv2
import os
from torchvision.models.resnet import resnet50


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class SENet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_resnet_50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def equalization(im):
    """
    直方图均衡化，用于王泰蔚的分类模型
    """
    nums = 256
    # if(len(im.shape)!=2):
    r, c, d = im.shape
    if (d == 3):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    elif (d == 0):
        print('WTF is wrong about the image sizes?')
        return None

    hist = {}
    for bin in range(nums):
        hist[bin] = 0
    for i in range(r):
        for j in range(c):
            if (hist.get(im[i][j]) is None):
                hist[im[i][j]] = 0
            hist[im[i][j]] += 1
    # normalise
    n = r * c
    for key in hist.keys():
        hist[key] = float(hist[key]) / n

    val = 0.0
    temp = hist.copy()
    for i in range(256):
        val = val + hist[i]
        temp[i] = val

    grayImg = np.zeros((r, c), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            grayImg[i][j] = int((nums - 1) * temp[im[i][j]] + 0.5)
    return grayImg


def screw_classify(im_array: np.array, model_path: str):
    """
    by zk
    螺钉螺帽分类调用接口
    """
    # config
    number_classes = 2
    class_names = ['001', '002']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    if 'traced' in model_path:
        model = torch.jit.load(model_path)
    else:
        model = se_resnet_50(num_classes=number_classes)
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # input
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.fromarray(im_array).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0)  # 拓展维度
    img_tensor = img_tensor.to(device)

    # inference
    output = model(img_tensor)  # 全连接层输出
    probs = torch.softmax(output, 1)
    max_prob = torch.max(probs, 1)[0].data.cpu()  # 最大softmax概率
    _, pred_cls_index = torch.max(output.data, 1)
    predict_cls = class_names[int(pred_cls_index)]

    if predict_cls == '001':
        result = 1
    else:
        result = 0

    return result, max_prob


def Distance_Saddles_classify(im_array: np.array, model_path: str):
    """
    by zk
    离墙码分类函数调用接口
    """
    # config
    number_classes = 50
    class_names = list(range(1, 51))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    if model_path.endswith('.pt'):
        model = torch.jit.load(model_path)
    else:
        model = resnet50(num_classes=number_classes)  # for WTW
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # input
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    im_array = equalization(im_array)  # for WTW
    img = Image.fromarray(im_array).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0)  # 拓展维度
    img_tensor = img_tensor.to(device)

    # inference
    output = model(img_tensor)  # 全连接层输出
    probs = torch.softmax(output, 1)
    max_prob = torch.max(probs, 1)[0].data.cpu()  # 最大softmax概率
    _, pred_cls_index = torch.max(output.data, 1)
    predict_cls = class_names[int(pred_cls_index)]

    if predict_cls == 6:
        result = 0
    else:
        result = 1

    return result, max_prob


def UBolts_classify(im_array: np.array, model_path: str):
    """
    by zk
    U型码分类函数调用接口
    """
    # config
    number_classes = 50
    class_names = list(range(1, 51))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    if model_path.endswith('.pt'):
        model = torch.jit.load(model_path)
    else:
        model = resnet50(num_classes=number_classes)  # for WTW
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # input
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    im_array = equalization(im_array)  # for WTW
    img = Image.fromarray(im_array).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0)  # 拓展维度
    img_tensor = img_tensor.to(device)

    # inference
    output = model(img_tensor)  # 全连接层输出
    probs = torch.softmax(output, 1)
    max_prob = torch.max(probs, 1)[0].data.cpu()  # 最大softmax概率
    _, pred_cls_index = torch.max(output.data, 1)
    predict_cls = class_names[int(pred_cls_index)]

    if predict_cls == 7:
        result = 0
    else:
        result = 1

    return result, max_prob


def Leakage_cable_Fixture_classify(im_array, model_path):
    """
    漏缆卡具分类函数调用接口
    """
    # config
    number_classes = 50
    class_names = list(range(1, 51))
    device = "cpu"

    # model
    if model_path.endswith('.pt'):
        model = torch.jit.load(model_path)
    else:
        model = resnet50(num_classes=number_classes)  # for WTW
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # input
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    im_array = equalization(im_array)  # for WTW
    img = Image.fromarray(im_array).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0)  # 拓展维度
    img_tensor = img_tensor.to(device)

    # inference
    output = model(img_tensor)  # 全连接层输出
    probs = torch.softmax(output, 1)
    max_prob = torch.max(probs, 1)[0].data.cpu()  # 最大softmax概率
    _, pred_cls_index = torch.max(output.data, 1)
    predict_cls = class_names[int(pred_cls_index)]

    if predict_cls > 4:
        result = 0
    else:
        result = 1

    return result, max_prob


if __name__ == "__main__":

    # path
    # model_path = 'experiment/test/results_screw/20210830/41.pkl'
    model_path = '../25_traced.pt'
    test_path = r'C:\Users\zk\Desktop\Project-411_classification\datasets\screw20211206\relabeled\smallImg\001'

    # 1. test single image
    img_path = os.path.join(test_path, '8_0_021_1_0.png')
    img_array = cv2.imread(img_path, 1)
    result, prob = Leakage_cable_Fixture_classify(img_array, model_path)
    print('predicted class: {} with prob: {}'.format(result, prob))

    # # 2. test multiple images
    # for file in os.listdir(test_path):
    #     # input
    #     img_path = os.path.join(test_path, file)
    #     img_array = cv2.imread(img_path, 1)
    #
    #     # inference
    #     result, prob = Leakage_cable_Fixture_classify(img_array, model_path)
    #     print('predicted class: {} with prob: {}'.format(result, prob))
