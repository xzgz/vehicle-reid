import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from model.resnet import ResNet50
import math
import numpy as np


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        out += residual
        out = self.relu(out)

        return out


class CRNNResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(CRNNResNet, self).__init__()
        # self.inplanes, self.init_inplanes = 16, 16
        self.inplanes, self.init_inplanes = 64, 64

        self.conv1 = nn.Conv2d(3, self.init_inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.init_inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, self.init_inplanes, layers[0])
        self.layer2 = self._make_layer(block, 2 * self.init_inplanes, layers[1],
                                       kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.layer3 = self._make_layer(block, 4 * self.init_inplanes, layers[2],
                                       kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.layer4 = self._make_layer(block, 8 * self.init_inplanes, layers[3],
                                       kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if type(stride) == tuple:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=kernel_size, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, padding, downsample))
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
        x = self.layer5(x)

        return x


class CRNN(nn.Module):
    def __init__(self, shape, num_classes, loss_type, **kwargs):
        super(CRNN, self).__init__()
        self.loss_type = loss_type
        self.h, self.w = shape
        assert self.h == 32 and self.w == 96, "Image height and width should be 32 and 96."

        # self.resnet50 = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.crnn_resnet = CRNNResNet(Bottleneck, [3, 4, 6, 3])

        self.num_layers = 2
        self.lstm_inp_sz = 512
        self.seq_len = self.w // 4
        self.hidden_size = self.lstm_inp_sz // 2

        # self.feature = nn.Sequential(
        #     # conv1
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # conv2
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # conv3
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # conv4
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        #     # conv5
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     # conv6
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        #     # conv7
        #     nn.Conv2d(512, self.lstm_inp_sz, kernel_size=(2, 1)),
        #     nn.BatchNorm2d(self.lstm_inp_sz),
        #     nn.ReLU(),
        # )

        self.bilstm = nn.LSTM(input_size=self.lstm_inp_sz, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, bidirectional=True)

        # self.conv_redu = nn.Sequential(
        #     nn.Conv2d(2048, 512, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )
        self.dropout = nn.Dropout2d(0.5)
        # self.avgpool = nn.AvgPool2d(kernel_size=(1, self.seq_len), stride=1)
        self.fc = nn.Linear(self.lstm_inp_sz * self.seq_len, num_classes)

    def forward(self, x):
        x = self.crnn_resnet(x)
        # x = self.resnet50(x)
        # x = self.conv_redu(x)

        x = x.squeeze().permute(2, 0, 1)
        x = self.bilstm(x)[0].permute(1, 2, 0)
        x = x[:, :, np.newaxis, :]
        x = self.dropout(x)

        # x = self.avgpool(x)
        feat = x.contiguous().view(x.size(0), -1)
        x = self.fc(feat)
        feat_norm = feat / feat.norm(dim=1, keepdim=True)

        if self.training:
            if self.loss_type == 'xent':
                return x
            elif self.loss_type in ['xent_triplet', 'xent_tripletv2']:
                return x, feat_norm
        else:
            return feat_norm


class CRNNV2(nn.Module):
    def __init__(self, shape, num_classes, loss_type, **kwargs):
        super(CRNNV2, self).__init__()
        self.loss_type = loss_type
        self.h, self.w = shape
        assert self.h == 32 and self.w == 96, "Image height and width should be 32 and 96."

        self.crnn_resnet = CRNNResNet(Bottleneck, [3, 4, 6, 3])

        self.use_bilstm = True
        # self.use_bilstm = False

        self.lstm_num_layers = 2
        self.lstm_inp_sz = 512
        self.lstm_seq_len = self.w // 4
        self.num_plt_chars = 70
        if self.use_bilstm:
            self.lstm_hidden_size = self.lstm_inp_sz // 2
            self.bilstm = nn.LSTM(input_size=self.lstm_inp_sz, hidden_size=self.lstm_hidden_size,
                                  num_layers=self.lstm_num_layers, bidirectional=True)
        else:
            self.lstm_hidden_size = self.lstm_inp_sz
            self.bilstm = nn.LSTM(input_size=self.lstm_inp_sz, hidden_size=self.lstm_hidden_size,
                              num_layers=self.lstm_num_layers)

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.8)
        self.fc1 = nn.Linear(self.lstm_inp_sz, self.num_plt_chars)
        self.fc2 = nn.Linear(self.num_plt_chars * self.lstm_seq_len, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.crnn_resnet(x)
        x = x.squeeze().permute(2, 0, 1)
        x = self.bilstm(x)[0].permute(1, 2, 0)
        x = self.dropout1(x)
        x = x.contiguous().view(-1, self.lstm_inp_sz)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        feat = x.view(-1, self.lstm_seq_len * self.num_plt_chars)
        x = self.fc2(feat)
        feat_norm = feat / feat.norm(dim=1, keepdim=True)

        if self.training:
            if self.loss_type == 'xent':
                return x
            elif self.loss_type in ['xent_triplet', 'xent_tripletv2']:
                return x, feat_norm
        else:
            return feat_norm
