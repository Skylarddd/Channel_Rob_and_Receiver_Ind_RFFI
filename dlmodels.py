
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sympy.abc import x
import torchvision.models as models


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, first_layer=False):
        super(ResBlock, self).__init__()
        self.first_layer = first_layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.first_layer:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.first_layer:
            identity = self.shortcut(x)
        out += identity
        out = F.relu(out)
        return out
    

class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.resblock1 = ResBlock(32, 32, 3)
        self.resblock2 = ResBlock(32, 32, 3)
        self.resblock3 = ResBlock(32, 64, 3, first_layer=True)
        self.resblock4 = ResBlock(64, 64, 3)
        self.avgpool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(26624, 512)
        # self.fc1 = nn.Linear(24960, 512)
        self.fc2 = nn.Linear(512, 256)
        self.reset()
    
    def reset(L):
        # Initialization using fan-in
        if isinstance(L, nn.Conv2d):
            n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
            nn.init.kaiming_normal_(L.weight.data)

        elif isinstance(L, nn.BatchNorm2d):
            L.weight.data.fill_(1)
            L.bias.data.fill_(0)

    def forward(self, x):

        x = F.relu(self.conv(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avgpool(x)
        flatten = torch.flatten(x, 1)

        dim512 = self.fc1(flatten)
        dim512 = F.normalize(dim512, p=2, dim=1)
        # relu
        # dim512 = F.relu(dim512)
        dim256 = self.fc2(dim512)
        # return flatten, dim512, dim256
        # return flatten, dim256
        return dim256



class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GradientReversal(nn.Module):
    def forward(self, input):
        return GradientReversalLayer.apply(input)

    
class Clf_Model_reverse(nn.Module):
    def __init__(self, base_model, num_classes, num_rx):
        super(Clf_Model_reverse, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.num_rx = num_rx
        hidden_lay = 256
        hidden_lay1 = 128
        hidden_tx = self.num_classes
        hidden_rx = self.num_rx

        self.gradient_reversal = GradientReversal()
        self.dense_layer1 = nn.Linear(hidden_lay, hidden_lay1)
        self.fc_tx = nn.Linear(hidden_lay1, hidden_tx)
        self.fc_rx = nn.Linear(hidden_lay1, hidden_rx)
        self.reset()

    def reset(L):
        # Initialization using fan-in
        if isinstance(L, nn.Conv2d):
            n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
            nn.init.kaiming_normal_(L.weight.data)

        elif isinstance(L, nn.BatchNorm2d):
            L.weight.data.fill_(1)
            L.bias.data.fill_(0)

    def base_out(self, x):
        projected = self.base_model(x)
        # _, _, projected = self.base_model(x)
        return projected

    def forward(self, x):
        feature = self.base_out(x)
        dim128 = self.dense_layer1(feature)
        # relu
        tx = F.relu(dim128)
        clf_out = self.fc_tx(tx)
        dim128_rx = self.gradient_reversal(feature)
        project2 = self.dense_layer1(dim128_rx)
        project2 = F.relu(project2)
        # project2 = F.leaky_relu(project2)
        rx_clf = self.fc_rx(project2)
        rx_clf = F.softmax(rx_clf, dim=1)

        return clf_out, rx_clf

class Clf_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super(Clf_Model, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        hidden_lay1 = 256
        hidden_lay2 = self.num_classes
        self.dense_layer1 = nn.Linear(hidden_lay1, hidden_lay2)

        self.reset()

    def reset(L):
        # Initialization using fan-in
        if isinstance(L, nn.Conv2d):
            n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
            nn.init.kaiming_normal_(L.weight.data)

        elif isinstance(L, nn.BatchNorm2d):
            L.weight.data.fill_(1)
            L.bias.data.fill_(0)

    def base_out(self, x):
        # dim512, projected = self.base_model(x)
        projected = self.base_model(x)
        # _, _, projected = self.base_model(x)
        # return dim512, projected
        return projected

    def clf(self, x):
        label = self.dense_layer1(x)
        
        return label

    def forward(self, x):

        project = self.base_out(x)
        clf_out = self.clf(project)

        return clf_out, project
    
class ResBlock_iq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResBlock_iq, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2))
        
        # If the input and output channels are not the same or stride > 1, use a 1x1 convolution for the shortcut
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        
        out += identity
        out = F.relu(out)
        
        return out
    
        
class ClassificationNet_iq(nn.Module):
    def __init__(self):
        super(ClassificationNet_iq, self).__init__()
        
        # Initial convolution layer with kernel size (7, 2) since the last dimension is only 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(128, 1), stride=1, padding=(3, 0))  # 7x2 conv, 32 channels
        self.resblock1 = ResBlock_iq(32, 32, kernel_size=(3, 1))  # 3x1 conv, 32 channels
        self.resblock2 = ResBlock_iq(32, 32, kernel_size=(3, 1))  # 3x1 conv, 32 channels
        self.resblock3 = ResBlock_iq(32, 64, kernel_size=(3, 1), stride=(2, 1))  # 3x1 conv, 64 channels, downsample
        
        self.resblock4 = ResBlock_iq(64, 64, kernel_size=(3, 1))  # 3x1 conv, 64 channels

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1))  # Average pooling
        self.flatten = nn.Flatten()
        
        # Calculate the input features for the first fully connected layer
        self.fc1 = nn.Linear(258304, 512)  # Flatten layer and fully connected layer, assuming input size
        self.fc2 = nn.Linear(512, 256)
        self.reset()
    
    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        
        dim512 = F.relu(self.fc1(x))
        dim256 = self.fc2(dim512)
        
        return dim256
        # return dim512, dim256