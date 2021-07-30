import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


def weights_init(m):
    if (type(m) == nn.Conv2d or type(m) == nn.Linear) and hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act=None, use_bn=False, dropout=None, **kwargs):
        super(LinearLayer, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.lin = nn.Linear(input_dim, output_dim, **kwargs)
        self.bn = nn.BatchNorm1d(output_dim) if use_bn else None
        if act == 'relu':
            self.act = nn.ReLU()
        if act == 'elt_wise':
            self.act = SliceMax()
        else:
            self.act = act

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        if self.bn:
            return self.act(self.bn(self.lin(x)))
        return self.act(self.lin(x))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, s=1, pad=1, act=None, use_bn=False, dropout=None, **kwargs):
        super(ConvLayer, self).__init__()
        self.dp = nn.Dropout2d(dropout) if dropout else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=s, padding=pad, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'elt_wise':
            self.act = SliceMax()
        else:
            self.act = act

    def forward(self, x):
        if self.dp:
            x = self.dp(x)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class SliceLayer(nn.Module):
    def __init__(self, slice_point=2):
        super(SliceLayer, self).__init__()
        self.slice = slice_point

    def forward(self, x):
        split_point = x.size(1) // self.slice
        if len(x.size()) == 2:
            return x[:, :split_point], x[:, split_point:]
        else:
            return x[:, :split_point, :, :], x[:, split_point:, :, :]


class SliceMax(nn.Module):
    def __init__(self, **kwargs):
        super(SliceMax, self).__init__()
        self.slice = SliceLayer(**kwargs)

    def forward(self, x):
        t1, t2 = self.slice(x)
        return torch.max(t1, t2)


class PreTrained(object):
    def __init__(self, architecture, base_network=False, fine_tune=None):
        self.architecture = architecture
        self.base_network = base_network
        self.frozen = fine_tune

    def set_parameter_requires_grad(self, model):
        if self.base_network:
            print('Using frozen base')
            for param in model.parameters():
                param.requires_grad = False
        if self.frozen:
            for name, params in model.named_parameters():
                if any([n in name for n in self.frozen]):
                    params.requires_grad = True
                else:
                    params.requires_grad = False

    def initialize_model(self, num_classes=4):
        if self.architecture == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            self.set_parameter_requires_grad(model)
            feats_in = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Linear(feats_in, num_classes)
            )
        elif self.architecture == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.set_parameter_requires_grad(model)
            feats_in = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Linear(feats_in, num_classes, bias=False)
            )
        elif self.architecture == 'resnet50':
            model = models.resnet50(pretrained=True)
            self.set_parameter_requires_grad(model)
            feats_in = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Linear(feats_in, num_classes, bias=False)
            )
        elif self.architecture == 'squeezenet':
            model = models.squeezenet1_1(pretrained=True)
            self.set_parameter_requires_grad(model)
            model.classifier[1] = nn.Conv2d(512, 4, kernel_size=1, stride=1)
        elif self.architecture == 'mnasnet':
            model = models.mnasnet1_0(pretrained=True)
            self.set_parameter_requires_grad(model)
            feats_in = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Linear(feats_in, num_classes)
            )
        elif self.architecture == 'densenet':
            model = models.densenet161(pretrained=True)
            self.set_parameter_requires_grad(model)
            feats_in = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Linear(feats_in, num_classes)
            )
        else:
            model = None
            print('No valid structure name')
            exit()
        return model


class SOTANetwork(nn.Module):
    def __init__(self, in_channels=1, num_outputs=4, **kwargs):
        super(SOTANetwork, self).__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        if in_channels == 1:
            self.conv1 = nn.Conv2d(in_channels, 3, 1, padding=0)
        self.pre_trained = self.get_conv_base(**kwargs)

    def get_conv_base(self, **kwargs):
        tuner = PreTrained(**kwargs)
        model = tuner.initialize_model(self.num_outputs)
        return model

    def forward(self, x):
        if self.in_channels == 1:
            x = self.conv1(x)
        x = self.pre_trained(x)
        return x

