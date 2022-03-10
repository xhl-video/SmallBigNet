import argparse
import torchvision

import torch.nn as nn
import numpy as np
from torch.utils import model_zoo

from models.blocks import *
from tools.tools import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class SmallBigNet(nn.Module):
    def __init__(
            self,
            Test,
            block,
            layers,
            imagenet_pre,
            num_classes=1000,
            feat=False,
            t_length=8,
            **kwargs):
        if not isinstance(block, list):
            block = [block] * 4
        self.inplanes = 64
        super(SmallBigNet, self).__init__()

        self.feat = feat
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7),
                               stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(
            block[1],
            128,
            layers[1],
            stride=2,
            t_stride=1,
            t_length=t_length)
        self.layer3 = self._make_layer(
            block[2],
            256,
            layers[2],
            stride=2,
            t_stride=1,
            t_length=t_length)
        self.layer4 = self._make_layer(
            block[3],
            512,
            layers[3],
            stride=2,
            t_stride=1,
            t_length=t_length)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.feat_dim = 512 * block[0].expansion
        self.test = Test
        if imagenet_pre and is_main_process():
            print('using imagenet pretraining weight set the BN as zero')

        if not feat:
            self.fc = nn.Linear(self.feat_dim, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if 'big' in n:
                if isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            t_stride=1,
            t_length=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False),
                nn.BatchNorm3d(
                    planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                t_stride=t_stride,
                downsample=downsample,
                t_length=t_length))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, t_length=t_length))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if not self.test:
            x = x.view(x.size(0), -1)
        if not self.feat:
            x = self.fc(x)
        return x


def use_image_pre_train(model, args):
    if '50' or '23' in args.model_name:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
    elif '101' in args.model_name:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
    new_state_dict = part_state_dict(state_dict, model.state_dict())
    idx = 0
    model_dict = model.state_dict()
    for k, v in new_state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                model_dict[k] = v.cuda()
                idx += 1

    if is_main_process():
        print(len(new_state_dict))
        print(idx)
        print('imagenet pre-trained weight upload already')
    model.load_state_dict(model_dict)

    return model
def res50(args):
    model = SmallBigNet(
        args.test, [Bottleneck3D_000, Bottleneck3D_000, Bottleneck3D_000, Bottleneck3D_000],
        [3, 4, 6, 3], args.imagenet, num_classes=args.num_classes, feat=args.feat)
    #model = torchvision.models.resnet50(pretrained=False)
    if is_main_process():
       # print_model_parm_flops(model, frame=args.t_length)
        #print(model)
        print(sum([np.prod(param.data.shape) for param in model.parameters()]))
    if args.imagenet:
        model = use_image_pre_train(model, args)
    return model

def slowonly50(args):
    model = SmallBigNet(
        args.test, [Bottleneck3D_000, Bottleneck3D_000, Bottleneck3D_100, Bottleneck3D_100],
        [3, 4, 6, 3], args.imagenet, num_classes=args.num_classes, feat=args.feat)
    if is_main_process():
        #print_model_parm_flops(model, frame=args.t_length)
        #print(model)
        print(sum([np.prod(param.data.shape) for param in model.parameters()]))
    if args.imagenet:
        model = use_image_pre_train(model, args)
    return model

def slowonly50_extra(args):
    model = SmallBigNet(
        args.test, [Bottleneck3D_000, Bottleneck3D_000, Bottleneck3D_100_extra, Bottleneck3D_100_extra],
        [3, 4, 6, 3], args.imagenet, num_classes=args.num_classes, feat=args.feat)
    if is_main_process():
        #print_model_parm_flops(model, frame=args.t_length)
        #print(model)
        print(sum([np.prod(param.data.shape) for param in model.parameters()]))
    if args.imagenet:
        model = use_image_pre_train(model, args)
    return model


def smallbig50_no_extra(args):
    model = SmallBigNet(
        args.test, [Bottleneck3D_000, SmallBig_module, SmallBig_module, SmallBig_module],
        [3, 4, 6, 3], args.imagenet, num_classes=args.num_classes, feat=args.feat, t_length=args.t_length)
    if is_main_process():
        #print(model)
        #print_model_parm_flops(model, frame=args.t_length)
        print(sum([np.prod(param.data.shape) for param in model.parameters()]))
    if args.imagenet:
        model = use_image_pre_train(model, args)
    return model


def smallbig23_no_extra(args):
    model = SmallBigNet(
        args.test, [Bottleneck3D_000, SmallBig_module, SmallBig_module, SmallBig_module],
        [1, 2, 3, 1], args.imagenet, num_classes=args.num_classes, feat=args.feat, t_length=args.t_length)
    if is_main_process():
        #print_model_parm_flops(model, frame=args.t_length)
        #print(model)
        print(sum([np.prod(param.data.shape) for param in model.parameters()]))
    if args.imagenet:
        model = use_image_pre_train(model, args)
    return model


def smallbig101_no_extra(args):
    model = SmallBigNet(
        args.test, [Bottleneck3D_000, SmallBig_module, SmallBig_module, SmallBig_module],
        [3, 4, 23, 3],args.imagenet, num_classes=args.num_classes, feat=args.feat, t_length=args.t_length)
    if is_main_process():
        #print_model_parm_flops(model, frame=args.t_length)
       # print(model)
        print(sum([np.prod(param.data.shape) for param in model.parameters()]))
    if args.imagenet:
        model = use_image_pre_train(model, args)
    return model

def smallbig50_extra(args):
    model = SmallBigNet(
        args.test, [Bottleneck3D_000, SmallBig_module_extra, SmallBig_module_extra, SmallBig_module_extra],
        [3, 4, 6, 3], args.imagenet, num_classes=args.num_classes, feat=args.feat, t_length=args.t_length)
    if is_main_process():
        #print(model)
       # print_model_parm_flops(model, frame=args.t_length)
        print(sum([np.prod(param.data.shape) for param in model.parameters()]))
    if args.imagenet:
        model = use_image_pre_train(model, args)
    return model

def get_model(args):
    model_name_dict = {
        'res50': res50(args),
        'slowonly50' : slowonly50(args),
        'slowonly50_extra': slowonly50_extra(args),
        'smallbig23_no_extra' : smallbig23_no_extra(args),
        'smallbig50_no_extra' : smallbig50_no_extra(args),
        'smallbig101_no_extra': smallbig101_no_extra(args),
        'smallbig50_extra': smallbig50_extra(args),
    }

    return model_name_dict[args.model_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SmallBig Training')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--feat', action='store_true')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--num_classes', default=400, type=int,
                        help="num classes of your dataset")
    parser.add_argument('--t_length', default=8, type=int,
                        help="Total length of sampling frames.")
    parser.add_argument('--model_name', default="smallbig50_no_extra",
                        choices=[
                            "slowonly50",
                            "smallbig23_no_extra",
                            "smallbig50_no_extra",
                            "smallbig101_no_extra"],
                        help="name of your model")
    args = parser.parse_args()
    net = get_model(args)
