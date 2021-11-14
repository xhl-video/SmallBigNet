import os.path
import logging
import math
import shutil
import time
from collections import defaultdict

import torch
import os
import random
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        self.avg = self.sum / (self.count + 1e-5)


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def save_checkpoint(state, is_best, epoch, experiment_root,
                    filename='checkpoint_{}epoch.pth'):
    filename = os.path.join(experiment_root, filename.format(epoch))
    logging.info("saving model to {}...".format(filename))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(experiment_root, 'model_best.pth')
        shutil.copyfile(filename, best_name)
    logging.info("saving done.")


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.contiguous().t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(
        state,
        is_best,
        epoch,
        experiment_root,
        args = None,
        name='name',
        filename='checkpoint_{}epoch.pth'):
    checkdir = os.path.join(experiment_root, args.model_name)
    if not os.path.exists(checkdir):
        os.makedirs(checkdir)
    file = args.model_name + '_' + args.dataset + '_t_length_' + str(args.t_length) + '_t_stride_' + str(
        args.t_stride) + '_batch_' + str(
        args.batch_size) + '_lr_' + str(args.lr) + time.strftime("%d_%b_%Y_%H:%M:%S",  time.localtime())
    file_dir = os.path.join(checkdir, file)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)

    file = os.path.join(file_dir, filename.format(epoch))

    torch.save(state, file)
    if is_best:
        best_name = os.path.join(
            file_dir,
            'model_best_' + name + '.pth')
        shutil.copyfile(file, best_name)


def adjust_learning_rate1(optimizer, base_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    alpha = (epoch + 2000) / 2000
    warm = (1. / 10) * (1 - alpha) + alpha
    lr = base_lr * warm
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, base_lr, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = base_lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def convert_arr2matrix(actions):
    node = []
    output = []
    intermit = []
    operator = []
    graph = np.zeros((5, 5))
    node.append([actions[0]])

    operator.append([actions[1]])
    idx = 2
    for i in range(1, 4):
        node.append([actions[idx * i + j] for j in range(i + 1)])

        operator.append([actions[idx * (i + 1) + j] for j in range(i + 1)])
        idx += 1

    for j in range(len(node)):
        ip = True
        for p in node[j + 1:]:
            if j + 2 in p:
                ip = False

        if ip:
            output.append(j + 2)
        else:
            intermit.append(j + 2)
    idy = 1
    for i in range(len(node)):
        for j in range(len(node[i])):

            if node[i][j] != 0:

                if i + 2 in intermit:
                    graph[node[i][j] - 1][idy] = node[i][j] + 1
                    graph[idy][node[i][j] - 1] = node[i][j] + 1
                else:
                    graph[node[i][j] - 1][idy] = node[i][j]
                    graph[idy][node[i][j] - 1] = node[i][j]
        idy += 1
    return graph


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def part_state_dict(state_dict, model_dict):
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print(k)
    pretrained_dict = inflate_state_dict(pretrained_dict, model_dict)
   # model_dict.update(pretrained_dict)
    return pretrained_dict


def adjust_learning_rate2(optimizer, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def inflate_state_dict(pretrained_dict, model_dict):
    for k in pretrained_dict.keys():
        if k in model_dict.keys():
            if pretrained_dict[k].size() != model_dict[k].size():
                assert(
                    pretrained_dict[k].size()[
                        :2] == model_dict[k].size()[
                        :2]), "To inflate, channel number should match."
                assert(pretrained_dict[k].size()[-2:] == model_dict[k].size()
                       [-2:]), "To inflate, spatial kernel size should match."
                #print("Layer {} needs inflation.".format(k))
                shape = list(pretrained_dict[k].shape)
                shape.insert(2, 1)
                t_length = model_dict[k].shape[2]
                pretrained_dict[k] = pretrained_dict[k].reshape(shape)
                if t_length != 1:
                    pretrained_dict[k] = pretrained_dict[k].expand_as(
                        model_dict[k]) / t_length
                assert(pretrained_dict[k].size() == model_dict[k].size()), \
                    "After inflation, model shape should match."

    return pretrained_dict


def print_model_parm_flops(model, frame=8):

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
                # print 'flops:{}'.format(self.__class__.__name__)
                # print 'input:{}'.format(input)
                # print '_dim:{}'.format(input[0].dim())
                # print 'input_shape:{}'.format(np.prod(input[0].shape))
                # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, time_stride, input_height, input_width = input[0].size(
        )
        output_channels, out_time, output_height, output_width = output[0].size(
        )

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (
            self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width * out_time

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, time_stride, input_height, input_width = input[0].size(
        )
        output_channels, out_time, output_height, output_width = output[0].size(
        )

        kernel_ops = self.kernel_size[0] * \
            self.kernel_size[1] * self.kernel_size[2]
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width * out_time

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv3d):
                    # net.register_forward_hook(save_hook(net.__class__.__name__))
                    # net.register_forward_hook(simple_hook)
                    # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(
                    net, torch.nn.MaxPool3d) or isinstance(
                    net, torch.nn.AvgPool3d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    criterion = nn.CrossEntropyLoss().cuda()

    model = model

    foo(model)
    input = Variable(torch.rand(1, 3, frame, 224, 224), requires_grad=True)
    out = model(input)

    total_flops = (
        sum(list_conv) +
        sum(list_linear) +
        sum(list_bn) +
        sum(list_relu) +
        sum(list_pooling))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
