import argparse
import logging
import os
import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from dataset.prepare_dataset import get_dataloader
from models.SmallBig import get_model

from tools.tools import save_checkpoint, is_main_process

from train_val import train, validate

parser = argparse.ArgumentParser(description='SmallBig Training')
parser.add_argument('--batch_size', default=1, type=int,
                    help="Total batch size for training.")
parser.add_argument('--t_length', default=8, type=int,
                    help="Total length of sampling frames.")
parser.add_argument('--t_stride', default=8, type=int,
                    help="Temporal stride between each frame.")
parser.add_argument('--num_clips', default=1, type=int,
                    help="Total number of clips for training or testing.")
parser.add_argument(
    '--crop_num',
    default=1,
    type=int,
    help="Total number of crops for each frame during full-resolution testing.")
parser.add_argument('--image_tmpl', default='image_{:06d}.jpg', type=str,
                    help="The name format of each frames you saved.")
parser.add_argument('--seed', default=0, type=int,
                    help="Random Seed")
parser.add_argument(
    '--dataset',
    default='kinetics',
    choices=[
        "kinetics",
        "imagenet",
        "something"],
    help="Choose dataset for training and validation")
parser.add_argument(
    '--phase',
    default='Val',
    choices=[
        "Train",
        "Val",
        "Fntest"],
    help="Different phases have different sampling methods.")
parser.add_argument('--root_path', default='/dataset/kinetics', type=str,
                    help='root path for accessing your image data')
parser.add_argument('--val_list_file', default='/dataset/kinetics/val.txt',
                    type=str, help='path for your data list(txt)')
parser.add_argument(
    '--train_list_file',
    default='/dataset/kinetics/train.txt',
    type=str,
    help='path for your data list(txt)')
parser.add_argument('--model_name', default="smallbig50_no_extra",
                    choices=[
                        "res50",
                        "slowonly50",
                        "slowonly50_extra",
                        "smallbig23_no_extra",
                        "smallbig50_no_extra",
                        "smallbig101_no_extra",
                        "smallbig50_extra"],
                    help="name of your model")
parser.add_argument('--local_rank', type=int, default=0,
                    help='node rank for distributed training')
parser.add_argument('--distribute', action='store_true')
parser.add_argument('--print_freq', type=int, default=50,
                    help='print frequency')
parser.add_argument('--test', action='store_true')
parser.add_argument('--feat', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--imagenet', action='store_false')
parser.add_argument('--num_classes', default=400, type=int,
                    help="num classes of your dataset")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--resume', type=str,
                    help="Checkpoint path that you want to restart training.")
parser.add_argument('--check_dir', type=str,
                    help="Location to store your model")
parser.add_argument('--log_dir', type=str,
                    help="Location to store your logs")


def set_logger(args):
    import time
    logdir = os.path.join(args.log_dir, args.model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    log_file = args.model_name + '_' + args.dataset + '_t_length_' + str(args.t_length) + '_t_stride_' + str(args.t_stride) + '_batch_' + str(
        args.batch_size) + '_lr_' + str(args.lr) + "_logfile_" + time.strftime("%d_%b_%Y_%H:%M:%S", time.localtime())
    log_file = os.path.join(logdir, log_file)
    if not os.path.exists(log_file):
        os.makedirs(log_file, exist_ok=True)
    log_file = os.path.join(log_file, "logfile_" + time.strftime("%d_%b_%Y_%H:%M:%S",
                                                                                   time.localtime()))
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]

    """ add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
    logging.basicConfig(level= logging.INFO,
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


def train_model(args):
    global best_metric, epoch_resume
    epoch_resume = 0
    best_metric = 0
    model = get_model(args)

    if args.distribute:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model).cuda()
    writer = None
    if is_main_process():
        log_file = args.model_name + '_' + args.dataset + '_t_length_' + str(args.t_length) + '_t_stride_' + str(
            args.t_stride) + '_batch_' + str(
            args.batch_size) + '_lr_' + str(args.lr) + "_logfile_" + time.strftime("%d_%b_%Y_%H:%M:%S",
                                                                                   time.localtime())
        log_file = os.path.join(args.log_dir, args.model_name, log_file)
        writer = SummaryWriter(log_dir=log_file)
        print(model)
    dataloaders, dataset_sizes, samplers = get_dataloader(args)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        momentum=0.9)
    criterion = nn.CrossEntropyLoss().cuda()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        epoch_resume = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
        model_dict = model.state_dict()
        idx = 0
        print(len(model_dict))
        print(len(checkpoint['state_dict']))
        for k, v in checkpoint['state_dict'].items():
            k = k.replace('module.', '')
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v.cuda()
                    idx += 1
        print(idx)
        print('upload parameter already')
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(("=> loaded checkpoint '{}' (epoch {})"
               .format(args.resume, checkpoint['epoch'])))
        print(best_metric)
    elif is_main_process():
        print(("=> no checkpoint found at '{}'".format(args.resume)))

    for epoch in range(epoch_resume, args.num_epochs):
        if args.distribute:
            samplers['train'].set_epoch(epoch)
            samplers['val'].set_epoch(epoch)
        end = time.time()
        train(dataloaders['train'], model, criterion, optimizer,
                                    epoch, args.print_freq, writer, args=args)
        scheduler.step()
        if epoch >= 0:
            metric = validate(
                dataloaders['val'],
                model,
                criterion,
                args.print_freq,
                epoch + 1,
                writer,
                args=args)
            if is_main_process():
                print(metric)
        #       remember best prec@1 and save checkpoint
                is_best = metric > best_metric
                best_metric = max(metric, best_metric)
                print(best_metric)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_metric': best_metric,
                    'optimizer': optimizer.state_dict(),
                }, is_best,
                   str('current'),
                    args.check_dir,
                    args = args,
                    name=args.model_name)

        time_elapsed = time.time() - end
        if is_main_process():
            print(
                f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")


if __name__ == '__main__':
    args = parser.parse_args()
    if is_main_process():
        set_logger(args)
        logging.info(args)
    if args.distribute:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://'
        )
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    train_model(args)
