import argparse
import os


import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset.kinetics import Kinetics
from dataset.something_something import Someting_something
from dataset.transform import *
from dataset.transforms import Lighting, To_3DTensor
from tools.tools import is_main_process

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

def get_dataloader(args):
    if args.dataset == 'kinetics':
        train_transform = torchvision.transforms.Compose([
            GroupMultiScaleCrop(input_size=224, scales=[1, .875, .75, .66]),
            GroupRandomHorizontalFlip(),
            Stack(mode='3D'),
            ToTorchFormatTensor(),
            GroupNormalize(),
        ])
        train_dataset = Kinetics(
            root_path=args.root_path,
            list_file=args.train_list_file,
            t_length=args.t_length,
            t_stride=args.t_stride,
            crop_num=args.crop_num,
            num_clips=args.num_clips,
            image_tmpl=args.image_tmpl,
            transform=train_transform,
            phase='Train',
            seed=args.seed)
        val_transform = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(mode='3D'),
            ToTorchFormatTensor(),
            GroupNormalize(),
        ])
        val_dataset = Kinetics(
            root_path=args.root_path,
            list_file=args.val_list_file,
            t_length=args.t_length,
            t_stride=args.t_stride,
            crop_num=args.crop_num,
            num_clips=args.num_clips,
            image_tmpl=args.image_tmpl,
            transform=val_transform,
            phase='Val',
            seed=args.seed)
    elif args.dataset == 'something':
        train_transform = torchvision.transforms.Compose([
            GroupMultiScaleCrop(input_size=224, scales=[1, .875, .75, .66]),
            Stack(mode='3D'),
            ToTorchFormatTensor(),
            GroupNormalize(),
        ])
        train_dataset = Someting_something(
            root_path=args.root_path,
            list_file=args.train_list_file,
            t_length=args.t_length,
            t_stride=args.t_stride,
            crop_num=args.crop_num,
            num_clips=args.num_clips,
            image_tmpl=args.image_tmpl,
            transform=train_transform,
            phase='Train',
            seed=args.seed)
        val_transform = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(mode='3D'),
            ToTorchFormatTensor(),
            GroupNormalize(),
        ])
        val_dataset = Someting_something(
            root_path=args.root_path,
            list_file=args.val_list_file,
            t_length=args.t_length,
            t_stride=args.t_stride,
            crop_num=args.crop_num,
            num_clips=args.num_clips,
            image_tmpl=args.image_tmpl,
            transform=val_transform,
            phase='Val',
            seed=args.seed)
    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.root_path, 'train')
        valdir = os.path.join(args.root_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(.4, .4, .4),
                transforms.ToTensor(),
                Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
                normalize,
                To_3DTensor()
            ]))
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                To_3DTensor()
            ]))

    if args.distribute:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
            num_replicas=torch.distributed.get_world_size(),
            rank=args.local_rank)

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
            num_replicas=torch.distributed.get_world_size(),
            rank=args.local_rank)

        batch_size = args.batch_size // torch.distributed.get_world_size()

    else:
        train_sampler = None
        val_sampler = None
        batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not args.distribute,
        drop_last=True,
        num_workers=8,
        sampler=train_sampler,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    samplers = {'train': train_sampler, 'val': val_sampler}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    if is_main_process():
        print(args.dataset, 'has the size: ', dataset_sizes)
    return dataloaders, dataset_sizes, samplers


if __name__ == "__main__":
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
    parser.add_argument(
        '--val_list_file',
        default='/dataset/kinetics/val.txt',
        type=str,
        help='path for your data list(txt)')
    parser.add_argument(
        '--train_list_file',
        default='/dataset/kinetics/train.txt',
        type=str,
        help='path for your data list(txt)')

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='node rank for distributed training')
    parser.add_argument('--distribute', action='store_true')

    args = parser.parse_args()

    dataloaders, dataset_sizes, samplers = get_dataloader(args)
    for k, v in dataset_sizes.items():
        print(k, v)
