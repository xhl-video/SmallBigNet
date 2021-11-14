import argparse

import cv2
import torch.utils.data as data

import os
import os.path
import numpy as np
import torchvision
from numpy.random import randint

import torch

from dataset import transforms
from dataset.transform import *


class VideoRecord(object):
    def __init__(
            self,
            row,
            root_path,
            phase='Train',
            copy_id=0,
            crop=0,
            vid=0):
        self._data = row
        self.crop_pos = crop

        self.phase = phase
        self.copy_id = copy_id
        self.vid = vid
        self._root_path = root_path

    @property
    def path(self):
        if self.phase == 'Train':
            return os.path.join(
                self._root_path, self._data[0].replace(
                    'RGB_train/', ''))
        else:
            return os.path.join(
                self._root_path,
                self._data[0].replace(
                    'RGB_val/',
                    ''))

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class Someting_something(data.Dataset):
    def __init__(self,
                 root_path,
                 list_file,
                 t_length=32,
                 t_stride=2,
                 num_clips=10,
                 image_tmpl='img_{:05d}.jpg',
                 transform=None,
                 crop_num=1,
                 style="Dense",
                 phase="Train",
                 seed=1):
        """
        :style: Dense, for 2D and 3D model, and Sparse for TSN model
        :phase: Train, Val, Test
        """

        self.root_path = root_path
        self.list_file = list_file
        self.crop_num = crop_num
        self.t_length = t_length
        self.t_stride = t_stride
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.n_times = num_clips
        self.rng = np.random.RandomState(seed)
        assert(style in ("Dense", "UnevenDense")
               ), "Only support Dense and UnevenDense"
        self.style = style
        self.phase = phase

        assert(t_length > 0), "Length of time must be bigger than zero."
        assert(t_stride > 0), "Stride of time must be bigger than zero."

        self._parse_list()

    def _load_image(self, directory, idx):
        from PIL import Image
        if os.path.exists(
            os.path.join(
                directory,
                self.image_tmpl.format(idx))):

            cv_img = Image.open(
                os.path.join(
                    directory,
                    self.image_tmpl.format(idx))).convert('RGB')
        else:
            print(
                'no frames at ',
                os.path.join(
                    directory,
                    self.image_tmpl.format(idx)))
            while True:
                idx += 1
                if os.path.exists(
                    os.path.join(
                        directory,
                        self.image_tmpl.format(idx))):
                    cv_img = Image.open(
                        os.path.join(
                            directory,
                            self.image_tmpl.format(idx))).convert('RGB')
                    break
        return [cv_img]

    def _parse_list(self):
        self.video_list = []

        if self.phase == 'Fntest':
            vid = 0
            for x in open(self.list_file):
                idx = 0
                for i in range(self.n_times):
                    for j in range(self.crop_num):
                        data = x.strip().split(' ')[0]
                        name = data.split('/')[-1].split('.')[0]
                        path = self.root_path
                        if os.path.exists(os.path.join(path, name)):
                            self.video_list.append(VideoRecord([name, x.split(' ')[1], x.split(
                                ' ')[2]], self.root_path, phase='Val', copy_id=i, crop=j, vid=vid))
                            idx += 1
                vid += 1

        elif self.phase == 'Val':
            for x in open(self.list_file):
                data = x.strip().split(' ')[0]
                name = data.split('/')[-1].split('.')[0]
                path = self.root_path

                if os.path.exists(os.path.join(path, name)):
                    self.video_list.append(VideoRecord(
                        [name, x.split(' ')[1], x.split(' ')[2]], self.root_path, ))

        else:
            for x in open(self.list_file):
                data = x.strip().split(' ')[0]
                name = data.split('/')[-1].split('.')[0]
                path = self.root_path
                if os.path.exists(os.path.join(path, name)):
                    self.video_list.append(VideoRecord(
                        [name, x.split(' ')[1], x.split(' ')[2]], self.root_path, ))
            self.rng.shuffle(self.video_list)

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.style == "Dense":

            average_duration = record.num_frames // self.t_length

            if average_duration > 0:

                offsets = np.multiply(list(range(self.t_length)),
                                      average_duration) + randint(average_duration,
                                                                  size=self.t_length)
            elif record.num_frames > self.t_length:

                offsets = np.sort(
                    randint(
                        record.num_frames,
                        size=self.t_length))

            else:

                offsets = np.zeros((self.t_length,))

            return {"dense": offsets + 1}

    def _get_val_indices(self, record):
        """
        get indices in val phase
        """
        if record.num_frames > self.t_length - 1:

            tick = (record.num_frames - 1) / float(self.t_length)

            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.t_length)])

        else:

            offsets = np.zeros((self.t_length,))

        return {"dense": offsets + 1}

    def _get_test_indices(self, record):

        tick = (record.num_frames) / float(self.t_length)

        offsets = [[int(tick / 2.0 + tick * x) + 1for x in range(self.t_length)],
                   [int(tick * x) + 1for x in range(self.t_length)]]
        return {"dense": offsets}

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.phase == "Train":
            indices = self._sample_indices(record)

            return self.get(record, indices, self.phase, index)
        elif self.phase == "Val":
            indices = self._get_val_indices(record)

            return self.get(record, indices, self.phase, index)
        elif self.phase == "Fntest":

            indices = self._get_test_indices(record)
            idx = record.copy_id % self.n_times
            indices['dense'] = indices['dense'][idx]

            return self.get(record, indices, self.phase, index)
        else:
            raise TypeError("Unsuported phase {}".format(self.phase))

    def get(self, record, indices, phase, index):
        # dense process data
        def dense_process_data(index):
            images = list()
            for ind in indices['dense']:
                ptr = int(ind)

                if ptr <= record.num_frames:
                    imgs = self._load_image(record.path, ptr)
                else:
                    imgs = self._load_image(record.path, record.num_frames)
                images.extend(imgs)

            if self.phase == 'Fntest':

                images = [np.asarray(im) for im in images]
                clip_input = np.concatenate(images, axis=2)

                self.t = transforms.Compose([
                    transforms.Resize(256)])
                clip_input = self.t(clip_input)

                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

                if record.crop_pos == 0:
                    self.transform = transforms.Compose([

                        transforms.CenterCrop((256, 256)),

                        transforms.ToTensor(),
                        normalize,
                    ])
                elif record.crop_pos == 1:
                    self.transform = transforms.Compose([

                        transforms.CornerCrop2((256, 256),),

                        transforms.ToTensor(),
                        normalize,
                    ])
                elif record.crop_pos == 2:
                    self.transform = transforms.Compose([
                        transforms.CornerCrop1((256, 256)),
                        transforms.ToTensor(),
                        normalize,
                    ])

                return self.transform(clip_input)

            return self.transform(images)

        if phase == "Train":
            if self.style == "Dense":
                process_data = dense_process_data(index)

        elif phase in ("Val", "Test"):
            process_data = dense_process_data(index)
            return process_data, record.label, indices
        else:
            process_data = dense_process_data(index)
            return process_data, record.label, indices

        return process_data, record.label, indices

    def __len__(self):
        return len(self.video_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SmallBig Training')
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--t_length', default=8, type=int,
                        help="Total length of sampling frames.")
    parser.add_argument('--t_stride', default=8, type=int,
                        help="Temporal stride between each frame.")
    parser.add_argument('--num_clips', default=2, type=int,
                        help="Total number of clips for training or testing.")
    parser.add_argument(
        '--crop_num',
        default=1,
        type=int,
        help="Total number of crops for each frame during full-resolution testing.")
    parser.add_argument('--image_tmpl', default='{:05d}.jpg', type=str,
                        help="The name format of each frames you saved.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random Seed")
    parser.add_argument(
        '--phase',
        default='Fntest',
        choices=[
            "Train",
            "Val",
            "Fntest"],
        help="Different phases have different sampling methods.")
    parser.add_argument('--root_path', default='/dataset/sthv1', type=str,
                        help='root path for accessing your image data')
    parser.add_argument(
        '--list_file',
        default='/dataset/sthv1_val.txt',
        type=str,
        help='path for your data list(txt)')

    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        GroupMultiScaleCrop(input_size=224, scales=[1, .875, .75, .66]),
        Stack(mode='3D'),
        ToTorchFormatTensor(),
        GroupNormalize(),
    ])
    dataset = Someting_something(
        root_path=args.root_path,
        list_file=args.list_file,
        t_length=args.t_length,
        t_stride=args.t_stride,
        crop_num=args.crop_num,
        num_clips=args.num_clips,
        image_tmpl=args.image_tmpl,
        transform=transform,
        phase=args.phase,
        seed=args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    for ind, (data, label, indices) in enumerate(loader):
        label = label.cuda(non_blocking=True)
        print(indices)
