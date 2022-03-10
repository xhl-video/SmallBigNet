import argparse

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


class Kinetics(data.Dataset):
    def __init__(self,
                 root_path,
                 list_file,
                 t_length=8,
                 t_stride=8,
                 num_clips=1,
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
        self.num_clips = num_clips
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.video_frame = {}
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
                for j in range(self.crop_num):

                    for i in range(self.num_clips):
                        data = x.strip().split(' ')[0]
                        name = data.split('/')[-1].split('.')[0][0:11]
                        path = self.root_path
                        if os.path.exists(os.path.join(path, name)):
                            self.video_list.append(VideoRecord([name, x.strip().split(' ')[1], int(x.strip(
                            ).split(' ')[2])], self.root_path, phase='Val', copy_id=idx, crop=j, vid=vid))
                            idx += 1
                vid += 1

        elif self.phase == 'Val':
            for x in open(self.list_file):
                data = x.strip().split(' ')[0]
                name = data.split('/')[-1].split('.')[0][0:11]
               # name = os.path.join('val', name[0:11])
                path = self.root_path
                if os.path.exists(os.path.join(path, name)):
                    self.video_list.append(VideoRecord([name, x.strip().split(' ')[1], int(
                        x.strip().split(' ')[2])], self.root_path, phase='Val', ))

        elif self.phase == 'Train':
            for x in open(self.list_file):
                idx = 0
                for i in range(self.num_clips):
                    data = x.strip().split(' ')[0]
                    name = data.split('/')[-1].split('.')[0][0:11]
                    #name = os.path.join('train', name[0:11])
                    path = self.root_path
                   # print(name)
                    if os.path.exists(os.path.join(path, name)):
                        self.video_list.append(VideoRecord([name, x.strip().split(' ')[1], int(
                            x.strip().split(' ')[2])], self.root_path, phase='Train'))
                    idx += 1
            self.rng.shuffle(self.video_list)

    @staticmethod
    def dense_sampler(num_frames, length, stride=1):
        t_length = length
        t_stride = stride
        offset = 0
        average_duration = num_frames - (t_length - 1) * t_stride - 1
        if average_duration >= 0:
            offset = randint(average_duration + 1)
        elif num_frames > t_length:
            while (t_stride - 1 > 0):
                t_stride -= 1
                average_duration = num_frames - (t_length - 1) * t_stride - 1
                if average_duration >= 0:
                    offset = randint(average_duration + 1)
                    break
            assert (t_stride >= 1), "temporal stride must be bigger than zero."
        else:
            t_stride = 1
        # sampling
        samples = []
        for i in range(t_length):
            samples.append(offset + i * t_stride + 1)
        return samples

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.style == "Dense":
            frames = []
            average_duration = record.num_frames / self.num_clips
            offsets = [average_duration * i for i in range(self.num_clips)]
            for i in range(self.num_clips):
                samples = self.dense_sampler(
                    average_duration, self.t_length, self.t_stride)
                samples = [int(sample + offsets[i]) for sample in samples]
                frames.append(samples)
            return {"dense": frames[0]}

    def _get_val_indices(self, record):
        """
        get indices in val phase
        """
        valid_offset_range = record.num_frames - \
            (self.t_length - 1) * self.t_stride
        if valid_offset_range > 0:
            offset = randint(valid_offset_range)
        else:
            offset = valid_offset_range
        if offset < 0:
            offset = 0
        samples = []
        for i in range(self.t_length):
            samples.append(offset + i * self.t_stride + 1)
        return {"dense": samples}

    def _get_test_index(self, record):
        sample_op = max(1, record.num_frames - self.t_length * self.t_stride)
        t_stride = self.t_stride
        start_list = np.linspace(
            0, sample_op - 1, num=self.num_clips, dtype=int)
        offsets = []
        for start in start_list.tolist():
            offsets.append([(idx * t_stride + start) %
                            record.num_frames for idx in range(self.t_length)])
        frame = np.array(offsets) + 1
        return {"dense": frame.tolist()}

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

            idx = record.copy_id % self.num_clips

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

                        transforms.CornerCrop2((256, 256)),

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

            if self.phase == 'Train':
                return self.transform(images)
            if self.phase == 'Val':
                return self.transform(images)

        if phase == "Train":
            if self.style == "Dense":
                process_data = dense_process_data(index)
        elif phase in ("Val", "Test"):
            process_data = dense_process_data(index)
            return process_data, record.label  # , indices
        else:
            process_data = dense_process_data(index)
            return process_data, record.label  # ,record.vid

        return process_data, record.label  # , indices

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
        '--list_file',
        default='/dataset/kinetics/val.txt',
        type=str,
        help='path for your data list(txt)')

    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        GroupMultiScaleCrop(input_size=224, scales=[1, .875, .75, .66]),
        GroupRandomHorizontalFlip(),
        Stack(mode='3D'),
        ToTorchFormatTensor(),
        GroupNormalize(),
    ])
    dataset = Kinetics(
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
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    for ind, (data, label) in enumerate(loader):
        label = label.cuda(non_blocking=True)

