import cv2
import numpy as np

import torch


class Compose(object):
    """Composes several video_transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> video_transforms.Compose([
        >>>     video_transforms.CenterCrop(10),
        >>>     video_transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, aug_seed=0):
        self.transforms = transforms

        for i, t in enumerate(self.transforms):
            t.set_random_state(seed=(aug_seed + i))

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Transform(object):
    """basse class for all transformation"""

    def set_random_state(self, seed=None):
        self.rng = np.random.RandomState(seed)


####################################
# Customized Transformations
####################################

class Normalize(Transform):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Resize(Transform):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size  # [w, h]
        self.interpolation = interpolation

    def __call__(self, data):
        h, w, c = data.shape

        if isinstance(self.size, int):
            slen = self.size
            if min(w, h) == slen:
                return data
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]

        if (h != new_h) or (w != new_w):
            scaled_data = cv2.resize(data, (new_w, new_h), self.interpolation)
        else:
            scaled_data = data

        return scaled_data


class RandomScale_nonlocal(Transform):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self,

                 slen=[224, 288],
                 interpolation=cv2.INTER_LINEAR):

        self.slen = slen  # [min factor, max factor]

        self.interpolation = interpolation
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        random_slen = self.rng.uniform(self.slen[0], self.slen[1])
        resize = Resize(int(random_slen))
        scaled_data = resize(data)
        return scaled_data


class RandomScale(Transform):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, make_square=False,
                 aspect_ratio=[1.0, 1.0],
                 slen=[224, 288],
                 interpolation=cv2.INTER_LINEAR):
        # assert slen[1] >= slen[0], \
        #        "slen ({}) should be in increase order".format(scale)
        # assert aspect_ratio[1] >= aspect_ratio[0], \
        #        "aspect_ratio ({}) should be in increase order".format(aspect_ratio)
        self.slen = slen  # [min factor, max factor]
        self.aspect_ratio = aspect_ratio
        self.make_square = make_square
        self.interpolation = interpolation
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        new_w = w
        new_h = h if not self.make_square else w
        if self.aspect_ratio:
            random_aspect_ratio = self.rng.uniform(
                self.aspect_ratio[0], self.aspect_ratio[1])
            if self.rng.rand() > 0.5:
                random_aspect_ratio = 1.0 / random_aspect_ratio
            new_w *= random_aspect_ratio
            new_h /= random_aspect_ratio
        resize_factor = self.rng.uniform(
            self.slen[0], self.slen[1]) / min(new_w, new_h)
        new_w *= resize_factor
        new_h *= resize_factor
        scaled_data = cv2.resize(
            data, (int(new_w + 1), int(new_h + 1)), self.interpolation)
        return scaled_data


class CornerCrop1(Transform):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = int(round((w - tw)) / 4)
        y1 = int(round((h - th)) / 4)
        x1 = 0
        y1 = 0
        # if x1==0 and y1!=0:
        #   y1=int((y1*3)/4))
        # if
        cropped_data = data[y1:(y1 + th), x1:(x1 + tw), :]
        return cropped_data


class CornerCrop2(Transform):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = int(round((w - tw)))
        y1 = int(round((h - th)))
        #x1=int(round((w - tw)))
        #y1=int(round((w - tw)))
        cropped_data = data[y1:(y1 + th), x1:(x1 + tw), :]
        return cropped_data


class CenterCrop(Transform):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        cropped_data = data[y1:(y1 + th), x1:(x1 + tw), :]
        return cropped_data


class GroupCrop(Transform):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data, crop_time=3):
        h, w, c = data.shape
        th, tw = self.size
        img = []

        x1 = [np.random.randint(0, w - tw)for i in range(crop_time)]
        y1 = [0 for i in range(crop_time)]

        for i in range(crop_time):

            cropped_data = data[y1[i]:(y1[i] + th), x1[i]:(x1[i] + tw), :]
            img.append(cropped_data)
        return np.concatenate(img, axis=2)


class Crop(Transform):
    """Crops the given numpy array at the random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, crop):

        self.size = size
        self.crop = crop
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        tw = self.size[0]
        th = self.size[1]
        x1 = self.crop[1]
        y1 = self.crop[0]
        cropped_data = data[y1:(y1 + th), x1:(x1 + tw), :]
        return cropped_data


class RandomCrop(Transform):
    """Crops the given numpy array at the random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):

        self.size = size
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        tw = self.size[0]
        th = self.size[1]
        # p=w-tw
        # q=h-th
        # if p!=0:
        #    x1 = self.rng.choice(range(w - tw))
        #    y1 = 0
        # elif q!=0:
        #    x1 = 0
        #    y1 = self.rng.choice(range(h - th))
        # elif p==0 and q==0:
        #    x1=0
        #    y1=0
        if tw < w and th < h:
            x1 = self.rng.choice(range(w - tw))
            y1 = self.rng.choice(range(h - th))
         #  cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        # else:

        #    resize=Resize([th,tw])
        #    cropped_data = resize(data)
        cropped_data = data[y1:(y1 + th), x1:(x1 + tw), :]

        return cropped_data


class RandomHorizontalFlip(Transform):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """

    def __init__(self):
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        if self.rng.rand() < 0.5:
            data = np.fliplr(data)
            data = np.ascontiguousarray(data)
        return data


class RandomVerticalFlip(Transform):
    """Randomly vertically flips the given numpy array with a probability of 0.5
    """

    def __init__(self):
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        if self.rng.rand() < 0.5:
            data = np.flipud(data)
            data = np.ascontiguousarray(data)
        return data


class RandomRGB(Transform):
    def __init__(self, vars=[10, 10, 10]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape

        random_vars = [int(round(self.rng.uniform(-x, x))) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape)
        for ic in range(0, c):
            var = random_vars[ic % base]
            augmented_data[:, :, ic] = np.minimum(
                np.maximum(data[:, :, ic] + var, 0), 255)
        return augmented_data


class RandomHLS(Transform):
    def __init__(self, vars=[15, 35, 25]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        assert c % 3 == 0, "input channel = %d, illegal" % c

        random_vars = [int(round(self.rng.uniform(-x, x))) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape, )

        for i_im in range(0, int(c / 3)):
            augmented_data[:,
                           :,
                           3 * i_im:(3 * i_im + 3)] = cv2.cvtColor(data[:,
                                                                        :,
                                                                        3 * i_im:(3 * i_im + 3)],
                                                                   cv2.COLOR_RGB2HLS)

        hls_limits = [180, 255, 255]
        for ic in range(0, c):
            var = random_vars[ic % base]
            limit = hls_limits[ic % base]
            augmented_data[:, :, ic] = np.minimum(
                np.maximum(augmented_data[:, :, ic] + var, 0), limit)

        for i_im in range(0, int(c / 3)):
            augmented_data[:, :, 3 *
                           i_im:(3 *
                                 i_im +
                                 3)] = cv2.cvtColor(augmented_data[:, :, 3 *
                                                                   i_im:(3 *
                                                                         i_im +
                                                                         3)].astype(np.uint8), cv2.COLOR_HLS2RGB)

        return augmented_data


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # H, W, C = image.shape
            # handle numpy array
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            # backward compatibility
            return image.float() / 255.0


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x (T x C)) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            H, W, _ = clips.shape
            # handle numpy array
            clips = torch.from_numpy(clips.reshape(
                (H, W, -1, self.dim)).transpose((3, 2, 0, 1)))
            # backward compatibility
            return clips.float() / 255.0
class To_3DTensor(Transform):

    def __init__(self, dim=2):
        self.dim = 2

    def __call__(self, images):
        if isinstance(images, torch.Tensor):
            images = images.unsqueeze(1)
            # backward compatibility
            return images
class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = np.random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise).
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = np.random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)

class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = np.random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)

class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        np.random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)