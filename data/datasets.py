from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data
NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95


def _transforms(self, scale, H, W):
    """
    Up(down)scale and randomly crop to `crop_size` x `crop_size`
    """
    transforms_list = [  # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
        transforms.RandomCrop(self.crop_size),
        transforms.ToTensor()]

    if self.normalize is True:
        transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transforms_list)


class OpenImages(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        # print(data_dir)
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX
        self.normalize = config.norm

    def _transforms(self, scale, H, W):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
            transforms.RandomCrop((self.im_height, self.im_width)),
            # transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # filesize = os.path.getsize(img_path)

        # This is faster but less convenient
        # H X W X C `ndarray`
        # img = imread(img_path)
        # img_dims = img.shape
        # H, W = img_dims[0], img_dims[1]
        # PIL
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')
        except:
            img_path = self.imgs[idx + 1]
            img = Image.open(img_path)
            img = img.convert('RGB')
            print("ERROR!")
        W, H = img.size
        # bpp = filesize * 8. / (H * W)

        shortest_side_length = min(H, W)

        minimum_scale_factor = float(self.crop_size) / float(shortest_side_length)
        scale_low = max(minimum_scale_factor, self.scale_min)
        scale_high = max(scale_low, self.scale_max)
        scale = np.random.uniform(scale_low, scale_high)

        dynamic_transform = self._transforms(scale, H, W)
        transformed = dynamic_transform(img)
        # apply random scaling + crop, put each pixel
        # in [-1.,1.] and reshape to (C x H x W)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, config):
        self.data_dir = config.test_data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()
        _, self.im_height, self.im_width = config.image_dims
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(0),
            # transforms.CenterCrop((self.im_height, self.im_width)),
            transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        img = self.transform(image)
        # if img.shape[1] > img.shape[2]:
        #     img = img.transpose(2, 1)
        return img

    def __len__(self):
        return len(self.imgs)


def get_loader(config):
    if config.trainset == 'OpenImages':
        train_dataset = OpenImages(config, config.train_data_dir)
    else:
        train_dataset = Datasets(config)

    test_dataset = Datasets(config)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader


def get_test_loader(config):
    test_dataset = Datasets(config)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    return test_loader


if __name__ == '__main__':
    import sys

    sys.path.append("/media/D/wangsixian/DJSCC")
    from NTSCC.config import config_v3 as config

    config.train_data_dir = ['/home/wangsixian/Dataset/openimages/**']
    train_loader, test_loader = get_cifar10_loader(config)
    print(train_loader.__len__())
