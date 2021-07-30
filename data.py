import os
import math
import random
import h5py
from PIL import Image
from pathlib import Path

import torch
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import imgaug.augmenters as iaa
from sklearn import preprocessing
from tqdm import tqdm

import util


def encode_label_generic(label, class_names):
    le = preprocessing.LabelEncoder()
    le.fit(class_names)
    return int(le.transform(label))


def decode_label_generic(_class, class_names):
    _class = util.get_detached_tensor(_class, numpy=True).reshape(1)
    le = preprocessing.LabelEncoder()
    le.fit(class_names)
    return str(le.inverse_transform(_class)[0])


def save_class_dataset(dataset: list, filename: str, class_names=None):
    try:
        # assert dataset contains only one label
        assert len(set([ex['label'] for ex in dataset])) == 1
    except AssertionError:
        print('Dataset has more than one class')
    images = np.array([ex['crop'] for ex in dataset])
    if isinstance(dataset[0]['label'], int):
        labels = np.array([ex['label'] for ex in dataset])
    else:
        labels = np.array([encode_label_generic(ex['label'], class_names) for ex in dataset])
    with h5py.File(filename, 'a') as writer:
        writer.create_dataset("images", np.shape(images), dtype='float32', data=images)
        writer.create_dataset('meta', np.shape(labels), dtype='float32', data=labels)
        writer.close()


def shard_dataset(dataset, num_records=5):
    chunk = len(dataset) // num_records
    parts = [(k * chunk) for k in range(len(dataset)) if (k * chunk) < len(dataset)]
    return chunk, parts


def write_class_dataset(dataset, label, data_name, class_names=None, training=True, num_records=5, **kwargs):
    """
    :param dataset: list of dictionaries
    :param label: label - class
    :param data_name: dataset name
    :param class_names: list of strings containing the class names of the full dataset
    :param training: true or false
    :param num_records:
    :param kwargs: pass data path to write data (fix this)
    :return: write datasets by label
    """
    _prefix = 'train' if training is True else 'test'
    train_check = 0
    if len(dataset) > 100:
        chunk, parts = shard_dataset(dataset, num_records)
        for i, j in enumerate(tqdm(parts)):
            shard = dataset[j:(j + chunk)]
            num_samples = len(shard)
            fn = '{}_{}-{}-{}_{:03d}-{:03d}.h5'.format(_prefix, label, data_name, num_samples, i + 1, len(parts))
            save_class_dataset(shard, os.path.join(kwargs.get('data_path'), fn), class_names=class_names)
            train_check += len(shard)
        print('Number of saved samples for {}: {}'.format(label, train_check))
    else:
        num_samples = len(dataset)
        fn = '{}_{}-{}-{}_{:03d}-{:03d}.h5'.format(_prefix, label, data_name, num_samples, 1, 1)
        save_class_dataset(dataset, os.path.join(kwargs.get('data_path'), fn), class_names=class_names)
        print('Small dataset with {} samples'.format(len(dataset)))
    return None


def augmentation_policy(crop):
    crop = np.array(crop)# transform it in case it is a PIL image
    crop = np.fliplr(crop) if random.random() < 0.5 else crop
    af_prob = lambda aug: iaa.Sometimes(0.5, aug)
    cl_prob = lambda aug: iaa.Sometimes(0.5, aug)
    sp_prob = lambda aug: iaa.Sometimes(0.3, aug)
    # blur_prob = lambda aug: iaa.Sometimes(0.2, aug)
    # hue_prob = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential([
        sp_prob(iaa.SaltAndPepper(0.01)),
        # hue_prob(iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5))),
        # cl_prob(iaa.CLAHE(clip_limit=(1, 10))),
        cl_prob(iaa.LinearContrast((0.4, 1.6))), # iaa.GammaContrast((0.5, 2.0))
        # blur_prob(iaa.GaussianBlur(sigma=(0.0, 3.0))),
        af_prob(iaa.Affine(scale=(1.0, 1.1),
                           translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                           rotate=(-17, 17),
                           shear=(-8, 8),
                           mode='edge'))])
    return seq.augment_image(crop)


def standardize_sample(img):
    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    img = img.float()
    mean = img.mean()
    n = img.size(0)
    adjusted_stddev = max(img.std(), 1.0 / math.sqrt(n))
    return (img-mean)/adjusted_stddev


def normalize_sample(img):
    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    img = img.float()
    img /= 255
    return img


def process_mlr():
    transformation = transforms.Compose([transforms.Lambda(lambda x: augmentation_policy(x)),
                                         transforms.Lambda(lambda x: standardize_sample(x))])
    return transformation


def process_adversaries(train=True):
    to_tensor = transforms.ToTensor()
    if train:
        transformations = [transforms.Lambda(lambda x: augmentation_policy(x)), to_tensor]
    else:
        transformations = [to_tensor]
    return transforms.Compose(transformations)


def process_imagenet(train=True):
    if train:
        transformation = transforms.Compose([transforms.Lambda(lambda x: augmentation_policy(x)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    else:
        transformation = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    return transformation


def imagenet_preprocessing(train=True, resize=256, image_size=224):
    hflip = transforms.RandomHorizontalFlip()
    rcrop = transforms.RandomCrop((image_size, image_size))
    ccrop = transforms.CenterCrop((image_size, image_size))
    totensor = transforms.ToTensor()
    cnorm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean and std for imagenet
    if train:
        transform = [transforms.Resize((resize, resize)), hflip, rcrop, totensor, cnorm]
    else:
        transform = [transforms.Resize((image_size, image_size)), ccrop, totensor, cnorm]
    return transforms.Compose(transform)


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def inverse_normalization(img, show=False, title=None, transform=True):
    import matplotlib.pyplot as plt
    """
    Invert normalization RGB images for displaying purposes
    :param img: img
    :param show: show the transformation
    :return: original image
    """
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                         std=[1/0.229, 1/0.224, 1/0.255])
    img = inv_normalize(img)
    npimg = img.numpy()
    if show:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        if title is not None:
            plt.title(title)
        plt.show()
    if transform:
        return np.transpose(npimg, (1, 2, 0))
    else:
        return npimg


def get_stl_datasets(adversarial_crafting=False):
    if adversarial_crafting:
        transformation = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        train_data = torchvision.datasets.STL10(root='./datasets', split='train', transform=transformation)
    else:
        train_data = torchvision.datasets.STL10(root='./datasets', split='train', transform=process_imagenet(True))
    test_data = torchvision.datasets.STL10(root='./datasets', split='test', transform=process_imagenet(False))
    return train_data, test_data, train_data.classes


class HDF5Dataset(Dataset):
    def __init__(self, data_dir, phase, load_data=False, cache_size=10, dataset_by_labels=None,
                 convert_rgb=False, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.cache_size = cache_size
        self.class_datasets = dataset_by_labels
        self.files = self.get_files(data_dir, phase)
        self.transform = transform
        self.rgb = convert_rgb
        for f in self.files:
            self.add_data_info(str(f.resolve()), load_data)

    def __getitem__(self, idx):
        x = self.get_data('images', idx)
        # if len(x.shape) < 3:
        #     x = x[None, ...]  # add channel dimension (gray-scale images)
        x = x[None, ...] if self.rgb and len(x.shape) < 3 else x
        y = self.get_data('meta', idx)
        x = self.transform(x) if self.transform else x
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.get_data_info('images'))

    def get_files(self, data_dir, phase):
        p = Path(data_dir)
        assert p.is_dir()
        try:
            class_datasets = '*' if not self.class_datasets else self.class_datasets
            if class_datasets != '*' and isinstance(class_datasets, list):
                files = []
                for cl in class_datasets:
                    files.extend(p.glob('{}_{}.h5'.format(phase, cl)))
            else:
                files = p.glob('{}_{}.h5'.format(phase, class_datasets))
            files = sorted(files)
            assert len(files) > 1
            return files
        except AssertionError:
            print('No dataset is being loaded')
            return None

    def load_data(self, file_path):
        file = h5py.File(file_path)
        attributes = [k for k in file.keys()]
        for att in attributes:
            data = file[att]
            for example in data:
                idx = self.add_data_cache(example, file_path)
                file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)
                self.data_info[file_idx + idx]['cache_idx'] = idx
        if len(self.data_cache) > self.cache_size:
            removal_keys = list(self.data_cache)  # list of h5 files in cache
            removal_keys.remove(file_path)  # remove current file from list removal
            self.data_cache.pop(removal_keys[0])  # pop one of the datasets from cache
            self.data_info = [{'file_path': di['file_path'],
                               'type': di['type'],
                               'shape': di['shape'],
                               'cache_idx': -1}
                              if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def add_data_info(self, file_path: str, load_data: bool):
        file = h5py.File(file_path)
        attributes = [k for k in file.keys()]
        for att in attributes:
            data = file[att]
            for example in data:
                idx = -1
                if load_data:
                    idx = self.add_data_cache(example, file_path)
                self.data_info.append({'file_path': file_path,
                                       'type': att,
                                       'shape': data.shape,
                                       'cache_idx': idx})

    def add_data_cache(self, data, file_path):
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_info(self, _type: str):
        return [di for di in self.data_info if di['type'] == _type]

    def get_data(self, _type, idx):
        fp = self.get_data_info(_type)[idx]['file_path']
        if fp not in self.data_cache:
            self.load_data(fp)
        cache_idx = self.get_data_info(_type)[idx]['cache_idx']
        return self.data_cache[fp][cache_idx]


if __name__ == '__main__':
    pass
