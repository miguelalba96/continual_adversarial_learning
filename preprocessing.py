import os
import glob
import math
import random
from collections import Counter
from pprint import pprint

import cv2
import numpy as np

from sklearn.model_selection import ShuffleSplit

import util
from data import write_class_dataset

random.seed(1)


class PreProcessAddition(object):
    """
    This class helps to preprocess the MRL open dataset for eye states
    http://mrl.cs.vsb.cz/eyedataset
    """
    def __init__(self, data_path, sample_size=600):
        self.data_path = data_path
        self.resize = (100, 68)
        self.sample = sample_size
        self.bad_samples_tolerance = 50

    def fitler_person_samples(self, person, person_fns):
        data = []
        tolerance = 0
        for fn in person_fns:
            img = util.imread(fn, gray_scale=True)
            img = cv2.resize(img, self.resize)
            assert img.shape == (68, 100)
            attributes = os.path.basename(fn).split('_')
            meta = {
                'person': attributes[0],
                'idx': attributes[1],
                'label': 'closed' if int(attributes[4]) == 0 else 'open',
                'crop': img.astype(np.float32),
            }
            if int(attributes[6]) == 0:
                tolerance += 1
            data.append(meta)
        data = random.sample(data, len(data))
        _closed = [ex for ex in data if ex['label'] == 'closed']
        _open = [ex for ex in data if ex['label'] == 'open']
        closed_samples = random.sample(_closed, self.sample // 2) if len(_closed) > self.sample // 2 else _closed
        open_samples = random.sample(_open, self.sample // 2) if len(_open) > self.sample // 2 else _open
        print('{} samples from {} loaded, bad samples {}'.format(len(data), person, tolerance))
        return closed_samples, open_samples

    def generate_example_sets(self, **kwargs):
        random.seed(1)
        persons = [d for d in os.listdir(self.data_path) if not (str(d).endswith('txt') or str(d).endswith('ods'))]
        counts = dict(open=0, closed=0)
        data = dict(open=[], closed=[])
        for p in persons:
            person_fns = glob.glob(os.path.join(self.data_path, p, '*'))
            closed_samples, open_samples = self.fitler_person_samples(p, person_fns)
            counts['open'] += len(open_samples)
            counts['closed'] += len(closed_samples)
            data['closed'].extend(closed_samples)
            data['open'].extend(open_samples)

        print('Data Statistics:', counts)
        closed_eye = random.sample(data['closed'], len(data['closed']))
        open_eye = random.sample(data['open'], len(data['open']))
        write_class_dataset(closed_eye, 'closed', 'eye_mlr', data_path=self.data_path, **kwargs)
        write_class_dataset(open_eye, 'open', 'eye_mlr', data_path=self.data_path, **kwargs)


def generate_adversarial_set(data_fn, out_name, class_names=None, sample=None, **kwargs):
    """
    function to write adversarial data as hd5 files
    :param data_fn: pickle file containing the adversarial examples
    :param out_name: output name for the folder containing the hd5
    :param class_names: class names
    :param sample: train with a subset of adversarial examples
    :param kwargs: additional arguments
    """
    data = util.load(data_fn)
    out_dir = os.path.join(os.path.dirname(data_fn), out_name)
    util.mdir(out_dir)
    if class_names is None:
        classes = class_names
    else:
        classes = list(set([ex['class_name'] for ex in data]))
    print('Data Statistics:', Counter([ex['class_name'] for ex in data]))
    random.seed(1)
    print('train data num samples {}'.format(len(data)))
    # pre shuffle
    train_data = random.sample(data, len(data))
    for cl in classes:
        train = [ex for ex in train_data if ex['class_name'] == cl]
        if sample:
            train = random.sample(train, sample)
        write_class_dataset(train, cl, 'adv_samples', class_names=class_names, data_path=out_dir, **kwargs)


if __name__ == '__main__':
    # for the STL dataset
    _classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    generate_adversarial_set(data_fn='./datasets/PGD_samples/train_adversarial_samples', out_name='all_adversarial_set',
                             class_names=_classes, num_records=12, training=True)
    generate_adversarial_set(data_fn='./datasets/PGD_samples/test_adversarial_samples', out_name='all_adversarial_set',
                             class_names=_classes, num_records=12, training=False)
    pass
