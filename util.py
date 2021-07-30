import os
import math
import random
import glob
import gzip
import pickle
import h5py

import cv2
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def slurpjson(fn):
    import json
    with open(fn, 'r') as f:
        return json.load(f)


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def mdir(path, verbose=True):
    try:
        os.makedirs(path)
    except FileExistsError:
        if verbose:
            print("Directory ", path, " already exists")


def save(fn, a):
    with gzip.open(fn, 'wb', compresslevel=2) as f:
        pickle.dump(a, f, 2)


def load(fn):
    with gzip.open(fn, 'rb') as f:
        return pickle.load(f)


def save_json(fn, data):
    import json
    with open(fn, 'wb') as outfile:
        outfile.write(json.dumps(data, indent=4, default=str).encode("utf-8"))


def get_filenames(path):
    return glob.glob('{}/*'.format(path))


def get_detached_tensor(tensor, numpy=False):
    new_tensor = tensor.cpu().detach().numpy() if numpy else tensor.cpu().detach()
    return new_tensor


def imread(fn, gray_scale=False):
    if gray_scale:
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(fn)
    return img


def show(img, **kwargs):
    if kwargs.get('cmap', False):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    if kwargs.get('title', None):
        plt.title(kwargs['title'])
    plt.show()


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def set_device():
    if torch.cuda.is_available():
        device = "cuda:{}".format(get_freer_gpu())
        # torch.backends.cudnn.benchmark = True
        print('Using GPU')
    else:
        device = 'cpu'
    return torch.device(device)


def accuracy(predicted, targets):
    _, predicted = torch.max(predicted, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    acc = correct / total
    return acc


def create_dirs(modelname, experiment=None):
    base_path = os.path.join('./trained_models', modelname)
    model_path = os.path.join(base_path, experiment) if experiment else base_path
    mdir(model_path)
    return model_path


def compute_averages_old(metrics, step):
    avg_loss = torch.stack([torch.FloatTensor(metrics['train_loss'][-step:])]).mean()
    test_avg_loss = torch.stack([torch.FloatTensor(metrics['test_loss'][-step:])]).mean()
    avg_acc = torch.stack([torch.FloatTensor(metrics['train_acc'][-step:])]).mean()
    test_avg_acc = torch.stack([torch.FloatTensor(metrics['test_acc'][-step:])]).mean()
    return avg_loss, test_avg_loss, avg_acc, test_avg_acc


def compute_averages(metrics, step):
    metric_list = []
    for k in metrics.keys():
        metric = torch.stack([torch.FloatTensor(metrics[k][-step:])]).mean()  # smoothing with history
        metric_list.append(metric)
    return tuple(metric_list)


def load_weights(model, weights_path):
    pretrained_dict = torch.load(weights_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    return model


def get_weights_tensorflow_model(file_path):
    keys = []
    params_dict = dict()
    with h5py.File(file_path, 'r') as f:
        f.visit(keys.append)
        for key in keys:
            if ':' in key:
                name = f[key].name.rsplit('/', 1)
                layer = name[0]
                weight_type = name[1]
                group = f[layer]
                weights = group[weight_type][:]
                params_dict[f[key].name] = weights
    for layer, values in params_dict.items():
        if 'kernel' in layer:
            if 'conv' in layer:
                params_dict[layer] = np.transpose(values, (3, 2, 0, 1))
            else:
                params_dict[layer] = np.transpose(values, (1, 0))
        elif 'bias' in layer:
            params_dict[layer] = values
    return keys, params_dict


def load_tf_state_dict(file_path, model):
    params, params_dict = get_weights_tensorflow_model(file_path)
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        filtered = {k: v for k, v in params_dict.items() if key.split('.')[0] == k.split('/')[2]}
        for layer, values in filtered.items():
            values = torch.from_numpy(values)
            param_type = key.split('.')[-1]
            if 'kernel' in layer and param_type == 'weight':
                model_state_dict[key] = values
            elif 'bias' in layer and param_type == 'bias':
                model_state_dict[key] = values
    model.load_state_dict(model_state_dict)
    return model


def plot_images(inputs, adversaries, labels, predictions, rows, cols, class_names):
    from data import decode_label_generic, inverse_normalization
    f, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(rows, cols*1.3))
    for i in range(rows):
        for j in range(cols):
            good = inverse_normalization(get_detached_tensor(inputs[i*cols+j]), transform=True)
            bad = inverse_normalization(get_detached_tensor(adversaries[i*cols+j]), transform=True)
            vis_crop = np.concatenate((good, bad), axis=1)
            ax[i][j].imshow(vis_crop, cmap='gray')
            title = ax[i][j].set_title("{} Pred: {}".format(decode_label_generic(labels[i*cols+j], class_names),
                                                            decode_label_generic(predictions[i*cols+j].max(dim=0)[1],
                                                                                 class_names)))
            plt.setp(title, color=('g' if predictions[i*cols+j].max(dim=0)[1] == labels[i*cols+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    plt.show()


def interval_metrics(metrics_dict, interval=100):
    """
    :param metrics_dict: metrics dict
    :param interval: steps interval
    :return: filtered metrics list
    """
    idxs = np.arange(len(metrics_dict['train_loss'])) * interval
    metrics = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[])
    for k, v in metrics_dict.items():
        m = metrics_dict[k]
        metrics[k] = [ex for i, ex in enumerate(m) if (i % interval) == 0]
    idxs = idxs[:len(metrics[k])]
    return idxs, metrics


def display_metrics(metrics_path, metrics_names, interval=100, eval_metrics=None, visualize=False):
    plt.style.use('ggplot')
    metrics_dict = load(metrics_path)
    indexes, metrics_dict = interval_metrics(metrics_dict, interval)
    if eval_metrics:
        eval_metrics = load(eval_metrics)
    for s in metrics_names: # ['loss', 'acc']
        plot_metric(metrics_dict, indexes, scalar=s, eval_metrics=eval_metrics, visualize=visualize)


def plot_metric(metrics_dict, indexes, scalar='loss', eval_metrics=None, visualize=False):
    plt.plot(indexes, metrics_dict['train_{}'.format(scalar)])
    plt.plot(indexes, metrics_dict['test_{}'.format(scalar)])
    plt.title('Training {}'.format(scalar))
    if eval_metrics:
        plt.plot(eval_metrics['step'], eval_metrics[scalar], ':', c='darkorange')
    if visualize:
        plt.show()


def boxplot_fisher(data, file_path):
    sns.set_theme(style="whitegrid")
    all_layers = []
    names = []
    for p in data:
        for k in p.keys():
            all_layers.append(p[k].numpy().ravel())
            names.append(k.split('fisher_')[1])
    plot = sns.boxplot(data=all_layers, orient='h')
    plt.title(os.path.basename(file_path).split('.')[0])
    plt.yticks(range(len(names)), names, fontsize=5)
    # plt.tight_layout()
    plt.autoscale()
    # plt.savefig(os.path.join(model_path, 'fisher_params.jpg'), dpi=300)
    plt.savefig(file_path, dpi=300)
    plt.close()


def set_seed(seed=1508):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



