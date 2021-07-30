import os
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from pprint import pprint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import util
import data
import models


device = util.set_device()


def least_likely_label(model, inputs):
    out = model(inputs)
    _, targets = torch.min(out.data, 1)
    targets = targets.detach()
    return targets


def get_model(structure, weights_path, **params):
    model = getattr(models, str(structure))(**params)
    weights = torch.load(weights_path)
    weights = {k: v for k, v in weights.items() if k in model.state_dict()}
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def compute_metrics(predictions, model_name=None, class_names=None):
    if not model_name:
        model_name = 'adversarial_generic'

    results = dict(model_name=model_name, report=dict())

    all_labels = [ex['label'] for ex in predictions]
    all_preds = [ex['pred_adv'] for ex in predictions]
    base_conf_matrix = confusion_matrix(all_labels, all_preds)
    pprint({'baseline_conf_matrix': base_conf_matrix.tolist()})

    for t in [0.5, 0.6, 0.7, 0.85, 0.9, 0.95]:
        filtered = []
        for ex in predictions:
            probabilities = ex['prob_adv']
            predicted_as = ex['pred_adv']
            if probabilities[predicted_as] >= t:
                filtered.append(ex)

        labels = [ex['label'] for ex in filtered]
        preds = [ex['pred_adv'] for ex in filtered]
        conf_matrix = confusion_matrix(labels, preds)

        meta = {
            'threshold_{}'.format(t): {
                'availability': len(filtered) / len(predictions),
                'classes': class_names,
                'confusion_matrix': conf_matrix.tolist(),
                'class_report': classification_report(labels,
                                                      preds, target_names=class_names,
                                                      labels=[i for i in range(len(class_names))],
                                                      output_dict=True)
            }
        }
        results['report'].update(meta)
    return results


class MomentumBoostFSGM(object):
    def __init__(self, model, epsilon=0.01, steps=40, decay=1.0, targeted=False, **kwargs):
        self.model = model
        self.eps = epsilon
        self.alpha = epsilon / steps
        self.steps = steps
        self.decay = decay
        self.targeted = -1 if targeted else 1
        self.criterion = nn.CrossEntropyLoss()

    def get_adversarial_sample(self, inputs):
        inputs = inputs.to(device)
        # inputs_vars = inputs.clone().to(device)
        targets = least_likely_label(self.model, inputs)
        momentum = torch.zeros_like(inputs).to(device)

        for i in range(self.steps):
            inputs.requires_grad = True
            # inputs_vars.requires_grad = True
            out = self.model(inputs)
            loss = self.targeted * self.criterion(out, targets).to(device)
            grad = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad, p=1)
            grad /= grad_norm
            grad += momentum * self.decay
            momentum = grad

            adv_samples = inputs + self.alpha * grad.sign()

            a = torch.clamp(inputs - self.eps, min=float(torch.min(inputs)))
            b = (adv_samples >= a).float() * adv_samples + (a > adv_samples).float() * a
            c = (b > inputs + self.eps).float() * (inputs + self.eps) + (inputs + self.eps >= b).float() * b
            inputs = torch.clamp(c, max=float(torch.max(inputs))).detach()

        adv_samples = torch.clamp(adv_samples, min=float(torch.min(inputs)), max=float(torch.max(inputs))).detach()
        return adv_samples


class BIM(object):
    def __init__(self, model, epsilon=0.001, alpha=0.001, steps=10, norm=False, targeted=False, **kwargs):
        self.model = model
        self.eps = epsilon
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.targeted = -1 if targeted else 1
        self.criterion = nn.CrossEntropyLoss()

    def get_adversarial_sample(self, inputs):
        inputs = inputs.to(device)
        targets = least_likely_label(self.model, inputs)
        for i in range(self.steps):
            inputs.requires_grad = True
            out = self.model(inputs)
            loss = self.targeted * self.criterion(out, targets).to(device)
            grad = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)[0]
            if self.norm:
                grad = self.alpha * grad / grad.view(grad.shape[0], -1).norm(dim=1)[:, None, None, None]
            else:
                grad = self.alpha * grad.sign()
            adv_samples = inputs + grad

            a = torch.clamp(inputs - self.eps, min=float(torch.min(inputs)))
            b = (adv_samples >= a).float() * adv_samples + (a > adv_samples).float() * a
            c = (b > inputs + self.eps).float() * (inputs + self.eps) + (inputs + self.eps >= b).float() * b
            inputs = torch.clamp(c, max=float(torch.max(inputs))).detach()

        adv_samples = torch.clamp(adv_samples, min=float(torch.min(inputs)), max=float(torch.max(inputs))).detach()
        return adv_samples


def projection(inputs, adv_inputs, epsilon, norm=None):
    """
    Define the range of the adversarial perturbation in the p-norm ball
    :param inputs:
    :param adv_inputs:
    :param epsilon:
    :param norm:
    :return:
    """
    if not norm:  # inf attacks
        adv_inputs = torch.max(torch.min(adv_inputs, inputs + epsilon), inputs - epsilon)
        # torch.clamp(adv_samples, min=float(torch.min(inputs)), max=float(torch.max(inputs))).detach()
    else:
        delta = adv_inputs - inputs
        mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= epsilon
        # compute the euclidean norm of the delta element-wisely in the batch
        scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
        # take the perturbations which are less than epsilon
        scaling_factor[mask] = epsilon
        delta *= epsilon

        delta *= epsilon / scaling_factor.view(-1, 1, 1, 1)
        adv_inputs = inputs + delta
    return adv_inputs


class PGD(object):
    def __init__(self, model, epsilon=0.01, alpha=0.1, steps=40, norm=None, random_start=False,
                 targeted=False, **kwargs):
        self.model = model
        self.eps = epsilon
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.random_start = random_start
        self.targeted = -1 if targeted else 1
        self.criterion = nn.CrossEntropyLoss()

    def random_perturbation(self, inputs):
        perturbation = torch.normal(torch.zeros_like(inputs), torch.ones_like(inputs))
        if not self.norm:
            perturbation = torch.sign(perturbation) * self.eps  # inf norm
        else:
            perturbation = projection(torch.zeros_like(inputs), perturbation, self.eps, self.norm)
        return inputs + perturbation

    def get_adversarial_sample(self, inputs):
        targets = least_likely_label(self.model, inputs)
        adv_samples = inputs.clone().to(device)
        if self.random_start:
            adv_samples = self.random_perturbation(adv_samples)

        for i in range(self.steps):
            adversaries = adv_samples.clone().detach()
            adversaries.requires_grad = True
            out = self.model(adversaries)
            loss = self.targeted * self.criterion(out, targets).to(device)
            grad = torch.autograd.grad(loss, adversaries, retain_graph=False, create_graph=False)[0]
            if not self.norm:  # inf norm
                grad = grad.sign() * self.alpha
            else:
                grad = grad * self.alpha / grad.view(adv_samples.shape[0], -1).norm(self.norm, dim=-1).view(-1, 1, 1, 1)

            adv_samples = adv_samples + grad
            adv_samples = projection(inputs, adv_samples, self.eps, self.norm)
            adv_samples = torch.clamp(adv_samples, min=float(torch.min(inputs)), max=float(torch.max(inputs))).detach()

        return adv_samples


class FGSM(object):
    def __init__(self, model, epsilon=0.001, targeted=False, **kwargs):
        self.model = model
        self.eps = epsilon
        self.targeted = -1 if targeted else 1
        self.criterion = nn.CrossEntropyLoss()

    def get_adversarial_sample(self, inputs):
        inputs = inputs.to(device)
        targets = least_likely_label(self.model, inputs)
        inputs.requires_grad = True
        loss = self.targeted * self.criterion(self.model(inputs), targets).to(device)
        grad = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)[0]
        adv_samples = inputs + self.eps * grad.sign()
        adv_samples = torch.clamp(adv_samples, min=float(torch.min(inputs)), max=float(torch.max(inputs))).detach()
        return adv_samples


class AdversarialSampler(object):
    def __init__(self, attack, model, threshold=None, print_results=True, class_names=None):
        self.attack_type = attack
        self.model = model
        self.sample_threshold = threshold
        self.print_results = print_results
        self.class_names = class_names

    def adversarial_attack(self, attack, **params):
        if attack == 'FGSM':
            adv_sampler = FGSM(self.model, **params)
        elif attack == 'BIM':
            adv_sampler = BIM(self.model, **params)
        elif attack == 'MOMENTUM':
            adv_sampler = MomentumBoostFSGM(self.model, **params)
        elif attack == 'PGD':
            adv_sampler = PGD(self.model, **params)
        else:
            raise NotImplementedError
        return adv_sampler

    def evaluate(self, inputs):
        scores = self.model(inputs)
        probabilities = F.softmax(scores, dim=1)
        _, predictions = torch.max(scores, dim=1)
        return probabilities, predictions

    def attack(self, data_loader, **params):
        adv_samples = []
        counter = 0
        mse = []
        adv_sampler = self.adversarial_attack(self.attack_type, **params)
        start = datetime.datetime.now()
        for batch in tqdm(data_loader):
            img, labels = batch
            img, labels = img.to(device), labels.to(device)
            prob, pred = self.evaluate(img)
            samples = adv_sampler.get_adversarial_sample(img)
            adv_prob, adv_pred = self.evaluate(samples)
            prob = util.get_detached_tensor(prob, numpy=True)
            adv_prob = util.get_detached_tensor(adv_prob, numpy=True)
            mse.append(F.mse_loss(samples, img).item())
            for i, adv in enumerate(samples):
                adv = util.get_detached_tensor(adv, numpy=True)
                meta = {
                    'crop': np.transpose(adv, (1, 2, 0)),  # save raw versions of the adversarial examples
                    'prob_clean': prob[i],
                    'prob_adv': adv_prob[i],
                    'pred_clean': int(pred[i]),
                    'pred_adv': int(adv_pred[i]),
                    'class_name': data.decode_label_generic(labels[i], self.class_names),
                    'label': int(labels[i]),
                }
                if self.sample_threshold and meta['prob_clean'][meta['pred_clean']] < self.sample_threshold:
                    continue
                else:
                    adv_samples.append(meta)
            counter += len(img)
        print('Generating results for {} percent of the original dataset'.format(len(adv_samples) / counter))
        end = datetime.datetime.now()
        avg_difference = torch.stack([torch.FloatTensor(mse)]).mean()
        print('Adversarial Samples MSE {}'.format(avg_difference))
        results = compute_metrics(adv_samples, class_names=self.class_names)
        results['MSE'] = avg_difference.numpy().tolist()
        results['attack'] = self.attack_type
        results['crafting_duration'] = end - start
        results['params'] = dict()
        attack_args = adv_sampler.__dict__
        for k in attack_args.keys():
            if k == 'model' or k == 'criterion':
                continue
            results['params'][k] = attack_args[k]
        if self.print_results:
            pprint(results)
        return adv_samples, results

    def get_samples(self, inputs, **params):
        adv_sampler = self.adversarial_attack(self.attack_type, **params)
        samples = adv_sampler.get_adversarial_sample(inputs)
        mse = F.mse_loss(samples, inputs).item()
        return samples, mse


def visualize_adversarial_attack(train=False, mode='pgd', **kwargs):
    torch.manual_seed(2)
    net = get_model(structure='SOTANetwork',
                    weights_path='./trained_models/STL-10/202104013_baseline_STL_densenet/final.pt',
                    architecture='densenet',
                    base_network=False,
                    in_channels=3,
                    num_outputs=10)
    train_data, test_data = data.get_stl_datasets(adversarial_crafting=True)
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    if train:
        dataset = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    else:
        dataset = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    if mode == 'mifgsm':
        adv = MomentumBoostFSGM(model=net, epsilon=0.05, steps=40, decay=0.9, targeted=True, **kwargs)
    elif mode == 'pgd':
        adv = PGD(net, epsilon=.035, alpha=0.01, steps=10, random_start=True, targeted=True, **kwargs)
    elif mode == 'l2':
        adv = PGD(net, epsilon=0.8, alpha=0.1, steps=40, norm=2, random_start=True, targeted=True, **kwargs)
    else:
        adv = FGSM(model=net, epsilon=0.035, targeted=True, **kwargs)

    for i, batch in enumerate(dataset):
        img, labels = batch
        img, labels = img.to(device), labels.to(device)
        output = net(img)
        samples = adv.get_adversarial_sample(img)
        _output = net(samples)
        print('Clean: {}'.format(util.accuracy(output, labels)), 'Bad: {}'.format(util.accuracy(_output, labels)))
        _, clean_predictions = torch.max(output, 1)
        _, bad_samples_predictions = torch.max(_output, 1)
        t = int(labels[1])
        o = clean_predictions[1]
        _o = bad_samples_predictions[1]
        data.inverse_normalization(util.get_detached_tensor(img[1]),
                                   show=True, title='clean {} pred as {}'.format(t, o))
        data.inverse_normalization(util.get_detached_tensor(samples[1]),
                                   show=True, title='bad {} pred as {}'.format(t, _o))
        if i > 5:
            break

    util.plot_images(img, samples, labels, _output, 5, 3, class_names)


def get_pgd_samples(data, model, class_names, save_path=None, adversarial_sets=False):
    """
    Generate PGD data to test robustness or train
    """
    sampler = AdversarialSampler(attack='PGD', model=model, class_names=class_names)
    if save_path is not None:
        steps = 40
    else:
        steps = 10

    epsilons = [0.015, 0.025, 0.035]
    sample_set = []
    for i, ep in enumerate(epsilons):
        if not adversarial_sets:
            if i == 0 or i == 1:
                continue
        print('Generating adversarial samples for epsilon: {}'.format(ep))
        samples, results = sampler.attack(data, epsilon=ep, alpha=0.01, steps=steps, random_start=True, targeted=True)
        sample_set.extend(samples)
    return sample_set, results


def generate_adversarial_samples(test=False, save_path=None, method='PDG_inf', weights_path=None,
                                 adversarial_sets=False):
    if save_path:
        util.mdir(save_path)
    torch.manual_seed(1)
    if weights_path is None:
        weights_path = './trained_models/STL-10/202104013_baseline_STL_densenet/final.pt'
    net = get_model(structure='SOTANetwork',
                    weights_path=weights_path,
                    architecture='densenet',
                    base_network=False,
                    in_channels=3,
                    num_outputs=10)
    train_data, test_data, class_names = data.get_stl_datasets(adversarial_crafting=True)
    if test:
        suffix = 'test'
        data_ = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    else:
        suffix = 'train'
        data_ = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    if method == 'FGSM':
        sampler = AdversarialSampler(attack='FGSM', model=net, class_names=class_names)
        samples, results = sampler.attack(data_, epsilon=0.035, targeted=True)
    elif method == 'MOMENTUM':
        sampler = AdversarialSampler(attack='MOMENTUM', model=net, class_names=class_names)
        samples, results = sampler.attack(data_,  epsilon=0.04, steps=40, decay=0.9, targeted=True)
    elif method == 'PDG_inf':
        samples, results = get_pgd_samples(data_, net, class_names, save_path, adversarial_sets)
    elif method == 'PGD_l2':
        sampler = AdversarialSampler(attack='PGD', model=net, class_names=class_names)
        samples, results = sampler.attack(data_, epsilon=0.9, alpha=0.1, steps=40, norm=2, random_start=True, targeted=True)
    else:
        raise NotImplementedError

    if save_path is not None:
        util.save(os.path.join(save_path, '{}_adversarial_samples'.format(suffix)), samples)
        util.save_json(os.path.join(save_path, 'adv_results.json'), results)


def adversarial_curves(attack, weights_path=None, phase='train', **kwargs):
    torch.manual_seed(1)
    if weights_path is None:
        weights_path = './trained_models/STL-10/202104013_baseline_STL_densenet/final.pt'
    net = get_model(structure='SOTANetwork',
                    weights_path=weights_path,
                    architecture='densenet',
                    base_network=False,
                    in_channels=3,
                    num_outputs=10)
    train_data, test_data, class_names = data.get_stl_datasets()
    train = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    test = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    sampler = AdversarialSampler(attack=attack, model=net, print_results=False, class_names=class_names)
    metrics = dict(accuracy=[], mse=[])
    start = datetime.datetime.now()
    if kwargs.get('l2_attack', False):
        epsilon = np.arange(0.0, 6.1, 1./3)
    else:
        epsilon = np.arange(0.0, 0.52, 0.025)
    for ep in epsilon:
        _, results = sampler.attack(train if phase == 'train' else test,
                                    epsilon=ep, targeted=True, **kwargs)
        acc = results['report']['threshold_0.5']['class_report']['accuracy']
        mse = results['MSE']
        print('Epsilon: {} , Accuracy: {}'.format(ep, acc))
        metrics['accuracy'].append(acc)
        metrics['mse'].append(mse)
    end = datetime.datetime.now()
    print('Total time (s)', end - start)
    return metrics


if __name__ == '__main__':
    # samples_path = '/media/miguel/ALICIUM/Miguel/DATASETS/RESEARCH_EWC/PGD_samples'
    # visualize_adversarial_attack()

    # generate_adversarial_samples(save_path=samples_path, adversarial_sets=True)
    # generate_adversarial_samples(test=True, save_path=samples_path, adversarial_sets=True)

    # STL: test adversarial attacks
    # generate_adversarial_samples(test=True) # 0.554484
    # generate_adversarial_samples(test=True, method='PDG_inf',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_STL_densenet/final.pt') # 0.8919511860688332
    # generate_adversarial_samples(test=True, method='PDG_inf',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_mixed_STL_densenet/final.pt') # 0.90658
    # generate_adversarial_samples(test=True, method='PDG_inf',
    #                              weights_path='./trained_models/STL-10/2021_adversarial_ewc_online/final.pt') #  0.9011563876651982,

    # generate_adversarial_samples(test=True, method='MOMENTUM') # 0.546487
    # generate_adversarial_samples(test=True, method='MOMENTUM',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_STL_densenet/final.pt') # 0.798396
    #
    # generate_adversarial_samples(test=True, method='MOMENTUM',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_mixed_STL_densenet/final.pt') # 0.797116

    # generate_adversarial_samples(test=True, method='MOMENTUM',
    #                              weights_path='./trained_models/STL-10/2021_adversarial_ewc_online/final.pt') # 0.824107683000605
    #

    # generate_adversarial_samples(test=True, method='FGSM') # 0.5713309647444657,
    # generate_adversarial_samples(test=True, method='FGSM',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_STL_densenet/final.pt') #  0.8030186748529036
    #
    # generate_adversarial_samples(test=True, method='FGSM',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_mixed_STL_densenet/final.pt') # 0.811306242789386

    # generate_adversarial_samples(test=True, method='FGSM',
    #                              weights_path='./trained_models/STL-10/2021_adversarial_ewc_online/final.pt') # 0.7674689440993789

    # generate_adversarial_samples(test=True, method='PGD_l2') # 0.546487
    # generate_adversarial_samples(test=True, method='PGD_l2',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_STL_densenet/final.pt') # 0.946716
    #
    # generate_adversarial_samples(test=True, method='PGD_l2',
    #                              weights_path='./trained_models/STL-10/20210522_adversarial_mixed_STL_densenet/final.pt') # 0.914618
    #
    # generate_adversarial_samples(test=True, method='PGD_l2',
    #                              weights_path='./trained_models/STL-10/2021_adversarial_ewc_online/final.pt') # 0.9316929386133539,


    pass

