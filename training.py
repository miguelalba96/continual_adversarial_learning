import os
import copy

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import util
import models
import data
from util import accuracy, set_device

from attacks import AdversarialSampler
from evaluation import EvalDataset, EvalSOTA
from ewc import ElasticWeightConsolidation

device = set_device()


OPTIMIZATION_PARAMS = {
            'base_lr': 1e-4,
            'epochs': 50,
            'batch_size': 64,
            'test_iter': 50,
            'decay_step': 2,
            'gamma': 0.5,
            'checkpoint': 10,
            'lambda': 0.1,
            'weight_decay': 0.1,
            'fisher_samples': 4000,
            'schedule': True,
            'optimizer': 'SGDM'}


ADVERSARIAL_PATH = './datasets/PGD_samples'


@torch.no_grad()
def evaluate_dataset(model, dataset, criterion):
    avg_loss = []
    avg_acc = []
    with torch.no_grad():
        for batch in dataset:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            avg_loss.append(loss.item())
            avg_acc.append(accuracy(output, targets))
    avg_loss = torch.stack([torch.FloatTensor(avg_loss)]).mean()
    avg_acc = torch.stack([torch.FloatTensor(avg_acc)]).mean()
    return avg_loss, avg_acc


def save_adversaries(adv_samples, labels, class_names, phase, iteration):
    save_path = os.path.join(ADVERSARIAL_PATH, 'adversarial_data')
    if not os.path.isdir(save_path):
        util.mdir(save_path, verbose=False)
    samples = []
    for i, adv in enumerate(adv_samples):
        samples.append(
            {'crop': np.transpose(util.get_detached_tensor(adv, numpy=True), (1, 2, 0)),
             'class_name': data.decode_label_generic(labels[i], class_names),
             'label': int(labels[i])
            }
        )
    util.save(os.path.join(save_path, '{}_{}_adversarial_samples'.format(phase, iteration)), samples)


def get_adversarial_samples(inputs, targets, model, adv_params, save_data=False, class_names=None, phase=None,
                            iteration=None):
    inputs, targets = inputs.to(device), targets.to(device)
    model.eval()
    sampler = AdversarialSampler(attack=adv_params['method'], model=model)
    method_params = copy.deepcopy(adv_params)
    del method_params['method']
    adv_samples, mse = sampler.get_samples(inputs, **method_params)
    if save_data and all([arg is not None for arg in [class_names, phase, iteration]]):
        save_adversaries(adv_samples, targets, class_names, phase, iteration)
    return adv_samples, mse


def create_adversarial_summary_buffers(metrics_dict):
    for phase in ['train', 'test']:
        metrics_dict['{}_adv_loss'.format(phase)] = []
        metrics_dict['{}_adv_acc'.format(phase)] = []
        metrics_dict['{}_adv_mse'.format(phase)] = []
    return metrics_dict


def log_fisher_information(fisher_data, model_path):
    fim, transformed = fisher_data
    util.save(os.path.join(model_path, 'weight_importance.pz'), fim)
    fisher_fn = os.path.join(model_path, 'fisher_params.jpg')
    util.boxplot_fisher(fim, fisher_fn)


class Trainer(object):
    def __init__(self, model_name, model, experiment='source_task', hyperparams=None, start_weights=None,
                 fine_tune=None, class_names=None, **kwargs):
        self.model = model
        self.cuda = kwargs.get('use_gpu', True)
        if self.cuda:
            self.model = model.to(device)
        self.params = dict(OPTIMIZATION_PARAMS, **hyperparams) if hyperparams else OPTIMIZATION_PARAMS
        self.start_weights = start_weights
        self.criterion, self.optimizer = self.set_optimization(fine_tune if fine_tune else self.model.parameters())
        self.model_path = util.create_dirs(model_name, experiment)
        self.class_names = class_names

    def set_optimization(self, params):
        criterion = nn.CrossEntropyLoss()
        if self.params['optimizer'] == 'ADAM':
            optimizer = optim.AdamW(params, lr=self.params['base_lr'], weight_decay=self.params['weight_decay'])  # 0.25
        else:
            optimizer = optim.SGD(params, lr=self.params['base_lr'], momentum=0.9)
        return criterion, optimizer

    def train_step(self, inputs, targets):
        if self.cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        summary = {
            'Mode': 'Train',
            'Loss': loss.item(),
            'Accuracy': accuracy(output, targets),
        }
        return summary

    def test_step(self, inputs, targets):
        if self.cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            output = self.model(inputs)
            loss = self.criterion(output, targets)
        summary = {'Mode': 'Test',
                   'Loss': loss.item(),
                   'Accuracy': accuracy(output, targets)}
        return summary

    def adversarial_step(self, img, labels, phase, metrics_dict, adversarial_params, double_step,
                         consolidate=None, **kwargs):
        if double_step:
            if phase == 'train':
                summary = self.train_step(img, labels)
            else:
                summary = self.test_step(img, labels)
            metrics_dict['{}_loss'.format(phase)].append(summary['Loss'])
            metrics_dict['{}_acc'.format(phase)].append(summary['Accuracy'])

        adversaries, mse = get_adversarial_samples(img, labels, self.model, adversarial_params,
                                                   class_names=self.class_names, **kwargs)
        if phase == 'train':
            if consolidate:
                self.model, adv_summary = consolidate(self.model, adversaries, labels)  # before img, labels
            else:
                adv_summary = self.train_step(adversaries, labels)
        else:
            adv_summary = self.test_step(adversaries, labels)
        metrics_dict['{}_adv_mse'.format(phase)].append(mse)
        metrics_dict['{}_adv_loss'.format(phase)].append(adv_summary['Loss'])
        metrics_dict['{}_adv_acc'.format(phase)].append(adv_summary['Accuracy'])
        return metrics_dict

    def optimize(self, train_dataset, test_dataset, eval_dataset=None, source_dataset=None,
                 adversarial_params=None, double_step=False, consolidate_weights=False, **kwargs):
        print(self.params)
        if self.start_weights:
            print('Loading pre-trained weights')
            self.model = util.load_weights(self.model, self.start_weights)
        test = iter(test_dataset)
        metrics = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[])
        eval_metrics = dict(step=[], loss=[], acc=[]) if eval_dataset else None
        step = 0
        if self.params['schedule']:
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params['decay_step'],
                                                        gamma=self.params['gamma'])
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 1, T_mult=1)
        if consolidate_weights:
            consolidate = ElasticWeightConsolidation(self.optimizer,
                                                     self.criterion,
                                                     self.params['weight_decay'], 0,
                                                     transformation='original',
                                                     empirical=False,
                                                     device=device)
            self.model = consolidate.estimate_fisher_information(self.model,
                                                                 source_dataset,
                                                                 sample_size=self.params['fisher_samples'])
            fisher, transform = consolidate.get_fisher_diagonal(self.model)
            log_fisher_information(fisher_data=(fisher, transform), model_path=self.model_path)
        else:
            consolidate = None

        if adversarial_params:
            metrics = create_adversarial_summary_buffers(metrics)
            util.save_json(os.path.join(self.model_path, 'adversarial_params.json'), adversarial_params)
        for e in range(self.params['epochs']):
            print('Learning rate:', scheduler.get_last_lr() if self.params['schedule'] else self.params['base_lr'])
            for i, batch in enumerate(train_dataset):
                img, labels = batch
                if adversarial_params:
                    metrics = self.adversarial_step(img, labels, 'train', metrics,
                                                    adversarial_params, double_step, consolidate=consolidate, **kwargs)
                else:
                    if consolidate:
                        self.model, train_summary = consolidate(self.model, img, labels)
                    else:
                        train_summary = self.train_step(img, labels)
                    metrics['train_loss'].append(train_summary['Loss'])
                    metrics['train_acc'].append(train_summary['Accuracy'])
                try:
                    test_img, test_labels = next(test)
                except StopIteration:
                    test = iter(test_dataset)
                    test_img, test_labels = next(test)
                if adversarial_params:
                    metrics = self.adversarial_step(test_img, test_labels, 'test', metrics,
                                                    adversarial_params, double_step, **kwargs)
                else:
                    test_summary = self.test_step(test_img, test_labels)
                    metrics['test_loss'].append(test_summary['Loss'])
                    metrics['test_acc'].append(test_summary['Accuracy'])
                if (step % self.params['test_iter']) == 0:
                    average_results = util.compute_averages(metrics, step)
                    if adversarial_params:
                        template = 'Epoch: {}, Step: {}, train loss: {}, test loss: {}, train acc: {}, test acc: {}, ' \
                                   'adv train loss: {}, adv train acc: {}, adv train MSE: {}, adv test loss: {}, ' \
                                   'adv test acc: {}, adv test MSE: {}'
                    else:
                        template = 'Epoch: {}, Step: {}, train loss: {}, test loss: {}, train acc: {}, test acc: {}'
                    print(template.format(e, step, *average_results))
                step += 1

            if self.params['schedule']:
                scheduler.step()
            if eval_dataset:
                avg_loss, avg_acc = evaluate_dataset(self.model, eval_dataset, self.criterion)
                eval_metrics['step'].append(step)
                eval_metrics['loss'].append(avg_loss)
                eval_metrics['acc'].append(avg_acc)
                print('Epoch: {}, eval loss: {}, eval acc {}'.format(e, avg_loss, avg_acc))
            if (e % self.params['checkpoint']) == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_path, 'checkpoint_{}.pt'.format(e)))
        print('Training completed')
        return metrics, eval_metrics


def train_baseline_model():
    """
    Train on the STL-10 dataset using image net as pre training
    """
    experiment = '202104013_baseline_STL_densenet'
    train_data, test_data, class_names = data.get_stl_datasets()
    train = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    test = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    model = models.SOTANetwork(architecture='densenet', in_channels=3, num_outputs=10)
    trainer = Trainer(model_name='STL-10',
                      model=model,
                      hyperparams={'base_lr': 0.0008,  # 0.00075
                                   'batch_size': 32,
                                   'weight_decay': 0.35,
                                   'epochs': 25,  # 70
                                   'decay_step': 1},  # 4
                      experiment=experiment,
                      fine_tune=[{'params': model.pre_trained.features.parameters(), 'lr': 5e-5, 'weight_decay': 0.4},
                                 {'params': model.pre_trained.classifier.parameters(), 'lr': 2e-4,
                                  'weight_decay': 0.15}],
                      use_gpu=True)
    metrics, eval_metrics = trainer.optimize(train, test, test)
    torch.save(trainer.model.state_dict(), os.path.join(trainer.model_path, 'final.pt'))
    util.save(os.path.join(trainer.model_path, 'history.pz'), metrics)
    util.save(os.path.join(trainer.model_path, 'eval_history.pz'), eval_metrics)
    evaluator = EvalSOTA(model_name='STL-10/{}'.format(experiment),
                         model=model,
                         class_names=class_names,
                         save_predictions=False,
                         use_gpu=True)
    results = evaluator.evaluate(test)
    return results


def train_adversarial(dourble_step=False):
    """
    Adversarial training on STL-10 dataset
    :param dourble_step: True to update parameters with adversarial examples and then clean data, False
                         in case of standard adversarial training
    """
    torch.manual_seed(1)
    experiment = '20210522_adversarial_STL_densenet'
    train_data, test_data, _ = data.get_stl_datasets()
    train = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    test = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    model = models.SOTANetwork(architecture='densenet', in_channels=3, num_outputs=10)
    trainer = Trainer(model_name='STL-10',
                      model=model,
                      hyperparams={'base_lr': 0.0002,  # 0.00075
                                   'batch_size': 32,
                                   'weight_decay': 0.35,
                                   'epochs': 25,  # 70
                                   'decay_step': 1},  # 4
                      fine_tune=[{'params': model.pre_trained.features.parameters(), 'lr': 5e-5, 'weight_decay': 0.4},
                                 {'params': model.pre_trained.classifier.parameters(), 'lr': 2e-4,
                                  'weight_decay': 0.15}],
                      experiment=experiment,
                      use_gpu=True)
    adversarial_params = {
        'method': 'PGD',
        'epsilon': 0.035,
        'alpha': 0.01,
        'steps': 10,
        'norm': None,
        'random_start': True,
        'targeted': True
    }
    metrics, eval_metrics = trainer.optimize(train, test, test, adversarial_params=adversarial_params,
                                             double_step=dourble_step)
    torch.save(trainer.model.state_dict(), os.path.join(trainer.model_path, 'final.pt'))
    util.save(os.path.join(trainer.model_path, 'history.pz'), metrics)
    util.save(os.path.join(trainer.model_path, 'eval_history.pz'), eval_metrics)


def train_ewc_v1(start_weights=None, debug=False):
    """
    Train with adversarial examples generated from the best model (EWC-v1)
    :param start_weights: None to use default pre training
    :param debug: True to use 10 observations to compute the fisher information matrix of the model
    :return: None
    """
    torch.manual_seed(1)
    data_path = './datasets/PGD_samples/all_adversarial_set'
    train_data = data.HDF5Dataset(data_dir=data_path, phase='train', cache_size=10,
                                  transform=data.process_adversaries())
    test_data = data.HDF5Dataset(data_dir=data_path, phase='test', cache_size=10,
                                 transform=data.process_adversaries(False))
    # eval data is the clean data
    source_data, eval_data, class_names = data.get_stl_datasets()

    model = models.SOTANetwork(architecture='densenet', in_channels=3, num_outputs=10)
    experiment = '2021_adversarial_ewc-v1_STL_densenet'
    train = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    source = DataLoader(source_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    eval = DataLoader(eval_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    fisher_samples = 10 if debug else 4000

    if start_weights is None:
        start_weights = './trained_models/STL-10/202104013_baseline_STL_densenet/final.pt'

    trainer = Trainer(model_name='STL-10',
                      model=model,
                      hyperparams={'base_lr': 0.0002,  # 0.00075
                                   'batch_size': 32,
                                   'weight_decay': 0.1,  # 0.35
                                   'epochs': 20,  # 70
                                   'decay_step': 1,
                                   'fisher_samples': fisher_samples},  # 4
                      experiment=experiment,
                      start_weights=start_weights,
                      fine_tune=[{'params': model.pre_trained.features.parameters(), 'lr': 5e-5},
                                 {'params': model.pre_trained.classifier.parameters(), 'lr': 2e-4}],
                      use_gpu=True)

    metrics, eval_metrics = trainer.optimize(train, test, eval, source, consolidate_weights=True)
    torch.save(trainer.model.state_dict(), os.path.join(trainer.model_path, 'final.pt'))
    util.save(os.path.join(trainer.model_path, 'history.pz'), metrics)
    util.save(os.path.join(trainer.model_path, 'eval_history.pz'), eval_metrics)
    evaluator = EvalSOTA(model_name='STL-10/{}'.format(experiment),
                         model=model,
                         class_names=class_names,
                         save_predictions=False,
                         use_gpu=True)
    evaluator.evaluate(test)


def train_adversarial_ewc(start_weights=None, debug=False):
    """
    Adversarial training using EWC online (EWC-AT)
    :param start_weights: None to use default pre training
    :param debug: True to use 10 observations to compute the fisher information matrix of the model
    :return: None
    """
    torch.manual_seed(1)
    train_data, test_data, class_names = data.get_stl_datasets()
    model = models.SOTANetwork(architecture='densenet', in_channels=3, num_outputs=10)
    experiment = '2021_adversarial_ewc_online'
    train = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    test = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    fisher_samples = 10 if debug else 4000

    if start_weights is None:
        start_weights = './trained_models/STL-10/202104013_baseline_STL_densenet/final.pt'

    trainer = Trainer(model_name='STL-10',
                      model=model,
                      hyperparams={'base_lr': 0.0002,  # 0.00075
                                   'batch_size': 32,
                                   'weight_decay': 0.5,
                                   'epochs': 20,  # 70
                                   'decay_step': 1,
                                   'fisher_samples': fisher_samples},  # 4
                      experiment=experiment,
                      start_weights=start_weights,
                      fine_tune=[{'params': model.pre_trained.features.parameters(), 'lr': 5e-5},
                                 {'params': model.pre_trained.classifier.parameters(), 'lr': 2e-4}],
                      use_gpu=True)

    adversarial_params = {
        'method': 'PGD',
        'epsilon': 0.035,
        'alpha': 0.01,
        'steps': 10,
        'norm': None,
        'random_start': True,
        'targeted': True
    }

    metrics, eval_metrics = trainer.optimize(train, test, eval_dataset=test, source_dataset=train,
                                             adversarial_params=adversarial_params,
                                             consolidate_weights=True)
    torch.save(trainer.model.state_dict(), os.path.join(trainer.model_path, 'final.pt'))
    util.save(os.path.join(trainer.model_path, 'history.pz'), metrics)
    util.save(os.path.join(trainer.model_path, 'eval_history.pz'), eval_metrics)
    evaluator = EvalSOTA(model_name='STL-10/{}'.format(experiment),
                         model=model,
                         class_names=class_names,
                         save_predictions=False,
                         use_gpu=True)
    evaluator.evaluate(test)


if __name__ == '__main__':
    # train STL-10
    # train_baseline_model()
    # train_adversarial(dourble_step=True)
    # train_adversarial(dourble_step=False)
    # train_ewc_v1()
    train_adversarial_ewc()
    pass
