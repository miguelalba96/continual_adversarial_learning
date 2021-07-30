from functools import reduce

import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from util import accuracy


class ElasticWeightConsolidation(object):
    def __init__(self, optimizer, criterion, _lambda=1, gamma=0, transformation='abs_dist',
                 empirical=False, verbose=1000, device=None, **kwargs):
        self._lambda = _lambda
        self.gamma = gamma
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.trans = transformation
        self.empirical = empirical
        self.verbose = verbose
        self.add_params = dict(**kwargs)

    def transformation(self, estimated_fisher):
        if self.trans == 'inverse':
            return 1 / (self._lambda * estimated_fisher + self.gamma)
        elif self.trans == 'abs_dist':
            maximum = torch.full(estimated_fisher.shape, estimated_fisher.max()).to(self.device)
            t = maximum - torch.abs(estimated_fisher - estimated_fisher.min()).to(self.device)
            return (self._lambda * t) + self.gamma
        else:
            return self._lambda * estimated_fisher

    def get_fisher_diagonal(self, model):
        fisher = []
        transform = []
        for name, param in model.named_parameters():
            buffer_name_param = name.replace('.', '_')
            estimated_fisher = getattr(model, '{}_estimated_fisher'.format(buffer_name_param))
            fisher_transform = self.transformation(estimated_fisher)
            fisher.append({'fisher_{}'.format(buffer_name_param): estimated_fisher.cpu()})
            transform.append({'fisher_{}'.format(buffer_name_param): fisher_transform.cpu()})
        return fisher, transform

    @staticmethod
    def update_mean_params(model):
        for name, param in model.named_parameters():
            buffer_param_name = name.replace('.', '_')
            model.register_buffer('{}_estimated_mean'.format(buffer_param_name), param.data.clone())
        return model

    def update_fisher_params(self, model, data_loader, sample_size=None):
        fisher_diagonals = dict()
        model.eval()
        # initialize in zero fisher values
        for name, param in model.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '_')
                fisher_diagonals[buffer_name] = torch.zeros_like(param)
        # sample size should be the number of batches desired to compute the fisher information None is all the data
        num_samples = 0
        # sample batches constantly, repeat the dataset if necessary'
        for index, (inputs, targets) in enumerate(data_loader):
            if num_samples >= sample_size:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            log_likelihoods = F.log_softmax(model(inputs), dim=1)
            if self.empirical:
                labels = targets.unsqueeze(1)
            else:
                # sample from the distribution of the posterior MC estimate E[grad loglik * grad loglik^T]
                labels = Categorical(logits=log_likelihoods).sample().unsqueeze(1).detach()
            samples = log_likelihoods.gather(1, labels)
            idx = 0
            batch_size = inputs.size(0)
            while idx < batch_size and num_samples < sample_size:
                model.zero_grad()
                torch.autograd.backward(samples[idx], retain_graph=True)
                num_samples += 1
                idx += 1
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        buffer_name = name.replace('.', '_')
                        fisher_diagonals[buffer_name] += (param.grad * param.grad)  # /num_samples
                        fisher_diagonals[buffer_name].detach_()
                if self.verbose and (num_samples % self.verbose) == 0:
                    print('Num of samples used for Fisher estimation: {}'.format(num_samples))

        # todo: theoretically is the sum of the expected values.
        for name, square_grad in fisher_diagonals.items():
            square_grad /= float(num_samples)

        for name, param in model.named_parameters():
            buffer_name = name.replace('.', '_')
            model.register_buffer('{}_estimated_fisher'.format(buffer_name), fisher_diagonals[buffer_name].clone())
        return model

    def old_update_fisher_params(self, model, dataset, sample_size):
        log_likelihoods = []
        for i, (inputs, targets) in enumerate(dataset):
            if i > sample_size:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = F.log_softmax(model(inputs), dim=1)
            log_likelihoods.append(outputs[:, targets])

        estimate = torch.cat(log_likelihoods).mean()  # nnloss
        grads = autograd.grad(estimate, model.parameters())
        buffer_param_names = [name.replace('.', '_') for name, param in model.named_parameters()]
        for buffer_name, param in zip(buffer_param_names, grads):
            model.register_buffer('{}_estimated_fisher'.format(buffer_name), param.data.clone() ** 2)
        return model

    def estimate_fisher_information(self, model, dataset, sample_size):
        model = self.update_mean_params(model)
        model = self.update_fisher_params(model, dataset, sample_size)
        return model

    def elastic_loss(self, model):
        try:
            losses = []
            for name, param in model.named_parameters():
                buffer_name_param = name.replace('.', '_')
                estimated_mean = getattr(model, '{}_estimated_mean'.format(buffer_name_param))
                estimated_fisher = getattr(model, '{}_estimated_fisher'.format(buffer_name_param))
                regularization = (self.transformation(estimated_fisher) * (param - estimated_mean) ** 2).sum()
                losses.append(regularization)
            return 0.5 * sum(losses)
        except AttributeError:
            return 0

    def __call__(self, model, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        output = model(inputs)
        org_loss = self.criterion(output, targets)
        ewc_loss = self.elastic_loss(model)
        loss = org_loss + ewc_loss
        loss.backward()
        self.optimizer.step()
        summary = {
            'Loss': loss,
            'EWC_loss': ewc_loss,
            'OrgLoss': org_loss,
            'Accuracy': accuracy(output, targets),
        }
        return model, summary
