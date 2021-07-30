import os
import pprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import util
import data
import models


device = util.set_device()
print(device)


class EvalDataset(object):
    def __init__(self, model_name, data_path, structure, class_names, explain=False,
                 save_predictions=False, use_gpu=True, **params):
        self.weights_path = os.path.join('./trained_models', model_name, 'final.pt')  # 'checkpoint_30.pt'
        print('Loading model from: {}'.format(self.weights_path))
        self.model_name = model_name
        self.data = data.HDF5Dataset(data_dir=data_path, phase='test', cache_size=20)
        self.cuda = use_gpu
        self.explain = explain
        self.out_dir = os.path.join('./trained_models', model_name, 'results')
        self.class_names = class_names
        self.save_predictions = save_predictions
        self.model = self.get_model(structure, **params)
        util.mdir(self.out_dir)

    def get_model(self, structure, **params):
        model = getattr(models, str(structure))(**params)
        weights = torch.load(self.weights_path)
        weights = {k: v for k, v in weights.items() if k in model.state_dict()}
        model.load_state_dict(weights)
        if self.cuda:
            model.to(device)
        model.eval()
        return model

    def compute_metrics(self, predictions):
        results = dict(model_name=self.model_name, report=dict())

        all_labels = [ex['label'] for ex in predictions]
        all_preds = [ex['pred_class'] for ex in predictions]
        base_conf_matrix = confusion_matrix(all_labels, all_preds)
        pprint.pprint({'baseline_conf_matrix': base_conf_matrix.tolist()})

        for t in [0.5, 0.6, 0.7, 0.85, 0.9, 0.95]:
            filtered = []
            for ex in predictions:
                probabilities = ex['probabilities']
                predicted_as = ex['pred_class']
                if probabilities[predicted_as] >= t:
                    filtered.append(ex)

            labels = [ex['label'] for ex in filtered]
            preds = [ex['pred_class'] for ex in filtered]
            conf_matrix = confusion_matrix(labels, preds)

            meta = {
                'threshold_{}'.format(t): {
                    'availability': len(filtered) / len(predictions),
                    'classes': self.class_names,
                    'confusion_matrix': conf_matrix.tolist(),
                    'class_report': classification_report(labels,
                                                          preds, target_names=self.class_names,
                                                          labels=[0, 1, 2, 3],
                                                          output_dict=True)
                }
            }
            results['report'].update(meta)
        pprint.pprint(results)
        return results

    def evaluate(self):
        evaluated = []
        test = DataLoader(self.data, batch_size=32, num_workers=2)
        with torch.no_grad():
            for batch in tqdm(test):
                inputs, labels = batch
                if self.cuda:
                    inputs, labels = inputs.to(device), labels.to(device)
                scores = self.model(inputs)
                probabilities = F.softmax(scores, dim=1)
                _, predictions = torch.max(scores, dim=1)
                for i, ex in enumerate(inputs):
                    meta = {
                        'img': ex,
                        'probabilities': probabilities[i],
                        'pred_class': int(predictions[i]),
                        'label': int(labels[i])
                    }
                    if self.explain:
                        self.explanations()
                    evaluated.append(meta)
        results = self.compute_metrics(evaluated)
        util.save_json(os.path.join(self.out_dir, 'results.json'), results)
        if self.save_predictions:
            util.save(os.path.join(self.out_dir, 'imgs.pdata'), predictions)
        return predictions

    def explanations(self):
        raise NotImplementedError


class EvalSOTA(object):
    """
    Evaluate on a torch builtin dataset
    """
    def __init__(self, model_name, model, class_names, save_predictions=False, use_gpu=True):
        self.weights_path = os.path.join('./trained_models', model_name, 'final.pt')
        print('Loading model from: {}'.format(self.weights_path))
        self.model_name = model_name
        self.cuda = use_gpu
        self.out_dir = os.path.join('./trained_models', model_name, 'results')
        self.class_names = class_names
        self.save_predictions = save_predictions
        self.model = self.get_model(model)
        util.mdir(self.out_dir)

    def get_model(self, model):
        model = util.load_weights(model, self.weights_path)
        if self.cuda:
            model.to(device)
        model.eval()
        return model

    def compute_metrics(self, predictions):
        results = dict(model_name=self.model_name, report=dict())

        all_labels = [ex['label'] for ex in predictions]
        all_preds = [ex['pred_class'] for ex in predictions]
        base_conf_matrix = confusion_matrix(all_labels, all_preds)
        pprint.pprint({'baseline_conf_matrix': base_conf_matrix.tolist()})

        for t in [0.5, 0.6, 0.7, 0.85, 0.9, 0.95]:
            filtered = []
            for ex in predictions:
                probabilities = ex['probabilities']
                predicted_as = ex['pred_class']
                if probabilities[predicted_as] >= t:
                    filtered.append(ex)

            labels = [ex['label'] for ex in filtered]
            preds = [ex['pred_class'] for ex in filtered]
            conf_matrix = confusion_matrix(labels, preds)

            meta = {
                'threshold_{}'.format(t): {
                    'availability': len(filtered) / len(predictions),
                    'classes': self.class_names,
                    'confusion_matrix': conf_matrix.tolist(),
                    'class_report': classification_report(labels,
                                                          preds, target_names=self.class_names,
                                                          labels=range(len(self.class_names)),
                                                          output_dict=True)
                }
            }
            results['report'].update(meta)
        pprint.pprint(results)
        return results

    def evaluate(self, test):
        evaluated = []
        with torch.no_grad():
            for batch in tqdm(test):
                inputs, labels = batch
                if self.cuda:
                    inputs, labels = inputs.to(device), labels.to(device)
                scores = self.model(inputs)
                probabilities = F.softmax(scores, dim=1)
                _, predictions = torch.max(scores, dim=1)
                for i, ex in enumerate(inputs):
                    meta = {
                        'img': ex,
                        'probabilities': probabilities[i],
                        'pred_class': int(predictions[i]),
                        'label': int(labels[i])
                    }
                    evaluated.append(meta)
        results = self.compute_metrics(evaluated)
        util.save_json(os.path.join(self.out_dir, 'results.json'), results)
        if self.save_predictions:
            util.save(os.path.join(self.out_dir, 'imgs.pdata'), predictions)
        return results


def clean_stl_evaluation(experiment, model='densenet'):
    # data_path = '/media/miguel/ALICIUM/Miguel/DATASETS/RESEARCH_EWC/PGD_samples/all_adversarial_set'
    # test_data = data.HDF5Dataset(data_dir=data_path, phase='test', cache_size=10,
    #                              transform=data.process_adversaries(False))
    _, test_data, class_names = data.get_stl_datasets()
    test = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)
    model = models.SOTANetwork(architecture=model, in_channels=3, num_outputs=10)
    evaluator = EvalSOTA(model_name='STL-10/{}'.format(experiment),
                         model=model,
                         class_names=class_names,
                         save_predictions=False,
                         use_gpu=True)
    results = evaluator.evaluate(test)
    return results


if __name__ == '__main__':

    # STL clean evaluations
    # clean_stl_evaluation('202104013_baseline_STL_densenet') # 0.945713
    # clean_stl_evaluation('20210522_adversarial_STL_densenet') # 0.914084
    # clean_stl_evaluation('20210522_adversarial_mixed_STL_densenet') # 0.93299
    # clean_stl_evaluation('2021_adversarial_ewc_online') # 0.9236957888120679,

    pass
