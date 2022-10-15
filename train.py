import os
import copy
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score

from src.i3dpt import Unit3Dpy
from utils.temporal_transforms import TemporalRandomCrop
from utils.utils import *
from src.i3dpt import I3D
from DataLoader import RGBFlowDataset
from opts import parser
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator


def load_and_freeze_model(model, weight_path, num_classes, num_freeze=15):
    model.load_state_dict(torch.load(weight_path))
    log("Pre-trained model {} loaded successfully!".format(weight_path))
    counter = 0
    for child in model.children():
        counter += 1
        if counter < num_freeze:
            log("Layer {} frozen!".format(child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
    model.num_classes = num_classes
    model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                   out_channels=num_classes,
                                   kernel_size=(1, 1, 1),
                                   activation=None,
                                   use_bias=True,
                                   use_bn=False)
    model.to(device)


class Estimator(BaseEstimator):

    def __init__(self, dropout_prob=0, lr=0.001):
        super(Estimator, self).__init__()
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.weight_paths = {"rgb": args.rgb_weights_path, "flow": args.flow_weights_path}
        self.optimizers = {}
        self.exp_lr_schedulers = {}

        # Here we must use 400 num_class because we have to load the weight from original file. We change it later.
        self.models = {"rgb": I3D(num_classes=400, modality='rgb', dropout_prob=dropout_prob),
                  "flow": I3D(num_classes=400, modality='flow', dropout_prob=dropout_prob)}

        for stream in streams:
            load_and_freeze_model(model=self.models[stream], num_classes=len(class_names), weight_path=self.weight_paths[stream])
            self.optimizers[stream] = optim.SGD(filter(lambda p: p.requires_grad, self.models[stream].parameters()), lr=lr,
                                           momentum=0.9)
            self.exp_lr_schedulers[stream] = lr_scheduler.StepLR(self.optimizers[stream], step_size=7, gamma=0.1)

        self.criterion = nn.CrossEntropyLoss()

    def fit(self, data_paths):
        rgb_flow_dataset = RGBFlowDataset(data_paths, class_dicts,
                                           sample_rate=args.sample_num,
                                           sample_type=args.sample_type,
                                           fps=args.out_fps,
                                           out_frame_num=args.out_frame_num,
                                           augment=True,
                                           output_size=args.output_size)
        data_loader = torch.utils.data.DataLoader(rgb_flow_dataset, batch_size=8,
                                                       shuffle=True, num_workers=0)
        dataset_size = len(rgb_flow_dataset)

        num_epochs = 50

        best_model_wts = {}
        for stream in streams:
            best_model_wts[stream] = copy.deepcopy(self.models[stream].state_dict())

        best_accs = {stream: 0.0 for stream in streams}
        best_accs['composed'] = 0.0
        best_f1s = {stream: 0.0 for stream in streams}
        best_f1s['composed'] = 0.0

        log(f'---param {self.dropout_prob} {self.lr}---')
        for epoch in range(num_epochs):
            log('Epoch {}/{}'.format(epoch, num_epochs - 1))
            log('-' * 10)

            [i.train() for i in self.models.values()]  # Set model to training mode
            running_losses = {"rgb": 0.0, "flow": 0.0, "composed": 0.0}
            running_corrects = {"rgb": 0, "flow": 0, "composed": 0}
            running_f1s = {"rgb": [], "flow": [], "composed": []}

            # Iterate over data.
            data = {}
            for data["rgb"], data["flow"], labels in data_loader:
                for stream in streams:
                    data[stream] = data[stream].to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                out_logits = {}
                losses = {}
                with torch.set_grad_enabled(True):
                    # Calculate the joint output of two model
                    for stream in streams:
                        _, out_logits[stream] = self.models[stream](data[stream])
                        out_softmax = torch.nn.functional.softmax(out_logits[stream], 1)
                        _, preds = torch.max(out_softmax.data.cpu(), 1)
                        losses[stream] = self.criterion(out_softmax.cpu(), labels.cpu())

                        # backward + optimize only if in training phase
                        losses[stream].backward()
                        self.optimizers[stream].step()

                        # statistics
                        running_losses[stream] += losses[stream].item() * data[stream].shape[0]
                        running_corrects[stream] += torch.sum(preds == labels.data.cpu())
                        running_f1s[stream].append(
                            f1_score(labels.data.cpu().numpy(), preds.numpy(), average='macro'))

                    out_logits["composed"] = out_logits["rgb"] + out_logits["flow"]
                    out_softmax = torch.nn.functional.softmax(out_logits["composed"], 1)
                    _, preds = torch.max(out_softmax.data.cpu(), 1)
                    running_corrects["composed"] += torch.sum(preds == labels.data.cpu())
                    running_f1s["composed"].append(f1_score(labels.data.cpu().numpy(), preds.numpy()))

            for scheduler in self.exp_lr_schedulers.values():
                scheduler.step()

            epoch_losses = {}
            epoch_accs = {}
            epoch_f1s = {}
            for stream in losses.keys():
                epoch_losses[stream] = running_losses[stream] / dataset_size
                epoch_accs[stream] = running_corrects[stream].double() / dataset_size
                epoch_f1s[stream] = np.average(running_f1s[stream])
                log('Loss ({}): {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                    stream, epoch_losses[stream], epoch_accs[stream], epoch_f1s[stream]))
                best_accs[stream] = max(best_accs[stream], epoch_accs[stream])
                best_f1s[stream] = max(best_f1s[stream], epoch_f1s[stream])
            best_accs['composed'] = max(best_accs['composed'], epoch_accs[stream])
            best_f1s['composed'] = max(best_f1s['composed'], epoch_f1s[stream])

        for stream in streams:
            log('Best val Acc({}): {:4f}'.format(stream, best_accs[stream]))
            log('Best val F1({}): {:4f}'.format(stream, best_f1s[stream]))
        log('Best val Acc({}): {:4f}'.format('composed', best_accs['composed']))
        log('Best val F1({}): {:4f}'.format('composed', best_f1s['composed']))

        # load best model weights
        for stream in streams:
            self.models[stream].load_state_dict(best_model_wts[stream])
            temp_model_path = 'model/{}_{}_model_{}.pth'.format(self.dropout_prob, self.lr, stream)
            torch.save(self.models[stream].state_dict(), temp_model_path)

    def score(self, data_paths):
        rgb_flow_dataset = RGBFlowDataset(data_paths, class_dicts,
                                           sample_rate=args.sample_num,
                                           sample_type=args.sample_type,
                                           fps=args.out_fps,
                                           out_frame_num=args.out_frame_num,
                                           augment=False,
                                           output_size=args.output_size)
        data_loader = torch.utils.data.DataLoader(rgb_flow_dataset, batch_size=8,
                                                       shuffle=True, num_workers=0)

        running_f1s = []
        data = {}
        for data["rgb"], data["flow"], labels in data_loader:
            for stream in streams:
                data[stream] = data[stream].to(device)
            labels = labels.to(device)

            out_logits = {}
            with torch.set_grad_enabled(False):
                # Calculate the joint output of two model
                for stream in streams:
                    _, out_logits[stream] = self.models[stream](data[stream])

                out_logits["composed"] = out_logits["rgb"] + out_logits["flow"]
                out_softmax = torch.nn.functional.softmax(out_logits["composed"], 1)
                _, preds = torch.max(out_softmax.data.cpu(), 1)
                running_f1s.append(f1_score(labels.data.cpu().numpy(), preds.numpy()))

        return np.average(running_f1s)


def model_selection(dropout_prob=0, lr=0.001):
    return cross_val_score(
        Estimator(dropout_prob=dropout_prob, lr=lr),
        data_paths, cv=4
    ).mean()


if __name__ == "__main__":
    args = parser.parse_args()
    class_names = [i.strip() for i in open(args.classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}
    data_dir = Path('data/videos/pre-processed')
    streams = ["rgb", "flow"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_paths = []
    for class_name in class_names:
        for video in os.listdir(data_dir / 'train' / class_name):
            data_paths.append((class_name, video, data_dir / 'train' / class_name / video))
    print(f'load {len(data_paths)} samples.')
    np.random.shuffle(data_paths)

    bayes_optimizer = BayesianOptimization(model_selection, {
        'dropout_prob': (0, 0.8),
        'lr': (0.0001, 0.01),
    })
    bayes_optimizer.maximize()

    print(bayes_optimizer.max)
