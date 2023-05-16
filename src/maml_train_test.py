#!/usr/bin/env python3

##############
# Parameters #
##############

import argparse
import json

#import numpy as np
import tqdm
from src.utils2 import Profiler
from src.zoo.trident_utils import inner_adapt_trident, setup

import os
import random

import numpy as np
import torch as th
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

parser = argparse.ArgumentParser()
parser.add_argument('--cnfg', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--n_ways', type=int)
parser.add_argument('--k_shots', type=int)
parser.add_argument('--q_shots', type=int)
parser.add_argument('--inner-adapt-steps-train', type=int)
parser.add_argument('--inner-adapt-steps-test', type=int)
parser.add_argument('--inner-lr', type=float)
parser.add_argument('--meta-lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--reconstr', type=str)
parser.add_argument('--wt-ce', type=float)
parser.add_argument('--klwt', type=str)
parser.add_argument('--rec-wt', type=float)
parser.add_argument('--beta-l', type=float)
parser.add_argument('--beta-s', type=float)
parser.add_argument('--zl', type=int, default=64)
parser.add_argument('--zs', type=int, default=64)
parser.add_argument('--wm-channels', type=int, default=64)
parser.add_argument('--wn-channels', type=int, default=32)
parser.add_argument('--task-adapt', type=str)
parser.add_argument('--experiment', type=str)
parser.add_argument('--order', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--download', type=str)


args = parser.parse_args()
with open(args.cnfg) as f:
    parser = argparse.ArgumentParser()
    argparse_dict = vars(args)
    argparse_dict.update(json.load(f))

    args = argparse.Namespace()
    args.__dict__.update(argparse_dict)


# TODO: fix this bool/str shit

if args.order == 'True':
    args.order = True
elif args.order == 'False':
    args.order = False

if args.download == 'True':
    args.download = True
elif args.download == 'False':
    args.download = False

if args.klwt == 'True':
    args.klwt = True
elif args.klwt == 'False':
    args.klwt = False

if args.task_adapt == 'True':
    args.task_adapt = True
elif args.task_adapt == 'False':
    args.task_adapt = False
args.device = 'cpu'

# Generating Tasks, initializing learners, loss, meta - optimizer and profilers
train_tasks, valid_tasks, test_tasks, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.order, args.inner_lr, args.device, download=args.download, task_adapt=args.task_adapt, args=args)
print("Phase 9")
opt = optim.Adam(learner.parameters(), args.meta_lr)
reconst_loss = nn.MSELoss(reduction='none')
start = 0

experiment_name = 'MAML_{}_{}-way_{}-shot_{}-queries_iter_{}_batch_{}'.format(args.dataset,args.n_ways, args.k_shots,
                                                                              args.q_shots, args.iterations,args.meta_batch_size)
if args.order == False:
    profiler = Profiler(experiment_name, args.experiment, args)

elif args.order == True:
    profiler = Profiler('FO-{}'.format(experiment_name), args.experiment, args)

save_dir = os.path.join(os.getcwd(), 'MAML_results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


res_dir = os.path.join(save_dir, experiment_name)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
print('results directory: {}'.format(res_dir))

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    print(np.shape(data))
    # Separate data into adaptation/evalutation sets
    adaptation_indices = th.zeros(data.size(0)).byte()
    # original: select the elements based on number of shots and ways, then seperate the training and validation here
    # now just separate in 80% train - 20 % val
    train_indices = random.sample(range(data.size(0)), int(data.size(0)*0.8))
    adaptation_indices[train_indices] = 1

    #adaptation_indices[th.arange(int(data.size(0)*0.8))] = 1
    #adaptation_indices[np.arange(shots*ways) * 2] = True

    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[1 - adaptation_indices], labels[1 - adaptation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=args.n_ways,
        shots=args.k_shots,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=args.meta_batch_size,
        adaptation_steps=1,
        num_iterations=args.iterations,
        cuda=False,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    device = th.device('cpu')
    if cuda and th.cuda.device_count():
        th.cuda.manual_seed(seed)
        device = th.device('cuda')

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    results_dict = {}
    results_dict['train_err'] = {}
    results_dict['train_acc'] = {}
    results_dict['valid_err'] = {}
    results_dict['valid_acc'] = {}
    results_dict['test_err'] = {}
    results_dict['test_acc'] = {}

    for iteration in range(num_iterations):

        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        # meta_test_error = 0.0
        # meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # # Compute meta-testing loss
            # learner = maml.clone()
            # batch = test_tasks.sample()
            # evaluation_error, evaluation_accuracy = fast_adapt(batch,
            #                                                    learner,
            #                                                    loss,
            #                                                    adaptation_steps,
            #                                                    shots,
            #                                                    ways,
            #                                                    device)
            # meta_test_error += evaluation_error.item()
            # meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        train_err = meta_train_error / meta_batch_size
        train_acc =  meta_train_accuracy / meta_batch_size
        valid_err = meta_valid_error / meta_batch_size
        valid_acc = meta_valid_accuracy / meta_batch_size
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', train_err)
        print('Meta Train Accuracy', train_acc)
        print('Meta Valid Error', valid_err)
        print('Meta Valid Accuracy', valid_acc)
        # print('Meta Test Error', meta_test_error / meta_batch_size)
        # print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        results_dict["train_err"][iteration] = train_err
        results_dict["train_acc"][iteration] = train_acc
        results_dict["valid_err"][iteration] = valid_err
        results_dict["valid_acc"][iteration] = valid_acc

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = test_tasks.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    test_err = meta_test_error / meta_batch_size
    test_acc = meta_test_accuracy / meta_batch_size

    print('Meta Test Error', test_err)
    print('Meta Test Accuracy', test_acc)
    results_dict["test_err"] = test_err
    results_dict["test_acc"] = test_acc

    with open(os.path.join(res_dir, 'results.json'), "w") as json_file:
        json.dump(results_dict, json_file, indent=4)

if __name__ == '__main__':
    main() 
