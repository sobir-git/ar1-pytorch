#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020. Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo           #
# Pellegrini, Davide Maltoni. All rights reserved.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2020                                                             #
# Authors: Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo Pellegrini, Davide   #
# Maltoni.                                                                     #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Simple AR1* implementation in PyTorch with Latent Replay """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from data_loader import CORE50
import copy
import os
import json
from utils import *
import configparser
import argparse
from pprint import pprint
from models.mobilenet import MyMobilenetV1
from torch.utils.tensorboard import SummaryWriter

# --------------------------------- Setup --------------------------------------

# recover exp configuration name
parser = argparse.ArgumentParser(description='Run CL experiments')
parser.add_argument('--name', dest='exp_name', default='DEFAULT',
                    help='name of the experiment you want to run.')
parser.add_argument('--datadir', default='./core50',
                    help='Folder where core50 dataset exists: sessions folders, paths.pkl, LUP.pkl, labels.pkl')
parser.add_argument('--logdir', default='../logs')
parser.add_argument('--config', default='./params.cfg', help='Configuration file containing hyperparameters, etc.')

args = parser.parse_args()

# set cuda device (based on your hardware)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# recover config file for the experiment
config = configparser.ConfigParser()
config.read(args.config)
exp_config = config[args.exp_name]
print("Experiment name:", args.exp_name)
pprint(dict(exp_config))

# recover parameters from the cfg file and compute the dependent ones.
exp_name = eval(exp_config['exp_name'])
comment = eval(exp_config['comment'])
use_cuda = eval(exp_config['use_cuda'])
init_lr = eval(exp_config['init_lr'])
inc_lr = eval(exp_config['inc_lr'])
mb_size = eval(exp_config['mb_size'])
init_train_ep = eval(exp_config['init_train_ep'])  # training epochs for the initial batch
inc_train_ep = eval(exp_config['inc_train_ep'])  # training epochs for incremental batches
init_update_rate = eval(exp_config['init_update_rate'])
inc_update_rate = eval(exp_config['inc_update_rate'])
max_r_max = eval(exp_config['max_r_max'])
max_d_max = eval(exp_config['max_d_max'])
inc_step = eval(exp_config['inc_step'])
rm_sz = eval(exp_config['rm_sz'])
momentum = eval(exp_config['momentum'])
l2 = eval(exp_config['l2'])
freeze_below_layer = eval(exp_config['freeze_below_layer'])
latent_layer_num = eval(exp_config['latent_layer_num'])
reg_lambda = eval(exp_config['reg_lambda'])

# setting up log dir for tensorboard
log_dir = args.logdir + '/' + exp_name
writer = SummaryWriter(log_dir)

# Saving params
hyper = json.dumps(dict(exp_config))
writer.add_text("parameters", hyper, 0)

# Other variables init
tot_it_step = 0
rm = None

# Create the dataset object
dataset = CORE50(root=args.datadir, scenario=exp_config['scenario'])
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()

# Model setup
model = MyMobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
# we replace BN layers with Batch Renormalization layers
replace_bn_with_brn(
    model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
    max_r_max=max_r_max, max_d_max=max_d_max
)
model.saved_weights = {}
model.past_j = {i: 0 for i in range(50)}  # number of patterns of each class seen in the past
model.cur_j = {i: 0 for i in range(50)}  # number of patterns of each class in the current batch

ewcData, synData = None, None
if reg_lambda != 0:
    # the regularization is based on Synaptic Intelligence as described in the
    # paper. ewcData is a list of two elements (best parametes, importance)
    # while synData is a dictionary with all the trajectory data needed by SI
    ewcData, synData = create_syn_data(model)

# Optimizer setup
optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
)
criterion = torch.nn.CrossEntropyLoss()

# --------------------------------- Training -----------------------------------

# we freeze the layer below the replay layer since the first batch
freeze_up_to(model, freeze_below_layer, only_conv=False)

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    if reg_lambda != 0:
        init_batch(model, ewcData, synData)

    if i == 1:
        change_brn_pars(
            model, momentum=inc_update_rate, r_d_max_inc_step=0,
            r_max=max_r_max, d_max=max_d_max)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
        )

    train_x, train_y = train_batch

    cur_classes = None
    if i == 0:
        cur_classes = [int(o) for o in set(train_y)]
        model.cur_j = examples_per_class(train_y)
    else:
        cur_classes = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]))

    print("----------- batch {0} -------------".format(i))
    print("train_x shape: {}, train_y shape: {}"
          .format(train_x.shape, train_y.shape))

    model.train()
    model.lat_features.eval()

    # load the weights for the seen classes, reset others to zero
    reset_weights(model, cur_classes)

    # for the first batch, pad training set to be multiple of mini-batch size
    it_x_ep = None
    if i == 0:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = AverageMeter()
    ave_loss = 0
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if i == 0:
        train_ep = init_train_ep
    else:
        train_ep = inc_train_ep

    chosen_latent_acts = None  # variable used to gather latent activations
    chosen_y = None

    # maximum number of latent patterns to gather for saving in memory
    h = rm_sz // (i + 1)

    # loop through current batch multiple epochs
    # each epoch runs once though the training set + replay memory (if non-empty)
    for ep in range(train_ep):

        # index used to shuffle training set
        shuffle_idx = np.random.permutation(len(train_x))

        print("training ep: ", ep)
        ave_loss = AverageMeter()  # compute average loss for this epoch
        train_sz = train_x.shape[0]

        # computing how many patterns to inject in the latent replay layer
        if i > 0:
            # a minibatch consists of patterns from the current batch and replay memory
            cur_sz = train_sz // ((train_sz + rm_sz) // mb_size)  # number of patterns from current batch
            it_x_ep = train_sz // cur_sz  # number of iterations to complete the batch
            n2inject = max(0, mb_size - cur_sz)  # number of patterns from replay memory which will be injected at the latent layer
        else:
            n2inject = 0   # in the initial batch, the replay memory is empty
        print("total sz:", train_sz + rm_sz)
        print("n2inject", n2inject)
        print("it x ep: ", it_x_ep)

        for it in range(it_x_ep):

            if reg_lambda != 0:
                pre_update(model, synData)

            # compute start and end indices of the training set that go into the current mini-batch
            start = it * (mb_size - n2inject)
            end = (it + 1) * (mb_size - n2inject)
            # construct the initial mini-batch from training set
            x_mb = train_x[shuffle_idx[start:end]].astype(np.float32)
            x_mb = preproc(x_mb)
            # convert train data to torch tensors
            x_mb = torch.from_numpy(x_mb).type(torch.FloatTensor)

            x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)

            if i == 0:  # if the initial batch, then no latent patterns
                lat_mb_x = None
                y_mb = maybe_cuda(train_y[shuffle_idx[start:end]], use_cuda=use_cuda)

            else:  # otherwise get latent replay patterns from memory
                lat_mb_x = rm[0][it * n2inject: (it + 1) * n2inject]
                lat_mb_y = rm[1][it * n2inject: (it + 1) * n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[shuffle_idx[start:end]], lat_mb_y), 0),
                    use_cuda=use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

            # if lat_mb_x is not None, this tensor will be concatenated in
            # the forward pass on-the-fly in the latent replay layer
            logits, lat_acts = model(
                x_mb, latent_input=lat_mb_x, return_lat_acts=True)

            # collect latent volumes only for the first epoch
            # we need to store them to eventually add them into the external
            # replay memory
            if ep == 0:  #
                if it == 0:
                    this_h = min(h, len(lat_acts))
                    chosen_latent_acts = lat_acts[:this_h].cpu().detach()
                    chosen_y = y_mb[:this_h].cpu()
                elif chosen_latent_acts.size(0) < h:
                    maxtake = min(len(lat_acts), h - chosen_latent_acts.size(0))
                    if maxtake > 0:
                        chosen_latent_acts = torch.cat((chosen_latent_acts, lat_acts[:maxtake].cpu().detach()), 0)
                        chosen_y = torch.cat([chosen_y, y_mb[:maxtake].cpu()], 0)

            pred_label = torch.argmax(logits, 1)
            acc.update(torch.eq(pred_label, y_mb).type(torch.FloatTensor).mean().item(), len(y_mb))

            loss = criterion(logits, y_mb)
            if reg_lambda != 0:
                loss += compute_ewc_loss(model, ewcData, lambd=reg_lambda)
            ave_loss.update(loss.item(), len(y_mb))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if reg_lambda != 0:
                post_update(model, synData)

            if it % 10 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss.avg, acc.avg)
                )

            # Log scalar values (scalar summary) to TB
            tot_it_step += 1
            writer.add_scalar('train_loss', ave_loss.avg, tot_it_step)
            writer.add_scalar('train_accuracy', acc.avg, tot_it_step)

    # at the end of the batch, consolidate weights
    consolidate_weights(model, cur_classes)
    if reg_lambda != 0:
        update_ewc_data(model, ewcData, synData, 0.001, 1)

    # how many patterns to save for next iter
    # replay memory should contain equal number of patterns (h) from each batch
    h = min(h, len(chosen_latent_acts))
    print("h", h)
    print("cur_acts sz:", chosen_latent_acts.size(0))
    assert len(chosen_latent_acts) <= h
    assert len(chosen_latent_acts) == len(chosen_y), (len(chosen_latent_acts), len(chosen_y))

    rm_add = [chosen_latent_acts, chosen_y]
    print("rm_add size", rm_add[0].size(0))

    # replace patterns in random memory
    if i == 0:  # if initial batch, just copy chosen activations, nothing to replace
        rm = rm_add
    else:  # choose h patterns from the replay memory, which will be replaced with current patterns
        idxs_2_replace = np.random.choice(
            rm[0].size(0), h, replace=False
        )
        for j, idx in enumerate(idxs_2_replace):
            rm[0][idx] = rm_add[0][j]
            rm[1][idx] = rm_add[1][j]

    set_consolidate_weights(model)
    # test the model
    test_avg_loss, test_acc, _ = get_accuracy(
        model, criterion, mb_size, test_x, test_y, preproc=preproc
    )

    # Log scalar values (scalar summary) to TB
    writer.add_scalar('test_loss', test_avg_loss, i)
    writer.add_scalar('test_accuracy', test_acc, i)

    # update number examples encountered over time
    for c, n in model.cur_j.items():
        model.past_j[c] += n

    print("---------------------------------")
    print("Accuracy: ", test_acc)
    print("---------------------------------")

writer.close()
