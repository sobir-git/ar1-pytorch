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
from utils import *
import argparse
from pprint import pprint
from models.mobilenet import MyMobilenetV1
import wandb
import yaml

# --------------------------------- Setup --------------------------------------

# recover exp configuration name
parser = argparse.ArgumentParser(description='Run CL experiments')
parser.add_argument('--name', dest='exp_name', default='DEFAULT',
                    help='name of the experiment you want to run.')
parser.add_argument('--datadir', default='./core50',
                    help='Folder where core50 dataset exists: sessions folders, paths.pkl, LUP.pkl, labels.pkl')
parser.add_argument('--logdir', default='../logs')
parser.add_argument('--config', default='./config-nic.yaml', help='Configuration file containing hyperparameters, etc.')
args = parser.parse_args()

# recover config file for the experiment
with open(args.config) as f:
    config = yaml.safe_load(f)

logger = get_console_logger('main')
logger.info("Experiment name: %s", args.exp_name)
pprint(dict(config))

wandb.init(project='ar1', config=config)
config = wandb.config

# Other variables init
tot_it_step = 0
rm = None

# Create the dataset object
dataset = CORE50(root=args.datadir, scenario=config['scenario'])
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()

# Model setup
model = MyMobilenetV1(pretrained=True, latent_layer_num=config.latent_layer_num)

# we replace BN layers with Batch Renormalization layers
if config['scenario'].startswith('nic'):
    replace_bn_with_brn(
        model, momentum=config.init_update_rate, r_d_max_inc_step=config.inc_step,
        max_r_max=config.max_r_max, max_d_max=config.max_d_max
    )
model.saved_weights = {}
model.past_j = {i: 0 for i in range(50)}  # number of patterns of each class seen in the past
model.cur_j = {i: 0 for i in range(50)}  # number of patterns of each class in the current batch

ewcData, synData = None, None
if config.reg_lambda != 0:
    # the regularization is based on Synaptic Intelligence as described in the
    # paper. ewcData is a list of two elements (best parametes, importance)
    # while synData is a dictionary with all the trajectory data needed by SI
    ewcData, synData = create_syn_data(model)

# Optimizer setup
optimizer = torch.optim.SGD(
    model.parameters(), lr=config.init_lr, momentum=config.momentum, weight_decay=config.l2
)
criterion = torch.nn.CrossEntropyLoss()

# --------------------------------- Training -----------------------------------

# we freeze the layer below the replay layer since the first batch
freeze_up_to(model, config.freeze_below_layer, only_conv=False)

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    if config.reg_lambda != 0:
        init_batch(model, ewcData, synData)

    if i == 1:
        if config['scenario'].startswith('nic'):
            change_brn_pars(
                model, momentum=config.inc_update_rate, r_d_max_inc_step=0,
                r_max=config.max_r_max, d_max=config.max_d_max)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.inc_lr, momentum=config.momentum, weight_decay=config.l2
        )

    train_x, train_y = train_batch

    cur_classes = None
    if i == 0:
        cur_classes = [int(o) for o in set(train_y)]
        model.cur_j = examples_per_class(train_y)
    else:
        cur_classes = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]))

    logger.info("----------- batch {0} -------------".format(i))
    logger.info("train_x shape: {}, train_y shape: {}"
                .format(train_x.shape, train_y.shape))

    model.train()
    model.lat_features.eval()

    # load the weights for the seen classes, reset others to zero
    reset_weights(model, cur_classes)

    # for the first batch, pad training set to be multiple of mini-batch size
    it_x_ep = None
    if i == 0:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], config.mb_size)

    model = maybe_cuda(model, use_cuda=config.use_cuda)
    acc = AverageMeter()
    ave_loss = 0
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if i == 0:
        train_ep = config.init_train_ep
    else:
        train_ep = config.inc_train_ep

    chosen_latent_acts = None  # variable used to gather latent activations
    chosen_y = None

    # maximum number of latent patterns to gather for saving in memory
    h = config.rm_sz // (i + 1)

    # loop through current batch multiple epochs
    # each epoch runs once though the training set + replay memory (if non-empty)
    for ep in range(train_ep):

        # index used to shuffle training set
        shuffle_idx = np.random.permutation(len(train_x))

        logger.info("training ep: %s", ep)
        ave_loss = AverageMeter()  # compute average loss for this epoch
        train_sz = train_x.shape[0]

        # computing how many patterns to inject in the latent replay layer
        if i > 0:
            # a minibatch consists of patterns from the current batch and replay memory
            cur_sz = train_sz // ((train_sz + config.rm_sz) // config.mb_size)  # number of patterns from current batch
            it_x_ep = train_sz // cur_sz  # number of iterations to complete the batch
            n2inject = max(0,
                           config.mb_size - cur_sz)  # number of patterns from replay memory which will be injected at the latent layer
        else:
            n2inject = 0  # in the initial batch, the replay memory is empty
        logger.info("total sz: %s", train_sz + config.rm_sz)
        logger.info("n2inject: %s", n2inject)
        logger.info("it x ep: %s", it_x_ep)

        for it in range(it_x_ep):

            if config.reg_lambda != 0:
                pre_update(model, synData)

            # compute start and end indices of the training set that go into the current mini-batch
            start = it * (config.mb_size - n2inject)
            end = (it + 1) * (config.mb_size - n2inject)
            # construct the initial mini-batch from training set
            x_mb = train_x[shuffle_idx[start:end]].astype(np.float32)
            x_mb = preproc(x_mb)
            # convert train data to torch tensors
            x_mb = torch.from_numpy(x_mb).type(torch.FloatTensor)

            x_mb = maybe_cuda(x_mb, use_cuda=config.use_cuda)

            if i == 0:  # if the initial batch, then no latent patterns
                lat_mb_x = None
                y_mb = maybe_cuda(train_y[shuffle_idx[start:end]], use_cuda=config.use_cuda)

            else:  # otherwise get latent replay patterns from memory
                lat_mb_x = rm[0][it * n2inject: (it + 1) * n2inject]
                lat_mb_y = rm[1][it * n2inject: (it + 1) * n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[shuffle_idx[start:end]], lat_mb_y), 0),
                    use_cuda=config.use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=config.use_cuda)

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
            if config.reg_lambda != 0:
                loss += compute_ewc_loss(model, ewcData, lambd=config.reg_lambda)
            ave_loss.update(loss.item(), len(y_mb))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config.reg_lambda != 0:
                post_update(model, synData)

            if it % 10 == 0:
                logger.info(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss.avg, acc.avg)
                )

            # Log scalar values (scalar summary) to TB
            tot_it_step += 1
            wandb.log({'train_loss': ave_loss.avg, 'train_accuracy': acc.avg}, step=tot_it_step)

    # at the end of the batch, consolidate weights
    consolidate_weights(model, cur_classes)
    if config.reg_lambda != 0:
        update_ewc_data(model, ewcData, synData, 0.001, 1)

    # how many patterns to save for next iter
    # replay memory should contain equal number of patterns (h) from each batch
    h = min(h, len(chosen_latent_acts))
    logger.info("h: %s", h)
    logger.info("cur_acts sz: %s", chosen_latent_acts.size(0))
    assert len(chosen_latent_acts) <= h
    assert len(chosen_latent_acts) == len(chosen_y), (len(chosen_latent_acts), len(chosen_y))

    rm_add = [chosen_latent_acts, chosen_y]
    logger.info("rm_add size: %s", rm_add[0].size(0))

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
        model, criterion, config.mb_size, test_x, test_y, preproc=preproc
    )

    # Log scalar values (scalar summary) to TB
    wandb.log({'test_loss': test_avg_loss, 'test_accuracy': test_acc}, step=i)

    # update number examples encountered over time
    for c, n in model.cur_j.items():
        model.past_j[c] += n

    logger.info("---------------------------------")
    logger.info("Accuracy: %s", test_acc)
    logger.info("---------------------------------")
