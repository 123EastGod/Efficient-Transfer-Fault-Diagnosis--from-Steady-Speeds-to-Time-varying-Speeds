#!/usr/bin/python
# -*- coding:utf-8 -*-
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim
import models
import datasets
from loss.DAN import DAN
from loss.mmd_cauthy import mmd_cauthy_noaccelerate
class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        args = self.args
        # Using GPU/CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Select datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
           args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)


        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(False if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # Select model
        self.model = getattr(models, args.model_name)(args.pretrained)
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)

            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)
        # Select parameters
        if args.bottleneck:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                              {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]
        else:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]


        # Select optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")


        # Selcet learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        else:
            raise Exception("Please enter right keyword ")


        self.start_epoch = 0
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        self.classifier_layer.to(self.device)


        # Select the type of MMD
        if args.distance_metric:
            if args.distance_loss == 'MK-MMD':
                self.distance_loss = DAN
            elif args.distance_loss == "Cauthy":
                self.distance_loss = mmd_cauthy_noaccelerate
            else:
                raise Exception("loss not implement")
        else:
            self.distance_loss = None

        #Select Cross-Entropy
        self.criterion = nn.CrossEntropyLoss()


    def train(self,d2 = np.array([], dtype=np.float), d = np.array([], dtype=np.float)):
        #Initation
        args = self.args
        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()
        iter_num = 0
        #Start training
        for epoch in range(self.start_epoch, args.max_epoch):
            #Log
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))
            iter_target = iter(self.dataloaders['target_train'])# Load target samples
            len_target_loader = len(self.dataloaders['target_train'])# Number of target samples
            # Training phase & val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Init
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0
                # Define train/test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    self.classifier_layer.eval()
                #Each batch
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train' or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, _ = iter_target.next()
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # Forward
                        features = self.model(inputs)
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)
                            # Calculate MMD
                            if self.distance_loss is not None:
                                if args.distance_loss == 'MK-MMD':
                                    distance_loss = self.distance_loss(features.narrow(0, 0, labels.size(0)),
                                                                       features.narrow(0, labels.size(0), inputs.size(0)-labels.size(0)))
                                elif args.distance_loss == 'Cauthy':
                                    distance_loss = self.distance_loss(features.narrow(0, 0, labels.size(0)),
                                                                       features.narrow(0, labels.size(0),
                                                                                       inputs.size(0) - labels.size(0)))
                                else:
                                    raise Exception("loss not exist")

                            else:
                                distance_loss = 0

                            # Calculate the balance factor
                            if args.trade_off_distance == 'Cons':
                                lam_distance = args.lam_distance
                            elif args.trade_off_distance == 'Step':
                                if epoch<args.middle_epoch:
                                    lam_distance=0
                                else:
                                    lam_distance = -4 / (1 + math.sqrt(epoch/args.max_epoch) ) + 4
                            else:
                                raise Exception("balance factor not exist")
                            # if batch_idx<=7 and epoch>40:
                            #     nd = np.array(distance_loss.item())
                            #     d2=np.append(d2,nd)
                            #     if d2.size==9:
                            #         d2/=9
                            #         d=np.append(d,d2.sum())
                            #         d2 = np.array([], dtype=np.float)
                            # if d.size==300:
                            #     x = np.linspace(0, d.size, d.size)
                            #     plt.plot(x,d)
                            #     plt.tick_params(labelsize=20)  # 刻度
                            #     plt.xlim(0, 300)
                            #     #io.savemat(r'D://target_val.mat', {'A': d})
                            #     plt.xlabel("Epochs",fontsize=30)  # xlabel 设置 x 轴的标签名称
                            #    # plt.ylabel("MMD",fontsize=30)  # ylabel 设置 y 轴的标签名称
                            #     plt.show()# TODO MMD图
                            #Calculate total loss
                            loss = classifier_loss + lam_distance * distance_loss
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training log
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)
                            # Print log
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))
                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    # save the best model according to the val accuracy
                    if (epoch_acc > best_acc or epoch > args.max_epoch-2) and (epoch > args.middle_epoch-1):
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()














