#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import *
from torchvision import transforms
import numpy as np
import IPython
import copy
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import time
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(
            self, dataset, idxs, idx=0, noniid=False, noniid_prob=1.0, xshift_type="rot"
    ):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.idx = idx
        self.noniid = noniid
        self.classes = self.dataset.classes
        self.targets = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate(object):
    def __init__(
            self,
            args,
            dataset,
            idx,
            idxs,
            logger=None,
            test_dataset=None,
            memory_dataset=None,
            output_dir="",
    ):
        self.args = args
        self.logger = logger
        self.id = idx  # user id
        self.idxs = idxs  # dataset id
        self.reg_scale = args.reg_scale
        self.output_dir = output_dir

        if dataset is not None:
            self.trainloader, self.validloader, self.testloader = self.train_val_test(
                dataset, list(idxs), test_dataset, memory_dataset
            )

        self.device = "cuda" if args.gpu else "cpu"
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def get_model(self):
        return self.model

    def init_dataset(self, dataset):
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs), test_dataset, memory_dataset
        )

    def init_model(self, model):
        """Initialize local models"""
        train_lr = self.args.lr
        self.model = model

        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_lr, weight_decay=1e-6
            )

        total_epochs = self.args.local_ep * self.args.epochs
        self.schedule = [
            int(total_epochs * 0.3),
            int(total_epochs * 0.6),
            int(total_epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.schedule, gamma=0.3
        )
        self.scheduler = scheduler
        self.optimizer = optimizer

    def train_val_test(self, dataset, idxs, test_dataset=None, memory_dataset=None):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes. split indexes for train, validation, and test (80, 10, 10)
        """
        idxs_train = idxs[: int(0.8 * len(idxs))]
        self.idxs_train = idxs_train
        idxs_val = idxs[int(0.8 * len(idxs)): int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        train_dataset = DatasetSplit(
            dataset,
            idxs_train,
            idx=self.id,
        )

        if not self.args.distributed_training:
            trainloader = DataLoader(
                train_dataset,
                batch_size=self.args.local_bs,
                shuffle=True,
                num_workers=16,
                pin_memory=True,
                drop_last=True if len(train_dataset) > self.args.local_bs else False,
            )
        else:
            self.dist_sampler = DistributedSampler(train_dataset, shuffle=True)
            trainloader = DataLoader(
                train_dataset,
                sampler=self.dist_sampler,
                batch_size=self.args.local_bs,
                num_workers=16,
                pin_memory=True,
                drop_last=True,
            )

        validloader = DataLoader(
            DatasetSplit(
                dataset,
                idxs_val,
                idx=self.id,
            ),
            batch_size=self.args.local_bs,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        testloader = DataLoader(
            DatasetSplit(
                dataset,
                idxs_test,
                idx=self.id,
            ),
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        if test_dataset is not None:
            # such that the memory loader is the original dataset without pair augmentation
            memoryloader = DataLoader(
                DatasetSplit(
                    memory_dataset,
                    idxs_train,
                    idx=self.id,
                ),
                batch_size=64,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )

        else:
            memoryloader = DataLoader(
                DatasetSplit(
                    dataset,
                    idxs_train,
                    idx=self.id,
                ),
                batch_size=self.args.local_bs,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )

        self.memory_loader = memoryloader
        self.test_loader = testloader

        return trainloader, validloader, testloader

    def update_fc_weights(self, model, global_round, train_dataset=None):
        """Train the linear layer with the encode frozen"""
        model.train()
        epoch_loss = []
        if train_dataset is not None:
            trainloader = DataLoader(
                train_dataset,
                batch_size=self.args.local_bs,
                shuffle=True,
                num_workers=16,
                pin_memory=True,
            )
        else:
            trainloader = self.trainloader

        # only adam
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
        for param in model.f.parameters():
            param.requires_grad = False

        for iter in range(int(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "Update FC || User : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            self.id,
                            global_round,
                            self.args.local_ep * global_round + iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                if self.logger is not None:
                    self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_ssl_weights(
            self,
            model,
            global_round,
            additionl_feature_banks=None,
            lr=None,
            epoch_num=None,
            vis_feature=False,
    ):
        """Train the local model with self-superivsed learning"""
        epoch_loss = [0]
        global_model_copy = copy.deepcopy(model)
        global_model_copy.eval()

        # Set optimizer for the local updates
        train_epoch = epoch_num if epoch_num is not None else self.args.local_ep

        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)

            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size

            train_lr = lr if lr is not None else train_lr
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        elif self.args.optimizer == "adam":
            train_lr = lr if lr is not None else self.args.lr
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_lr, weight_decay=1e-6
            )

        if self.args.ssl_method == "mae":
            train_lr = lr if lr is not None else self.args.lr
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=train_lr, weight_decay=0.05
            )

        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())

        schedule = [
            int(self.args.local_ep * self.args.epochs * 0.3),
            int(self.args.local_ep * self.args.epochs * 0.6),
            int(self.args.local_ep * self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )
        global_step = 0
        max_steps = len(self.trainloader) * self.args.local_ep
        if additionl_feature_banks is not None:
            # hack: append the global model features to target for later usage
            targets = (
                np.array(self.trainloader.dataset.dataset.targets).reshape(-1).copy()
            )
            self.trainloader.dataset.dataset.target_copy = targets.copy()
            self.trainloader.dataset.dataset.targets = np.concatenate(
                (targets[:, None], additionl_feature_banks.detach().cpu().numpy().T),
                axis=1,
            )

        train_epoch_ = int(np.ceil(train_epoch))
        max_iter = int(train_epoch * len(self.trainloader))
        epoch_start_time = time.time()

        for iter in range(train_epoch_):
            model.train()
            local_curr_ep = self.args.local_ep * global_round + iter

            if self.args.optimizer == "sgd":
                adjust_learning_rate(
                    optimizer,
                    train_lr,
                    local_curr_ep,
                    self.args.epochs * self.args.local_ep,
                    iter,
                )

            batch_loss = []
            batch_size = self.args.local_bs
            temperature = self.args.temperature
            start_time = time.time()

            if self.args.distributed_training:
                self.dist_sampler.set_epoch(int(local_curr_ep))

            for batch_idx, data in enumerate(self.trainloader):
                data_time = time.time() - start_time
                start_time = time.time()

                if additionl_feature_banks is not None:
                    (pos_1, pos_2, labels) = data
                    labels, addition_features = (
                        labels[:, [0]],
                        labels[:, 1:].to(self.device),
                    )

                    loss, feat = model(
                        pos_1.to(self.device),
                        pos_2.to(self.device),
                        addition_features,
                        self.reg_scale,
                        return_feat=True,
                    )
                else:
                    if self.args.ssl_method == "mae":
                        images, labels = data
                        images, labels = images.to(self.device), labels.to(self.device)
                        predicted_img, mask = model(images)
                        feat = mask
                        mask_ratio = 0.75
                        loss = (
                                torch.mean((predicted_img - images) ** 2 * mask)
                                / mask_ratio
                        )
                    else:
                        (pos_1, pos_2, labels) = data
                        loss, feat = model(
                            pos_1.to(self.device, non_blocking=True),
                            pos_2.to(self.device, non_blocking=True),
                            return_feat=True,
                        )

                loss = loss.mean()
                optimizer.zero_grad()
                if not loss.isnan().any():
                    loss.backward()
                    optimizer.step()

                model_time = time.time() - start_time
                start_time = time.time()

                if batch_idx % 10 == 0:
                    print(
                        "Update SSL || User : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} \
                        LR: {:.4f}  Feat: {:.3f} Epoch Time: {:.3f} Model Time: {:.3f} Data Time: {:.3f} Model: {}".format(
                            self.id,
                            global_round,
                            local_curr_ep,
                            batch_idx * len(labels),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                            optimizer.param_groups[0]["lr"],
                            feat.mean().item(),
                            time.time() - epoch_start_time,
                            model_time,
                            data_time,
                            self.args.model_time,
                        )
                    )
                if self.logger is not None:
                    self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
                data_start_time = time.time()
                scheduler.step(int(local_curr_ep))

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if additionl_feature_banks is not None:
            self.trainloader.dataset.dataset.targets = (
                self.trainloader.dataset.dataset.target_copy
            )

        self.model = model
        self.optimizer = optimizer
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights(
            self, model, global_round, vis_feature=False, lr=None, epoch_num=None
    ):
        """Train the local model with superivsed learning"""
        self.model = model
        model.train()
        epoch_loss = []

        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-6
            )
        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())

        schedule = [
            int(self.args.local_ep * self.args.epochs * 0.3),
            int(self.args.local_ep * self.args.epochs * 0.6),
            int(self.args.local_ep * self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )

        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter
            batch_loss = []
            feature_bank, label_bank, image_bank = [], [], []

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                label_bank.append(labels.detach().cpu().numpy())
                optimizer.zero_grad()
                log_probs, feat = model(images, return_feat=True)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print(
                        "Inference || User : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            self.id,
                            global_round,
                            self.args.local_ep * global_round + iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / (len(batch_loss) + 1e-4))
            scheduler.step(int(local_curr_ep))
        self.optimizer = optimizer
        self.model = model
        return model.state_dict(), sum(epoch_loss) / (len(epoch_loss) + 1e-4)
    # def proto_based_inference(self, model, test_dataset, test_user, agg_protos):
    #     """Returns the inference accuracy and loss for a local client based on prototype matching."""
    #     model.eval()
    #     total, correct = 0.0, 0.0
    #
    #     testloader = DataLoader(
    #         DatasetSplit(test_dataset, test_user),
    #         batch_size=64,
    #         shuffle=False,
    #         num_workers=4,
    #         pin_memory=True,
    #     )
    #
    #     for batch_idx, (images, labels) in enumerate(testloader):
    #         images, labels = images.to(self.device), labels.to(self.device)
    #
    #         # Inference to get the feature representations
    #         _, features = model(images)
    #
    #         # Prediction by finding the nearest prototype
    #         pred_labels = []
    #         for feature in features:
    #             min_dist = float('inf')
    #             pred_label = -1
    #             for key, protos in agg_protos.items():
    #                 for proto in protos:
    #                     dist = torch.norm(feature - proto)
    #                     if dist < min_dist:
    #                         min_dist = dist
    #                         pred_label = key
    #             pred_labels.append(pred_label)
    #
    #         pred_labels = torch.tensor(pred_labels).to(self.device)
    #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #         total += len(labels)
    #
    #     accuracy = correct / total
    #     return accuracy, correct, total
    # cosine

    # def proto_based_inference(self, model, test_dataset, test_user, agg_protos):
    #     """Returns the inference accuracy, loss, predicted labels, and actual labels for a local client based on prototype matching."""
    #     model.eval()
    #     total, correct = 0.0, 0.0
    #     pred_labels_list = []
    #     actual_labels_list = []
    #     similarity_scores = []
    #
    #     # Convert aggregated prototypes to tensors and move to device
    #     protos_tensors = {key: torch.stack([torch.tensor(proto).to(self.device) for proto in protos]) for key, protos in
    #                       agg_protos.items()}
    #
    #     testloader = DataLoader(
    #         DatasetSplit(test_dataset, test_user),
    #         batch_size=64,
    #         shuffle=False,
    #         num_workers=4,
    #         pin_memory=True,
    #     )
    #
    #     with torch.no_grad():
    #         for batch_idx, (images, labels) in enumerate(testloader):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #
    #             # Inference to get the feature representations
    #             _, features = model(images)
    #
    #             # Check and possibly reshape features
    #             if features.dim() == 4:
    #                 features = features.view(features.size(0), -1)  # Flatten if features are 4D (e.g., images)
    #
    #             # Prediction by finding the nearest prototype using cosine similarity
    #             pred_labels = []
    #             for feature in features:
    #                 max_similarity = float('-inf')
    #                 pred_label = -1
    #                 for key, protos in protos_tensors.items():
    #                     protos = protos.view(protos.size(0), -1)  # Ensure protos are flattened if needed
    #                     similarities = torch.nn.functional.cosine_similarity(feature.unsqueeze(0), protos,
    #                                                                          dim=1)  # Calculate cosine similarities
    #                     similarity, idx = torch.max(similarities, dim=0)
    #                     if similarity > max_similarity:
    #                         max_similarity = similarity
    #                         pred_label = key
    #                 pred_labels.append(pred_label)
    #                 similarity_scores.append(max_similarity.item())
    #
    #             pred_labels = torch.tensor(pred_labels).to(self.device)
    #             correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #             total += len(labels)
    #
    #             pred_labels_list.extend(pred_labels.cpu().tolist())
    #             actual_labels_list.extend(labels.cpu().tolist())
    #
    #     accuracy = correct / total
    #     return accuracy, correct, total, pred_labels_list, actual_labels_list, similarity_scores
    # cosine qujian
    def proto_based_inference(self, model, test_dataset, test_user, agg_protos, num_intervals=10):
        """Returns the inference accuracy, loss, predicted labels, actual labels, and similarity scores for a local client based on prototype matching."""
        model.eval()
        total, correct = 0.0, 0.0
        pred_labels_list = []
        actual_labels_list = []
        similarity_scores = []

        # Initialize interval counts
        interval_counts = [0] * num_intervals
        interval_correct_counts = [0] * num_intervals
        interval_incorrect_counts = [0] * num_intervals

        # Convert aggregated prototypes to tensors and move to device
        protos_tensors = {key: torch.stack([torch.tensor(proto).to(self.device) for proto in protos]) for key, protos in
                          agg_protos.items()}

        testloader = DataLoader(
            DatasetSplit(test_dataset, test_user),
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference to get the feature representations
                _, features = model(images)

                # Check and possibly reshape features
                if features.dim() == 4:
                    features = features.view(features.size(0), -1)  # Flatten if features are 4D (e.g., images)

                # Prediction by finding the nearest prototype using cosine similarity
                pred_labels = []
                for feature in features:
                    max_similarity = float('-inf')
                    pred_label = -1
                    for key, protos in protos_tensors.items():
                        protos = protos.view(protos.size(0), -1)  # Ensure protos are flattened if needed
                        similarities = torch.nn.functional.cosine_similarity(feature.unsqueeze(0), protos,
                                                                             dim=1)  # Calculate cosine similarities
                        similarity, idx = torch.max(similarities, dim=0)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            pred_label = key
                    pred_labels.append(pred_label)
                    similarity_scores.append(max_similarity.item())

                    # Determine which interval this similarity score falls into
                    interval_idx = min(int(max_similarity * num_intervals), num_intervals - 1)
                    interval_counts[interval_idx] += 1
                    if pred_label == labels.cpu().tolist()[0]:
                        interval_correct_counts[interval_idx] += 1
                    else:
                        interval_incorrect_counts[interval_idx] += 1

                pred_labels = torch.tensor(pred_labels).to(self.device)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                pred_labels_list.extend(pred_labels.cpu().tolist())
                actual_labels_list.extend(labels.cpu().tolist())

        accuracy = correct / total

        return accuracy, correct, total, pred_labels_list, actual_labels_list, similarity_scores, interval_counts, interval_correct_counts, interval_incorrect_counts
    # L2norm

    # def proto_based_inference(self, model, test_dataset, test_user, agg_protos):
    #     """Returns the inference accuracy, loss, predicted labels, and actual labels for a local client based on prototype matching."""
    #     model.eval()
    #     total, correct = 0.0, 0.0
    #     pred_labels_list = []
    #     actual_labels_list = []
    #
    #     # Convert aggregated prototypes to tensors and move to device
    #     protos_tensors = {key: torch.stack([torch.tensor(proto).to(self.device) for proto in protos]) for key, protos in
    #                       agg_protos.items()}
    #
    #     testloader = DataLoader(
    #         DatasetSplit(test_dataset, test_user),
    #         batch_size=64,
    #         shuffle=False,
    #         num_workers=4,
    #         pin_memory=True,
    #     )
    #
    #     with torch.no_grad():
    #         for batch_idx, (images, labels) in enumerate(testloader):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #
    #             # Inference to get the feature representations
    #             _, features = model(images)
    #
    #             # Check and possibly reshape features
    #             if features.dim() == 4:
    #                 features = features.view(features.size(0), -1)  # Flatten if features are 4D (e.g., images)
    #
    #             # Prediction by finding the nearest prototype using L2 distance
    #             pred_labels = []
    #             for feature in features:
    #                 min_dist = float('inf')
    #                 pred_label = -1
    #                 for key, protos in protos_tensors.items():
    #                     protos = protos.view(protos.size(0), -1)  # Ensure protos are flattened if needed
    #                     dists = torch.norm(feature.unsqueeze(0) - protos, p=2, dim=1)  # Calculate L2 distances
    #                     dist, idx = torch.min(dists, dim=0)
    #                     if dist < min_dist:
    #                         min_dist = dist
    #                         pred_label = key
    #                 pred_labels.append(pred_label)
    #
    #             pred_labels = torch.tensor(pred_labels).to(self.device)
    #             correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #             total += len(labels)
    #
    #             pred_labels_list.extend(pred_labels.cpu().tolist())
    #             actual_labels_list.extend(labels.cpu().tolist())
    #
    #     accuracy = correct / total
    #     return accuracy, correct, total, pred_labels_list, actual_labels_list

    def confidence(self, model, test_dataset=None, test_user=None):
        """Returns the inference accuracy and loss for a local client."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if test_dataset is not None:
            self.testloader = DataLoader(
                DatasetSplit(test_dataset, test_user, idx=self.id),
                batch_size=64,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs,_ = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss
    # test acc with proto
    def inference(self, model, test_dataset=None, test_user=None):
        """Returns the inference accuracy and loss for a local client."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        if test_dataset is not None:
            self.testloader = DataLoader(
                DatasetSplit(test_dataset, test_user, idx=self.id),
                batch_size=64,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs,_ = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss, correct, total

    def evaluate_model_with_confidence(self, model, test_dataset, test_user):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        high_conf_total, high_conf_correct = 0.0, 0.0

        if test_dataset is not None:
            self.testloader = DataLoader(
                DatasetSplit(test_dataset, test_user, idx=self.id),
                batch_size=64,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs, _ = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)


            # Calculate confidence (probabilities) using softmax
            # predicted = torch.softmax(outputs.detach() / self.args.T, dim=-1)
            # confidence_scores = predicted.max(dim=1).values

            # Count high confidence samples
            # high_conf_mask = confidence_scores > 0.95
            # high_conf_total += torch.sum(high_conf_mask).item()
            # high_conf_correct += torch.sum(high_conf_mask & (pred_labels == labels)).item()

        accuracy = correct / total
        high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0.0
        # high_conf_total：通过high_conf_mask 统计置信度超过95%的样本数量。
        # high_conf_correct：统计置信度超过95%且预测正确的样本数量，通过high_conf_mask和预测正确样本的联合条件来计算。
        # high_conf_accuracy：通过high_conf_correct除以high_conf_total来计算高置信度样本的准确率，如果没有高置信度样本，结果设为0.0。
        return accuracy, loss, correct, total, high_conf_total, high_conf_correct, high_conf_accuracy

    def confidence_qujian(self, model, agg_protos, test_dataset=None, test_user=None, temperature=0.7):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        confidence_buckets_correct = [[] for _ in range(20)]  # To store confidence values for correct predictions
        confidence_buckets_incorrect = [[] for _ in range(20)]  # To store confidence values for incorrect predictions
        confidence_sample_counts = [0] * 20  # To store total samples in each confidence interval

        if test_dataset is not None:
            self.testloader = DataLoader(
                DatasetSplit(test_dataset, test_user, idx=self.id),
                batch_size=64,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs, features = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct_mask = torch.eq(pred_labels, labels)
            correct += torch.sum(correct_mask).item()
            total += len(labels)

            # Calculate confidence (probabilities) using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Apply the custom confidence calculation
            transformed_probs = probabilities ** (1 / temperature)
            confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
            confidence_scores = confidence_scores.max(dim=1).values * 100  # Convert to percentage

            # Calculate proto-based predictions and cosine similarities
            proto_labels = []
            cosine_sims = []
            for feature in features:
                max_cos_sim = -1
                best_proto_label = -1
                for key, proto_tensor in agg_protos.items():
                    cos_similarities = F.cosine_similarity(feature.unsqueeze(0), proto_tensor, dim=1)
                    if cos_similarities.max().item() > max_cos_sim:
                        max_cos_sim = cos_similarities.max().item()
                        best_proto_label = key
                proto_labels.append(best_proto_label)
                cosine_sims.append(max_cos_sim)

            proto_labels = torch.tensor(proto_labels, device=self.device)
            cosine_sims = torch.tensor(cosine_sims, device=self.device)

            # Count samples in different confidence intervals
            for i in range(20):
                lower_bound = i * 5
                upper_bound = (i + 1) * 5
                bucket_mask = (confidence_scores >= lower_bound) & (confidence_scores < upper_bound)
                bucket_size = torch.sum(bucket_mask).item()
                confidence_sample_counts[i] += bucket_size

                # Track detailed information in this confidence interval
                for score, correct_flag, true_label, pred_label, proto_label, cos_sim in zip(
                        confidence_scores[bucket_mask], correct_mask[bucket_mask], labels[bucket_mask],
                        pred_labels[bucket_mask], proto_labels[bucket_mask], cosine_sims[bucket_mask]):
                    info = {
                        'confidence': score.item(),
                        'true_label': true_label.item(),
                        'model_pred': pred_label.item(),
                        'proto_pred': proto_label,
                        'cosine_sim': cos_sim.item()
                    }
                    if correct_flag:
                        confidence_buckets_correct[i].append(info)
                    else:
                        confidence_buckets_incorrect[i].append(info)

        # Calculate percentages for each confidence interval relative to total samples in each interval
        confidence_percentages_correct = [
            (len(bucket_correct) / count * 100 if count > 0 else 0.0)
            for bucket_correct, count in zip(confidence_buckets_correct, confidence_sample_counts)]
        confidence_percentages_incorrect = [
            (len(bucket_incorrect) / count * 100 if count > 0 else 0.0)
            for bucket_incorrect, count in zip(confidence_buckets_incorrect, confidence_sample_counts)]

        total_samples = sum(confidence_sample_counts)
        confidence_sample_percentages = [count / total_samples * 100 for count in confidence_sample_counts]

        accuracy = correct / total

        # Return accuracy, loss, correct, total, confidence percentages for correct and incorrect predictions,
        # confidence sample percentages, correct samples confidence values, and incorrect samples confidence values for each interval
        return accuracy, loss, correct, total, confidence_percentages_correct, confidence_percentages_incorrect, confidence_sample_percentages, confidence_buckets_correct, confidence_buckets_incorrect

    def update_local_protos(self, model, dataset=None, test_user=None, temperature=0.8):
        """Returns the test accuracy and loss."""
        agg_protos_label = {}
        model.eval()
        if dataset is not None:
            self.testloader = DataLoader(
                DatasetSplit(dataset, test_user, idx=self.id),
                batch_size=64,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
            )

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            log_probs, protos = model(images)
            probabilities = torch.nn.functional.softmax(log_probs, dim=1)

            # Apply the custom confidence calculation
            transformed_probs = probabilities ** (1 / temperature)
            confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
            confidence_scores = confidence_scores.max(dim=1).values * 100  # Convert to percentage

            # Inference
            pred_labels = torch.argmax(probabilities, dim=1)
            for i in range(len(labels)):
                if confidence_scores[i] >= 95:
                    pred_label = pred_labels[i].item()
                    if pred_label in agg_protos_label:
                        agg_protos_label[pred_label].append(protos[i, :])
                    else:
                        agg_protos_label[pred_label] = [protos[i, :]]
                else:
                    continue
        return agg_protos_label

def test_inference(args, model, test_dataset):
    """Returns the test accuracy and loss."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = "cuda" if args.gpu else "cpu"
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    test_bar = tqdm((testloader), desc="Linear Probing", disable=False)

    for (images, labels) in test_bar:
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs,_ = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        test_bar.set_postfix({"Accuracy": correct / total * 100})

    accuracy = correct / total
    return accuracy, loss


def test_confidence(args, model, test_dataset):
    """Returns the test accuracy and loss."""
    # model.eval() 将模型设置为评估模式，以确保在推理期间关闭dropout层等训练时特有的层。
    # 初始化 loss、total 和 correct 为 0，用于累积损失和正确预测的数量。
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    all_labels = []
    all_predictions = []
    all_confidences = []
    device = "cuda" if args.gpu else "cpu"
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    test_bar = tqdm((testloader), desc="Linear Probing", disable=False)

    for (images, labels) in test_bar:
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        # Calculate confidence (probabilities) using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence_scores = probabilities.max(dim=1).values

        # Store current batch labels, predictions, and confidence scores
        all_labels.extend(labels.detach().cpu().numpy())
        all_predictions.extend(pred_labels.detach().cpu().numpy())
        all_confidences.extend(confidence_scores.detach().cpu().numpy())

        test_bar.set_postfix({"Accuracy": correct / total * 100})
        test_bar.refresh()  # 强制刷新进度条

    accuracy = correct / total
    print(f"test_acc {accuracy}")

    # Print or return all test labels, predictions, and confidence scores
    for label, prediction, confidence in zip(all_labels, all_predictions, all_confidences):
        print(f"Actual Label: {label}, Predicted Label: {prediction}, Confidence: {confidence:.4f}")
    return accuracy, loss



def update_global_protos(args, model, dataset):
    """Returns the test accuracy and loss."""
    agg_protos_label = {}
    model.eval()
    device = "cuda" if args.gpu else "cpu"
    globalloader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    test_bar = tqdm((globalloader), desc="feature extraction", disable=False)

    for (images, labels) in test_bar:
        images, labels = images.to(device), labels.to(device)
        log_probs, protos = model(images)
        # Inference
        for i in range(len(labels)):
            if labels[i].item() in agg_protos_label:
                agg_protos_label[labels[i].item()].append(protos[i, :])
            else:
                agg_protos_label[labels[i].item()] = [protos[i, :]]

    return agg_protos_label

def agg_func(protos):
    """
    Returns the average of the weights.
    """
    # label,list
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()

    # Aggregate local prototypes
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    # Final aggregation and convert to single tensor per label
    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            proto_sum = sum(proto.data for proto in proto_list)
            agg_protos_label[label] = proto_sum / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label



def test_function(args, global_model, test_train_dataset, test_val_dataset, device):
    test_train_loader = DataLoader(
        test_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )

    test_val_loader = DataLoader(
        test_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    global_model.eval()
    protos = update_global_protos(args, global_model, test_train_dataset)
    agg_protos = agg_func(protos)

    # Convert agg_protos to numpy array for cosine similarity calculation
    agg_protos = {label: proto.cpu().numpy() for label, proto in agg_protos.items()}
    agg_protos = {label: agg_protos[label] for label in sorted(agg_protos)}

    results = []

    with torch.no_grad():
        for data, labels in test_val_loader:
            data = data.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs, _ = global_model(data)
            _, preds = torch.max(outputs, 1)

            # Calculate confidence (probabilities) using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Apply the custom confidence calculation
            temperature = 0.5  # You need to define the temperature here
            transformed_probs = probabilities ** (1 / temperature)
            confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
            confidence_scores = confidence_scores.max(dim=1).values * 100  # Convert to percentage

            # Get features for cosine similarity calculation
            _, features = global_model(data)
            features = features.cpu().numpy()

            # Calculate cosine similarity and predictions based on prototypes
            cos_sims = np.array([cosine_similarity(features, proto[np.newaxis]) for proto in agg_protos.values()])
            cos_sims = np.transpose(cos_sims, (1, 0, 2)).reshape(features.shape[0], -1)
            cos_preds = np.argmax(cos_sims, axis=1)

            # Normalize cosine similarities to range [0, 1]
            normalized_cos_sims = (cos_sims + 1) / 2

            # Get predicted labels based on agg_protos dictionary
            pred_labels = [list(agg_protos.keys())[idx % len(agg_protos)] for idx in cos_preds]

            # Collect results for each sample in the batch
            for i in range(len(labels)):
                result = {
                    "true_label": labels[i].item(),
                    "model_pred": preds[i].item(),
                    "confidence": confidence_scores[i].item(),
                    "dist_pred": pred_labels[i],
                    "dists": normalized_cos_sims[i].tolist(),
                    "cosine_sim": normalized_cos_sims[i][cos_preds[i]]  # Add cosine similarity of the predicted label
                }
                results.append(result)

        # Calculate accuracy
        total_samples = len(results)
        equal = sum(1 for result in results if result["dist_pred"] == result["model_pred"])
        true = sum(1 for result in results if result["dist_pred"] == result["model_pred"]and result["true_label"] == result["model_pred"])
        correct_model_pred = sum(1 for result in results if result["true_label"] == result["model_pred"])
        correct_dist_pred = sum(1 for result in results if result["true_label"] == result["dist_pred"])
        accuracy_model_pred = correct_model_pred / total_samples * 100
        accuracy_dist_pred = correct_dist_pred / total_samples * 100
        true_equal_rate = true/equal * 100

        # Print accuracy
        print(f"Model Prediction Accuracy: {accuracy_model_pred:.4f}%")
        print(f"Distance Prediction Accuracy: {accuracy_dist_pred:.4f}%")
        print(f"proto==model_output: {equal:}")
        print(f"true_equal_rate: {true_equal_rate:.4f}%")



        return results