# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
# dataaugmentation
import logging
import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple
import copy
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

PARAMETER_MAX = 10




class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(
            self, dataset, idxs, idx=0, noniid=False, noniid_prob=1.0, xshift_type="rot"
    ):
        self.dataset = dataset
        try:
            self.idxs = [int(i) for i in idxs]
        except ValueError as e:
            print(f"Error converting idxs to integers: {e}")
            raise ValueError("All idxs must be convertible to integers.")

        print(f"Initialized DatasetSplit with {len(self.idxs)} indices.")

        self.idx = idx
        self.noniid = noniid
        self.classes = self.dataset.classes
        self.targets = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]



class LocalUpdate:
    def __init__(self, args, dataset, idx, idxs, logger=None, test_dataset=None, memory_dataset=None, output_dir=""):
        self.args = args
        self.logger = logger
        self.id = idx  # user id
        self.idxs = idxs  # dataset id
        self.reg_scale = args.reg_scale
        self.output_dir = output_dir
        self.dataset = dataset
        if dataset is not None:
            self.trainloader, self.testloader, self.loader = self.train_val_test(dataset, list(idxs), test_dataset, memory_dataset)

        self.device = "cuda" if args.gpu else "cpu"
        self.criterion = torch.nn.NLLLoss().to(self.device)
        self.model = None  # 初始化 model 属性

        # 添加数据增强变换
        self.transform = TransformFixMatch(mean=args.mean, std=args.std)
        # 初始化 memoryloader 属性
        self.memoryloader = DataLoader(memory_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True) if memory_dataset else None
        # 定义这两个
        self.critical_parameter = None  # record the critical parameter positions in FedCAC
        self.customized_model = copy.deepcopy(self.model)  # customized global model
    # def train_val_test(self, dataset, idxs, test_dataset=None, memory_dataset=None):
    #     """
    #     Returns train, validation and test dataloaders for a given dataset
    #     and user indexes. split indexes for train, validation, and test (80, 10, 10)
    #     """
    #     idxs_train = idxs[: int(0.9 * len(idxs))]
    #     self.idxs_train = idxs_train
    #     idxs_test = idxs[int(0.9 * len(idxs)):]
    #
    #     print(f"Training samples: {len(idxs_train)}, Test samples: {len(idxs_test)}")
    #
    #     train_dataset = DatasetSplit(dataset, idxs_train, idx=self.id)
    #
    #     if not self.args.distributed_training:
    #         trainloader = DataLoader(
    #             train_dataset,
    #             batch_size=self.args.local_bs,
    #             shuffle=True,
    #             num_workers=16,
    #             pin_memory=True,
    #             drop_last=True if len(train_dataset) > self.args.local_bs else False,
    #         )
    #     else:
    #         self.dist_sampler = DistributedSampler(train_dataset, shuffle=True)
    #         trainloader = DataLoader(
    #             train_dataset,
    #             sampler=self.dist_sampler,
    #             batch_size=self.args.local_bs,
    #             num_workers=16,
    #             pin_memory=True,
    #             drop_last=True,
    #         )
    #
    #     testloader = DataLoader(
    #         DatasetSplit(dataset, idxs_test, idx=self.id),
    #         batch_size=64,
    #         shuffle=False,
    #         num_workers=1,
    #         pin_memory=True,
    #     )
    #
    #     if test_dataset is not None:
    #         memoryloader = DataLoader(
    #             DatasetSplit(memory_dataset, idxs_train, idx=self.id),
    #             batch_size=64,
    #             shuffle=False,
    #             num_workers=1,
    #             pin_memory=True,
    #             drop_last=False,
    #         )
    #     else:
    #         memoryloader = DataLoader(
    #             DatasetSplit(dataset, idxs_train, idx=self.id),
    #             batch_size=self.args.local_bs,
    #             shuffle=False,
    #             num_workers=1,
    #             pin_memory=True,
    #             drop_last=False,
    #         )
    #
    #     self.memory_loader = memoryloader
    #     self.test_loader = testloader
    #
    #     return trainloader, testloader
    def train_val_test(self, dataset, idxs, test_dataset=None, memory_dataset=None):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes. split indexes for train, validation, and test (90, 10)
        """
        # 固定随机种子
        seed_value = 1
        g = torch.Generator()
        g.manual_seed(seed_value)

        # 确保数据索引的一致性

        idxs_train = idxs[: int(0.9 * len(idxs))]
        self.idxs_train = idxs_train
        idxs_test = idxs[int(0.9 * len(idxs)):]

        print(f"Training samples: {len(idxs_train)}, Test samples: {len(idxs_test)}")

        train_dataset = DatasetSplit(dataset, idxs_train, idx=self.id)
        test_dataset = DatasetSplit(dataset, idxs_test, idx=self.id)
        dataset_all = DatasetSplit(dataset, idxs, idx=self.id)
        if not self.args.distributed_training:
            loader = DataLoader(
                dataset_all,
                batch_size=self.args.local_bs,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                drop_last= False,
                generator=g
            )
            trainloader = DataLoader(
                train_dataset,
                batch_size=self.args.local_bs,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                drop_last=True if len(train_dataset) > self.args.local_bs else False,
                generator=g
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

        testloader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        if memory_dataset is not None:
            memoryloader = DataLoader(
                DatasetSplit(memory_dataset, idxs_train, idx=self.id),
                batch_size=64,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )
        else:
            memoryloader = DataLoader(
                DatasetSplit(dataset, idxs_train, idx=self.id),
                batch_size=self.args.local_bs,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )


        return trainloader, testloader, loader
    # def train_val_test(self, dataset, idxs, test_dataset=None, memory_dataset=None):
    #     """
    #     Returns train, validation and test dataloaders for a given dataset
    #     and user indexes. split indexes for train, validation, and test (80, 10, 10)
    #     """
    #     idxs_train = idxs[: int(0.9 * len(idxs))]
    #     self.idxs_train = idxs_train
    #     # idxs_val = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
    #     idxs_test = idxs[int(0.9 * len(idxs)) :]
    #
    #     train_dataset = DatasetSplit(dataset, idxs_train, idx=self.id)
    #
    #     if not self.args.distributed_training:
    #         trainloader = DataLoader(
    #             train_dataset,
    #             batch_size=self.args.local_bs,
    #             shuffle=True,
    #             num_workers=16,
    #             pin_memory=True,
    #             drop_last=True if len(train_dataset) > self.args.local_bs else False,
    #         )
    #     else:
    #         self.dist_sampler = DistributedSampler(train_dataset, shuffle=True)
    #         trainloader = DataLoader(
    #             train_dataset,
    #             sampler=self.dist_sampler,
    #             batch_size=self.args.local_bs,
    #             num_workers=16,
    #             pin_memory=True,
    #             drop_last=True,
    #         )
    #
    #     # validloader = DataLoader(
    #     #     DatasetSplit(dataset, idxs_val, idx=self.id),
    #     #     batch_size=self.args.local_bs,
    #     #     shuffle=False,
    #     #     num_workers=1,
    #     #     pin_memory=True,
    #     # )
    #
    #     testloader = DataLoader(
    #         DatasetSplit(dataset, idxs_test, idx=self.id),
    #         batch_size=64,
    #         shuffle=False,
    #         num_workers=1,
    #         pin_memory=True,
    #     )
    #
    #     if test_dataset is not None:
    #         memoryloader = DataLoader(
    #             DatasetSplit(memory_dataset, idxs_train, idx=self.id),
    #             batch_size=64,
    #             shuffle=False,
    #             num_workers=1,
    #             pin_memory=True,
    #             drop_last=False,
    #         )
    #     else:
    #         memoryloader = DataLoader(
    #             DatasetSplit(dataset, idxs_train, idx=self.id),
    #             batch_size=self.args.local_bs,
    #             shuffle=False,
    #             num_workers=1,
    #             pin_memory=True,
    #             drop_last=False,
    #         )
    #
    #     self.memory_loader = memoryloader
    #     self.test_loader = testloader
    #
    #     # return trainloader, validloader, testloader
    #     return trainloader, testloader

    def get_model(self):
        return self.model

    def init_dataset(self, dataset):
        self.trainloader, self.testloader = self.train_val_test(dataset, list(self.idxs), self.testloader.dataset, self.memoryloader.dataset)

    def init_model(self, model):
        """Initialize local models"""
        train_lr = self.args.lr
        self.model = model

        if self.args.optimizer == "sgd":
            train_lr = self.args.lr
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
                model.parameters(), lr=train_lr, weight_decay=5e-4
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

    def evaluate_accuracy(self, model, dataloader, device):
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_confidences = []
        top2_predictions = []
        top2_confidences = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                # Inference
                outputs = model(images)
                predicted = torch.softmax(outputs.detach() / self.args.T, dim=-1)

                max_probs, targets_u = torch.max(predicted, dim=-1)
                correct += torch.sum(torch.eq(targets_u, labels)).item()
                total += labels.size(0)

                # For each sample, get the top 2 predictions and their confidence scores
                top2_vals, top2_idx = torch.topk(predicted, k=2, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(targets_u.cpu().numpy())
                all_confidences.extend(max_probs.cpu().numpy())
                top2_predictions.extend(top2_idx.cpu().numpy())
                top2_confidences.extend(top2_vals.cpu().numpy())

        accuracy = 100 * correct / total

        # Print the accuracy
        print(f"Accuracy: {accuracy:.2f}%")

        # Print all test labels, predictions, and confidence scores
        for label, prediction, confidence, top2_pred, top2_conf in zip(all_labels, all_predictions, all_confidences,
                                                                       top2_predictions, top2_confidences):
            print(f"Actual Label: {label}, Predicted Label: {prediction}, Confidence: {confidence:.4f}")
            print(f"Top 2 Predicted Labels: {top2_pred}, Top 2 Confidences: {top2_conf}")

        return accuracy

    # def update_semi_weights(self, model, global_round, additional_feature_banks=None):
    #     self.model = model  # 设置 self.model 属性
    #     self.model.to(self.device)
    #     self.model.train()
    #     epoch_loss = []
    #
    #     # 选择优化器
    #     if self.args.optimizer == "sgd":
    #         train_lr = self.args.lr * (self.args.batch_size / 256)
    #         if self.args.distributed_training:
    #             train_lr = train_lr * self.args.world_size
    #         optimizer = torch.optim.SGD(
    #             self.model.parameters(),
    #             lr=train_lr,
    #             momentum=self.args.momentum,
    #             weight_decay=self.args.weight_decay,
    #         )
    #     elif self.args.optimizer == "adam":
    #         optimizer = torch.optim.Adam(
    #             self.model.parameters(), lr=self.args.lr, weight_decay=1e-6
    #         )
    #
    #     # 如果继续训练，加载优化器状态
    #     if self.args.model_continue_training and hasattr(self, "optimizer"):
    #         optimizer.load_state_dict(self.optimizer.state_dict())
    #     schedule = [
    #         int(self.args.epochs * 0.3),
    #         int(self.args.epochs * 0.6),
    #         int(self.args.epochs * 0.9),
    #     ]
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, milestones=schedule, gamma=0.3
    #     )
    #
    #     criterion = torch.nn.CrossEntropyLoss().to(self.device)
    #
    #     for iter in range(int(self.args.local_ep)):
    #         local_curr_ep = self.args.local_ep * global_round + iter
    #
    #         # 在每个 epoch 开始前计算并打印准确率
    #         accuracy_before, loss_before, correct_before, total_before = self.inference(self.model)
    #         print(f'User: {self.id} \tEpoch: {iter} \tAccuracy before training: {100. * accuracy_before:.4f}%')
    #         print(
    #             f"Loss before training: {loss_before}, Correct before training: {correct_before}, Total before training: {total_before}")
    #
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.trainloader):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #
    #             # 进行强弱数据增强
    #             weak, strong = self.transform(images)
    #
    #             # 移动到设备
    #             weak, strong = weak.to(self.device), strong.to(self.device)
    #
    #             # 计算弱增强的输出
    #             # softmax,feature
    #             logits_u_w, _ = self.model(weak)
    #             probabilities = torch.nn.functional.softmax(logits_u_w, dim=1)
    #             # Apply the custom confidence calculation
    #             transformed_probs = probabilities ** (1 / self.args.T)
    #             confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
    #
    #             max_probs, targets_u = torch.max(confidence_scores, dim=-1)
    #             mask = max_probs.ge(self.args.threshold).float()
    #
    #             # 计算强增强的输出
    #             logits_u_s, _ = self.model(strong)
    #
    #             # 计算未标记数据的损失
    #             Lu = (criterion(logits_u_s, targets_u) * mask).mean()
    #
    #             # 计算当前 batch 总样本数
    #             total_samples = images.size(0)
    #
    #             # 计算伪标签正确的样本数
    #             correct_pseudo_labels = (targets_u == labels).float().sum().item()
    #
    #             # 计算超过 threshold 的样本数量
    #             threshold_samples = mask.sum().item()
    #
    #             # 计算超过 threshold 且伪标签正确的样本数量
    #             correct_threshold_samples = ((targets_u == labels).float() * mask).sum().item()
    #
    #             print(
    #                 f"Batch {batch_idx}: Total Samples = {total_samples}, Correct Pseudo Labels = {correct_pseudo_labels}, Threshold Samples = {threshold_samples}, Correct Threshold Samples = {correct_threshold_samples}")
    #
    #             optimizer.zero_grad()
    #             Lu.backward()
    #             optimizer.step()
    #
    #             batch_loss.append(Lu.item())
    #             if batch_idx % 50 == 0:
    #                 print(
    #                     f'Local Training Epoch: {iter} [{batch_idx * len(images)}/{len(self.trainloader.dataset)} ({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {Lu.item():.6f}')
    #
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #         scheduler.step(int(local_curr_ep))
    #
    #         # 在每个 epoch 结束后进行评估
    #         # accuracy_after = self.evaluate_accuracy(self.model, self.trainloader, self.device)
    #         # print(f'User: {self.id} \tEpoch {iter + 1}/{self.args.local_ep}, Test Accuracy: {accuracy_after:.4f}%')
    #     self.optimizer = optimizer
    #     self.model = model
    #     accuracy_after, loss_after, correct_after, total_after = self.inference(self.model)
    #     # 打印训练之后的值
    #     print(f'User: {self.id} \tEpoch: {global_round} \tAccuracy after training: {100. * accuracy_after:.4f}')
    #     print(
    #         f"Loss after training: {loss_after}, Correct after training: {correct_after}, Total after training: {total_after}")
    #
    #     return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    # 弱扰动，大于95的进行强扰动，然后Lce
    def update_semi_weights_WEAKSTRONG(self, model, global_round, additional_feature_banks=None):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=1e-6
            )
        # 如果继续训练，加载优化器状态
        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())
        schedule = [
            int(self.args.epochs * 0.3),
            int(self.args.epochs * 0.6),
            int(self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter

            # 在每个 epoch 开始前计算并打印准确率
            # accuracy_before, loss_before, correct_before, total_before = self.inference(self.model)
            # print(f'User: {self.id} \tEpoch: {iter} \tAccuracy before training: {100.*accuracy_before:.4f}%')
            # print(
            #     f"Loss before training: {loss_before}, Correct before training: {correct_before}, Total before training: {total_before}")

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 进行强弱数据增强
                weak, strong = self.transform(images)

                # 移动到设备
                weak, strong = weak.to(self.device), strong.to(self.device)

                # 计算弱增强的输出
                # softmax,feature
                logits_u_w, _ = self.model(weak)
                probabilities = torch.nn.functional.softmax(logits_u_w, dim=1)
                # Apply the custom confidence calculation
                transformed_probs = probabilities ** (1 / self.args.T)
                confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
                # targets_u弱增强对的
                max_probs, targets_u = torch.max(confidence_scores, dim=-1)
                mask = max_probs.ge(self.args.threshold).float()

                # 计算强增强的输出
                logits_u_s, _ = self.model(strong)

                # 计算未标记数据的损失
                Lu = (criterion(logits_u_s, targets_u) * mask).mean()

                # 计算当前 batch 总样本数
                total_samples = images.size(0)

                # 计算伪标签正确的样本数
                correct_pseudo_labels = (targets_u == labels).float().sum().item()

                # 计算超过 threshold 的样本数量
                threshold_samples = mask.sum().item()

                # 计算超过 threshold 且伪标签正确的样本数量
                correct_threshold_samples = ((targets_u == labels).float() * mask).sum().item()
                #  Correct Pseudo Labels 弱扰动模型推理对的，弱扰动后超过threshold的
                print(
                    f"Batch {batch_idx}: Total Samples = {total_samples}, Correct Pseudo Labels = {correct_pseudo_labels}, Threshold Samples = {threshold_samples}, Correct Threshold Samples = {correct_threshold_samples}")

                optimizer.zero_grad()
                Lu.backward()
                optimizer.step()

                batch_loss.append(Lu.item())
                #
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            scheduler.step(int(local_curr_ep))

            # 在每个 epoch 结束后进行评估
            # accuracy_after = self.evaluate_accuracy(self.model, self.trainloader, self.device)
            # print(f'User: {self.id} \tEpoch {iter + 1}/{self.args.local_ep}, Test Accuracy: {accuracy_after:.4f}%')
        self.optimizer = optimizer
        self.model = model
        # accuracy_after, loss_after, correct_after, total_after = self.inference(self.model)
        # 打印训练之后的值
        # print(f'User: {self.id} \tEpoch: {global_round} \tAccuracy after training: {100.*accuracy_after:.4f}')
        # print(
        #     f"Loss after training: {loss_after}, Correct after training: {correct_after}, Total after training: {total_after}")

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def update_semi_weights_withoutproto(self, model, global_round, additional_feature_banks=None):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        ema_model = model.to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = self.args.lr
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=1e-6
            )

        # 如果继续训练，加载优化器状态
        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())
        schedule = [
            int(self.args.epochs * 0.3),
            int(self.args.epochs * 0.6),
            int(self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )

        criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
        criterion_mse = torch.nn.MSELoss().to(self.device)
        initial_model = copy.deepcopy(self.model)

        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                self.model.eval()
                images, labels = images.to(self.device), labels.to(self.device)
                total_samples = images.size(0)

                # 计算原始数据的输出
                logits_u, features = self.model(images)
                probabilities = torch.nn.functional.softmax(logits_u, dim=1)
                transformed_probs = probabilities
                confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
                max_probs, targets_u = torch.max(confidence_scores, dim=-1)
                mask = max_probs.ge(self.args.threshold).float()

                loss_u = 0.0
                loss_u_T = 0.0
                over_thre = 0

                for i in range(total_samples):
                    loss_u += criterion_ce(logits_u[i], targets_u[i])
                    if mask[i]:
                        over_thre += 1
                        loss_u_T += criterion_ce(logits_u[i], targets_u[i])
                autual_acc = self.inference(self.model)
                print(f"|---- Before Batch {batch_idx}-------")
                print(f"real_acc_for_current_model {autual_acc}")
                print(
                    f"Batch {batch_idx}: Loss = {loss_u}/{loss_u / total_samples}, Loss_u_T = {loss_u_T}/{loss_u_T / over_thre}")
                correct_predictions = targets_u == labels
                inference_accuracy = correct_predictions.float().mean().item()

                weak, strong = self.transform(images)
                weak, strong = weak.to(self.device), strong.to(self.device)

                logits_u_w, features_w = self.model(weak)
                logits_u_s, features_s = self.model(strong)
                probabilities_weak = torch.nn.functional.softmax(logits_u_w, dim=1)
                transformed_probs = probabilities_weak
                confidence_scores_weak = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
                max_probs_weak, targets_u_weak = torch.max(confidence_scores_weak, dim=-1)
                mask_weak = max_probs_weak.ge(self.args.threshold).float()

                loss_s = 0.0
                loss_w = 0.0
                correct_over_thre = 0
                over_thre = 0
                num_weak = 0
                num_strong = 0

                for i in range(total_samples):
                    if mask[i]:
                        over_thre += 1
                        if targets_u[i] == labels[i].item():
                            correct_over_thre += 1
                        if int(mask_weak[i]) & targets_u_weak[i] == targets_u[i]:
                            loss_s += criterion_ce(logits_u_s[i], targets_u[i])
                            num_strong += 1
                        else:
                            loss_w += criterion_ce(logits_u_w[i], targets_u[i])
                            num_weak += 1

                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                # Lu = loss_s/max(num_strong,1) + (num_weak/total_samples)*loss_w/max(num_weak, 1) + self.args.weight_decay * l1_norm
                Lu = loss_s/max(num_strong,1) + (num_weak/total_samples)*loss_w/max(num_weak, 1)

                correct_pseudo_labels = (targets_u == labels).float().sum().item()
                print(
                    f"Batch {batch_idx}: Correct pseudo labels: {correct_pseudo_labels}, Loss_s = {loss_s/max(num_strong,1)}, Loss_w = {(num_weak/total_samples)*loss_w/max(num_weak, 1)}, l1_norm = {self.args.weight_decay*l1_norm}")
                print(f"Model inference accuracy: {inference_accuracy * 100:.2f}%")
                print(f"Over_thre: {over_thre}")
                print(f"Correct over_thre: {correct_over_thre}")
                self.model.train()

                optimizer.zero_grad()
                Lu.backward()
                optimizer.step()

                print(f"|---- After Batch {batch_idx}-------")

                # 打印参数更新后的模型参数
                # print("Parameters after optimizer step:")
                # for name, param in self.model.named_parameters():
                #     print(name, param.data)

                epoch_loss.append(Lu.item())
                self.model.eval()

                # 再次计算模型输出
                logits_u, features = self.model(images)
                probabilities = torch.nn.functional.softmax(logits_u, dim=1)
                transformed_probs = probabilities
                confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
                max_probs, targets_u = torch.max(confidence_scores, dim=-1)
                mask = max_probs.ge(self.args.threshold).float()
                correct_predictions = targets_u == labels
                inference_accuracy = correct_predictions.float().mean().item()
                autual_acc = self.inference(self.model)
                autual_acc_1 = self.inference(self.model)
                autual_acc_2 = self.inference(self.model)

                print(f"real_acc_for_current_model {autual_acc}, {autual_acc_1}, {autual_acc_2}")
                print(f"Model inference accuracy: {inference_accuracy * 100:.2f}%")
                print(f"Over_thre: {mask.sum().item()}")
                print(f"Correct over_thre: {((targets_u == labels).float() * mask).sum().item()}")
                loss_u = 0.0
                loss_u_T = 0.0
                over_thre = 0
                for i in range(total_samples):
                    loss_u += criterion_ce(logits_u[i], targets_u[i])
                    if mask[i]:
                        over_thre += 1
                        loss_u_T += criterion_ce(logits_u[i], targets_u[i])
                print(
                    f"Batch {batch_idx}: Loss = {loss_u}/{loss_u / total_samples}, Loss_u_T = {loss_u_T}/{loss_u_T / over_thre}")

            breakpoint()  # 这里添加断点



            scheduler.step(int(local_curr_ep))
        self.model.eval()
        self.model.to(self.device)
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        self.optimizer = optimizer
        self.model = model

        return self.model.state_dict(), sum(epoch_loss) / len(
            epoch_loss), self.critical_parameter, self.global_mask, self.local_mask

    # def update_semi_weights_myself(self, model, global_round, agg_protos, additional_feature_banks=None):
    #     self.model = model  # 设置 self.model 属性
    #     self.model.to(self.device)
    #     self.model.train()
    #     epoch_loss = []
    #     ema_model = model.to(self.device)
    #     ema_model.load_state_dict(self.model.state_dict())
    #     # 选择优化器
    #     if self.args.optimizer == "sgd":
    #         train_lr = self.args.lr * (self.args.batch_size / 256)
    #         if self.args.distributed_training:
    #             train_lr = train_lr * self.args.world_size
    #         optimizer = torch.optim.SGD(
    #             self.model.parameters(),
    #             lr=train_lr,
    #             momentum=self.args.momentum,
    #             weight_decay=self.args.weight_decay,
    #         )
    #     elif self.args.optimizer == "adam":
    #         optimizer = torch.optim.Adam(
    #             self.model.parameters(), lr=self.args.lr, weight_decay=1e-6
    #         )
    #
    #     # 如果继续训练，加载优化器状态
    #     if self.args.model_continue_training and hasattr(self, "optimizer"):
    #         optimizer.load_state_dict(self.optimizer.state_dict())
    #     schedule = [
    #         int(self.args.epochs * 0.3),
    #         int(self.args.epochs * 0.6),
    #         int(self.args.epochs * 0.9),
    #     ]
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, milestones=schedule, gamma=0.3
    #     )
    #
    #     criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
    #     criterion_mse = torch.nn.MSELoss().to(self.device)
    #     initial_model = copy.deepcopy(self.model)
    #
    #     for iter in range(int(self.args.local_ep)):
    #         local_curr_ep = self.args.local_ep * global_round + iter
    #
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.trainloader):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             total_samples = images.size(0)
    #
    #             # 计算原始数据的输出
    #             logits_u, features = self.model(images)
    #             probabilities = torch.nn.functional.softmax(logits_u, dim=1)
    #             transformed_probs = probabilities ** (1 / self.args.T)
    #             confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
    #             # 模型预测结果
    #             max_probs, targets_u = torch.max(confidence_scores, dim=-1)
    #             mask = max_probs.ge(self.args.threshold).float()
    #
    #             # 计算余弦相似度和原型标签
    #             cos_sims = []
    #             proto_labels = []
    #             for feature in features:
    #                 single_cos_sims = []
    #                 for key, proto_tensor in agg_protos.items():
    #                     cos_sim = torch.nn.functional.cosine_similarity(feature.unsqueeze(0), proto_tensor, dim=1)
    #                     single_cos_sims.append(cos_sim)
    #                 single_cos_sims = torch.stack(single_cos_sims, dim=0)
    #                 max_cos_sim, best_proto_idx = torch.max(single_cos_sims, dim=0)
    #                 cos_sims.append(max_cos_sim)
    #                 proto_labels.append(best_proto_idx.item())
    #             cos_sims = torch.stack(cos_sims, dim=0)
    #             # 计算特征向量相似度精度
    #             proto_correct_predictions = torch.tensor(proto_labels).to(self.device) == labels
    #             proto_accuracy = proto_correct_predictions.float().mean().item()
    #
    #             # 计算模型推理精度
    #             correct_predictions = targets_u == labels
    #             inference_accuracy = correct_predictions.float().mean().item()
    #             # proto_labels = torch.tensor(proto_labels, device=self.device)
    #             # 调试输出
    #             print(f"|---- Before Batch {batch_idx}-------")
    #             print(f"Feature vector similarity accuracy: {proto_accuracy * 100:.2f}%")
    #             print(f"Model inference accuracy: {inference_accuracy * 100:.2f}%")
    #             # for i in range(total_samples):
    #             #     print(
    #             #         f"Sample {i}: label={labels[i].item()}, targets_u={targets_u[i].item()}, max_probs={max_probs[i].item()}, proto_labels={proto_labels[i]}, cos_sims={cos_sims[i].item()}")
    #             # 对输入进行增强
    #             weak, strong = self.transform(images)
    #             weak, strong = weak.to(self.device), strong.to(self.device)
    #
    #             # 计算强增强的输出
    #             logits_u_w, features_w = self.model(weak)
    #             logits_u_s, features_s = self.model(strong)
    #             # 计算原始数据的输出
    #             probabilities_weak = torch.nn.functional.softmax(logits_u_w, dim=1)
    #             transformed_probs = probabilities_weak ** (1 / self.args.T)
    #             confidence_scores_weak = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
    #             # 模型预测结果
    #             max_probs_weak, targets_u_weak = torch.max(confidence_scores_weak, dim=-1)
    #             mask_weak = max_probs_weak.ge(self.args.threshold).float()
    #
    #             loss_s = 0.0
    #             loss_p = 0.0
    #             recorrect = 0
    #             correct_over_thre = 0
    #             over_thre = 0
    #             inter_class_loss = 0
    #             inter_class_loss_num = 0
    #             margin = 1
    #             for i in range(total_samples):
    #                 if mask[i]:
    #                     over_thre += 1
    #                     if targets_u[i] == proto_labels[i]:
    #                         if targets_u[i]==labels[i].item():
    #                             correct_over_thre +=1
    #                         if int(mask_weak[i]) & targets_u_weak[i] == targets_u[i] :
    #                             loss_s += criterion_ce(logits_u_s[i], targets_u[i])
    #                             loss_p += criterion_mse(features_s[i], agg_protos[proto_labels[i]])
    #                         else:
    #                             loss_s += criterion_ce(logits_u_w[i], targets_u[i])
    #                             loss_p += criterion_mse(features_w[i], agg_protos[proto_labels[i]])
    #
    #                     else:
    #                         if cos_sims[i] < 0.85:
    #                             inter_class_loss_num += 1
    #                             if targets_u[i] == labels[i].item():
    #                                 correct_over_thre += 1
    #                             loss_s += criterion_ce(logits_u_w[i], targets_u[i])
    #                             target_key = targets_u[i].item() if isinstance(targets_u[i], torch.Tensor) else \
    #                                 targets_u[i]
    #                             # print(f"agg_protos shape: {features_w[i].shape}")
    #                             # print(f"proto_labels shape: {agg_protos[proto_labels[i]].shape}")
    #                             inter_class_loss += torch.clamp(
    #                                 margin - F.cosine_similarity(features_w[i].unsqueeze(0), agg_protos[proto_labels[i]].unsqueeze(0)),
    #                                 min=0)
    #
    #                             if target_key in agg_protos:
    #                                 loss_p += criterion_mse(features_w[i], agg_protos[target_key])
    #
    #                             else: return false
    #                         else:
    #                             if proto_labels[i] == labels[i].item():
    #                                 recorrect += 1
    #                                 correct_over_thre += 1
    #                             loss_s += criterion_ce(logits_u_w[i], torch.tensor(proto_labels[i], device=self.device))
    #                             loss_p += criterion_mse(features_w[i], agg_protos[proto_labels[i]])
    #
    #             # 总损失
    #
    #             Lu = (loss_s + loss_p) / over_thre + inter_class_loss/inter_class_loss_num
    #             # for i in range(total_samples):
    #             #     if mask[i]:
    #             #         over_thre += 1
    #             #         if targets_u[i] == proto_labels[i]:
    #             #             if targets_u[i]==labels[i].item():
    #             #                 correct_over_thre +=1
    #             #             if cos_sims[i] < 0.85:
    #             #                 loss_s += criterion_ce(logits_u_w[i], targets_u[i])
    #             #                 loss_p += criterion_mse(features_w[i], agg_protos[proto_labels[i]])
    #             #             else:
    #             #                 if
    #             #                 loss_s += criterion_ce(logits_u_s[i], targets_u[i])
    #             #                 loss_p += criterion_mse(features_s[i], agg_protos[proto_labels[i]])
    #             #         else:
    #             #             if cos_sims[i] < 0.85:
    #             #                 inter_class_loss_num += 1
    #             #                 if targets_u[i] == labels[i].item():
    #             #                     correct_over_thre += 1
    #             #                 loss_s += criterion_ce(logits_u_w[i], targets_u[i])
    #             #                 target_key = targets_u[i].item() if isinstance(targets_u[i], torch.Tensor) else \
    #             #                     targets_u[i]
    #             #                 # print(f"agg_protos shape: {features_w[i].shape}")
    #             #                 # print(f"proto_labels shape: {agg_protos[proto_labels[i]].shape}")
    #             #                 inter_class_loss += torch.clamp(
    #             #                     margin - F.cosine_similarity(features_w[i].unsqueeze(0), agg_protos[proto_labels[i]].unsqueeze(0)),
    #             #                     min=0)
    #             #
    #             #                 if target_key in agg_protos:
    #             #                     loss_p += criterion_mse(features_w[i], agg_protos[target_key])
    #             #
    #             #                 else: return false
    #             #             else:
    #             #                 if proto_labels[i] == labels[i].item():
    #             #                     recorrect += 1
    #             #                     correct_over_thre += 1
    #             #                 loss_s += criterion_ce(logits_u_w[i], torch.tensor(proto_labels[i], device=self.device))
    #             #                 loss_p += criterion_mse(features_w[i], agg_protos[proto_labels[i]])
    #             #
    #             # # 总损失
    #             #
    #             # Lu = (loss_s + loss_p) / over_thre + inter_class_loss/inter_class_loss_num
    #             correct_pseudo_labels = (targets_u == labels).float().sum().item()
    #             # 对输入进行增强
    #             print(
    #                 f"Batch {batch_idx}: Correct pseudo labels: {correct_pseudo_labels}, Loss_s = {loss_s.item()/over_thre}, Loss_p = {loss_p.item()/over_thre}, Inter_Class_Loss = {inter_class_loss/inter_class_loss_num}")
    #
    #             print(f"Over_thre: {over_thre}")
    #             print(f"Correct over_thre: {correct_over_thre}")
    #             print(f"Recorrect: {recorrect}")
    #             optimizer.zero_grad()
    #             Lu.backward()
    #             print(f"|---- After Batch {batch_idx}-------")
    #
    #             # 打印梯度
    #             # for name, param in model.named_parameters():
    #             #     print(f"Gradient of {name} before optimizer.step(): {param.grad}")
    #             optimizer.step()
    #             epoch_loss.append(Lu.item())
    #             images, labels = images.to(self.device), labels.to(self.device)
    #
    #             # 计算原始数据的输出
    #             logits_u, features = self.model(images)
    #             probabilities = torch.nn.functional.softmax(logits_u, dim=1)
    #             transformed_probs = probabilities ** (1 / self.args.T)
    #             confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
    #             # 模型预测结果
    #             max_probs, targets_u = torch.max(confidence_scores, dim=-1)
    #             mask = max_probs.ge(self.args.threshold).float()
    #
    #             # 计算余弦相似度和原型标签
    #             cos_sims = []
    #             proto_labels = []
    #             for feature in features:
    #                 single_cos_sims = []
    #                 for key, proto_tensor in agg_protos.items():
    #                     cos_sim = torch.nn.functional.cosine_similarity(feature.unsqueeze(0), proto_tensor, dim=1)
    #                     single_cos_sims.append(cos_sim)
    #                 single_cos_sims = torch.stack(single_cos_sims, dim=0)
    #                 max_cos_sim, best_proto_idx = torch.max(single_cos_sims, dim=0)
    #                 cos_sims.append(max_cos_sim)
    #                 proto_labels.append(best_proto_idx.item())
    #             cos_sims = torch.stack(cos_sims, dim=0)
    #             # 计算特征向量相似度精度
    #             proto_correct_predictions = torch.tensor(proto_labels).to(self.device) == labels
    #             proto_accuracy = proto_correct_predictions.float().mean().item()
    #
    #             # 计算模型推理精度
    #             correct_predictions = targets_u == labels
    #             inference_accuracy = correct_predictions.float().mean().item()
    #             # proto_labels = torch.tensor(proto_labels, device=self.device)
    #             # 调试输出
    #             print(f"Feature vector similarity accuracy: {proto_accuracy * 100:.2f}%")
    #             print(f"Model inference accuracy: {inference_accuracy * 100:.2f}%")
    #             # 打印结果
    #
    #
    #             # proto_labels = torch.tensor(proto_labels, device=self.device)
    #             # 调试输出
    #             # print(f"|---- After Batch {batch_idx}-------")
    #             # for i in range(total_samples):
    #             #     print(
    #             #         f"Sample {i}: label={labels[i].item()}, targets_u={targets_u[i].item()}, max_probs={max_probs[i].item()}, proto_labels={proto_labels[i]}, cos_sims={cos_sims[i].item()}")
    #             # 对输入进行增强
    #             # ema方式更新
    #             update_ema(ema_model, self.model, 0.999)
    #             self.model.load_state_dict(ema_model.state_dict())
    #
    #             # print(f"|---- After Batch {batch_idx}-------")
    #             # for i in range(total_samples):
    #             #     print(
    #             #         f"Sample {i}: label={labels[i].item()}, targets_u={targets_u[i].item()}, max_probs={max_probs[i].item()}, proto_labels={proto_labels[i]}, cos_sims={cos_sims[i].item()}")
    #
    #         scheduler.step(int(local_curr_ep))
    #     self.model.eval()
    #     self.model.to(self.device)
    #     self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
    #         prevModel=initial_model, model=self.model, tau=self.args.tau
    #     )
    #
    #     self.optimizer = optimizer
    #     self.model = model
    #
    #     return self.model.state_dict(), sum(epoch_loss) / len(
    #         epoch_loss), self.critical_parameter, self.global_mask, self.local_mask


    def update_semi_weights_STRONG(self, model, global_round, additional_feature_banks=None):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=1e-6
            )

        # 如果继续训练，加载优化器状态
        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())
        schedule = [
            int(self.args.epochs * 0.3),
            int(self.args.epochs * 0.6),
            int(self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter

            # 在每个 epoch 开始前计算并打印准确率
            # accuracy_before, loss_before, correct_before, total_before = self.inference(self.model)
            # print(f'User: {self.id} \tEpoch: {iter} \tAccuracy before training: {100.*accuracy_before:.4f}%')
            # print(
            #     f"Loss before training: {loss_before}, Correct before training: {correct_before}, Total before training: {total_before}")

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # 计算原始数据的输出
                logits_u_w, _ = self.model(images)
                probabilities = torch.nn.functional.softmax(logits_u_w, dim=1)
                transformed_probs = probabilities ** (1 / self.args.T)
                confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
                max_probs, targets_u = torch.max(confidence_scores, dim=-1)
                mask = max_probs.ge(self.args.threshold).float()
                # 进行强数据增强
                _, strong = self.transform(images)
                strong = strong.to(self.device)

                # 使用强增强的数据进行推理
                logits_u_s, _ = self.model(strong)

                # 计算交叉熵损失
                Lu = (criterion(logits_u_s, targets_u) * mask).mean()

                # 计算当前 batch 总样本数
                total_samples = images.size(0)

                # 计算伪标签正确的样本数
                correct_pseudo_labels = (targets_u == labels).float().sum().item()

                # 计算超过 threshold 的样本数量
                threshold_samples = mask.sum().item()

                # 计算超过 threshold 且伪标签正确的样本数量
                correct_threshold_samples = ((targets_u == labels).float() * mask).sum().item()

                print(
                    f"Batch {batch_idx}: Total Samples = {total_samples}, Correct Pseudo Labels = {correct_pseudo_labels}, Threshold Samples = {threshold_samples}, Correct Threshold Samples = {correct_threshold_samples}")

                optimizer.zero_grad()
                Lu.backward()
                optimizer.step()

                batch_loss.append(Lu.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            scheduler.step(int(local_curr_ep))

        self.optimizer = optimizer
        self.model = model

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def update_semi_weights_fling(self, model, global_round, additional_feature_banks=None):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=1e-6
            )

        # 如果继续训练，加载优化器状态
        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())
        schedule = [
            int(self.args.epochs * 0.3),
            int(self.args.epochs * 0.6),
            int(self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # 记录当前模型
        initial_model = copy.deepcopy(self.model)

        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter

            # 在每个 epoch 开始前计算并打印准确率
            # accuracy_before, loss_before, correct_before, total_before = self.inference(self.model)
            # print(f'User: {self.id} \tEpoch: {iter} \tAccuracy before training: {100.*accuracy_before:.4f}%')
            # print(
            #     f"Loss before training: {loss_before}, Correct before training: {correct_before}, Total before training: {total_before}")

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # 计算原始数据的输出
                logits_u_w, _ = self.model(images)
                probabilities = torch.nn.functional.softmax(logits_u_w, dim=1)
                transformed_probs = probabilities ** (1 / self.args.T)
                confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
                max_probs, targets_u = torch.max(confidence_scores, dim=-1)
                mask = max_probs.ge(self.args.threshold).float()
                # 进行强数据增强
                _, strong = self.transform(images)
                strong = strong.to(self.device)

                # 使用强增强的数据进行推理
                logits_u_s, _ = self.model(strong)

                # 计算交叉熵损失
                Lu = (criterion(logits_u_s, targets_u) * mask).mean()

                # 计算当前 batch 总样本数
                total_samples = images.size(0)

                # 计算伪标签正确的样本数
                correct_pseudo_labels = (targets_u == labels).float().sum().item()

                # 计算超过 threshold 的样本数量
                threshold_samples = mask.sum().item()

                # 计算超过 threshold 且伪标签正确的样本数量
                correct_threshold_samples = ((targets_u == labels).float() * mask).sum().item()

                print(
                    f"Batch {batch_idx}: Total Samples = {total_samples}, Correct Pseudo Labels = {correct_pseudo_labels}, Threshold Samples = {threshold_samples}, Correct Threshold Samples = {correct_threshold_samples}")

                optimizer.zero_grad()
                Lu.backward()
                optimizer.step()

                batch_loss.append(Lu.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            scheduler.step(int(local_curr_ep))

            # 计算关键参数
        self.model.eval()
        self.model.to(self.device)
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        self.optimizer = optimizer
        self.model = model

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.critical_parameter, self.global_mask, self.local_mask

    def evaluate_critical_parameter(self, prevModel, model,
                                    tau) -> Tuple[torch.Tensor, list, list]:
        r"""
        Overview:
            Implement critical parameter selection.
        """
        global_mask = []  # 用于标记非关键参数
        local_mask = []  # 用于标记关键参数
        critical_parameter = []  # 存储所有层的关键参数

        self.model.to(self.device)
        prevModel.to(self.device)

        # 在每一层选择特定数量的关键参数
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            # 计算参数的变化值
            g = (param.data - prevparam.data)
            v = param.data
            # 计算非负变化量
            c = torch.abs(g * v)
            # 将变化量展平为一维张量
            metric = c.view(-1)
            num_params = metric.size(0)
            # 根据 tau 参数确定要选择的关键参数数量
            nz = int(tau * num_params)
            # 选择前 nz 个最大的变化量
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            # 如果阈值小于等于 1e-10，选择最小的非零元素作为阈值
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-20]
                if len(new_metric) == 0:  # 如果所有元素都是零
                    print(f'Abnormal!!! metric:{metric}')
                else:
                    thresh = new_metric.sort()[0][0]

            # 生成局部掩码，将敏感性度量大于等于阈值的参数标记为1
            mask = (c >= thresh).int().to('cpu')
            # 生成全局掩码，将敏感性度量小于阈值的参数标记为1
            global_mask.append((c < thresh).int().to('cpu'))
            local_mask.append(mask)
            critical_parameter.append(mask.view(-1))
            # 打印每一层的名称、掩码形状和掩码内容（此处注释掉了）
            # print(f"Layer: {name1}, Mask Shape: {mask.shape}, Mask: {mask.numpy()}")

        model.zero_grad()
        # 将所有层的关键参数拼接在一起
        critical_parameter = torch.cat(critical_parameter)

        self.model.to('cpu')
        prevModel.to('cpu')

        # 打印 global_mask 的长度（此处注释掉了）
        # for idx, g_mask in enumerate(global_mask):
        #     print(f"Layer {idx} global_mask shape: {g_mask.shape}")

        # print(f'Global mask length: {len(global_mask)}')

        return critical_parameter, global_mask, local_mask

    # 测试弱增强和强增强后的准确率
    def confidence_qujian(self, model, temperature=0.5):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        confidence_buckets_correct = [0] * 20  # To store counts for confidence intervals of correct predictions
        confidence_buckets_incorrect = [0] * 20  # To store counts for confidence intervals of incorrect predictions
        confidence_sample_counts = [0] * 20  # To store total samples in each confidence interval

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # 进行强弱数据增强
            weak, strong = self.transform(images)
            images, labels = weak.to(self.device), weak.to(self.device)

            # Inference
            outputs, _ = model(images)
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

            # Count samples in different confidence intervals
            for i in range(20):
                lower_bound = i * 5
                upper_bound = (i + 1) * 5
                bucket_mask = (confidence_scores >= lower_bound) & (confidence_scores < upper_bound)
                bucket_size = torch.sum(bucket_mask).item()
                confidence_sample_counts[i] += bucket_size

                # Count correct samples in this confidence interval
                confidence_correct_mask = bucket_mask & correct_mask
                confidence_buckets_correct[i] += torch.sum(confidence_correct_mask).item()

                # Count incorrect samples in this confidence interval
                confidence_buckets_incorrect[i] += torch.sum(bucket_mask & ~correct_mask).item()

        # Calculate percentages for each confidence interval relative to total samples in each interval
        confidence_percentages_correct = [
            (bucket_correct_count / bucket_total_count * 100 if bucket_total_count > 0 else 0.0)
            for bucket_correct_count, bucket_total_count in
            zip(confidence_buckets_correct, confidence_sample_counts)]
        confidence_percentages_incorrect = [
            (bucket_incorrect_count / bucket_total_count * 100 if bucket_total_count > 0 else 0.0)
            for bucket_incorrect_count, bucket_total_count in
            zip(confidence_buckets_incorrect, confidence_sample_counts)]

        total_samples = sum(confidence_sample_counts)
        confidence_sample_percentages = [bucket_size / total_samples * 100 for bucket_size in confidence_sample_counts]

        accuracy = correct / total

        # Return accuracy, loss, correct, total, confidence percentages for correct and incorrect predictions,
        # confidence sample percentages, correct samples count and incorrect samples count for each interval
        return accuracy, loss, correct, total, confidence_percentages_correct, confidence_percentages_incorrect, confidence_sample_percentages, confidence_buckets_correct, confidence_buckets_incorrect

    # def update_local_protos(self, model, dataset=None, test_user=None,agg_protos_label=None, temperature=0.8):
    #     """Returns the test accuracy and loss."""
    #     model.eval()
    #     model = model.to(self.device)
    #     for batch_idx, (images, labels) in enumerate(self.trainloader):
    #         images, labels = images.to(self.device), labels.to(self.device)
    #         with torch.no_grad():  # 确保不记录计算图
    #             log_probs, protos = model(images)
    #             probabilities = torch.nn.functional.softmax(log_probs, dim=1)
    #
    #             # Apply the custom confidence calculation
    #             transformed_probs = probabilities ** (1 / temperature)
    #             confidence_scores = transformed_probs / transformed_probs.sum(dim=1, keepdim=True)
    #             confidence_scores = confidence_scores.max(dim=1).values * 100  # Convert to percentage
    #
    #             # Inference
    #             pred_labels = torch.argmax(probabilities, dim=1)
    #             for i in range(len(labels)):
    #                 if confidence_scores[i] >= 95:
    #                     pred_label = pred_labels[i].item()
    #                     if pred_label in agg_protos_label:
    #                         agg_protos_label[pred_label].append(protos[i, :])
    #                     else:
    #                         agg_protos_label[pred_label] = [protos[i, :]]
    #                 else:
    #                     continue
    #     return agg_protos_label
    # 不用temperature，加上数量
    def update_local_protos(self, model, dataset=None, test_user=None, agg_protos_label=None):
        """Returns the test accuracy and loss."""
        model.eval()
        model = model.to(self.device)
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():  # 确保不记录计算图
                log_probs, protos = model(images)
                probabilities = torch.nn.functional.softmax(log_probs, dim=1)

                # Apply the custom confidence calculation
                transformed_probs = probabilities
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
    # def inference(self, model):
    #     """Returns the accuracy on test data for no perturbation, weak perturbation, and strong perturbation."""
    #     model.eval()
    #     correct_normal, correct_weak, correct_strong = 0.0, 0.0, 0.0
    #     total = 0.0
    #
    #     for batch_idx, (images, labels) in enumerate(self.testloader):
    #         images, labels = images.to(self.device), labels.to(self.device)
    #
    #         # Inference without perturbation
    #         outputs_normal, _ = model(images)
    #         _, pred_labels_normal = torch.max(outputs_normal, 1)
    #         pred_labels_normal = pred_labels_normal.view(-1)
    #         correct_normal += torch.sum(torch.eq(pred_labels_normal, labels)).item()
    #
    #         # Apply weak and strong perturbations
    #         weak_images, strong_images = self.transform(images)
    #         weak_images, strong_images = weak_images.to(self.device), strong_images.to(self.device)
    #
    #         # Inference with weak perturbation
    #         outputs_weak, _ = model(weak_images)
    #         _, pred_labels_weak = torch.max(outputs_weak, 1)
    #         pred_labels_weak = pred_labels_weak.view(-1)
    #         correct_weak += torch.sum(torch.eq(pred_labels_weak, labels)).item()
    #
    #         # Inference with strong perturbation
    #         outputs_strong, _ = model(strong_images)
    #         _, pred_labels_strong = torch.max(outputs_strong, 1)
    #         pred_labels_strong = pred_labels_strong.view(-1)
    #         correct_strong += torch.sum(torch.eq(pred_labels_strong, labels)).item()
    #
    #         total += len(labels)
    #
    #     accuracy_normal = correct_normal / total
    #     accuracy_weak = correct_weak / total
    #     accuracy_strong = correct_strong / total
    #
    #     return accuracy_normal, accuracy_weak, accuracy_strong, total
    def inference(self, model):
        """Returns the inference accuracy and loss for a local client."""
        model.to(self.device)  # 将模型移动到指定设备
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.loader):
            print(f'Batch {batch_idx} - images type: {type(images)}, labels type: {type(labels)}')
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

        accuracy = correct / total
        return accuracy, loss, correct, total


    def finetune(self, lr, finetune_args, device=None, finetune_eps=None, override=False):
        """
        Finetune function. In this function, the local model will not be changed, but will return the finetune results.
        """
        # Back-up variables.
        if device is not None:
            device_bak = self.device
            self.device = device
        if not override:
            model_bak = copy.deepcopy(self.model)

        # Get default ``finetune_eps``.
        if finetune_eps is None:
            finetune_eps = self.args.learn.local_eps

        self.model.train()
        self.model.to(self.device)

        # Get weights to be fine-tuned.
        # For calculating train loss and train acc.
        weights = get_weights(self.model, parameter_args=finetune_args)

        # Get optimizer and loss.
        op = get_optimizer(weights=weights, **self.args.learn.optimizer)
        criterion = nn.CrossEntropyLoss()

        # Main loop.
        info = []
        for epoch in range(finetune_eps):
            self.model.train()
            self.model.to(self.device)
            monitor = VariableMonitor()
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.finetune_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=op)

            # Test model every epoch.
            mean_monitor_variables = monitor.variable_mean()
            mean_monitor_variables.update(self.test())
            info.append(mean_monitor_variables)

        # Retrieve the back-up variables.
        if not override:
            self.model = model_bak
        else:
            # Put the model to cpu after training to save GPU memory.
            self.model.to('cpu')
        if device is not None:
            self.device = device_bak

        return info

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
def update_global_protos(args, model, dataset,agg_protos_label):
    """Returns the test accuracy and loss."""
    model.eval()
    device = "cuda" if args.gpu else "cpu"
    model = model.to(device)
    globalloader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    test_bar = tqdm((globalloader), desc="feature extraction", disable=False)

    for (images, labels) in test_bar:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():  # 确保不记录计算图
            log_probs, protos = model(images)
            # Inference
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [protos[i, :]]
    return agg_protos_label

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


def update_server_weight(args, global_model=None, test_epoch=1, test_train_dataset=None):
    # global representation, global classifier
    from models import ResNetCifarClassifier
    import torch
    # 设置随机种子

    device = "cuda"
    # define model

    test_train_loader = DataLoader(
        test_train_dataset,
        batch_size=250,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    print("begin training ...")
    if args.ssl_method == "mae":
        global_model_classifer = ViT_Classifier(
            global_model.encoder, num_classes=10
        ).to(device)
        global_model_classifer = global_model_classifer.cuda()
        optimizer = torch.optim.AdamW(
            global_model_classifer.head.parameters(), lr=3e-4, weight_decay=0.05
        )

        # Move to GPU
    global_model.cuda()

    optimizer = torch.optim.SGD(
        global_model.parameters(),
        lr=0.03,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # sample
    # dist_sampler = (
    #     DistributedSampler(train_dataset)
    #     if args.distributed_training
    #     else RandomSampler(train_dataset)
    # )
    # 创建测试训练集的 DataLoader

    # trainloader = DataLoader(
    #     train_dataset,
    #     sampler=dist_sampler,
    #     batch_size=256,
    #     num_workers=16,
    #     pin_memory=False,
    # )
    criterion = (
        torch.nn.NLLLoss().to(device)
        if args.ssl_method != "mae"
        else torch.nn.CrossEntropyLoss().to(device)
    )

    # train global model on global dataset
    for epoch_idx in tqdm(range(test_epoch)):
        i=0
        if args.distributed_training:
            dist_sampler.set_epoch(epoch_idx)

        for batch_idx, (images, labels) in enumerate(test_train_loader):
            i += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs,_ = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'total number={i}')
    return global_model
# 可视化增强后的图像
def visualize_images(images, title, save_path='output', num_images=10):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 只选择前 num_images 张图像
    images = images[:num_images]
    grid = torchvision.utils.make_grid(images.cpu().data, nrow=4, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"{title}.png"))
    plt.close()
# 弱增强

def update_server_weight_save(args, global_model=None, test_epoch=1, test_train_dataset=None, test_val_dataset=None):
    # global representation, global classifier
    from models import ResNetCifarClassifier
    import torch
    from torch.optim.lr_scheduler import StepLR  # Import the scheduler

    device = "cuda"
    # define model
    model_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "_{}".format(
        str(os.getpid())
    )  # to avoid collision
    args.model_time = model_time
    model_output_dir = "save/" + model_time
    save_args_json(model_output_dir, args)
    test_train_loader = DataLoader(
        test_train_dataset,
        batch_size=250,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    print("begin training ...")
    if args.ssl_method == "mae":
        global_model_classifer = ViT_Classifier(
            global_model.encoder, num_classes=10
        ).to(device)
        global_model_classifer = global_model_classifer.cuda()
        optimizer = torch.optim.AdamW(
            global_model_classifer.head.parameters(), lr=3e-4, weight_decay=0.05
        )

    # Move to GPU
    global_model.cuda()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        global_model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4
    )

    # Add the learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=15, gamma=0.9)  # Adjust step_size and gamma as needed

    best_acc = 0.0
    # train global model on global dataset
    for epoch_idx in tqdm(range(test_epoch)):
        if args.distributed_training:
            dist_sampler.set_epoch(epoch_idx)

        for batch_idx, (images, labels) in enumerate(test_train_loader):
            images, labels = images.to(device), labels.to(device)
            transform = TransformFixMatch(mean=args.mean, std=args.std)
            # 进行强弱数据增强
            weak, _ = transform(images)

            # 移动到设备
            weak = weak.to(device)

            # 计算弱增强的输出
            outputs, _ = global_model(weak)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Step the scheduler at the end of the epoch
        # scheduler.step()

        test_acc, test_loss = test_inference(args, global_model, test_val_dataset)
        if test_acc > best_acc:
            best_acc = test_acc
            global_model.save_model(model_output_dir, suffix=f"best_{best_acc}_epoch{epoch_idx}")
            print("\n Downstream Train loss: {} Acc: {}".format(test_loss, best_acc))

    return best_acc

def save_args_json(path, args):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "args.json")
    with open(arg_json, "w") as f:
        args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)
def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.to_pil = transforms.ToPILImage()
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            RandAugmentPC(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
        ])

    def __call__(self, x):
        weak_images = []
        strong_images = []
        for img in x:
            if isinstance(img, torch.Tensor):
                img = self.denormalize(img)  # 反归一化
                img = transforms.ToPILImage()(img)  # 转换为 PIL 图像
            weak = self.weak(img)
            strong = self.strong(img)
            weak_images.append(self.normalize(weak))
            strong_images.append(self.normalize(strong))
        return torch.stack(weak_images), torch.stack(strong_images)


# 包含数据增强函数
def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwarg):
    return img

def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)

def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)

def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs

def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs

class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img
