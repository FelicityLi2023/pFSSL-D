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
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple
import copy
from tqdm import tqdm
import torch.optim as optim
from scipy.special import kl_div

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])
        return {'data': data, 'target': label}



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

class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        return input

    def __len__(self):
        return self.size

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
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = None  # 初始化 model 属性
        self.numbda = torch.distributions.beta.Beta(torch.tensor(self.args.a), torch.tensor(self.args.a))

        # 添加数据增强变换
        self.transform = TransformFixMatch(mean=args.mean, std=args.std)
        # 初始化 memoryloader 属性
        self.memoryloader = DataLoader(memory_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True) if memory_dataset else None
        # 定义这两个
        self.critical_parameter = None  # record the critical parameter positions in FedCAC
        self.customized_model = copy.deepcopy(self.model)  # customized global model
        self.update_vectors = None  # 用于存储模型更新向量

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
                shuffle=True,
                num_workers=16,
                pin_memory=True,
                drop_last= False,
                generator=g
            )
            trainloader = DataLoader(
                train_dataset,
                batch_size=self.args.local_bs,
                shuffle=True,
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
        self.loader = loader
        self.trainloader = trainloader
        self.testloader = testloader

        return trainloader, testloader, loader

    # def generate_fixmix_datasets(self, model,epoch):
    #     # 使用模型对 dataset_all 进行预测
    #     model.to(self.device)  # 将模型移动到指定设备
    #     model.eval()
    #     fix_data = []
    #     fix_labels = []
    #     correct_high_confidence = 0
    #     total_high_confidence = 0
    #     with torch.no_grad():
    #         for batch_idx, (images, labels) in enumerate(self.loader):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             # predict
    #             logits_u, _ = model(images)
    #             softmax_probs = torch.nn.functional.softmax(logits_u, dim=1)
    #             confidence_scores = softmax_probs / softmax_probs.sum(dim=1, keepdim=True)
    #             max_probs, targets_u = torch.max(confidence_scores, dim=-1)
    #             mask = max_probs >= (self.args.threshold-epoch*0.02)  # 创建一个掩码，筛选出高于阈值的样本
    #             # 获取高置信度样本的数量和正确的样本数量
    #             total_high_confidence += mask.sum().item()
    #             correct_high_confidence += (targets_u[mask] == labels[mask]).sum().item()
    #
    #
    #             # 获取对应的样本和预测标签
    #             fix_data.extend(images[mask].cpu().tolist())
    #             fix_labels.extend(targets_u[mask].cpu().tolist())
    #             # 创建 fix_dataset 和 mix_dataset
    #         fix_dataset = CustomDataset(fix_data, fix_labels)
    #         mix_dataset = MixDataset(len(fix_dataset), fix_dataset)
    #         # 计算高置信度样本的准确率
    #         if total_high_confidence > 0:
    #             accuracy_high_confidence = correct_high_confidence / total_high_confidence
    #         else:
    #             accuracy_high_confidence = 0.0
    #         print(f"Fix dataset length: {len(fix_dataset)}, Mix dataset length: {len(mix_dataset)}")
    #
    #         print(f"Accuracy of high confidence samples: {accuracy_high_confidence:.4f}")
    #     return fix_dataset, mix_dataset
    def generate_fixmix_datasets(self, model):
        # 使用模型对 dataset_all 进行预测
        model.to(self.device)  # 将模型移动到指定设备
        model.eval()
        fix_data = []
        fix_labels = []
        correct_high_confidence = 0
        total_high_confidence = 0
        total_samples = 0  # Initialize total samples counter

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                total_samples += len(images)
                # predict
                logits_u, _ = model(images)
                softmax_probs = torch.nn.functional.softmax(logits_u, dim=1)
                confidence_scores = softmax_probs / softmax_probs.sum(dim=1, keepdim=True)
                max_probs, targets_u = torch.max(confidence_scores, dim=-1)
                mask = max_probs >= self.args.threshold  # 创建一个掩码，筛选出高于阈值的样本
                # 获取高置信度样本的数量和正确的样本数量
                total_high_confidence += mask.sum().item()
                correct_high_confidence += (targets_u[mask] == labels[mask]).sum().item()


                # 获取对应的样本和预测标签
                fix_data.extend(images[mask].cpu().tolist())
                fix_labels.extend(targets_u[mask].cpu().tolist())
                # 创建 fix_dataset 和 mix_dataset
            fix_dataset = CustomDataset(fix_data, fix_labels)
            mix_dataset = MixDataset(len(fix_dataset), fix_dataset)
            # 计算高置信度样本的准确率
            if total_high_confidence > 0:
                accuracy_high_confidence = correct_high_confidence / total_high_confidence
            else:
                accuracy_high_confidence = 0.0
            print(f"Fix dataset length: {len(fix_dataset)}, Mix dataset length: {len(mix_dataset)}")

            print(f"Accuracy of high confidence samples: {accuracy_high_confidence:.4f}")
        return fix_dataset, mix_dataset, total_samples

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


    def update_semi_weights_semi(self, lr, model, global_round, fix_dataset, mix_dataset, additional_feature_banks=None):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        ema_model = model.to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        dataset_size = len(fix_dataset)  # 获取数据集的大小
        batch_size = min(dataset_size // 7, 256)  # 计算批量大小


        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = lr*batch_size/256
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

        criterion_ce = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        criterion_mse = torch.nn.MSELoss().to(self.device)
        initial_model = copy.deepcopy(self.model)
        fix_loader = DataLoader(
            fix_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        mix_loader = DataLoader(
            mix_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter
            self.model.eval()
            autual_acc = self.inference(self.model)
            print(f"before_train： {autual_acc}")
            batch_loss = []
            for batch_idx, (fix_batch, mix_batch) in enumerate(zip(fix_loader, mix_loader)):
                fix_images, fix_labels = fix_batch['data'], fix_batch['target']
                mix_images, mix_labels = mix_batch['data'], mix_batch['target']

                # 将数据移动到设备
                fix_images, fix_labels = fix_images.to(self.device), fix_labels.to(self.device)
                mix_images, mix_labels = mix_images.to(self.device), mix_labels.to(self.device)

                # 采样 mix-up 的参数 lam, 随机
                lam = self.numbda.sample().item()

                # 将 fix_images 和 mix_images 加权混合
                combined_images = lam * fix_images + (1 - lam) * mix_images

                # 处理 strong augmentation
                _, strong = self.transform(fix_images)
                strong = strong.to(self.device)
                logits_u_s, _ = self.model(strong)
                loss = criterion_ce(logits_u_s, fix_labels)

                # 处理 weak with strong
                weak, _ = self.transform(combined_images)
                weak = weak.to(self.device)
                logits_u_wws, _ = self.model(weak)
                loss_weak_with_strong = criterion_ce(logits_u_wws, fix_labels)

                # 处理 weak with self
                loss_weak_with_self = criterion_ce(logits_u_wws, mix_labels)

                # 计算总损失
                loss_total = loss + lam * loss_weak_with_strong + (1 - lam) * loss_weak_with_self

                # 对所有样本的加权损失取均值
                loss_total = loss_total.mean()  # 对整个批次的加权损失求均值

                self.model.train()
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # 梯度裁剪
                optimizer.step()
                print(f"|---- After batch {batch_idx} -------")

                # 打印参数更新后的模型参数
                # print("Parameters after optimizer step:")
                # for name, param in self.model.named_parameters():
                #     print(name, param.data)

                self.model.eval()
                autual_acc = self.inference(self.model)
                print(f"after_train_client： {autual_acc}")
                # self.model.train()

            print(f"|---- After epoch {iter} -------")

                # 打印参数更新后的模型参数
                # print("Parameters after optimizer step:")
                # for name, param in self.model.named_parameters():
                #     print(name, param.data)
            # keypoint
            self.model.eval()
            autual_acc, loss, correct, total = self.inference(self.model)
            print(f"after_train_client： { autual_acc,loss,correct,total}")

            scheduler.step(int(local_curr_ep))
        self.model.eval()
        self.model.to(self.device)
        self.critical_parameter, self.global_mask, self.local_mask, self.update_vectors = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        self.optimizer = optimizer
        return self.model.state_dict(), sum(epoch_loss), self.critical_parameter, self.global_mask, self.local_mask, self.update_vectors, correct, total
        # return self.model.state_dict(), sum(epoch_loss), correct, total





    def update_semi_weights_align(self, lr, model, global_round, fix_dataset, mix_dataset, additional_feature_banks=None):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        ema_model = model.to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        dataset_size = len(fix_dataset)  # 获取数据集的大小
        batch_size = min(dataset_size // 7, 256)  # 计算批量大小


        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = lr*batch_size/256
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

        criterion_ce = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        criterion_mse = torch.nn.MSELoss().to(self.device)
        initial_model = copy.deepcopy(self.model)
        fix_loader = DataLoader(
            fix_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        mix_loader = DataLoader(
            mix_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter
            self.model.eval()
            autual_acc = self.inference(self.model)
            print(f"before_train： {autual_acc}")
            batch_loss = []
            for batch_idx, (fix_batch, mix_batch) in enumerate(zip(fix_loader, mix_loader)):
                fix_images, fix_labels = fix_batch['data'], fix_batch['target']
                mix_images, mix_labels = mix_batch['data'], mix_batch['target']

                # 将数据移动到设备
                fix_images, fix_labels = fix_images.to(self.device), fix_labels.to(self.device)
                mix_images, mix_labels = mix_images.to(self.device), mix_labels.to(self.device)

                # 采样 mix-up 的参数 lam, 随机
                lam = self.numbda.sample().item()
                # print("lam:", lam)


                weak_fix, strong = self.transform(fix_images)
                strong = strong.to(self.device)
                weak_fix = weak_fix.to(self.device)
                logits_u_s, _ = self.model(strong)
                _, features_fix = self.model(weak_fix)
                loss = criterion_ce(logits_u_s, fix_labels)

                # 处理 weak with strong
                weak_mix, _ = self.transform(mix_images)
                weak_mix = weak_mix.to(self.device)
                _, features_mix = self.model(weak_mix)

                # 计算 combined_images
                combined_images = lam * weak_fix + (1 - lam) * weak_mix
                logits_u, features = self.model(combined_images)

                # 计算每张图片的特征差异
                a = features - features_fix
                b = features - features_mix

                # 计算每张图片的范数
                norm_a = torch.norm(a, p=2, dim=1)
                norm_b = torch.norm(b, p=2, dim=1)

                # 对每张图片的范数进行处理
                exp_norm_a = torch.exp(-norm_a)
                exp_norm_b = torch.exp(-norm_b)

                # 计算每张图片的 lam_p
                denominator = lam * exp_norm_a + (1 - lam) * exp_norm_b
                lam_p = lam * exp_norm_a / denominator
                # print("lam_p:", lam_p.cpu().detach().numpy())

                # 计算每张图片的损失
                loss_weak_with_strong = criterion_ce(logits_u, fix_labels)  # 每张图片的损失
                loss_weak_with_self = criterion_ce(logits_u, mix_labels)  # 每张图片的损失
                # print(f"loss_weak_with_strong:{loss_weak_with_strong.mean()}")
                # print(f"loss_weak_with_self:{loss_weak_with_self.mean()}")

                # 打印每张图片的 lam_p
                # print("Batch index:", batch_idx)
                # print("lam_p:", lam_p.cpu().detach().numpy())  # 将 lam_p 移到 CPU 并转换为 numpy 数组以进行打印
                # breakpoint()
                # 确保 lam_p 的形状与损失的形状匹配
                assert lam_p.shape == loss_weak_with_strong.shape, "Shape mismatch between lam_p and loss_weak_with_strong"
                assert lam_p.shape == loss_weak_with_self.shape, "Shape mismatch between lam_p and loss_weak_with_self"

                # 使用每张图片的 lam_p 加权损失
                weighted_loss_weak_with_strong = lam_p * loss_weak_with_strong
                weighted_loss_weak_with_self = (1 - lam_p) * loss_weak_with_self

                # 计算每张图片的加权损失总和
                # print("loss:", loss.cpu().detach().numpy())
                # print("weighted_loss_weak_with_strong:", weighted_loss_weak_with_strong.cpu().detach().numpy())
                # print("weighted_loss_weak_with_self:", weighted_loss_weak_with_self.cpu().detach().numpy())
                # breakpoint()

                loss_total = loss + weighted_loss_weak_with_strong + weighted_loss_weak_with_self

                # 对所有样本的加权损失取均值
                loss_total = loss_total.mean()  # 对整个批次的加权损失求均值

                self.model.train()
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # 梯度裁剪
                optimizer.step()
                print(f"|---- After batch {batch_idx} -------")

                # 打印参数更新后的模型参数
                # print("Parameters after optimizer step:")
                # for name, param in self.model.named_parameters():
                #     print(name, param.data)

                self.model.eval()
                autual_acc = self.inference(self.model)
                print(f"after_train_client： {autual_acc}")
                # self.model.train()

            print(f"|---- After epoch {iter} -------")

                # 打印参数更新后的模型参数
                # print("Parameters after optimizer step:")
                # for name, param in self.model.named_parameters():
                #     print(name, param.data)
            # keypoint
            self.model.eval()
            autual_acc, loss, correct, total = self.inference(self.model)
            print(f"after_train_client： { autual_acc,loss,correct,total}")

            scheduler.step(int(local_curr_ep))
        self.model.eval()
        self.model.to(self.device)
        self.critical_parameter, self.global_mask, self.local_mask, self.update_vectors = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        self.optimizer = optimizer
        return self.model.state_dict(), sum(epoch_loss), self.critical_parameter, self.global_mask, self.local_mask, self.update_vectors, correct, total
        # return self.model.state_dict(), sum(epoch_loss), correct, total

    def update_FedCAC(self, lr, model):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        ema_model = model.to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        # dataset_size = len(fix_dataset)  # 获取数据集的大小
        # batch_size = min(dataset_size // 7, 256)  # 计算批量大小


        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = lr
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

            # 使用交叉熵损失
        criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)

        # 保留初始模型，以便后续计算关键参数
        initial_model = copy.deepcopy(self.model)

        # 开始训练
        for epoch in range(int(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # 将数据移动到设备
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向传播
                outputs,_ = self.model(images)
                loss = criterion_ce(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # 梯度裁剪
                optimizer.step()  # 添加这行代码
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Layer {name} grad norm: {torch.norm(param.grad)}")

                # 记录损失
            #     batch_loss.append(loss.item())
            #     print(
            #         f"Epoch [{epoch + 1}/{self.args.local_ep}], Batch [{batch_idx + 1}/{len(self.trainloader)}], Loss: {loss.item()}")
            #
            # epoch_loss.append(sum(batch_loss) / len(batch_loss))  # 记录每个epoch的平均损失

            # 每个epoch后评估模型准确率
            self.model.eval()
            autual_acc, loss, correct, total = self.inference(self.model)
            print(f"after_train_client： {autual_acc, loss, correct, total}")
            self.model.train()

        # 计算关键参数
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        # 返回模型权重和训练结果
        self.optimizer = optimizer
        return self.model.state_dict(), self.critical_parameter, self.global_mask, self.local_mask, correct, total


    def update_semi_weights_withoutproto(self, lr, model, global_round, fix_dataset, mix_dataset, additional_feature_banks=None):
        self.model = model  # 设置 self.model 属性
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        ema_model = model.to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        dataset_size = len(fix_dataset)  # 获取数据集的大小
        batch_size = min(dataset_size // 7, 256)  # 计算批量大小
        # 选择优化器
        if self.args.optimizer == "sgd":
            train_lr = lr*batch_size/256
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
        fix_loader = DataLoader(
            fix_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        mix_loader = DataLoader(
            mix_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter

            batch_loss = []
            for batch_idx, (fix_batch, mix_batch) in enumerate(zip(fix_loader, mix_loader)):
                fix_images, fix_labels = fix_batch['data'], fix_batch['target']
                mix_images, mix_labels = mix_batch['data'], mix_batch['target']

                # 将数据移动到设备
                fix_images, fix_labels = fix_images.to(self.device), fix_labels.to(self.device)
                mix_images, mix_labels = mix_images.to(self.device), mix_labels.to(self.device)

                # 采样 mix-up 的参数 lam, 随机
                lam = self.numbda.sample().item()
                print(f"lam: {lam}")  # 输出lam是标量还是张量

                # 先数据增强后mix-up

                # 将 fix_images 和 mix_images 加权混合
                # combined_images = lam * fix_images + (1 - lam) * mix_images

                # self.model.eval()
                # total_samples = fix_images.size(0)

                # 处理 strong augmentation
                weak_fix, strong = self.transform(fix_images)
                strong = strong.to(self.device)
                weak_fix = weak_fix.to(self.device)
                logits_u_s, _ = self.model(strong)
                loss = criterion_ce(logits_u_s, fix_labels)
                print(f"loss = {loss}")


                # 处理 weak with strong
                weak_mix, _ = self.transform(mix_images)
                weak_mix = weak_mix.to(self.device)
                combined_images = lam * weak_fix + (1 - lam) * weak_mix
                logits_u_wws, _ = self.model(combined_images)
                loss_weak_with_strong = criterion_ce(logits_u_wws, fix_labels)


                # 处理 weak with self
                loss_weak_with_self = criterion_ce(logits_u_wws, mix_labels)
                print(f"Batch {batch_idx}:")
                print(f"Loss (Weak with Strong): {loss_weak_with_strong.item()}")
                print(f"Loss (Weak with Self): {loss_weak_with_self.item()}")
                # breakpoint()
                # loss_total = loss
                loss_total = lam * loss_weak_with_strong + (1 - lam) * loss_weak_with_self

                # loss_total = loss + lam * loss_weak_with_strong + (1 - lam) * loss_weak_with_self
                # 先mix-up后数据增强
                # # 将 fix_images 和 mix_images 加权混合
                # combined_images = lam * fix_images + (1 - lam) * mix_images
                #
                #
                # # self.model.eval()
                # # total_samples = fix_images.size(0)
                #
                # # 处理 strong augmentation
                # weak_, strong = self.transform(fix_images)
                # strong = strong.to(self.device)
                # logits_u_s, _ = self.model(strong)
                # loss = criterion_ce(logits_u_s, fix_labels)
                #
                # # for i in range(total_samples):
                # #     loss_strong += criterion_ce(logits_u_s[i], fix_labels[i])
                #     # fix_label_float = fix_labels[i].float()
                #     # loss_strong += criterion_mse(logits_u_s[i], fix_label_float)
                #
                # # 处理 weak with strong
                # weak, _ = self.transform(combined_images)
                # weak = weak.to(self.device)
                # logits_u_wws, _ = self.model(weak)
                # loss_weak_with_strong = criterion_ce(logits_u_wws, fix_labels)
                #     # fix_label_float = fix_labels[i].float()
                #     # loss_weak_with_strong += criterion_mse(logits_u_wws[i],  fix_label_float )
                #
                # # 处理 weak with self
                # loss_weak_with_self = criterion_ce(logits_u_wws, mix_labels)
                #     # fix_label_float = fix_labels[i].float()
                #     # loss_weak_with_self += criterion_mse(logits_u_wws[i], fix_label_float)
                #
                # # 计算总损失
                # # loss_total = loss_strong + lam * loss_weak_with_strong + (1 - lam) * loss_weak_with_self
                # loss_total = loss + lam * loss_weak_with_strong + (1 - lam) * loss_weak_with_self
                #
                self.model.train()

                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # 梯度裁剪
                optimizer.step()

            print(f"|---- After epoch {iter} -------")

                # 打印参数更新后的模型参数
                # print("Parameters after optimizer step:")
                # for name, param in self.model.named_parameters():
                #     print(name, param.data)

            self.model.eval()
            autual_acc = self.inference(self.model)
            print(f"after_train_client： {autual_acc}")

            scheduler.step(int(local_curr_ep))
        self.model.eval()
        self.model.to(self.device)
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        self.optimizer = optimizer
        return self.model.state_dict(), sum(epoch_loss), self.critical_parameter, self.global_mask, self.local_mask

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
        update_vectors = []  # 存储每一层的更新向量

        self.model.to(self.device)
        prevModel.to(self.device)

        # 在每一层选择特定数量的关键参数
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            # 计算参数的变化值
            g = (param.data - prevparam.data)
            update_vectors.append(g.view(-1).clone())  # 保存更新向量

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
                    # 打印调试信息
            # print(
            #         f"Layer {name1}: metric min = {metric.min().item()}, max = {metric.max().item()}, thresh = {thresh}")

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

        return critical_parameter, global_mask, local_mask, update_vectors

    def global_repr_global_classifier(args, global_model, test_epoch=60):
        # global representation, global classifier
        from models import ResNetCifarClassifier
        from update import test_inference, test_confidence
        import torch
        import random
        from torch.utils.data import random_split
        # 设置随机种子
        seed = 1
        torch.manual_seed(seed)
        random.seed(seed)

        device = "cuda"
        # define model
        model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(
            str(os.getpid())
        )  # to avoid collision
        args.model_time = model_time
        model_output_dir = "save/" + model_time
        save_args_json(model_output_dir, args)
        # dataset
        train_dataset, test_dataset = get_classifier_dataset(args)

        print("begin training classifier...")
        if args.ssl_method == "mae":
            global_model_classifer = ViT_Classifier(
                global_model.encoder, num_classes=10
            ).to(device)
            global_model_classifer = global_model_classifer.cuda()
            optimizer = torch.optim.AdamW(
                global_model_classifer.head.parameters(), lr=3e-4, weight_decay=0.05
            )

        else:
            print("begin training classifier...")
            # Initialize a new model
            global_model_classifer = ResNetCifarClassifier(args=args)

            # Check if the global model is wrapped in a DataParallel or DistributedDataParallel wrapper
            if hasattr(global_model, "module"):
                global_model_to_load = global_model.module
            else:
                global_model_to_load = global_model

            # Load the state dict
            global_model_classifer.load_state_dict(global_model_to_load.state_dict(), strict=False)

            # Move to GPU
            global_model_classifer = global_model_classifer.cuda()

            # Freeze parameters of certain layers
            for param in global_model_classifer.f.parameters():
                param.requires_grad = False

            # train only the last layer
            optimizer = torch.optim.Adam(
                global_model_classifer.fc.parameters(), lr=1e-3, weight_decay=1e-6
            )

        # sample
        # dist_sampler = (
        #     DistributedSampler(train_dataset)
        #     if args.distributed_training
        #     else RandomSampler(train_dataset)
        # )
        # 创建测试训练集的 DataLoader
        total_test_size = len(test_dataset)
        test_train_size = total_test_size // 2
        test_val_size = total_test_size - test_train_size
        test_train_dataset, test_val_dataset = random_split(test_dataset, [test_train_size, test_val_size])
        test_train_loader = DataLoader(
            test_train_dataset,
            batch_size=256,
            sampler=RandomSampler(test_train_dataset),
            num_workers=16,
            pin_memory=False,
            drop_last=True,
        )
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
        best_acc = 0

        # train global model on global dataset
        for epoch_idx in tqdm(range(test_epoch)):
            batch_loss = []
            if args.distributed_training:
                dist_sampler.set_epoch(epoch_idx)

            for batch_idx, (images, labels) in enumerate(test_train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = global_model_classifer(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print(
                        "Downstream Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch_idx + 1,
                            batch_idx * len(images),
                            len(test_train_loader.dataset),
                            100.0 * batch_idx / len(test_train_loader),
                            loss.item(),
                        )
                    )
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss) / len(batch_loss)
            test_acc, test_loss = test_inference(args, global_model_classifer, test_val_dataset)
            test_confidence(args, global_model_classifer, test_val_dataset)
            if test_acc > best_acc:
                best_acc = test_acc
                global_model_classifer.save_model(model_output_dir, suffix=f"best_{best_acc}")
            print("\n Downstream Train loss: {} Acc: {}".format(loss_avg, best_acc))
        return best_acc

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
    """返回测试集上的准确率。"""
    device = "cuda" if args.gpu else "cpu"

    model.eval()
    model.to(device)
    total, correct = 0.0, 0.0

    testloader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    with torch.no_grad():
        for batch, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            # 推理
            outputs, _ = model(images)



            # 预测
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # 打印调试信息
            # print(f'Pred Labels: {pred_labels}')
            # print(f'Correct Predictions: {correct}')
            # print(f'Total Samples: {total}')

            # 更新进度条上的当前准确率

    accuracy = correct / total
    # print(f'Final Accuracy: {accuracy}')
    return accuracy



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


def update_server_weight(args, global_model=None, test_epoch=1, train_dataset=None, test_dataset=None, lr=0.01 ):
    # global representation, global classifier
    from models import ResNetCifarClassifier
    import torch
    # 设置随机种子
    device = "cuda"
    # Set random seed for reproducibility
    seed_value = 16
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    # define model
    test_train_loader = DataLoader(
        train_dataset,
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
    lr = lr*1.5
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        global_model.parameters(), lr, momentum=0.9, weight_decay=5e-4
    )
    # 余弦调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=test_epoch, eta_min=0)

    acc = test_inference(args, model=global_model, test_dataset=test_dataset)
    # inference
    print(f"-------------- acc before training--------------------------")
    print(f"accuracy_test_on_testdataset = {acc}")
    # train global model on global dataset
    for epoch_idx in tqdm(range(test_epoch)):
        if args.distributed_training:
            dist_sampler.set_epoch(epoch_idx)

        # global_model.eval()
        for batch_idx, (images, labels) in enumerate(test_train_loader):
            images, labels = images.to(device), labels.to(device)
            transform = TransformFixMatch(mean=args.mean, std=args.std)
            # 进行强弱数据增强
            weak, _ = transform(images)

            weak = weak.to(device)

            outputs, _ = global_model(weak)
            loss = criterion(outputs, labels)

            global_model.train()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(global_model.parameters(), 1)  # 梯度裁剪
            optimizer.step()
            # global_model.eval()
            # 更新学习率
        scheduler.step()
        # test_acc = test_inference(args, global_model, test_dataset)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate after epoch {epoch_idx}: {lr}")
        # print(f"epoch_{epoch_idx}accuracy_test_on_testdataset = {test_acc}")
    # breakpoint()
    return global_model

def get_softmax_outputs(model, train_dataset):
    device = torch.device("cuda")
    model.to(device)  # 将模型移到GPU
    model.eval()  # 设置模型为评估模式
    total_softmax = None
    data_loader = DataLoader(
        train_dataset,
        batch_size=500,  # 增大批处理大小
        shuffle=False,
        num_workers=12,
        pin_memory=True  # 使用pin_memory以提高数据传输效率
    )

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)  # 将输入移到GPU
            outputs, _ = model(inputs)  # 计算模型输出
            softmax_outputs = F.softmax(outputs, dim=1)  # 计算softmax

            # 累加softmax输出
            if total_softmax is None:
                total_softmax = softmax_outputs.sum(dim=0)
            else:
                total_softmax += softmax_outputs.sum(dim=0)

    return total_softmax / len(data_loader.dataset)  # 归一化

def kl_divergence(p, q):
    p = p.cpu().numpy()
    q = q.cpu().numpy()
    return np.sum(kl_div(p, q))

def update_server_classifier(args, global_model=None, test_epoch=60, train_dataset=None, test_dataset=None):
    # Import necessary modules
    from models import ResNetCifarClassifier
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if args.gpu else "cpu"
    global_model_classifier = ResNetCifarClassifier(args=args)

    # Check if the global model is wrapped in a DataParallel or DistributedDataParallel wrapper
    if hasattr(global_model, "module"):
        global_model_to_load = global_model.module
    else:
        global_model_to_load = global_model

    # Get the state dict of the feature extractor and exclude the fully connected layers
    state_dict = global_model_to_load.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}

    # Load the state dict into the classifier model
    global_model_classifier.load_state_dict(new_state_dict, strict=False)

    # Move the classifier model to GPU
    global_model_classifier = global_model_classifier.to(device)

    # Freeze parameters of the feature extractor layers
    for param in global_model_classifier.f.parameters():
        param.requires_grad = False

    # Initialize the optimizer to only train the last layer (fully connected layer)
    optimizer = torch.optim.Adam(global_model_classifier.fc.parameters(), lr=1e-4, weight_decay=1e-6)

    # Initialize the loss criterion based on the SSL method
    criterion = (
        torch.nn.NLLLoss().to(device)
        if args.ssl_method != "mae"
        else torch.nn.CrossEntropyLoss().to(device)
    )

    transform = TransformFixMatch(mean=args.mean, std=args.std)

    # Setup DataLoader for training dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.local_bs,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
    )

    # Initial inference before training
    acc = test_inference(args, model=global_model, test_dataset=test_dataset)
    print(f"-------------- acc before training --------------------------")
    print(f"accuracy_test_on_testdataset = {acc}")
    global_model_classifier.train()  # Set model to training mode

    # Train the global model on the global dataset
    for epoch_idx in tqdm(range(test_epoch), desc="Server Training"):

        for batch_idx, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # weak, _ = transform(images)
            # weak = weak.to(device)

            logits_u_w, _ = global_model_classifier(images)
            loss = criterion(logits_u_w, labels)

            # Debugging: Print batch loss
            print(f"Epoch: {epoch_idx}, Batch: {batch_idx}, Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

            # Evaluation after each epoch
            # test acc
            acc = test_inference(args, model=global_model_classifier, test_dataset=test_dataset)
            print(f"in epoch {epoch_idx}: accuracy_test_on_testdataset = {acc}")

    print(f"-------------- acc after training --------------------------")
    return global_model_classifier





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
