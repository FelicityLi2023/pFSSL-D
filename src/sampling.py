#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import IPython
import matplotlib.pyplot as plt
import torch
import time
import os
from typing import List
import random


def mkdir_if_missing(dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_partition(dataset, num_users, shard_per_user=1):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_imgs = len(dataset) // num_users // shard_per_user
    num_shards = num_users
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def cifar_partition_skew(dataset, num_users, beta=1, vis=False, labels=None):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    skew_ratio = 1 - beta
    print("partition skew: {} {} ".format(num_users, skew_ratio))
    if labels is None:
        labels = np.array(dataset.targets)
    skewed_data = []
    unskewed_data = []
    data_num_per_user = len(dataset) // num_users
    data_num_per_user_skew = int(data_num_per_user * skew_ratio)
    data_num_per_user_unskew = int(data_num_per_user * (1 - skew_ratio))
    print(data_num_per_user, data_num_per_user_skew, data_num_per_user_unskew)

    K = len(np.unique(labels))
    dict_users = {i: np.array([]) for i in range(num_users)}

    for i in range(K):
        index = np.where(labels == i)[0]
        np.random.shuffle(index)
        split = int(len(index) * skew_ratio)
        skewed_data.append(index[:split])
        unskewed_data.append(index[split:])

    skewed_data = np.concatenate(skewed_data)
    unskewed_data = np.concatenate(unskewed_data)
    np.random.shuffle(unskewed_data)  # uniform
    print(
        "len of skewed: {} len of unskewed: {} data_num_per_user_skew: {}".format(
            len(skewed_data), len(unskewed_data), data_num_per_user_skew
        )
    )

    # divide and assign
    print(data_num_per_user, split, data_num_per_user_skew)
    for i in range(num_users):
        skew_base_idx = i * data_num_per_user_skew
        unskew_base_idx = i * data_num_per_user_unskew
        dict_users[i] = np.concatenate(
            (
                skewed_data[skew_base_idx : skew_base_idx + data_num_per_user_skew],
                unskewed_data[
                    unskew_base_idx : unskew_base_idx + data_num_per_user_unskew
                ],
            ),
            axis=0,
        )

    return dict_users


def cifar_noniid(dataset, num_users, vis=True):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def dirichlet_sampling(seed, labels, num_users, alpha, seed_value=None, vis=False, fig_name="cluster"):
    if seed_value is None:
        np.random.seed(seed)  # Set seed for reproducibility

    K = len(np.unique(labels))
    N = labels.shape[0]
    threshold = 0.5
    min_require_size = N / num_users * (1 - threshold)
    max_require_size = N / num_users * (1 + threshold)
    min_size, max_size = 0, 1e6
    iter_idx = 0

    while (
            min_size < min_require_size or max_size > max_require_size
    ) and iter_idx < 1000:
        idx_batch = [[] for _ in range(num_users)]
        plt.clf()
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))

            # avoid adding over
            proportions = np.array(
                [
                    p * (len(idx_j) < N / num_users)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]

            min_size = min([len(idx_j) for idx_j in idx_batch])
            max_size = max([len(idx_j) for idx_j in idx_batch])

        iter_idx += 1

    # 打乱用户分配的数据索引
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])

    # divide and assign
    dict_users = {i: idx for i, idx in enumerate(idx_batch)}
    return dict_users
def cifar_noniid_x_cluster(
    dataset, num_users, cluster_type="pixels", args=None, vis=False, test=False
):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([]) for i in range(num_users)}
    import scipy.cluster
    import sys
    from models import SimCLR

    data = np.load("save/CIFAR10_tuned_features.npz")
    X = data["features_training"] if not test else data["features_testing"]

    from sklearn import decomposition

    pca = decomposition.PCA(n_components=30, whiten=True)
    # IPython.embed()
    X = pca.fit_transform(X)
    features = np.array(X)

    clusters, dist = scipy.cluster.vq.kmeans(features, num_users)
    center_dists = np.linalg.norm(clusters[:, None] - features[None], axis=-1)
    center_dists_argmin = np.argmin(center_dists, axis=0)
    print("{} clustering distortion: {}".format(cluster_type, dist))

    for i in range(num_users):
        dict_users[i] = np.nonzero(center_dists_argmin == i)[0]
        print("cluster {} size: {}".format(i, len(dict_users[i])))

    labels = dataset.targets
    for i in range(num_users):
        cls_cnt = np.array(labels)[dict_users[i]]
        print("dominant class:", np.bincount(cls_cnt))

    # could do resampling and run dirichlet after this for the degree
    labels = np.zeros_like(labels)
    for i in range(num_users):
        labels[dict_users[i]] = i

    # divide and assign
    dict_users = dirichlet_sampling(
        labels, num_users, args.dir_beta, vis=vis, fig_name=cluster_type
    )
    return dict_users


def cifar_noniid_x(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def cifar_noniid_dirichlet(seed, dataset, num_users, beta=0.4, labels=None, vis=False):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if labels is None:
        labels = np.array(dataset.targets)

    dict_users = dirichlet_sampling(
        seed, labels, num_users, beta, vis=vis, fig_name="y_shift"
    )
    return dict_users


def pathological_sampling(dataset: List[tuple], client_number: int, sample_num: int, seed: int, alpha: int) -> List:
    # 获取数据集的大小
    num_indices = len(dataset)

    # 提取数据集中每个数据点的标签（类别）
    labels = np.array([dataset[i][1] for i in range(num_indices)])  # 修改访问方式

    # 计算数据集中独特标签的数量（类别数量）
    num_classes = len(np.unique(labels))

    # 创建一个二维列表，用于存储每个类别对应的数据索引
    idxs_classes = [[] for _ in range(num_classes)]

    # 如果 sample_num 为0，计算每个客户端应分配的数据数量
    if sample_num == 0:
        sample_num = num_indices // client_number

    # 根据标签将数据索引分配到不同的类别列表中
    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    # 创建一个二维列表，用于存储每个客户端分配到的数据索引
    client_indexes = [[] for _ in range(client_number)]

    # 设置随机数生成器的种子，以确保结果可重复
    random_state = np.random.RandomState(seed)

    # 为每个客户端分配数据
    for i in range(client_number):
        # 确定该客户端要分配的两个类别的起始类别索引
        class_start_idx = (i // 2) % num_classes
        class_idx = [class_start_idx, (class_start_idx + 1) % num_classes]

        for j in class_idx:
            # 计算要从该类别中选择的数据数量
            select_num = int(sample_num / alpha)

            # 从该类别中随机选择数据索引
            selected = random_state.choice(idxs_classes[j], select_num, replace=(select_num > len(idxs_classes[j])))

            # 将选择的数据索引添加到该客户端的索引列表中
            client_indexes[i] += list(selected)

        # 将该客户端的索引列表转换为 numpy 数组
        client_indexes[i] = np.array(client_indexes[i])

    # 返回每个客户端的数据集
    return client_indexes


def modified_sampling(dataset: List[tuple], client_number: int, seed: int) -> List[List[int]]:
    """
    Perform simple and deterministic sampling on the dataset.
    :param dataset: A list of tuples, where each tuple contains (data, label, ...).
    :param client_number: Number of clients.
    :param seed: Random seed for reproducibility.
    :return: List of lists, where each sublist contains the indices of the data for each client.
    """
    # 获取数据集的大小
    num_indices = len(dataset)

    # 提取数据集中每个数据点的标签（类别）
    labels = np.array([dataset[i][1] for i in range(num_indices)])

    # 计算数据集中独特标签的数量（类别数量）
    num_classes = len(np.unique(labels))

    # 创建一个二维列表，用于存储每个类别对应的数据索引
    idxs_classes = [[] for _ in range(num_classes)]
    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    # 确保结果可重复
    random.seed(seed)

    # 创建一个二维列表，用于存储每个客户端分配到的数据索引
    client_indexes = [[] for _ in range(client_number)]

    # 将每一类数据分配给两个客户端
    for class_id in range(num_classes):
        # 获取对应类的数据索引
        class_idxs = idxs_classes[class_id]

        # 计算要分配给每个客户端的样本数量
        split_size = len(class_idxs) // 2

        # 根据类id，确定要分配的客户端对
        client_pair = [class_id * 2 % client_number, (class_id * 2 + 1) % client_number]

        # 分配给两个客户端
        client_indexes[client_pair[0]].extend(class_idxs[:split_size])
        client_indexes[client_pair[1]].extend(class_idxs[split_size:])

    # 打乱每个客户端的数据顺序
    for i in range(client_number):
        random.shuffle(client_indexes[i])

    # 返回每个客户端的数据集索引
    return client_indexes


def cifar_noniid_pathological(seed, dataset, num_users, sample_num=0, alpha=2, labels=None, vis=False):
    """
    Sample non-I.I.D client data from CIFAR10 dataset using pathological sampling.
    :param seed: Random seed for reproducibility.
    :param dataset: CIFAR10 dataset.
    :param num_users: Number of clients.
    :param sample_num: Number of samples per client (default 0 to calculate based on dataset size).
    :param alpha: Control parameter for data distribution.
    :param labels: Labels of the dataset, if None, extract from dataset.
    :param vis: Whether to visualize the data distribution (not used).
    :return: Dictionary of user data indices.
    """
    if labels is None:
        labels = np.array([data[1] for data in dataset])  # 修改访问方式

    dict_users = pathological_sampling(dataset, num_users, sample_num, seed, alpha)
    return dict_users

def cifar_noniid_pathological_modify(seed, dataset, num_users, sample_num=0, alpha=2, labels=None, vis=False):
    """
    Sample non-I.I.D client data from CIFAR10 dataset using pathological sampling.
    :param seed: Random seed for reproducibility.
    :param dataset: CIFAR10 dataset.
    :param num_users: Number of clients.
    :param sample_num: Number of samples per client (default 0 to calculate based on dataset size).
    :param alpha: Control parameter for data distribution.
    :param labels: Labels of the dataset, if None, extract from dataset.
    :param vis: Whether to visualize the data distribution (not used).
    :return: Dictionary of user data indices.
    """
    if labels is None:
        labels = np.array([data[1] for data in dataset])  # 修改访问方式

    dict_users = modified_sampling(dataset, num_users, seed)
    return dict_users