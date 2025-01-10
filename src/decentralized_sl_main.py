#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import io

# 确保输出使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import copy
import time
import random
import csv
import numpy as np
from tqdm import tqdm
import torch

from tensorboardX import SummaryWriter
from options import args_parser
from models import *
from utils import *
from datetime import datetime
from update import LocalUpdate, test_inference
from pprint import pprint
import IPython

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import RandomSampler
import socket

if __name__ == "__main__":
    start_time = time.time()

    # define paths
    path_project = os.path.abspath("..")
    args = args_parser()
    exp_details(args)

    if args.distributed_training:
        global_rank, world_size = get_dist_env()
        hostname = socket.gethostname()

        print("initing distributed training")
        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            world_size=world_size,
            init_method=args.dist_url,
        )
        args.world_size = world_size
        args.batch_size *= world_size

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = "cuda" if args.gpu else "cpu"

    # load dataset and user groups
    set_seed(args.seed)
    (
        train_dataset,
        test_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args)

    batch_size = args.batch_size
    model_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "_{}".format(
        str(os.getpid())
    )  # to avoid collision
    model_output_dir = "save/" + model_time
    args.model_time = model_time
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    pprint(args)

    # BUILD MODEL
    # 全局模型初始化
    args.start_time = datetime.now()
    global_model = ResNetCifarClassifier(args=args).to(device)

    if args.distributed_training:
        global_model = DDP(global_model)
    else:
        global_model = torch.nn.DataParallel(global_model)
    global_model.train()

    # Training
    start_epoch = 0
    print_every = 5
    # 在分配给用户之前测一下全局模型在测试集上的精度
    print("Evaluating global model before training...")
    initial_test_acc, initial_test_loss = test_inference(args, global_model, test_dataset)
    print(f"Initial Test Accuracy: {initial_test_acc * 100:.2f}%, Initial Test Loss: {initial_test_loss:.4f}")
    train_loss, train_accuracy, global_model_accuracy = [], [], []
    epoch_accuracy = []
    # 各个客户端模型是通过对 global_model 进行深度复制来创建的
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]

    optimizer = torch.optim.Adam(
        global_model.parameters(), lr=args.lr, weight_decay=1e-6
    )
    total_epochs = int(args.epochs / args.local_ep)
    schedule = [
        int(total_epochs * 0.3),
        int(total_epochs * 0.6),
        int(total_epochs * 0.9),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule, gamma=0.3
    )

    print(
        "number of users per round: {}".format(max(int(args.frac * args.num_users), 1))
    )
    print("total number of rounds: {}".format(total_epochs))

    local_update_clients = [
        LocalUpdate(
            args=args,
            dataset=train_dataset,
            idx=idx,
            idxs=user_groups[idx],
            logger=logger,
            output_dir=model_output_dir,
        )
        for idx in range(args.num_users)
    ]
    best_test_acc = 0.0  # 初始化最佳测试准确率为0
    lr = optimizer.param_groups[0]["lr"]
    for epoch in tqdm(range(start_epoch, total_epochs)):

        # set local epoch
        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} | Model : {model_time}\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = local_update_clients[idx]
            # 评估本地模型在本地数据集上的准确率和损失
            # 确实是返回本地训练集的推理准确率
            local_accuracy, local_loss = local_model.inference(
                model=local_models[idx],
                test_dataset=train_dataset,
                test_user=user_groups[idx]
            )
            print(f"User {idx} - Local Accuracy: {local_accuracy * 100:.4f}%, Local Loss: {local_loss:.4f}")
            w, loss = local_model.update_weights(
                model=local_models[idx],
                global_round=epoch,
                lr=lr,
            )
            local_models[idx] = local_model.model
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        if args.average_without_bn:
            for i in range(args.num_users):
                local_models[i] = load_weights_without_batchnorm(
                    local_models[i], global_weights
                )
        else:
            for i in range(args.num_users):
                local_models[i] = load_weights(local_models[i], global_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        logger.add_scalar("train loss", loss_avg, epoch)
        #     # 每'print_every'轮打印一次全局训练损失
        # print global training loss after every 'i' rounds
        if (int(epoch * args.local_ep) + 1) % print_every == 0:
            print(f"Training Local Client Loss : {np.mean(np.array(train_loss))}")

        scheduler.step()
        lr = scheduler._last_lr[0]
        global_model.module.save_model(model_output_dir)
        # 在每一轮聚合生成新的模型后测试一次准确率
        test_acc, _ = test_inference(args, global_model.module, test_dataset)
        print(f" \n Results after {epoch + 1} global rounds of training:")
        print("|---- Test Accuracy: {:.4f}%".format(100 * test_acc))
        # 仅当当前测试准确率高于历史最佳测试准确率时才保存模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            global_model.module.save_model(model_output_dir)
        print(f"|---- Saved model with better test accuracy: {100 * test_acc:.4f}% at epoch {epoch + 1}")

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
    suffix = "{}_{}_{}_{}_dec_sl".format(
        args.model, args.batch_size, args.epochs, args.save_name_suffix
    )
    write_log_and_plot(model_time, model_output_dir, args, suffix, test_acc)
