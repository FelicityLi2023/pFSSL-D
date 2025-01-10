import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference, test_confidence

from options import args_parser
from models import *
from dataaugment import *
from utils import *
import numpy as np
import random
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP
from pprint import pprint
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, RandomSampler, random_split
from datetime import datetime
# 改成训练后的数据进行微调五轮？
if __name__ == "__main__":

    # define paths
    # 解析命令行参数
    args = args_parser()
    exp_details(args)

    # 设置随机种子
    seed_value = 1
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # load dataset and user groups
    (
        train_dataset,
        test_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args, seed=seed_value)  # Assuming get_dataset can take a seed parameter
    # 将测试集按类别划分
    class_indices = {}
    for idx, label in enumerate(test_dataset.targets):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    # 将每个类别的样本分配到训练和验证集five
    train_indices = []
    val_indices = []
    for label, indices in class_indices.items():
        train_idx, val_idx = train_test_split(indices, test_size=0.5, random_state=args.seed)
        # train_idx, val_idx = train_test_split(indices, test_size=0.95, random_state=args.seed)
        # train_idx, val_idx = train_test_split(indices, test_size=0.75, random_state=args.seed)
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
    start_epoch = 0
    #central strategy
    #
    # 创建训练和验证集的子集数据集
    # test_train_dataset = Subset(test_dataset, train_indices)
    # test_val_dataset = Subset(test_dataset, val_indices)
    #
    # # 创建数据加载器对象
    # batch_size = args.batch_size
    # device = "cuda" if args.gpu else "cpu"
    #
    # # Initialize global models
    # num_clusters = args.num_clusters
    # global_models = [ResNetCifarClassifier(args=args) for _ in range(num_clusters)]
    # for global_model in global_models:
    #     global_model.to(device)
    #     global_model.train()
    #
    # print("begin training classifier...")
    #
    # # Training loop
    # test_acc_small = np.max(
    #     [
    #         update_server_weight_save(args, global_model=global_model, test_epoch=60,
    #                                   test_train_dataset=test_train_dataset, test_val_dataset=train_dataset)
    #         for global_model in global_models
    #     ]
    # )
    #
    # print(f" \n Results after {args.epochs} global rounds of training:")
    # print("|---- Test Accuracy on all_update: {:.2f}%".format(100 * test_acc_small))
    #
    # 创建训练和验证集的子集数据集
    test_train_dataset = Subset(test_dataset, train_indices)
    test_val_dataset = Subset(test_dataset, val_indices)

    # 创建数据加载器对象
    batch_size = args.batch_size
    device = "cuda" if args.gpu else "cpu"

    num_clusters = args.num_clusters
    global_models = [ResNetCifarClassifier(args=args) for _ in range(num_clusters)]
    for global_model in global_models:
        global_model.to(device)
        global_model.train()
    # 替换为包含模型文件的目录路径
    model_directory = "/nfs/home/wt_liyuting/tempt/save/patho_cluster1"
    # model_directory = "/nfs/home/wt_liyuting/tempt/save/patho_dec_ssl"
    # model_directory = "/nfs/home/wt_liyuting/tempt/save/04_11_2024_19:49:01_64998"
    # model_directory = "/nfs/home/wt_liyuting/tempt/save/30_10_2024_03:27:42_11418"
    # 获取该目录下所有以 'model.pth' 结尾的文件
    model_files = [os.path.join(model_directory, f) for f in os.listdir(model_directory) if f.endswith('model.pth')]

    # 加载每个模型的权重
    for i in range(args.num_clusters):
    #     model_path = model_files[i]  # 获取文件路径
    #     print(f"Loading model from: {model_path}")  # 打印正在加载的模型路径
    #     global_models[i].load_state_dict(torch.load(model_path, map_location=device), strict=False)
    # model_directory = "save/53"
        model_files = [os.path.join(model_directory, f) for f in os.listdir(model_directory) if
                       f.startswith('model_cluster_') and f.endswith('.pth')]
    for i in range(args.num_clusters):
        global_models[i].load_state_dict(torch.load(model_files[i], map_location=device), strict=False)
    #

    print("begin training classifier...")


    # training loop
    test_acc_small = np.max(
        [
            update_server_weight_save(args, global_model=global_model, test_epoch=60,
                                      test_train_dataset=test_train_dataset, test_val_dataset=train_dataset)
            for global_model in global_models
        ]
    )
    # evaluate representations
    # test_acc_finetune_500 = np.max(
    #     [
    #         small_global_repr_global_classifier(args, global_model, test_epoch=60,test_train_dataset=test_train_dataset, test_val_dataset=train_dataset)
    #         for global_model in global_models
    #     ]
    # )

    # test_acc_large = np.max(
    #     [
    #         global_repr_global_classifier(args, global_model, args.finetuning_epoch)
    #         for global_model in global_models
    #     ]
    # )

    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|---- Test Accuracy on all_update: {:.2f}%".format(100 * test_acc_small))
    # print("|---- Test Accuracy on finetune: {:.2f}%".format(100 * test_acc_finetune_500))

