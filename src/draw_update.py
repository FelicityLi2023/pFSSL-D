from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import numpy as np
import random
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from models import ResNetCifarClassifier
from update import update_global_protos, agg_func
from options import args_parser
from utils import *

import torch
from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference, test_confidence
from options import args_parser
from models import *
import numpy as np
import random
import csv
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler, random_split, Dataset, DataLoader, Subset
from torch.nn.parallel import DataParallel as DP
from pprint import pprint
from update import *
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    # 定义颜色和颜色映射
    colors = ['#FF1493', '#0000FF', '#FFFF00', '#FF0000', '#000000', '#008000', '#800080', '#00FFFF', '#FFA500', '#FFC0CB']
    cmap = mcolors.ListedColormap(colors)

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

    # Split the dataset by class
    class_indices = {}
    for idx, label in enumerate(test_dataset.targets):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Allocate each class's samples to train and validation sets
    train_indices = []
    val_indices = []
    for label, indices in class_indices.items():
        train_idx, val_idx = train_test_split(indices, test_size=0.5, random_state=args.seed)
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
    batch_size = args.batch_size
    # Create Subset datasets for training and validation
    test_train_dataset = Subset(test_dataset, train_indices)
    test_val_dataset = Subset(test_dataset, val_indices)

    # Create DataLoader objects
    test_train_loader = DataLoader(
        test_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )

    test_val_loader = DataLoader(
        test_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )
    global_protos = []
    device = "cuda" if args.gpu else "cpu"
    global_model = ResNetCifarClassifier(args=args).to(device)
    # weight_path = 'save/final/model_best_0.6778.pth'  # 替换为你的预训练权重路径

    weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/04_07_2024_19:34:53_14244/model_best_0.6778.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)

    global_model.eval()
    # protos = update_global_protos(args, global_model, test_train_dataset)
    # agg_protos = agg_func(protos)

    # for key, value in agg_protos.items():
    #     print(f'Key: {key}, Value: {value}, Number of values: {len(value)}')
    # for key, value in agg_protos.items():
    #     if not isinstance(value, list) or not all(isinstance(tensor, torch.Tensor) for tensor in value):
    #         print(f"Error: agg_protos[{key}] is not a list of tensors.")

    # 创建用户模型列表，并将全局模型的权重赋值给每个用户模型
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]

    # 检查权重是否一致
    # training loop
    # initialize
    local_update_clients = [
        LocalUpdate(
            args=args,
            dataset=train_dataset,
            idx=idx,
            idxs=user_groups[idx],
        )
        for idx in range(args.num_users)
    ]
    # evaluate performance
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    # local agg
    global_protos = {}

    # Iterate over each idx
    for idx in idxs_users:
        local_model = local_update_clients[idx]

        # Update local prototypes
        local_protos = local_model.update_local_protos(
            model=local_models[idx],
            dataset=train_dataset,
            test_user=user_groups[idx],
        )

        # Aggregate local prototypes using agg_func
        agg_protos = agg_func(local_protos)

        # Convert each prototype list to tensors and store
        agg_protos_tensors = {}
        for key, protos_list in agg_protos.items():
            protos_tensor = torch.stack([proto.clone().detach().to(local_model.device) for proto in protos_list])
            agg_protos_tensors[key] = protos_tensor

        # Store aggregated prototypes as tensors
        global_protos[idx] = agg_protos_tensors

    # Aggregate global prototypes
    final_global_protos = proto_aggregation(global_protos)
    for label, proto_list in final_global_protos.items():
        if isinstance(proto_list, list) and len(proto_list) == 1:
            final_global_protos[label] = proto_list[0]
        # 将原型向量从 CUDA 设备复制到 CPU，并转换为 NumPy 数组
    vectors_agg = np.array([proto.cpu().detach().numpy() for proto in final_global_protos.values()])
    protos_each = update_global_protos(args, global_model, test_train_dataset)

    # 收集所有 protos_each 中的向量
    all_vectors_each = []
    for key in protos_each.keys():
        all_vectors_each.extend([vector.cpu().detach().numpy() for vector in protos_each[key]])

    vectors_each = np.array(all_vectors_each)

    num_points_agg = len(vectors_agg)  # final_global_protos 中的点数量
    num_points_each = len(vectors_each)  # protos_each 中的所有点数量
    print(f"protos_each keys: {list(protos_each.keys())}")

    print(f"Number of points in final_global_protos: {num_points_agg}")
    print(f"Number of points in protos_each (all keys): {num_points_each}")

    # 设置 t-SNE 参数并进行降维
    perplexity_agg = min(5, len(vectors_agg) - 1)  # 设置一个合适的 perplexity 值，比如 5
    tsne_agg = TSNE(n_components=2, perplexity=perplexity_agg, random_state=0)
    low_dim_embeddings_agg = tsne_agg.fit_transform(vectors_agg)

    perplexity_each = min(5, len(vectors_each) - 1)  # 设置一个合适的 perplexity 值，比如 5
    tsne_each = TSNE(n_components=2, perplexity=perplexity_each, random_state=0)
    low_dim_embeddings_each = tsne_each.fit_transform(vectors_each)

    # 准备标签信息用于颜色映射
    labels_agg = list(final_global_protos.keys())
    labels_each = []
    for key in protos_each.keys():
        labels_each.extend([key] * len(protos_each[key]))

    point_size_agg = 500  # 设置点的大小为 500
    point_size_each = 30  # 设置点的大小为 30

    # 绘制 t-SNE 可视化结果
    plt.figure(figsize=(40, 20))
    scatter_agg = plt.scatter(low_dim_embeddings_agg[:, 0], low_dim_embeddings_agg[:, 1], c=labels_agg, cmap=cmap,
                              s=point_size_agg, label="final_global_protos")
    scatter_each = plt.scatter(low_dim_embeddings_each[:, 0], low_dim_embeddings_each[:, 1], c=labels_each, cmap=cmap,
                               s=point_size_each, label="protos_each (all keys)")

    # 创建自定义图例
    legend_labels = sorted(final_global_protos.keys())  # 按键的大小排序
    legend_entries = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap.colors[i], markersize=10, label=f'key={key}')
        for i, key in enumerate(legend_labels)]
    plt.legend(handles=legend_entries, loc='lower right', prop={'size': 20})  # 调整字体大小



    plt.title('t-SNE Visualization with Custom Colors (All Keys)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.savefig('plot.png')
