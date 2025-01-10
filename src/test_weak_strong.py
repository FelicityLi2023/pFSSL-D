import torch
from tensorboardX import SummaryWriter
# from update import LocalUpdate, test_inference, test_confidence
from options import args_parser
from models import *
from utils import *
import numpy as np
import random
import csv
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.nn.parallel import DataParallel as DP
from pprint import pprint
# from update import *
from dataaugment import *
if __name__ == "__main__":
    # define paths
    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = "cuda" if args.gpu else "cpu"
    batch_size = args.batch_size

        # Set a fixed seed for reproducibility
    seed_value = 1
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Your existing code to define paths, load arguments, set GPU, etc.

    # Load dataset and user groups with fixed seed
    (
        train_dataset,
        test_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args, seed=seed_value)  # Assuming get_dataset can take a seed parameter

    # Other parts of your code to define DataLoader objects, global model, local models, etc.

    # Ensure DataLoader objects use the same seed
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=lambda _: np.random.seed(seed_value)  # Ensure workers use the same seed
    )
    memory_loader = DataLoader(
        memory_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=lambda _: np.random.seed(seed_value)
    )

    total_test_size = len(test_dataset)
    test_train_size = total_test_size // 2
    test_val_size = total_test_size - test_train_size
    test_train_dataset, test_val_dataset = random_split(test_dataset, [test_train_size, test_val_size])

    test_train_loader = DataLoader(
        test_train_dataset,
        batch_size=256,
        sampler=RandomSampler(test_train_dataset, replacement=False, num_samples=seed_value),
        num_workers=16,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=lambda _: np.random.seed(seed_value)
    )

    global_model = ResNetCifarClassifier(args=args).to(device)

    weight_path = 'save/final/model_best_0.6778.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)

    global_model.eval()
    # protos = update_global_protos(args, global_model, test_train_dataset)
    # agg_protos = agg_func(protos)
    # global_protos = {}
    # agg_protos = agg_func(local_protos)

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
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    for idx in idxs_users:
        local_model = local_update_clients[idx]
        accuracy_normal, accuracy_weak, accuracy_strong, total = local_model.inference(
            model=local_models[idx])

        # 打印结果
        print(f"User index: {idx}")
        print(f"Accuracy without perturbation: {accuracy_normal:.4f}")
        print(f"Weak accuracy with perturbation: {accuracy_weak:.4f}")
        print(f"Strong accuracy with perturbation: {accuracy_strong:.4f}")
        print(f"total: {total:}")


