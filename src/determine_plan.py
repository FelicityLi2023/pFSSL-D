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
from torch.utils.data import Subset, DataLoader, RandomSampler, random_split
from torch.nn.parallel import DataParallel as DP
from pprint import pprint
from dataaugment import *
from sklearn.model_selection import train_test_split

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
    # 将每个类别的样本分配到训练和验证集
    train_indices = []
    val_indices = []
    for label, indices in class_indices.items():
        train_idx, val_idx = train_test_split(indices, test_size=0.5, random_state=args.seed)
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
    start_epoch = 0

    # 创建训练和验证集的子集数据集
    test_train_dataset = Subset(test_dataset, train_indices)
    test_val_dataset = Subset(test_dataset, val_indices)

    # 创建数据加载器对象
    batch_size = args.batch_size
    device = "cuda" if args.gpu else "cpu"
    global_model = ResNetCifarClassifier(args=args).to(device)
    weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/04_07_2024_19:34:53_14244/model_best_0.6778.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)

    global_model.eval()
    # global_protos = []
    # protos = update_global_protos(args, global_model, test_train_dataset)
    # agg_protos = agg_func(protos)
    # global_protos = {}
    # agg_protos = agg_func(local_protos)

    # 创建用户模型列表，并将全局模型的权重赋值给每个用户模型
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
    # 搞个proto
    global_protos = {}
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


    lr = optimizer.param_groups[0]["lr"]
    for epoch in tqdm(range(start_epoch, total_epochs)):
        for idx in idxs_users:
            local_model = local_update_clients[idx]
        # 确实是返回本地训练集的推理准确率
            w, loss = local_model.update_semi_weights_myself(
                model=local_models[idx],
                global_round=epoch,
                agg_protos=final_global_protos

            )
            # w, loss = local_model.update_semi_weights_STRONG(
            #     model=local_models[idx],
            #     global_round=epoch
            #
            # )
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
