import torch
torch.cuda.empty_cache()
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
# from fix_mix_updata_myself_ali import *
from fix_mix_updata_myself_second import *
from sklearn.model_selection import train_test_split
from collections import defaultdict

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
    device = "cuda" if args.gpu else "cpu"
    global_model = ResNetCifarClassifier(args=args).to(device)

    global_model.to(device)
    global_model.train()


    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=5e-4)
    # args.epochs = 50
    total_epochs = int(args.epochs / args.local_ep)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)


    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]

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

    lr = optimizer.param_groups[0]["lr"]
    print(f"Learning rate: {lr}")
    #
    print(f"--------------initial acc before training--------------------------")
    accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
    num_clients = len(idxs_users)
    for idx in range(num_clients):
        # localupdate类型
        local_model = local_update_clients[idx]
        accuracy, loss, correct, total = local_model.inference(model=local_models[idx])
        accuracy_ava += accuracy
        loss_ava += loss
        correct_ava += correct
        total_ava += total
        print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    # 计算并打印平均准确率
    average_accuracy = correct_ava / total_ava
    average_loss = loss_ava / num_clients
    print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")


    # start training
    for epoch in tqdm(range(start_epoch, total_epochs)):
        total_nums = 0.0
        correct_total = 0.0
        total_total = 0.0
        local_weights, local_losses, local_nums = [], [], []
        # each user generate dataset
        for idx in range(num_clients):
            # localupdate类型
            local_model = local_update_clients[idx]
            # generate local fix & mix dataset
            print(f"Client_ID： {idx}")

            w, critical_parameter, global_mask, local_mask, correct, total = local_model.update_FedCAC(
                lr=lr,
                model=local_models[idx]
            )
            correct_total += correct
            total_total += total
            local_models[idx] = local_model
            local_weights.append(copy.deepcopy(w))
        global_weights = average_weights_direct(local_weights)
        global_model.load_state_dict(global_weights)
        print(f"--------------after local_Update--------------------------")
        print(f"average_acc： {correct_total / total_total}")
        # calculate similarity

        print(f"--------------test global model in epoch {epoch} --------------------------")
        accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
        num_clients = len(idxs_users)
        for idx in range(num_clients):
            accuracy, loss, correct, total = local_models[idx].inference(model=global_model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            local_models[idx] = global_model
            # local_models[idx] = local_models[idx].model
        # 计算并打印平均准确率
        average_accuracy = correct_ava / total_ava
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")

        scheduler.step()
