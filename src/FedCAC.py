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
        scheduler.step()
        models = [local_models[idx].model for idx in range(len(local_models))]
        critical_parameters = [local_models[idx].critical_parameter for idx in range(len(local_models))]
        num_models = len(models)

        # 初始化相似度矩阵（重叠率矩阵）
        similarity_matrix = np.zeros((num_models, num_models))

        # 计算所有模型对之间的重叠率（相似度）
        for i in range(num_models):
            for j in range(i, num_models):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 自身的相似度为 1
                else:
                    # 计算客户端 i 和 j 之间的重叠率
                    overlap_rate = 1 - torch.sum(
                        torch.abs(critical_parameters[i].to(device) - critical_parameters[j].to(device))
                    ) / float(torch.sum(critical_parameters[i].to(device)).cpu() * 2)
                    similarity_matrix[i, j] = overlap_rate
                    similarity_matrix[j, i] = overlap_rate

        # 打印相似度矩阵
        print("Model Critical Parameter Similarity Matrix:")
        print(similarity_matrix)

        # 计算阈值
        similarity_avg = np.mean(similarity_matrix)
        similarity_max = np.max(similarity_matrix)
        threshold = similarity_avg + (epoch + 1) / 100 * (similarity_max - similarity_avg)

        print("threshold:", threshold)

        # 为每个用户生成新的模型参数
        for i in range(num_models):
            # 为客户端 i 创建一个权重字典的深拷贝
            w_customized_global = copy.deepcopy(local_models[i].model.state_dict())
            # 添加当前客户端 i 到合作客户端列表中
            collaboration_clients = [i]

            # 找出与客户端 i 的相似度高于阈值的客户端
            for j in range(num_models):
                if i == j:
                    continue
                if similarity_matrix[i][j] >= threshold:
                    collaboration_clients.append(j)
            print(f"Client {i}: collaboration clients = {collaboration_clients}")

            # 累加合作客户端的权重
            for key in w_customized_global.keys():
                for client in collaboration_clients:
                    if client == i:
                        continue
                    w_customized_global[key] += local_models[client].model.state_dict()[key]

                # 取平均值
                w_customized_global[key] = torch.div(w_customized_global[key], float(len(collaboration_clients)))

            # 更新客户端 i 的模型权重
            local_models[i].model.load_state_dict(w_customized_global)

        # calculate the customized global model for each client
        # 计算每个客户端的定制化模型,感觉没问题啊

        #
        # 根据每个用户的mask来更新本地网络
        """
        Overview:
            Perform the critical and non-critical parameter initialization steps in FedCAC.
        """
        print(f"--------------test customed model in epoch {epoch} --------------------------")
        accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
        num_clients = len(idxs_users)
        for idx in range(num_models):

            accuracy, loss, correct, total = local_models[idx].inference(model=local_models[idx].model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            # local_models[idx] = local_models[idx].model
        # 计算并打印平均准确率
        average_accuracy = correct_ava / total_ava
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")


        #
        # 使用三个嵌套的迭代器同时遍历客户端的当前模型参数、全局模型参数和客户端的定制化模型参数。
        global_model.to(device)
        # param1.data（客户端模型的参数）更新为：
        # local_mask[index] 对应的 param3.data（定制化模型的参数）和 global_mask[index] 对应的 param2.data（全局模型的参数）的加权和。
        for client in range(len(idxs_users)):
            index = 0
            local_models[client].model.to(device)
            for (name1, param1), (name2, param2), (name3, param3) in zip(
                    local_models[client].model.named_parameters(), global_model.named_parameters(),
                    local_models[client].model.named_parameters()):
                param1.data = local_models[client].local_mask[index].to(
                    device).float() * param3.data + \
                              local_models[client].global_mask[index].to(
                                  device).float() * param2.data
                index += 1
            local_models[client].model.to('cpu')
        global_model.to('cpu')
        print("success update")

        print(f"--------------test PG-aggregation in epoch {epoch} --------------------------")
        accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
        num_clients = len(idxs_users)

        for idx in range(num_clients):
            # local_models[idx].model.load_state_dict(global_weights)
            accuracy, loss, correct, total = local_models[idx].inference(model=local_models[idx].model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            local_models[idx] = local_models[idx].model
        # 计算并打印平均准确率
        average_accuracy = correct_ava / total_ava
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")

