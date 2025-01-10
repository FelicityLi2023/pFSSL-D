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
    print("User groups:")
    # for user_id, indices in user_groups.items():
    #     print(f"User {user_id}: {indices}")

    # 将所有样本的标签提取出来
    all_labels = []
    for _, label in train_dataset:
        all_labels.append(label)
    all_labels = np.array(all_labels)

    # 打印每个用户的数据对应的标签
    for user_id, indices in enumerate(user_groups):
        user_labels = [all_labels[idx] for idx in indices]
        print(f"User {user_id}: {user_labels}")
    breakpoint()
    # 打印每个用户每个类别的样本数量
    # for user_id, class_count in user_class_counts.items():
    #     print(f"User {user_id}:")
    #     for class_label, count in class_count.items():
    #         print(f"  Class {class_label}: {count} samples")

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
    device = "cuda" if args.gpu else "cpu"
    global_model = ResNetCifarClassifier(args=args).to(device)
    # weight_path = 'save/final/model_best_0.6778.pth'  # 替换为你的预训练权重路径

    weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/21_08_LRStep_acc65.2/model_best_0.652_epoch59.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    acc = test_inference(args, model=global_model, test_dataset=test_val_dataset)
    # inference
    print(f"-------------- acc before training--------------------------")
    print(f"accuracy_test_on_testdataset = {acc}")
    # update global_model
    global_model.train()

    # 创建用户模型列表，并将全局模型的权重赋值给每个用户模型
    # optimizer = torch.optim.Adam(
    #     global_model.parameters(), lr=args.lr, weight_decay=1e-6
    # )
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=5e-4)

    total_epochs = int(args.epochs / args.local_ep)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10, gamma=0.5
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

    # global_model = update_server_weight(args, global_model=global_model, test_epoch=1, train_dataset=test_train_dataset, test_dataset=test_val_dataset)
    # breakpoint()
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]
    # breakpoint()
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

    print(f"--------------initial acc before training--------------------------")
    accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
    num_clients = len(idxs_users)
    for idx in idxs_users:
        # localupdate类型
        local_model = local_update_clients[idx]
        accuracy, loss, correct, total = local_model.inference(model=local_models[idx])
        accuracy_ava += accuracy
        loss_ava += loss
        correct_ava += correct
        total_ava += total
        print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    # 计算并打印平均准确率
    average_accuracy = accuracy_ava / num_clients
    average_loss = loss_ava / num_clients
    print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")


    # start training
    for epoch in tqdm(range(start_epoch, total_epochs)):
        total_nums = 0.0
        local_weights, local_losses, local_nums = [], [], []
        # each user generate dataset
        for idx in idxs_users:
            # localupdate类型
            local_model = local_update_clients[idx]
            # generate local fix & mix dataset
            print(f"Client_ID： {idx}")
            fix_dataset, mix_dataset = local_model.generate_fixmix_datasets(model=local_models[idx])
            local_num = len(fix_dataset)  # 获取每个客户端的样本数量
            local_nums.append(local_num)  # 添加到 local_nums 列表中            # lr = args.lr
            # print(f"Fix dataset type: {type(fix_dataset)}, Mix dataset type: {type(mix_dataset)}")
            # print(f"Fix dataset length: {len(fix_dataset)}, Mix dataset length: {len(mix_dataset)}")
            w, loss, critical_parameter, global_mask, local_mask = local_model.update_semi_weights_withoutproto(
                lr=args.lr,
                model=local_models[idx],
                global_round=epoch,
                fix_dataset=fix_dataset,
                mix_dataset=mix_dataset,
            )
            local_models[idx] = local_model
            local_weights.append(copy.deepcopy(w))
        global_weights = average_weights(local_weights, local_nums)
        global_model.load_state_dict(global_weights)
        #     w, loss, critical_parameter, global_mask, local_mask = local_model.update_semi_weights_fling(
        #         model=local_models[idx],
        #         global_round=epoch,
        #     )

            # breakpoint()
        # update global weights

        # update global_model

        # 测试更新后在各个模型上的acc

        # 聚合前acc
        print(f"--------------before aggregation in epoch {epoch} --------------------------")
        accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
        num_clients = len(idxs_users)

        for idx in idxs_users:
            accuracy, loss, correct, total = local_models[idx].inference(model=local_models[idx].model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            # local_models[idx] = local_models[idx].model
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

        # 计算并打印平均准确率
        average_accuracy = accuracy_ava / num_clients
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")
        #
        # breakpoint()
        # global_fintune
        #为什么加上这个就对不上？
        print(f"--------------update_global_model--------------------------")

        global_model.train()
        # #
        global_model = update_server_weight(args, global_model=global_model, test_epoch=5,
                                            train_dataset=test_train_dataset, test_dataset=test_val_dataset)
        print(f"--------------test_global_model on clients--------------------------")

        accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
        num_clients = len(idxs_users)
        for idx in idxs_users:
            accuracy, loss, correct, total = local_models[idx].inference(model=global_model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            # local_models[idx] = local_models[idx].model
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            # local_models[idx] = global_model（加了这一句就各种不对）
        # #     breakpoint()
        # # # 计算并打印平均准确率
        average_accuracy = accuracy_ava / num_clients
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")
        breakpoint()
        # # 如果需要，打印总的正确预测数和样本数

        # # calculating similarity
        overlap_buffer = [[] for _ in range(len(idxs_users))]
        overlap_buffer_dis = [[] for _ in range(len(idxs_users))]
        # 打印初始化后的 overlap_buffer 和 overlap_buffer_dis
        print("Initialized overlap_buffer:", overlap_buffer)
        print("Initialized overlap_buffer_dis:", overlap_buffer_dis)
        # # calculate overlap rate between client i and client j
        for i_index, i in enumerate(idxs_users):
            for j_index, j in enumerate(idxs_users):
                if i == j:
                    overlap_rate = 1
                    overlap_buffer_dis[i_index].append(overlap_rate)
                    continue
                overlap_rate = 1 - torch.sum(
                    torch.abs(local_models[i].critical_parameter.to(device) -
                              local_models[j].critical_parameter.to(device))
                ) / float(torch.sum(local_models[i].critical_parameter.to(device)).cpu() * 2)
                overlap_buffer[i_index].append(overlap_rate)
                overlap_buffer_dis[i_index].append(overlap_rate)
        #
        # # 计算全局阈值
        overlap_buffer_tensor = torch.tensor(overlap_buffer)
        overlap_sum = overlap_buffer_tensor.sum()
        overlap_avg = overlap_sum / (len(idxs_users) * (len(idxs_users) - 1))
        overlap_max = overlap_buffer_tensor.max()
        print(f"overlap_max = {overlap_max}")
        print(f"overlap_avg = {overlap_avg}")
        threshold = overlap_avg + (epoch +30) / args.beta * (overlap_max - overlap_avg)
        print(f"final_threshold = {threshold}")
        # 打印填充后的 overlap_buffer 和 overlap_buffer_dis
        print("Filled overlap_buffer:", overlap_buffer)
        print("Filled overlap_buffer_dis:", overlap_buffer_dis)
        # calculate the customized global model for each client
        # 计算每个客户端的定制化模型,感觉没问题啊
        for  i in range(args.num_users):
            # 这一行代码为客户端 i 创建了一个权重字典的深拷贝，这将作为基础来累加其他客户端的权重。
            w_customized_global = copy.deepcopy(local_models[i].model.state_dict())
            # 这段代码首先将当前客户端 i 添加到合作客户端列表中。
            # 然后，它遍历其他所有客户端 j，检查每一个客户端与客户端 i 的重叠率是否达到了预设的阈值。
            # 如果是，就将这些客户端 j 也加入到合作客户端列表中。
            collaboration_clients = [i]
            # 找出与客户端i重叠率较高的客户端
            index = 0
            for j in range(args.num_users):
                if i == j:
                    continue
                if overlap_buffer[i][index] >= threshold:
                    collaboration_clients.append(j)
                index += 1
            print(f"Client {i}: collaboration clients = {collaboration_clients}")

            for key in w_customized_global.keys():
                for client in collaboration_clients:
                    if client == i:
                        continue
                    w_customized_global[key] += local_models[client].model.state_dict()[key]

                w_customized_global[key] = torch.div(w_customized_global[key], float(len(collaboration_clients)))
            # global_model = local_models[idx].model
            local_models[i].model.load_state_dict(w_customized_global)  # 直接更新模型状态
        # breakpoint()
        #
        # 根据每个用户的mask来更新本地网络
        """
        Overview:
            Perform the critical and non-critical parameter initialization steps in FedCAC.
        """
        print(f"--------------test customed model in epoch {epoch} --------------------------")
        accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
        num_clients = len(idxs_users)
        for idx in idxs_users:

            accuracy, loss, correct, total = local_models[idx].inference(model=local_models[idx].model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            # local_models[idx] = local_models[idx].model
        # 计算并打印平均准确率
        average_accuracy = accuracy_ava / num_clients
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")

        # # # 如果需要，打印总的正确预测数和样本数
        # # print(f"Total Correct Predictions: {correct_ava}, Total Samples: {total_ava}")
        # #
        # breakpoint()
        # # finetune with server data
        # # for idx in idxs_users:
        # #
        # #     global_model.train()
        # #     # server finetuningal_
        # #     global_model = update_server_classifier(args, global_model=locmodels[idx].model, test_epoch=60,
        # #                           train_dataset=test_train_dataset, test_dataset=test_val_dataset)
        # #     # accuracy, loss, correct, total = local_models[idx].inference(model=local_models[idx].model)
        # #     accuracy, loss, correct, total = local_models[idx].inference(model=global_model)
        # #
        # #     accuracy_ava += accuracy
        # #     loss_ava += loss
        # #     correct_ava += correct
        # #     total_ava += total
        # #     print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        # #     # local_models[idx] = local_models[idx].model
        # #     local_models[idx] = global_model
        # #
        # # # 计算并打印平均准确率
        # # average_accuracy = accuracy_ava / num_clients
        # # average_loss = loss_ava / num_clients
        # # print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")
        #
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

        for idx in idxs_users:
            # local_models[idx].model.load_state_dict(global_weights)
            accuracy, loss, correct, total = local_models[idx].inference(model=local_models[idx].model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            local_models[idx] = local_models[idx].model
        # 计算并打印平均准确率
        average_accuracy = accuracy_ava / num_clients
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")
        #
        # # # 如果需要，打印总的正确预测数和样本数
        # # print(f"Total Correct Predictions: {correct_ava}, Total Samples: {total_ava}")
        #
        # # if args.average_without_bn:
        # #     for i in range(args.num_users):
        # #         local_models[i].model = load_weights_without_batchnorm(
        # #             local_models[i].model, global_weights
        # #         )
        # # else:
        # #     for i in range(args.num_users):
        # #         local_models[i].model = load_weights(local_models[i].model, global_weights)
        #
        # # update global weights
        # # scheduler.step()
        # # lr = scheduler._last_lr[0]
        # # print(f"Updated Learning rate: {lr}")
        #
        # # global_model.module.save_model(model_output_dir)
        # # 在每一轮聚合生成新的模型后测试一次准确率
