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
# 服务器？，e取5
#1、初始更新with弱增强，5LUN
# 2、改batch大小，服务器的
# 3、改mix-up方式
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
    # 打印用户组
    # print("User groups:")
    # for user_id, indices in enumerate(user_groups):  # 由于 user_groups 是列表，使用 enumerate 获取用户ID和索引列表
    #     print(f"User {user_id}: {indices}")

    # 将所有样本的标签提取出来
    # all_labels = []
    # for _, label in train_dataset:
    #     all_labels.append(label)
    # all_labels = np.array(all_labels)
    #
    # # 打印每个用户的数据对应的标签种类
    # for user_id, indices in enumerate(user_groups):  # user_groups 是列表，使用 enumerate 遍历
    #     user_labels = [all_labels[idx] for idx in indices]  # 根据用户的索引获取对应标签
    #     unique_labels = set(user_labels)  # 获取用户的唯一标签（类别）
    #     print(f"User {user_id}: {unique_labels}")
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
    # weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/66.86/model_best_0.6686_epoch53.pth'  # 替换为你的预训练权重路径
    weight_path = '/nfs/home/wangwenhua/Yuting/MY/save/09_11_2024_01_28_53_70208/model_best_0.52102_epoch17.pth'  # 替换为你的预训练权重路径

    # weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/05_08_2024_19_37_08_4661/model_best_0.7878_epoch42.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    # acc = test_inference(args, model=global_model, test_dataset=test_val_dataset)
    # inference
    # print(f"-------------- acc before training--------------------------")
    # print(f"accuracy_test_on_testdataset = {acc}")
    # update global_model
    global_model.train()

    # 创建用户模型列表，并将全局模型的权重赋值给每个用户模型
    # optimizer = torch.optim.Adam(
    #     global_model.parameters(), lr=args.lr, weight_decay=1e-6
    # )
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=5e-4)
    # args.epochs = 50
    total_epochs = int(args.epochs / args.local_ep)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10, gamma=0.5
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

    # global_model = update_server_weight(args, global_model=global_model, test_epoch=1, train_dataset=test_train_dataset, test_dataset=test_val_dataset)

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
            fix_dataset, mix_dataset, num_samples = local_model.generate_fixmix_datasets(model=local_models[idx])

            # fix_dataset, mix_dataset = local_model.generate_fixmix_datasets(model=local_models[idx],epoch=epoch)
            # local_num = len(fix_dataset)  # 获取每个客户端的样本数量
            local_num = num_samples
            local_nums.append(local_num)  # 添加到 local_nums 列表中            # lr = args.lr
            # print(f"Fix dataset type: {type(fix_dataset)}, Mix dataset type: {type(mix_dataset)}")
            # print(f"Fix dataset length: {len(fix_dataset)}, Mix dataset length: {len(mix_dataset)}")
            # w, loss, critical_parameter, global_mask, local_mask = local_model.update_semi_weights_ali(
            #     lr=args.lr,
            #     model=local_models[idx],
            #     global_round=epoch,
            #     fix_dataset=fix_dataset,
            #     mix_dataset=mix_dataset,
            # )
            w, loss, correct, total = local_model.update_semi_weights_semi(
                lr=lr,
                # lr=args.lr,
                model=local_models[idx],
                global_round=epoch,
                fix_dataset=fix_dataset,
                mix_dataset=mix_dataset,
            )
            correct_total += correct
            total_total += total
            local_models[idx] = local_model
            local_weights.append(copy.deepcopy(w))
        global_weights = average_weights(local_weights, local_nums)
        global_model.load_state_dict(global_weights)
        print(f"--------------after local_FT--------------------------")
        print(f"average_acc： {correct_total/total_total}")

        # calculats = [get_softmax_outputs(model, test_train_dataset) for model in models]

        print(f"--------------update_global_model--------------------------")

        # global_model.train()
        # #
        # breakpoint()
        global_model = update_server_weight(args, global_model=global_model, test_epoch=5,
                                            train_dataset=test_train_dataset, test_dataset=test_val_dataset,lr=lr)
        print(f"--------------test_global_model on clients--------------------------")

        accuracy_ava, loss_ava, correct_ava, total_ava = 0, 0, 0, 0
        num_clients = len(idxs_users)
        for idx in range(num_clients):
            accuracy, loss, correct, total = local_models[idx].inference(model=global_model)
            accuracy_ava += accuracy
            loss_ava += loss
            correct_ava += correct
            total_ava += total
            # local_models[idx] = local_models[idx].model
            print(f"Client {idx} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            local_models[idx] = global_model
    #     # #     breakpoint()
    #     # # # 计算并打印平均准确率
        average_accuracy = correct_ava / total_ava
        average_loss = loss_ava / num_clients
        print(f"Average Accuracy: {average_accuracy:.4f}, Average Loss: {average_loss:.4f}")

        # # 如果需要，打印总的正确预测数和样本数

        # 获取所有客户端的softmax输出累加结果
        # models = [local_models[idx].model for idx in range(len(local_models))]
        # softmax_outputs = [get_softmax_outputs(model, test_train_dataset) for model in models]
        # 调用调度器更新学习率
        scheduler.step()

        # 打印当前学习率
        lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate after epoch {epoch}: {lr}")
