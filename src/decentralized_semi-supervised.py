import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from update import  test_inference
from dataaugment import *
from options import args_parser
from models import *
from utils import *
import numpy as np
import random
import csv
from datetime import datetime
from torch.nn.parallel import DataParallel as DP
from pprint import pprint
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, random_split, Subset
from sklearn.model_selection import train_test_split
from  update import  update_global_protos, agg_func, proto_aggregation
from pprint import pprint

# 测试

if __name__ == "__main__":
    start_time = time.time()

    # define paths
    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = "cuda" if args.gpu else "cpu"
    batch_size = args.batch_size
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
    ) = get_dataset(args, seed=seed_value)
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

    suffix = "{}_{}_{}_{}_semi_dec_sl".format(args.model, args.batch_size, args.epochs, args.save_name_suffix)
    model_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "_{}".format(str(os.getpid()))  # to avoid collision
    args.model_time = model_time
    model_output_dir = "save/" + model_time
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    pprint(args)

    # build model
    global_models = []

    global_protos = []
    # warm-up

    global_model = ResNetCifarClassifier(args=args).to(device)
    # unsure
    weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/04_07_2024_19:34:53_14244/model_best_0.6778.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    global_model.train()
    if args.distributed_training:
        global_model = DDP(global_model)
    else:
        global_model = torch.nn.DataParallel(global_model)


    # global_model.eval()
    # protos = update_global_protos(args, global_model, test_train_dataset)
    # agg_protos = agg_func(protos)

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
        )
        for idx in range(args.num_users)
    ]
    global_protos = []
    train_loss, train_accuracy, global_model_accuracy = [], [], []
    epoch_accuracy = []
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]

    for epoch in tqdm(range(total_epochs)):
    # training pipeline
      #generate protos
        global_model.eval()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # initial
            # local agg
        # global_protos = {}
        # # Iterate over each idx
        # for idx in idxs_users:
        #     local_model = local_update_clients[idx]
        #     # Update local prototypes
        #     local_protos = local_model.update_local_protos(
        #         model=local_models[idx],
        #         dataset=train_dataset,
        #         test_user=user_groups[idx],
        #     )
        #     # Aggregate local prototypes using agg_func
        #     agg_protos = agg_func(local_protos)
        #     # Convert each prototype list to tensors and store
        #     agg_protos_tensors = {}
        #     for key, protos_list in agg_protos.items():
        #         protos_tensor = torch.stack([proto.clone().detach().to(local_model.device) for proto in protos_list])
        #         agg_protos_tensors[key] = protos_tensor
        #     # Store aggregated prototypes as tensors
        #     global_protos[idx] = agg_protos_tensors
        # # Aggregate global prototypes
        # final_global_protos = proto_aggregation(global_protos)
        # server-side finetune
        # Train global model on test_train data

        # for epoch in tqdm(range(0, 1)):
        #     if args.optimizer == "sgd":
        #         adjust_learning_rate(optimizer, args.lr, epoch, args)
        #     lr = optimizer.param_groups[0]["lr"]
        #     batch_loss = []
        # localupdate
        for idx in idxs_users:
            # 弱增强大于95的，强增强做CE
            local_weights, local_losses = [], []
            print_every = 1
            epoch_accuracy = []
            local_model = local_update_clients[idx]
            print(f"\n | Global Training Round : {epoch+1} |\n")
            w, loss = local_model.update_semi_weights(
                model=local_models[idx],
                global_round=epoch,
            )
            local_models[idx] = local_model.model
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            scheduler.step()
            lr = scheduler._last_lr[0]
            global_model.module.save_model(model_output_dir)
            if (epoch + 1) % print_every == 0:
                print(f"\nAvg Training Stats after {epoch + 1} global rounds:")
                print(f"Training Local Client Loss: {np.mean(np.array(train_loss))}")


        #   aggregate
