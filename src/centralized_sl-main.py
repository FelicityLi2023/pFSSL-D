import os
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from datetime import datetime
from pprint import pprint
from options import args_parser
from utils import *
from models import *
import socket
from update import LocalUpdate, test_inference

if __name__ == "__main__":
    args = args_parser()
    exp_details(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "_{}".format(
        str(os.getpid())
    )
    args.model_time = model_time

    model_output_dir = os.path.join("save/", model_time)
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    print("output dir:", model_output_dir)

    # load datasets
    train_dataset, test_dataset, _, memory_dataset, _ = get_dataset(args)
    batch_size = args.batch_size

    # 将测试集分为50%用于训练和50%用于测试
    total_test_size = len(test_dataset)
    test_train_size = total_test_size // 2
    test_val_size = total_test_size - test_train_size
    test_train_dataset, test_val_dataset = random_split(test_dataset, [test_train_size, test_val_size])

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
    )
    test_train_loader = DataLoader(
        test_train_dataset,
        sampler=RandomSampler(test_train_dataset),
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
    )
    test_val_loader = DataLoader(
        test_val_dataset,
        sampler=RandomSampler(test_val_dataset),
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        drop_last=False,
    )

    suffix = "{}_{}_{}_{}_baseline_supervised".format(
        args.model, args.batch_size, args.epochs, args.save_name_suffix
    )

    # BUILD MODEL
    global_model = ResNetCifarClassifier(args=args).to(device)
    global_model.train()
    start_epoch = 0
    print_every = 50

    # Training
    if args.optimizer == "sgd":
        args.lr = args.lr * (args.batch_size / 256)
        optimizer = torch.optim.SGD(
            global_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            global_model.parameters(), lr=args.lr, weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90], gamma=0.3
        )

    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []
    global_step = 0
    max_steps = len(train_loader) * args.epochs
    scaler = GradScaler()

    best_acc = 0.0

    for epoch in tqdm(range(0, args.epochs + 1)):
        global_model.train()
        if args.optimizer == "sgd":
            adjust_learning_rate(optimizer, args.lr, epoch, args)
        lr = optimizer.param_groups[0]["lr"]
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(test_train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = global_model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 50 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch + 1,
                        batch_idx * len(images),
                        len(test_train_loader.dataset),
                        100.0 * batch_idx / len(test_train_loader),
                        loss.item(),
                    )
                )
            batch_loss.append(loss.item())
            global_step += 1

        loss_avg = sum(batch_loss) / len(batch_loss)
        print("\nTrain Epoch: {} loss: {} lr: {}".format(epoch, loss_avg, lr))
        epoch_loss.append(loss_avg)

        # Evaluate model on test set
        test_acc, test_loss = test_inference(args, global_model, test_val_dataset)
        if test_acc > best_acc:
            best_acc = test_acc
            global_model.save_model(model_output_dir, suffix=f"best_{best_acc}")
        print("\n Downstream Train loss: {} Acc: {}".format(loss_avg, best_acc))
