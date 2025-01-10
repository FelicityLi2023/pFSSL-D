
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

    global_model = ResNetCifarClassifier(args=args).to(device)


    weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/04_07_2024_19:34:53_14244/model_best_0.6778.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)

    global_model.eval()
    protos = update_global_protos(args, global_model, test_train_dataset)
    agg_protos = agg_func(protos)

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
    for idx in idxs_users:
        results = local_model.proto_based_inference(
            model=local_models[idx],
            test_dataset=train_dataset,
            test_user=user_groups[idx],
            agg_protos=final_global_protos,
        )
        accuracy, correct, total, pred_labels, actual_labels, similarity_scores, interval_counts, interval_correct_counts, interval_incorrect_counts = results
        print(f'User {idx}, Accuracy: {accuracy:.4f}, Correct: {correct}, Total: {total}')
        for i in range(len(interval_counts)):
            lower_bound = i / len(interval_counts)
            upper_bound = (i + 1) / len(interval_counts)
            print(
                f'Similarity interval {lower_bound:.1f} ~ {upper_bound:.1f}, Total: {interval_counts[i]}, Correct: {interval_correct_counts[i]}, Incorrect: {interval_incorrect_counts[i]}')

        accuracy, loss, correct, total, confidence_percentages_correct, confidence_percentages_incorrect, confidence_sample_percentages, confidence_buckets_correct, confidence_buckets_incorrect = local_model.confidence_qujian(
            model=local_models[idx],
            test_dataset=train_dataset,
            test_user=user_groups[idx]
        )
        # Print evaluation results
        print(f"User {idx} - Accuracy: {accuracy * 100:.4f}%")
        print(f"User {idx} - Loss: {loss:.4f}")
        print(f"User {idx} - Correct: {correct:.4f}")
        print(f"User {idx} - Total: {total:.4f}")
        print("Confidence Intervals:")
        for i in range(20):
            lower_bound = i * 5
            upper_bound = (i + 1) * 5
            print(
                f"   {lower_bound}-{upper_bound}%: Correct - {confidence_percentages_correct[i]:.2f}%, Incorrect - {confidence_percentages_incorrect[i]:.2f}%, Total Sample Percentage - {confidence_sample_percentages[i]:.2f}%"
            )
            print(f"      Correct Samples: {confidence_buckets_correct[i]}")
            print(f"      Incorrect Samples: {confidence_buckets_incorrect[i]}")