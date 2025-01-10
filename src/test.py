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
from update import *
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

    weight_path = '/nfs/home/wt_liyuting/Dec-SSL-main/save/04_07_2024_19:34:53_14244/model_best_0.6778.pth'  # 替换为你的预训练权重路径
    global_model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)

    global_model.eval()
    global_protos = []
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

    # 确实是返回本地训练集的推理准确率
    for idx in idxs_users:
        local_model = local_update_clients[idx]
        # Evaluate the local model on the local dataset for accuracy, loss, and confidence interval ratios
        accuracy, loss, correct, total, confidence_percentages_correct, confidence_percentages_incorrect, confidence_sample_percentages, confidence_buckets_correct, confidence_buckets_incorrect = local_model.confidence_qujian(
            model=local_models[idx], agg_protos=final_global_protos)

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
            print(f"      Correct Samples: {len(confidence_buckets_correct[i])}")
            print(f"      Incorrect Samples: {len(confidence_buckets_incorrect[i])}")

            # Print detailed info for correct samples
            print("      Correct Samples Info:")
            for info in confidence_buckets_correct[i]:
                print(
                    f"         Confidence: {info['confidence']:.2f}, True Label: {info['true_label']}, Model Pred: {info['model_pred']}, Proto Pred: {info['proto_pred']}, Cosine Sim: {info['cosine_sim']:.4f}")

            # Print detailed info for incorrect samples
            print("      Incorrect Samples Info:")
            for info in confidence_buckets_incorrect[i]:
                print(
                    f"         Confidence: {info['confidence']:.2f}, True Label: {info['true_label']}, Model Pred: {info['model_pred']}, Proto Pred: {info['proto_pred']}, Cosine Sim: {info['cosine_sim']:.4f}")
