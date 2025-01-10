#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # decentralized arguments (Notation for the arguments followed from paper)
    parser.add_argument(
        "--epochs", type=int, default=51, help="number of rounds of training"
    )
    parser.add_argument("--num_users", type=int, default=5, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=1.0, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=float, default=1, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=250, help="local batch size: B")
    parser.add_argument("--local_bs_fix", type=int, default=100, help="local batch size: B")

    parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="thfe number of local epochs: E"
    )
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    # model arguments
    parser.add_argument("--model", type=str, default="resnet", help="model name")
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="Optimizer weight decay"
    )

    # other arguments
    parser.add_argument("--dataset", type=str, default="mnist", help="name of dataset")
    parser.add_argument(
        "--backbone", type=str, default="resnet18", help="name of backbone"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="number  of classes"
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="To use cuda, set to a specific GPU ID. Default set to use CPU.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="type  of optimizer"
    )
    parser.add_argument(
        "--save_name_suffix", type=str, default="", help="model name suffix"
    )
    parser.add_argument(
        "--iid", type=int, default=1, help="Default set to IID. Set to 0 for non-IID."
    )
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--seed", type=int, default=49, help="random seed")
    parser.add_argument(
        "--feature_dim", default=128, type=int, help="Feature dim for latent vector"
    )
    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="Temperature used in softmax in SIMCLR",
    )
    parser.add_argument(
        "--k",
        default=200,
        type=int,
        help="Top k most similar images used to predict the label",
    )
    parser.add_argument(
        "--ssl_method", type=str, default="simclr", help="simclr, byol, simsam"
    )
    parser.add_argument(
        "--x_noniid", action="store_true", help="non i.i.d x distribution"
    )
    parser.add_argument(
        "--dirichlet", action="store_true", help="dirichlet label distribution"
    )
    parser.add_argument(
        "--test_intermediate", action="store_true", help="dirichlet label distribution"
    )
    parser.add_argument(
        "--dir_beta", type=float, default=0.5, help="Dirichlet Parameters"
    )
    parser.add_argument(
        "--imagenet_based_cluster",
        action="store_true",
        help="x partition based on pretrained features",
    )
    parser.add_argument(
        "--y_partition", action="store_true", help="partition by y label"
    )
    parser.add_argument(
        "--log_file_name", type=str, default="", help="the log file name"
    )
    parser.add_argument(
        "--num_clusters",
        default=2,
        type=int,
        help="number of clusters in iterative decentralized clustering",
    )

    parser.add_argument("--imagenet100", action="store_true", help="use imagenet100 ")

    parser.add_argument(
        "--y_partition_skew", action="store_true", help="use skew partition for y label"
    )
    parser.add_argument(
        "--y_partition_ratio",
        type=float,
        default=1,
        help="the ratio for skew partition",
    )
    parser.add_argument(
        "--x_shift_dirichlet",
        action="store_true",
        help="dirichlet distribution sampling for input non-i.i.dness",
    )

    parser.add_argument(
        "--reg_scale",
        type=float,
        default=1,
        help="global model feature regularization scale",
    )

    parser.add_argument(
        "--load_pretrained_path",
        type=str,
        default="",
        help="load pretrained model and test on another task",
    )
    parser.add_argument(
        "--full_size", action="store_true", help="use fullsize imagenet image"
    )

    parser.add_argument(
        "--local_rank", default=0, type=int, help="rank for distributed training"
    )
    parser.add_argument(
        "--distributed_training", action="store_true", help="use distributed training"
    )
    parser.add_argument(
        "--log_directory", type=str, default="", help="the directory to save results"
    )
    parser.add_argument(
        "--emd",
        type=float,
        default=0,
        help="earth mover distance to measure noniid ness",
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default="env://",
        help="url used to distributed training",
    )
    parser.add_argument(
        "--average_without_bn",
        action="store_true",
        help="do not load the weights for the batchnorm layers",
    )
    parser.add_argument(
        "--model_continue_training",
        type=int,
        default=0,
        help="do not reinitialize optimizer in each round",
    )
    parser.add_argument(
        "--finetuning_epoch",
        type=int,
        default=50,
        help="do not reinitialize optimizer in each round",
    )
    parser.add_argument(
        "--script_name",
        type=str,
        default="",
        help="the commands to launch the scripts",
    )
    parser.add_argument(
        "--x_shift_skew",
        action="store_true",
        help="x skew ratio",
    )
    parser.add_argument(
        "--load_dataset_to_memory",
        action="store_true",
        help="load dataset into system memory",
    )
    parser.add_argument('--mean', type=float, nargs='+', default=[0.4914, 0.4822, 0.4465],
                        help='mean for normalization')
    parser.add_argument('--std', type=float, nargs='+', default=[0.2023, 0.1994, 0.2010], help='std for normalization')
    parser.add_argument('--T', type=float, default=0.9, help='Temperature for softmax')
    parser.add_argument('--tau', type=float, default=0.5, help='ratio of selecting')
    parser.add_argument('--beta', type=float, default=100, help='parameter for threshold')
    parser.add_argument('--alpha', type=float, default=0.1, help='ema')
    parser.add_argument('--a', type=float, default=0.75, help='mix-up parameter')
    parser.add_argument('--pathological', action='store_true', help='description for pathological')
    parser.add_argument('--pathological_modify', action='store_true', help='description for pathological')
    args = parser.parse_args()
    return args
