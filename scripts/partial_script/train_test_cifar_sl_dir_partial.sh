#!/bin/bash

n=${1-"10"}
i=${2-"1"}


time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 128 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 20 --frac 1   --dirichlet --dir_beta 0.1 --log_directory  "partial_scripts"  &


time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 128 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.7  --dirichlet --dir_beta 0.1 --log_directory  "partial_scripts" &
 

time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 128 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.5    --dirichlet --dir_beta 0.1 --log_directory  "partial_scripts"  &


time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 128 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.3 --dirichlet --dir_beta 0.1 --log_directory  "partial_scripts"   &
wait 

time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 128 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.2   --dirichlet --dir_beta 0.1  --log_directory  "partial_scripts" &  
 

time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 128 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.1    --dirichlet --dir_beta 0.1  --log_directory  "partial_scripts" &


time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 128 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.05  --dirichlet --dir_beta 0.1  --log_directory  "partial_scripts"  &
wait