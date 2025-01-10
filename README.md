#  Does Decentralized Learning with non-IID Unlabeled Data Benefit from Self Supervision?
This is the PyTorch implementation of the [paper](https://arxiv.org/abs/2210.10947) "Does Decentralized Learning with non-IID Unlabeled Data Benefit from Self Supervision?".  
![image](misc/comparison.png)

## Requirments
```
pip install -r requirements.txt
```
python src/decentralized_featarc_ssl_main.py --dataset=cifarssl --gpu=0 --iid=0 --pathological_modify --num_users=10 | tee 8.21.txt
## Main Training Command
0. Centralized SSL experiment ```python src/centralized_ssl_main.py --dataset=cifarssl --gpu=0 --iid=0  --dirichlet --dir_beta 0.02```
1. Decentralized SSL experiment ```python src/decentralized_ssl_main.py --dataset=cifarssl --gpu=0 --iid=0  --dirichlet --dir_beta 0.02```
2. Decentralized SL experiment ```python src/decentralized_sl_main.py --dataset=cifarssl --gpu=0 --iid=0  --dirichlet --dir_beta 0.02```
3. Decentralized SL Representation experiment ```python src/decentralized_sl_repr_main.py --dataset=cifarssl --gpu=0 --iid=0  --dirichlet --dir_beta 0.02```
4. Decentralized Feature Alignment SSL experiment ```python src/decentralized_featarc_ssl_main.py --dataset=cifarssl --gpu=0 --iid=0  --dirichlet --dir_beta 0.02```
 conda activate  D:\Anaconda3\envs\fling\envs\basic
python src/fix_mix_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological --num_users=10 | tee 8.7_2txt
python src/fix_mix_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 | tee 8.21.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological --num_users=10 --num_users=10 | tee 11.9_dir_0.1_80.txt
python src/KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological --num_users=10 | tee 8.12_ema.txt
python src/KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological --num_users=10 | tee 8.12_ema.txt
python src/Align_KL_FL_DA.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 | tee 11.9_dir0.1_DA.txt
python src/Align_KL_FL_DAFT.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 | tee 11.9_dir0.1_DAFT.txt
python src/Align_KL_FL_CMA.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 | tee 11.9_dir0.1_CMA.txt
python src/Semi-FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 --epochs=75 --local_ep=5 --local_bs=10 | tee 11.9_SemiFL_patho_e=5.txt
python src/FedCAC.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 --epochs=375 --local_ep=5 --lr=0.01 | tee 11.6_FedCAC_dir0.1.txt
python src/FedAVG.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 --epochs=375 --local_ep=5 --lr=0.01 | tee 11.7_FedAVG_dir0.3.txt
python src/finetune.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 --num_clusters=1 | tee 11.11_patho_fetarc.txt
python src/Align_KL_FL_CMA.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10
python src/Align_KL_FL_semifl.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 --epochs=76 | tee 11.19_semifl_dir0.3_5%.txt
python src/Align_KL_FL_semifl.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 --epochs=76 | tee 11.19_semifl_dir0.1_5%.txt
python src/Align_KL_FL_semifl.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 --epochs=76 | tee 11.19_semifl_patho_5%.txt
python src/Align_KL_FL_semifl.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 --epochs=76 | tee 11.21_semifl_patho_1%.txt

python src/Align_KL_FL_dir0.1.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 | tee 11.13_pFSSL-D_dir0.1.txt
python src/Align_KL_FL_dir0.3.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 | tee 11.13_pFSSL-D_dir0.3.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 | tee 11.18_patho_save.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 | tee 11.18_patho_5%.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 | tee 11.18_dir0.1_5%.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 | tee 11.18_dir0.3_5%.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 | tee 11.21_patho_1%.txt

python src/finetune.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 --num_clusters=1 | tee 11.9_cen20.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 | tee 11.18_patho_save.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10 | tee 9.10_mix-up_align.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10  | tee 9.11_mix-up_align.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10  | tee 9.25_mix-up_align.txt
python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10  | tee 9.26_finetune_lr_2.txt
python src/Align_KL_FL_dir0.1.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.1 --num_users=10 | tee 11.12_pFSSL-D_dir0.1.txt

 python src/test.py --dataset=cifar --gpu=0 
python src/determine_plan.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 | tee 7.16_myself.txt
python src/fling_based_myselfsemi.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 | tee 7.18_myself.txt
python src/test.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 | tee 7.13_test.txt
python src/scale_sim.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 | tee 7.15_confidence.txt
python src/draw_update.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 
python src/generate_proto.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10
python src/determine_plan.py --dataset=cifar --gpu=0 --iid=0 --dirichlet --dir_beta 0.3 --num_users=10 | tee 7.13_proto_confidence.txt
python src/test.py --dataset=cifar --gpu=0 --iid=0 --dirichlet --dir_beta 0.3 --num_users=10 | tee 7.13_test_confidence.txt

## Example Scripts on CIFAR
0. Run  Dirichlet non-i.i.d. SSL experiment with SimCLR scripts on CIFAR-10: 
```bash scripts/noniid_script/train_test_cifar_ssl_dir.sh```
1. Run  ablation study with Simsiam
```bash scripts/noniid_script/train_test_cifar_ssl_dir_simsiam.sh```roto.py --dataset=cifar --gpu=0 --iid=0  --dirichlet --dir_beta 0.3 --num_users=10 | tee 7.20_myself.txt
python src/fix_mix_FL.py --dataset=
2. Run  distributed Training
```bash scripts/noniid_script/train_test_cifar_sl_dir_distributed.sh```
3. Run  training with ray
```bash scripts/noniid_script/train_test_cifar_ssl_skewpartition_ray.sh```


## ImageNet Experiment  
0.  Generate ImageNet-100 dataset for smaller-scale experiments. 
```python misc/create_imagenet_subset.py [PATH_TO_EXISTING_IMAGENET] [PATH_TO_CREATE_SUBSET]```

1. To launch as a batch job with two V100 gpus on a cluster
```bash scripts/imagenet_script/train_test_dec_ssl_imagenetfull_simclr_mpi.sh``` 

2. To train on ImageNet-100
```bash scripts/imagenet_script/train_test_dec_ssl_imagenet100_simclr.sh```

3. To train on Full ImageNet
```bash scripts/imagenet_script/train_test_dec_ssl_imagenetfull_simclr.sh```


## Transfer Learning: Object Detection / Segmentation
0. Install [Detectron2](https://github.com/facebookresearch/detectron2) and set up data folders following Detectron2's [datasets instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

1. Convert pre-trained models to Detectron2 models:
```
python misc/convert_pretrained_model.py model.pth det_model.pkl
```

2. Go to Detectron2's folder, and run:
```
python tools/train_net.py --config-file /path/to/config/config.yaml MODEL.WEIGHTS /path/to/model/det_model.pkl
```
where `config.yaml` is the config file listed under the [configs](misc/configs) folder.

 
## File Structure
```angular2html
├── ...
├── Dec-SSL
|   |── data 			# training data
|   |── src 			# source code
|   |   |── options 	# parameters and config
|   |   |── sampling 	# different sampling regimes for non-IIDness
|   |   |── update 	    # pipeline for each local client
|   |   |── models 	    # network architecture
|   |   |── *_main 	    # main training and testing scripts
|   |   └── ...
|   |── save 			# logged results
|   |── scripts 		# experiment scripts
|   |── misc 			# related scripts for finetuning 
└── ...
```

## Citation
If you find Dec-SSL useful in your research, please consider citing:
```
@inproceedings{wang2022does,
	author    = {Lirui Wang, Kaiqing Zhang, Yunzhu Li, Yonglong Tian, and Russ Tedrake},
	title     = {Does Self-Supervised Learning Excel at Handling Decentralized and Non-IID Unlabeled Data?},
	booktitle = {arXiv:2210.10947},
	year      = {2022}
}
```
![img.png](img.png)

## Acknowledgements:
1. [FL](https://github.com/AshwinRJ/decentralized-Learning-PyTorch)
2. SSL ([1](https://github.com/SsnL/moco_align_uniform), [2](https://github.com/leftthomas/SimCLR), [3](https://github.com/PatrickHua/SimSiam), [4](https://github.com/HobbitLong/PyContrast), [5](https://github.com/IcarusWizard/MAE))

## License
MIT 
#   p F S S L - D - m a i n  
 