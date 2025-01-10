#  pFSSL-D: Generalization Meets Personalization in Dual-Phase Federated Semi-Supervised Learning
This is the PyTorch implementation of the [paper](https://cmt3.research.microsoft.com/ICDE2025/Submission/Index) "pFSSL-D: Generalization Meets Personalization in Dual-Phase Federated Semi-Supervised Learning".  
![image](misc/comparison.png)

## Requirments
```
pip install -r requirements.txt
```
## Main Training Command
1. Decentralized Feature Alignment-based Feature Extraction for Pre-training```python src/decentralized_featarc_ssl_main.py --dataset=cifarssl --gpu=0 --iid=0  --dirichlet --dir_beta 0.02```
2. Centralized classifier Pre-training ```python src/decentralized_sl_main.py --dataset=cifarssl --gpu=0 --iid=0  --dirichlet --dir_beta 0.02```
3. Federated semi-supervised learning for personal model ```python src/Align_KL_FL.py --dataset=cifar --gpu=0 --iid=0  --pathological_modify --num_users=10```



## Acknowledgements:
1. [Dec-SSL](https://github.com/liruiw/Dec-SSL)
2. [FL](https://github.com/AshwinRJ/decentralized-Learning-PyTorch)
3. SSL ([1](https://github.com/SsnL/moco_align_uniform), [2](https://github.com/leftthomas/SimCLR), [3](https://github.com/PatrickHua/SimSiam), [4](https://github.com/HobbitLong/PyContrast), [5](https://github.com/IcarusWizard/MAE))


