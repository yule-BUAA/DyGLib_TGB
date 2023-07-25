# An Empirical Evaluation of Temporal Graph Benchmark


## Overview
The DyGLib_TGB repository is developed upon [Dynamic Graph Library (DyGLib)](https://github.com/yule-BUAA/DyGLib) and [Temporal Graph Benchmark (TGB)](https://github.com/shenyangHuang/TGB). 
We extend DyGLib to TGB, which additionally includes several popular dynamic graph learning methods for more exhaustive comparisons. 
Details of DyGLib_TGB is [available here](https://arxiv.org/abs/2307.12510).
Feedback from the community is highly welcomed for further improvements of this repository.

## Dynamic Graph Learning Models
Eleven dynamic graph learning methods are included in DyGLib_TGB, including 
[JODIE](https://dl.acm.org/doi/10.1145/3292500.3330895), 
[DyRep](https://openreview.net/forum?id=HyePrhR5KX), 
[TGAT](https://openreview.net/forum?id=rJeW1yHYwH), 
[TGN](https://arxiv.org/abs/2006.10637), 
[CAWN](https://openreview.net/forum?id=KYPz4YsCPj), 
[EdgeBank](https://openreview.net/forum?id=1GVpwr2Tfdg), 
[TCL](https://arxiv.org/abs/2105.07944), 
[GraphMixer](https://openreview.net/forum?id=ayPPc0SyLv1), 
[DyGFormer](https://arxiv.org/abs/2303.13047),
[Persistent Forecast](https://arxiv.org/abs/2307.01026), and 
[Moving Average](https://arxiv.org/abs/2307.01026).


## Evaluation Tasks
DyGLib_TGB supports both dynamic link property prediction and dynamic node property prediction tasks on TGB.

## Environments
[PyTorch 2.0.1](https://pytorch.org/), 
[py-tgb](https://pypi.org/project/py-tgb/),
[numpy](https://pypi.org/project/numpy/), and
[tqdm](https://pypi.org/project/tqdm/).


## Executing Scripts
### Scripts for Dynamic Link Property Prediction
Dynamic link property prediction could be performed on five datasets, including `tgbl-wiki`, `tgbl-review`, `tgbl-coin`, `tgbl-comment`, and `tgbl-flight`. 
#### Model Training
* Example of training `DyGFormer` on `tgbl-wiki` dataset:
```{bash}
python train_link_prediction.py --dataset_name tgbl-wiki --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to train `DyGFormer` on `tgbl-wiki` dataset, run
```{bash}
python train_link_prediction.py --dataset_name tgbl-wiki --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0
```
#### Model Evaluation
* Example of evaluating `DyGFormer` on `tgbl-wiki` dataset:
```{bash}
python evaluate_link_prediction.py --dataset_name tgbl-wiki --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to evaluate `DyGFormer` on `tgbl-wiki` dataset, run
```{bash}
python evaluate_link_prediction.py --dataset_name tgbl-wiki --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0
```

### Scripts for Dynamic Node Property Prediction
Dynamic node property prediction could be performed on three datasets, including `tgbn-trade`, `tgbn-genre`, and `tgbn-reddit`.
Note that if you train or evaluate on a dynamic node property prediction dataset for the first time, the data pre-processing will take a while. However, this is a one-time process.
#### Model Training
* Example of training `DyGFormer` on `tgbn-trade` dataset:
```{bash}
python train_node_classification.py --dataset_name tgbn-trade --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to train `DyGFormer` on `tgbn-trade` dataset, run
```{bash}
python train_node_classification.py --dataset_name tgbn-trade --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0
```
#### Model Evaluation
* Example of evaluating `DyGFormer` on `tgbn-trade` dataset:
```{bash}
python evaluate_node_classification.py --dataset_name tgbn-trade --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to evaluate `DyGFormer` on `tgbn-trade` dataset, run
```{bash}
python evaluate_node_classification.py --dataset_name tgbn-trade --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0
```


## Citations
Please consider citing the following related references when using this project.
```{bibtex}
@article{yu2023empirical,
  title={An Empirical Evaluation of Temporal Graph Benchmark},
  author={Yu, Le},
  journal={arXiv preprint arXiv:2307.12510},
  year={2023}
}
```
```{bibtex}
@article{yu2023towards,
  title={Towards Better Dynamic Graph Learning: New Architecture and Unified Library},
  author={Yu, Le and Sun, Leilei and Du, Bowen and Lv, Weifeng},
  journal={arXiv preprint arXiv:2303.13047},
  year={2023}
}
```
```{bibtex}
@article{huang2023temporal,
  title={Temporal Graph Benchmark for Machine Learning on Temporal Graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={arXiv preprint arXiv:2307.01026},
  year={2023}
}
```
