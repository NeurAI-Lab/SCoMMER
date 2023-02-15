# SCoMMER
Official Repository for the **(Oral)** AAAI'23 paper [Sparse Coding in a Dual Memory System for Lifelong Learning](https://arxiv.org/pdf/2301.05058.pdf)

We extended the [CLS-ER](https://github.com/NeurAI-Lab/CLS-ER) repo with our method

## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following  

    `python main.py --dataset <dataset> --model <model> --experiment_id <experiment_id> --buffer_size <buffer_size> --load_best_args`

  ## Examples:

    ```
    python main.py --dataset seq-cifar10 --model scommer --buffer_size 200 --experiment_id scommer-c10-200 --load_best_args
    
    python main.py --dataset seq-cifar100 --model scommer --buffer_size 200 --experiment_id scommer-c100-200 --load_best_args
    ```

  ## For GCIL-CIFAR-100 Experiments

    ```
    python main.py --dataset gcil-cifar100 --weight_dist unif --model scommer --buffer_size 200 --experiment_id scommer-gcil-unif-200 --load_best_args
    
    python main.py --dataset gcil-cifar100 --weight_dist longtail --model scommer --buffer_size 200 --experiment_id scommer-gcil-longtail-200 --load_best_args
    ```

## Requirements

- torch==1.7.0

- torchvision==0.9.0 

- quadprog==0.1.7
