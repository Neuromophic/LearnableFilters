#!/usr/bin/env python
#SBATCH --job-name=SNNBase
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import torch
import os
from pprint import pprint
import math
import sys
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
import training as T
import psnn
import math
torch.pi = math.pi 


seed = int(sys.argv[1])

datasets = os.listdir('../ts_datasets/')
datasets = [dataset for dataset in datasets if dataset.endswith('.tsds')]
datasets.sort()
for ds_idx in range(len(datasets)):
    dataset = datasets[ds_idx]
    package = torch.load(f'../ts_datasets/{dataset}')

    name = package['name']

    N_train = package['N_train']
    N_valid = package['N_valid']
    N_test = package['N_test']

    N_class = package['N_class']

    N_channel = package['N_channel']
    N_length = package['N_length']

    N_feature = N_channel * N_length

    print(f'dataset: {name}, N_train: {N_train}, N_valid: {N_valid}, N_test: {N_test}, N_class: {N_class}, N_feature: {N_feature}, N_channel: {N_channel}, N_length: {N_length}')

    X_train = package['X_train']
    X_valid = package['X_valid']
    X_test = package['X_test']

    y_train = package['Y_train']
    y_valid = package['Y_valid']
    y_test = package['Y_test']

    setup = f'{name}_{seed}.snn'
    
    if os.path.exists(f'./baseline_result/SNN/{setup}'):
        print(f'{setup} exists.')

    else:
        torch.manual_seed(seed)

        topology = [N_channel, 3, N_class]
        model = psnn.SpikingNeuralNetwork(topology)

        loss_fn = psnn.SNNLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    
        best_nn = T.training_snn(model, loss_fn, optimizer, X_train, y_train, X_valid, y_valid, X_test, y_test)
        
        y_pred_train = best_nn(X_train)
        acc_train = (y_pred_train.sum(2).argmax(dim=1) == y_train).float().mean()
        y_pred_valid = best_nn(X_valid)
        acc_valid = (y_pred_valid.sum(2).argmax(dim=1) == y_valid).float().mean()
        y_pred_test = best_nn(X_test)
        acc_test = (y_pred_test.sum(2).argmax(dim=1) == y_test).float().mean()

        package = {'name': name, model: best_nn, 'acc_train': acc_train, 'acc_valid': acc_valid, 'acc_test': acc_test}
        print(package)

        if not os.path.exists('./baseline_result/SNN/'):
            os.makedirs('./baseline_result/SNN/')
        torch.save(package, f'./baseline_result/SNN/{setup}')
