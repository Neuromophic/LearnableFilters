#!/usr/bin/env python
#SBATCH --job-name=BSLRNN
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from configuration import *
import torch
import pprint
from utils import *
import BaselineModels as B

args = parser.parse_args()

for seed in range(10):

    args.SEED = seed
    args = FormulateArgs(args)

    print(f'Training network on device: {args.DEVICE}.')
    MakeFolder(args)

    train_loader, datainfo = GetDataLoader(args, 'train')
    valid_loader, datainfo = GetDataLoader(args, 'valid')
    test_loader, datainfo = GetDataLoader(args, 'test')
    pprint.pprint(datainfo)

    SetSeed(args.SEED)

    setup = f"baseline_model_RNN_data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}.model"
    print(f'Training setup: {setup}.')

    msglogger = GetMessageLogger(args, setup)
    msglogger.info(f'Training network on device: {args.DEVICE}.')
    msglogger.info(f'Training setup: {setup}.')
    msglogger.info(datainfo)

    if os.path.isfile(f'{args.savepath}/{setup}'):
        print(f'{setup} exists, skip this training.')
        msglogger.info('Training was already finished.')
    else:
        rnn = B.RNN(datainfo['N_feature'], datainfo['N_class'], 2).to(args.DEVICE)
        
        lossfunction = B.LossFN(args).to(args.DEVICE)
        optimizer = torch.optim.Adam(rnn.GetParam(), lr=args.LR)

        if args.PROGRESSIVE:
            rnn, best = train_pnn_progressive(rnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
        else:
            rnn, best = train_pnn(rnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)

        if best:
            if not os.path.exists(f'{args.savepath}/'):
                os.makedirs(f'{args.savepath}/')
            torch.save(rnn, f'{args.savepath}/{setup}')
            msglogger.info('Training if finished.')
        else:
            msglogger.warning('Time out, further training is necessary.')

    CloseLogger(msglogger)