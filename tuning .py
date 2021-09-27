"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""
import copy
import logging
from typing import Any, Dict, Tuple, Union
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
import optuna.logging
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import math
import time
import os
import datetime
import optuna

optuna_logger = logging.getLogger("optuna")

def objective(trial):
    args = dotdict()

    args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

    args.data = 'Custom' # data
    args.root_path = 'D:\GitHub库\Informer2020\data' # root path of data file
    args.data_path = 'select_feature_DF.csv' # data file
    args.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = 'Total_Area' # target feature in S or MS task
    args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.checkpoints = './informer_checkpoints' # location of model checkpoints

    args.seq_len = trial.suggest_categorical('seq_len', [12,24,48,96,192,384]) # input sequence length of Informer encoder
    args.label_len = trial.suggest_categorical('label_len', [12,24,48,96,192,384])  # start token length of Informer decoder
    args.pred_len = 24 # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.enc_in = 4 # encoder input size
    args.dec_in = 4 # decoder input size
    args.c_out = 1 # output size
    args.factor = 5 # probsparse attn factor
    args.d_model = 512 # dimension of model
    args.n_heads = trial.suggest_categorical('n_heads', [8,16]) # num of heads
    args.e_layers = trial.suggest_categorical('e_layers', [6,4,3,2]) # num of encoder layers
    args.d_layers = 2 # num of decoder layers
    args.d_ff = 2048 # dimension of fcn in model
    args.dropout = 0.05 # dropout
    args.attn = 'prob' # attention used in encoder, options:[prob, full]
    args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu' # activation
    args.distil = True # whether to use distilling in encoder
    args.output_attention = False # whether to output attention in ecoder
    args.mix = True
    args.padding = 0

    args.batch_size = 32
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False # whether to use automatic mixed precision training

    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 6
    args.patience = 3
    args.des = 'exp'

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Set augments by using data name
    data_parser = {
        'Custom':{'data':'select_feature_DF.csv','T':'Total_Area','M':[4,4,4],'S':[1,1,1],'MS':[4,4,1]},
        'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.detail_freq = args.freq
    args.freq = args.freq[-1:]


    Exp = Exp_Informer

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

        # set experiments
        exp = Exp(args)
    exp.train(setting)
    
    def predict(exp, setting, load=False):
        pred_data, pred_loader = exp._get_data(flag='pred')
            
        if load:
            path = os.path.join(exp.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            exp.model.load_state_dict(torch.load(best_model_path))

        exp.model.eval()
            
        preds = []
            
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-exp.args.pred_len:,:]).float()
            dec_inp = torch.cat([batch_y[:,:exp.args.label_len,:], dec_inp], dim=1).float().to(exp.device)
            # encoder - decoder
            if exp.args.use_amp:
                with torch.cuda.amp.autocast():
                    if exp.args.output_attention:
                        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if exp.args.output_attention:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if exp.args.features=='MS' else 0
            batch_y = batch_y[:,-exp.args.pred_len:,f_dim:].to(exp.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = batch_y.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)
            trues.append(true)

            preds = np.array(preds)
            trues = np.array(trues)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return preds,trues
    # you can also use this prediction function to get result
    prediction,trues = predict(exp, setting, True)

    def MSE(pred, true):
        return np.mean((pred-true)**2)
    
    return MSE(prediction, trues)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial



