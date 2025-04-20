import os
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.LightGTS_pretrain import LightGTS
from src.learner_2task import Learner, transfer_weights
from src.callback.tracking_2task import *
from src.callback.patch_mask_2task_predict import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from src.data.datamodule import *


import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=list, default=['ETTh2'], help='dataset name')
parser.add_argument('--dset', type=str, default='etth2', help='dataset name')
parser.add_argument('--dset_path', type=str, default='/home/Decoder_version_1/data/pretrain_datasets/monash_csv_downsmp', help='dataset path')
parser.add_argument('--context_points', type=int, default=576, help='sequence length')
parser.add_argument('--target_points', type=int, default=288, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--img', type=int, default=0, help='for multivariate model or univariate model')
# Patch
parser.add_argument('--target_patch_len', type=int, default=48, help='stride between patch')
parser.add_argument('--patch_len', type=int, default=48, help='patch length')
parser.add_argument('--stride', type=int, default=48, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--e_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--d_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.3, help='masking ratio for the input')
parser.add_argument('--mask_mode', type=str, default='freq_multi', help='masking ratio for the input')
parser.add_argument('--mask_nums', type=int, default=4, help='choice from patch point ')
parser.add_argument('--img_size', type=int, default=64, help='img_size')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='test_model_continue', help='for multivariate model or univariate model')
parser.add_argument('--is_half', type=float, default=0, help='half of the train_set')
parser.add_argument('--is_all', type=int, default=1, help='all of the dataset')
parser.add_argument('--one_channel', type=int, default=0, help='choose 1 channel')
parser.add_argument('--channel_num', type=int, default=0, help='cut random n channel')
parser.add_argument('--model_name', type=str, default='_bigmodel_7_300w_800M_2task_96', help='half of the train_set')
# model save
parser.add_argument('--is_checkpoints', type=bool, default=True, help='save the checkpoints or not')
parser.add_argument('--checkpoints_freq', type=int, default=1, help='the frequency of saving the checkpoints or not')
parser.add_argument('--checkpoints_path', type=str, default="checkpoints/", help='the path of saving the checkpoints')

args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/' + "Decoder_version_1" + '/masked_patchtst/' + args.model_type + '/'
args.save_checkpoints_path = args.checkpoints_path  + args.model_type + '/' + args.save_pretrained_model + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
if not os.path.exists(args.save_checkpoints_path): os.makedirs(args.save_checkpoints_path)

# get available GPU devide
set_device()


def get_model(c_in, args):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = LightGTS(c_in=c_in,
                target_dim=args.target_points,
                target_patch_len=args.target_patch_len,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                e_layers=args.e_layers,
                d_layers=args.d_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='pretrain',
                res_attention=False,
                mask_mode = args.mask_mode,
                mask_nums = args.mask_nums,
                img_size = args.img_size,
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio,maks_mode=args.mask_mode, mask_nums=args.mask_nums)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    # get dataloader
    # dls = get_dls(args)
    dls = DataProviders(args)
    # get model     
    model = get_model(1, args)
    model = nn.DataParallel(model, device_ids=[0,1,2,3]) 
    # model= nn.DataParallel(get_model(1, args), device_ids=[0,1,2,3,4,5,6,7]) 
    # get loss
    loss_func = torch.nn.L1Loss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(1, denorm=True)] if args.revin else []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio,maks_mode=args.mask_mode, mask_nums=args.mask_nums),
         SaveModelCB(monitor='train_loss', fname=args.save_pretrained_model,                       
                        path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        is_checkpoints = args.is_checkpoints,
                        checkpoints_freq = args.checkpoints_freq,
                        save_checkpoints_path = args.save_checkpoints_path,
                        #metrics=[mse]
                        )                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    train_mse = learn.recorder['train_mse']
    if args.is_all:
        valid_loss = 0
    else:
        valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'train_mse': train_mse, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)

def only_name(path):
  file_name=[]
  a = os.listdir(path)
  for j in a:
    if os.path.splitext(j)[1] == '.csv':
      name = os.path.splitext(j)[0]
      file_name.append(name)
    
  return file_name

if __name__ == '__main__':

    pretrain_list=only_name(args.dset_path)
    # final_pretrain_list = []
    # for name in pretrain_list:
    #    if 'london' or 'weather' in name:
    #       final_pretrain_list.append(name)
    args.dset_pretrain = pretrain_list
    # suggested_lr = find_lr()
    suggested_lr = 0.0001
    # Pretrain
    pretrain_func(suggested_lr)
    print('pretraining completed')