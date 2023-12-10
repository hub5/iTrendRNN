import os
import numpy as np
import torch.optim as optim
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# from utils import sample,normalize
from utils import normalize
from iTrendRNN.our_model import TrendRNN
import argparse
from utils.schedule_sampling import schedule_sampling

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import Config as con
batch_size=8

# type='temperature'
# vmin,vmax=con.tp_min,con.tp_max
# type='wind_speed'
# # vmin,vmax=con.ws_min,con.ws_max
type='relative_humidity'
vmin,vmax=con.rh_min,con.rh_max

# scheduled sampling
parser = argparse.ArgumentParser(description='PyTorch video prediction model - TrendRNN')
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)
parser.add_argument('--input_length', type=int, default=con.input_length)
parser.add_argument('--total_length', type=int, default=con.input_length+con.output_length)
parser.add_argument('--batch_size', type=int, default=batch_size)
parser.add_argument('--prior_distance', type=int, default=True)
parser.add_argument('--high_order_trend', type=int, default=False)
parser.add_argument('--size', type=int, default=80)
args = parser.parse_args()

def wrapper_train(model):
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    print('begin to train')

    epoch=0
    eta = args.sampling_start_value
    for itr in range(1, 500*800 + 1):
        ##Here, we show the random input. In your practice, you should provide your sample data.
        # ims = sample.sample_train(batch_size=batch_size,type=type)#b*(in_len+out_len)*h*w*1
        ims=torch.rand(batch_size,args.total_length,args.size,args.size,1)#batch_size,length,height,width,channel
        new_ims=normalize.nor(ims,vmin,vmax)
        new_ims=torch.Tensor(new_ims).cuda()
        optimizer.zero_grad()
        eta, real_input_flag = schedule_sampling(args,eta, itr)
        real_input_flag=torch.FloatTensor(real_input_flag).cuda()
        next_frames,loss= model(new_ims,input_flag=real_input_flag)

        print(loss)

        loss.backward()
        optimizer.step()

if __name__=='__main__':
    model = TrendRNN(args).cuda()
    wrapper_train(model)

