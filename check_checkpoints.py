import torch
import numpy as np


ckpt_path = 'outputs/magbs/drums-v1.ckpt'

checkpoint = torch.load(ckpt_path, map_location='cpu')
print('last epoch: ', np.log(checkpoint['lr_schedulers'][0]['_last_lr'][0]/0.001)/np.log(0.98)*2)

print(checkpoint['hyper_parameters']['time_layer'])
print(checkpoint['hyper_parameters']['band_layer'])
print(checkpoint['hyper_parameters']['feature_dim'])
print(checkpoint['hyper_parameters']['num_repeat'])
print(checkpoint['hyper_parameters']['n_att_head'])
print(checkpoint['hyper_parameters']['cfg_optim'])


