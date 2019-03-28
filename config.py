import numpy as np

debug = True
image_source_dir = './dataset/horse2zebra/'
input_channel = 3  # input image channels
output_channel = 3  # output image channels
lr = 0.0005
epoch = 200
crop_from = 286
image_size = 256
batch_size = 1
combined_filepath = 'best_weights.h5'
generator_a2b_filepath = 'generator_a2b.h5'
generator_b2a_filepath = 'generator_b2a.h5'
seed = 9584
imagenet_mean = np.array([0.5, 0.5, 0.5])
imagenet_std = np.array([0.5, 0.5, 0.5])
downsample=16
pretrain_epoch=2
pretrain_step_start=500
pretrain_step_end=520
output_dir='predict1/'
use_instance_norm=False
