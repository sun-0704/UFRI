from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.UFRI_dataset import *

from geoseg.models.MF2S import pyramid_mamba 
from tools.utils import Lookahead
from tools.utils import process_model_params
import pdb

# training hparam
max_epoch = 500
ignore_index = len(CLASSES)#10

train_batch_size = 16
val_batch_size = 1
lr = 1e-4
weight_decay = 1e-6
backbone_lr = 6e-5
backbone_weight_decay = 1e-6
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "mf2s" 
weights_path = "model_weights/UFRI/{}".format(weights_name)
test_weights_name = "mf2s" 
log_name = 'mf2s/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = [0]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = pyramid_mamba(embed_dim=64, num_classes=num_classes, decoder_channels=128)


# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index,dim=1),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index),1.0, 1.0)

use_aux_loss = False

# define the dataloader

train_dataset = UFRIDataset(data_root='train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)


val_dataset = UFRIDataset(transform=val_aug)
test_dataset = UFRIDataset(data_root='test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)


