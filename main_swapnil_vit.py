import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model_swap_vit import Lipreading
# from lipreading.model_swap_vit import vit_model
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
import os

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=100, help='Number of classes')
    parser.add_argument('--modality', default='video', choices=['video', 'raw_audio'], help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/100WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet','mixer','vit'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'], help='what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=1, help='Mini-batch size')
    parser.add_argument('--optimizer',type=str, default='adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default=None, help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')
    parser.add_argument('--resize', default=False, action='store_true',help = 'resize or not for vit')

    args = parser.parse_args()
    return args

args = load_args()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True
from lipreading.dataloaders_swap_3d import get_data_loaders, get_preprocessing_pipelines

def extract(model, dset_loader, key='train'):
    model.eval()
    for batch_idx, (input, lengths, labels, filename) in enumerate(dset_loader):

#         print('labels',labels)
#         print('filename',filename)
        word = filename[0].split('/')[2]
        file = filename[0].split('/')[-1]
        feats = model(input.unsqueeze(1).cuda(), lengths=lengths)
        feats = (feats.cpu().detach().numpy())
        folder = './datasets/vit_features/' + word + '/' + key + '/'
        fname = folder+file.split('.')[0]
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(fname, feats)
        
# def train(model, dset_loader, criterion, epoch, optimizer, logger=None):
#     data_time = AverageMeter()
#     batch_time = AverageMeter()
#     accum_iter = 10
#     lr = showLR(optimizer)

# #     logger.info('-' * 10)
# #     logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
# #     logger.info('Current learning rate: {}'.format(lr))

#     model.train()
#     running_loss = 0.
#     running_corrects = 0.
#     running_all = 0.

#     end = time.time()
#     for batch_idx, (input, lengths, labels, filename) in enumerate(dset_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)

#         # --
# #         input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
# #         labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

#         print('labels',labels)
#         print('filename',filename)
    
#         logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
        
# #         loss_func = mixup_criterion(labels_a, labels_b, lam)
# #         loss = loss_func(criterion, logits) / accum_iter
# #         loss = loss_func(criterion, logits)
        

# #         loss.backward()
# #         if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dset_loader)):
# #         optimizer.step()
# #         optimizer.zero_grad()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#         # -- compute running performance
#         _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
#         running_loss += loss.item()*input.size(0)
#         running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
#         running_all += input.size(0)
#         # -- log intermediate results
#         if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader)-1):
#             update_logger_batch( args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

#     return model

def main():

    # -- get model
    model = Lipreading( modality='video',
                        num_classes=args.num_classes,
                        backbone_type=args.backbone_type,
#                         backbone_type='vit',
                        relu_type=args.relu_type,
                        extract_feats=False).cuda()
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    extract(model, dset_loaders['train'], key = 'train')
    extract(model, dset_loaders['test'], key = 'test')
    extract(model, dset_loaders['val'], key = 'val')
    
if __name__ == '__main__':
    main()
