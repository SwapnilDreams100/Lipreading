import torch
import numpy as np
from lipreading.preprocess_swap_3d import *
from lipreading.dataset_3d import MyDataset, pad_packed_collate

def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        resize_size = 224
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    Resize(resize_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])

        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Resize(resize_size),
                                    Normalize(mean, std) ])

        preprocessing['test'] = preprocessing['val']

    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines( args.modality)

    # create dataset object for each partition
    dsets = {partition: MyDataset(
                modality=args.modality,
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz'
                ) for partition in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=6,
                        worker_init_fn=np.random.seed(1)) for x in ['train', 'val', 'test']}
    return dset_loaders
