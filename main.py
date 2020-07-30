import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_seg import Dataset
from torchvision import transforms
from augmentation import autoaugment
from model.iternet.iternet_model import Iternet
from model.r2u_unet.model import R2AttU_Net
from trainer.trainer import Trainer

import argparse


def main(args): 
    #
    mean = [104.00699, 116.66877, 122.67892]
    std = [0.225*255, 0.224*255, 0.229*255]
    
    train_transform = transform.Compose([
        transform.Resize((256, 256)),
        transform.RandomFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])
    val_transform = transform.Compose([
        transform.Resize((256, 256)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])
    Transform = {'train': val_transform, 'val': val_transform}
    
    # set datasets
    csv_dir = {
        'train': args.train_csv,
        'val': args.val_csv
    }
    
    datasets = {
        x: Dataset(csv_dir[x],
                   args.image_dir,
                   args.mask_dir,
                   batch_size=args.batch_size,
                   transform=Transform[x]) for x in ['train', 'val']
    }

    # set dataloaders
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True) for x in ['train', 'val']
    }

    # initialize the model
    if args.arch == 'iternet':
        model = Iternet(n_channels=3, n_classes=1, out_channels=32, iterations=3)
    elif args.arch == 'r2u_unet':
        model = R2AttU_Net(img_ch=3,output_ch=1)

    # set loss function and optimizer
    criteria = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    
    #
    writer = SummaryWriter('tensorboard/' + args.arch)
    # train the model
    trainer = Trainer(model, criteria, optimizer,
                      scheduler, args.gpus, args.seed, writer)
    exp = os.path.join(args.model_dir, args.arch)
    trainer(dataloaders, args.epochs, exp)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--gpus', default='0,1,2,3',
                        type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='iternet',
                        type=str, help='Architecture')
    parser.add_argument('--size', default='256', type=int,
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image_dir', default='data/',
                        type=str, help='Images folder path')
    parser.add_argument('--mask_dir', default='data/',
                        type=str, help='Masks folder path')
    parser.add_argument('--train_csv', default='data/train.csv',
                        type=str, help='list of training set')
    parser.add_argument('--val_csv', default='data/val.csv',
                        type=str, help='list of validation set')
    parser.add_argument('--lr', default='0.0001',
                        type=float, help='learning rate')
    parser.add_argument('--epochs', default='2',
                        type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default='32',
                        type=int, help='Batch Size')
    parser.add_argument('--model_dir', default='exp/',
                        type=str, help='Images folder path')
    parser.add_argument('--seed', default='2020123',
                        type=int, help='Random status')
    args = parser.parse_args()

    main(args)
