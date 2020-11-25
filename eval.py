import utils
# import dataset.transforms as T
import datetime
import os
import time
import torch.utils.data
from dataset.ratio import GroupedBatchSampler, create_aspect_ratio_groups
import argparse
import yaml
from dataset.dataset import *
from models import faster_rcnn as model_detector
from utils.engine import *
import wandb
from utils.logging import logger

# Init visualize for training process
wandb.init(project="faster_rcnn")


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Training')
    parser.add_argument('--config', default='./config/config.yaml', help='path to training config')
    return parser.parse_args()


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.Transform())
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main():
    args = get_args()
    with open(args.config) as f:
        config_parameters = yaml.load(f, Loader=yaml.FullLoader)

    if config_parameters['output_dir']:
        mkdir(config_parameters['output_dir'])

    init_distributed_mode(config_parameters)

    print("Loading data ... ")
    # train_set = load_dataset(config_parameters['train'], 'train', transforms=get_transform(True))
    valid_set = load_dataset(config_parameters['valid'], 'valid', transforms=get_transform(False))

    if config_parameters['distributed']:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set)
    else:
        valid_sampler = torch.utils.data.SequentialSampler(valid_set)

    print("Loading Parameters...")
    batch_size = config_parameters['batch_size']
    workers = config_parameters['workers']
    number_classes = int(config_parameters['num_classes'])
    pretrained = config_parameters['pretrained']
    device = config_parameters['device']
    backbone = config_parameters['backbone']
    trainable_backbone_layers = config_parameters['trainable_backbone_layers']
    lr_steps = config_parameters['lr_steps']
    lr_gamma = config_parameters['lr_gamma']
    learning_rate = config_parameters['learning_rate']
    momentum = config_parameters['momentum']
    weight_decay = float(config_parameters['weight_decay'])
    distributed = config_parameters['distributed']
    resume = config_parameters['resume']
    gpu = config_parameters['gpu']

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=workers, collate_fn=collate_fn)

    model = model_detector.resnet_backbone(num_classes=number_classes, pretrained=pretrained, name_backbone=backbone,
                                           trainable_backbone_layers=trainable_backbone_layers)

    model.to(device)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    if resume:
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    evaluate(model, valid_loader, device=device, wandb=wandb)


if __name__ == "__main__":
    main()
