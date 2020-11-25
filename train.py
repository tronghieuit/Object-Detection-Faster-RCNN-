import torch.utils.data
from dataset.ratio import GroupedBatchSampler, create_aspect_ratio_groups
import yaml
from dataset.dataset import *
from models import faster_rcnn as model_detector
from utils.engine import *
import wandb
from utils.logging import logger
import argparse

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

    logger.info("Loading data ... ")
    train_set = load_dataset(config_parameters['train'], 'train', transforms=get_transform(True))
    valid_set = load_dataset(config_parameters['valid'], 'valid', transforms=get_transform(False))

    if config_parameters['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        valid_sampler = torch.utils.data.SequentialSampler(valid_set)

    logger.info("Loading Parameters...")
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
    epochs = config_parameters['epochs']
    print_frequency = config_parameters['print_frequency']
    aspect_ratio_group_factor = config_parameters['aspect_ratio_group_factor']
    distributed = config_parameters['distributed']
    parallel = config_parameters['parallel']
    resume = config_parameters['resume']
    output_dir = config_parameters['output_dir']
    gpu = config_parameters['gpu']
    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_set, k=aspect_ratio_group_factor)
        train_batch = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    else:
        train_batch = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_batch, num_workers=workers,
                                               collate_fn=collate_fn)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=workers, collate_fn=collate_fn)

    model = model_detector.resnet_backbone(num_classes=number_classes, pretrained=pretrained, name_backbone=backbone,
                                           trainable_backbone_layers=trainable_backbone_layers)

    model.to(device)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_without_ddp = model.module

        # Parallel
    if parallel:
        logger.info('Training parallel')
        model = torch.nn.DataParallel(model).cuda()
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    if resume:
        logger.info("Loading checkpoint ...")
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    for epoch in range(epochs):
        model.train()
        logger.info(('\n' + '%20s' * 8) % ('Epoch', 'gpu_mem', 'img_size', 'loss_classifier', 'loss_box_reg',
                                           'loss_object', 'loss_rpn_box_reg', 'total_losses'))
        training(model, optimizer, train_loader, device, epoch, wandb, epochs, print_frequency)

        lr_scheduler.step()

        # evaluate after every epoch
        evaluate(model, valid_loader, device=device, wandb=wandb)

        if output_dir:
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(output_dir, 'model_{}.pth'.format(epoch)))


if __name__ == "__main__":
    main()
