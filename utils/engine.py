import math
import sys
import cv2
import numpy as np
from utils.utils import *
from dataset.eval import Evaluator
from dataset.dataset import get_coco_api_from_dataset
from config import labels_name
from tqdm import tqdm
from utils.logging import logger


def training(model, optimizer, data_loader, device, epoch, wandb, epochs, print_frequency):
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    nb = len(data_loader)
    progressbar = tqdm(enumerate(data_loader), total=nb)
    for iteration, (images, targets) in progressbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        loss_classifier = float(loss_dict_reduced['loss_classifier'])
        loss_box_reg = float(loss_dict_reduced['loss_box_reg'])
        loss_object_ness = float(loss_dict_reduced['loss_objectness'])
        loss_rpn_box_reg = float(loss_dict_reduced['loss_rpn_box_reg'])

        losses = sum(loss for loss in loss_dict.values())

        gpu_memory = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

        c, w, h = images[0].shape
        s = ('%20s' * 3 + '%20.4g' * 5) % ('%g/%g' % (epoch, epochs), gpu_memory, '%g-%g' % (w, h), loss_classifier,
                                           loss_box_reg, loss_object_ness, loss_rpn_box_reg, losses_reduced)
        progressbar.set_description(s)

        if iteration % print_frequency:
            wandb.log({'loss_classifier': loss_classifier,
                       'loss_box_reg': loss_box_reg,
                       'loss_object_ness': loss_object_ness,
                       'loss_rpn_box_reg': loss_rpn_box_reg,
                       'total_losses': losses})

        if not math.isfinite(loss_value):
            logger.warning("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()


@torch.no_grad()
def evaluate(model, data_loader, device, wandb):
    names = labels_name.names
    class_labels = {int(k): v for k, v in names.items()}
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = get_iou_types(model)
    coco_evaluator = Evaluator(coco, iou_types)

    logger_images = []

    nb = len(data_loader)
    progressbar = tqdm(enumerate(data_loader), total=nb)
    s = 'Evaluating ...'
    for iteration, (images, targets) in progressbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize(device)
        outputs = model(images)

        for image, output in zip(images, outputs):
            image = image.permute(1, 2, 0).cpu()
            image = np.array(image) * 255
            w, h, c = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pred = output.copy()
            pred['boxes'], pred['scores'], pred['labels'] = \
                np.array(pred['boxes'].cpu()), np.array(pred['scores'].cpu()), np.array(pred['labels'].cpu())

            box_data = [{"position": {"minX": box[0] / w, "minY": box[1] / h, "maxX": box[2] / w, "maxY": box[3] / h},
                         "class_id": int(cls),
                         "box_caption": "%s %.3f" % (names[str(cls)], conf),
                         "scores": {"score": float(conf)},
                         "domain": "pixel"} for box, conf, cls in
                        zip(pred['boxes'], pred['scores'], pred['labels'])]
            boxes = {"predictions": {"box_data": box_data, "class_labels": class_labels}}
            logger_images.append(wandb.Image(image, boxes=boxes))
        progressbar.set_description(s)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    wandb.log({"validation_images": logger_images})
    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    summarize = coco_evaluator.summarize()
    mAP_95, mAP_5 = summarize[0], summarize[1]
    wandb.log({'IoU=0.50:0.95': mAP_95, 'IoU=0.50': mAP_5})
    torch.set_num_threads(n_threads)
