import cv2
from config import labels_name
from models import faster_rcnn as model_detector
import argparse
import torch
import yaml
from utils.logging import logger


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Training')
    parser.add_argument('--config', default='./config/config.yaml', help='path to training config')
    parser.add_argument('--image_path', default='', help='path to image')
    parser.add_argument('--score', default=0.5, help='threshold for prediction')
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config) as f:
        config_parameters = yaml.load(f, Loader=yaml.FullLoader)
    names = labels_name.names

    logger.info("Loading Parameters...")
    number_classes = int(config_parameters['num_classes'])
    pretrained = config_parameters['pretrained']
    device = config_parameters['device']
    backbone = config_parameters['backbone']
    trainable_backbone_layers = config_parameters['trainable_backbone_layers']
    resume = config_parameters['resume']

    logger.info('Creating Model ...')
    model = model_detector.resnet_backbone(num_classes=number_classes, pretrained=pretrained, name_backbone=backbone,
                                           trainable_backbone_layers=trainable_backbone_layers)

    model.to(device)

    model.eval()

    save = torch.load(resume)
    model.load_state_dict(save['model'])
    src_img = cv2.imread(args.image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
    images = [img_tensor]
    out = model(images)
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    for idx in range(boxes.shape[0]):
        if scores[idx] >= args.score:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(src_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    cv2.imshow('result', src_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
