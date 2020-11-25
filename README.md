# Object Detection (Faster RCNN) with Best Visualization

## Installation

```bash
pip install -r requirements.txt
```

[wandb](wandb.ai) is a package for visualization, please register and login to avoid errors during training
![alt text](images/visualization.png)

## Usage
Before training, Please edit in config files (./config/config.yaml) and (./config/labels_name.py)\
With ./config/config.yaml:\
Config path for train and evaluate with format :\
```
        train/
               annotations/
                    annotations.json (with coco format)
               images/
        valid/
               annotations/
                    annotations.json (with coco format)
               images/
```              

With (./config/labels_name.py)\
config with format:
```
        names = {'0': 'background',
                 '1' : 'object_1',
                 '2': 'object_2', ... }
```               
                 
For training:
```
    python train.py
```
For evaluate:
```
    python eval.py
```
For Inference:
```
    python inference.py --image_path path/to/image --score 0.5
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
