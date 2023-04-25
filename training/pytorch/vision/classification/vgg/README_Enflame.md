This directory includes resnet50 pytorch implement.

## code base
### reference training code:
link: https://github.com/pytorch/vision/tree/release/0.8.0/references/classification


commit id: cb17a5551678897aefb99aaa644d068196ebe342


branch: release/0.8.0


Pytorch version: 1.7.0 Torchvision version: 0.8.0
### reference model code:
link: https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/resnet.py


commit id: cb17a5551678897aefb99aaa644d068196ebe342


branch: release/0.8.0


Pytorch version: 1.7.0 Torchvision version: 0.8.0

## Setup environment

Scripts are made available as a Python3.6 module.

Before running models, please add local path to python path by:
```shell
export PYTHONPATH=$PYTHONPATH:/path/to/enflame_resnet
```

The package tops_models is also necessary and need to be installed before
running any cases. To install tops_models, please move to the path/to/common
and follow README.md under "common" catalog.

```shell
cd path/to/common
python setup.py bdist_egg
sudo python setup.py install
```

Some dependencies are also required as listed in requirements.txt.
```
pip install -r requirements.txt
```

### run vgg16 pytorch training on gcu:
python train.py --data-path=imagenet50 --num_classes=50 --model=vgg16 --device=gcu --batch-size=192 --epochs=5 --workers=0 --lr 1e-2

```shell
python train.py --data-path=imagenet50 --num_classes=50 --model=vgg16 --device=gcu --batch-size=192 --epochs=5 --workers=0 --lr 1e-2 --training_step_per_epoch=10 --eval_step_per_epoch=10
```

#### Common parameters
- --data-path: path to dataset
- --num_classes: The number of classes
- --model: the name of network, default'vgg16'
- --device: device to running models, option: cpu, gcu, gpu
- --batch-size: batch size
- --workers: number of data loading workers
- --epochs: number of total epochs to run
- --training_step_per_epoch: Number of training steps for each epoch
- --eval_step_per_epoch: Number of evaluate steps for each epoch