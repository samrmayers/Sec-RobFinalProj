# Security and Robustness of Machine Learning Systems Final Project
Colin Brown, cdb2167 and Samanatha Mayers, srm2204

This codebase is structured so almost everything can be run from the command line; some example commands are included below. Trained weights can be saved in the ./networks folder.

To get started, download the required packages using ```pip install -r requirements.txt``` and download the data by setting the "download" flag in the Base Dataset to True.

## Example commands:
Train a basic network
```
python Trainer.py --train True --main_task BasicClassification --main_path ./networks/basicnet_weights --dataloader Base --epochs 10 --batch_size 16
```

Evaluate a basic network
```
python Trainer.py --train False --main_task BasicClassification --main_path ./networks/basicnet_weights --dataloader Base
```

Train a jigsaw feature extractor net
```
python Trainer.py --train True --main_task Jigsaw --main_path ./networks/jigsaw2_weights --dataloader Jigsaw --epochs 10 --batch_size 100 --lr 0.01
```

Train an aggregation network for jigsaw and resnet feature extractors with pretrained weights
```
python Trainer.py --train True --main_task AggNet --main_path ./networks/jigsaw_resnet_aggnet_weights --feature_nets Jigsaw ./networks/jigsaw2_weights ResNet18 ./networks/resnet18_weights --dataloader Base --batch_size 100 --epochs 10
```

Evaluate an aggregation network
```
python Trainer.py --train False --main_task AggNet --main_path ./networks/jigsaw_resnet_aggnet_weights --feature_nets Jigsaw ./networks/jigsaw2_weights ResNet18 ./networks/resnet18_weights --dataloader Base --batch_size 100 --epochs 10
```

Attack an aggregation network using FGSM
```
python Trainer.py --train False --main_task AggNet --main_path ./networks/jigsaw_resnet_aggnet_weights --feature_nets Jigsaw ./networks/jigsaw2_weights ResNet18 ./networks/resnet18_weights --dataloader Base --batch_size 100 --epochs 10 --attack FGSM
```

## Code References:
https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10

https://github.com/Harry24k/adversarial-attacks-pytorch

https://pytorch.org/hub/pytorch_vision_resnet/

https://github.com/KevinMusgrave/pytorch-metric-learning
