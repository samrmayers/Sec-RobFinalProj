# Sec-RobFinalProj
Work for Security and Robustness Final Project

Make sure data is downloaded before running.

Example commands:
Train a basic network
```
python Trainer.py --train True --main_task BasicClassification --main_path ./networks/basicnet_weights --dataloader Base --epochs 10 --batch_size 16
```

Evaluate a basic network
```
python Trainer.py --train False --main_task BasicClassification --main_path ./networks/basicnet_weights --dataloader Base
```

Train a pixel randomization network
```
python Trainer.py --train True --main_task PixelRandomization --main_path ./networks/pixel_distortion_weights --dataloader PixelRandomization --epochs 10 --batch_size 16
```

Train a basic network with the pixel randomization pretraining task incorporated
```
python Trainer.py --train True --main_task BasicClassification --main_path ./networks/combined_weights --feature_nets PixelRandomization ./networks/pixel_distortion_weights  --dataloader Base --epochs 1 --batch_size 16
```

Evaluate a combined network
```
python Trainer.py --train False --main_task BasicClassification --main_path ./networks/combined_weights --feature_nets PixelRandomization ./networks/pixel_distortion_weights  --dataloader Base
```

## Todos

- [ ] Add attacks as a dataloader
- [ ] Incorporate real architectures to test on (resnet50, wide resnet or vgg16), can use pretrained weights for early layers
- [ ] Implement patch-filling or some reconstruction based pretraining task
- [ ] Implement color prediction
- [ ] Implement Jigsaw puzzle solving
- [ ] Implement contrastive learning
- [ ] Implement at least one other (https://github.com/jason718/awesome-self-supervised-learning)
- [ ] Train on pairs or combinations of pretraining tasks
- [ ] (extra) include adversarial examples as part of contrastive learning

Workflow for each
- implement (dataloader, feature net, loss, choice code), train, and save
- train incorporated classifier using each of the 2-3 main architectures, save weights
- run attacks, record performance vs baseline

## References:
https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10

https://github.com/Harry24k/adversarial-attacks-pytorch

https://arxiv.org/pdf/2103.14222.pdf

(We should include one for color prediction)

cite pytorch resnet implementation
