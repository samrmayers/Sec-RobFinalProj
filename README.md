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
python Trainer.py --train False --main_task BasicClassification --path ./networks/basicnet_weights --dataloader Base
```

Train a pixel randomization network
```
python Trainer.py --train True --main_task PixelRandomization --main_path ./networks/pixel_distortion_weights --dataloader PixelRandomization --epochs 10 --batch_size ```

Train a basic network with the pixel randomization pretraining task incorporated
```
python Trainer.py --train True --main_task BasicClassification --main_path ./networks/combined_weights --feature_nets PixelRandomization ./networks/pixel_distortion_weights  --dataloader Base --epochs 1 --batch_size 16
```

Evaluate a combined network
```
python Trainer.py --train False --main_task BasicClassification --main_path ./networks/combined_weights --feature_nets PixelRandomization ./networks/pixel_distortion_weights  --dataloader Base
```

## References:
https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10

https://github.com/Harry24k/adversarial-attacks-pytorch

https://arxiv.org/pdf/2103.14222.pdf

(We should include one for color prediction)