# Models

This directory contains the available classification models. All of these models were copied from the [TensorFlow Models repo](https://github.com/tensorflow/models/tree/master/slim/nets) and updated to TensorFlow r1.0. 

The table below lists relevant information for each model. To use one of these models (e.g. when using the training scripts), simply set the `--model_name` flag to the appropriate name. The number of parameters and the number of flops were computed using the `profile` function in [net_profile.py](net_profile.py). I assumed a batch size of 1, and 1000 classes for all models except the CifarNet and LeNet, which assumed 10 classes. All available checkpoint files are from models trained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) dataset. Top-1 and Top-5 numbers correspond to performance on that datasets. When fine-tuning from one of these checkpoints, it is recommended to use the same image size as the default image size for that model.

| Model | Name | TF-Slim File | Checkpoint | Top-1 Accuracy | Top-5 Accuracy | Default Image Size | Num Params | Num Flops |
:----:|:----:|:------------:|:----------:|:-------:|:--------:|:--------:|:--------:|:--------:|
[LeNet](http://yann.lecun.com/exdb/lenet/) | lenet | [Code](lenet.py) | | | | 28px | 3,276,234 | 32.34m |
[CifarNet](https://github.com/akrizhevsky/cuda-convnet2/) | cifarnet | [Code](cifarnet.py) | | | | 32px | 1,756,426 | 65.64m |
[AlexNet](https://arxiv.org/abs/1404.5997) | alexnet_v2 | [Code](alexnet.py) | | | | 224px | 50,303,912 | 1.47b |
[Overfeat](http://arxiv.org/abs/1312.6229) | overfeat | [Code](overfeat.py) | | | | 231px | 145,704,424 | 5.29b |
[Inception V1](http://arxiv.org/abs/1409.4842v1) | inception_v1 | [Code](inception_v1.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz) | 69.8 | 89.6 | 224px | 6,617,624 | 3.00b |
[Inception V2](http://arxiv.org/abs/1502.03167) | inception_v2 | [Code](inception_v2.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz) | 73.9 | 91.8 | 224px | 11,178,336 | 3.87b |
[Inception V3](http://arxiv.org/abs/1512.00567) | inception_v3 | [Code](inception_v3.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) | 78.0 | 93.9 | 299px | 27,143,152 | 11.44b |
[Inception V4](http://arxiv.org/abs/1602.07261) | inception_v4 | [Code](inception_v4.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) | 80.2 | 95.2 | 299px | 46,006,800 | 24.52b |
[Inception-ResNet-v2](http://arxiv.org/abs/1602.07261) | inception_resnet_v2 | [Code](inception_resnet_v2.py) | [Checkpoint](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz) | 80.4 | 95.3 | 299px | 59,179,952 | 26.34b |
[ResNet 50 v1](https://arxiv.org/abs/1512.03385) | resnet_v1_50 | [Code](resnet_v1.py) | [Checkpoint](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) | 75.2 | 92.2 | 224px | 25,557,032 | 6.96b |
[ResNet 101 v1](https://arxiv.org/abs/1512.03385) | resnet_v1_101 | [Code](resnet_v1.py) | [Checkpoint](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) | 76.4 | 92.9 | 224px | 44,549,160 | 14.39b |
[ResNet 152 v1](https://arxiv.org/abs/1512.03385) | resnet_v1_152 | [Code](resnet_v1.py) | [Checkpoint](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz) | 76.8 | 93.2 | 224px | 60,192,808 | 21.81b |
[ResNet 200 v1](https://arxiv.org/abs/1512.03385) | resnet_v1_200 | [Code](resnet_v1.py) | | | | 224px | 64,673,832 | 28.80b |
[ResNet 50 v2](https://arxiv.org/abs/1603.05027) | resnet_v2_50 | [Code](resnet_v2.py) | | | | 224px | 25,568,360 | 6.97b |
[ResNet 101 v2](https://arxiv.org/abs/1603.05027)| resnet_v2_101 | [Code](resnet_v2.py) | | | | 224px | 44,577,896 | 14.40b |
[ResNet 152 v2](https://arxiv.org/abs/1603.05027) | resnet_v2_152 | [Code](resnet_v2.py) | | | | 224px | 60,236,904 | 21.83b |
[ResNet 200 v2](https://arxiv.org/abs/1603.05027) | resnet_v2_200 | [Code](resnet_v2.py) | | | | 224px | 64,726,120 | 28.82b |
[VGG a](http://arxiv.org/abs/1409.1556.pdf) | vgg_a | [Code](vgg.py) | | | | 224px | 132,863,336 | 15.23b |
[VGG 16](http://arxiv.org/abs/1409.1556.pdf)| vgg_16 | [Code](vgg.py) | [Checkpoint](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) | 71.5 | 89.8 | 224px | 138,357,544 | 30.95b |
[VGG 19](http://arxiv.org/abs/1409.1556.pdf)| vgg_19 | [Code](vgg.py) | [Checkpoint](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) | 71.1 | 89.8 | 224px | 143,667,240 | 39.28b |
