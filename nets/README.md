# Models

This directory contains the available classification models. All of these models were copied from the [TensorFlow Models repo](https://github.com/tensorflow/models/tree/master/slim/nets) and updated to TensorFlow r1.0.

The table below lists relevant information for each model. To use one of these models (e.g. when using the training scripts), simply set the `--model_name` flag to the appropriate name. The number of parameters and the number of flops were computed using the `profile` function in [net_profile.py](net_profile.py). I assumed a batch size of 1, and 1000 classes for all models. All available checkpoint files are from models trained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) dataset. Top-1 and Top-5 numbers correspond to performance on that datasets. When fine-tuning from one of these checkpoints, it is recommended to use the same image size as the default image size for that model.

| Model | Name | TF-Slim File | Checkpoint | Top-1 Accuracy | Top-5 Accuracy | Default Image Size | Num Params | Num Flops |
:----:|:----:|:------------:|:----------:|:-------:|:--------:|:--------:|:--------:|:--------:|
[Inception V1](http://arxiv.org/abs/1409.4842v1) | inception_v1 | [Code](inception_v1.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz) | 69.8 | 89.6 | 224px | 6,617,624 | 3.00b |
[Inception V2](http://arxiv.org/abs/1502.03167) | inception_v2 | [Code](inception_v2.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz) | 73.9 | 91.8 | 224px | 11,178,336 | 3.87b |
[Inception V3](http://arxiv.org/abs/1512.00567) | inception_v3 | [Code](inception_v3.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) | 78.0 | 93.9 | 299px | 27,143,152 | 11.44b |
[Inception V4](http://arxiv.org/abs/1602.07261) | inception_v4 | [Code](inception_v4.py) | [Checkpoint](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) | 80.2 | 95.2 | 299px | 46,006,800 | 24.52b |
[Inception-ResNet-v2](http://arxiv.org/abs/1602.07261) | inception_resnet_v2 | [Code](inception_resnet_v2.py) | [Checkpoint](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz) | 80.4 | 95.3 | 299px | 59,179,952 | 26.34b |
[ResNet V2 50](https://arxiv.org/abs/1603.05027) | resnet_v2_50 | [Code](resnet_v2.py) | [Checkpoint](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz) | 75.6 | 92.8 | 299px | 25,568,360 | 13.08b |
[ResNet V2 101](https://arxiv.org/abs/1603.05027) | resnet_v2_101 | [Code](resnet_v2.py) | [Checkpoint](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) | 77.0 | 93.7 | 299px | 44,577,896 | 26.77b |
[ResNet V2 152](https://arxiv.org/abs/1603.05027) | resnet_v2_152 | [Code](resnet_v2.py) | [Checkpoint](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz) | 77.8 | 94.1 | 299px | 60,236,904 | 40.45b |
[MobileNet-v1](https://arxiv.org/abs/1704.04861) | mobilenet_v1 | [Code](mobilenet_v1.py) | [Checkpoint](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz) | 70.7 | 89.5 | 224px | 4,231,976 | 1.14b |