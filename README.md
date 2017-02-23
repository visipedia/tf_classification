# TensorFlow Classification
This repo contains training, testing and classifcation code for image classification using [TensorFlow](https://www.tensorflow.org/).

TensorFlow 1.0 is required. The code is tested with Python 2.7 on Ubuntu 14.04 and Mac OSX 10.11

---

## Prepare the Data
The models require the image data to be in a specific format. You can use the Visipedia [tfrecords repo](https://github.com/visipedia/tfrecords) to produce the files. 

---

## Directory Structure
I have found that its useful to have the following directory and file setup:
* experiment/
  * logdir/
    * train_summaries/
    * val_summaries/
    * test_summaries/
    * results/
    * finetune/
      * train_summaries/
      * val_summaries/
  * cmds.txt
  * config_train.yaml
  * config_test.yaml
  * config_export.yaml

The purpose of each directory and file will be explained below. 

The `cmds.txt` is useful to save the different training and testing commands. There are quite a few command-line arguments to some of the scripts, so its convienent to compose the commands in an editor. 

---

## Configuration
There are example configuration files in the [config directory](config/). At the very least you'll need a `config_train.yaml` file, and you'll probably want a `config_test.yaml` file. It is convienent to copy the example configuration files into your `experiment` directory. See the configuration [README](config/README.md) for more details.

### Choose a Network Architecture
This repo currently supports the Google Inception flavor of networks. See the nets [README](nets/README.md) for more information on the different Inception versions. At the moment, `inception_v3` probably offers the best tradeoff in terms of size and performance. The [README](nets/README.md) also contains links where you can download checkpoint files for the models. In most cases you should start your training from these checkpoint files rather than training from scratch. 

You can specify the name of the choosen network in the configuration yaml file. Alternatively you can pass it in as a command-line argument to most of the scripts. 

---

## Training and Validating
It's recommended to start from a pretrained network when training a network on your own data. However, this isn't necessary and you can train from scratch if you have enough data. The following warmup section assumes you are starting from a pretrained network. See the nets [README](nets/README.md) to find links to pretrained checkpoint files.

### Warm Up A Pretrained Network
You'll need to scrap the final layer of a pretrained network in order to output new predictions for your data. Its recommended that you "warm up" this new final layer before training all layers in the network. This is called "fine-tuning."  So the first part of training is to load a pretrained network, and train only the final layer. We can put the trained models in the `experiment/logdir/finetune` directory. 

```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/train/* \
--logdir /Users/GVH/Desktop/cub_train/logdir/finetune \
--config /Users/GVH/Desktop/cub_train/config_train.yaml \
--pretrained_model /Users/GVH/Desktop/Inception_Models/inception_v3.ckpt \
--trainable_scopes InceptionV3/Logits InceptionV3/AuxLogits \
--checkpoint_exclude_scopes InceptionV3/Logits InceptionV3/AuxLogits \
--learning_rate_decay_type fixed \
--lr 0.01 
```

#### Monitoring Progress
We'll want to monitor performance of the model on a validation set. Once the model performance starts to plateau we can assume that the final layer is warmed up and we can switch to full training. We can monitor the validation performance by running:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/val/* \
--save_dir /Users/GVH/Desktop/cub_train/logdir/finetune/val_summaries \
--checkpoint_path /Users/GVH/Desktop/cub_train/logdir/finetune \
--config /Users/GVH/Desktop/cub_train/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

You may want to also monitor the accuracy on the train set. Simply pass in the train tfrecords to the `test.py` script and change the output directory:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/train/* \
--save_dir /Users/GVH/Desktop/cub_train/logdir/finetune/train_summaries \
--checkpoint_path /Users/GVH/Desktop/cub_train/logdir/finetune \
--config /Users/GVH/Desktop/cub_train/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

Keeping the train summaries and val summaries in separate directories will keep the tensorboard ui clean. To monitor the training process you can fireup tensorboard:
```
$ tensorboard --logdir=/Users/GVH/Desktop/cub_train/logdir --port=6006
```

### Full Train
Now we can train all of the layers of the model. We'll store the generated files in the `experiment/logdir` directory:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/* \
--logdir /Users/GVH/Desktop/cub_train/logdir \
--config /Users/GVH/Desktop/cub_train/config_train.yaml \
--pretrained_model /Users/GVH/Desktop/cub_train/logdir/finetune
```

#### Monitoring Progress

And for watching the validation performance we can do:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/val/* \
--save_dir /Users/GVH/Desktop/cub_train/logdir/val_summaries \
--checkpoint_path /Users/GVH/Desktop/cub_train/logdir \
--config /Users/GVH/Desktop/cub_train/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

Similar for the train data: 
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/train/* \
--save_dir /Users/GVH/Desktop/cub_train/logdir/train_summaries \
--checkpoint_path /Users/GVH/Desktop/cub_train/logdir \
--config /Users/GVH/Desktop/cub_train/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

The command for tensorboard doesn't need to change:
```
$ tensorboard --logdir=/Users/GVH/Desktop/cub_train/logdir --port=6006
```
You will be able to see the fine-tune and the full train data.

### Training Without Warming Up
In this case, the `experiment/logdir/finetune` is not necessary. We'll allow all layers to be trained, but we won't load the final layer from the pretrained network. 
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/train/* \
--logdir /Users/GVH/Desktop/cub_train/logdir/ \
--config /Users/GVH/Desktop/cub_train/config_train.yaml \
--pretrained_model /Users/GVH/Desktop/Inception_Models/inception_v3.ckpt \
--checkpoint_exclude_scopes InceptionV3/Logits InceptionV3/AuxLogits
```

### Training From Scratch
If you have enough data to train the network from scratch, then you can do the following:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/train/* \
--logdir /Users/GVH/Desktop/cub_train/logdir/ \
--config /Users/GVH/Desktop/cub_train/config_train.yaml
``` 

---

## Test
Once performance on the validation has plateaued, you can test the model on a held out set of images to see how well it generalizes to new data:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/test/* \
--save_dir /Users/GVH/Desktop/cub_train/logdir/test_summaries \
--checkpoint_path /Users/GVH/Desktop/cub_train/logdir \
--config /Users/GVH/Desktop/cub_train/config_test.yaml \
--batch_size 32 \
--model_name inception_v3 \
--batches 100 \
```

If you are happy with the performance of the model, then you are ready to classify new images and export the model for production use. Otherwise its back to the drawing board to figure out how to increase performance. 

---

## Classifying 
If you want to classify data offline using the trained model then you can do:
```
CUDA_VISIBLE_DEVICES=1 python classify.py \
--tfrecords /Users/GVH/Desktop/cub_tfrecords/new/* \
--checkpoint_path /Users/GVH/Desktop/cub_train \
--save_path /Users/GVH/Desktop/cub_train/logdir/results/classification_results.npz \
--config /Users/GVH/Desktop/cub_train/config_test.yaml \
--batch_size 32 \
--batches 1000 \
--save_logits
```

The output of the script is a numpy uncompressed .npz file saved at `--save_path`. The file will contain at least 2 arrays: one that contains ids and one that contains the predicted class label. If `--save_logits` is specified, then the raw logits (before going through the softmax) will also be saved. 

---

## Export & Compress
To export a model for easy use on a mobile device you can use:
```
python export.py \
--checkpoint_path /Users/GVH/Desktop/cub_train/logdir \
--export_dir /Users/GVH/Desktop/cub_train \
--export_version 1 \
--config /Users/GVH/Desktop/cub_train/config_export.yaml
```
The input node is called `images` and the output node is called `Predictions`.

If you are going to use the model with [TensorFlow Serving](https://www.tensorflow.org/deploy/tfserve) then you can use the following:
```
python export.py \
--checkpoint_path /Users/GVH/Desktop/cub_train/logdir \
--export_dir /Users/GVH/Desktop/cub_train \
--export_version 1 \
--config /Users/GVH/Desktop/cub_train/config_export.yaml \
--serving
```
