# TensorFlow Classification
This repo contains training, testing and classifcation code for image classification using [TensorFlow](https://www.tensorflow.org/). Whole image classification as well as multi instance bounding box classification is supported. 

Checkout the [Wiki](https://github.com/visipedia/tf_classification/wiki) for more detailed tutorials. 

---

## Requirements
TensorFlow 1.0+ is required. The code is tested with TensorFlow 1.3 and Python 2.7 on Ubuntu 16.04 and Mac OSX 10.11. Check out the [requirements.txt](requirements.txt) file for a list of python dependencies. 

---

## Prepare the Data
The models require the image data to be in a specific format. You can use the Visipedia [tfrecords repo](https://github.com/visipedia/tfrecords) to produce the files. 

For the commands below, I'll assume that you have created a `DATASET_DIR` environment variable that points to the directory that contains your tfrecords:
```
$ export DATASET_DIR=/home/ubuntu/tf_datasets/cub
```

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

For the commands below, I'll assume that you have created a `EXPERIMENT_DIR` environment variable that points to your experiment directory:
```
$ export EXPERIMENT_DIR=/home/ubuntu/tf_experiments/cub
```

---

## Configuration
There are example configuration files in the [config directory](config/). At the very least you'll need a `config_train.yaml` file, and you'll probably want a `config_test.yaml` file. It is convienent to copy the example configuration files into your `experiment` directory. See the configuration [README](config/README.md) for more details.

### Choose a Network Architecture
This repo currently supports the Google Inception, ResNet and MobileNet flavor of networks. See the nets [README](nets/README.md) for more information on the different Inception versions. At the moment, `inception_v3` probably offers the best tradeoff in terms of size and performance, although its always worth experimenting with a few different architectures. The [README](nets/README.md) also contains links where you can download checkpoint files for the models. In most cases you should start your training from these checkpoint files rather than training from scratch. 

You can specify the name of the choosen network in the configuration yaml file. Alternatively you can pass it in as a command-line argument to most of the scripts. 

For the commands below, I'll assume that you have created an environment variable that points to the pretrained checkpoint file that you downloaded:
```
$ export PRETRAINED_MODEL=/home/ubuntu/tf_models/inception_v3.ckpt
```

---

## Data Visualization
Now that you have a configuration script for training, it is a good idea to visualize the inputs to the network and ensure that they look good. This allows you to debug any problems with your tfrecords and lets you play with different augmentation techniques. Visualize your data by doing:
```
$ CUDA_VISIBLE_DEVICES=1 python visualize_train_inputs.py \
--tfrecords $DATASET_DIR/train* \
--config $EXPERIMENT_DIR/config_train.yaml
```

If you are in a virtualenv and Matplotlib is complaining, then you may need to modify your environment. See this [FAQ](http://matplotlib.org/faq/virtualenv_faq.html) and [this document](http://matplotlib.org/faq/osx_framework.html#osxframework-faq) for fixing this issue. I use a virtualenv on my Mac OSX 10.11 machine and I needed to do the `PYTHONHOME` [work around](http://matplotlib.org/faq/osx_framework.html#pythonhome-function) for Matplotlib to work properly. In this case the command looks like:
```
$ CUDA_VISIBLE_DEVICES=1 frameworkpython visualize_train_inputs.py \
--tfrecords $DATASET_DIR/train* \
--config $EXPERIMENT_DIR/config_train.yaml
```

---

## Training and Validating
It's recommended to start from a pretrained network when training a network on your own data. However, this isn't necessary and you can train from scratch if you have enough data. The following warmup section assumes you are starting from a pretrained network. See the nets [README](nets/README.md) to find links to pretrained checkpoint files.

### Finetune A Pretrained Network
Finetuning a pretrained network essentially uses the pretrained network as a generic feature extractor and learns a new final layer that will output predictions for your target classes (rather than the original classes that the pretrained network was trained on). To do this, we will specify the pretrained model as the starting point, and only allow the logits layers to be modified. We can put the trained models in the `experiment/logdir/finetune` directory. 

```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logdir/finetune \
--config $EXPERIMENT_DIR/config_train.yaml \
--pretrained_model $PRETRAINED_MODEL \
--trainable_scopes InceptionV3/Logits InceptionV3/AuxLogits \
--checkpoint_exclude_scopes InceptionV3/Logits InceptionV3/AuxLogits \
--learning_rate_decay_type fixed \
--lr 0.01 
```

#### Monitoring Progress
We'll want to monitor performance of the model on a validation set. Once the model performance starts to plateau we can assume that the final layer is warmed up and we can switch to full training. We can monitor the validation performance by running:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords $DATASET_DIR/val* \
--save_dir $EXPERIMENT_DIR/logdir/finetune/val_summaries \
--checkpoint_path $EXPERIMENT_DIR/logdir/finetune \
--config $EXPERIMENT_DIR/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

You may want to also monitor the accuracy on the train set. Simply pass in the train tfrecords to the `test.py` script and change the output directory:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords $DATASET_DIR/train* \
--save_dir $EXPERIMENT_DIR/logdir/finetune/train_summaries \
--checkpoint_path $EXPERIMENT_DIR/logdir/finetune \
--config $EXPERIMENT_DIR/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

Keeping the train summaries and val summaries in separate directories will keep the tensorboard ui clean. To monitor the training process you can fireup tensorboard:
```
$ tensorboard --logdir=$EXPERIMENT_DIR/logdir --port=6006
```

### Training the Entire Network
The benefit of finetuning a network is that the training is very fast, as only the last layer is modified. However, to get the best performance you'll typically want to modify more (or all) of the layers of the network. Starting from a pretrained network (which can happen to be a finetuned network), this full training step essentially adapts the network to operating on the domain of your specific dataset.  We'll store the generated files in the `experiment/logdir` directory. You can do the finetuning process as a warmup and then start the full train:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logdir \
--config $EXPERIMENT_DIR/config_train.yaml \
--pretrained_model $EXPERIMENT_DIR/logdir/finetune
```

Or you can just start the full train from a pretrained model:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logdir \
--config $EXPERIMENT_DIR/config_train.yaml \
--pretrained_model $PRETRAINED_MODEL \
--checkpoint_exclude_scopes InceptionV3/Logits InceptionV3/AuxLogits
```

Or if you have enough data, you may not want to even use the pretrained model. Rather you can train from scratch:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logdir/ \
--config $EXPERIMENT_DIR/config_train.yaml
``` 

#### Monitoring Progress

For watching the validation performance we can do:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords $DATASET_DIR/val* \
--save_dir $EXPERIMENT_DIR/logdir/val_summaries \
--checkpoint_path $EXPERIMENT_DIR/logdir \
--config $EXPERIMENT_DIR/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

Similar for the train data: 
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords $DATASET_DIR/train* \
--save_dir $EXPERIMENT_DIR/train_summaries \
--checkpoint_path $EXPERIMENT_DIR/logdir \
--config $EXPERIMENT_DIR/config_test.yaml \
--batches 100 \
--eval_interval_secs 300
```

The command for tensorboard doesn't need to change:
```
$ tensorboard --logdir=$EXPERIMENT_DIR/logdir --port=6006
```
You will be able to see the fine-tune and the full train data plotted on the same plots. 

---

## Test
Once performance on the validation data has plateaued (or some other criterion has been met), you can test the model on a held out set of images to see how well it generalizes to new data:
```
$ CUDA_VISIBLE_DEVICES=1 python test.py \
--tfrecords $DATASET_DIR/test* \
--save_dir $EXPERIMENT_DIR/logdir/test_summaries \
--checkpoint_path $EXPERIMENT_DIR/logdir \
--config $EXPERIMENT_DIR/config_test.yaml \
--batch_size 32 \
--batches 100
```

If you are happy with the performance of the model, then you are ready to classify new images and export the model for production use. Otherwise its back to the drawing board to figure out how to increase performance. 

---

## Classifying 
If you want to classify data offline using the trained model then you can do:
```
CUDA_VISIBLE_DEVICES=1 python classify.py \
--tfrecords $DATASET_DIR/new/* \
--checkpoint_path $EXPERIMENT_DIR/logdir \
--save_path $EXPERIMENT_DIR/logdir/results/classification_results.npz \
--config $EXPERIMENT_DIR/config_test.yaml \
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
--checkpoint_path model.ckpt-399739 \
--export_dir ./export \
--export_version 1 \
--config config_export.yaml \
--class_names class-codes.txt
```
The input node is called `images` and the output node is called `Predictions`. Checkout [this](https://github.com/visipedia/tf_classification/wiki/Exporting-an-Optimized-Model) wiki article for more tips. 

If you are going to use the model with [TensorFlow Serving](https://www.tensorflow.org/deploy/tfserve) then you can use the following:
```
python export.py \
--checkpoint_path model.ckpt-399739 \
--export_dir ./export \
--export_version 1 \
--config config_export.yaml \
--serving \
--add_preprocess \
--class_names class-codes.txt
```
Check out the resources in the [tfserving](tfserving/) directory for more help with deploying on TensorFlow Serving.
