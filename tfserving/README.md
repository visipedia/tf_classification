# TensorFlow Serving Utilities

This directory contains utility code for interacting with a [TensorFlow Serving](https://www.tensorflow.org/serving/) instance. I'll walk through the basic steps of using TensorFlow Serving below.

## Export a Trained Model
When your training process has finished you will be left with a training checkpoint file created by the [tf.train.Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver) class. We need to convert this checkpoint file for use with TensorFlow Serving. You'll need to create a yaml configuration file for the export (that species the . An example:

```yaml
# Export specific configuration

RANDOM_SEED : 1.0

SESSION_CONFIG : {
  # If true, then the device location of each variable will be printed
  LOG_DEVICE_PLACEMENT : false,

  # How much GPU memory we are allowed to pre-allocate
  PER_PROCESS_GPU_MEMORY_FRACTION : 0.9
}

#################################################
# Dataset Info
# The number of classes we are classifying
NUM_CLASSES : 200

# The model architecture to use.
MODEL_NAME : 'inception_v3'

# END: Dataset Info
#################################################
# Image Processing and Augmentation 

IMAGE_PROCESSING : {
    # Images are assumed to be raveled, and have length  INPUT_SIZE * INPUT_SIZE * 3
    INPUT_SIZE : 299
}

# END: Image Processing and Augmentation
#################################################
# Regularization 
#
# The decay to use for the moving average. If 0, then moving average is not computed
# When restoring models, this value is needed to determine whether to restore moving
# average variables or not.
MOVING_AVERAGE_DECAY : 0.9999

# End: Regularization
#################################################
```

To export the model, we'll use the [export.py](export.py) script:
```
python export.py \
--checkpoint_path model.ckpt-399739 \
--export_dir export \
--export_version 1 \
--config config_export.yaml \
--serving \
--add_preprocess \
--class_names class-codes.txt
```
We've passed in identifiers for the classes using the `--class_names` argument. This will allow clients to receive semantically meaningful identifiers for the prediction results. The class-codes.txt file contains one identifier per line, with each line corresponding to one index in the scores array. 

This will create a directory called `1` in the `export_dir` directory and will contain the files that TensorFlow Serving requires. 

## Server Machine
Spin up an Ubuntu 16.04 instance on your favorite cloud provider, or use your personal machine. You'll need to add the TensorFlow Serving distribution URI as a package source prior to installing:
```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server
```
You can also install from [source](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md#installation).

Now you can start the server:
```
tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/home/ubuntu/serving/models
```

# Client Machine
To query the server from a client machine you'll need to install the `tensorflow-serving-api` PIP package along with tensorflow. I use numpy for some operations so I'll install that too:
```
pip install numpy tensorflow tensorflow-serving-api
```

We can now query the server using the [client.py](client.py) file:
```
python client.py \
--images IMG_0932_sm.jpg \
--num_results 10 \
--model_name inception \
--host localhost \
--port 9000 \
--timeout 10
```
This command will send the `IMG_0932_sm.jpg` file to the TensorFlow Serving instance at `localhost:9000` and print the top 10 class predictions. 
