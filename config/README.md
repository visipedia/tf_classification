This directory contains example configuration scripts for training, testing, classifying and exporting models. I find it easy to copy these configuration files to my experiment directory and make the necessary changes. 

## Training Configuration
See the [example training config file](config_train.yaml). 

The training configuration script contains the most configurations. The other scripts mainly contain subsets of the training configuration. The `Learning Rate Parameters`, `Regularization`, and `Optimization` configurations provided experimenters fine-grained control over the learning process. Non-researchers will probably find most of the default settings adequate. I will not go into detail for these configuration parameters, but there are comments for these parameters in the [example training config file](config_train.yaml).

The configuration sections that you will want to pay attention to are the `Dataset Info` section and the `Image Processing and Augmentation` section. You'll most likely be modifying these for each experiment. Once you determine good settings for the `Queues` and `Saving Models and Summaries` you'll probably reuse these values across experiments.

### Dataset Info
| Config Name | Type | Description |
:----:|:----:|------------|
NUM_CLASSES | int | This is how you specify how many classes are in your dataset. |
NUM_TRAIN_EXAMPLES | int | This is the number of images (or bounding boxes) in your training tfrecords. This value, along with the `BATCH_SIZE` is used to compute the number of iterations in an epoch (i.e. the number of batches it takes to go through the whole training set) |
NUM_TRAIN_ITERATIONS | int | The maximum number of iterations to execute before stopping. If you are manually monitoring the training, then you can set this to a large number (e.g. 1000000) |
BATCH_SIZE | int | The number of images to process in one iteration. This number is constrained by the amount of GPU memory you have. The larger the batch size, the more GPU memory you need. You typically want the largest batch size that will fit on your GPU. |
MODEL_NAME | str | The architecture to use. Its important to keep this configuration parameter constant in all of your configuration files. |

### Image Processing and Augmentation
Deep neural networks are notoriously data hungry. One technique for increasing the amount of data that you can pass through the network is to augment your training data. Augmentations can be as simple as randomly flipping the images horizontally, or as complex as extracting crops and perturbing the pixel values. You will typically only want to augment data for the training phase. 

`IMAGE_PROCESSING` contains the parameters for controlling how to extract data from the images:

| Config Name | Type | Description |
:----:|:----:|------------|
INPUT_SIZE | int | All images will be resized to [`INPUT_SIZE`, `INPUT_SIZE`, 3] prior to passing through the network. You'll want to set this to the same value that the pretrained model used. See the nets [README](../nets/README.md) for the input size of each model architecture. |
REGION_TYPE | str | Which region should be used when creating an example? Possible values are `image` and `bbox`. |
MAINTAIN_ASPECT_RATIO | bool | When we resize an extracted region, should we maintain the aspect ratio? Or just squish it? 
RESIZE_FAST | bool | If true, then slower resize operations will be avoided and only [bilinear resizing](https://en.wikipedia.org/wiki/Bilinear_interpolation) will be used. Otherwise, a random choice between [bilinear](), [nearest neighbor](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation), [bicubic](https://en.wikipedia.org/wiki/Bicubic_interpolation) and area interpolation will be used. |
DO_RANDOM_FLIP_LEFT_RIGHT | bool | If true, then each region has a 50% chance of being flipped. | 
DO_COLOR_DISTORTION | float | Value between 0 and 1. 0 means never distort the color, and 1 means always distort the color. |
COLOR_DISTORT_FAST | bool | Its possible to distort the brightness, saturation, hue and contrast of an image. If true, then slower modifications (hue and contrast) are avoided. |

#### Region Extraction

Currently there are two different region extraction protocols: 
* `image`: The entire image is extracted and passed to the next phase of augmentation 
* `bbox`: Each bounding box in the tfrecord is used to crop out an image region. These regions are passed on to the next phase of augmentation. If there are `n` bounding boxes in a tfrecord, then `n` regions will be extracted from the image. 

For bounding boxes, we can specify wether we want to enlarge the box. This can be used as another form of augmentation (loose bounding boxes vs tight bounding boxes).

| Config Name | Type | Description |
:----:|:----:|------------|
DO_EXPANSION | float | Value between 0 and 1. 0 means never expand the box. 1 means always expand the box. |
EXPANSION_CFG | | Contains the parameters controlling the expansion of the bounding box. | 
EXPANSION_CFG.<br />WIDTH_EXPANSION_FACTOR | float | Scaling factor for the width of the box. | 
EXPANSION_CFG.<br />HEIGHT_EXPANSION_FACTOR | float | Scaling factor for the height of the box. | 


#### Random Cropping

Each region that is extracted from an image can then be randomly cropped. Again, this is a form of data augmentation. We are trying to make the network robust to changes in the data that do not effect the class label. 

`RANDOM_CROP_CFG` contains parameters for cropping out a rectangular patch from each region. 

| Config Name | Type | Description |
:----:|:----:|------------|
DO_RANDOM_CROP | float | Value between 0 and 1. 0 means never crop a region. 1 means always take a crop. |
RANDOM_CROP_CFG | | This contains parameters that controls the types of crops that are possible. |
RANDOM_CROP_CFG.<br />MIN_AREA | float | Value between 0 and 1. This controls how much of the region is required to be in the crop, essentially controlling how small a crop can be. |
RANDOM_CROP_CFG.<br />MAX_AREA | float | Value between 0 and 1. This controls the maximum size of the crop. |
RANDOM_CROP_CFG.<br />MIN_ASPECT_RATIO | float | The minimum [aspect ratio](https://en.wikipedia.org/wiki/Aspect_ratio_(image)) of the crop. Don't forget that this crop will be resized to [`INPUT_SIZE`, `INPUT_SIZE`, 3] prior to passing through the network. |
RANDOM_CROP_CFG.<br />MAX_ASPECT_RATIO | float | The maximum [aspect ratio](https://en.wikipedia.org/wiki/Aspect_ratio_(image)) of the crop. Don't forget that this crop will be resized to [`INPUT_SIZE`, `INPUT_SIZE`, 3] prior to passing through the network. |
RANDOM_CROP_CFG.<br />MAX_ATTEMPTS | int | The number of crop attempts to try before returning the whole region. |

### Queues
This section of the config file contains parameters for controlling the queueing of data to feed the network. These setting depend on the number of cores in your machine and the amount of memory available. Please see the comments in the example config file for more information. 

### Saving Models and Summaries 
This section of the config file contains parameters for controlling how often a model checkpoint should be created and how often tensorboard summary files should be generated. Please see the comments in the example config file for more information. 

## Testing Configuration
See the [example testing config file](config_test.yaml). 

The `Learning Rate Parameters`, `Optimization`, and `Saving Models and Summaries` parameters are not necessary for testing. The remaining parameters from the training config carry over to testing. In addition there are a few new configurations:

| Config Name | Type | Description |
:----:|:----:|------------|
PRECISION_AT_K_METRIC | array of ints | You can track top-k metrics using this array. Top-1 (i.e. accuracy) will always be plotted |
NUM_TEST_EXAMPLES | int | The number of images (or bounding boxes) in the tfrecords. This can be ignored if you use the `--batches` command line flag. | 

Typically in a testing situation you'll want to turn off the augmentations to the extracted image regions. This way you are passing "real" data to the network. See the `Image Processing and Augmentation` section of the [example testing config file](config_test.yaml) to see how to extract regions without augmentations.

## Classification Configuration
See the [example classification config file](config_classify.yaml).

The classification configuration contains even fewer necessary fields than the testing configuration. The `Metrics` section is removed and you'll need to pass batch size and total batch information through command-line arguments. 

## Export Configuration
See the [example export config file](config_export.yaml).

The export configuration is the smallest configuration file. See the [example](config_export.yaml) for which fields are required. 
