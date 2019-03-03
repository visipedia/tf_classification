# Some of this code came from the https://github.com/tensorflow/models/tree/master/slim
# directory, so lets keep the Google license around for now.
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from preprocessing.decode_example import decode_serialized_example



def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return tf.tuple([cropped_image, distort_bbox])

def _largest_size_at_most(height, width, largest_side):
  """Computes new shape with the largest side equal to `largest_side`.
  Computes new shape with the largest side equal to `largest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    largest_side: A python integer or scalar `Tensor` indicating the size of
      the largest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  largest_side = tf.convert_to_tensor(largest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  largest_side = tf.to_float(largest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: largest_side / height,
                  lambda: largest_side / width)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

class DistortedInputs():

    def __init__(self, cfg, add_summaries):
        self.cfg = cfg
        self.add_summaries = add_summaries

    def apply(self, original_image, bboxes, distorted_inputs, image_summaries, current_index):

        cfg = self.cfg
        add_summaries = self.add_summaries

        image_shape = tf.shape(original_image)
        image_height = tf.cast(image_shape[0], dtype=tf.float32) # cast so that we can multiply them by the bbox coords
        image_width = tf.cast(image_shape[1], dtype=tf.float32)

        # First thing we need to do is crop out the bbox region from the image
        bbox = bboxes[current_index]
        xmin = tf.cast(bbox[0] * image_width, tf.int32)
        ymin = tf.cast(bbox[1] * image_height, tf.int32)
        xmax = tf.cast(bbox[2] * image_width, tf.int32)
        ymax = tf.cast(bbox[3] * image_height, tf.int32)
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        image = tf.image.crop_to_bounding_box(
            image=original_image,
            offset_height=ymin,
            offset_width=xmin,
            target_height=bbox_height,
            target_width=bbox_width
        )
        image_height = bbox_height
        image_width = bbox_width

        # Convert the pixel values to be in the range [0,1]
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Add a summary of the original data
        if add_summaries:
            new_height, new_width = _largest_size_at_most(image_height, image_width, cfg.INPUT_SIZE)
            resized_original_image = tf.image.resize_bilinear(tf.expand_dims(image, 0), [new_height, new_width])
            resized_original_image = tf.squeeze(resized_original_image)
            resized_original_image = tf.image.pad_to_bounding_box(resized_original_image, 0, 0, cfg.INPUT_SIZE, cfg.INPUT_SIZE)

            # If there are multiple boxes for an image, we only want to write to the TensorArray once.
            #image_summaries = image_summaries.write(0, tf.expand_dims(resized_original_image, 0))
            image_summaries = tf.cond(tf.equal(current_index, 0),
                lambda: image_summaries.write(0, tf.expand_dims(resized_original_image, 0)),
                lambda: image_summaries.identity()
            )

        # Extract a distorted bbox
        if cfg.DO_RANDOM_CROP > 0:
            r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            do_crop = tf.less(r, cfg.DO_RANDOM_CROP)
            rc_cfg = cfg.RANDOM_CROP_CFG
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
            distorted_image, distorted_bbox = tf.cond(do_crop,
                    lambda: distorted_bounding_box_crop(image, bbox,
                                                           aspect_ratio_range=(rc_cfg.MIN_ASPECT_RATIO, rc_cfg.MAX_ASPECT_RATIO),
                                                           area_range=(rc_cfg.MIN_AREA, rc_cfg.MAX_AREA),
                                                           max_attempts=rc_cfg.MAX_ATTEMPTS),
                    lambda: tf.tuple([image, bbox])
                )
        else:
            distorted_image = tf.identity(image)
            distorted_bbox = tf.constant([[[0.0, 0.0, 1.0, 1.0]]]) # ymin, xmin, ymax, xmax

        if cfg.DO_CENTRAL_CROP > 0:
            r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            do_crop = tf.less(r, cfg.DO_CENTRAL_CROP)
            distorted_image = tf.cond(do_crop,
                lambda: tf.image.central_crop(distorted_image, cfg.CENTRAL_CROP_FRACTION),
                lambda: tf.identity(distorted_image)
            )

        distorted_image.set_shape([None, None, 3])

        # Add a summary
        if add_summaries:
            image_with_bbox = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distorted_bbox)
            new_height, new_width = _largest_size_at_most(image_height, image_width, cfg.INPUT_SIZE)
            resized_image_with_bbox = tf.image.resize_bilinear(image_with_bbox, [new_height, new_width])
            resized_image_with_bbox = tf.squeeze(resized_image_with_bbox)
            resized_image_with_bbox = tf.image.pad_to_bounding_box(resized_image_with_bbox, 0, 0, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
            #image_summaries = image_summaries.write(1, tf.expand_dims(resized_image_with_bbox, 0))
            image_summaries = tf.cond(tf.equal(current_index, 0),
                lambda: image_summaries.write(1, tf.expand_dims(resized_image_with_bbox, 0)),
                lambda: image_summaries.identity()
            )

        # Resize the distorted image to the correct dimensions for the network
        if cfg.MAINTAIN_ASPECT_RATIO:
            shape = tf.shape(distorted_image)
            height = shape[0]
            width = shape[1]
            new_height, new_width = _largest_size_at_most(height, width, cfg.INPUT_SIZE)
        else:
            new_height = cfg.INPUT_SIZE
            new_width = cfg.INPUT_SIZE

        num_resize_cases = 1 if cfg.RESIZE_FAST else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [new_height, new_width], method=method),
            num_cases=num_resize_cases)

        distorted_image = tf.image.pad_to_bounding_box(distorted_image, 0, 0, cfg.INPUT_SIZE, cfg.INPUT_SIZE)

        if add_summaries:
            #image_summaries = image_summaries.write(2, tf.expand_dims(distorted_image, 0))
            image_summaries = tf.cond(tf.equal(current_index, 0),
                lambda: image_summaries.write(2, tf.expand_dims(distorted_image, 0)),
                lambda: image_summaries.identity()
            )

        # Randomly flip the image:
        if cfg.DO_RANDOM_FLIP_LEFT_RIGHT > 0:
          r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
          do_flip = tf.less(r, 0.5)
          distorted_image = tf.cond(do_flip, lambda: tf.image.flip_left_right(distorted_image), lambda: tf.identity(distorted_image))

        # TODO: Can this be changed so that we don't always distort the colors?
        # Distort the colors
        if cfg.DO_COLOR_DISTORTION > 0:
            r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            do_color_distortion = tf.less(r, cfg.DO_COLOR_DISTORTION)
            num_color_cases = 1 if cfg.COLOR_DISTORT_FAST else 4
            distorted_color_image = apply_with_random_selector(
              distorted_image,
              lambda x, ordering: distort_color(x, ordering, fast_mode=cfg.COLOR_DISTORT_FAST),
              num_cases=num_color_cases)
            distorted_image = tf.cond(do_color_distortion, lambda: tf.identity(distorted_color_image), lambda: tf.identity(distorted_image))

        distorted_image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])

        # Add a summary
        if add_summaries:
            #image_summaries = image_summaries.write(3, tf.expand_dims(distorted_image, 0))
            image_summaries = tf.cond(tf.equal(current_index, 0),
                lambda: image_summaries.write(3, tf.expand_dims(distorted_image, 0)),
                lambda: image_summaries.identity()
            )

        # Add the distorted image to the TensorArray
        distorted_inputs = distorted_inputs.write(current_index, tf.expand_dims(distorted_image, 0))

        return [original_image, bboxes, distorted_inputs, image_summaries, current_index + 1]

def check_normalized_box_values(xmin, ymin, xmax, ymax, maximum_normalized_coordinate=1.01, prefix=""):
    """ Make sure the normalized coordinates are less than 1
    """

    xmin_maximum = tf.reduce_max(xmin)
    xmin_assert = tf.Assert(
        tf.greater_equal(1.01, xmin_maximum),
        ['%s, maximum xmin coordinate value is larger '
         'than %f: ' % (prefix, maximum_normalized_coordinate), xmin_maximum])
    with tf.control_dependencies([xmin_assert]):
        xmin = tf.identity(xmin)

    ymin_maximum = tf.reduce_max(ymin)
    ymin_assert = tf.Assert(
        tf.greater_equal(1.01, ymin_maximum),
        ['%s, maximum ymin coordinate value is larger '
        'than %f: ' % (prefix, maximum_normalized_coordinate), ymin_maximum])
    with tf.control_dependencies([ymin_assert]):
        ymin = tf.identity(ymin)

    xmax_maximum = tf.reduce_max(xmax)
    xmax_assert = tf.Assert(
        tf.greater_equal(1.01, xmax_maximum),
        ['%s, maximum xmax coordinate value is larger '
        'than %f: ' % (prefix, maximum_normalized_coordinate), xmax_maximum])
    with tf.control_dependencies([xmax_assert]):
        xmax = tf.identity(xmax)

    ymax_maximum = tf.reduce_max(ymax)
    ymax_assert = tf.Assert(
        tf.greater_equal(1.01, ymax_maximum),
        ['%s, maximum ymax coordinate value is larger '
        'than %f: ' % (prefix, maximum_normalized_coordinate), ymax_maximum])
    with tf.control_dependencies([ymax_assert]):
        ymax = tf.identity(ymax)

    return xmin, ymin, xmax, ymax

def expand_bboxes(xmin, xmax, ymin, ymax, cfg):
    """
    Expand the bboxes. Don't allow to expand past image boundaries.
    """

    w = xmax - xmin
    h = ymax - ymin

    w = w * cfg.WIDTH_EXPANSION_FACTOR
    h = h * cfg.HEIGHT_EXPANSION_FACTOR

    half_w = w / 2.
    half_h = h / 2.

    xmin = tf.clip_by_value(xmin - half_w, 0, 1)
    xmax = tf.clip_by_value(xmax + half_w, 0, 1)
    ymin = tf.clip_by_value(ymin - half_h, 0, 1)
    ymax = tf.clip_by_value(ymax + half_h, 0, 1)

    return tf.tuple([xmin, xmax, ymin, ymax])

def get_region_data(serialized_example, cfg, fetch_ids=True, fetch_labels=True, fetch_text_labels=True, read_filename=False):
    """
    Return the image, an array of bounding boxes, and an array of ids.
    """

    feature_dict = {}

    if cfg.REGION_TYPE == 'bbox':

        bbox_cfg = cfg.BBOX_CFG

        features_to_extract = [('image/object/bbox/xmin', 'xmin'),
                               ('image/object/bbox/xmax', 'xmax'),
                               ('image/object/bbox/ymin', 'ymin'),
                               ('image/object/bbox/ymax', 'ymax'),
                               ('image/object/bbox/ymax', 'ymax')]

        if read_filename:
            features_to_extract.append(('image/filename', 'filename'))
        else:
            features_to_extract.append(('image/encoded', 'image'))

        if fetch_ids:
            features_to_extract.append(('image/object/id', 'id'))

        if fetch_labels:
            features_to_extract.append(('image/object/bbox/label', 'label'))

        if fetch_text_labels:
            features_to_extract.append(('image/object/bbox/text', 'text'))

        features = decode_serialized_example(serialized_example, features_to_extract)

        if read_filename:
            image_buffer = tf.read_file(features['filename'])
            image = tf.image.decode_jpeg(image_buffer, channels=3)
        else:
            image = features['image']

        feature_dict['image'] = image

        xmin = tf.expand_dims(features['xmin'], 0)
        ymin = tf.expand_dims(features['ymin'], 0)
        xmax = tf.expand_dims(features['xmax'], 0)
        ymax = tf.expand_dims(features['ymax'], 0)

        xmin, ymin, xmax, ymax = check_normalized_box_values(xmin, ymin, xmax, ymax, prefix="From tfrecords ")

        if 'DO_EXPANSION' in bbox_cfg and bbox_cfg.DO_EXPANSION > 0:
            r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            do_expansion = tf.less(r, bbox_cfg.DO_EXPANSION)
            xmin, xmax, ymin, ymax = tf.cond(do_expansion,
                lambda: expand_bboxes(xmin, xmax, ymin, ymax, bbox_cfg.EXPANSION_CFG),
                lambda: tf.tuple([xmin, xmax, ymin, ymax])
            )

            xmin, ymin, xmax, ymax = check_normalized_box_values(xmin, ymin, xmax, ymax, prefix="After expansion ")

        # combine the bounding boxes
        bboxes = tf.concat(values=[xmin, ymin, xmax, ymax], axis=0)
        # order the bboxes so that they have the shape: [num_bboxes, bbox_coords]
        bboxes = tf.transpose(bboxes, [1, 0])

        feature_dict['bboxes'] = bboxes

        if fetch_ids:
            ids = features['id']
            feature_dict['ids'] = ids

        if fetch_labels:
            labels = features['label']
            feature_dict['labels'] = labels

        if fetch_text_labels:
            text = features['text']
            feature_dict['text'] = text

    elif cfg.REGION_TYPE == 'image':

        features_to_extract = []

        if read_filename:
            features_to_extract.append(('image/filename', 'filename'))
        else:
            features_to_extract.append(('image/encoded', 'image'))

        if fetch_ids:
            features_to_extract.append(('image/id', 'id'))

        if fetch_labels:
            features_to_extract.append(('image/class/label', 'label'))

        if fetch_text_labels:
            features_to_extract.append(('image/class/text', 'text'))

        features = decode_serialized_example(serialized_example, features_to_extract)

        if read_filename:
            image_buffer = tf.read_file(features['filename'])
            image = tf.image.decode_jpeg(image_buffer, channels=3)
        else:
            image = features['image']

        feature_dict['image'] = image

        bboxes = tf.constant([[0.0, 0.0, 1.0, 1.0]])
        feature_dict['bboxes'] = bboxes

        if fetch_ids:
            ids = [features['id']]
            feature_dict['ids'] = ids

        if fetch_labels:
            labels = [features['label']]
            feature_dict['labels'] = labels

        if fetch_text_labels:
            text = [features['text']]
            feature_dict['text'] = text

    else:
        raise ValueError("Unknown REGION_TYPE: %s" % (cfg.REGION_TYPE,))

    return feature_dict

def bbox_crop_loop_cond(original_image, bboxes, distorted_inputs, image_summaries, current_index):
    num_bboxes = tf.shape(bboxes)[0]
    return current_index < num_bboxes

def get_distorted_inputs(original_image, bboxes, cfg, add_summaries):

    distorter = DistortedInputs(cfg, add_summaries)
    num_bboxes = tf.shape(bboxes)[0]
    distorted_inputs = tf.TensorArray(
        dtype=tf.float32,
        size=num_bboxes,
        element_shape=tf.TensorShape([1, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    )

    if add_summaries:
        image_summaries = tf.TensorArray(
            dtype=tf.float32,
            size=4,
            element_shape=tf.TensorShape([1, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
        )
    else:
        image_summaries = tf.constant([])

    current_index = tf.constant(0, dtype=tf.int32)

    loop_vars = [original_image, bboxes, distorted_inputs, image_summaries, current_index]
    original_image, bboxes, distorted_inputs, image_summaries, current_index = tf.while_loop(
        cond=bbox_crop_loop_cond,
        body=distorter.apply,
        loop_vars=loop_vars,
        parallel_iterations=10, back_prop=False, swap_memory=False
    )

    distorted_inputs = distorted_inputs.concat()

    if add_summaries:
        tf.summary.image('0.original_image', image_summaries.read(0))
        tf.summary.image('1.image_with_random_crop', image_summaries.read(1))
        tf.summary.image('2.cropped_resized_image', image_summaries.read(2))
        tf.summary.image('3.final_distorted_image', image_summaries.read(3))


    return distorted_inputs

def create_training_batch(serialized_example, cfg, add_summaries, read_filenames=False):

    features = get_region_data(serialized_example, cfg, fetch_ids=False,
                               fetch_labels=True, fetch_text_labels=False, read_filename=read_filenames)

    original_image = features['image']
    bboxes = features['bboxes']
    labels = features['labels']

    distorted_inputs = get_distorted_inputs(original_image, bboxes, cfg, add_summaries)

    distorted_inputs = tf.subtract(distorted_inputs, 0.5)
    distorted_inputs = tf.multiply(distorted_inputs, 2.0)

    names = ('inputs', 'labels')
    tensors = [distorted_inputs, labels]
    return [names, tensors]

def create_visualization_batch(serialized_example, cfg, add_summaries, fetch_text_labels=False, read_filenames=False):

    features = get_region_data(serialized_example, cfg, fetch_ids=True,
                               fetch_labels=True, fetch_text_labels=fetch_text_labels, read_filename=read_filenames)

    original_image = features['image']
    ids = features['ids']
    bboxes = features['bboxes']
    labels = features['labels']
    if fetch_text_labels:
        text_labels = features['text']

    cpy_original_image = tf.identity(original_image)

    distorted_inputs = get_distorted_inputs(original_image, bboxes, cfg, add_summaries)

    original_image = cpy_original_image

    # Resize the original image
    if original_image.dtype != tf.float32:
      original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)
    shape = tf.shape(original_image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _largest_size_at_most(height, width, cfg.INPUT_SIZE)
    original_image = tf.image.resize_images(original_image, [new_height, new_width], method=0)
    original_image = tf.image.pad_to_bounding_box(original_image, 0, 0, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
    original_image = tf.image.convert_image_dtype(original_image, dtype=tf.uint8)

    # make a copy of the original image for each bounding box
    num_bboxes = tf.shape(bboxes)[0]
    expanded_original_image = tf.expand_dims(original_image, 0)
    concatenated_original_images = tf.tile(expanded_original_image, [num_bboxes, 1, 1, 1])

    names = ['original_inputs', 'inputs', 'ids', 'labels']
    tensors = [concatenated_original_images, distorted_inputs, ids, labels]

    if fetch_text_labels:
        names.append('text_labels')
        tensors.append(text_labels)

    return [names, tensors]

def create_classification_batch(serialized_example, cfg, add_summaries, read_filenames=False):

    features = get_region_data(serialized_example, cfg, fetch_ids=True,
                               fetch_labels=False, fetch_text_labels=False, read_filename=read_filenames)

    original_image = features['image']
    bboxes = features['bboxes']
    ids = features['ids']

    distorted_inputs = get_distorted_inputs(original_image, bboxes, cfg, add_summaries)

    distorted_inputs = tf.subtract(distorted_inputs, 0.5)
    distorted_inputs = tf.multiply(distorted_inputs, 2.0)

    names = ('inputs', 'ids')
    tensors = [distorted_inputs, ids]
    return [names, tensors]

def input_nodes(tfrecords, cfg, num_epochs=None, batch_size=32, num_threads=2,
                shuffle_batch = True, random_seed=1, capacity = 1000, min_after_dequeue = 96,
                add_summaries=True, input_type='train', fetch_text_labels=False,
                read_filenames=False):
    """
    Args:
        tfrecords:
        cfg:
        num_epochs: number of times to read the tfrecords
        batch_size:
        num_threads:
        shuffle_batch:
        capacity:
        min_after_dequeue:
        add_summaries: Add tensorboard summaries of the images
        input_type: 'train', 'visualize', 'test', 'classification'
    """
    with tf.name_scope('inputs'):

        # A producer to generate tfrecord file paths
        filename_queue = tf.train.string_input_producer(
          tfrecords,
          num_epochs=num_epochs
        )

        # Construct a Reader to read examples from the tfrecords file
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        if input_type=='train' or input_type=='test':
            batch_keys, data_to_batch = create_training_batch(serialized_example, cfg, add_summaries, read_filenames)
        elif input_type=='visualize':
            batch_keys, data_to_batch = create_visualization_batch(serialized_example, cfg, add_summaries, fetch_text_labels, read_filenames)
        elif input_type=='classification':
            batch_keys, data_to_batch = create_classification_batch(serialized_example, cfg, add_summaries, read_filenames)
        else:
            raise ValueError("Unknown input type: %s. Options are `train`, `test`, " \
                             "`visualize`, and `classification`." % (input_type,))

        if shuffle_batch:
            batch = tf.train.shuffle_batch(
                data_to_batch,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity= capacity,
                min_after_dequeue= min_after_dequeue,
                seed = random_seed,
                enqueue_many=True
            )

        else:
            batch = tf.train.batch(
                data_to_batch,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity= capacity,
                enqueue_many=True
            )

        batch_dict = {k : v for k, v in zip(batch_keys, batch)}

        return batch_dict