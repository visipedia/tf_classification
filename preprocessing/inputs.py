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
    return cropped_image, distort_bbox

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


def apply_distortions(image, cfg, add_summaries=True):

    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    # Convert the pixel values to be in the range [0,1]
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Add a summary of the original data
    if add_summaries:
      tf.summary.image('original_images', tf.expand_dims(image, 0))

    # Extract a distorted bbox
    r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    do_crop = tf.less(r, cfg.WHOLE_IMAGE_CFG.DO_RANDOM_CROP)
    rc_cfg = cfg.WHOLE_IMAGE_CFG.RANDOM_CROP_CFG
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    distorted_image, distorted_bbox = tf.cond(do_crop,
            lambda: distorted_bounding_box_crop(image, bbox,
                                                   aspect_ratio_range=(rc_cfg.MIN_ASPECT_RATIO, rc_cfg.MAX_ASPECT_RATIO),
                                                   area_range=(rc_cfg.MIN_AREA, rc_cfg.MAX_AREA),
                                                   max_attempts=rc_cfg.MAX_ATTEMPTS),
            lambda: tf.tuple([image, bbox])
        )

    distorted_image.set_shape([None, None, 3])

    # Add a summary
    if add_summaries:
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox)
        tf.summary.image('images_with_random_crop',
                         image_with_distorted_box)

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
        tf.summary.image('cropped_resized_images',
                     tf.expand_dims(distorted_image, 0))

    # Randomly flip the image:
    if cfg.DO_RANDOM_FLIP_LEFT_RIGHT:
      r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
      do_flip = tf.less(r, 0.5)
      distorted_image = tf.cond(do_flip, lambda: tf.image.flip_left_right(distorted_image), lambda: tf.identity(distorted_image))

    # TODO: Can this be changed so that we don't always distort the colors?
    # Distort the colors
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
      tf.summary.image('final_distorted_images', tf.expand_dims(distorted_image, 0))

    return distorted_image

def create_training_batch(serialized_example, cfg, add_summaries):
    features = decode_serialized_example(serialized_example,
                                            [('image/encoded', 'image'),
                                             ('image/class/label', 'label')])


    image = features['image']
    label = features['label']

    distorted_image = apply_distortions(image, cfg, add_summaries=add_summaries)

    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)

    return [('inputs', 'labels'), [distorted_image, label]]

def create_visualization_batch(serialized_example, cfg, add_summaries):

    features = decode_serialized_example(serialized_example,
                                            [('image/id', 'image_id'),
                                             ('image/encoded', 'image'),
                                             ('image/class/label', 'label'),
                                             ('image/class/text', 'text_label')])


    image = features['image']
    image_id = features['image_id']
    label = features['label']
    text_label = features['text_label']

    original_image = tf.identity(image)
    distorted_image = apply_distortions(image, cfg, add_summaries=add_summaries)

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

    # Bring the distorted image back to ints
    if distorted_image.dtype == tf.float32:
      distorted_image = tf.image.convert_image_dtype(distorted_image, dtype=tf.uint8)

    return [('original_inputs', 'inputs', 'image_ids', 'labels', 'text_labels'), [original_image, distorted_image, image_id, label, text_label]]

def input_nodes(tfrecords, cfg, num_epochs=None, batch_size=32, num_threads=2,
                shuffle_batch = True, random_seed=1, capacity = 1000, min_after_dequeue = 96,
                add_summaries=True, visualize=False):
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
        visualize:
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

        if visualize:
            batch_keys, data_to_batch = create_visualization_batch(serialized_example, cfg, add_summaries)
        else:
            batch_keys, data_to_batch = create_training_batch(serialized_example, cfg, add_summaries)

        if shuffle_batch:
            batch = tf.train.shuffle_batch(
                data_to_batch,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity= capacity,
                min_after_dequeue= min_after_dequeue,
                seed = random_seed,
                enqueue_many=False
            )

        else:
            batch = tf.train.batch(
                data_to_batch,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity= capacity,
                enqueue_many=False
            )

        batch_dict = {k : v for k, v in zip(batch_keys, batch)}

        return batch_dict