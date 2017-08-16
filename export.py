"""
Export a trained model for application use.

Example for use with TensorFlow Serving:
python export.py \
--checkpoint_path model.ckpt-399739 \
--export_dir export \
--export_version 1 \
--config config_export.yaml \
--serving \
--add_preprocess \
--class_names class-codes.txt

Example for use with TensorFlow Mobile:
python export.py \
--checkpoint_path model.ckpt-399739 \
--export_dir export \
--export_version 1 \
--config config_export.yaml \
--class_names class-codes.txt

Author: Grant Van Horn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import optimize_for_inference_lib
slim = tf.contrib.slim

from config.parse_config import parse_config_file
from nets import nets_factory

def export(checkpoint_path, export_dir, export_version, export_for_serving, add_preprocess_step, class_names, cfg):
  """Export a model for use with TensorFlow Serving or for more conveinent use on mobile devices, etc.
  Arguments:
    checkpoint_path (str): Path to the specific model checkpoint file to export.
    export_dir (str): Path to a directory to store the export files.
    export_version (int): The version number of this export. If `export_for_serving` is True, then this version
      number must not exist in the `export_dir`.
    export_for_serving (bool): Export a model for use with TensorFlow Serving.
    add_preprocess_step (bool): If True, then an input path for handling image byte strings will be added to the graph.
    class_names (list): A list of semantic class identifiers to embed within the model that correspond to the prediction
      indices. Set to None to not embed.
    cfg (dict): Configuration dictionary.
  """

  if not os.path.exists(export_dir):
     print("Making export directory: %s" % (export_dir,))
     os.makedirs(export_dir)

  graph = tf.Graph()

  array_input_node_name = "images"
  bytes_input_node_name = "image_bytes"

  output_node_name = "Predictions"
  class_names_node_name = "names"

  with graph.as_default():

      global_step = slim.get_or_create_global_step()

      input_height = cfg.IMAGE_PROCESSING.INPUT_SIZE
      input_width = cfg.IMAGE_PROCESSING.INPUT_SIZE
      input_depth = 3

      # We want to store the preprocessing operation in the graph
      if add_preprocess_step:

        # The TensorFlow map_fn() function passes one argument only,
        # so I have put this method here to take advantage of scope
        # (to access input_height, etc.)
        def preprocess_image(image_buffer):
          """Preprocess image bytes to 3D float Tensor."""

          # Decode image bytes
          image = tf.image.decode_image(image_buffer)
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)

          # make sure the image is of rank 3
          image = tf.cond(
            tf.equal(tf.rank(image), 2),
            lambda: tf.expand_dims(image, 2),
            lambda: image
          )

          num_channels = tf.shape(image)[2]

          # if we decoded 1 channel (grayscale), then convert to a RGB image
          image = tf.cond(
            tf.equal(num_channels, 1),
            lambda: tf.image.grayscale_to_rgb(image),
            lambda: image
          )

          # if we decoded 2 channels (grayscale + alpha), then strip off the last dim and convert to rgb
          image = tf.cond(
            tf.equal(num_channels, 2),
            lambda: tf.image.grayscale_to_rgb(tf.expand_dims(image[:,:,0], 2)),
            lambda: image
          )

          # if we decoded 4 or more channels (rgb + alpha), then take the first three channels
          image = tf.cond(
            tf.greater(num_channels, 3),
            lambda : image[:,:,:3],
            lambda : image
          )

          # Resize the image to the input height and width for the network.
          image = tf.expand_dims(image, 0)
          image = tf.image.resize_bilinear(image,
                              [input_height, input_width],
                              align_corners=False)
          image = tf.squeeze(image, [0])
          # Finally, rescale to [-1,1] instead of [0, 1)
          image = tf.subtract(image, 0.5)
          image = tf.multiply(image, 2.0)
          return image

        image_bytes_placeholder = tf.placeholder(tf.string, name=bytes_input_node_name)
        preped_images = tf.map_fn(preprocess_image, image_bytes_placeholder, dtype=tf.float32)
        input_placeholder = tf.identity(preped_images, name=array_input_node_name) # Explicit name (we can't name the map_fn)

      # We assume the client has preprocessed the data for us
      else:
        input_placeholder = tf.placeholder(tf.float32, [None, input_height * input_width * input_depth], name=array_input_node_name)

      images = tf.reshape(input_placeholder, [-1, input_height, input_width, input_depth])

      arg_scope = nets_factory.arg_scopes_map[cfg.MODEL_NAME]()

      with slim.arg_scope(arg_scope):
        logits, end_points = nets_factory.networks_map[cfg.MODEL_NAME](
          inputs=images,
          num_classes=cfg.NUM_CLASSES,
          is_training=False
        )

      class_scores = end_points['Predictions']
      if class_names == None:
        class_names = tf.range(class_scores.get_shape().as_list()[1])
      predicted_classes = tf.tile(tf.expand_dims(class_names, 0), [tf.shape(class_scores)[0], 1], name=class_names_node_name)


      # GVH: I would like to use tf.identity here, but the function tensorflow.python.framework.graph_util.remove_training_nodes
      # called in (optimize_for_inference_lib.optimize_for_inference) removes the identity function.
      # Sticking with an add 0 operation for now.
      # We are doing this so that we can rename the output to `output_node_name` (i.e. something consistent)
      output_node = tf.add(end_points['Predictions'], 0., name=output_node_name)
      output_node_name = output_node.op.name

      if 'MOVING_AVERAGE_DECAY' in cfg and cfg.MOVING_AVERAGE_DECAY > 0:
        variable_averages = tf.train.ExponentialMovingAverage(
          cfg.MOVING_AVERAGE_DECAY, global_step)
        variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      else:
        variables_to_restore = slim.get_variables_to_restore()

      saver = tf.train.Saver(variables_to_restore, reshape=True)

      if os.path.isdir(checkpoint_path):
        checkpoint_dir = checkpoint_path
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

        if checkpoint_path is None:
          raise ValueError("Unable to find a model checkpoint in the " \
                           "directory %s" % (checkpoint_dir,))

      tf.logging.info('Exporting model: %s' % checkpoint_path)

      sess_config = tf.ConfigProto(
        log_device_placement=cfg.SESSION_CONFIG.LOG_DEVICE_PLACEMENT,
        allow_soft_placement = True,
        gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
        )
      )
      sess = tf.Session(graph=graph, config=sess_config)

      if export_for_serving:

        with tf.Session(graph=graph) as sess:

          tf.global_variables_initializer().run()

          saver.restore(sess, checkpoint_path)

          save_path = os.path.join(export_dir, "%d" % (export_version,))

          builder = saved_model_builder.SavedModelBuilder(save_path)

          # Build the signature_def_map.
          signature_def_map={}
          classes_output_tensor_info = utils.build_tensor_info(predicted_classes)
          scores_output_tensor_info = utils.build_tensor_info(class_scores)

          # image bytes input
          if add_preprocess_step:
            image_bytes_tensor_info = utils.build_tensor_info(image_bytes_placeholder)
            image_bytes_prediction_signature = signature_def_utils.build_signature_def(
              inputs={'images': image_bytes_tensor_info},
              outputs={
                'classes': classes_output_tensor_info,
                'scores': scores_output_tensor_info
              },
              method_name=signature_constants.PREDICT_METHOD_NAME
            )
            signature_def_map['predict_image_bytes']=image_bytes_prediction_signature

          # image array input
          image_array_tensor_info = utils.build_tensor_info(input_placeholder)
          image_array_prediction_signature = signature_def_utils.build_signature_def(
            inputs={'images': image_array_tensor_info},
            outputs={
              'classes': classes_output_tensor_info,
              'scores': scores_output_tensor_info
            },
            method_name=signature_constants.PREDICT_METHOD_NAME
          )
          signature_def_map['predict_image_array']=image_array_prediction_signature
          signature_def_map[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]=image_array_prediction_signature

          legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

          builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map=signature_def_map,
            legacy_init_op=legacy_init_op
          )

          builder.save()

          print("Saved optimized model for TensorFlow Serving.")


      else:
        with sess.as_default():

          tf.global_variables_initializer().run()

          saver.restore(sess, checkpoint_path)

          input_graph_def = graph.as_graph_def()
          input_node_names= [array_input_node_name]
          if add_preprocess_step:
            input_node_names.append(bytes_input_node_name)
          output_node_names = [output_node_name, class_names_node_name]

          constant_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names,
            variable_names_whitelist=None,
            variable_names_blacklist=None
          )

          if add_preprocess_step:
            optimized_graph_def = constant_graph_def
          else:
            optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
              input_graph_def=constant_graph_def,
              input_node_names=input_node_names,
              output_node_names=output_node_names,
              placeholder_type_enum=dtypes.float32.as_datatype_enum
            )

          save_dir = os.path.join(export_dir, str(export_version))
          if not os.path.exists(save_dir):
            print("Making version directory in export directory: %s" % (save_dir,))
            os.makedirs(save_dir)
          save_path = os.path.join(save_dir, 'optimized_model.pb')
          with open(save_path, 'w') as f:
            f.write(optimized_graph_def.SerializeToString())

          print("Saved optimized model for mobile devices at: %s." % (save_path,))
          print("Input node names: %s" % (input_node_names,))
          print("Output node name: %s" % (output_node_names,))

def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to the specific model you want to export.',
                          required=True, type=str)

    parser.add_argument('--export_dir', dest='export_dir',
                          help='Path to a directory where the exported model will be saved.',
                          required=True, type=str)

    parser.add_argument('--export_version', dest='export_version',
                        help='Version number of the model.',
                        required=True, type=int)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--serving', dest='serving',
                        help='Export for TensorFlow Serving usage. Otherwise, a constant graph will be generated.',
                        action='store_true', default=False)

    parser.add_argument('--add_preprocess', dest='add_preprocess',
                        help='Add the image decoding and preprocessing nodes to the graph so that image bytes can be passed in.',
                        action='store_true', default=False)

    parser.add_argument('--class_names', dest='class_names_path',
                          help='Path to the class names corresponding to each entry in the predictions output. This file should have one line for each index.',
                          required=False, type=str, default=None)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    if args.class_names_path != None:
      class_names = []
      with open(args.class_names_path) as f:
        for line in f:
          class_names.append(line.strip())
    else:
      class_names=None

    export(checkpoint_path=args.checkpoint_path,
      export_dir=args.export_dir,
      export_version=args.export_version,
      export_for_serving=args.serving,
      add_preprocess_step=args.add_preprocess,
      class_names=class_names,
      cfg=cfg
    )