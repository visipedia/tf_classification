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

def export(checkpoint_path, export_dir, export_version, export_for_serving, do_preprocess, cfg):

    graph = tf.Graph()

    input_node_name = "images"
    output_node_name = None
    jpegs = None

    with graph.as_default():

        global_step = slim.get_or_create_global_step()

        input_height = cfg.IMAGE_PROCESSING.INPUT_SIZE
        input_width = cfg.IMAGE_PROCESSING.INPUT_SIZE
        input_depth = 3
        # We want to store the preprocessing operation in the graph
        if do_preprocess:

            def preprocess_image(image_buffer):
              """Preprocess JPEG encoded bytes to 3D float Tensor."""

              # Decode the string as an RGB JPEG.
              image = tf.image.decode_jpeg(image_buffer, channels=3)
              image = tf.image.convert_image_dtype(image, dtype=tf.float32)
              # Resize the image to the original height and width.
              image = tf.expand_dims(image, 0)
              image = tf.image.resize_bilinear(image,
                                               [input_height, input_width],
                                               align_corners=False)
              image = tf.squeeze(image, [0])
              # Finally, rescale to [-1,1] instead of [0, 1)
              image = tf.subtract(image, 0.5)
              image = tf.multiply(image, 2.0)
              return image

            input_placeholder = tf.placeholder(tf.string, name=input_node_name)
            feature_configs = {
                'image/encoded': tf.FixedLenFeature(
                    shape=[], dtype=tf.string),
            }
            tf_example = tf.parse_example(input_placeholder, feature_configs)

            jpegs = tf_example['image/encoded']
            encoded_jpeg_node_name = jpegs.name[:-2]
            images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

        # We assume the client has preprocessed the data for us
        else:
            input_placeholder = tf.placeholder(tf.float32, [None, input_height * input_width * input_depth], name=input_node_name)
            images = tf.reshape(input_placeholder, [-1, input_height, input_width, input_depth])

        arg_scope = nets_factory.arg_scopes_map[cfg.MODEL_NAME]()

        with slim.arg_scope(arg_scope):
            logits, end_points = nets_factory.networks_map[cfg.MODEL_NAME](
                inputs=images,
                num_classes=cfg.NUM_CLASSES,
                is_training=False
            )
        
        # GVH: I would like to use tf.identity here, but the function tensorflow.python.framework.graph_util.remove_training_nodes 
        # called in (optimize_for_inference_lib.optimize_for_inference) removes the identity function.
        # Sticking with an add 0 operation for now. 
        output_node = tf.add(end_points['Predictions'], 0., name='Predictions')
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
            
            classification_input_node = input_placeholder
            if do_preprocess:
                prediction_input_node = jpegs
            else:
                prediction_input_node = classification_input_node

            class_scores, predicted_classes = tf.nn.top_k(end_points['Predictions'], k=cfg.NUM_CLASSES)

            with tf.Session(graph=graph) as sess:

                tf.global_variables_initializer().run()

                saver.restore(sess, checkpoint_path)

                save_path = os.path.join(export_dir, "%d" % (export_version,))

                builder = saved_model_builder.SavedModelBuilder(save_path)

                # Build the signature_def_map.

                classify_inputs_tensor_info = utils.build_tensor_info(classification_input_node)
                classes_output_tensor_info = utils.build_tensor_info(predicted_classes)
                scores_output_tensor_info = utils.build_tensor_info(class_scores)

                classification_signature = signature_def_utils.build_signature_def(
                    inputs={
                        signature_constants.CLASSIFY_INPUTS: classify_inputs_tensor_info
                    },
                    outputs={
                        signature_constants.CLASSIFY_OUTPUT_CLASSES:
                            classes_output_tensor_info,
                        signature_constants.CLASSIFY_OUTPUT_SCORES:
                            scores_output_tensor_info
                    },
                    method_name=signature_constants.CLASSIFY_METHOD_NAME
                )
                
                predict_inputs_tensor_info = utils.build_tensor_info(prediction_input_node)

                prediction_signature = signature_def_utils.build_signature_def(
                    inputs={'images': predict_inputs_tensor_info},
                    outputs={
                        'classes': classes_output_tensor_info,
                        'scores': scores_output_tensor_info
                },
                method_name=signature_constants.PREDICT_METHOD_NAME)

                legacy_init_op = tf.group(
                  tf.tables_initializer(), name='legacy_init_op')

                builder.add_meta_graph_and_variables(
                    sess, [tag_constants.SERVING],
                    signature_def_map={
                        'predict_images':
                            prediction_signature,
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            classification_signature,
                    },
                    legacy_init_op=legacy_init_op
                )

                builder.save()

                print("Saved optimized model for TensorFlow Serving.")


        else:
            with sess.as_default():

                tf.global_variables_initializer().run()

                saver.restore(sess, checkpoint_path)

                input_graph_def = graph.as_graph_def()
                input_node_names= [input_node_name]
                output_node_names = [output_node_name]

                constant_graph_def = graph_util.convert_variables_to_constants(
                    sess=sess,
                    input_graph_def=input_graph_def,
                    output_node_names=output_node_names,
                    variable_names_whitelist=None,
                    variable_names_blacklist=None)

                if do_preprocess:
                    optimized_graph_def = constant_graph_def
                else:
                    optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
                        input_graph_def=constant_graph_def,
                        input_node_names=input_node_names,
                        output_node_names=output_node_names,
                        placeholder_type_enum=dtypes.float32.as_datatype_enum)
                
                if not os.path.exists(export_dir):
                    os.makedirs(export_dir)
                save_path = os.path.join(export_dir, 'optimized_model-%d.pb' % (export_version,))
                with open(save_path, 'w') as f:
                    f.write(optimized_graph_def.SerializeToString())

                print("Saved optimized model for mobile devices at: %s." % (save_path,))
                print("Input node name: %s" % (input_node_name,))
                print("Output node name: %s" % (output_node_name,))

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

    parser.add_argument('--do_preprocess', dest='do_preprocess',
                        help='Add the image decoding and preprocessing nodes to the graph.',
                        action='store_true', default=False)


    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    export(args.checkpoint_path, args.export_dir, args.export_version, args.serving, args.do_preprocess, cfg=cfg)