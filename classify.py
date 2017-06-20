from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config.parse_config import parse_config_file
from nets import nets_factory
from preprocessing import inputs

def classify(tfrecords, checkpoint_path, save_path, max_iterations, save_logits, cfg):
    """
    Args:
        tfrecords (list)
        checkpoint_path (str)
        save_dir (str)
        max_iterations (int)
        save_logits (bool)
        cfg (EasyDict)
    """
    tf.logging.set_verbosity(tf.logging.DEBUG)

    graph = tf.Graph()

    with graph.as_default():

        global_step = slim.get_or_create_global_step()

        with tf.device('/cpu:0'):
            batch_dict = inputs.input_nodes(
                tfrecords=tfrecords,
                cfg=cfg.IMAGE_PROCESSING,
                num_epochs=1,
                batch_size=cfg.BATCH_SIZE,
                num_threads=cfg.NUM_INPUT_THREADS,
                shuffle_batch =cfg.SHUFFLE_QUEUE,
                random_seed=cfg.RANDOM_SEED,
                capacity=cfg.QUEUE_CAPACITY,
                min_after_dequeue=cfg.QUEUE_MIN,
                add_summaries=False,
                input_type='classification'
            )

        arg_scope = nets_factory.arg_scopes_map[cfg.MODEL_NAME]()

        with slim.arg_scope(arg_scope):
            logits, end_points = nets_factory.networks_map[cfg.MODEL_NAME](
                inputs=batch_dict['inputs'],
                num_classes=cfg.NUM_CLASSES,
                is_training=False
            )

            predicted_labels = tf.argmax(end_points['Predictions'], 1)

        if 'MOVING_AVERAGE_DECAY' in cfg and cfg.MOVING_AVERAGE_DECAY > 0:
            variable_averages = tf.train.ExponentialMovingAverage(
                cfg.MOVING_AVERAGE_DECAY, global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[global_step.op.name] = global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()
            variables_to_restore.append(global_step)

        saver = tf.train.Saver(variables_to_restore, reshape=True)

        num_batches = max_iterations
        num_images = num_batches * cfg.BATCH_SIZE
        label_array = np.empty(num_images, dtype=np.int32)
        id_array = np.empty(num_images, dtype=np.object)
        fetches = [predicted_labels, batch_dict['ids']]
        if save_logits:
            fetches.append(logits)
            logits_array = np.empty((num_images, cfg.NUM_CLASSES), dtype=np.float32)

        if os.path.isdir(checkpoint_path):
            checkpoint_dir = checkpoint_path
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

            if checkpoint_path is None:
                raise ValueError("Unable to find a model checkpoint in the " \
                                 "directory %s" % (checkpoint_dir,))

        tf.logging.info('Classifying records using %s' % checkpoint_path)

        coord = tf.train.Coordinator()

        sess_config = tf.ConfigProto(
                log_device_placement=cfg.SESSION_CONFIG.LOG_DEVICE_PLACEMENT,
                allow_soft_placement = True,
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
                )
            )
        sess = tf.Session(graph=graph, config=sess_config)

        with sess.as_default():

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:

                # Restore from checkpoint
                saver.restore(sess, checkpoint_path)

                print_str = ', '.join([
                  'Step: %d',
                  'Time/image (ms): %.1f'
                ])

                step = 0
                while not coord.should_stop():

                    t = time.time()
                    outputs = sess.run(fetches)
                    dt = time.time()-t

                    idx1 = cfg.BATCH_SIZE * step
                    idx2 = idx1 + cfg.BATCH_SIZE
                    label_array[idx1:idx2] = outputs[0]
                    id_array[idx1:idx2] = outputs[1]
                    if save_logits:
                        logits_array[idx1:idx2] = outputs[2]

                    step += 1
                    print(print_str % (step, (dt / cfg.BATCH_SIZE) * 1000))

                    if max_iterations > 0 and step == max_iterations:
                        break

            except tf.errors.OutOfRangeError as e:
                pass

        coord.request_stop()
        coord.join(threads)

        # save the results
        if save_logits:
            np.savez(save_path, labels=label_array, ids=id_array, logits=logits_array)
        else:
            np.savez(save_path, labels=label_array, ids=id_array)


def parse_args():

    parser = argparse.ArgumentParser(description='Classify images, optionally saving the logits.')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='Paths to tfrecords.', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to a specific model to test against. If a directory, then the newest checkpoint file will be used.', type=str,
                          required=True, default=None)

    parser.add_argument('--save_path', dest='save_path',
                          help='File name path to a save the classification results.', type=str,
                          required=True, default=None)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--batch_size', dest='batch_size',
                        help='The number of images in a batch.',
                        required=True, type=int, default=None)

    parser.add_argument('--batches', dest='batches',
                        help='Maximum number of iterations to run. Default is all records (modulo the batch size).',
                        required=True, type=int, default=0)

    parser.add_argument('--save_logits', dest='save_logits',
                        help='Should the logits be saved?',
                        action='store_true', default=False)

    parser.add_argument('--model_name', dest='model_name',
                        help='The name of the architecture to use.',
                        required=False, type=str, default=None)



    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = parse_config_file(args.config_file)

    if args.batch_size != None:
        cfg.BATCH_SIZE = args.batch_size

    if args.model_name != None:
        cfg.MODEL_NAME = args.model_name

    classify(
        tfrecords=args.tfrecords,
        checkpoint_path=args.checkpoint_path,
        save_path = args.save_path,
        max_iterations=args.batches,
        save_logits=args.save_logits,
        cfg=cfg
    )

if __name__ == '__main__':
    main()
