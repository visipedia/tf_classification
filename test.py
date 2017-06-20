from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config.parse_config import parse_config_file
from nets import nets_factory
from preprocessing import inputs

def test(tfrecords, checkpoint_path, save_dir, max_iterations, eval_interval_secs, cfg):
    """
    Args:
        tfrecords (list)
        checkpoint_path (str)
        savedir (str)
        max_iterations (int)
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
                input_type='test'
            )

            batched_one_hot_labels = slim.one_hot_encoding(batch_dict['labels'],
                                                        num_classes=cfg.NUM_CLASSES)

        arg_scope = nets_factory.arg_scopes_map[cfg.MODEL_NAME]()

        with slim.arg_scope(arg_scope):
            logits, end_points = nets_factory.networks_map[cfg.MODEL_NAME](
                inputs=batch_dict['inputs'],
                num_classes=cfg.NUM_CLASSES,
                is_training=False
            )

            predictions = end_points['Predictions']
            #labels = tf.squeeze(batch_dict['labels'])
            labels = batch_dict['labels']

            # Add the loss summary
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=batched_one_hot_labels, label_smoothing=0., weights=1.0)

        if 'MOVING_AVERAGE_DECAY' in cfg and cfg.MOVING_AVERAGE_DECAY > 0:
            variable_averages = tf.train.ExponentialMovingAverage(
                cfg.MOVING_AVERAGE_DECAY, global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[global_step.op.name] = global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()
            variables_to_restore.append(global_step)
        

        # Define the metrics:
        metric_map = {
            'Accuracy': slim.metrics.streaming_accuracy(labels=labels, predictions=tf.argmax(predictions, 1)),
            loss.op.name : slim.metrics.streaming_mean(loss)
        }
        if len(cfg.ACCURACY_AT_K_METRIC) > 0:
            bool_labels = tf.ones([cfg.BATCH_SIZE], dtype=tf.bool)
            for k in cfg.ACCURACY_AT_K_METRIC:
                if k <= 1 or k > cfg.NUM_CLASSES:
                    continue
                in_top_k = tf.nn.in_top_k(predictions=predictions, targets=labels, k=k)
                metric_map['Accuracy_at_%s' % k] = slim.metrics.streaming_accuracy(labels=bool_labels, predictions=in_top_k)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric_map)

        # Print the summaries to screen.
        print_global_step = True
        for name, value in names_to_values.iteritems():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            if print_global_step:
                op=tf.Print(op, [global_step], "Model Step ")
                print_global_step = False
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op) 

        if max_iterations > 0:
            num_batches = max_iterations
        else:
            # This ensures that we make a single pass over all of the data.
            # We could use ceil if the batch queue is allowed to pad the last batch
            num_batches = np.floor(cfg.NUM_TEST_EXAMPLES / float(cfg.BATCH_SIZE))


        sess_config = tf.ConfigProto(
            log_device_placement=cfg.SESSION_CONFIG.LOG_DEVICE_PLACEMENT,
            allow_soft_placement = True,
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
            )
        )

        if eval_interval_secs > 0:

            if not os.path.isdir(checkpoint_path):
                raise ValueError("checkpoint_path should be a path to a directory when " \
                                 "evaluating in a loop.")

            slim.evaluation.evaluation_loop(
                master='',
                checkpoint_dir=checkpoint_path,
                logdir=save_dir,
                num_evals=num_batches,
                initial_op=None,
                initial_op_feed_dict=None,
                eval_op=names_to_updates.values(),
                eval_op_feed_dict=None,
                final_op=None,
                final_op_feed_dict=None,
                summary_op=tf.summary.merge_all(),
                summary_op_feed_dict=None,
                variables_to_restore=variables_to_restore,
                eval_interval_secs=eval_interval_secs,
                max_number_of_evaluations=None,
                session_config=sess_config,
                timeout=None
            )

        else:
            if os.path.isdir(checkpoint_path):
                checkpoint_dir = checkpoint_path
                checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

                if checkpoint_path is None:
                    raise ValueError("Unable to find a model checkpoint in the " \
                                     "directory %s" % (checkpoint_dir,))

            tf.logging.info('Evaluating %s' % checkpoint_path)

            slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=checkpoint_path,
                logdir=save_dir,
                num_evals=num_batches,
                eval_op=names_to_updates.values(),
                variables_to_restore=variables_to_restore,
                session_config=sess_config
            )

def parse_args():

    parser = argparse.ArgumentParser(description='Test the person classifier')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='Paths to tfrecords.', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to a specific model to test against. If a directory, then the newest checkpoint file will be used.', type=str,
                          required=True, default=None)

    parser.add_argument('--save_dir', dest='savedir',
                          help='Path to directory to store summary files.', type=str,
                          required=True)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file.',
                        required=True, type=str)

    parser.add_argument('--eval_interval_secs', dest='eval_interval_secs',
                        help='Go into an evaluation loop, waiting this many seconds between evaluations. Default is to evaluate once.',
                        required=False, type=int, default=0)

    parser.add_argument('--batch_size', dest='batch_size',
                        help='The number of images in a batch.',
                        required=False, type=int, default=None)

    parser.add_argument('--batches', dest='batches',
                        help='Maximum number of iterations to run. Default is all records (modulo the batch size).',
                        required=False, type=int, default=0)

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

    test(
        tfrecords=args.tfrecords,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.savedir,
        max_iterations=args.batches,
        eval_interval_secs=args.eval_interval_secs,
        cfg=cfg
    )

if __name__ == '__main__':
    main()
