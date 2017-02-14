from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

from nets import nets_factory

def profile(model_name, num_classes, image_size, batch_size):

    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with graph.as_default(), sess.as_default():

        network_fn = nets_factory.get_network_fn(model_name, num_classes=num_classes)
        inputs = tf.random_uniform((batch_size, image_size, image_size, 3))
        logits, _ = network_fn(inputs)

        print("Profiling model %s" % model_name)

        # Print trainable variable parameter statistics to stdout.
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

        # param_stats is tensorflow.tfprof.TFProfNode proto. It organize the statistics
        # of each graph node in tree scructure. Let's print the root below.
        print('total_params: %d\n' % param_stats.total_parameters)

        print()

        # Print to stdout an analysis of the number of floating point operations in the
        # model broken down by individual operations.
        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)


def parse_args():

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_name', dest='model_name',
                        help='The name of the architecture to profile.', type=str,
                        required=False, default='inception_v3')

    parser.add_argument('--num_classes', dest='num_classes',
                        help='The number of classes.', type=int,
                        required=False, default=1000)

    parser.add_argument('--image_size', dest='image_size',
                          help='The size of the input image.', type=int,
                          required=False, default=299)

    parser.add_argument('--batch_size', dest='batch_size',
                        help='The number of images in a batch.', type=int,
                        required=False, default=1)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    profile(args.model_name, args.num_classes, args.image_size, args.batch_size)

if __name__ == '__main__':
    main()