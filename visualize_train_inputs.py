from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from config.parse_config import parse_config_file
from preprocessing.inputs import input_nodes

def visualize_train_inputs(tfrecords, cfg, show_text_labels=False):

    graph = tf.Graph()
    sess = tf.Session(graph = graph)

    # run a session to look at the images...
    with sess.as_default(), graph.as_default():

        # Input Nodes
        with tf.device('/cpu:0'):
            batch_dict = input_nodes(
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
                input_type='visualize',
                fetch_text_labels=show_text_labels
            )

        # Convert float images to uint8 images
        image_to_convert = tf.placeholder(dtype=tf.float32,
                                          shape=[cfg.IMAGE_PROCESSING.INPUT_SIZE,
                                                 cfg.IMAGE_PROCESSING.INPUT_SIZE, 3])
        uint8_image = tf.image.convert_image_dtype(image_to_convert, dtype=tf.uint8)


        coord = tf.train.Coordinator()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        plt.ion()
        done = False
        while not done:

            output = sess.run(batch_dict)

            original_images = output['original_inputs']
            distorted_images = output['inputs']
            image_ids = output['ids']
            labels = output['labels']
            if show_text_labels:
                text_labels = output['text_labels']

            for b in range(cfg.BATCH_SIZE):

                original_image = original_images[b]
                distorted_image = distorted_images[b]

                if original_image.dtype != np.uint8:
                    original_image = sess.run(uint8_image, {image_to_convert : original_image})

                if distorted_image.dtype != np.uint8:
                    distorted_image = sess.run(uint8_image, {image_to_convert : distorted_image})

                image_id = image_ids[b]
                label = labels[b]

                fig = plt.figure('Train Inputs')

                if show_text_labels:
                    text_label = text_labels[b]
                    st = fig.suptitle("Image: %s\nLabel: %d\nText: %s" %
                                      (image_id, label, text_label), fontsize=12)
                else:
                    st = fig.suptitle("Image: %s\nLabel: %d" % (image_id, label), fontsize=12)

                plt.subplot(2, 1, 1)
                plt.imshow(original_image)
                plt.title("Original")
                plt.axis('off')

                plt.subplot(2, 1, 2)
                plt.imshow(distorted_image)
                plt.title("Modified")
                plt.axis('off')

                # Shift the subplots down a bit to make room for the super title
                st.set_y(0.95)
                fig.subplots_adjust(top=0.75)

                plt.show(block=False)

                t = raw_input("Press Enter to view next image. Press any key followed " \
                              "by Enter to quite: ")
                if t != '':
                    done = True
                    break
                plt.clf()


def parse_args():

    parser = argparse.ArgumentParser(description='Visualize the inputs to train the classification system.')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='Paths to tfrecord files.', type=str,
                        nargs='+', required=True)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--text_labels', dest='show_text_labels',
                        help='If text labels have been stored in the tfrecords, then you can use this flag to show them.',
                        action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
  args = parse_args()
  cfg = parse_config_file(args.config_file)
  visualize_train_inputs(
    tfrecords=args.tfrecords,
    cfg=cfg,
    show_text_labels=args.show_text_labels
  )



if __name__ == '__main__':
  main()
