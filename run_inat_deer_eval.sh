#!/bin/bash

EXPERIMENT_DIR="/home/ubuntu/efs/sim_classification/unity/1_4M_all"

EXP="inat_deer"

DATASET_DIR="/home/ubuntu/efs/tfrecords/inat_deer_tfrecords"

python classify.py --tfrecords $DATASET_DIR/* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.npz" --config $EXPERIMENT_DIR/config_test.yaml --batches 3000 --batch_size 1 --save_logits

python extract_activations.py --tfrecords $DATASET_DIR/*-00001-* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batch_size 1 --batches 1000 --batch_size 1 --save_logits




