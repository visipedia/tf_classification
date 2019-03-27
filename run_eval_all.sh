#!/bin/bash

EXPERIMENT_DIR="/home/ubuntu/efs/sim_classification/general/resnet_101_v2_train_on_cct"

EXP="eval_all"

DATASET_DIR="/home/ubuntu/efs/tfrecords/cct_new/eval_all"

python classify.py --tfrecords $DATASET_DIR/* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.npz" --config $EXPERIMENT_DIR/config_test.yaml --batches 80000 --batch_size 1 --save_logits

EXP="inat_deer"

DATASET_DIR="/home/ubuntu/efs/tfrecords/inat_deer_tfrecords"

python classify.py --tfrecords $DATASET_DIR/* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.npz" --config $EXPERIMENT_DIR/config_test.yaml --batches 3000 --batch_size 1 --save_logits



