#!/bin/bash

EXPERIMENT_DIR="/home/ubuntu/efs/sim_classification/unity/model_2_100K"

EXP="eval_all"

DATASET_DIR="/home/ubuntu/efs/tfrecords/cct_new/eval_all"

python classify.py --tfrecords $DATASET_DIR/* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.npz" --config $EXPERIMENT_DIR/config_test.yaml --batches 80000 --batch_size 1 --save_logits


