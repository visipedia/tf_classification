#!/bin/bash

EXPERIMENT_DIR="/home/ubuntu/efs/sim_classification/unity/unity_all_models_625K_day"

EXP="cis_test_deer"

DATASET_DIR="/home/ubuntu/efs/tfrecords/eccv_cct_tfrecords/cis_test_deer/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batch_size 1 --batches 1000 --batch_size 1 --save_logits

EXP="imerit_deer"

DATASET_DIR="/home/ubuntu/efs/tfrecords/eccv_cct_tfrecords/imerit/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="train"

DATASET_DIR="/home/ubuntu/efs/tfrecords/cct_new/multiclass/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="trans_val"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="cis_val"

DATASET_DIR="/home/ubuntu/efs/tfrecords/cct_new/multiclass/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="trans_test"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="cis_test"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="unity_deer"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/eval_day"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml  --batches 1000 --batch_size 1 --save_logits

EXP="unity_night"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/eval_night"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits


