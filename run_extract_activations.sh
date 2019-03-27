#!/bin/bash

EXPERIMENT_DIR="/home/ubuntu/efs/sim_classification/unity/1_4M_all"

EXP="cis_test_deer"

DATASET_DIR="/home/ubuntu/efs/tfrecords/eccv_cct_tfrecords/cis_test_deer/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batch_size 1 --batches 1000 --batch_size 1 --save_logits

EXP="imerit_deer"

DATASET_DIR="/home/ubuntu/efs/tfrecords/eccv_cct_tfrecords/imerit/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="train"

DATASET_DIR="/home/ubuntu/efs/tfrecords/cct_new/multiclass/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP*0000[1-5]-* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="trans_val"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="cis_val"

DATASET_DIR="/home/ubuntu/efs/tfrecords/cct_new/multiclass/"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="trans_test"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP*0000[1-5]-* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="cis_test"

python extract_activations.py --tfrecords $DATASET_DIR/$EXP*0000[1-5]-* --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD0_day"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerMixedDay_5_500_min_pix/"*-00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml  --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD0_night"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerMixedNight_6_500_min_pix/"*-00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD1_day"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerDay_SD_MOD1_500_min_pix/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD1_night"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerNight_SD_MOD1/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD2_day"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerDay_SD_MOD2_500_min_pix/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD2_night"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerNight_SD_MOD2/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD3_day"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerDay_SD_MOD3_500_min_pix/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD3_night"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerNight_SD_MOD3/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD4_day"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerDay_SD_MOD4_500_min_pix/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits

EXP="SD_MOD4_night"
DATASET_DIR="/home/ubuntu/efs/tfrecords/unity_tfrecords/DeerNight_SD_MOD4/"*00001-*

python extract_activations.py --tfrecords $DATASET_DIR --checkpoint_path $EXPERIMENT_DIR/logdir --save_path $EXPERIMENT_DIR/logdir/results/$EXP"_results.p" --config $EXPERIMENT_DIR/config_test.yaml --batches 1000 --batch_size 1 --save_logits


