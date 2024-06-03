#!/bin/bash
# train.sh 
#
while getopts "" opt; do
    case $opt in
	\?)
	    echo "Invalid option: -$OPTARG" >&2
	    ;;
    esac
done
if [[ "$#" -ne 0 ]]; then
    echo "Illegal number of parameters" >&2
    exit 1
fi

train_mlp_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 10 --batch_size 10 --learning_rate 0.001'
# more epoch
train_mlp_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 20 --batch_size 10 --learning_rate 0.001'

# Note that the model needs to be changed inside the train_model.py
# file. This only changes the hyperparameters
train_rnn_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 10 --batch_size 10 --learning_rate 0.001'
# Higher learning rate
train_rnn_2='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 10 --batch_size 10 --learning_rate 0.01'

train_cnn_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 10 --batch_size 10 --learning_rate 0.001'
# more epoch
train_cnn_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 20 --batch_size 10 --learning_rate 0.001'

train_cmd=$train_cnn_1
echo $train_cmd
read -r -p 'Run command? ' yn
if [[ "$yn" != 'y' ]]; then
    exit 1
fi
eval "$train_cmd"
