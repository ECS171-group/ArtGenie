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

train_mlp_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 10 --batch_size 10 --learning_rate 0.001 --model_type mlp'
# more epoch
train_mlp_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 20 --batch_size 10 --learning_rate 0.001 --model_type mlp'

train_cnn_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 10 --batch_size 10 --learning_rate 0.001 --model_type cnn'
# more epoch
train_cnn_1='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 20 --batch_size 10 --learning_rate 0.001 --model_type cnn'

# Best model by grid search
train_best='python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 20 --batch_size 64 --learning_rate 0.001 --model_type cnn'


train_cmd=$train_best
echo $train_cmd
read -r -p 'Run command? ' yn
if [[ "$yn" != 'y' ]]; then
    exit 1
fi
eval "$train_cmd"
