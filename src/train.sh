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

echo 'python train_model.py --trainingdata_dir data/rnn_subset --eval_data_dir data/rnn_subset --output_dir data/rnn_subset_output --job-dir data/quickdraw_model_subset3 --num_epochs 10 --batch_size 32 --learning_rate 0.001'
read -r -p 'Run command? ' yn
if [[ "$yn" != 'y' ]]; then
    exit 1
fi
python train_model.py --trainingdata_dir data/final512data --eval_data_dir data/final512data --output_dir data/final512output --job-dir data/final512job --num_epochs 10 --batch_size 10 --learning_rate 0.001
