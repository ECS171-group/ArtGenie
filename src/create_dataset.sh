#!/bin/bash
# makedataset.sh 
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
python create_dataset.py --recognized_only --ndjson_path data/download --output_path data/final512data --classes_file data/subset2.txt --train_observations_per_class 512 --eval_observations_per_class 128
