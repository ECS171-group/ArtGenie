import argparse
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf

def parse_line(ndjson_line):
    """Parse an ndjson line and return ink (as np array) and classname."""
    sample = json.loads(ndjson_line)
    class_name = sample["word"]
    recognized = sample.get("recognized", True)  # Get the "recognized" field, default to True if not present
    inkarray = sample["drawing"]
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    for stroke in inkarray:
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1  # stroke_end
    # Preprocessing.
    # 1. Size normalization.
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    # 2. Compute deltas.
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]
    return np_ink, class_name, recognized  # Return recognized along with np_ink and class_name

def convert_data(trainingdata_dir,
                 observations_per_class,
                 output_file,
                 classnames,
                 output_shards=10,
                 offset=0,
                 recognized_only=False):
    """Convert training data from ndjson files into tf.Example format."""
    writers = []
    for i in range(output_shards):
        writers.append(
            tf.io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i, output_shards)))

    train_files = []
    for class_name in classnames:
        class_files = tf.io.gfile.glob(os.path.join(trainingdata_dir,
                                                    f"full_simplified_{class_name}.ndjson"))
        train_files.extend(class_files)

    num_per_class = {}
    for class_name in classnames:
        num_per_class[class_name] = 0
    for filename in train_files:
        print(f"Processing file: {filename}")  # Debug print
        with tf.io.gfile.GFile(filename, "r") as f:
            for line in f:
                ink, class_name, recognized = parse_line(line)  # Get recognized value from parse_line
                if recognized_only and not recognized:  # Skip entries that are not correctly recognized if recognized_only is True
                    print(f"Skipping entry: {class_name} (Not recognized)")
                    continue
                if class_name not in classnames:
                    print(f"Skipping entry: {class_name} (Class not in classnames)")
                    continue
                if num_per_class[class_name] >= observations_per_class:
                    print(f"Skipping entry: {class_name} (Reached observations_per_class limit)")
                    continue
                num_per_class[class_name] += 1
                if num_per_class[class_name] < offset:
                    continue
                features = {}
                features["class_index"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[classnames.index(class_name)]))
                features["ink"] = tf.train.Feature(float_list=tf.train.FloatList(value=ink.flatten()))
                features["shape"] = tf.train.Feature(int64_list=tf.train.Int64List(value=ink.shape))
                f = tf.train.Features(feature=features)
                example = tf.train.Example(features=f)
                writers[random.randint(0, output_shards - 1)].write(example.SerializeToString())

    for writer in writers:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndjson_path", type=str, default="", help="Directory containing ndjson files.")
    parser.add_argument("--output_path", type=str, default="", help="Path to store output TFRecord files.")
    parser.add_argument("--train_observations_per_class", type=int, default=10000, help="Number of observations per class for training data.")
    parser.add_argument("--eval_observations_per_class", type=int, default=1000, help="Number of observations per class for evaluation data.")
    parser.add_argument("--classes_file", type=str, default="", help="File containing class names.")
    parser.add_argument("--recognized_only", action="store_true", help="Only include correctly recognized items.")

    args = parser.parse_args()

    assert args.ndjson_path, "Must provide --ndjson_path"
    assert args.output_path, "Must provide --output_path"
    assert args.classes_file, "Must provide --classes_file"

    with tf.io.gfile.GFile(args.classes_file, "r") as f:
        classnames = [x.strip() for x in f]

    print(f"Found {len(classnames)} classes: {classnames}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    convert_data(args.ndjson_path,
             args.train_observations_per_class,
             os.path.join(args.output_path, "training.tfrecord"),
             classnames,
             recognized_only=args.recognized_only)

    convert_data(args.ndjson_path,
             args.eval_observations_per_class,
             os.path.join(args.output_path, "eval.tfrecord"),
             classnames,
             offset=args.train_observations_per_class,
             recognized_only=args.recognized_only)


    with tf.io.gfile.GFile(os.path.join(args.output_path, "training.tfrecord.classes"), "w") as f:
        for class_name in classnames:
            f.write(class_name + "\n")

if __name__ == "__main__":
    main()
