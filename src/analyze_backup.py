""" Post-training analysis """
import argparse
import json
import tensorflow as tf
from tensorflow import keras


def load_dataset(tfrecord_dir, batch_size=1, shuffle=False, buffer_size=10000):
    """Load and preprocess dataset."""
    def _parse_tfexample(example_proto):
        """Parse a single TFExample."""
        feature_description = {
            "ink": tf.io.FixedLenFeature([200, 3], tf.float32),
            "class_index": tf.io.FixedLenFeature([], tf.int64),
            "shape": tf.io.FixedLenFeature([2], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(
            example_proto, feature_description)
        ink = parsed_features["ink"]
        class_index = parsed_features["class_index"]
        return ink, class_index

    # Use Dataset.list_files to read all TFRecord files
    files = tf.data.Dataset.list_files(f"{tfrecord_dir}/*.tfrecord*-of-*")
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=10)

    dataset = dataset.map(_parse_tfexample)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([200, 3], []))
    return dataset


def main(args):
    model = keras.models.load_model(args.model_path)

    eval_dataset = load_dataset(args.tfrecord_dir, args.batch_size)

    # Load the class map from JSON file
    with open(args.class_map_path, 'r') as f:
        class_map = json.load(f)

    # Perform predictions on the evaluation dataset
    predictions = model.predict(eval_dataset)

    # Get the original true labels
    true_labels = []
    for _, class_index in eval_dataset:
        true_labels.extend(class_index.numpy())

    # Convert predicted numbers to labels using the class map
    predicted_labels = [class_map[str(prediction.argmax())]
                        for prediction in predictions]

    # Create a dictionary to store mismatch counts for each combination
    mismatch_counts = {}
    mismatch_counts["Total"] = 0
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_label_name = class_map[str(true_label)]
        if true_label_name != predicted_label:
            key = f"True: {true_label_name}, Predicted: {predicted_label}"
            mismatch_counts[key] = mismatch_counts.get(key, 0) + 1
            mismatch_counts["Total"] += 1

    # Print the mismatch counts for each combination
    for combination, count in mismatch_counts.items():
        print(f"Mismatch Combination: {combination}, Mismatch Count: {count}")

    accuracy = (1 - mismatch_counts["Total"] / len(true_labels)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total mismatch counts: {
          mismatch_counts['Total']}/{len(true_labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Load a model and perform predictions with TFRecord files.')
    parser.add_argument('--model_path', type=str, default="./saved_results/best_model.keras",
                        help='Path to the saved Keras model file')
    parser.add_argument('--tfrecord_dir', type=str,
                        default="./data/final512data", help='Path to the evaluation TFRecord file')
    parser.add_argument('--class_map_path', type=str, default="./saved_results/class_map.json",
                        help='Path to the class map JSON file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for prediction')

    args = parser.parse_args()
    main(args)
