import argparse
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

    eval_dataset = load_dataset(args.eval_tfrecord, args.batch_size)

    # Perform predictions on the evaluation dataset
    predictions = model.predict(eval_dataset)

    # Print or process the predictions as needed
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a model and perform predictions with TFRecord files.')
    parser.add_argument('--model_path', type=str, help='Path to the saved Keras model file')
    parser.add_argument('--eval_tfrecord', type=str, help='Path to the evaluation TFRecord file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')

    args = parser.parse_args()
    main(args)

