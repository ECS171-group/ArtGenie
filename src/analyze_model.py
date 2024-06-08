""" Post-training analysis """
import argparse
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os


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


def get_metrics(y_true, y_pred):
    """Calculate Performance Statistics."""
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    recall = []
    specificity = []
    precision = []
    f1 = []

    for i in range(cm.shape[0]):  # Iterate through classes
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i]) - tp
        tn = np.sum(cm) - tp - fp - fn

        # Calculate metrics
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        f1.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))

    # Calculate Accuracy
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples

    return cm, recall, specificity, precision, accuracy, f1


def plot_metrics(confusion_matrix, classes, save_path=None,
                 img_name="confusion_matrix.png"):
    """Plot and log metrics to TensorBoard."""
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'), ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
    if save_path:
        print("Saving confusion matrix to", os.path.join(save_path,
                                                         img_name))
        plt.savefig(os.path.join(save_path, img_name))
    plt.show()


def main(args):
    model = keras.models.load_model(args.model_path)

    eval_dataset = load_dataset(args.tfrecord_dir, args.batch_size)

    # Load the class map from JSON file
    with open(args.class_map_path, 'r') as f:
        class_map = json.load(f)

    classes = list(class_map.values())

    y_true = []  # Initialize list for true labels
    y_pred = []  # Initialize list for predicted labels
    for x, y in eval_dataset:
        y_true.extend(y.numpy())  # Append true labels to the list
        # Append predicted labels to the list
        y_pred.extend(np.argmax(model.predict(x), axis=1))

    y_true = np.array(y_true)  # Convert list to numpy array
    y_pred = np.array(y_pred)  # Convert list to numpy array

    # Calculate and print metrics
    cm, recall, specificity, precision, model_accuracy, f1 = get_metrics(
        y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {model_accuracy}")
    print(f"F1-score: {f1}")
    plot_metrics(cm, classes, args.plt_dir)


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
    parser.add_argument('--plt_dir', type=str, default="./data/final512output",
                        help='Batch size for prediction')

    args = parser.parse_args()
    main(args)
