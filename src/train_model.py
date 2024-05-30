import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def parse_line(ndjson_line):
    """Parse an ndjson line and return ink (as np array) and classname."""
    sample = json.loads(ndjson_line)
    class_name = sample["word"]
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
    return np_ink, class_name

def load_dataset(tfrecord_dir, batch_size, shuffle=True, buffer_size=10000):
    """Load and preprocess dataset."""
    def _parse_tfexample(example_proto):
        """Parse a single TFExample."""
        feature_description = {
            "ink": tf.io.VarLenFeature(tf.float32),
            "class_index": tf.io.FixedLenFeature([], tf.int64),
            "shape": tf.io.FixedLenFeature([2], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        ink = tf.sparse.to_dense(tf.sparse.reshape(parsed_features["ink"], parsed_features["shape"]))
        class_index = parsed_features["class_index"]
        return ink, class_index

    # Use Dataset.list_files to read all TFRecord files
    files = tf.data.Dataset.list_files(f"{tfrecord_dir}/*.tfrecord*-of-*")
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=10)

    dataset = dataset.map(_parse_tfexample)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 3], []))
    return dataset

def create_model(num_classes, learning_rate=0.001):
    """Create the drawing classification model."""
    # Simplified model by Eric
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes)
    ])

    # Previous model by Ling
    #model = tf.keras.Sequential([
    #    layers.Conv1D(32, 5, activation="relu", padding="same"),
    #    layers.Conv1D(64, 5, activation="relu", padding="same"),
    #    layers.Conv1D(128, 3, activation="relu", padding="same"),
    #    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    #    layers.Bidirectional(layers.LSTM(128)),
    #    layers.Dense(512, activation="relu"),
    #    layers.Dropout(0.5),
    #    layers.Dense(num_classes)
    #])

    #inputs = tf.keras.layers.Input(shape=(None, 3), dtype=tf.float32)
    #print("Input shape:", inputs.shape)  # Debug statement

    #reshape = tf.keras.layers.Reshape((-1, 1))(inputs)
    #print("Reshape shape:", reshape.shape)  # Debug statement

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model

def get_num_classes(classes_file):
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f]
    return len(classes)

def train_and_evaluate(args):
    """Train and evaluate the model."""

    print("Loading train dataset...")
    train_dataset = load_dataset(
        args.trainingdata_dir,
        args.batch_size
    )
    print("Loading eval dataset...")
    eval_dataset = load_dataset(
        args.eval_data_dir,
        args.batch_size,
        shuffle=False
    )
    print("Train dataset:")
    # Take the first 3 elements from the dataset
    subset = train_dataset.take(3)

    # Iterate over the subset and print the elements
    for element in subset:
        print(element)

    num_classes = get_num_classes(os.path.join(args.trainingdata_dir, 'training.tfrecord.classes'))

    # Create model
    print("Creating model...")
    model = create_model(num_classes, args.learning_rate)

    # Set up callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.job_dir, "checkpoints", "model-{epoch:02d}.weights.h5"),
        save_freq="epoch",
        save_weights_only=True,
        verbose=1
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.job_dir, "logs"))

    # Restore from the latest checkpoint if available
    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(args.job_dir, "checkpoints"))
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        model.load_weights(latest_checkpoint)
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Train the model
    print("Starting training...")
    model.fit(train_dataset,
              epochs=args.num_epochs,
              validation_data=eval_dataset,
              callbacks=[checkpoint_callback, tensorboard_callback])

    # Evaluate the model
    print("Evaluating model...")
    eval_loss, eval_accuracy = model.evaluate(eval_dataset)
    print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")

    # Save the trained model
    print("Saving model...")
    model.save(os.path.join(args.job_dir, "model"))

def main(args):
    # Set up output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Train and evaluate
    train_and_evaluate(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingdata_dir", type=str, required=True,
                        help="Directory containing the training QuickDraw data in ndjson format.")
    parser.add_argument("--eval_data_dir", type=str, required=True,
                        help="Directory containing the eval QuickDraw data in ndjson format.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store the converted TFRecord files and class map.")
    parser.add_argument("--job-dir", type=str, required=True,
                        help="Directory to store the trained model, checkpoints, and logs.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer.")

    args = parser.parse_args()
    main(args)
