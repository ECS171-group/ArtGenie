import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(len(train_labels))
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Rest of your code remains the same
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

# Get the predictions for the test set
predictions = model.predict(ds_test)

# Convert the predictions to class labels
predicted_labels = tf.argmax(predictions, axis=1)

# Plot the images and their predicted/true labels
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(12, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, ax in enumerate(axes.flat):
    # Get the image, predicted label, and true label
    image = test_images[i]
    predicted_label = predicted_labels[i].numpy()
    true_label = test_labels[i]
    
    # Plot the image
    ax.imshow(image, cmap='gray')
    
    # Set the title with the predicted and true labels
    if predicted_label == true_label:
        ax.set_title(f"Predicted: {predicted_label}, True: {true_label}", color='green')
    else:
        ax.set_title(f"Predicted: {predicted_label}, True: {true_label}", color='red')
    
    # Remove the axis labels
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
