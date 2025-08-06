import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import numpy as np
from sklearn.model_selection import train_test_split

# --- GPU Configuration ---
# Enable GPU memory growth (avoids reserving all memory)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Enable XLA (Accelerated Linear Algebra) JIT compilation
tf.config.optimizer.set_jit(True)

# Enable mixed precision training (float16)
mixed_precision.set_global_policy('mixed_float16')

# --- Load and preprocess data ---
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_full = np.expand_dims(x_train_full, -1)  # Shape: (num_samples, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# --- Train/validation split ---
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=42
)

# --- tf.image-based data augmentation function ---
def augment(image, label):
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    # please list some best augments for the data set
    return image, label

# --- Prepare tf.data pipelines ---
batch_size = 128

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000)
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Build optimized CNN model with L2 regularization ---
l2_reg = keras.regularizers.l2(0.001)

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),

    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),

    # Ensure final output is float32 for numerical stability
    keras.layers.Dense(10, activation='softmax', dtype='float32')
])

model.summary()

# --- Compile model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# --- Train the model ---
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop, reduce_lr]
)

# --- Evaluate on test data ---
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")