import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

BATCH_SIZE = 512
NUM_CLASSES = 62
EPOCHS = 30

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(True)

def fix_orientation(images):
    return np.flip(np.rot90(images, k=3, axes=(1, 2)), axis=2)

df_train = pd.read_csv("./emnist-byclass-train.csv", header=None)
df_test = pd.read_csv("./emnist-byclass-test.csv", header=None)

X = df_train.drop(columns=[0]).to_numpy().reshape(-1, 28, 28, 1)
y = df_train[0].to_numpy()

X_test = df_test.drop(columns=[0]).to_numpy().reshape(-1, 28, 28, 1)
y_test = df_test[0].to_numpy()

X = fix_orientation(X).astype(np.float32) / 255.0
X_test = fix_orientation(X_test).astype(np.float32) / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_oh = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05),
    layers.GaussianNoise(0.05),
])

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test_oh))
    .batch(BATCH_SIZE)
)

inputs = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, 3, strides=2, padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(16, 3, strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(48, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x) 
x = layers.ReLU()(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x) 

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

steps_per_epoch = len(X_train) // BATCH_SIZE

total_steps = steps_per_epoch * EPOCHS

cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=total_steps,
    alpha=0.01
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=cosine_decay, weight_decay=1e-4
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

class_weights = {
    i: w for i, w in enumerate(
        compute_class_weight(
            class_weight="balanced",
            classes=np.arange(NUM_CLASSES),
            y=y_train
        )
    )
}

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
]

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

pred_probs = model.predict(test_ds)
pred_labels = tf.argmax(pred_probs, axis=1)

accuracy = accuracy_score(y_test, pred_labels)
print("Test accuracy:", accuracy)
