import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BATCH_SIZE = 512
NUM_CLASSES = 62
EPOCHS = 50

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

blocks = [
    (16, False),
    (48, True),
    (64, False),
    (128, False)
]

for filters, downsample in blocks:
    shortcut = x
    stride = 2 if downsample else 1

    x = layers.Conv2D(filters, 3, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x) 

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

lr = 1e-3

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr, weight_decay=1e-4
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2
)

pred_probs = model.predict(test_ds)
pred_labels = tf.argmax(pred_probs, axis=1)

accuracy = accuracy_score(y_test, pred_labels)
print("Test accuracy:", accuracy)
#86