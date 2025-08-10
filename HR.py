import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 10
BATCH_SIZE = 64
IMG_SHAPE = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 36

df_train = pd.read_csv("./emnist-digits-train.csv", header=None)
df_test = pd.read_csv("./emnist-digits-test.csv", header=None)

def prep(X_raw):
    return np.rot90(np.flip(X_raw.reshape(-1, 28, 28), axis=2),k=1, axes=(1, 2))[..., np.newaxis] / 255.0

X_train = prep(df_train.drop(columns=[0]).values)
y_train = df_train[0].values

X_test = prep(df_test.drop(columns=[0]).values)
y_test = df_test[0].values

y_train_onehot = to_categorical(y_train, NUM_CLASSES)
y_test_onehot = to_categorical(y_test, NUM_CLASSES)

X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
    X_train, y_train_onehot, test_size=0.15, random_state=42, stratify=y_train
)

augmentation = tf.keras.Sequential([
    layers.RandomTranslation(height_factor=0.01, width_factor=0.01, fill_mode='nearest'),
    layers.RandomRotation(factor=0.01, fill_mode='nearest'),
    layers.RandomZoom(height_factor=(-0.01, 0.01), width_factor=(-0.01, 0.01), fill_mode='nearest')
])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
train_ds = train_ds.shuffle(10000)
train_ds = train_ds.map(lambda img, lbl: (augmentation(img, training=True), lbl), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test_onehot))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

steps_per_epoch = len(train_ds)
total_steps = steps_per_epoch * EPOCHS
initial_learning_rate = 1e-3
alpha = 1e-5 / initial_learning_rate

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=total_steps,
    alpha=alpha
)

model = models.Sequential([
    layers.Input(shape=IMG_SHAPE),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"test_acc: {test_acc:.4f} | test_loss: {test_loss:.4f}")
#test_acc: 0.9967 | test_loss: 0.5125