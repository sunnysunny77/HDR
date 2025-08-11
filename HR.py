import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(True)

NUM_CLASSES = 10
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 512
EPOCHS = 66
AUTOTUNE = tf.data.AUTOTUNE
initial_learning_rate = 1e-3

df_train = pd.read_csv("./emnist-digits-train.csv", header=None)
df_test = pd.read_csv("./emnist-digits-test.csv", header=None)

def prep(X):
    return (np.rot90(np.flip(X.reshape(-1, 28, 28), axis=2), k=1, axes=(1, 2))[..., None] / 255.0).astype(np.float32)

X_train_all = prep(df_train.drop(columns=[0]).values)
y_train_all = df_train[0].values
X_test = prep(df_test.drop(columns=[0]).values)
y_test = df_test[0].values

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.15, random_state=42, stratify=y_train_all
)

model = models.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=IMG_SHAPE),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(NUM_CLASSES, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=(len(X_train) // BATCH_SIZE) * EPOCHS,
    alpha=1e-5 / initial_learning_rate
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=11, restore_best_weights=True)
]

y_train_onehot = tf.one_hot(y_train, depth=NUM_CLASSES)
y_val_onehot = tf.one_hot(y_val, depth=NUM_CLASSES)
y_test_onehot = tf.one_hot(y_test, depth=NUM_CLASSES)

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
    .shuffle(10000)
    .map(lambda x, y: (tf.cast(data_augmentation(x), tf.float32), y), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test_onehot)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"test_acc: {test_acc:.4f} | test_loss: {test_loss:.4f}")
#test_acc: 0.9944 | test_loss: 0.5389