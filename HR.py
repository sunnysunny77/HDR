import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

BATCH_SIZE = 256
NUM_CLASSES = 10
EPOCHS = 100

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(True)

def fix_orientation(images):
    images = np.rot90(images, k=3, axes=(1, 2))
    images = np.flip(images, axis=2)
    return images

df_train = pd.read_csv("./emnist-digits-train.csv", header=None)
df_test = pd.read_csv("./emnist-digits-test.csv", header=None)

X = df_train.drop(columns=[0]).to_numpy().reshape(-1, 28, 28, 1)
y = df_train[0]

X_test = df_test.drop(columns=[0]).to_numpy().reshape(-1, 28, 28, 1)
y_test = df_test[0]

X = fix_orientation(X).astype(np.float32) / 255.0
X_test = fix_orientation(X_test).astype(np.float32) / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),   
    layers.RandomTranslation(0.1, 0.1), 
    layers.RandomZoom(0.1),      
    layers.RandomContrast(0.1)  
])

def augment(images, labels):
    images = data_augmentation(images)
    return images, labels

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

model = tf.keras.models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stop],
    verbose=2
)

pred_probs = model.predict(test_ds)
pred_labels = pred_probs.argmax(axis=1)

accuracy = accuracy_score(y_test, pred_labels)
print("Test accuracy:", accuracy)
#Test accuracy: 0.9944