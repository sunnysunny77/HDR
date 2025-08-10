import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models

NUM_CLASSES = 10
BATCH_SIZE = 256
IMG_SHAPE = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

df_train = pd.read_csv("./emnist-digits-train.csv", header=None)
df_test = pd.read_csv("./emnist-digits-test.csv", header=None)

X_train = df_train.drop(columns=[0]).values
y_train = df_train[0].values

X_test = df_test.drop(columns=[0]).values
y_test = df_test[0].values

def prep(X_scaled):
    return np.fliplr(np.rot90(X_scaled.reshape(-1, 28, 28), 1, axes=(1, 2)))[..., None].astype(np.float32)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_fixed = prep(X_train_scaled)
X_test_fixed = prep(X_test_scaled)

X_train = X_train_fixed.reshape(-1, 28, 28, 1)
X_test = X_test_fixed.reshape(-1, 28, 28, 1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

augment_layers = tf.keras.Sequential([
    layers.RandomRotation(0.15, fill_mode='nearest'),
    layers.RandomTranslation(0.2, 0.2, fill_mode='nearest'),
    layers.RandomZoom(0.15, 0.15, fill_mode='nearest'),
    layers.RandomContrast(0.1),
])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000)
train_ds = train_ds.map(lambda img, lbl: (augment_layers(img, training=True), lbl), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

model = models.Sequential([
    
    layers.Input(shape=IMG_SHAPE),

    layers.Conv2D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation='softmax'),
    
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=33,
    verbose=2,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(test_ds)
print("test_acc, {} | test_loss, {}".format(test_acc, test_loss))
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models

NUM_CLASSES = 10
BATCH_SIZE = 256
IMG_SHAPE = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

df_train = pd.read_csv("./emnist-digits-train.csv", header=None)
df_test = pd.read_csv("./emnist-digits-test.csv", header=None)

X_train = df_train.drop(columns=[0]).values
y_train = df_train[0].values

X_test = df_test.drop(columns=[0]).values
y_test = df_test[0].values

def prep(X_scaled):
    return np.fliplr(np.rot90(X_scaled.reshape(-1, 28, 28), 1, axes=(1, 2)))[..., None].astype(np.float32)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_fixed = prep(X_train_scaled)
X_test_fixed = prep(X_test_scaled)

X_train = X_train_fixed.reshape(-1, 28, 28, 1)
X_test = X_test_fixed.reshape(-1, 28, 28, 1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

augment_layers = tf.keras.Sequential([
    layers.RandomRotation(0.15, fill_mode='nearest'),
    layers.RandomTranslation(0.2, 0.2, fill_mode='nearest'),
    layers.RandomZoom(0.15, 0.15, fill_mode='nearest'),
    layers.RandomContrast(0.1),
])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000)
train_ds = train_ds.map(lambda img, lbl: (augment_layers(img, training=True), lbl), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

model = models.Sequential([
    
    layers.Input(shape=IMG_SHAPE),

    layers.Conv2D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation='softmax'),
    
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=33,
    verbose=2,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(test_ds)
print("test_acc, {} | test_loss, {}".format(test_acc, test_loss))
#test_acc, 0.9937750101089478 | test_loss, 0.021786343306303024