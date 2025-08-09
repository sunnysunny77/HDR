import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

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

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

train_datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

model = models.Sequential([
    
    layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.25),

    layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.35),

    layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_datagen.flow(X_train, y_train, batch_size=256),
    validation_data=(X_val, y_val),
    epochs=17,
    verbose=2
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("test_acc, {} | test_loss, {}".format(test_acc, test_loss))
#test_acc, 0.9973250031471252 | test_loss, 0.012162618339061737