import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(True)

NUM_CLASSES = 10
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 512
EPOCHS = 40
AUTOTUNE = tf.data.AUTOTUNE
WEIGHT_DECAY = 1e-4
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

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', 
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

inputs = layers.Input(shape=IMG_SHAPE)

x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, 64)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.25)(x)

x = residual_block(x, 128, stride=2) 
x = layers.Dropout(0.25)(x)

x = residual_block(x, 256, stride=2)
x = layers.Dropout(0.25)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

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
    .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
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
#test_acc: 0.9975 | test_loss: 0.5173