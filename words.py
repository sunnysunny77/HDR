import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import string
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

with open("./words.txt") as f:
    lines = f.readlines()
    
RAW = lines[18:]

LABELS = []
for line in RAW:
    parts = line.strip().split()
    label = parts[-1]
    if len(label) == 4:
        LABELS.append(label)

BATCH_SIZE = 128
NUM_SAMPLES_PER_WORD = 500
LETTERS = np.array(list(string.ascii_uppercase))
LETTER_INDEX = {ch: i for i, ch in enumerate(LETTERS)}

LABELS = np.char.upper(np.array(LABELS))
UNIQUE, COUNTS = np.unique(LABELS, return_counts=True)
ORDER = np.argsort(-COUNTS)
TOP = UNIQUE[ORDER][:100]

MASK = np.array([set(w).issubset(set(LETTERS)) for w in TOP])
WORDS = TOP[MASK]

WORD_INDEX = {w: i for i, w in enumerate(WORDS)}
NUM_WORDS = len(WORDS)
MAX_LETTERS = np.max(np.char.str_len(WORDS))
IMG_HEIGHT = 28
IMG_WIDTH = IMG_HEIGHT * MAX_LETTERS

print(WORDS)

df_train = pd.read_csv("./emnist-byclass-train.csv", header=None)

X_train = df_train.drop(columns=[0]).to_numpy()
y_train = df_train[0].to_numpy()

MASK = (10 <= y_train) & (y_train <= 35)
X_train, y_train = X_train[MASK], y_train[MASK] - 10

X_train = X_train.reshape(-1, 28, 28)

X_train = np.flip(np.rot90(X_train, k=3, axes=(1, 2)), axis=2)

CANDIDATES = {i: np.where(y_train == i)[0] for i in range(len(LETTERS))}

def generate_word():
    X_words, y_words = [], []
    for word in WORDS:
        for _ in range(NUM_SAMPLES_PER_WORD):
            imgs = []
            for char in word.upper():
                idx = LETTER_INDEX[char]
                candidates = CANDIDATES[idx]
                img = X_train[np.random.choice(candidates)]
                imgs.append(img)
            word_img = np.hstack(imgs)
            word_img = cv2.resize(word_img, (IMG_WIDTH, IMG_HEIGHT))
            X_words.append(word_img[..., np.newaxis].astype(np.float32) / 255.0)
            y_words.append(WORD_INDEX[word])
    X_words = np.array(X_words)
    y_words = to_categorical(y_words, num_classes=NUM_WORDS)
    idx = np.random.permutation(len(X_words))
    return X_words[idx], y_words[idx]

X_train, y_train = generate_word()

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train.argmax(axis=1)
)

augmentation = Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05)
])

def prepare_augmentation(x, y):
    x = augmentation(x, training=True)
    return x, y

train_ds = (
    Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .map(prepare_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

x = layers.Conv2D(16, 3, strides=1, padding="same", use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
x = layers.SpatialDropout2D(0.1)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False)(x)
x = layers.SpatialDropout2D(0.1)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False)(x)
x = layers.SpatialDropout2D(0.15)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dense(128, use_bias=False)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_WORDS, activation="softmax", dtype="float32")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=CategoricalFocalCrossentropy(gamma=2.0, from_logits=False, label_smoothing=0.1),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
)

y_val_pred = model.predict(val_ds)
y_val_labels = y_val.argmax(axis=1)
y_pred_labels = y_val_pred.argmax(axis=1)

accuracy = accuracy_score(y_val_labels, y_pred_labels)
print("Validation Accuracy:", accuracy)