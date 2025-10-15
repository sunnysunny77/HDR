# Importing necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, accuracy_score

# Setting gpu optimizations
tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

# Loading and inspecting the data
with open("./words.txt") as text:
    word_meta = text.readlines()

# Whole words
for line in word_meta[:20]:
    print(repr(line))

df = pd.read_csv("./emnist-byclass-test.csv", header=None)

# Individual letters
df

# Perparing the data

# Drop unnecessary comments
word_meta_raw = word_meta[18:]

# 784 pixel letters
X_letters = df.drop(columns=[0]).to_numpy()

# True labels letters
y_letters = df[0].to_numpy()

# True labels words
y_words = []

for line in word_meta_raw:
    
    # Select four letter words
    parts = line.strip().split()
    y = parts[-1]
    if len(y) == 4:
        y_words.append(y)

# Uppercase numpy
y_words = np.char.upper(np.array(y_words))

# Top 75 counted unquie words
unique, counts = np.unique(y_words, return_counts=True)
ordered = np.argsort(-counts)
top = unique[ordered][:75]

# Create numpy of valid words
vaild_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
word_array = [word for word in top if set(word).issubset(vaild_chars)]
word_array = np.array(word_array)

print(word_array)

# Word dictionary
word_dict = {word: index for index, word in enumerate(word_array)}

# Use captial letters from byclass
mask = (10 <= y_letters) & (y_letters <= 35)
X_letters, y_letters = X_letters[mask], y_letters[mask] - 10

# Fix orientaion of letters
before = X_letters[0].reshape(28, 28)
plt.imshow(before, cmap='gray')
plt.title("Before Fix (Original EMNIST Orientation)")
plt.axis("off")
plt.show()

X_letters = X_letters.reshape(-1, 28, 28)
X_letters = np.flip(np.rot90(X_letters, k=3, axes=(1, 2)), axis=2)

after = X_letters[0]
plt.figure(figsize=(3, 3))
plt.imshow(after, cmap='gray')
plt.title("After Fix (Upright Orientation)")
plt.axis("off")
plt.show()


# Generate words from letters

# Constants
num_samples_per_word = 500
num_words = len(word_array)
max_letters = np.max(np.char.str_len(word_array))
img_height = 28
img_width = img_height * max_letters
candidate_dict = {index: np.where(y_letters == index)[0] for index in range(26)}

def generate_word():
    X_words, y_words = [], []
    for word in word_array:
        for _ in range(num_samples_per_word):
            imgs = []
            # Combine letters
            for char in word.upper():
                idx = ord(char) - ord('A')
                candidates = candidate_dict[idx]
                img = X_letters[np.random.choice(candidates)]
                imgs.append(img)
            word_img = np.hstack(imgs)
            word_img = cv2.resize(word_img, (img_width, img_height))
            # Normalize and create feature
            X_words.append(word_img[..., np.newaxis].astype(np.float32) / 255.0)
            # Create true labels
            y_words.append(word_dict[word])
    # Feature numpy
    X_words = np.array(X_words)
    # One hot encode for loss calculation
    y_words = to_categorical(y_words, num_classes=num_words)
    idx = np.random.permutation(len(X_words))
    return X_words[idx], y_words[idx]

X_train, y_train = generate_word()

# Plot the first generated word image
plt.figure(figsize=(6, 3))
plt.imshow(X_train[0].squeeze(), cmap='gray')
plt.title("Generated Word Example")
plt.axis("off")
plt.show()

# Split off test set (10% of original data)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train.argmax(axis=1)
)

# Split remaining training data into train and validation (10% of train as validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train.argmax(axis=1)
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Augmentation pipeline
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.05)
])

def prepare_augmentation(x, y):
    x = augmentation(x, training=True)
    return x, y

batch_size = 128

# Create TensorFlow datasets
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(2048)
    .batch(batch_size)
    .map(prepare_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# Define model
inputs = layers.Input(shape=(img_height, img_width, 1))

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

outputs = layers.Dense(num_words, activation="softmax", dtype="float32")(x)

model = models.Model(inputs, outputs)

# Compile the model
# Focal loss focuses more on hard-to-classify examples (those the model predicts with low confidence) by down-weighting easy examples.
# Label smoothing replaces the hard 1/0 with slightly “softer” values:
model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=CategoricalFocalCrossentropy(gamma=2.0, from_logits=False, label_smoothing=0.1),
    metrics=["accuracy"]
)

# Training the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
)

# Predict on test set
y_test_pred = model.predict(test_ds)
y_test_labels = y_test.argmax(axis=1)
y_pred_test_labels = y_test_pred.argmax(axis=1)

# Overall accuracy on test
overall_test_accuracy = accuracy_score(y_test_labels, y_pred_test_labels)
print("Test Accuracy:", overall_test_accuracy)

# Per-class confusion matrices
mcm_test = multilabel_confusion_matrix(y_test_labels, y_pred_test_labels)

# Per-class metrics
precisions_test = precision_score(y_test_labels, y_pred_test_labels, average=None, zero_division=0)
recalls_test = recall_score(y_test_labels, y_pred_test_labels, average=None, zero_division=0)

# Build DataFrame for test set
results_test = []
for cls, cm in enumerate(mcm_test):
    tn, fp, fn, tp = cm.ravel()
    y_true_bin = (y_test_labels == cls).astype(int)
    y_pred_bin = (y_pred_test_labels == cls).astype(int)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    results_test.append({
        "Word": word_array[cls],
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precisions_test[cls],
        "Recall": recalls_test[cls],
        "Accuracy": acc,
    })

df_metrics_test = pd.DataFrame(results_test)
df_metrics_test = df_metrics_test.sort_values(by="Accuracy", ascending=False)

pd.set_option("display.max_rows", None)
print(df_metrics_test)

# Save the model
tf.saved_model.save(model, "HR")

# Convert the model for web
!tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model HR tfjs_model