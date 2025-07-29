#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Load and preprocess data ---
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_full = np.expand_dims(x_train_full, -1)  # Shape: (num_samples, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# --- Split training into train/validation ---
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

# --- Data Augmentation ---
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
# datagen.fit(x_train)  # Not needed since no featurewise normalization

# --- Build the model with L2 regularization ---
l2_reg = keras.regularizers.l2(0.001)

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),

    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(10, activation='softmax')
])
model.summary()

# --- Compile the model ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- EarlyStopping ---
early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# --- Train with augmented data ---
batch_size = 64
steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size

model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    validation_data=(x_val, y_val),
    epochs=30,
    callbacks=[early_stop]
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# --- Visualize predictions ---
for i in range(5):
    img = x_test[i]
    true_label = y_test[i]
    pred = model.predict(img[np.newaxis, ...], verbose=0)
    pred_label = np.argmax(pred, axis=1)[0]

    color = 'green' if pred_label == true_label else 'red'
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"True: {true_label} | Pred: {pred_label}", color=color)
    plt.axis('off')
    plt.show()


# In[3]:


model.export('HDR')


# In[5]:


get_ipython().system('tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model HDR tfjs_model_HDR')


# In[ ]:




