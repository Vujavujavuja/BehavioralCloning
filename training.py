import tensorflow as tf
import numpy as np

x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

x = np.array(x)

y = tf.keras.utils.to_categorical(y, num_classes=7)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2], x.shape[3])),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=1000, batch_size=32)

model.save('trained_model.keras')
