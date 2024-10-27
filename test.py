import tensorflow as tf

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(20,), activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Dummy data
import numpy as np
X = np.random.rand(100, 20)
y = np.random.rand(100, 1)

# Train the model
model.fit(X, y, epochs=5)

# Save the model
model.save('test_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('test_model.h5')
