import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(1, input_shape=[1])])
model.compile(optimizer="sgd", loss='mean_squared_error')
xs = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

# ys = ys / 100_000.0

model.fit(xs, ys, epochs=1000)
print(model.predict([7.0]) * 100000)
