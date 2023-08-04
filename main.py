import tensorflow as tf
import numpy as np

#create a tensor
# scalar = tf.constant(7)

# #check number of dimensions 
# scalar.ndim

# #create a vector
# vector = tf.constant([10, 10])

# #create a matrix(2 dimensions)
# matrix = tf.constant([10, 7],
#                      [10, 7])

# escalar : um único número.
# vetor : um número com direção (por exemplo, velocidade do vento com direção).
# matriz : uma matriz bidimensional de números.
# tensor : uma matriz n-dimensional de números (onde n pode ser qualquer número, um tensor de dimensão 0 é um escalar, um tensor de dimensão 1 é um vetor).


X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Set random seed
tf.random.set_seed(42)

# Create a model (same as above)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Fit model (this time we'll train for longer)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100) # train for 100 epochs not 10