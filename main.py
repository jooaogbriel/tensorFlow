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


X = np.arange(-100, 100, 4)

# Create labels (using tensors)
y = np.arange(-90, 110, 4)

# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.Dense(1)
])

model_1.compile()