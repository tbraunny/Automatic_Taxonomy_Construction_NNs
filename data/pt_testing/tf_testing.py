import tensorflow as tf

class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            return self.dense(x)