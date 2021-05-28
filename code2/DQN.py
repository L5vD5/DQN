import tensorflow as tf

class DQNNetwork(tf.keras.Model):
    def __init__(self, odim, adim):
        super(DQNNetwork, self).__init__()
        self.layer1 = tf.keras.layers.Dense(24, input_shape=(odim,), activation='relu', kernel_initializer='he_uniform')
        self.layer2 = tf.keras.layers.Dense(24, activation='relu', kernel_initializer='he_uniform')
        self.output_layer = tf.keras.layers.Dense(adim, activation='linear', kernel_initializer='he_uniform')

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x
