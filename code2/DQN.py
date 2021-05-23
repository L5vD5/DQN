import tensorflow as tf

class QNetWorkDense(tf.keras.Model):
    def __init__(self, odim, adim):
        super(QNetWorkDense, self).__init__()
        self.layer1 = tf.keras.layers.Dense(24, activation='relu', kernel_initializer='he_uniform')
        self.output_layer = tf.keras.layers.Dense(adim, activation='linear', kernel_initializer='he_uniform')
        self.build(input_shape=(None,) + odim)

    def call(self, x):
        x = self.layer1(x)

QNet = QNetWorkDense(28, 8)