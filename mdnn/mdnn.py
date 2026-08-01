import tensorflow as tf
from tensorflow.keras import layers, Model

class FourColumnMDNN(Model):
    def __init__(self):
        super(FourColumnMDNN, self).__init__()

        # Very Large Receptive Field Column
        self.column1 = tf.keras.Sequential([
            layers.Conv2D(16, kernel_size=11, padding='same', use_bias=True),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(32, kernel_size=9, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(64, kernel_size=7, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # Large Receptive Field Column
        self.column2 = tf.keras.Sequential([
            layers.Conv2D(16, kernel_size=9, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(32, kernel_size=7, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(64, kernel_size=5, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # Medium Receptive Field Column
        self.column3 = tf.keras.Sequential([
            layers.Conv2D(20, kernel_size=7, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(40, kernel_size=5, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(80, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # Small Receptive Field Column
        self.column4 = tf.keras.Sequential([
            layers.Conv2D(24, kernel_size=5, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(48, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(96, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # Fusion Layer
        self.fusion = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=1, activation='relu'),
            layers.Conv2D(1, kernel_size=1, activation='relu')
        ])

    def call(self, inputs, training=False):
        x1 = self.column1(inputs, training=training)
        x2 = self.column2(inputs, training=training)
        x3 = self.column3(inputs, training=training)
        x4 = self.column4(inputs, training=training)

        x = tf.concat([x1, x2, x3, x4], axis=-1)
        output = self.fusion(x, training=training)

        return output


# model = FourColumnMDNN()
# model.summary()

# dummy_input = tf.random.normal((1, 256, 256, 3))
# output = model(dummy_input)

# print("Output shape:", output.shape)