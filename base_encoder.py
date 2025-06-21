import tensorflow as tf
from tensorflow.keras import layers

class FlexibleCNNEncoder(tf.keras.Model):
    def __init__(self, input_channels=12, name='flexible_cnn_encoder'):
        super(FlexibleCNNEncoder, self).__init__(name=name)
        self.input_channels = input_channels
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1')
        self.conv2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2')
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), name='pool')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3')

    def call(self, x):
        """
        Processes input through the CNN encoder and returns feature maps at different scales.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, input_channels).
        
        Returns:
            Tuple of three feature maps (x1, x2, x3).
        """
        x1 = self.conv1(x)  # (batch_size, height, width, 64)
        x2 = self.pool(x1)  # (batch_size, height/2, width/2, 64)
        x2 = self.conv2(x2)  # (batch_size, height/2, width/2, 128)
        x3 = self.pool(x2)  # (batch_size, height/4, width/4, 128)
        x3 = self.conv3(x3)  # (batch_size, height/4, width/4, 256)
        return x1, x2, x3

    def get_config(self):
        config = super(FlexibleCNNEncoder, self).get_config()
        config.update({
            'input_channels': self.input_channels,
            'name': self.name
        })
        return config