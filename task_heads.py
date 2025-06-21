import tensorflow as tf
from tensorflow.keras import layers

class SegmentationHead(tf.keras.layers.Layer):
    def __init__(self, filters=1, name='segmentation_head'):
        """
        Initializes the segmentation head for pixel-wise classification.
        
        Args:
            filters: Number of output channels (e.g., 1 for binary segmentation).
            name: Name of the layer.
        """
        super(SegmentationHead, self).__init__(name=name)
        self.filters = filters
        self.conv = layers.Conv2D(filters, 1, activation='sigmoid', name='seg_conv')

    def call(self, x):
        """
        Applies 1x1 convolution for segmentation output.
        
        Args:
            x: Input feature map of shape (batch_size, height, width, channels).
        
        Returns:
            Segmentation output of shape (batch_size, height, width, filters).
        """
        return self.conv(x)

    def get_config(self):
        config = super(SegmentationHead, self).get_config()
        config.update({
            'filters': self.filters,
            'name': self.name
        })
        return config

class RegressionHead(tf.keras.layers.Layer):
    def __init__(self, name='regression_head'):
        """
        Initializes the regression head for scalar output prediction.
        
        Args:
            name: Name of the layer.
        """
        super(RegressionHead, self).__init__(name=name)
        self.global_pool = layers.GlobalAveragePooling2D(name='global_pool')
        self.fc = layers.Dense(1, name='fc')

    def call(self, x):
        """
        Applies global average pooling and dense layer for regression output.
        
        Args:
            x: Input feature map of shape (batch_size, height, width, channels).
        
        Returns:
            Scalar output of shape (batch_size, 1).
        """
        x = self.global_pool(x)
        return self.fc(x)

    def get_config(self):
        config = super(RegressionHead, self).get_config()
        config.update({
            'name': self.name
        })
        return config