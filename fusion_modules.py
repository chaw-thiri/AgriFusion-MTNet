import tensorflow as tf
from tensorflow.keras import layers

class EarlyFusion(tf.keras.layers.Layer):
    """Concatenates multiple input tensors along the last axis."""
    def __init__(self, name='early_fusion'):
        super(EarlyFusion, self).__init__(name=name)

    def call(self, inputs):
        """
        Concatenates input tensors along the last axis.
        
        Args:
            inputs: List of tensors with compatible shapes except for the last dimension.
        
        Returns:
            Concatenated tensor along the last axis.
        """
        return tf.concat(inputs, axis=-1)

    def get_config(self):
        return super(EarlyFusion, self).get_config()

class LateFusion(tf.keras.layers.Layer):
    """Computes the mean of multiple input tensors stacked along a new axis."""
    def __init__(self, name='late_fusion'):
        super(LateFusion, self).__init__(name=name)

    def call(self, inputs):
        """
        Computes the mean of stacked input tensors.
        
        Args:
            inputs: List of tensors with identical shapes.
        
        Returns:
            Mean tensor across the stacked inputs.
        """
        return tf.reduce_mean(tf.stack(inputs, axis=0), axis=0)

    def get_config(self):
        return super(LateFusion, self).get_config()

class GatedFusion(tf.keras.layers.Layer):
    """Applies gated fusion to two input tensors using a 1x1 convolution."""
    def __init__(self, filters, use_batch_norm=True, name='gated_fusion'):
        """
        Initializes the GatedFusion layer.
        
        Args:
            filters: Number of filters for the 1x1 convolution.
            use_batch_norm: Whether to apply batch normalization after convolution.
            name: Name of the layer.
        """
        super(GatedFusion, self).__init__(name=name)
        self.filters = filters
        self.use_batch_norm = use_batch_norm
        self.gate_conv = layers.Conv2D(filters, 1, activation=None)
        self.batch_norm = layers.BatchNormalization() if use_batch_norm else None
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, inputs):
        """
        Applies gated fusion to two input tensors.
        
        Args:
            inputs: List or tuple of two tensors [x1, x2] with compatible shapes.
        
        Returns:
            Fused tensor computed as gate * x1 + (1 - gate) * x2.
        """
        x1, x2 = inputs
        gate = self.gate_conv(x1 + x2)
        if self.use_batch_norm:
            gate = self.batch_norm(gate)
        gate = self.sigmoid(gate)
        return gate * x1 + (1 - gate) * x2

    def get_config(self):
        config = super(GatedFusion, self).get_config()
        config.update({
            'filters': self.filters,
            'use_batch_norm': self.use_batch_norm
        })
        return config