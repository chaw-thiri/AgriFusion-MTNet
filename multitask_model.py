import tensorflow as tf
from tensorflow.keras import layers
from base_encoder import FlexibleCNNEncoder
from fusion_modules import GatedFusion
from task_heads import SegmentationHead, RegressionHead

class MultiTaskModel(tf.keras.Model):
    def __init__(self, input_channels=12, name='multitask_model'):
        """
        Initializes the multi-task model for segmentation, phenology, and yield prediction.
        
        Args:
            input_channels: Number of input channels for the encoder.
            name: Name of the model.
        """
        super(MultiTaskModel, self).__init__(name=name)
        self.input_channels = input_channels
        self.encoder = FlexibleCNNEncoder(input_channels=input_channels, name='encoder')
        self.upsample = layers.UpSampling2D(size=(2, 2), name='upsample_x3')
        self.gated_fusion = GatedFusion(filters=128, name='gated_fusion')
        self.segmentation_head = SegmentationHead(name='segmentation_head')
        self.phenology_head = RegressionHead(name='phenology_head')
        self.yield_head = RegressionHead(name='yield_head')

    def call(self, inputs):
        """
        Processes inputs through the encoder, fusion, and task heads.
        
        Args:
            inputs: Input tensor of shape (batch_size, height, width, input_channels).
        
        Returns:
            Dictionary with outputs for segmentation, phenology, and yield tasks.
        """
        x1, x2, x3 = self.encoder(inputs)
        # Upsample x3 to match x2's spatial dimensions
        x3_upsampled = self.upsample(x3)  # (batch_size, height/2, width/2, 256)
        # Adjust x3 channels to match x2
        x3_adjusted = layers.Conv2D(128, 1, padding='same', name='adjust_x3')(x3_upsampled)
        fused = self.gated_fusion([x2, x3_adjusted])

        seg_out = self.segmentation_head(x1)
        phen_out = self.phenology_head(fused)
        yield_out = self.yield_head(fused)

        return {
            'segmentation': seg_out,
            'phenology': phen_out,
            'yield': yield_out
        }

    def get_config(self):
        config = super(MultiTaskModel, self).get_config()
        config.update({
            'input_channels': self.input_channels,
            'name': self.name
        })
        return config
    

model = MultiTaskModel(input_channels=12)
sample_input = tf.random.normal((1, 256, 256, 12))
outputs = model(sample_input)
print({k: v.shape for k, v in outputs.items()})