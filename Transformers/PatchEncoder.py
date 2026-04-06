import tensorflow as tf
from tensorflow.keras import layers as Layer

class PatchEncoder(Layer):
    def __init__(self, N_PATCHES, HIDDEN_SIZE):
        super(PatchEncoder,self).__init__(name = 'patch_encoder')

        self.linear_projection = Dense(HIDDEN_SIZE)
        self.positional_embbeding = Embedding(N_PATCHES,HIDDEN_SIZE)
        self.N_Patches = N_PATCHES

    def call(self,x):
        patches = tf.image.extract_patches(
            images = x,
            sizes=[1,16,16,1],
            strides=[1,16,16,1],
            rates=[1,1,1,1],
            padding="VALID"
        )

        patches = tf.reshape((patches)(patches.shape[0],-1,patches.shape(-1)))
        embbeding_input = tf.range(start = 0, limit = self.N_Patches, delta = 1)
        output = self.linear_projection(patches) + self.positional_embbeding(embbeding_input)

        x = self.conv(x)
        x = self.batch_norm(x, training)

        return output