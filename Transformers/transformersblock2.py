import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout_rate=0.1,
        layer_norm_eps=1e-6,
        name="transformer_block"
    ):
        super(TransformerBlock, self).__init__(name=name)

        # =========================
        # 1. MULTI-HEAD ATTENTION
        # =========================
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate,
            name="multihead_attention"
        )

        # =========================
        # 2. FEED FORWARD NETWORK
        # =========================
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu", name="ffn_dense_1"),
            layers.Dense(embed_dim, name="ffn_dense_2"),
        ], name="feed_forward_network")

        # =========================
        # 3. NORMALIZACIÓN
        # =========================
        self.layernorm1 = layers.LayerNormalization(epsilon=layer_norm_eps, name="layer_norm_1")
        self.layernorm2 = layers.LayerNormalization(epsilon=layer_norm_eps, name="layer_norm_2")

        # =========================
        # 4. DROPOUT
        # =========================
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        """
        inputs: shape (batch_size, seq_len, embed_dim)
        """

        # =========================
        # 1. SELF-ATTENTION
        # =========================
        attn_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )

        # Dropout
        attn_output = self.dropout1(attn_output, training=training)

        # Residual Connection + LayerNorm
        out1 = self.layernorm1(inputs + attn_output)

        # =========================
        # 2. FEED FORWARD NETWORK
        # =========================
        ffn_output = self.ffn(out1)

        # Dropout
        ffn_output = self.dropout2(ffn_output, training=training)

        # Residual Connection + LayerNorm
        output = self.layernorm2(out1 + ffn_output)

        return output