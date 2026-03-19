
import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import GlobalAveragePooling1D, MultiHeadAttention, Dense, Dropout, LayerNormalization, Input, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

def make_windows(features, targets, input_len, forecast_len):
    """Create sliding windows of features and corresponding target forecasts."""
    X, y_win = [], []
    for i in range(len(features) - input_len - forecast_len + 1):
        X.append(features[i : i + input_len])
        y_win.append(targets[i + input_len : i + input_len + forecast_len])
    return np.array(X), np.array(y_win)

def create_sequences(data, lookback=168, forecast_len=24):
    """Create sequences of input data and corresponding forecast targets."""
    X, y_seq = [], []
    for i in range(lookback, len(data) - forecast_len + 1):
        X.append(data[i-lookback:i, 0])
        y_seq.append(data[i:i+forecast_len, 0])
    return np.array(X), np.array(y_seq)

def build_lstm_model(lookback, forecast_horizon, num_features=1, learning_rate=0.001):
    """Build and compile bidirectional LSTM model."""
    inputs = Input(shape=(lookback, num_features))
    x = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(units=64, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(units=forecast_horizon)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, weight_decay=1e-6),
        loss='mse',
        metrics=['mae']
    )
    return model


class PositionalEncoding(layers.Layer):
    """Positional Encoding Layer for Transformer models."""
    def __init__(self, max_len=512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model   = input_shape[-1]
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims      = np.arange(0, d_model, 2)[np.newaxis, :]
        angles    = positions / np.power(10000, dims / d_model)

        pe = np.zeros((self.max_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

        self.pe = self.add_weight(
            name="pe", shape=(self.max_len, d_model),
            initializer=keras.initializers.Constant(pe),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        import tensorflow as tf
        seq_len = tf.shape(x)[1]
        return x + tf.cast(self.pe[:seq_len, :], x.dtype)

class TransformerEncoderBlock(layers.Layer):
    """Transformer Encoder Block with Multi-Head Attention and Feed-Forward Network."""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.rate      = dropout

        self.attn  = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout
        )
        self.ffn   = keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dropout(dropout),
            Dense(d_model),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":   self.d_model,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.rate,
        })
        return config

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x


def build_transformer_model(input_len, forecast_horizon, learning_rate=0.001, num_features=2):
    """Build and compile Transformer model with positional encoding and two encoder blocks."""
    input_layer = Input(shape=(input_len, num_features))
    x = Dense(64)(input_layer)
    x = PositionalEncoding(max_len=input_len * 2)(x)
    x = TransformerEncoderBlock(d_model=64, num_heads=4, ff_dim=128, dropout=0.1)(x)
    x = TransformerEncoderBlock(d_model=64, num_heads=4, ff_dim=128, dropout=0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    output = Dense(forecast_horizon)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, weight_decay=1e-6),
        loss='mse',
        metrics=['mae']
    )
    return model

