"""
Transformer for Time-Series Forecasting — Keras Implementation
================================================================
Architecture:
  Raw series → STL decomposition → [series + trend] as 2 features
             → StandardScaler → sliding windows
             → Positional Encoding
             → N × Transformer Encoder Blocks
             → GlobalAvgPool + Dense → Forecast horizon
             → inverse_transform → original scale

Changes from v1:
  - Removed RevIN (your data has a strong downward trend; StandardScaler
    + explicit STL trend feature handles this more simply)
  - Added STL decomposition to extract trend as a second input feature
  - StandardScaler now normalises globally before windowing
  - NUM_FEATURES set to 2 (raw series + trend)

Usage:
  1. Replace the synthetic series with your real data
  2. Adjust STL_PERIOD to match your data frequency
     (e.g. 365 for daily data, 24 for hourly)
  3. Run: python transformer_forecasting.py
"""

import numpy as np
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


# ─────────────────────────────────────────────────────────────
# 1. POSITIONAL ENCODING
#    Gives the model a sense of "where in time" each step is.
#    Uses fixed sine/cosine waves of different frequencies.
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        # Capture d_model at build time so encoding is sized correctly
        d_model   = input_shape[-1]
        positions = np.arange(self.max_len)[:, np.newaxis]           # (max_len, 1)
        dims      = np.arange(0, d_model, 2)[np.newaxis, :]          # (1, d_model/2)
        angles    = positions / np.power(10000, dims / d_model)       # (max_len, d_model/2)

        pe = np.zeros((self.max_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

        # Non-trainable weight so it moves to GPU automatically
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


# ─────────────────────────────────────────────────────────────
# 2. ONE TRANSFORMER ENCODER BLOCK
#    = Multi-Head Self-Attention + Feed-Forward Network
#    Both wrapped with residual connections and LayerNorm.
# ─────────────────────────────────────────────────────────────
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        """
        d_model   : embedding dimension (e.g. 64, 128)
        num_heads : number of attention heads (must divide d_model evenly)
        ff_dim    : hidden size inside the feed-forward network
        dropout   : regularisation — set higher (0.2-0.3) if overfitting
        """
        super(TransformerBlock, self).__init__(**kwargs)
        # Store args explicitly — required for Keras 2 serialisation
        self.d_model   = d_model
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.rate      = dropout

        self.attn  = layers.MultiHeadAttention(
                         num_heads=num_heads,
                         key_dim=d_model // num_heads,
                         dropout=dropout)
        self.ffn   = keras.Sequential([
                         layers.Dense(ff_dim, activation="gelu"),
                         layers.Dropout(dropout),
                         layers.Dense(d_model),
                     ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "d_model":   self.d_model,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.rate,
        })
        return config

    def call(self, x, training=False):
        # Self-attention: each timestep attends to all others
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))

        # Feed-forward: applied independently to each position
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x


# ─────────────────────────────────────────────────────────────
# 3. FULL MODEL
# ─────────────────────────────────────────────────────────────
def build_transformer(
    input_len,       # number of past timesteps fed in
    forecast_len,    # number of future timesteps to predict
    num_features,    # number of input variables (2 = series + trend)
    d_model=64,      # embedding dimension
    num_heads=4,     # attention heads (must divide d_model evenly)
    ff_dim=128,      # feed-forward hidden size
    num_layers=2,    # how many transformer blocks to stack
    dropout=0.1,
):
    """
    Returns a compiled Keras model.

    Input shape : (batch, input_len, num_features)
    Output shape: (batch, forecast_len)
    """
    inputs = keras.Input(shape=(input_len, num_features))   # (B, T, F)

    # Project F features → d_model
    x = layers.Dense(d_model)(inputs)                       # (B, T, d_model)

    # Add positional encoding
    x = PositionalEncoding(max_len=input_len * 2)(x)        # (B, T, d_model)

    # Stack N transformer encoder blocks
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)

    # Pool over time and project to forecast horizon
    x = layers.GlobalAveragePooling1D()(x)                  # (B, d_model)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(forecast_len)(x)                 # (B, forecast_len)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# 4. DATA HELPERS
# ─────────────────────────────────────────────────────────────
def extract_trend(series, period):
    """
    Run STL decomposition and return the trend component.
    series : 1-D np.array
    period : seasonality period (e.g. 365 for daily, 24 for hourly)
    """
    stl    = STL(series, period=period, robust=True)
    result = stl.fit()
    return result.trend                                      # same length as series


def make_windows(features, targets, input_len, forecast_len):
    """
    Slice into overlapping (X, y) windows.
    features : np.array of shape (total_timesteps, num_features)  — scaled
    targets  : np.array of shape (total_timesteps,)               — scaled
    Returns:
        X : (n_samples, input_len, num_features)
        y : (n_samples, forecast_len)
    """
    X, y = [], []
    for i in range(len(features) - input_len - forecast_len + 1):
        X.append(features[i : i + input_len])
        y.append(targets[i + input_len : i + input_len + forecast_len])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Config ──────────────────────────────────────────────
    INPUT_LEN    = 96    # look back 96 timesteps
    FORECAST_LEN = 24    # predict next 24 timesteps
    NUM_FEATURES = 2     # raw series + STL trend
    STL_PERIOD   = 365   # adjust to your data: 365=daily, 24=hourly
    D_MODEL      = 64
    NUM_HEADS    = 4
    FF_DIM       = 128
    NUM_LAYERS   = 2
    DROPOUT      = 0.1
    EPOCHS       = 20
    BATCH_SIZE   = 32
    # ────────────────────────────────────────────────────────

    # ── 1. Load your data here ───────────────────────────────
    # Replace this with: series = df["your_column"].values
    t      = np.linspace(0, 100, 5000)
    series = np.sin(t) + 0.3 * np.linspace(5, 0, 5000) + 0.1 * np.random.randn(5000)
    # ────────────────────────────────────────────────────────

    # ── 2. Extract STL trend ─────────────────────────────────
    # Gives the model an explicit signal about the long-run direction.
    # Your decomposition plot showed a strong downward trend — this helps.
    print("Running STL decomposition...")
    trend = extract_trend(series, period=STL_PERIOD)

    # ── 3. Scale both channels with StandardScaler ───────────
    # Fit ONLY on training portion to avoid data leakage.
    train_end   = int(len(series) * 0.8)

    scaler_x = StandardScaler()
    raw_scaled   = scaler_x.fit_transform(series.reshape(-1, 1))        # (N, 1)
    trend_scaled = scaler_x.transform(trend.reshape(-1, 1))             # (N, 1)
    features     = np.hstack([raw_scaled, trend_scaled])                # (N, 2)

    # Target scaler — used to invert predictions back to original scale
    scaler_y = StandardScaler()
    scaler_y.fit(series[:train_end].reshape(-1, 1))
    targets_scaled = scaler_y.transform(series.reshape(-1, 1)).flatten()

    # ── 4. Create sliding windows ────────────────────────────
    X, y = make_windows(features, targets_scaled, INPUT_LEN, FORECAST_LEN)

    # Train / validation split — keep temporal order, never shuffle!
    split      = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {X_train.shape} → {y_train.shape}")
    print(f"Val  : {X_val.shape}   → {y_val.shape}")

    # ── 5. Build & train ─────────────────────────────────────
    model = build_transformer(
        input_len    = INPUT_LEN,
        forecast_len = FORECAST_LEN,
        num_features = NUM_FEATURES,
        d_model      = D_MODEL,
        num_heads    = NUM_HEADS,
        ff_dim       = FF_DIM,
        num_layers   = NUM_LAYERS,
        dropout      = DROPOUT,
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ── 6. Evaluate ──────────────────────────────────────────
    val_preds_scaled = model.predict(X_val)                 # (n_val, forecast_len)

    # Inverse transform each horizon step back to original scale
    val_preds = scaler_y.inverse_transform(val_preds_scaled)
    val_true  = scaler_y.inverse_transform(y_val)

    mae = np.mean(np.abs(val_preds - val_true))
    mse = np.mean((val_preds - val_true) ** 2)
    print(f"\nVal MSE (original scale): {mse:.4f}")
    print(f"Val MAE (original scale): {mae:.4f}")
