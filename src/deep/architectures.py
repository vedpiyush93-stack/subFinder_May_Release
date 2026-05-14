"""Four deep architectures from the paper, Keras 3.

Architectures (used as keys throughout the codebase):
  LSTM       single-layer LSTM(128) + dropout(0.6) + dense(12, softmax)
  LSTMattn   LSTM(128) + soft-attention + dense(12, softmax)
  JustAttn   non-recurrent attention block (bag-of-vectors → attn → dense)
  Trans      4-block transformer encoder (4 heads × head_size=128 × ff=512, dropout=0.5)

All four take a (batch, seq_len, embed_dim) tensor of stacked token vectors and
output (batch, 12) class probabilities via softmax. Training is paper-verbatim:
Adam(1e-4), EarlyStopping(patience=30, restore_best=True), max_epochs=2000,
75/25 stratified val split (random_state=42). Batch sizes deviate from paper's
256 to fit M4 Max throughput: 1024 for LSTM family, 4096 for transformer.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import keras
    from keras import layers
except ImportError:  # pragma: no cover
    keras = None
    layers = None


def _attention_block(x, head_size: int = 128, num_heads: int = 4, ff_dim: int = 512, dropout: float = 0.5):
    """One transformer-encoder block (MHA + residual + FFN + residual)."""
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    h = layers.Dropout(dropout)(h)
    h = layers.LayerNormalization(epsilon=1e-6)(x + h)
    f = layers.Conv1D(ff_dim, 1, activation="relu")(h)
    f = layers.Dropout(dropout)(f)
    f = layers.Conv1D(x.shape[-1], 1)(f)
    return layers.LayerNormalization(epsilon=1e-6)(h + f)


def _lstm(input_shape, n_classes=12, attention=False):
    inp = keras.Input(shape=input_shape)
    h = layers.LSTM(128, return_sequences=attention, dropout=0.6)(inp)
    if attention:
        # Bahdanau-style additive attention pooled to a single vector
        a = layers.Dense(1, activation="tanh")(h)
        a = layers.Flatten()(a); a = layers.Activation("softmax")(a)
        a = layers.RepeatVector(128)(a); a = layers.Permute([2,1])(a)
        h = layers.Multiply()([h, a])
        h = layers.Lambda(lambda z: keras.ops.sum(z, axis=1))(h)
    h = layers.Dense(64, activation="relu")(h); h = layers.Dropout(0.6)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["acc"])
    return m


def _just_attn(input_shape, n_classes=12):
    inp = keras.Input(shape=input_shape)
    h = _attention_block(inp, dropout=0.4)
    h = layers.GlobalAveragePooling1D()(h)
    h = layers.Dense(64, activation="relu")(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["acc"])
    return m


def _transformer(input_shape, n_classes=12, n_blocks=4):
    inp = keras.Input(shape=input_shape)
    h = inp
    for _ in range(n_blocks): h = _attention_block(h, dropout=0.5)
    h = layers.GlobalAveragePooling1D()(h)
    h = layers.Dense(64, activation="relu")(h); h = layers.Dropout(0.5)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["acc"])
    return m


DL_ARCHITECTURES = {
    "LSTM":     lambda shp: _lstm(shp, attention=False),
    "LSTMattn": lambda shp: _lstm(shp, attention=True),
    "JustAttn": _just_attn,
    "Trans":    _transformer,
}
DL_BATCH = {"LSTM": 1024, "LSTMattn": 1024, "JustAttn": 1024, "Trans": 4096}


def build_dl(name: str, input_shape: tuple):
    """Return a freshly-instantiated DL classifier given input shape (seq_len, embed_dim)."""
    if name not in DL_ARCHITECTURES:
        raise KeyError(f"unknown DL arch {name!r}; choose from {list(DL_ARCHITECTURES)}")
    return DL_ARCHITECTURES[name](input_shape)


def train_dl(model, X, y_onehot, name: str, max_epochs: int = 2000, verbose: int = 0):
    """Train a DL model with paper-verbatim 75/25 stratified val split + EarlyStopping(p=30)."""
    y_idx = y_onehot.argmax(axis=1)
    tr, val = train_test_split(np.arange(len(X)), test_size=0.25,
                                random_state=42, stratify=y_idx)
    cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    h = model.fit(X[tr], y_onehot[tr], validation_data=(X[val], y_onehot[val]),
                  batch_size=DL_BATCH[name], epochs=max_epochs, callbacks=[cb], verbose=verbose)
    return model, h.history
