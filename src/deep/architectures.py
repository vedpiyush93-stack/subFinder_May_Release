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
import os
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import keras
    from keras import layers
except ImportError:  # pragma: no cover
    keras = None
    layers = None


def _rep_seed() -> int:
    return int(os.environ.get("REPRO_REP_SEED", "42"))


def _set_keras_seed():
    """Seed all RNGs Keras / TF / NumPy / Python touch during model init + fit."""
    s = _rep_seed()
    if keras is None:
        return
    import random, tensorflow as tf
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    try: keras.utils.set_random_seed(s)
    except Exception: pass


def _attention_block(x, head_size: int = 128, num_heads: int = 4, ff_dim: int = 512, dropout: float = 0.5):
    """One transformer-encoder block (MHA + residual + FFN + residual).

    Generic block — used as-is by JustAttn (with dropout=0.4) and by the
    legacy Trans path. The 'apples-to-apples paper-faithful' Trans block
    is in _attention_block_paper below, which mirrors the source repo's
    ordering exactly (LN BEFORE residual, Dense FFN, no extra projection).
    """
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    h = layers.Dropout(dropout)(h)
    h = layers.LayerNormalization(epsilon=1e-6)(x + h)
    f = layers.Conv1D(ff_dim, 1, activation="relu")(h)
    f = layers.Dropout(dropout)(f)
    f = layers.Conv1D(x.shape[-1], 1)(f)
    return layers.LayerNormalization(epsilon=1e-6)(h + f)


def _attention_block_paper(x, head_size: int = 128, num_heads: int = 4, ff_dim: int = 512, dropout: float = 0.5):
    """Paper-verbatim transformer-encoder block (matches subFinder/Codes/
    Model_architectures_tran.py:transformer_encoder exactly):

        h = MHA(x, x)
        h = Dropout(h)
        h = LN(h)            ← LN BEFORE the residual sum
        res = h + x

        f = Dense(ff_dim, relu)(res)
        f = Dropout(f)
        f = Dense(emb_dim)(f)
        f = LN(f)            ← LN BEFORE the residual sum
        return f + res
    """
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    h = layers.Dropout(dropout)(h)
    h = layers.LayerNormalization(epsilon=1e-6)(h)
    res = h + x

    f = layers.Dense(ff_dim, activation="relu")(res)
    f = layers.Dropout(dropout)(f)
    f = layers.Dense(x.shape[-1])(f)
    f = layers.LayerNormalization(epsilon=1e-6)(f)
    return f + res


def _bahdanau_attn(seq, emb_dim, drop_inner: float):
    """Source-paper Bahdanau-style additive attention (used by LSTMattn + JustAttn).

    Mirrors subFinder/Codes/Model_architectures_tran.py:
        x_a = Dense(emb_dim//2, tanh, glorot_uniform)(seq)
        x_a = Dropout(drop_inner)(x_a)
        x_a = Dense(1, linear, glorot_uniform)(x_a)
        x_a = Flatten()(x_a)
        att = Activation('softmax')(x_a)
        att = RepeatVector(emb_dim)(att); att = Permute([2,1])(att)
        out = Multiply()([seq, att])
        out = Lambda(sum over time)(out)
    """
    a = layers.Dense(emb_dim // 2, kernel_initializer="glorot_uniform",
                      activation="tanh", name=None)(seq)
    a = layers.Dropout(drop_inner)(a)
    a = layers.Dense(1, kernel_initializer="glorot_uniform", activation="linear")(a)
    a = layers.Flatten()(a)
    a = layers.Activation("softmax")(a)
    a = layers.RepeatVector(emb_dim)(a)
    a = layers.Permute([2, 1])(a)
    pooled = layers.Multiply()([seq, a])
    pooled = layers.Lambda(lambda z: keras.ops.sum(z, axis=1))(pooled)
    return pooled


# ── Legacy non-paper LSTM kept for backwards compat / callers that import it ──
def _lstm(input_shape, n_classes=12, attention=False):
    inp = keras.Input(shape=input_shape)
    h = layers.LSTM(128, return_sequences=attention, dropout=0.6)(inp)
    if attention:
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


# ── Legacy non-paper JustAttn (transformer-style) kept for backwards compat ──
def _just_attn(input_shape, n_classes=12):
    inp = keras.Input(shape=input_shape)
    h = _attention_block(inp, dropout=0.4)
    h = layers.GlobalAveragePooling1D()(h)
    h = layers.Dense(64, activation="relu")(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["acc"])
    return m


# ─────────────────────────────────────────────────────────────────────────────
# PAPER-FAITHFUL DL ARCHS — match subFinder/Codes/Model_architectures_tran.py
# Two intentional deviations kept from May_Release:
#   * batch_size : 4096 for Trans, 1024 for LSTM family (vs paper 256) — set in DL_BATCH
#   * embedding  : EXTERNAL pre-vectorized input (vs in-model TextVectorization+Embedding)
# Other deviation: source uses SparseCCE(from_logits=True) + Dense(n, no_softmax).
# We keep softmax in the last Dense + SparseCCE(from_logits=False). Mathematically
# identical training (same gradients); the difference is purely numerical-stability
# ordering in the loss op. We do this so model.predict() returns proper probs that
# downstream calibration / ECE / inference code consumes unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def _lstm_paper(input_shape, n_classes=12):
    """LSTM source: simple_lstm — single LSTM(100, dropout=0.6) + Dropout(0.6) head.
    Adam(1e-3), SparseCCE."""
    inp = keras.Input(shape=input_shape)
    h = layers.LSTM(100, dropout=0.6)(inp)
    h = layers.Dropout(0.6)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return m


def _lstmattn_paper(input_shape, n_classes=12):
    """LSTMattn source: attention_lstm_model — LSTM(100, dropout=0.5, return_sequences=True)
    + Bahdanau attention (tanh+drop+linear+softmax) + Dropout(0.65) head.
    Adam(1e-3), SparseCCE."""
    inp = keras.Input(shape=input_shape)
    seq = layers.LSTM(100, return_sequences=True, dropout=0.5)(inp)
    pooled = _bahdanau_attn(seq, emb_dim=100, drop_inner=0.5)
    h = layers.Dropout(0.65)(pooled)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return m


def _just_attn_paper(input_shape, n_classes=12):
    """JustAttn source: non_recurrent_attention_model — Bahdanau attention applied
    DIRECTLY to the (seq_len, emb_dim) input (no LSTM). Dropout(0.65) head.
    Adam(1e-3), SparseCCE."""
    inp = keras.Input(shape=input_shape)
    emb_dim = input_shape[-1]
    pooled = _bahdanau_attn(inp, emb_dim=emb_dim, drop_inner=0.5)
    h = layers.Dropout(0.65)(pooled)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return m


def _transformer(input_shape, n_classes=12, n_blocks=4):
    """Paper-verbatim transformer (matches subFinder/Codes/Model_architectures_tran.py
    apples-to-apples) EXCEPT for two intentional differences kept from May_Release:
      * batch_size = 4096 (vs paper 256) — set in DL_BATCH for M4 Max throughput
      * embedding layer is EXTERNAL (input is pre-vectorized (seq_len, embed_dim))
        instead of baked-in TextVectorization+Embedding inside the model

    Everything else matches source exactly:
      - 4 transformer-encoder blocks via _attention_block_paper (LN BEFORE residual,
        Dense FFN, no extra Dense head before the classifier)
      - GlobalAveragePooling1D → Dropout(0.5) → Dense(n_classes, softmax)
      - Adam(lr=1e-4)
      - SparseCategoricalCrossentropy(from_logits=False) loss + metrics=['accuracy']
        ↳ train_dl detects 'Trans' name + passes int labels accordingly
    """
    inp = keras.Input(shape=input_shape)
    h = inp
    for _ in range(n_blocks):
        h = _attention_block_paper(h, dropout=0.5)
    h = layers.GlobalAveragePooling1D()(h)
    h = layers.Dropout(0.5)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return m


DL_ARCHITECTURES = {
    "LSTM":     _lstm_paper,
    "LSTMattn": _lstmattn_paper,
    "JustAttn": _just_attn_paper,
    "Trans":    _transformer,
}
DL_BATCH = {"LSTM": 1024, "LSTMattn": 1024, "JustAttn": 1024, "Trans": 4096}


def build_dl(name: str, input_shape: tuple):
    """Return a freshly-instantiated DL classifier given input shape (seq_len, embed_dim)."""
    if name not in DL_ARCHITECTURES:
        raise KeyError(f"unknown DL arch {name!r}; choose from {list(DL_ARCHITECTURES)}")
    _set_keras_seed()
    return DL_ARCHITECTURES[name](input_shape)


def train_dl(model, X, y_onehot, name: str, max_epochs: int = 2000, verbose: int = 0):
    """Train a DL model with 75/25 stratified val split + EarlyStopping(p=30).

    The val split here MUST match the val split that was excluded from the
    embedding training corpus (in src.splits.rskf_splits), otherwise val rows
    leak through the embedding's training set and inflate val accuracy.

    rskf_splits uses random_state=42 for the inner train_test_split. We mirror
    that exactly here regardless of REPRO_REP_SEED, so val is the SAME 206 rows
    that the embedding excluded. REPRO_REP_SEED still controls model init
    (seeded via _set_keras_seed in build_dl) — that's the variance we want
    across reps. Data partition stays deterministic.

    The naturally-rotating "k-fold flavor" the val partition gives us comes
    from the OUTER 5x5 RSKF: each of the 25 outer folds has a different
    outer_train, so the inner 75/25 split lands on different actual rows even
    though random_state is fixed. Over the 25 outer folds every row gets a turn
    being in val for at least one fold.

    Falls back to a random (non-stratified) 75/25 split if any class has <2
    members in the input — happens for very small classes in some outer folds.
    """
    y_idx = y_onehot.argmax(axis=1)
    val_rs = 42  # MUST match src.splits.rskf_splits' inner random_state
    try:
        tr, val = train_test_split(np.arange(len(X)), test_size=0.25,
                                    random_state=val_rs, stratify=y_idx)
    except ValueError:
        # Class with <2 members — drop stratification, keep same random_state.
        tr, val = train_test_split(np.arange(len(X)), test_size=0.25, random_state=val_rs)
    cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    # All 4 DL archs are now paper-faithful → SparseCategoricalCrossentropy → int labels.
    y_tr_target  = y_idx[tr].astype(np.int32)
    y_val_target = y_idx[val].astype(np.int32)
    h = model.fit(X[tr], y_tr_target, validation_data=(X[val], y_val_target),
                  batch_size=DL_BATCH[name], epochs=max_epochs, callbacks=[cb], verbose=verbose)
    return model, h.history
