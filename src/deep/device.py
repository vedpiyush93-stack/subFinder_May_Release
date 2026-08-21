"""Compute-device selection and reporting for the deep configs.

On Apple Silicon, TensorFlow reaches the GPU through the ``tensorflow-metal``
PluggableDevice (Apple's Metal Performance Shaders backend — "MPS"). When that
plugin is installed, TF registers a ``GPU:0`` device and places ops on it
automatically; there is no flag to switch on. What this module adds is making
that *explicit and recorded*, so a run states which device produced it instead
of leaving it to be inferred.

Measured on an M4 Max (824x30x300 inputs, one outer-training fold, 60 epochs):

    architecture   Metal GPU     CPU     speedup
    Trans           451 ms    2447 ms      5.4x
    JustAttn         82 ms     104 ms      1.3x
    LSTMattn        122 ms     129 ms      1.05x
    LSTM            118 ms     103 ms      0.87x

The transformer is matmul-bound and gains a great deal. The LSTM family does
not, and that is not a configuration fault: with 824 training rows and a batch
size of 1024 there is exactly one batch per epoch, so those fits are bound by
per-epoch overhead rather than by arithmetic. No backend removes that.
"""
from __future__ import annotations
import os


def configure_device(prefer: str = "auto") -> dict:
    """Select the compute device and return a description of what is active.

    prefer: "auto" (GPU when available), "gpu" (require it), "cpu" (force CPU).
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if prefer == "cpu":
        tf.config.set_visible_devices([], "GPU")
        gpus = []
    elif prefer == "gpu" and not gpus:
        raise RuntimeError(
            "no GPU visible to TensorFlow. On Apple Silicon install the Metal "
            "plugin: pip install 'tensorflow==2.18.0' 'tensorflow-metal==1.2.0'")

    plugin = None
    if gpus:
        try:
            import importlib.metadata as md
            plugin = f"tensorflow-metal {md.version('tensorflow-metal')}"
        except Exception:
            plugin = "pluggable GPU device"

    return {
        "device": "gpu" if gpus else "cpu",
        "backend": "metal (Apple MPS)" if (gpus and plugin and "metal" in plugin) else
                   ("gpu" if gpus else "cpu"),
        "plugin": plugin,
        "tensorflow": tf.__version__,
        "n_gpus": len(gpus),
    }


def device_summary(info: dict) -> str:
    if info["device"] == "gpu":
        return (f"GPU via {info['backend']} [{info['plugin']}] — "
                f"tensorflow {info['tensorflow']}")
    return f"CPU only — tensorflow {info['tensorflow']} (no GPU device visible)"
