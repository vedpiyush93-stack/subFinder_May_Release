"""Deep classifier architectures (paper-verbatim) and device selection."""
from .architectures import DL_ARCHITECTURES, build_dl, train_dl
from .device import configure_device, device_summary
__all__ = ["DL_ARCHITECTURES", "build_dl", "train_dl",
           "configure_device", "device_summary"]
