"""Temperature scaling — calibrate a deployed OvR(ExtraTrees) classifier."""
from .temperature import (fit_temperature, fit_temperature_inner_cv,
                          apply_temperature, CalibratedClassifier)
__all__ = ["fit_temperature", "fit_temperature_inner_cv",
           "apply_temperature", "CalibratedClassifier"]
