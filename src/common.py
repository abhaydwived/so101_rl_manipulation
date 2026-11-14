"""
Shared environment setup utilities.

Call ``configure_env()`` at the top of any entry-point script (training,
evaluation, live-test) to redirect Windows-incompatible cache paths and
suppress TensorFlow import side-effects.
"""

from __future__ import annotations

import os
import sys
import tempfile


def configure_env() -> None:
    """
    Redirect Keras / Matplotlib caches to a safe temp directory and block
    accidental TensorFlow imports (which cause EPERM on Windows when the
    default cache paths are under %USERPROFILE%).

    Must be called **before** importing stable-baselines3 or matplotlib.
    """
    tmp = tempfile.gettempdir()
    os.environ.setdefault("KERAS_HOME",           os.path.join(tmp, "keras_cache"))
    os.environ.setdefault("MPLCONFIGDIR",         os.path.join(tmp, "mpl_cache"))
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    os.makedirs(os.environ["KERAS_HOME"],   exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    _write_keras_json(os.environ["KERAS_HOME"])

    # Block accidental tensorflow import (not needed for SB3-PPO CPU training)
    sys.modules.setdefault("tensorflow", None)   # type: ignore[assignment]


def _write_keras_json(keras_home: str) -> None:
    """Write a minimal valid keras.json to prevent Keras from complaining."""
    path = os.path.join(keras_home, "keras.json")
    try:
        with open(path, "w") as f:
            f.write("{}")
    except OSError:
        pass  # Read-only filesystem — ignore

