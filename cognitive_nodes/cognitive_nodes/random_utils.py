"""
Utilities for deterministic random number generation across the cognitive
architecture.

The cognitive architecture contains several stochastic components (space
learning, exploration policies, dummy activations, neural-network weight
initialisation, ...). For debugging and reproducibility it is often useful to
force every source of randomness to depend on a single, user-provided seed.

Each cognitive node receives its seed through the ``random_seed`` node
parameter. Set it to the same value in every node's configuration to make a
whole experiment reproducible. If ``random_seed`` is left undefined (``None``)
the previous, non-deterministic behaviour is preserved.

This module centralises two helpers:

* :func:`get_rng` returns an isolated ``numpy.random.Generator`` seeded with the
  given seed. Prefer this over the global ``numpy.random`` / ``random`` state so
  that different components do not interfere with each other's sequences.
* :func:`set_global_seeds` seeds the *global* RNGs of the standard library,
  NumPy and (if available) TensorFlow and PyTorch. Use it for third-party code
  whose randomness cannot be routed through an explicit generator (e.g. Keras
  weight initialisation or ``torch`` operations).
"""

from __future__ import annotations

import random as _py_random
from typing import Optional

import numpy as _np

from core.utils import resolve_seed


def get_rng(seed: Optional[int] = None) -> "_np.random.Generator":
    """Return a fresh, isolated NumPy random generator.

    The seed is resolved through :func:`core.utils.resolve_seed`, so a value of
    ``None`` or ``0`` yields a fresh time-based (genuinely random) seed rather
    than a fixed default. Any other value is used deterministically.

    :param seed: Requested seed. ``None`` or ``0`` request a random run.
    :type seed: int or None
    :return: A seeded ``numpy.random.Generator`` instance.
    :rtype: numpy.random.Generator
    """
    return _np.random.default_rng(resolve_seed(seed))


def set_global_seeds(seed: Optional[int] = None) -> None:
    """Seed every *global* random number generator that may be in use.

    This seeds the Python ``random`` module, NumPy's legacy global RNG and, when
    the libraries are importable, TensorFlow and PyTorch (including a request for
    deterministic cuDNN behaviour). Framework imports are optional: if a
    framework is not installed the corresponding seeding is silently skipped.

    The seed is resolved through :func:`core.utils.resolve_seed`, so a value of
    ``None`` or ``0`` produces a fresh time-based seed (random weight
    initialisation) instead of applying a fixed default.

    :param seed: Requested seed. ``None`` or ``0`` request a random run.
    :type seed: int or None
    """
    seed = resolve_seed(seed)

    # Standard library and NumPy global state.
    _py_random.seed(seed)
    _np.random.seed(seed)

    # TensorFlow (optional). ``set_random_seed`` also re-seeds python + numpy and
    # enables op-level determinism for Keras weight initialisation.
    try:
        import tensorflow as tf  # type: ignore

        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            tf.random.set_seed(seed)
    except Exception:
        pass

    # PyTorch (optional).
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass
