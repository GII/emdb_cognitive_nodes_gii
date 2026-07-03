"""
Utilities for deterministic / genuinely-random number generation across the
cognitive architecture.

The architecture has several stochastic components (space learning, neural
network weight initialisation and shuffling, exploration policies, dreaming,
episodic-buffer sampling, ...). For debugging and reproducibility it is useful
to force every source of randomness to depend on a single, user-provided seed,
while making sure that when no seed is requested the run is genuinely random.

Each node receives its seed through the ``random_seed`` parameter, which the
commander propagates to every node via its ``global_params``. The seed is
resolved with :func:`core.utils.resolve_seed`, so a value of ``None`` or ``0``
yields a fresh time-based seed (never a fixed default).

This module centralises two helpers:

* :func:`get_rng` returns an isolated ``numpy.random.Generator`` seeded with the
  resolved seed. Prefer this over the global ``numpy.random`` / ``random`` state.
* :func:`set_global_seeds` seeds the *global* RNGs of the standard library,
  NumPy and (when importable) PyTorch, for third-party code whose randomness
  cannot be routed through an explicit generator (e.g. torch weight
  initialisation or ``DataLoader`` shuffling).
"""

from __future__ import annotations

import random as _py_random
from typing import Optional

import numpy as _np

from core.utils import resolve_seed


def get_rng(seed: Optional[int] = None) -> "_np.random.Generator":
    """Return a fresh, isolated NumPy random generator.

    The seed is resolved through :func:`core.utils.resolve_seed`, so ``None`` or
    ``0`` yields a fresh time-based (genuinely random) seed rather than a fixed
    default. Any other value is used deterministically.

    :param seed: Requested seed. ``None`` or ``0`` request a random run.
    :type seed: int or None
    :return: A seeded ``numpy.random.Generator`` instance.
    :rtype: numpy.random.Generator
    """
    return _np.random.default_rng(resolve_seed(seed))


def set_global_seeds(seed: Optional[int] = None) -> int:
    """Seed every *global* random number generator that may be in use.

    Seeds the Python ``random`` module, NumPy's legacy global RNG and, when the
    library is importable, PyTorch (including a request for deterministic cuDNN
    behaviour). The seed is resolved with :func:`core.utils.resolve_seed`, so
    ``None`` or ``0`` produces a fresh time-based seed instead of a fixed
    default. The resolved seed is returned so callers can log it.

    :param seed: Requested seed. ``None`` or ``0`` request a random run.
    :type seed: int or None
    :return: The concrete seed that was applied.
    :rtype: int
    """
    seed = resolve_seed(seed)

    _py_random.seed(seed)
    _np.random.seed(seed)

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

    return seed
