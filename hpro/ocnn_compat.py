"""Compatibility shim for ocnn under triton >= 3.3 -- import before ``ocnn``.

ocnn 2.3.2's Triton convolution kernels fail to compile against the triton 3.3
that ships with torch 2.7::

    File ".../ocnn/nn/kernels/conv_fwd_implicit_gemm.py", line 77
        bias_block = tl.load(bias + offset_co)
    AttributeError("'NoneType' object has no attribute 'type'")

The kernel guards that load with ``if bias is not None``, but ``bias`` is a
plain kernel argument, not a ``tl.constexpr``. Newer triton therefore does not
fold the branch at compile time and traces its body with ``bias=None``. The NVPS
network's convolutions are bias-free, so this fires on every forward pass and
makes the whole NVPS backbone unusable.

ocnn ships a supported fallback -- a pure-PyTorch convolution path selected by
``OCNN_DISABLE_TRITON=1``, read once at ocnn import time. That path is also what
the NVPS timings in ``RESEARCH_PLAN.md`` §3/§4 were measured on (~30 s per cloud
at N=3000, against the paper's reported 75 ms), so enabling it changes no
published number here.

We use ``setdefault`` so an explicit ``OCNN_DISABLE_TRITON=0`` still opts back
in (e.g. after an ocnn upgrade that fixes the kernel).

Usage -- must precede any ``import ocnn``, directly or transitively::

    import ocnn_compat  # noqa: F401
    import ocnn
"""

import os

os.environ.setdefault("OCNN_DISABLE_TRITON", "1")
