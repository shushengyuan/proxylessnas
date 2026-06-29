"""Minimal local stub for Proxyless search imports.

The search path imports CRF helpers unconditionally, but search cost runs use
postprocess=none and never execute DenseCRF. This stub prevents an optional
inference-only dependency from blocking architecture search reproduction.
"""
