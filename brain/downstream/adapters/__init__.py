"""Adapters that put published encoders and baselines behind the suite's encoder interface.

Each adapter subclasses ``brain.downstream.models.ExternalEncoder`` and is named by its ``configs/bench/models.yaml``
entry (``adapter: module:Class``). An adapter owns its model's preprocessing: it receives the stored volume (1 mm MNI
grid, 256 x 256 x 160 voxels as (X, Y, Z); CT in HU, MRI as a robust z-score), with ``needs_brain`` the stored brain
mask, and the volume's stored attributes (``ExternalEncoder.raw_intensity`` restores the MRI intensity scale; native
spacing and slice thickness; the registration affine back to the source scan), and returns a global embedding and
optionally tokens.
"""
