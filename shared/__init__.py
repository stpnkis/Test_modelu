"""
shared — Single source of truth for the anomaly-detection benchmark.

Every model container imports from this package (mounted via Docker volume
and added to PYTHONPATH).  No benchmark logic lives inside individual
model directories.
"""
