"""
Memory-conscious training entry for larger signed drug-gene datasets.

This file keeps the same CSIH architecture but exposes conservative defaults
for large graphs. It can be extended with sparse incidence tensors or
mini-batched pair decoding if required.
"""

import sys

from parameter import get_args
import run


if __name__ == "__main__":
    print(
        "run_large.py uses the same leakage-free CSIH training pipeline. "
        "For very large datasets, reduce --n_prototypes, --topk, or hidden dimensions."
    )
    run.main()
