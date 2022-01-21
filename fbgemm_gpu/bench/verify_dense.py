import itertools
import logging
import math
import random
import statistics
import time
from typing import Callable, List, Optional, Tuple

import click
import numpy as np
import torch

haveAIBench = False
try:
    from aibench_observer.utils.observer import emitMetric

    haveAIBench = True
except Exception:
    haveAIBench = False

from split_table_batched_embeddings_benchmark import (
    generate_requests,
    benchmark_requests_refer,
)

from fbgemm_gpu.split_table_batched_embeddings_ops import (
    BoundsCheckMode,
    CacheAlgorithm,
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
    OptimType,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    PoolingMode,
)
from numpy.random import default_rng
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)


def verify(B, D, L, E, T):

    # The verification is done on cuda.
    device = torch.cuda.current_device()

    np.random.seed(42)
    torch.manual_seed(42)

    Ds = [D] * T

    ## FBGEMM GPU implementation of Table Batched Embedding.
    emb = DenseTableBatchedEmbeddingBagsCodegen([(E, d,) for d in Ds],)
    emb = emb.to(device)

    ## Extract the weights to check the correctness of reference implementation.
    emb_weights = emb.weights.reshape(T, E, D)

    ## Number of iterations is currently set to 1.
    iters = 1
    requests = generate_requests(iters, B, T, L, E,)

    ## Reference torch implementation.
    reference_result = benchmark_requests_refer(
        requests, T, B, L, E, D, "sum", False, pre_trained_weight=emb_weights
    )

    ## FBGEMM gpu result.
    for indices, offsets, per_sample_weights in requests:
        fbgpu_result = emb.forward(indices.long(), offsets.long(), per_sample_weights)

    return reference_result, fbgpu_result


## Play with numbers.
B = 2  # batch_size
D = 4  # embedding_dim
L = 3  # bag_size
E = 2  # num_embeddings
T = 2  # num_tables

reference_result, fbgpu_result = verify(B, D, L, E, T)
check_equal = torch.all(reference_result == fbgpu_result)

print(check_equal)  # tensor(True, device='cuda:x')
