from typing import List, cast
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
NDArray = npt.NDArray[Any]
NDArrayFloat = npt.NDArray[np.float_]
NDArrayInt = npt.NDArray[np.int64]
NDArrays = List[NDArray]


N = 2**50 #50  #56
M = 2 ** 59 #59 #60
CR = 3.0 #3  

modulus = np.vectorize(lambda x : x % M)

def add_mod(x, y):
    return (x + y) % M


def _stochastic_round(arr: NDArrayFloat) -> NDArrayInt:
    ret: NDArrayInt = np.ceil(arr).astype(np.int64)
    rand_arr = np.random.rand(*ret.shape)
    ret[rand_arr < ret - arr] -= 1    
    return ret


def quantize(
    parameters: List[NDArrayFloat], clipping_range: float, target_range: np.int64
) -> List[NDArrayInt]:
    """Quantize float Numpy arrays to integer Numpy arrays."""
    quantized_list: List[NDArrayInt] = []
    quantizer = target_range / (2 * clipping_range)
    for arr in parameters:
        # Stochastic quantization
        pre_quantized = cast(
            NDArrayFloat,
            (np.clip(arr, -clipping_range, clipping_range) + clipping_range)
            * quantizer,
        )
        quantized = _stochastic_round(pre_quantized)
        quantized_list.append(quantized)
    return quantized_list


def dequantize(
    quantized_parameters: List[NDArrayInt],
    clipping_range: float,
    target_range: np.int64, ag_no: int,
) -> List[NDArrayFloat]:
    """Dequantize integer Numpy arrays to float Numpy arrays within range [-clipping_range, clipping_range]"""
    reverse_quantized_list: List[NDArrayFloat] = []
    quantizer = (2 * clipping_range) / target_range
    shift = -clipping_range*ag_no
    for arr in quantized_parameters:
        recon_arr = arr.view(np.ndarray).astype(np.float_)
        print('recon_arr', recon_arr)
        print('quantizer', quantizer)
        print('shift', shift)
        recon_arr = cast(NDArrayFloat, recon_arr * quantizer + shift)   
        reverse_quantized_list.append(recon_arr)
    return reverse_quantized_list

# Dequantize parameters to range [-clipping_range, clipping_range]
def dequantize_mean(
    quantized_parameters: List[NDArrayInt],
    clipping_range: float,
    target_range: np.int64, ag_no: int,
) -> List[NDArrayFloat]:
    """Dequantize integer Numpy arrays to float Numpy arrays."""
    reverse_quantized_list: List[NDArrayFloat] = []
    quantizer = (2 * clipping_range) / (target_range*ag_no)
    shift = -clipping_range
    for arr in quantized_parameters:
        recon_arr = arr.view(np.ndarray).astype(np.float_)
        recon_arr = cast(NDArrayFloat, recon_arr *quantizer + shift)   
        reverse_quantized_list.append(recon_arr)
    return reverse_quantized_list
