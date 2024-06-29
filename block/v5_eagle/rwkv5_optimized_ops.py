# 
# Random collection of optimized operations used within the RWKV v5 implementation.
# ---

import torch
import torch.nn.functional as F
from torch import Tensor

def modified_lerp(start_mul, start, weight):
	'''
	Modified LERP operation, which is used to compute the 
	time mixing and channel mixing components. 

	This is slightly different from the standard LERP operation
	due to the presence of the start_mul parameter.
	'''
	return start_mul * start + weight * (1 - start)
