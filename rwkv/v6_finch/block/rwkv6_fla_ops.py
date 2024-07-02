import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

# FLA implementation of tmix
from fla.ops.rwkv6 import chunk_rwkv6

def RUN_RWKVx060_FLA(B, T, C, H, r, k, v, w, u, in_wkv_state):
    r = r.view(B,T,H,-1).transpose(1,2).float()
    k = k.view(B,T,H,-1).transpose(1,2).float()
    v = v.view(B,T,H,-1).transpose(1,2).float()
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2).float())
    o, out_wkv_state = RUN_RWKVx060_CHUNK(r, k, v, w, u=u.float(), scale=1, initial_state=in_wkv_state.float(), output_final_state=True)
    return o.bfloat16().transpose(1,2).reshape(B,T,C), out_wkv_state.bfloat16()

@torch.compiler.disable
def RUN_RWKVx060_CHUNK(r,k,v,w,u,scale,initial_state,output_final_state):
    return chunk_rwkv6(r, k, v, w, u=u, scale=scale, initial_state=initial_state, output_final_state=output_final_state)
