import torch
from torch import nn
from typing import Union
from .rwkv5_block_config_map import RWKV5BlockConfigMap

class RWKV5ChannelMix(torch.nn.Module):
    '''
    ChannelMix block for RWKV
    This is similar to transformer FFN block
    '''

    def __init__(self, configMap: Union[RWKV5BlockConfigMap, any]):
        super().__init__()

        cMap:RWKV5BlockConfigMap = RWKV5BlockConfigMap.normalize(configMap)
        self.configMap = cMap

        # Get required props
        n_dim = cMap.n_dim
        n_layer = cMap.n_layer

        # Get optional props
        n_dim_ffn = cMap.get_n_dim_ffn()
        layer_id = cMap.get_layer_id(0)
        device = cMap.get_device('cpu')
        dtype = cMap.get_dtype('bfloat16')

        # Build the various params
        # ---
        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_dim)
            for i in range(n_dim):
                ddd[0, 0, i] = i / n_dim
            self.time_maa_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0).to(device, dtype=dtype))
            self.time_maa_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0).to(device, dtype=dtype))

        self.key = nn.Linear(n_dim, n_dim_ffn, bias=False, device=device, dtype=dtype)
        self.receptance = nn.Linear(n_dim, n_dim, bias=False, device=device, dtype=dtype)
        self.value = nn.Linear(n_dim_ffn, n_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, last_state: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        '''
        Forwarding channel mix given the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming channel mix, shift states of the various batches [batch_size, state_size]
        
        Returns a pair 
        - Output embedding of shape [batch_size, seq_len, embedding_size]
        - Output channel mix, shift state of shape [batch_size, state_size]
        '''
        # last_state = last_state.to(self.key.weight.device)

        ##########
        ## x060
        ##########

        dxprev = torch.concat((last_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, x[:,-1]


    @torch.compile(mode="default", fullgraph=True)
    def forward_with_compile(self, in_x: torch.Tensor, in_state: torch.Tensor, out_x: torch.Tensor, out_state: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], out_state[:] = self.forward(in_x, in_state)
        return out_x, out_state

    def load_from_model_state_dict(self, state_dict: dict, layer_id:int, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the RWKV5ChannelMix model weights, using the layer_id
        '''
        # Copy the values from the state_dict
        self.time_maa_k.data.copy_(state_dict[f"blocks.{layer_id}.ffn.time_maa_k"], non_blocking=non_blocking)
        self.time_maa_r.data.copy_(state_dict[f"blocks.{layer_id}.ffn.time_maa_r"], non_blocking=non_blocking)
        self.key.weight.data.copy_(state_dict[f"blocks.{layer_id}.ffn.key.weight"], non_blocking=non_blocking)
        self.receptance.weight.data.copy_(state_dict[f"blocks.{layer_id}.ffn.receptance.weight"], non_blocking=non_blocking)
        self.value.weight.data.copy_(state_dict[f"blocks.{layer_id}.ffn.value.weight"], non_blocking=non_blocking)
        