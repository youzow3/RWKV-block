import torch
from torch import nn
from torch import Tensor
from typing import Union
from .rwkv5_block_config_map import RWKV5BlockConfigMap

class RWKV5ChannelMix(torch.nn.Module):
    '''
    ChannelMix block for RWKV
    This is similar to transformer FFN block
    '''

    def __init__(self, configMap: Union[RWKV5BlockConfigMap, any]):
        super().__init__()

        configMap:RWKV5BlockConfigMap = RWKV5BlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        num_hidden_layers = configMap.num_hidden_layers

        # Get optional props
        hidden_size_ffn = configMap.get_hidden_size_ffn()
        layer_id = configMap.get_layer_id(0)
        device = configMap.get_device('cpu')
        dtype = configMap.get_dtype('bfloat16')

        # Build the various params
        # ---
        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, hidden_size, device=device, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(hidden_size, hidden_size_ffn, bias=False, device=device, dtype=dtype)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.value = nn.Linear(hidden_size_ffn, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor, last_state: Tensor) -> tuple[Tensor,Tensor]:
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

        # 7B 4090
        # - no compile:      0.3243246078491211 ms
        # - default compile: 0.3113429546356201 ms
        # - max auto tune:   0.30689454078674316 ms
        # ---
        xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        kv = self.value( torch.relu( self.key(xk) ) ** 2 )
        return (torch.sigmoid(self.receptance(xr)) * kv,
                (x[:, -1]))

        # 7B 4090 
        # - no compile:      0.3181338310241699 ms
        # - default compile: 0.3143901824951172 ms
        # - max auto tune:   0.3115804195404053 ms
        # ---
        # xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]), dim=1)
        # xk,xr = torch.lerp(xx, x, torch.stack((self.time_mix_k,self.time_mix_r)))
        # kv = self.value( torch.relu( self.key(xk) ) ** 2 )
        # return torch.sigmoid(self.receptance(xr)) * kv, x[:,-1]

    @torch.compile(mode="default", fullgraph=True)
    def forward_with_default_compile(self, in_x: Tensor, in_state: Tensor, out_x: Tensor, out_state: Tensor) -> tuple[Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], out_state[:] = self.forward(in_x, in_state)
        return out_x, out_state

    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def forward_with_reduce_compile(self, x: Tensor, last_state: Tensor) -> tuple[Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no input tensor being modified. 
        Useful for reduce-overhead compile mode
        '''
        return self.forward(x, last_state)
    
    def load_from_model_state_dict(self, model_state_dict: dict, layer_id:int, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the the current module weights, using the layer_id
        '''
        # Get the current state_dict
        current_state_dict = self.state_dict()

        # Iterate each parameter in the state_dict, and compare from the model
        for n in current_state_dict:
            model_key = f"blocks.{layer_id}.ffn.{n}"
            if model_key not in model_state_dict:
                continue

            # Copy the values from the state_dict
            current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
        