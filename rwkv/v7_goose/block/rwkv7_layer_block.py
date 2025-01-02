import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .rwkv7_block_config_map import RWKV7BlockConfigMap
from .rwkv7_channel_mix import RWKV7ChannelMix
from .rwkv7_time_mix import RWKV7TimeMix

class RWKV7LayerBlock(torch.nn.Module):
    '''
    layer block for RWKV V7
    '''

    def __init__(self, configMap: Union[RWKV7BlockConfigMap, any]):
        super().__init__()

        cMap:RWKV7BlockConfigMap = RWKV7BlockConfigMap.normalize(configMap)
        self.configMap = cMap

        # Get required props
        n_dim = cMap.n_dim
        layer_id = cMap.get_layer_id(-1)
        device = cMap.get_device('cpu')
        dtype = cMap.get_dtype('bfloat16')
        dropout_rate = cMap.dropout_rate

        # Validate the layer_id
        assert layer_id >= 0, f'layer_id must be >= 0, got {layer_id}'

        # Setup the layernorms, and mixes
        self.ln1 = nn.LayerNorm(n_dim, device=device, dtype=dtype)
        self.ln2 = nn.LayerNorm(n_dim, device=device, dtype=dtype)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_dim, device=device, dtype=dtype)
        else:
            self.ln0 = nn.Identity(device=device)

        # Setup the time and channel mix
        self.att = RWKV7TimeMix(configMap)
        self.ffn = RWKV7ChannelMix(configMap)

        # Setup droupout at block level
        if dropout_rate > 0.0:            
            self.drop0 = nn.Dropout(p = dropout_rate,device=device)
            self.drop1 = nn.Dropout(p = dropout_rate,device=device)
        else:
            self.drop0 = nn.Identity(device=device)
            self.drop1 = nn.Identity(device=device)
    
    def forward(
        self, x:torch.Tensor,
        last_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor], 
        v_first:torch.Tensor
        ) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Forward the block given the input tokens and the last state
        Last state is a tuple of the following
        - time mix shift state
        - time mix wkv state
        - channel mix shift state

        Returns a pair of the output embedding, v_first and the next state
        '''

        # # Ensure everything is in the right device
        # x = x.to(self.ln1.weight.device)
        # last_state = [ s.to(self.ln1.weight.device) for s in last_state ]

        x = self.ln0(x)

        # assert self.ln1(x) is not None
        # assert last_state.tmix_shift is not None
        # assert last_state.tmix_wkv is not None

        xx, tmix_shift, tmix_wkv, v_first = self.att(
            self.ln1(x),
            last_state[0], # tmix_shift,
            last_state[1], # tmix_wkv
            v_first
        )

        # x = x + att_out
        x = self.drop0(x + xx)

        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state[2] # cmix_shift,
        )

        # x = x + ffn_out
        x = self.drop1(x + ffn_out)
        
        return x, (tmix_shift, tmix_wkv, ffn_state), v_first
    
    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, 
        in_x:torch.Tensor, 
        in_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor],
        in_v_first:torch.Tensor,
        out_x:torch.Tensor, 
        out_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor],
        out_v_first:torch.Tensor
        ) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], tmp_state, out_v_first[:] = self.forward(in_x, in_state, in_v_first)
        out_state[0][:], out_state[1][:], out_state[2][:] = tmp_state
        return out_x, out_state, out_v_first

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x: torch.Tensor, in_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor], in_v_first:torch.Tensor) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Compiled varient of the forward function
        '''
        return self.forward(in_x, in_state, in_v_first)
    
    def load_from_model_state_dict(self, model_state_dict:dict, layer_id:int=-1, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, load the block weights accordingly
        '''
        if layer_id == -1:
            layer_id = self.configMap.get_layer_id(1)
            
        # Get the current state_dict
        current_state_dict = self.state_dict()

        # Iterate each parameter in the state_dict, and compare from the model
        for n in current_state_dict:
            model_key = f"blocks.{layer_id}.{n}"
            if model_key not in model_state_dict:
                continue

            # Copy the values from the state_dict
            try:
                current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
            except Exception as e:
                print(f"[ERROR] loading: {model_key} | model shape: {current_state_dict[n].shape} | weight shape: {model_state_dict[model_key].shape}")
                raise e