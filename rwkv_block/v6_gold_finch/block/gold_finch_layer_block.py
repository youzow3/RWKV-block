import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .gold_finch_block_config_map import GoldFinchBlockConfigMap
from ...v6_finch.block.rwkv6_channel_mix import RWKV6ChannelMix
from ...v6_finch.block.rwkv6_time_mix import RWKV6TimeMix
from ...v6_finch.block.rwkv6_time_mix_b2 import RWKV6TimeMixB2
from .gold_finch_gpt_alpha_goco import GoldFinchGPTAlphaGoCo

class GoldFinchLayerBlock(nn.Module):
    '''
    layer block for Gold Finch
    '''

    def __init__(self, configMap: Union[GoldFinchBlockConfigMap, any]):
        super().__init__()

        configMap:GoldFinchBlockConfigMap = GoldFinchBlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        layer_id = configMap.get_layer_id(-1)
        device = configMap.get_device('cpu')
        dtype = configMap.get_dtype('bfloat16')
        dropout_rate = configMap.dropout_rate

        att_type = configMap.att_type
        self.att_type = att_type

        # Validate the layer_id
        assert layer_id >= 0, f'layer_id must be >= 0, got {layer_id}'

        # Setup the layernorms, and mixes
        self.ln1 = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        else:
            self.ln0 = nn.Identity(device=device)

        # Setup the attention block
        if att_type == 'x060':
            self.att = RWKV6TimeMix(configMap)
        elif att_type == 'x060b2':
            self.att = RWKV6TimeMixB2(configMap)
        elif att_type == 'gptalpha_goco':
            self.att = GoldFinchGPTAlphaGoCo(configMap)
        else:
            raise ValueError(f'Invalid att_type {att_type}')

        # Setup the channelmix 
        self.ffn = RWKV6ChannelMix(configMap)

        # Setup droupout at block level
        if dropout_rate > 0.0:            
            self.drop0 = nn.Dropout(p = dropout_rate,device=device)
            self.drop1 = nn.Dropout(p = dropout_rate,device=device)
        else:
            self.drop0 = nn.Identity(device=device)
            self.drop1 = nn.Identity(device=device)
        
    
    def forward(
            self, 
            x:Tensor, 
            last_state: tuple[Tensor,Tensor,Tensor],
            x_original_cache:Tensor = None,
            kv_cache:Tensor = None
        ) -> tuple[Tensor,tuple[Tensor,Tensor,Tensor]]:
        '''
        Forward the block given the input tokens and the last state
        Last state is a tuple of the following
        - time mix shift state
        - time mix wkv state
        - channel mix shift state

        Returns a pair of the output embedding and the next state
        If any specific state is not required, it will be None in the tuple
        '''
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()

        # tmix shift and wkv values
        tmix_shift = None
        tmix_wkv = None

        # Run tmix according to the att_type
        if self.att_type == 'x060' or self.att_type == 'x060b2':
            dx, tmix_shift, tmix_wkv = self.att(
                self.ln1(x),
                last_state[0], # tmix_shift
                last_state[1] # tmix_wkv
            )
        elif self.att_type == 'gptalpha_goco':
            dx, tmix_shift = self.att(
                self.ln1(x),
                last_state[0], # tmix_shift
                x_original_cache,
                kv_cache
            )
        else:
            raise ValueError(f'Invalid att_type {self.att_type}')

        # x = x + dx
        x = self.drop0(x + dx)

        # Run the channel mix
        dx, ffn_state = self.ffn(
            self.ln2(x),
            last_state[2] # cmix_shift,
        )

        # x = x + dx
        x = self.drop1(x + dx)

        # return with block state
        return x, (tmix_shift, tmix_wkv, ffn_state)
    
    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, 
        in_x:Tensor, 
        in_state: tuple[Tensor,Tensor,Tensor],
        out_x:Tensor, 
        out_state: tuple[Tensor,Tensor,Tensor]
        ) -> tuple[Tensor,tuple[Tensor,Tensor,Tensor]]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], tmp_state = self.forward(in_x, in_state)
        out_state[0][:], out_state[1][:], out_state[2][:] = tmp_state
        return out_x, out_state

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(
            self, in_x: Tensor, in_state: tuple[Tensor,Tensor,Tensor],
            x_original_cache:Tensor = None, kv_cache:Tensor = None
        ) -> tuple[Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        '''
        return self.forward(in_x, in_state)
    
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
            current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
        