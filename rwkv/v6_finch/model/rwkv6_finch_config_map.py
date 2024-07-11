from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

from ..block.rwkv6_block_config_map import RWKV6BlockConfigMap

@dataclass
class RWKV6FinchConfigMap(RWKV6BlockConfigMap):
    # This is the world tokenizer size
    n_vocab: int = 65536 
    init_state_wkv: bool = False

    # State tuning flag
    perform_state_tuning: bool = False

    @staticmethod
    def normalize(config_map: any) -> 'RWKV6FinchConfigMap':
        '''
        Converts either maps, objs or RWKV6BlockConfigMap
        '''
        if isinstance(config_map, RWKV6FinchConfigMap):
            return config_map
        
        if isinstance(config_map, dict):
            return RWKV6FinchConfigMap(**config_map)

        if hasattr(config_map, '__dict__'):
            return RWKV6FinchConfigMap(**config_map.__dict__)
        
        raise ValueError(f"Unsupported config_map type: {type(config_map)}")

    @staticmethod
    def from_model_state_dict(state_dict: dict, **kwargs) -> 'RWKV6FinchConfigMap':
        '''
        Converts the state dict to the config map
        '''

        # Iterate and count the layers
        n_layer = 0
        for key in state_dict.keys():
            if key.startswith('blocks.'):
                idx = key.split('.')[1]
                n_layer = max(n_layer, int(idx)+1)

        # Enable wkv_state
        if 'init_state.0.wkv' in state_dict:
            kwargs['init_state_wkv'] = True
        
        # Initialize the config map, with the configured values
        return RWKV6FinchConfigMap(
            n_layer=n_layer,
            n_dim=state_dict['emb.weight'].shape[1],
            n_vocab=state_dict['emb.weight'].shape[0],
            # init_state_wkv=hasattr(state_dict, 'init_state.0.wkv'),

            n_head=state_dict['blocks.0.att.time_faaaa'].shape[0],
            head_size=state_dict['blocks.0.att.time_faaaa'].shape[1],

            n_dim_att=state_dict['blocks.0.att.key.weight'].shape[0],
            n_dim_ffn=state_dict['blocks.0.ffn.key.weight'].shape[0],

            **kwargs
        )
        