from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

from ..block.rwkv5_block_config_map import RWKV5BlockConfigMap

@dataclass
class RWKV5EagleConfigMap(RWKV5BlockConfigMap):
    # This is the world tokenizer size
    vocab_size: int = 65536 
    init_wkv_state: bool = False 

    @staticmethod
    def normalize(config_map: any) -> 'RWKV5EagleConfigMap':
        '''
        Converts either maps, objs or RWKV5BlockConfigMap
        '''
        if isinstance(config_map, RWKV5EagleConfigMap):
            return config_map
        
        if isinstance(config_map, dict):
            return RWKV5EagleConfigMap(**config_map)

        if hasattr(config_map, '__dict__'):
            return RWKV5EagleConfigMap(**config_map.__dict__)
        
        raise ValueError(f"Unsupported config_map type: {type(config_map)}")

    @staticmethod
    def from_model_state_dict(state_dict: dict, **kwargs) -> 'RWKV5EagleConfigMap':
        '''
        Converts the state dict to the config map
        '''

        # Iterate and count the layers
        num_hidden_layers = 0
        for key in state_dict.keys():
            if key.startswith('blocks.'):
                idx = key.split('.')[1]
                num_hidden_layers = max(num_hidden_layers, int(idx)+1)
        
        # Initialize the config map, with the configured values
        return RWKV5EagleConfigMap(
            num_hidden_layers=num_hidden_layers,
            hidden_size=state_dict['emb.weight'].shape[1],
            vocab_size=state_dict['emb.weight'].shape[0],
            init_wkv_state=hasattr(state_dict, 'init_state.0.wkv'),

            n_head=state_dict['blocks.0.att.time_decay'].shape[0],
            head_size=state_dict['blocks.0.att.time_decay'].shape[1],

            hidden_size_att=state_dict['blocks.0.att.key.weight'].shape[0],
            hidden_size_ffn=state_dict['blocks.0.ffn.key.weight'].shape[0],

            **kwargs
        )
        