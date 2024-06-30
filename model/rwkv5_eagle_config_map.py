from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

from ..block.v5_eagle.rwkv5_block_config_map import RWKV5BlockConfigMap

@dataclass
class RWKV5EagleConfigMap(RWKV5BlockConfigMap):

    n_vocab: int
    init_state_wkv: bool = False

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
