from dataclasses import dataclass
from typing import Optional

@dataclass
class RWKV5BlockConfigMap:
    """Configuration map for RWKV based models"""
    # Key properties for the block / model
    n_layer: int
    n_embed: int

    # ---
    # OPTIONAL PROPS
    #
    # Optional properties which can be derived
    # or can be overwritten by the user
    # ---

    # Channel mix / FFN block dimension size
    dim_ffn: Optional[int] = None,

    # Current layer_id of the block
    layer_id: Optional[int] = None

    # Device and Data type
    device: Optional[str] = None
    dtype: Optional[str] = None

    # ---
    # OPTIONAL PROPS FETCHER
    # ---

    def get_dim_ffn(self) -> int:
        '''
        Returns the dimension of feed forward network
        '''
        if self.dim_ffn is not None:
            return self.dim_ffn
        return int((self.n_embed * 3.5) // 32 * 32)
    
    def get_layer_id(self, fallback:int) -> int:
        '''
        Returns the layer id
        '''
        if self.layer_id is not None:
            return self.layer_id
        return fallback
    
    def get_device(self, fallback:str) -> str:
        '''
        Returns the device
        '''
        if self.device is not None:
            return self.device
        return fallback
    
    def get_dtype(self, fallback:str) -> str:
        '''
        Returns the dtype
        '''
        if self.dtype is not None:
            return self.dtype
        return fallback

def RWKV5BlockConfigMapNormalizer(config_map: any) -> RWKV5BlockConfigMap:
    '''
    Converts either maps, objs or RWKV5BlockConfigMap
    '''
    if isinstance(config_map, RWKV5BlockConfigMap):
        return config_map
    
    if isinstance(config_map, dict):
        return RWKV5BlockConfigMap(**config_map)
    