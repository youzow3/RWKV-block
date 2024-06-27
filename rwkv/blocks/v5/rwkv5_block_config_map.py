from dataclasses import dataclass
from typing import Optional

@dataclass
class RWKV5BlockConfigMap:
    """Configuration map for RWKV based models"""
    # Key properties for the block / model
    n_layer: int
    n_embed: int

    head_size: int = 64
    head_size_divisor: int = 8

    # ---
    # OPTIONAL PROPS
    #
    # Optional properties which can be derived
    # or can be overwritten by the user
    # ---

    # Channel mix / FFN block dimension size
    dim_ffn: Optional[int] = None,
    dim_att: Optional[int] = None

    # Current layer_id of the block
    layer_id: Optional[int] = None

    # number of heads
    n_head: Optional[int] = None

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
            dim_ffn = self.dim_ffn
        else:
            n_embed = self.n_embed
            assert n_embed  % 32 == 0, f"n_embed must be divisible by 32"
            dim_ffn = int((self.n_embed * 3.5) // 32 * 32)
        assert dim_ffn % 32 == 0, f"dim_att must be divisible by 32"
        return dim_ffn
    
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
    
    def get_dim_att(self) -> int:
        '''
        Returns the dimension of attention
        '''
        if self.dim_att is not None:
            dim_att = self.dim_att
        else:
            n_embed = self.n_embed
            assert n_embed  % 32 == 0, f"n_embed must be divisible by 32"
            dim_att = n_embed
        assert dim_att % 32 == 0, f"dim_att must be divisible by 32 ({dim_att})"
        return dim_att
    
    def get_n_head(self) -> int:
        '''
        Returns the number of heads
        '''
        if self.n_head is not None:
            n_head = self.n_head
        else:
            dim_att = self.get_dim_att()
            n_head = self.get_dim_att() // self.head_size
            assert dim_att % n_head == 0 ,  f"dim_att must be divisible by head_size ({self.head_size})"

        return n_head


def RWKV5BlockConfigMapNormalizer(config_map: any) -> RWKV5BlockConfigMap:
    '''
    Converts either maps, objs or RWKV5BlockConfigMap
    '''
    if isinstance(config_map, RWKV5BlockConfigMap):
        return config_map
    
    if isinstance(config_map, dict):
        return RWKV5BlockConfigMap(**config_map)

    if hasattr(config_map, '__dict__'):
        return RWKV5BlockConfigMap(**config_map.__dict__)
    
    raise ValueError(f"Unsupported config_map type: {type(config_map)}")