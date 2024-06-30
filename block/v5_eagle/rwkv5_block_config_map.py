from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

@dataclass
class RWKV5BlockConfigMap:
    """Configuration map for RWKV based models"""
    # Key properties for the block / model
    n_layer: int
    n_dim: int

    head_size: int = 64
    head_size_divisor: int = 8

    # Dropout rate, should only be used in training
    dropout_rate: float = 0.0

    # ---
    # OPTIONAL PROPS
    #
    # Optional properties which can be derived
    # or can be overwritten by the user
    # ---

    # Channel mix / FFN block dimension size
    n_dim_ffn: Optional[int] = None
    n_dim_att: Optional[int] = None

    # Current layer_id of the block
    layer_id: Optional[int] = None

    # number of heads
    n_head: Optional[int] = None

    # Device and Data type
    device: Union[torch.device, str, None] = None
    dtype: Union[torch.dtype, str, None] = None

    # ---
    # OPTIONAL PROPS FETCHER
    # ---

    def get_n_dim_ffn(self) -> int:
        '''
        Returns the dimension of feed forward network
        '''
        if self.n_dim_ffn is not None:
            n_dim_ffn = self.n_dim_ffn
        else:
            n_dim = self.n_dim
            assert n_dim  % 32 == 0, f"n_dim must be divisible by 32"
            n_dim_ffn = (self.n_dim * 3.5) // 32 * 32

        n_dim_ffn = int(n_dim_ffn)
        assert n_dim_ffn % 32 == 0, f"n_dim_att must be divisible by 32"
        return n_dim_ffn
    
    def get_layer_id(self, fallback:int) -> int:
        '''
        Returns the layer id
        '''
        if self.layer_id is not None:
            return self.layer_id
        return fallback
    
    def get_device(self, fallback:str) -> torch.device:
        '''
        Returns the device
        '''
        if self.device is not None:
            return torch.device(self.device)
        return torch.device(fallback)
    
    def get_dtype(self, fallback:str) -> torch.dtype:
        '''
        Returns the dtype
        '''
        if self.dtype is not None:
            key = self.dtype
        else:
            key = fallback

        # if dtype is already torch.dtype
        if isinstance(key, torch.dtype):
            return key
        
        # Get and Check if the dtype is instance of torch.dtype
        ret = getattr(torch, key) 
        assert isinstance(ret, torch.dtype), f"Invalid dtype: {self.dtype}"
        return ret
    
    def get_n_dim_att(self) -> int:
        '''
        Returns the dimension of attention
        '''
        if self.n_dim_att is not None:
            n_dim_att = self.n_dim_att
        else:
            n_dim = self.n_dim
            assert n_dim  % 32 == 0, f"n_dim must be divisible by 32"
            n_dim_att = n_dim
        assert n_dim_att % 32 == 0, f"n_dim_att must be divisible by 32 ({n_dim_att})"
        return n_dim_att
    
    def get_n_head(self) -> int:
        '''
        Returns the number of heads
        '''
        if self.n_head is not None:
            n_head = self.n_head
        else:
            n_dim_att = self.get_n_dim_att()
            n_head = self.get_n_dim_att() // self.head_size
            assert n_dim_att % n_head == 0 ,  f"n_dim_att must be divisible by head_size ({self.head_size})"

        return n_head

    # ---
    # Duplicator & Normalizer
    # ---

    def get_new_config_map(self, **kwargs) -> 'RWKV5BlockConfigMap':
        '''
        Returns a new config map with updated values
        '''
        return RWKV5BlockConfigMap(
            **self.__dict__,
            **kwargs
        )

    @staticmethod
    def normalize(config_map: any) -> 'RWKV5BlockConfigMap':
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
