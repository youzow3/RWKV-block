from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

@dataclass
class GoldFinchBlockConfigMap:

    """Configuration map for GoldFinch hybrid based models"""
    # Key properties for the block / model
    num_hidden_layers: int
    hidden_size: int

    head_size: int = 64
    head_size_divisor: int = 8

    # Dropout rate, should only be used in training
    dropout_rate: float = 0.0

    # Implementation backend to use
    tmix_backend: str = "auto"

    # Block attention type
    # Its either x060 or x060b2 or gptalpha_goco
    att_type: str = "x060"

    # ---
    # OPTIONAL PROPS
    #
    # Optional properties which can be derived
    # or can be overwritten by the user
    # ---

    # Channel mix / FFN block dimension size
    hidden_size_ffn: Optional[int] = None
    hidden_size_att: Optional[int] = None

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

    def get_hidden_size_ffn(self) -> int:
        '''
        Returns the dimension of feed forward network
        '''
        if self.hidden_size_ffn is not None:
            hidden_size_ffn = self.hidden_size_ffn
        else:
            hidden_size = self.hidden_size
            assert hidden_size  % 32 == 0, f"hidden_size must be divisible by 32"
            hidden_size_ffn = (self.hidden_size * 3.5) // 32 * 32

        hidden_size_ffn = int(hidden_size_ffn)
        assert hidden_size_ffn % 32 == 0, f"hidden_size_att must be divisible by 32"
        return hidden_size_ffn
    
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
        if fallback is not None:
            return torch.device(fallback)
        return torch.get_default_device()
    
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
    
    def get_hidden_size_att(self) -> int:
        '''
        Returns the dimension of attention
        '''
        if self.hidden_size_att is not None:
            hidden_size_att = self.hidden_size_att
        else:
            hidden_size = self.hidden_size
            assert hidden_size  % 32 == 0, f"hidden_size must be divisible by 32"
            hidden_size_att = hidden_size
        assert hidden_size_att % 32 == 0, f"hidden_size_att must be divisible by 32 ({hidden_size_att})"
        return hidden_size_att
    
    def get_n_head(self) -> int:
        '''
        Returns the number of heads
        '''
        if self.n_head is not None:
            n_head = self.n_head
        else:
            hidden_size_att = self.get_hidden_size_att()
            n_head = self.get_hidden_size_att() // self.head_size
            assert hidden_size_att % n_head == 0 ,  f"hidden_size_att must be divisible by head_size ({self.head_size})"

        return n_head

    # ---
    # Duplicator & Normalizer
    # ---

    def new_block_config_map(self, **kwargs) -> 'GoldFinchBlockConfigMap':
        '''
        Returns a new config map with updated values
        '''

        new_dict = {}
        for key in GoldFinchBlockConfigMap.__dataclass_fields__:
            if key in self:
                new_dict[key] = self[key]
        new_dict.update(kwargs)

        return GoldFinchBlockConfigMap(**new_dict)

    @staticmethod
    def normalize(config_map: any) -> 'GoldFinchBlockConfigMap':
        '''
        Converts either maps, objs or GoldFinchBlockConfigMap
        '''
        if isinstance(config_map, GoldFinchBlockConfigMap):
            return config_map
        
        dict_obj = None
        if isinstance(config_map, dict):
            dict_obj = config_map
        elif hasattr(config_map, '__dict__'):
            dict_obj = config_map.__dict__
        
        if dict_obj is not None:
            # Filter for only valeus in GoldFinchBlockConfigMap
            new_dict = {}
            for key, value in dict_obj.items():
                if key in GoldFinchBlockConfigMap.__dataclass_fields__:
                    new_dict[key] = value
            return GoldFinchBlockConfigMap(**new_dict)

        raise ValueError(f"Unsupported config_map type: {type(config_map)}")