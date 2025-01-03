import torch
from torch import nn
from torch import Tensor
from typing import Union

from .rwkv5_eagle_config_map import RWKV5EagleConfigMap
from ..block.rwkv5_layer_block import RWKV5LayerBlock

class RWKV5EagleModel(nn.Module):
    '''
    RWKV5 Eagle Model architecture
    Simplified implementation
    '''

    def __init__(self, config: Union[RWKV5EagleConfigMap, any]):
        super().__init__()

        configMap:RWKV5EagleConfigMap = RWKV5EagleConfigMap.normalize(config)
        self.configMap = configMap

        # Get the required prop
        n_layer = configMap.n_layer
        n_vocab = configMap.n_vocab
        device = configMap.get_device('cpu')
        dtype = configMap.get_dtype('bfloat16')
        n_dim = configMap.n_dim
        
        # Embedding layer
        self.emb = nn.Embedding(n_vocab, n_dim, device=device, dtype=dtype)

        # main block layers
        blockList = [None]*n_layer
        for i in range(n_layer):
            blockList[i] = RWKV5LayerBlock(configMap.new_block_config_map(layer_id=i))
        self.blocks = nn.ModuleList(blockList)

        # ln_out and head
        self.ln_out = nn.LayerNorm(n_dim, device=device, dtype=dtype)
        self.head = nn.Linear(n_dim, n_vocab, bias=False, device=device, dtype=dtype)

        # init state tuning support
        if configMap.init_state_wkv:
            stateTuneList = [None]*n_layer
            for i in range(n_layer):
                stateTuneList[i] = nn.ParameterDict({
                    "wkv": nn.Parameter(torch.zeros(n_dim // 64, 64, 64, device=device, dtype=dtype)),
                })
            self.init_state = nn.ParameterList(stateTuneList)

    def load_from_model_state_dict(self, state_dict: dict, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the RWKV_TimeMix model weights, using the layer_id
        '''
        for i, block in enumerate(self.blocks):
            block.load_from_model_state_dict(state_dict, i, non_blocking=non_blocking)
        
        self.ln_out.weight.data.copy_(state_dict['ln_out.weight'], non_blocking=True)
        self.ln_out.bias.data.copy_(state_dict['ln_out.bias'], non_blocking=True)
        self.head.weight.data.copy_(state_dict['head.weight'], non_blocking=True)
        self.emb.weight.data.copy_(state_dict['emb.weight'], non_blocking=True)

    ### ---
    ###
    ### Init state handling
    ###
    ### ---

    def get_init_state(self, batch_size:int=1, skip_init_state:bool=False) -> list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]:
        '''
        Get an initialized copy of the model state, for the given batch size
        '''
        # Get required configs
        n_dim = self.configMap.n_dim
        init_state_wkv = self.configMap.init_state_wkv
        n_layer = self.configMap.n_layer

        # Prepare the initial state
        init_state = [ None for i in range(n_layer) ]
        for i in range(n_layer):
            device = self.blocks[i].ln1.weight.data.device
            dtype = self.blocks[i].ln1.weight.data.dtype

            # Use the saved init_state if enabled
            wkv_state = torch.zeros(batch_size, n_dim // 64, 64, 64, device=device, dtype=dtype)
            if init_state_wkv and skip_init_state == False:
                init_wkv = self.init_state[i]["wkv"]
                for b in range(batch_size):
                    wkv_state[b][:] = init_wkv

            # Prepare the state
            init_state[i] = (
                torch.zeros(batch_size, n_dim, device=device, dtype=dtype),
                wkv_state,
                torch.zeros(batch_size, n_dim, device=device, dtype=dtype)
            )
        return init_state

    ### ---
    ###
    ### Model Forward
    ###
    ### ---

    def forward(
        self, idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = None,  
        ret_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = None,
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Forward the block set, given the input tokens and the last state
        Last state is a list of tuple of the following
        - time mix shift state
        - time mix wkv state
        - channel mix shift state

        Returns a pair of the output embedding and the next state
        '''
        # Prepare the state, with the batch size
        if prv_stateList is None:
            prv_stateList = self.get_init_state(idx.shape[0])

        # If no return state is set, let _forward_internal, set it up
        if ret_stateList is None:
            ret_stateList = [ None for i in range(self.configMap.n_layer) ]
            return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=False)

        # Forward internally
        return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=True)

    def _forward_internal(
        self, idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],  
        ret_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],
        overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Internal forward operations, which assumes the state is already initialized
        Due to the lack of safety checks, this should not be used directly
        '''

        # Lets get the embedding
        idx = idx.to(self.emb.weight.device, non_blocking=True)
        x_emb = self.emb(idx)

        # Iterate the block layers, compute the x embedding
        if overwrite_ret_tensor:
            for i, block in enumerate(self.blocks):
                x_emb = x_emb.to(block.ln1.weight.device, non_blocking=True)
                x_emb, last_block_state = block(x_emb, prv_stateList[i])
                ret_stateList[i][0][:] = last_block_state[0]
                ret_stateList[i][1][:] = last_block_state[1]
                ret_stateList[i][2][:] = last_block_state[2]
        else:
            for i, block in enumerate(self.blocks):
                x_emb = x_emb.to(block.ln1.weight.device, non_blocking=True)
                x_emb, ret_stateList[i] = block(x_emb, prv_stateList[i])

        # Final layer norm, and head
        x_emb = x_emb.to(self.ln_out.weight.device, non_blocking=True)
        x_emb = self.ln_out(x_emb)
        x_out = self.head(x_emb)

        # Return the output and the state list
        return x_out, ret_stateList
    
    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],
        ret_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        # Lets get the embedding
        idx = idx.to(self.emb.weight.device, non_blocking=True)
        x_emb = self.emb(idx)

        # Iterate the block layers, compute the x embedding
        for i, block in enumerate(self.blocks):
            x_emb = x_emb.to(block.ln1.weight.device, non_blocking=True)
            block.forward_with_default_compile(
                x_emb, prv_stateList[i], x_emb, ret_stateList[i]
            )

        # Final layer norm, and head
        x_emb = x_emb.to(self.ln_out.weight.device, non_blocking=True)
        x_emb = self.ln_out(x_emb)

        # Due to large embeeding X size, 
        # this cannot be preinit and compiled
        out_emb = self.head(x_emb)

        # Return the output and the state list
        return out_emb, ret_stateList

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(
        self, idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = None,  
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Compiled varient of the forward function
        With no input tensor being modified.
        Useful for reduce-overhead compile mode
        '''
        return self.forward(idx, prv_stateList, prv_stateList)